import os
import time
import logging
from time import gmtime, strftime
from pathlib import Path
import json

import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

from model.clip import _transform, load
from model.model import convert_weights, CLIP
from train import train, evaluate
from data.data import get_data
from training.params import parse_args
from training.logger import setup_primary_logging, setup_worker_logging
from training.scheduler import cosine_lr
from transformers import CLIPModel, CLIPConfig, CLIPProcessor


# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def main_worker(gpu, log_queue, args):

    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Log and save params.
    logging.info("Params:")
    params_file = os.path.join(args.logs, args.name, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    args.batch_size *= args.world_size

    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    # Do not use skip_reset unless you want to use on of the CLIP model
    model_config_file = Path(__file__).parent / f"model/model_configs/{args.model.replace('/', '-')}.json"
    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIPModel(CLIPConfig())
    model_empty = CLIP(**model_info)
#    convert_weights(model)
    preprocess_train = _transform(model_empty.visual.input_resolution, is_train=True)
    preprocess_val = _transform(model_empty.visual.input_resolution, is_train=False)

#    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
#        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
        logging.warning("This will run on CPU.")
    else:
        model.cuda(args.gpu)
#        if args.precision == "fp16":
#            convert_weights(model)

        if args.world_size > 1:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)

#        if args.precision == "fp16":
#            convert_weights(model)

    data = get_data(args, (preprocess_train, preprocess_val))

    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, )
        total_steps = data["train"].num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"]
            state_dic = checkpoint["state_dict"]
            state_dic = {k[len('module.'):]: v for k, v in state_dic.items()}
            model.load_state_dict(state_dic)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    args.save_logs = (args.logs != 'none') and (args.gpu == min(args.multigpu))
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    if args.wandb:
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="open-clip",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if args.train_data is None:
        evaluate(model, data, start_epoch, args, writer)
        return
#    elif start_epoch == 0 and args.val_data is not None:
#        evaluate(model, data, 0, args, writer)

    for epoch in range(start_epoch, args.epochs):
        if (args.gpu == min(args.multigpu)):
            logging.info(f'Start epoch {epoch}')
        train(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        steps = data["train"].num_batches * (epoch + 1)
        if args.val_data is not None:
            evaluate(model, data, epoch + 1, args, writer, steps)

        # Saving checkpoints.
        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            if (epoch + 1) == args.epochs or (
                    args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0
            ):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
                )

    if args.wandb and (args.gpu == 0 or (not args.distributed)):
        wandb.finish()

def main():
    args = parse_args()

    # get the name of the experiments
    if args.name is None:
        args.name = strftime(
            f'parameters:_'
            f"lr={args.lr}_"
            f"wd={args.wd}_"
            f"agg={args.aggregate}_"
            f"model={args.model}_"
            f"sorted={args.sorted}_"
            f"batchsize={args.batch_size}_workers={args.workers}_date=%Y-%m-%d-%H-%M-%S",
            gmtime(),
        )

    args.log_path = os.path.join(args.logs, args.name, "app.log")
    if os.path.exists(args.log_path):
        print("There already is such an experiment, specify a new name in the args.")
        return -1

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    torch.multiprocessing.set_start_method("spawn")

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    args.gpu = args.multigpu[0]
    args.world_size = len(args.multigpu)

    main_worker(args.gpu, log_queue, args)


if __name__ == "__main__":
    main()
