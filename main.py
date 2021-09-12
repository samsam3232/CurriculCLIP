import os
import logging
from time import gmtime, strftime
from pathlib import Path
import json
import torch
from torch import optim
import torch.backends.cudnn as cudnn

from model.clip import _transform
from model.model import convert_weights, CLIP
from train import train, evaluate
from data.data import get_data
from training.params import parse_args
from training.logger import setup_primary_logging, setup_worker_logging


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def main_worker(gpu, log_queue, args):

    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)

    logging.info("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        logging.info(f"  {name}: {val}")

    args.batch_size *= args.world_size

    # Do not use skip_reset unless you want to use on of the CLIP model
    model_config_file = Path(__file__).parent / f"model/model_configs/{args.model.replace('/', '-')}.json"
    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)

    model = CLIP(**model_info)
    convert_weights(model)
    preprocess_train = _transform(model.visual.input_resolution, is_train=True)
    preprocess_val = _transform(model.visual.input_resolution, is_train=False)
    convert_models_to_fp32(model)

    data = get_data(args, (preprocess_train, preprocess_val))

    if not torch.cuda.is_available():
        model.float()
        logging.warning("This will run on CPU.")
    else:
        model.cuda(args.gpu)

        if args.world_size > 1:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)

    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, )

    start_epoch = 0
    cudnn.benchmark = True
    cudnn.deterministic = False

    args.save_logs = (args.logs != 'none') and (args.gpu == min(args.multigpu))

    if args.train_data is None:
        evaluate(model, data, start_epoch, args)
        return
    elif start_epoch == 0 and args.val_data is not None:
        evaluate(model, data, 0, args)

    for epoch in range(start_epoch, args.epochs):
        if (args.gpu == min(args.multigpu)):
            logging.info(f'Start epoch {epoch}')
        train(model, data, epoch, optimizer, args)
        data['trainset'].reshuffle()
        data['train'] = torch.utils.data.DataLoader(data['trainset'], batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                            pin_memory=True, drop_last=True)
        data['train'].num_batches = len(data['train'])
        data['train'].num_samples = len(data['trainset'])
        logging.info("RESHUFFLED DATA")
        if args.val_data is not None:
            evaluate(model, data, epoch + 1, args)

        if args.save_logs and (args.gpu == min(args.multigpu)):
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


def main():
    args = parse_args()

    if args.name is None:
        args.name = strftime(
            f'parameters:_'
            f"lr={args.lr}_"
            f"wd={args.wd}_"
            f"model={args.model}_"
            f"sorted={args.sorted}_"
            f"batchsize={args.batch_size}_workers={args.workers}_date=%Y-%m-%d-%H-%M-%S",
            gmtime(),
        )

    args.log_path = os.path.join(args.logs, args.name, "app.log")
    if os.path.exists(args.log_path):
        print("There already is such an experiment, specify a new name in the args.")
        return -1

    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    torch.multiprocessing.set_start_method("spawn")

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    assert len(args.multigpu) > 0, "You have no GPU to train on, training or running without GPU is unrealistic"
    args.gpu = args.multigpu[0]
    args.world_size = len(args.multigpu)

    main_worker(args.gpu, log_queue, args)


if __name__ == "__main__":
    main()
