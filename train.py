import torch
import logging
import os
import json
import time
from training.zero_shot_imagenet_util import zero_shot_evaluation
from training.utils import get_predictions_metrics, get_loss
import torch.nn as nn



def train(model, data, epoch, optimizer, args):
    os.environ["WDS_EPOCH"] = str(epoch)

    model.train()

    loader = data['train']

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    num_batches_per_epoch = loader.num_batches

    end = time.time()
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        images, texts = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        total_loss = get_loss(model, images, texts, loss_img, loss_txt, args)
        total_loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        if (i % 100) == 0:
            num_samples = i * len(images)
            samples_per_epoch = loader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}"
            )

        if (i % 2000) == 1999:
            logging.info(f'Epoch {epoch}, step {i} evaluation:')
            evaluate(model, data, epoch, args)


def evaluate(model, data, epoch, args):

    model.eval()
    if epoch % args.zeroshot_frequency == 0:
        zshot_metric = zero_shot_evaluation(model, data['imagenet-val'], epoch, args) if ('imagenet-val' in data and args.zeroshot_frequency != 0) else {}
    else:
        zshot_metric = {}


    loader = data['val']
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    total_loss, samples_num = 0.0, 0.0
    all_im_features, all_t_features = list(), list()

    with torch.no_grad():
        for ims, texts in loader:
            if args.gpu is not None:
                ims, texts = ims.cuda(args.gpu, non_blocking=True), texts.cuda(args.gpu, non_blocking=True)

            im_features, t_features, logit_scale = model(ims, texts)
            all_im_features.append(im_features)
            all_t_features.append(t_features)
            logit_scale = logit_scale.mean()
            im_logits = logit_scale * im_features @ t_features.t()
            t_logits = im_logits.t()

            gt = torch.arange(len(ims)).long()
            if args.gpu is not None:
                gt = gt.cuda(args.gpu, non_blocking=True)

            curr_loss = (loss_img(im_logits, gt) + loss_txt(t_logits, gt)) / 2
            batch_size = len(ims)
            total_loss += curr_loss * batch_size
            samples_num += batch_size

        metrics = get_predictions_metrics(torch.cat(all_im_features), torch.cat(all_t_features))
        metrics.update(zshot_metric)
        metrics.update({"loss:": (total_loss / samples_num).item(), "epoch": epoch, "samples num": samples_num})

        logging.info(f"Eval Epoch: {epoch} " + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics