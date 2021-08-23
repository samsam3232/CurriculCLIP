import torch
import logging
import wandb
import os
import json
from torch.cuda.amp import autocast
import time
from training.zero_shot_imagenet_util import zero_shot_evaluation
from training.utils import get_predictions_metrics, get_loss
import torch.nn as nn
import numpy as np



def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
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
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()

        images, texts = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts['input_ids'] = texts['input_ids'].squeeze(1).cuda(args.gpu, non_blocking=True)
            texts['attention_mask'] = texts['attention_mask'].squeeze(1).cuda(args.gpu, non_blocking=True)

        inputs = {'input_ids': texts['input_ids'], 'attention_mask': texts['attention_mask'], 'pixel_values': images}
        data_time = time.time() - end

#        m = model.module

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = get_loss(model, images, texts, loss_img, loss_txt, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss(model, inputs, loss_img, loss_txt, args)
            total_loss.backward()
            optimizer.step()

#        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)
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
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
#                "scale": m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})

        if (i % 250) == 249:
            logging.info(f'Epoch {epoch}, step {i} evaluation:')
            evaluate(model, data, epoch, args, tb_writer)


def evaluate(model, data, epoch, args, tb_writer=None):

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
                ims = ims.cuda(args.gpu, non_blocking=True)
                texts['input_ids'] = texts['input_ids'].squeeze(1).cuda(args.gpu, non_blocking=True)
                texts['attention_mask'] = texts['attention_mask'].squeeze(1).cuda(args.gpu, non_blocking=True)

            inputs = {'input_ids': texts['input_ids'], 'attention_mask': texts['attention_mask'], 'pixel_values': ims}

            outputs = model(**inputs)
            all_im_features.append(outputs['image_embeds'])
            all_t_features.append(outputs['text_embeds'])
            im_logits = outputs['logits_per_image'].mean() * outputs['image_embeds'] @ outputs['text_embeds'].t()
            t_logits = outputs['logits_per_text'].mean() * outputs['text_embeds'] @ outputs['image_embeds'].t()

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
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)
        if args.wandb:
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch})

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics