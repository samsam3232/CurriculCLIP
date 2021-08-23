import torch
import numpy as np
import torch.nn as nn


def get_predictions_metrics(t_features, im_features):

    im_logits = im_features @ t_features.t()
    t_logits = im_logits.t()

    logits = {"t_2_im": t_logits, "im_2_t": im_logits}
    gt = (torch.arange(len(t_features)).view(-1, 1).to(im_logits.device))

    results = dict()
    for name, logit in logits.items():
        preds = (torch.where(torch.argsort(logit, descending=True) == gt)[1]).detach().cpu().numpy()
        results[f"{name}_mean_rank"] = preds.mean() + 1
        results[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            results[f"{name}_R@{k}"] = np.mean(preds < k)

    return results


def get_loss(model, inputs, loss_img, loss_txt, args):

    outputs = model(**inputs)
    im_logits = outputs['logits_per_image'].mean() * outputs['image_embeds'] @ outputs['text_embeds'].t()
    t_logits = outputs['logits_per_text'].mean() * outputs['text_embeds'] @ outputs['image_embeds'].t()

    ground_truth = torch.arange(len(im_logits)).long()
    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    total_loss = (loss_img(im_logits, ground_truth)+ loss_txt(t_logits, ground_truth)) / 2
    return total_loss