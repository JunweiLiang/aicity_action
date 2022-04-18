#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_multi_dataset_loss(preds, labels, masks, datasets, loss_funcs,
                               dataset_loss_weights=None,
                               add_cross_proj=False,
                               cross_proj_add_to_pred=False,
                               proj_loss_func=None,
                               proj_loss_weight=0.5):
    """
    The preds should be {dataset_name: [B, num_classes]}
    labels should be {dataset_name: [B] or [B, num_classes]}
    masks should be {dataset_name: [B]} of 1/0, 1 for the loss that we want

    # junwei: 03042022, added cross-dataset projection layer
    # preds will also include dataset1_dataset2 output, same dimension as dataset2
    """
    losses = []
    assert len(datasets) == len(loss_funcs)
    if proj_loss_func is not None:
        proj_loss_func = _LOSSES[proj_loss_func](reduction="none")

    for dataset_name, loss_name in zip(datasets, loss_funcs):
        if loss_name not in _SOFT_TARGET_LOSSES.keys():
            raise NotImplementedError("Loss {} is not supported for multi-dataset".format(loss_name))

        loss_func = _LOSSES[loss_name](reduction="none")

        pred = preds[dataset_name]
        if add_cross_proj and cross_proj_add_to_pred:
            for d1_d2 in preds.keys():
                if d1_d2 in datasets:
                    continue
                d1_name, d2_name = d1_d2.split("_")
                if d2_name == dataset_name:
                    proj_pred = preds[d1_d2]  # [B, num_class]
                    pred = pred + proj_pred * proj_loss_weight
                    # don't use += , in-place change


        loss = loss_func(pred, labels[dataset_name])

        # TODO: change this to a gather function, watch out for mask all zero case
        # https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

        # loss could be [B, num_class] for bce loss
        if len(loss.shape) == 2:
            loss = loss.mean(-1)  # along the class dimension

        # add additional projection loss from other head
        if add_cross_proj and not cross_proj_add_to_pred:
            proj_losses = []
            # get all the projected outputs
            for d1_d2 in preds.keys():
                if d1_d2 in datasets:
                    continue
                d1_name, d2_name = d1_d2.split("_")
                if d2_name == dataset_name:
                    proj_pred = preds[d1_d2]  # [B, num_class]
                    # [B]
                    proj_loss = proj_loss_func(proj_pred, labels[dataset_name])
                    proj_losses.append(proj_loss * proj_loss_weight)

            loss = loss + torch.stack(proj_losses, dim=1).mean(dim=1)

        # loss is [B], mask is also [B]
        if dataset_loss_weights is not None:
            # bce loss is 0.6 and ce loss could be 5.0, so some balancing is needed
            loss = dataset_loss_weights[dataset_name] * loss
        # mask out samples in this batch that is not this dataset's
        loss_masked = masks[dataset_name] * loss
        losses.append(loss_masked)
    #print(len(losses))  # num dataset
    #print([l.shape for l in losses])  # each is batch_size
    # we do the reduction after getting all the loss for each sample
    # losses are B * 3 items with lots of zeros
    # TODO(junwei): check this under DDP, the loss is not gathered yet
    loss_per_minibatch = torch.cat(losses, dim=0).sum() / losses[0].shape[0]
    #print(loss_per_minibatch)
    #sys.exit()
    return loss_per_minibatch


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_vicreg_loss(emb, std_weight=25.0, cov_weight=1.0):
    """
    https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    enforce the representation/embedding to be more informative
    """
    # need to all_gather all emb of the whole batch to compute
    # [B, C]
    emb = emb - emb.mean(dim=0)
    std_emb = torch.sqrt(emb.var(dim=0) + 1e-4)
    # [B] -> scalar
    std_loss = torch.mean(F.relu(1 - std_emb))  # hinge loss

    batch_size = emb.shape[0]
    feature_size = emb.shape[1]
    cov_emb = (emb.T @ emb) / (batch_size - 1)  # 1/(n-1)

    # [B, B] -> scalar
    cov_loss = off_diagonal(cov_emb).pow_(2).sum().div(feature_size)

    return std_loss * std_weight + cov_loss * cov_weight


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target. For MixUp train
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # x [B, num_classes]
        # y [B, C], 0 - 1.0
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

class NormalizedSoftTargetCrossEntropy(nn.Module):
    """
    Normalized Cross entropy loss with soft target. For MixUp train
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(NormalizedSoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # x [B, num_classes], logits
        # y [B, C], 0 - 1.0
        # softmax on logits and then log
        pred = F.log_softmax(x, dim=-1) # [B, num_classes]
        loss = - torch.sum(y * pred, dim=-1) / (- pred.sum(dim=-1))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class ReverseSoftTargetCrossEntropy(nn.Module):
    """
    Reverse Cross entropy loss with soft target. For MixUp train
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(ReverseSoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # x [B, num_classes], logits
        # y [B, C], 0 - 1.0
        # softmax on logits and then log
        pred = F.softmax(x, dim=-1) # [B, num_classes]
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        # so low confidence class will be ignore
        y = torch.clamp(y, min=1e-4, max=1.0)  # cannot be zeros
        y = torch.log(y)
        loss = - torch.sum(y * pred, dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class MeanAbsoluteError(nn.Module):
    """
    mean absolute error loss, it is said to be more robust to label noise compared
    to CE loss
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MeanAbsoluteError, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # x [B, num_classes], logits
        # y [B, C], 0 - 1.0
        # softmax on logits and then log
        pred = F.softmax(x, dim=-1) # [B, num_classes]
        loss = 1. - torch.sum(y*pred, dim=-1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

# combined loss with noramlized ce
# http://proceedings.mlr.press/v119/ma20c/ma20c.pdf
class NCEandRCE(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, reduction="mean"):
        super(NCEandRCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nce = NormalizedSoftTargetCrossEntropy(reduction=reduction)
        self.rce = ReverseSoftTargetCrossEntropy(reduction=reduction)

    def forward(self, pred, labels):
        return self.alpha * self.nce(pred, labels) + self.beta * self.rce(pred, labels)


class LSEPLoss(nn.Module):
    """
    # http://openaccess.thecvf.com/content_cvpr_2017/html/Li_Improving_Pairwise_Ranking_CVPR_2017_paper.html
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(LSEPLoss, self).__init__()
        self.reduction = reduction

    def forward(self, scores, labels):
        # scores [B, num_classes]
        # labels [B, C], 0 -1 1.0

        mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
                     labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
        loss = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
                     scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
        loss = loss.exp().mul(mask).sum().add(1).log()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

# AVA models use bce
# Kinetics models use MViT - soft_cross_entropy, slowfast-> cross_entropy
_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,  # this can be used with soft targets
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "mean_absolute_error": MeanAbsoluteError,
    "reverse_soft_cross_entropy": ReverseSoftTargetCrossEntropy,
    "normalized_soft_cross_entropy": NormalizedSoftTargetCrossEntropy,
    "nce_and_rce": NCEandRCE,
    "lsep": LSEPLoss,
}

# these losses takes labels of [B, C] as inputs
_SOFT_TARGET_LOSSES = {
    "bce": nn.BCELoss,  # this can be used with soft targets
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "nce_and_rce": NCEandRCE,
    "normalized_soft_cross_entropy": NormalizedSoftTargetCrossEntropy,
    "reverse_soft_cross_entropy": ReverseSoftTargetCrossEntropy,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]






