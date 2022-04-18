#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch
import numpy as np

def make_contrastive_minibatch_gt(labels):
    # given [B] of long tensor, return [B, B] gt of floats
    # TODO(junwei): make this faster!
    batch_size = labels.shape[0]

    # cannot be long, some label would be fractions
    gt = torch.zeros((batch_size, batch_size), dtype=torch.float32).cuda(non_blocking=True)

    for i in range(batch_size):
        same_text_label = labels == labels[i]  # [B]
        same_text_label = same_text_label.float()
        gt[i, :] = same_text_label / same_text_label.sum()  # make sure it sums to 1

    return gt


def compute_recall_at_rank(simi_matrix, labels, recalls):
    # simi_matrix torch tensors of [B, B]
    # labels [B, B], non-zero for a positive match
    # recalls [1, 5, 10]

    # for each sample, check whether the top-rank k match has any positive
    # [B, B], largest index at front
    rankings = torch.argsort(simi_matrix, descending=True)
    batch_size = len(rankings)

    # TODO: make this faster
    ranking_to_labels = torch.zeros((batch_size, batch_size))
    for i in range(batch_size):
        for j in range(batch_size):
            ranking_to_labels[i, j] = labels[i, rankings[i, j]]
    # for each sample, from left to right, if the score is not zero, then it is correct
    recall_at_ranks = []
    for r in recalls:
        # each sample's top rank r has any positive (True/False)
        top_rank_has_pos = ranking_to_labels[:, :r].sum(dim=1) > 0 # [B]
        # averaged
        recall_at_ranks.append(float(top_rank_has_pos.float().mean().detach().cpu().numpy()))

    return recall_at_ranks


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topks_correct_full_label(preds, labels, ks, no_stacking=False):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (list): a list of num_class tensors
        labels (list): a list of num_class tensors, could be one-hot or multi-label
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert len(preds) == len(labels), "Batch dim of predictions and labels must match"
    if not no_stacking:  # this means preds/labels are already [N, num_class]
        # [N, num_class]
        preds = torch.stack(preds, dim=0)
    # Find the top max_k predictions for each sample
    # top_max_k_inds: (batch_size, max_k)
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )


    topks_correct = [[] for k in ks]
    for b in range(len(labels)):
        # [max_k]
        pred_topk = top_max_k_inds[b].cpu().numpy().tolist()
        # K labels
        # a tensor of index
        #print(labels[b], labels[b].shape)
        this_labels = [o[0] for o in (labels[b] == 1.0).nonzero(as_tuple=False).cpu().numpy().tolist()]
        # [110, 135, 156, 164] [278, 133, 34, 220, 154]
        #print(this_labels, pred_topk)



        for i_k, k in enumerate(ks):
            pred_inds = set(pred_topk[:k])
            label_inds = set(this_labels)
            inter_count = len(pred_inds.intersection(label_inds))
            # for each top k, as long as the prediction has all the k - labels, it is correct
            #if inter_count >= min(k, len(label_inds)):

            # the multi-moments-in-time paper uses top-5 like this: it is correct if top-5
            # predictions has any positive labels
            if inter_count >= 1:
                topks_correct[i_k].append(1.0)
            else:
                topks_correct[i_k].append(0.0)


    topks_correct = [np.mean(k) for k in topks_correct]

    return topks_correct

def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]
