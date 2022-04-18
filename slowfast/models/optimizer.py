#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer


try:
    # https://github.com/facebookresearch/bitsandbytes
    import bitsandbytes as bnb
    # this can reduce GPU memory usage while achieving similar performance
    # not useful as the paper claimed
except Exception as e:
    #print(e)
    pass

import torch.optim as optim


import slowfast.utils.lr_policy as lr_policy
#import slowfast.utils.distributed as du


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    no_grad_parameters = []

    # for multi-dataset head training
    cross_proj_parameters = []
    skip = {}
    if cfg.NUM_GPUS > 1:
        if hasattr(model.module, "no_weight_decay"):
            skip = model.module.no_weight_decay()
            skip = {"module." + v for v in skip}
    else:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
    for name, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        if name.startswith("head.cross_dataset_heads"):
            for p in m.parameters(recurse=False):
                cross_proj_parameters.append(p)
            continue
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                no_grad_parameters.append(p)
            elif is_bn:
                bn_parameters.append(p)
            elif name in skip:
                zero_parameters.append(p)

            # True for MViT
            elif cfg.SOLVER.ZERO_WD_1D_PARAM and \
                (len(p.shape) == 1 or name.endswith(".bias")):
                zero_parameters.append(p)
            else:
                non_bn_parameters.append(p)

    # construct per parameter options
    """
    optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
    """
    cross_proj_lr = cfg.SOLVER.BASE_LR
    cross_proj_momentum = cfg.SOLVER.MOMENTUM
    if cfg.MODEL.MULTI_PROJ_TRAIN_DIFF_LR:
        cross_proj_lr = cfg.MODEL.MULTI_PROJ_LR
        cross_proj_momentum = cfg.MODEL.MULTI_PROJ_MOMENTUM
    optim_params = [
        {"params": bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        {"params": zero_parameters, "weight_decay": 0.0},
        {
            "params": cross_proj_parameters,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "lr": cross_proj_lr,
            "momentum": cross_proj_momentum,
        },
    ]
    optim_params = [x for x in optim_params if len(x["params"])]

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_parameters
    ) + len(zero_parameters) + len(
        no_grad_parameters
    ) + len(cross_proj_parameters), "parameter size does not match: {} + {} + {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_grad_parameters),
        len(cross_proj_parameters),
        len(list(model.parameters())),
    )

    # bn 0, non bn 106, zero 205 no grad 0 for MViT 16x4
    # MViT not setting weight decay for 1-dim parameters
    """
    print(
        "bn {}, non bn {}, zero {} no grad {}".format(
            len(bn_parameters),
            len(non_bn_parameters),
            len(zero_parameters),
            len(no_grad_parameters),
        )
    )
    """
    #print(model.parameters())
    #print(len(optim_params))  # 3 [non_bn, zero, ]
    #sys.exit()

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    # zero redundacy optimizer does not seem to reduce memory on 3GPU single machine 1080 TI
    elif cfg.SOLVER.OPTIMIZING_METHOD == "zero_sgd":
        if zero_parameters:
            print("warning, not setting zero parameters weight decay to zero")
        # this will need NUM_GPU > 1
        return ZeroRedundancyOptimizer(
            model.parameters(),
            #process_group=du.get_local_process_group(),  # it could already set to default local process group
            optimizer_class=optim.SGD,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            parameters_as_bucket_view=True,
            nesterov=cfg.SOLVER.NESTEROV)
    elif cfg.SOLVER.OPTIMIZING_METHOD == "sgd_8bit":
        print("using 8-bit optimizer")
        # this does not seem to help as of 11/17/2021 (on 1080 TI)
        return bnb.optim.SGD8bit(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
            # # parameter tensors with less than 2048 values are optimized in 32-bit
            min_8bit_size=4096,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam_8bit":
        print("using 8-bit optimizer")
        # does not seeme to help
        return bnb.optim.Adam8bit(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            # # parameter tensors with less than 2048 values are optimized in 32-bit
            min_8bit_size=4096,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    # zero redundacy optimizer does not seem to reduce memory on 3GPU single machine 1080 TI
    elif cfg.SOLVER.OPTIMIZING_METHOD == "zero_adamw":
        if zero_parameters:
            print("warning, not setting zero parameters weight decay to zero")
        return ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=optim.AdamW,
            lr=cfg.SOLVER.BASE_LR,
            eps=1e-08,
            parameters_as_bucket_view=True,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr, skip_last_group=False):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    # print(len(optimizer.param_groups))  # correspond to the optim group above

    for i, param_group in enumerate(optimizer.param_groups):
        if skip_last_group and i == len(optimizer.param_groups) - 1:
            continue
        param_group["lr"] = new_lr
