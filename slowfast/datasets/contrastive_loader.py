#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
from functools import partial
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from slowfast.datasets.multigrid_helper import ShortCycleBatchSampler

from . import utils as utils
from .build import build_dataset


def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.

    if is multi_dataset, label and extra_data need to be repackaged as well
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, tokens, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    tokens = [item for sublist in tokens for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]

    inputs, labels, tokens, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(tokens),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    return inputs, labels, tokens, video_idx, extra_data


def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        # per machine?
        #batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        # junwei: change to TRAIN.BATCH_SIZE means global batch_size,
        # here we compute the batch_size per machine per gpu
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS) / max(1, cfg.NUM_SHARDS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        #batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS) / max(1, cfg.NUM_SHARDS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        #batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS) / max(1, cfg.NUM_SHARDS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    # Create a sampler for multi-process training
    # this will sample a part of a batch for each process
    # DistributedSampler(dataset)
    # we set world_size/rank to be the total GPUs in utils.multiprocessing
    # None if GPU==0/1
    sampler = utils.create_sampler(dataset, shuffle, cfg)
    # Create a loader
    if cfg.AUG.ENABLE and cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
        collate_func = partial(
            multiple_samples_collate,
        )
    else:
        collate_func = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        # this shuffle is for NUM_GPU==0/1, with a shufflesampler,
        # otherwise the shuffling will be done in the sampler
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_func,
        worker_init_fn=utils.loader_worker_init_fn(dataset),
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    if (
        loader._dataset_kind
        == torch.utils.data.dataloader._DatasetKind.Iterable
    ):
        if hasattr(loader.dataset, "sampler"):
            sampler = loader.dataset.sampler
        else:
            raise RuntimeError(
                "Unknown sampler for IterableDataset when shuffling dataset"
            )
    else:
        sampler = (
            loader.batch_sampler.sampler
            if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
            else loader.sampler
        )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
