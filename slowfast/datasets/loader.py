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


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.

    if is multi_dataset, label and extra_data need to be repackaged as well
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]

    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:  # for imgnet only
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data

def multi_dataset_collate(batch, dataset_names=[], dataset_num_classes=[], is_multiple_aug=False):
    assert len(dataset_names) == len(dataset_num_classes)

    inputs, labels, video_idx, extra_data = zip(*batch)
    # print(extra_data)({'dataset_name': 'mmit'}, {'dataset_name': 'mmit'}, {'dataset_name': 'mmit'})
    if is_multiple_aug:
        # Rand Aug with multiple samples
        inputs = [item for sublist in inputs for item in sublist]
        labels = [item for sublist in labels for item in sublist]
        video_idx = [item for sublist in video_idx for item in sublist]
        # Aug Sample == 2
        # ({'dataset_name': ['mmit', 'mmit']}, {'dataset_name': ['activitynet', 'activitynet']}, {'dataset_name': ['mmit', 'mmit']})
        extra_data = [{"dataset_name": d_name} for subdict in extra_data for d_name in subdict["dataset_name"]]

    inputs, video_idx = default_collate(inputs), default_collate(video_idx)

    # labels are a list [B] items of [num_classes] or single int, will convert to one-hot if needed
    # extra_data["dataset_name"] is also B items

    # we need to make labels {dataset_name: [B, num_classes]}
    # extra_data["label_mask"] {dataset_name: [B]}
    dataset_labels = {}
    dataset_masks = {}
    batch_size = len(extra_data)
    assert len(labels) == len(extra_data)
    for dataset_name, num_class in zip(dataset_names, dataset_num_classes):
        dataset_labels[dataset_name] = torch.zeros((batch_size, num_class), dtype=torch.float32)
        dataset_masks[dataset_name] = torch.zeros((batch_size), dtype=torch.float32)

    #print(labels, extra_data)
    # labels is a [B] item, each could be a one-hot array or an int

    for i, (label, extra_data_item) in enumerate(zip(labels, extra_data)):
        this_dataset_name = extra_data_item["dataset_name"]
        this_dataset_num_class = dataset_num_classes[dataset_names.index(this_dataset_name)]
        if isinstance(label, int):
            # convert to one-hot
            label_arr = np.zeros((this_dataset_num_class), dtype=np.float32)
            label_arr[label] = 1.
            label = label_arr

        dataset_labels[this_dataset_name][i, :] = torch.tensor(label)
        dataset_masks[this_dataset_name][i] = 1.

    extra_data = {}
    extra_data["label_masks"] = dataset_masks

    labels = dataset_labels
    return inputs, labels, video_idx, extra_data


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "boxes" or key == "ori_boxes":
            # Append idx info to the bboxes before concatenating them for ROI Align
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, collated_extra_data


def construct_loader(cfg, split, is_precise_bn=False, return_dataset=False):
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

    if isinstance(dataset, torch.utils.data.IterableDataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    else:
        if (
            cfg.MULTIGRID.SHORT_CYCLE
            and split in ["train"]
            and not is_precise_bn
        ):
            # Create a sampler for multi-process training
            sampler = utils.create_sampler(dataset, shuffle, cfg)
            batch_sampler = ShortCycleBatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
            )
            # Create a loader
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                worker_init_fn=utils.loader_worker_init_fn(dataset),
            )
        else:
            # Create a sampler for multi-process training
            # this will sample a part of a batch for each process
            # DistributedSampler(dataset)
            # we set world_size/rank to be the total GPUs in utils.multiprocessing
            # None if GPU==0/1
            sampler = utils.create_sampler(dataset, shuffle, cfg)
            # Create a loader
            if cfg.DETECTION.ENABLE:
                collate_func = detection_collate
            elif not cfg.MODEL.USE_MULTI_HEAD and cfg.AUG.ENABLE and cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
                collate_func = partial(
                    multiple_samples_collate, fold="imagenet" in dataset_name,
                )
            # so for multi_head model during testing, we assume you run with a single dataset
            elif cfg.MODEL.USE_MULTI_HEAD and split in ["train", "val"]:
                collate_func = partial(
                    multi_dataset_collate,
                    dataset_names=cfg.MODEL.MULTI_DATASETS,
                    dataset_num_classes=cfg.MODEL.MULTI_NUM_CLASSES,
                    is_multiple_aug=cfg.AUG.ENABLE and cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]
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

    if return_dataset:
        return loader, dataset
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
