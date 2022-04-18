#!/usr/bin/env python3

import os
import random
import torch
import torch.utils.data
from torchvision import transforms

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment


from .build import build_dataset

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Multi_dataset_seq(torch.utils.data.ConcatDataset):
    """
        from
        https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset
    """
    def __init__(self, cfg, mode):
        self.mode = mode
        self.cfg = cfg

        assert cfg.MODEL.USE_MULTI_HEAD

        # construct each datasets
        self.dataset_names = cfg.MODEL.MULTI_DATASETS
        # [1, 2, 1] then dataset[1] will be used twice
        self.dataset_replicas = cfg.MODEL.MULTI_REPLICAS
        assert len(self.dataset_replicas) == len(self.dataset_names)

        datasets = []
        for num_replica, dataset_name in zip(self.dataset_replicas, self.dataset_names):
            if mode != "train":
                num_replica = 1
            for i in range(int(num_replica)):
                dataset = build_dataset(dataset_name, cfg, mode)
                datasets.append(dataset)

        super().__init__(datasets)
