# coding=utf-8

from .ava_dataset import Ava
from .build import DATASET_REGISTRY, build_dataset
from .charades import Charades
from .imagenet import Imagenet
from .kinetics import Kinetics
from .mmit import Mmit
from .activitynet import Activitynet
from .multi_dataset_seq import Multi_dataset_seq
from .mit import Mit
from .ssv2 import Ssv2
from .aicity import Aicity
from .ssv2_frames import Ssv2_frames
from .web_video_text import Web_video_text

try:
    from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2
except Exception:
    #print("Please update your PyTorchVideo to latest master")
    pass
