#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import av
import decord
import slowfast.utils.distributed as du

def get_video_container(
        path_to_vid, multi_thread_decode=False, backend="pyav",
        use_gpu_decode=False):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        container = av.open(path_to_vid)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        return container
    elif backend == "decord":
        if use_gpu_decode:
            container = decord.VideoReader(path_to_vid, ctx=decord.gpu(du.get_rank()))
        else:
            container = decord.VideoReader(path_to_vid, ctx=decord.cpu(0))
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))
