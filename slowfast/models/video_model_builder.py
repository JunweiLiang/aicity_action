# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.attention import MultiScaleBlock
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.stem_helper import PatchEmbed
from slowfast.models.utils import round_width, validate_checkpoint_wrapper_import

# for contrastive learning
from slowfast.models.text_models import Transformer

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

from copy import deepcopy
import numpy as np

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            #print(pool_size[pathway]) [1, 1, 1] for Slowfast 101 non-local
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        # x: slow_frames, fast_frames
        # x: [torch.Size([1, 3, 8, 256, 256]), torch.Size([1, 3, 32, 256, 256])]
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        #x: [torch.Size([1, 2048, 8, 16, 16]), torch.Size([1, 256, 32, 16, 16])]
        # 256 -> 16, so spatial is 16x downsample
        # so DETECTION.SPATIAL_SCALE_FACTOR should be 16,
        # the bboxes will be scaled down 16 times from its original coordindates
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        # Based on profiling data of activation size, s1 and s2 have the activation sizes
        # that are 4X larger than the second largest. Therefore, checkpointing them gives
        # best memory savings. Further tuning is possible for better memory saving and tradeoffs
        # with recomputing FLOPs.
        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)
            self.s1 = checkpoint_wrapper(s1)
            self.s2 = checkpoint_wrapper(s2)
        else:
            self.s1 = s1
            self.s2 = s2

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        y = []  # Don't modify x list in place due to activation checkpoint.
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            y.append(pool(x[pathway]))
        x = self.s3(y)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if cfg.X3D.CHANNELWISE_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            NotImplementedError
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x


@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg

        # by junwei, Improved MViT not using CLS_EMB already
        #assert not cfg.MVIT.CLS_EMBED_ON, "dont use this!"
        if cfg.MVIT.CLS_EMBED_ON:
            print("warning, using CLS_EMBED_ON")

        # by junwei, MViT version 2
        # https://arxiv.org/pdf/2112.01526v1.pdf
        self.use_query_residual_pool = cfg.MVIT.Q_POOL_RESIDUAL
        self.q_pool_all = cfg.MVIT.Q_POOL_ALL
        self.channel_expand_front = cfg.MVIT.CHANNEL_EXPAND_FRONT
        self.pool_skip_use_conv = cfg.MVIT.POOL_SKIP_USE_CONV

        # whether to use x = x[0] for inputs
        self.direct_input = cfg.MVIT.DIRECT_INPUT


        pool_first = cfg.MVIT.POOL_FIRST

        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:  # default False for 16x4 and 32x3 model
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        dim_out = embed_dim
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS  # 1
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            # create a new function with this eps
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes

        # simple convolutions,
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,  # [3, 7, 7]
            stride=cfg.MVIT.PATCH_STRIDE,  # [2, 4, 4]
            padding=cfg.MVIT.PATCH_PADDING,  #[1, 3, 3]
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]

        # 16x224x224 inputs with path_stride (2, 4, 4) -> 8x56x56
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        # 8*56*56
        num_patches = math.prod(self.patch_dims)

        """
        >>> torch.linspace(0, 0.5, 16)
        tensor([0.0000, 0.0333, 0.0667, 0.1000, 0.1333, 0.1667, 0.2000, 0.2333, 0.2667,
        0.3000, 0.3333, 0.3667, 0.4000, 0.4333, 0.4667, 0.5000])
        """
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # pos_embed_dim is not used for sep_pos_embed == True
        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:  # true for mvit 16x4 model and 32x3 model
            # this makes MViT model can only be used for same input size
            # patch_dim: 8x56x56
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, embed_dim)
                )
        else:
            # no separate positional embeddings, so
            # with or without cls_embeding, pos_embed_dim = patch_hwt + 1 or patch_hwt
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        # dropout rate = 0.5
        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        # depth = 16/24
        # dimention multiplier
        # 16x4 model: [[1, 2.0], [3, 2.0], [14, 2.0]]
        # 32x3 model: [[2, 2.0], [5, 2.0], [21, 2.0]]
        # increase the dimension of the next (i+1) block
        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        # depth=16/24
        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            # q pooling stride:
            # [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
            # so [1, 2, 2] at layer 1, 3, 14
            # no q pooling at other layers
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            # q pooling kernel (Conv3D)
            if cfg.MVIT.POOL_KVQ_KERNEL is not None: # this is set by default
            # so q_pooling conv3D kernel always 3x3x3
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        if self.q_pool_all:
            for i in range(len(pool_q)):
                if not pool_q[i]:
                    pool_q[i] = cfg.MVIT.POOL_KVQ_KERNEL
                    stride_q[i] = [1, 1, 1]  # stride=1 conv3d

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        # Default False
        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        self.blocks = nn.ModuleList()

        # activation checkpointing to save GPU memory
        if cfg.MODEL.ACT_CHECKPOINT:
            # check for
            # from fairscale.nn.checkpoint import checkpoint_wrapper
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        # head_mul [[1, 2.0], [3, 2.0], [14, 2.0]]
        # dim_mul [[1, 2.0], [3, 2.0], [14, 2.0]]
        #print(dim_mul, head_mul)  # all ones, 2 at the above index
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])

            # junwei: MViT version 2, this helps reduce parameters and FLOPs
            if self.channel_expand_front:
                if i == 0:
                    embed_dim_mul = 1.0
                else:
                    embed_dim_mul = dim_mul[i-1]
                embed_dim = round_width(embed_dim, embed_dim_mul, divisor=num_heads)
                dim_out = round_width(dim_out, dim_mul[i], divisor=num_heads)
            else:
                embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
                # compute the output dimension of each block
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                use_query_residual_pool=self.use_query_residual_pool,
                channel_expand_front=self.channel_expand_front,
                pool_skip_use_conv=self.pool_skip_use_conv,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)
        # 768
        embed_dim = dim_out

        # MoCo v3 says ViT should not use LN+Pool if CLS token is not used
        # but on K400, without this is 3% worse on Top-1
        self.norm = norm_layer(embed_dim) if not cfg.MVIT.NO_NORM_BEFORE_AVG else None

        if self.sep_pos_embed:
            # from torch.nn.init
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)

        self.add_spatial_max_pool_before_proj = cfg.DETECTION.USE_SPATIAL_MAXPOOL_BEFORE_PROJ

        self.use_roi_head = cfg.DETECTION.ENABLE and not cfg.DETECTION.USE_CUBE_PROP
        self.use_multi_head = cfg.MODEL.USE_MULTI_HEAD

        if self.cfg.CONTRA.ENABLE:
            # for contrastive learning, no classification head,
            # mean pool -> layer_norm -> linear projection
            self.head = head_helper.ContrastiveProjectionHead(
                embed_dim,
                self.cfg.CONTRA.embed_dim,  # the dimension of the projected space vector
                use_MLP=self.cfg.CONTRA.use_MLP,
                dropout_rate=self.cfg.MODEL.DROPOUT_RATE
            )
        else:
            if not self.use_multi_head:
                if self.use_roi_head:
                    temporal_pools = [o[1] for o in cfg.MVIT.POOL_Q_STRIDE]
                    t_pool_kernel = cfg.DATA.NUM_FRAMES // cfg.MVIT.PATCH_STRIDE[0]
                    for t_pool in temporal_pools:
                        t_pool_kernel = t_pool_kernel // t_pool

                    self.head = head_helper.ResNetRoIHead(
                        dim_in=[
                            embed_dim,
                        ],
                        num_classes=num_classes,

                        pool_size=[
                            # we may downsample temporal dim in other places
                            #[cfg.DATA.NUM_FRAMES // cfg.MVIT.PATCH_STRIDE[0], 1, 1]
                            [t_pool_kernel, 1, 1]
                        ],
                        resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION]*2],
                        scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],

                        dropout_rate=cfg.MODEL.DROPOUT_RATE,
                        act_func=cfg.MODEL.HEAD_ACT,
                        aligned=cfg.DETECTION.ALIGNED,
                    )
                else:
                    self.head = head_helper.TransformerBasicHead(
                        embed_dim,
                        num_classes,
                        dropout_rate=cfg.MODEL.DROPOUT_RATE,
                        act_func=cfg.MODEL.HEAD_ACT,
                        use_act_in_train=cfg.MODEL.USE_HEAD_ACT_IN_TRAIN,
                    )
            else:
                assert not self.use_roi_head, "not supported yet"

                self.head = head_helper.TransformerMultiHead(
                    embed_dim,
                    cfg.MODEL.MULTI_DATASETS,
                    cfg.MODEL.MULTI_NUM_CLASSES,
                    cfg.MODEL.MULTI_HEAD_ACT,
                    dropout_rate=cfg.MODEL.DROPOUT_RATE,
                    use_MLP=cfg.MODEL.MULTI_USE_MLP,
                    add_cross_proj=cfg.MODEL.MULTI_ADD_CROSS_PROJ,
                    use_moco=cfg.MODEL.MULTI_USE_MOCO,
                )

        self.apply(self._init_weights)
        if cfg.MODEL.MULTI_USE_MOCO:
            self.init_head_moco()  # copy the weights to moco encoder

    def init_head_moco(self):
        self.head.init_moco()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # False for MViT 16x4
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x, bboxes=None, dataset_name=None, run_cross_proj=False,
                use_moco=False, moco_momentum=0.9):
        # for MViT 16x4, 224 model
        #x: torch.Size([1, 3, 16, 224, 224])
        if not self.direct_input:
            # for slowfast inputs
            x = x[0]

        # Conv3D with kernel (3, 7, 7), stride (2, 4, 4), padding (1, 3, 3)
        # conv weights: [96, 3, 3, 7, 7]
        # so 16x224x224 -> 8x56x56
        x = self.patch_embed(x)

        # dimensions after patch convolutions
        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, _, _ = x.shape


        if self.cls_embed_on:  # default config on

            # nn.Parameter(torch.zeros(1, 1, embed_dim))
            # [B, 1, 96]
            # class embedding is the same for all samples
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks

            # class embedding is at the beginning of this tensor
            # random initialized class embedding concat to the input patch tokens
            # [B, 1 + T*H*W, 96]
            # so each sample has the beginning the same across samples
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:  # default true for MVIT B 32x3 and 16x4

            # self.patch_dims [8, 56, 56] for (16, 224, 224) inputs

            # so each token location has a embedding
            # same temporal location has the same position embedding

            # pos_embed_spatial: [1, 56*56, emb_dim]
            # pos_embed_temporal: [1, 8, emb_dim]
            # -> [1, 8*56*56, 96]
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )

            if self.cls_embed_on:
                # positional embedding for the class [token] embedding
                # pos_embed_class: (1, 1, emb_dim)
                # -> [1, 1+T*H*W, 96]
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)

            # [B, 1 + T*H*W, 96]; broadcast to all sample in the mini-batch
            # all samples, has the same positional embedding vectors at the same location
            # and the class embedding is the same
            x = x + pos_embed

            # x[B, 0, emb_dim] all equal across different batch

        else:
            x = x + self.pos_embed

        if self.drop_rate:
            # drop_out=0.5 here
            x = self.pos_drop(x)

        if self.norm_stem:  # default false
            # layernorm
            x = self.norm_stem(x)

        # 8, 56, 56 for 16x224x224 inputs
        thw = [T, H, W]
        # [B, 1 + T*H*W, 96] -> # [B, 1 + T*new_hw, new_channel]
        for blk in self.blocks:
            # x.size == T*H*W + 1, +1 is because CLS_EMBED_ON = True
            x, thw = blk(x, thw)
        # thw changes from 56x56 to 7x7, t is still 8 (16/2) for 16x4 model
        # so the spatial downsample is 32x

        # layer norm
        if self.norm:
            x = self.norm(x)
        # for MViT model
        # POOL_Q_STRIDE: [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
        # torch.Size([1, 392, 768])
        # if POOL_Q_STRIDE: [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 1, 1]]
        # torch.Size([1, 1568, 768])

        # [B, (1+)T*H*W, c=768], 1+ for class embedding
        if not self.use_multi_head:
            if self.use_roi_head:
                T, H, W = thw
                channel = x.shape[-1]
                if self.cls_embed_on:  # 16x4 model has cls_embedding on
                    x = x[:, 1:, :]  # ignore cls embedding for now

                    x = x.reshape(B, T, H, W, channel)
                else:
                    x = x.reshape(B, T, H, W, channel)
                # [B, C, T, H, W]
                x = x.permute(0, 4, 1, 2, 3)
                #sys.exit()
                x = [x]  # slowfast format
                x = self.head(x, bboxes)
            else:
                if self.add_spatial_max_pool_before_proj:
                    # this will produce the same outputs as ROI Align
                    # if using the whole HW as box
                    # junwei: this is just a temporal function
                    if self.cls_embed_on:
                        x = x[:, 1:, :]  # ignore cls embedding for now
                    # [0,0,W,H] boxes
                    T, H, W = thw
                    channel = x.shape[-1]
                    x = x.reshape(B, T, H, W, channel)
                    x = x.mean(1)
                    # [B, C, H, W]
                    x = x.permute(0, 3, 1, 2)
                    # if the H and W != spatial resolution
                    feat_size = self.cfg.DATA.TEST_CROP_SIZE // self.cfg.DETECTION.SPATIAL_SCALE_FACTOR

                    # ONNX will include an if route if used H
                    #if H != self.cfg.DETECTION.ROI_XFORM_RESOLUTION:
                    if feat_size != self.cfg.DETECTION.ROI_XFORM_RESOLUTION:
                        roi_size = self.cfg.DETECTION.ROI_XFORM_RESOLUTION
                        x = torch.nn.functional.interpolate(
                            x,
                            size=(roi_size, roi_size),
                            mode="bilinear", align_corners=True)

                    # this is not supported by ONNX opset 12
                    #x = x.amax((-2, -1), keepdim=False)
                    #b = x.amax((-2, -1))
                    x, _ = x.max(dim=3)
                    x, _ = x.max(dim=2)
                    #assert torch.allclose(b, x)
                else:
                    if self.cls_embed_on:
                        # x is [B, 1+T*new_hw, channel]
                        x = x[:, 0]  # THW+1, first dim is cls_emb
                    else:
                        # [B, T*new_hw, channel]
                        x = x.mean(1)

                # [B, channel] -> [B, classes]  # or in the contrastive learning
                # case, [B, emb_dim]
                x = self.head(x)
        else:
            # for multi dataset forward,
            # only classification supported for now
            if self.cls_embed_on:
                # x is [B, 1+T*new_hw, channel]
                x = x[:, 0]  # THW+1, first dim is cls_emb
            else:
                x = x.mean(1)
            # we have x [B, channel], will output {dataset_name: [B, num_classes]}

            if self.cfg.MODEL.USE_VICREG_LOSS:
                # x is considered as the video representation
                # apply the robust loss on x
                return self.head(
                    x, dataset_name, run_cross_proj=run_cross_proj,
                    use_moco=use_moco, moco_momentum=moco_momentum), x
            else:
                x = self.head(
                        x, dataset_name, run_cross_proj=run_cross_proj,
                        use_moco=use_moco, moco_momentum=moco_momentum)
        return x



# TODO(junweil): think of a better name
@MODEL_REGISTRY.register()
class ActionCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # init parameters within the module
        if self.cfg.MODEL.LOAD_VISUAL:
            self.video_encoder = MViT(self.cfg)

        # init parameters within the module
        self.text_encoder = Transformer(
            width=self.cfg.CONTRA.transformer_width,
            layers=self.cfg.CONTRA.transformer_layers,
            heads=self.cfg.CONTRA.transformer_heads,
            context_length=self.cfg.CONTRA.CONTEXT_LENGTH,
            vocab_size=self.cfg.CONTRA.vocab_size,
            embed_dim=self.cfg.CONTRA.embed_dim,  # the projection joint space's dimension
            use_gradient_checkpoint=self.cfg.MODEL.ACT_CHECKPOINT,
            use_MLP=self.cfg.CONTRA.use_MLP,
            dropout_rate=self.cfg.MODEL.DROPOUT_RATE,
        )

        # the learnable temperture parameter as in the paper
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.cfg.TRAIN.USE_MOCO:
            # TODO(junwei): add extra predictor head for encoder but not momentum encoder
            # like MoCo v3 paper and BYOL paper

            # momentum encoder for both video and text

            """
            self.video_encoder_moco = MViT(self.cfg)
            self.text_encoder_moco = Transformer(
                width=self.cfg.CONTRA.transformer_width,
                layers=self.cfg.CONTRA.transformer_layers,
                heads=self.cfg.CONTRA.transformer_heads,
                context_length=self.cfg.CONTRA.CONTEXT_LENGTH,
                vocab_size=self.cfg.CONTRA.vocab_size,
                embed_dim=self.cfg.CONTRA.embed_dim,  # the projection joint space's dimension
                use_gradient_checkpoint=self.cfg.MODEL.ACT_CHECKPOINT,
                use_MLP=self.cfg.CONTRA.use_MLP,
                dropout_rate=self.cfg.MODEL.DROPOUT_RATE,
            )
            """

            self.video_encoder_moco = deepcopy(self.video_encoder)
            self.text_encoder_moco = deepcopy(self.text_encoder)

            # copy the initial params from the original encoder to the moco encoder
            # these momentum encoder params will be saved to checkpoint as well
            for param_b, param_m in zip(self.video_encoder.parameters(), self.video_encoder_moco.parameters()):
                param_m.data.copy_(param_b.data)
                param_m.requires_grad = False

            for param_b, param_m in zip(self.text_encoder.parameters(), self.text_encoder_moco.parameters()):
                param_m.data.copy_(param_b.data)
                param_m.requires_grad = False


    def forward(self, frames, tokens, use_moco=False, moco_momentum=0.99):
        # check https://github.com/mlfoundations/open_clip/blob/main/src/clip/model.py

        # given [B, C, T, H, W], [B, L]

        # [B, emb_dim/512]
        text_features = self.text_encoder.encode_text(tokens)

        # [B, emb_dim/512]
        video_features = self.video_encoder(frames)

        # TODO: add predictor layer (2-layer MLP, not used by moco encoder) as the MoCo V3 paper

        # L2 norm
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if use_moco:
            with torch.no_grad():
                self._moco_update(moco_momentum)

                text_features_moco = self.text_encoder_moco.encode_text(tokens)
                video_features_moco = self.video_encoder_moco(frames)
                video_features_moco = video_features_moco / video_features_moco.norm(dim=-1, keepdim=True)
                text_features_moco = text_features_moco / text_features_moco.norm(dim=-1, keepdim=True)

            return video_features, text_features, self.logit_scale.exp(), video_features_moco, text_features_moco

        else:

            return video_features, text_features, self.logit_scale.exp()


    @torch.no_grad()
    def _moco_update(self, momentum):
        for param_b, param_m in zip(self.video_encoder.parameters(), self.video_encoder_moco.parameters()):
            param_m.data = param_m.data * momentum + param_b.data * (1 - momentum)

        for param_b, param_m in zip(self.text_encoder.parameters(), self.text_encoder_moco.parameters()):
            param_m.data = param_m.data * momentum + param_b.data * (1 - momentum)

