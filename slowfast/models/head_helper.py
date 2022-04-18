# coding=utf-8

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
from slowfast.models.common import Mlp
from detectron2.layers import ROIAlign

from functools import partial

from copy import deepcopy

class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,  # scale the input box by this
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            #print(inputs[pathway].size())  # [B, C, T, H, W]
            out = t_pool(inputs[pathway])  #[B, C, 1, H, W]
            #print(out.size())
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            # N -> M box features
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.act(x)  # during training and test we will both use sigmoid
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class X3DHead(nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        bn_lin5_on=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.

        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(X3DHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self._construct_head(dim_in, dim_inner, dim_out, norm_module)

    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):

        self.conv_5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.conv_5_bn = norm_module(
            num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt
        )
        self.conv_5_relu = nn.ReLU(self.inplace_relu)

        if self.pool_size is None:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1)

        self.lin_5 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(
                num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
            )
        self.lin_5_relu = nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        if self.act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif self.act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(self.act_func)
            )

    def forward(self, inputs):
        # In its current design the X3D head is only useable for a single
        # pathway input.
        assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool(x)

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        use_act_in_train=False,
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        self.use_act_in_train = use_act_in_train

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # remember to use bce_logits for loss for sigmoid head
        if self.use_act_in_train or not self.training:
            x = self.act(x)
        return x

class ContrastiveProjectionHead(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_MLP=False,
        dropout_rate=0.0,
    ):
        """
         Given [B, D], layernorm -> linear
        """
        super(ContrastiveProjectionHead, self).__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(dim_in)
        # WenLan paper and MoCo v2 paper uses 2-layer MLP, 2048-d, RELU
        if use_MLP:
            self.projection = Mlp(
                in_features=dim_in,
                hidden_features=2048,
                out_features=dim_out,
                act_layer=nn.GELU,
                drop_rate=dropout_rate,
            )
        else:
            self.projection = nn.Linear(dim_in, dim_out, bias=False)



    def forward(self, x):
        x = self.norm(x)
        x = self.projection(x)
        return x


def get_act_func(func_name):
    if func_name == "softmax":
        return nn.Softmax(dim=1)
    elif func_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(func_name)
        )

class TransformerMultiHead(nn.Module):
    """
    multiple classification head
    """

    def __init__(
        self,
        dim_in,
        dataset_names,
        dataset_num_classes,
        act_funcs,
        dropout_rate=0.0,
        use_MLP=False,
        add_cross_proj=False,  # add pair-wise dataset class projection layers
        use_moco=False, # use moco encoder for the cross dataset proj part
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerMultiHead, self).__init__()
        if dropout_rate > 0.0 and not use_MLP:
            # we will use dropout within MLP if used
            self.dropout = nn.Dropout(dropout_rate)

        # for cross_entropy loss, we dont use act during training , but bce need it
        self.heads = {}
        self.acts = {}

        self.cross_dataset_heads = {}  # dataset-to-dataset projection
        self.add_cross_proj = add_cross_proj
        self.use_moco = use_moco

        assert len(dataset_names) == len(dataset_num_classes) == len(act_funcs)
        for i, dataset_name in enumerate(dataset_names):
            num_classes = dataset_num_classes[i]
            if use_MLP:
                self.heads[dataset_name] = Mlp(
                    in_features=dim_in,
                    hidden_features=2048,
                    out_features=num_classes,
                    act_layer=nn.GELU,
                    drop_rate=dropout_rate,
                )
            else:
                self.heads[dataset_name] = nn.Linear(dim_in, num_classes, bias=True)
            self.acts[dataset_name] = get_act_func(act_funcs[i])

            if self.add_cross_proj:
                for j, other_dataset_name in enumerate(dataset_names):
                    if other_dataset_name == dataset_name:
                        continue
                    proj_name = "%s_%s" % (other_dataset_name, dataset_name)
                    other_dataset_num_classes = dataset_num_classes[j]

                    # mit_k700
                    # 305 x 700
                    self.cross_dataset_heads[proj_name] = nn.Linear(
                        other_dataset_num_classes, num_classes, bias=False)


        self.heads = nn.ModuleDict([[k, self.heads[k]] for k in self.heads])
        self.acts = nn.ModuleDict([[k, self.acts[k]] for k in self.acts])

        if self.add_cross_proj:
            self.cross_dataset_heads = nn.ModuleDict(
                [[k, self.cross_dataset_heads[k]]
                 for k in self.cross_dataset_heads])

            if self.use_moco:
                self.heads_moco = deepcopy(self.heads)

    def init_moco(self):
        for param_b, param_m in zip(self.heads.parameters(), \
                self.heads_moco.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def _moco_update(self, momentum):
        for param_b, param_m in zip(self.heads.parameters(), \
                self.heads_moco.parameters()):
            param_m.data = param_m.data * momentum + param_b.data * (1 - momentum)

    def forward(self, x, dataset_name=None,
                run_cross_proj=False, use_moco=False, moco_momentum=0.9):
        """
        output {dataset_name: outputs}
        """
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        head_outputs = {}

        if use_moco:
            self._moco_update(moco_momentum)
            head_outputs_moco = {}

        if dataset_name is None:
            run_names = self.heads.keys()
        else:
            run_names = [dataset_name]
            assert dataset_name in self.heads.keys()

        for dataset_name in run_names:
            x_head = self.heads[dataset_name](x)

            # no activation func during training,
            # so for sigmoid head remember to use bce_logit for loss
            if not self.training:
                x_head = self.acts[dataset_name](x_head)

            head_outputs[dataset_name] = x_head
            if use_moco:
                head_outputs_moco[dataset_name] = self.heads_moco[dataset_name](x)

        # should only be used during training
        if self.add_cross_proj and run_cross_proj:
            assert self.training, "cross dataset projection is not supposed to be used during inf"


            for d1_d2 in self.cross_dataset_heads.keys():
                d1_name, d2_name = d1_d2.split("_")
                # so the output should be the same dim as d2_name
                if use_moco:
                    proj_inputs = head_outputs_moco[d1_name]
                else:
                    proj_inputs = head_outputs[d1_name]
                # junweil: should we add softmax here?
                #proj_inputs = self.acts[d1_name](proj_inputs)
                head_outputs[d1_d2] = self.cross_dataset_heads[d1_d2](proj_inputs)

        return head_outputs
