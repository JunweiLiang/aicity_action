#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import numpy
import torch
import torch.nn as nn

from slowfast.models.common import DropPath, Mlp


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None, pool2d=None):
    # junwei: pool2d is for TNN, it does not support [1,3,3] maxpool3d
    if pool is None:
        return tensor, thw_shape

    # tensor could be multi-head
    # # [B, num_head, thw, C // num_head]
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        # [B, THW, dim] -> [B, 1, THW, dim]
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    # N is the number of head
    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    """  # junwei: test for pool2d vs pool3d
    if pool2d is not None:
        tensor_test = pool2d(tensor.view(-1, H, W)).view(B * N, C, T, H//2, W//2)
    #print(tensor.shape) # N, C, T, H, W
    # pool kernel/stride/pad is # [1, 3, 3] [1, 2, 2] [0, 1, 1]
    tensor = pool(tensor)
    if pool2d is not None:
        print(torch.allclose(tensor, tensor_test))
    """
    # junwei: change to use pool2d
    if pool2d is not None:
        # https://discuss.pytorch.org/t/ceil-mode-in-pooling/54646/2
        # output_hw size: floor((W-K+2P)/S) + 1
        # if K=3, P=1, S=2
        # so floor[(W - 1)/2] + 1 == W/2 if W is even number,
        # for odd number it is W/2 + 1
        tensor = pool2d(tensor.view(-1, H, W)).view(B * N, C, T, H//2, W//2)
    else:
        tensor = pool(tensor)  # could be maxpool or conv

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, tensor.shape[1], L_pooled).transpose(2, 3)

    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
        L_pooled = torch.add(L_pooled, 1)

    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    # 3 [8, 14, 14] torch.Size([1, 1, 1568, 384])
    # 4 [8, 7, 7] torch.Size([1, 4, 392, 96])
    #print(tensor_dim, thw_shape, tensor.shape)
    # pool: Conv3d or MaxPool3d
    #print(tensor_dim, thw_shape, isinstance(pool, nn.MaxPool3d))
    #if tensor_dim == 4:
    #    pass
    #else:  #  tensor_dim == 3:
    if tensor_dim == 3:
        # [B, 1, THW, dim] -> [B, THW, dim]
        # this with CLS_EMBED_ON=False, ONNX will generate onnx::If
        # and TensorRT v8.0 will complain
        #tensor = tensor.squeeze(1)
        tensor = tensor.reshape(B, L_pooled, tensor.shape[3])
    return tensor, thw_shape


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        pool_first=False,
        # added by junwei for MViT version 2
        use_query_residual_pool=False,
        # we expand the channel in the last project?
        expand_channel=False,
        expand_to_dim=None,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads

        dim_in = dim
        dim_out = dim
        if expand_channel:
            dim_out = expand_to_dim
        self.dim_out = dim_out

        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim_in, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode == "avg":
            self.pool_q = (
                nn.AvgPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "max":
            self.pool_q = (
                nn.MaxPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )

        elif mode == "conv":  # this is default for MViT 16x4, 32x3 model from the paper

            # input is # [B, num_head, thw, C // num_head]
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")


        # junwei: https://arxiv.org/pdf/2112.01526v1.pdf

        self.use_query_residual_pool = use_query_residual_pool

    def forward(self, x, thw_shape):

        # N is thw
        B, N, C = x.shape
        C = self.dim_out

        # x linearly projected to query, key, value tensors
        # [B, N, C] -> [B, N, 3*C] -> each [B, N, num_head, C//num_head]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # [B, num_head, thw, C // num_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # pool with Conv3D
        # [B, num_head, thw, C // num_head] -> [B, num_head, new_thw, C // num_head]
        # so the final output only depends on query's new_thw
        q, out_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, _ = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, _ = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        # matmul
        # each token and head in query attends each token and head in key
        # q [B, num_head, new_thw, dim] -> k [B, num_head, dim, new_thw']
        # new_thw and new_thw' could be different
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn: [B, num_head, new_thw, new_thw']
        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        # attn [B, num_head, new_thw, new_thw'] -> v [B, num_head, new_thw', dim]
        # -> [B, num_head, new_thw, dim]
        # -> [B, new_thw (out_shape), num_head*dim]
        # so we have attention from each token in the query to all of value's token,
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.use_query_residual_pool:
            # q [B, num_head, new_thw, dim]
            x = x + q.transpose(1, 2).reshape(B, N, C)  # this does not add to FLOPs?
        # linear, dim -> dim
        x = self.proj(x)
        if self.drop_rate > 0.0:  # drop out, 0.0 by default
            x = self.proj_drop(x)
        return x, out_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        use_query_residual_pool=False,
        channel_expand_front=False,
        pool_skip_use_conv=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]

        # junwei: MViT Version 2, dim became 2*dim during attention compute
        expand_channel = False
        dim_in = dim
        if channel_expand_front:
            if dim != dim_out:
                expand_channel = True
        self.expand_channel = expand_channel
        self.pool_skip_use_conv = pool_skip_use_conv

        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
            use_query_residual_pool=use_query_residual_pool,
            expand_channel=expand_channel,
            expand_to_dim=dim_out,
        )
        if self.expand_channel:
            dim = dim_out
            self.dim = dim_out

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)  # normalize along last dimension
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed

        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        # this does not seem to be used in MViT V2
        """
        if pool_skip_use_conv:
            self.pool_skip = (
                nn.Conv3d(
                    dim_in, dim_out,
                    kernel_skip, stride_skip, padding_skip, bias=False
                )
                if len(kernel_skip) > 0
                else None
            )
            self.pool_skip_norm = nn.LayerNorm(dim_out)
        else:
        """
        if self.expand_channel:
            self.proj_max_pool = nn.Linear(dim_in, dim_out)

        # pooling for the skip connections
        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(kernel_skip) > 0
            else None
        )
        self.pool_skip_norm = None
        #print(kernel_skip)  # empty for 16x4 and 32x3 model
        # junwei: 12/2021, use MaxPool2d instead
        # for both 16x4, 32x3, some blocks are [] some are [1, 3, 3] for kernel_skip
        # [1, 3, 3] [1, 2, 2] [0, 1, 1]
        #print(kernel_skip, stride_skip, padding_skip)
        """
        self.pool_skip2d = (
            nn.MaxPool2d(
                kernel_skip[1:], stride_skip[1:], padding_skip[1:], ceil_mode=False
            )
            if len(kernel_skip) > 0 and stride_skip[1] != 1
            else None
        )
        """

    def forward(self, x, thw_shape):
        # x: [B, (1+)thw, channel]
        # layernorm -> attention
        # [8, 56, 56] [8, 28, 28], changed 3 times for MViT 16x4
        # there are attention_pool within attn() as well

        # here the attentionblock will pool the feature from [B, THW, C]
        # to [B, new_THW, C]
        # the query of self attention block determines the output shape
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        # attention pool
        # x_res: torch.Size([1, 6273, 192])
        if not self.pool_skip_use_conv and self.expand_channel:
            # need to change the channel for the residual connection
            x = self.proj_max_pool(x)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed,
            norm=self.pool_skip_norm,
            # pool2d does not support (2, 3, 3) yet
            #pool2d=self.pool_skip2d  # not using it during training/testing
        )
        # x_res should be pooled to thw_shape_new as well
        x = x_res + self.drop_path(x_block)

        x_norm = self.norm2(x)

        x_mlp = self.mlp(x_norm)

        # in this case x is normed before +, but it may not be normed if dim equals
        if self.dim != self.dim_out:
            # residual connect for the MLP block
            x = self.proj(x_norm)

        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new
