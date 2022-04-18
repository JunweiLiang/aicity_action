# coding=utf-8
"""
Definition of some text models
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from slowfast.models.common import Mlp
from collections import OrderedDict


from slowfast.models.utils import validate_checkpoint_wrapper_import

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None

# junwei: let's directly use torch layernorm and see what happens
from torch.nn import LayerNorm
"""
class LayerNorm(nn.LayerNorm):
    #Subclass torch's LayerNorm to handle fp16.

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
"""

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask  # T x T of -inf

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 # added by junwei for using the text model only
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 embed_dim: int = 512,
                 use_gradient_checkpoint: bool = False,
                 dropout_rate: float = 0.5,
                 use_MLP: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.context_length = context_length
        if attn_mask is None:
            # lazily create causal attention mask, with full attention between the vision tokens
            # pytorch uses additive attention mask; fill with -inf
            mask = torch.empty(self.context_length, self.context_length)
            mask.fill_(float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal
            attn_mask = mask

        # unwrap this so we could add gradient checkpointing
        """
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)])
        """

        # activation checkpointing to save GPU memory
        if self.use_gradient_checkpoint:
            # check for
            # from fairscale.nn.checkpoint import checkpoint_wrapper
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        self.resblocks = nn.ModuleList()
        for i in range(layers):

            attention_block = ResidualAttentionBlock(width, heads, attn_mask)

            if self.use_gradient_checkpoint:
                attention_block = checkpoint_wrapper(attention_block)

            self.resblocks.append(attention_block)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = LayerNorm(width)

        # WenLan paper and MoCo v2 paper uses 2-layer MLP, 2048-d, RELU
        self.use_MLP = use_MLP
        if self.use_MLP:
            self.text_projection =  Mlp(
                in_features=width,
                hidden_features=2048,
                out_features=embed_dim,
                act_layer=nn.GELU,
                drop_rate=dropout_rate,
            )
        else:
            self.text_projection = nn.Parameter(torch.empty(width, embed_dim))


        self.initialize_parameters()

        #self.dtype = self.text_projection.dtype


    def forward(self, x: torch.Tensor):
        #return self.resblocks(x)  # sequential
        for blk in self.resblocks:
            x = blk(x)
        return x


    def encode_text(self, text):
        # B x L (text) -> B x L x emb_dim
        # text.argmax(dim=-1) -> the eot_token
        x = self.token_embedding(text) # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.forward(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.use_MLP:
            x = self.text_projection(x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    # junwei: so print(model) can show extra nn.parameters as well
    def extra_repr(self):

        # save the module names that will be printed in __repr__ already
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '('+ name +'): ' \
                    +'tensor(' + str(tuple(p[1].shape))+ ', requires_grad='+ str(p[1].requires_grad) +')\n'

        return string_repr


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)


        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            # for nn.Parameters
            if not self.use_MLP:
                nn.init.normal_(self.text_projection, std=self.width ** -0.5)
            else:
                nn.init.normal_(self.text_projection.fc1.weight, std=self.width ** -0.5)
                nn.init.normal_(self.text_projection.fc2.weight, std=self.width ** -0.5)
                nn.init.constant_(self.text_projection.fc2.bias, 0)
                nn.init.constant_(self.text_projection.fc1.bias, 0)
