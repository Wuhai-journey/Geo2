import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from IPython import embed

from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
from timm.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def build_dropout_layer(p: Optional[float], **kwargs) -> nn.Module:
    r"""Factory function for dropout layer."""
    if p is None or p == 0:
        return nn.Identity()
    else:
        return nn.Dropout(p=p, **kwargs)

class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None,):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.k=35


        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_q1 = nn.Linear(d_model, d_model) 
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_p = nn.Linear(d_model, d_model) 

        self.dropout = build_dropout_layer(dropout)

    def forward(self, input_q, input_k, input_v, rpe_knn_embeddings, knn_idx, key_weights=None, key_masks=None, attention_factors=None):

        B, N, C = input_q.shape
        H = self.num_heads
        D = self.d_model_per_head
        k_sp = min(self.k, N)

        q = self.proj_q(input_q).view(B, N, H, D)
        q1 = self.proj_q1(input_q).view(B, N, H, D)
        k_all = self.proj_k(input_k)
        v_all = self.proj_v(input_v)

        k_neighbors = knn_gather(k_all, knn_idx).contiguous() # (B, N, k, C)
        v_neighbors = knn_gather(v_all, knn_idx).contiguous() # (B, N, k, C)

        k_neighbors = k_neighbors.view(B, N, k_sp, H, D)
        v_neighbors = v_neighbors.view(B, N, k_sp, H, D)

        p = self.proj_p(rpe_knn_embeddings).contiguous().view(B, N, k_sp, H, D)
        attn_e = torch.einsum('bnhd,bnkhd->bnhk', q, k_neighbors)

        attn_p = torch.einsum('bnhd,bnkhd->bnhk', q1, p)

        attention_scores = (attn_e + attn_p) / (D ** 0.5)

        attention_probs = F.softmax(attention_scores, dim=-1) # (B, N, H, k)
        attention_probs = self.dropout(attention_probs)

        hidden_states = torch.einsum('bnhk,bnkhd->bnhd', attention_probs, v_neighbors)
        
        hidden_states = hidden_states.reshape(B, N, C)

        return hidden_states, attention_probs


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        knn_idx,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            knn_idx,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        knn_idx,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            knn_idx,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores

def build_act_layer(act_cfg: Optional[Union[str, Dict]]) -> nn.Module:
    r"""Factory function for activation functions."""
    if act_cfg is None:
        return nn.Identity()
    layer, kwargs = parse_cfg(act_cfg)
    assert layer in ACT_LAYERS, f'Illegal activation: {layer}.'
    if layer == 'LeakyReLU':
        if 'negative_slope' not in kwargs:
            kwargs['negative_slope'] = 0.2
    return ACT_LAYERS[layer](**kwargs)

class AttentionOutput(nn.Module):
    def __init__(self, d_model, dropout=None, activation_fn='ReLU'):
        super(AttentionOutput, self).__init__()
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = build_act_layer(activation_fn)
        self.squeeze = nn.Linear(d_model * 2, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states):
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(input_states + hidden_states)
        return output_states

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02, 
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim) 
        self.norm = norm_cls(dim)

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states  
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype)) 
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params) 
        return hidden_states, residual 

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block



class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos=None, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        if pos is None:
            hidden_states = hidden_states
        else:
            hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            ) 
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states 
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype)) 
        else:

            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states
