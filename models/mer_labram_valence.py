# Derived from AdaBrain-Bench, LaBraM and CBraMod; see THIRD_PARTY_NOTICES.md.
from util.model_storage import load_pretrained
'Complete FIPT (LaBraM) model for MER valence.\nAll model components are defined in this file; parameter names retain source compatibility.\n'
from pyexpat import model
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import util.utils as utils
from functools import partial
from scipy.signal import welch
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from einops import rearrange

class front_model_LinearWithConstraint(nn.Linear):

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(front_model_LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(front_model_LinearWithConstraint, self).forward(x)

def front_model__cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), **kwargs}

class front_model_DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(front_model_DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class front_model_Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class front_model_Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.0, proj_drop=0.0, window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer('relative_position_index', relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if return_attention:
            return attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_qkv:
            return (x, qkv)
        return x

class front_model_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = front_model_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = front_model_DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = front_model_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = (None, None)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return (x, qkv)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class front_model_PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class front_model_TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, in_chans=1, out_chans=8):
        """
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x

class front_model_Front_NeuralTransformer(nn.Module):

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_values=0.1, use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001, num_ch=22, num_t=4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size
        self.psd_proj = nn.Linear(101, embed_dim)
        self.ctx = nn.Parameter(torch.zeros(16, embed_dim))
        self.pos_prompt_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
            self.new_pos_embed = nn.Parameter(torch.zeros(1, num_ch + 1, embed_dim), requires_grad=True)
        if num_t > 16:
            self.time_embed = nn.Parameter(torch.zeros(1, 32, embed_dim), requires_grad=True)
        else:
            self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.rel_pos_bias = None
        self.use_rel_pos_bias = use_rel_pos_bias
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, psd=None, shape_x=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        psd = self.psd_proj(psd)
        batch_size, n, a, t = shape_x
        input_time_window = a if t == self.patch_size else t
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        ctx = self.ctx.expand(batch_size, -1, -1)
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.new_pos_embed
        if pos_embed_used is not None:
            x_pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            cls_pos_embed = pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1)
            x = x + x_pos_embed
            cls_tokens = cls_tokens + cls_pos_embed
            ctx = ctx + self.pos_prompt_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
            x += time_embed
        x = torch.cat((cls_tokens, x, psd, ctx), dim=1)
        x = self.pos_drop(x)
        return x

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        """
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        """
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f'Not support for layer id is {layer_id} now!')

    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)
        return features

def front_model_front_labram_base_patch200_200(pretrained=False, **kwargs):
    model = front_model_Front_NeuralTransformer(patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-06), norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)
    model.default_cfg = front_model__cfg()
    return model

class mid_model_LinearWithConstraint(nn.Linear):

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(mid_model_LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(mid_model_LinearWithConstraint, self).forward(x)

def mid_model__cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), **kwargs}

class mid_model_DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(mid_model_DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class mid_model_Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class mid_model_Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.0, proj_drop=0.0, window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer('relative_position_index', relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if return_attention:
            return attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_qkv:
            return (x, qkv)
        return x

class mid_model_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mid_model_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = mid_model_DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mid_model_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = (None, None)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return (x, qkv)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class mid_model_PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class mid_model_TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, in_chans=1, out_chans=8):
        """
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x

class mid_model_Mid_NeuralTransformer(nn.Module):

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_values=0.1, use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001, num_ch=22, num_t=4, **kwargs):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([mid_model_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=None) for i in range(depth)])

def mid_model_mid_labram_base_patch200_200(pretrained=False, **kwargs):
    model = mid_model_Mid_NeuralTransformer(patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-06), norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)
    model.default_cfg = mid_model__cfg()
    return model

class rear_model_LinearWithConstraint(nn.Linear):

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(rear_model_LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(rear_model_LinearWithConstraint, self).forward(x)

def rear_model__cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), **kwargs}

class rear_model_DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(rear_model_DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class rear_model_Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class rear_model_Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.0, proj_drop=0.0, window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer('relative_position_index', relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if return_attention:
            return attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_qkv:
            return (x, qkv)
        return x

class rear_model_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = rear_model_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = rear_model_DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = rear_model_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = (None, None)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return (x, qkv)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class rear_model_PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class rear_model_TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, in_chans=1, out_chans=8):
        """
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x

class rear_model_Rear_NeuralTransformer(nn.Module):

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_values=0.1, use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001, num_ch=22, num_t=4, **kwargs):
        super().__init__()
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        elif return_all_tokens:
            return x
        elif return_patch_tokens:
            return x[:, 1:]
        else:
            return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        """
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        """
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f'Not support for layer id is {layer_id} now!')

    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)
        return features

def rear_model_rear_labram_base_patch200_200(pretrained=False, **kwargs):
    model = rear_model_Rear_NeuralTransformer(patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-06), norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)
    model.default_cfg = rear_model__cfg()
    return model

class patch_block_LinearWithConstraint(nn.Linear):

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(patch_block_LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(patch_block_LinearWithConstraint, self).forward(x)

def patch_block__cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), **kwargs}

class patch_block_DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(patch_block_DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class patch_block_Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class patch_block_Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.0, proj_drop=0.0, window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer('relative_position_index', relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if return_attention:
            return attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_qkv:
            return (x, qkv)
        return x

class patch_block_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = patch_block_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = patch_block_DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = patch_block_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = (None, None)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return (x, qkv)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class patch_block_PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class patch_block_TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, in_chans=1, out_chans=8):
        """
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x

class patch_block_Patch_NeuralTransformer(nn.Module):

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.0, qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_values=0.1, use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001, num_ch=22, num_t=4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = patch_block_TemporalConv(out_chans=out_chans) if in_chans == 1 else patch_block_PatchEmbed(EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x = self.patch_embed(x)
        return x

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        """
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        """
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f'Not support for layer id is {layer_id} now!')

    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)
        return features

def patch_block_patch_labram_base_patch200_200(pretrained=False, **kwargs):
    model = patch_block_Patch_NeuralTransformer(patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-06), norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)
    model.default_cfg = patch_block__cfg()
    return model

class LinearWithConstraint(nn.Linear):

    def __init__(self, *args, doWeightNorm=True, max_norm=1, flatten=0, dropout=0, patch_mean=False, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        self.flatten = flatten
        self.patch_mean = patch_mean
        self.drop_out = nn.Dropout(p=dropout) if dropout else None
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.flatten:
            x = x.flatten(self.flatten)
        elif self.patch_mean:
            x = x.reshape(x.shape[0], -1, x.shape[-1]).mean(1)
        if self.drop_out is not None:
            x = self.drop_out(x)
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(LinearWithConstraint, self).forward(x)

def setup_finetuning_parameters(model: nn.Module):
    """
    根据指定策略设置模型的参数梯度。
    - 冻结除了 cls_token, pos_embed, time_embed, 分类头, 所有 LayerNorm 参数之外的所有权重。
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
    print(f'--- 初始状态: 所有 {len(list(model.parameters()))} 个参数已冻结 ---')
    unfrozen_params = []
    if hasattr(model, 'ctx') and model.ctx is not None:
        model.ctx.requires_grad = True
        unfrozen_params.append('ctx')
    if hasattr(model, 'pos_prompt_embed') and model.pos_prompt_embed is not None:
        for param in model.psd_proj.parameters():
            param.requires_grad = True
        unfrozen_params.append('psd_proj.*')
    if hasattr(model, 'new_pos_embed'):
        model.new_pos_embed.requires_grad = True
        unfrozen_params.append('new_pos_embed')
    if hasattr(model, 'psd_proj'):
        for param in model.psd_proj.parameters():
            param.requires_grad = True
        unfrozen_params.append('psd_proj.*')
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        model.pos_embed.requires_grad = True
        unfrozen_params.append('pos_embed')
    if hasattr(model, 'time_embed') and model.time_embed is not None:
        model.time_embed.requires_grad = True
        unfrozen_params.append('time_embed')
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
        unfrozen_params.append('head.*')
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True
                unfrozen_params.append(f'{name}.*')
    total_trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    total_all_params = sum((p.numel() for p in model.parameters()))
    print('\n--- 微调策略总结 ---')
    print(f'总参数量: {total_all_params:,}')
    print(f'可训练参数量: {total_trainable_params:,}')
    print(f'可训练参数占比: {total_trainable_params / total_all_params:.2%}')
    print('\n已解冻的参数集合 (或类型):')
    print(set([name.split('.')[0] for name in unfrozen_params]))
    return model

def load_model_dict(args, ch_names, num_t, use_abs_pos_emb, model_name, from_pretrain=True):
    model = create_model(model_name, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop, drop_path_rate=args.drop_path, attn_drop_rate=args.attn_drop_rate, drop_block_rate=None, use_mean_pooling=args.use_mean_pooling, init_scale=args.init_scale, use_rel_pos_bias=args.rel_pos_bias, use_abs_pos_emb=use_abs_pos_emb, init_values=args.layer_scale_init_value, qkv_bias=args.qkv_bias, num_ch=ch_names, num_t=num_t)
    model = setup_finetuning_parameters(model)
    if from_pretrain and (not getattr(args, 'from_scratch', False)):
        if args.pretrained.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.pretrained, map_location='cpu', check_hash=True)
        else:
            checkpoint = load_pretrained(args)
        print('Load ckpt from %s' % args.pretrained)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print('Load state_dict by model_key = %s' % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if checkpoint_model is not None and args.model_filter_name != '':
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f'Removing key {k} from pretrained checkpoint')
                del checkpoint_model[k]
        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if 'relative_position_index' in key:
                checkpoint_model.pop(key)
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    return model

class DenseCoAttn(nn.Module):
    """
    实现多模态的交叉注意力机制。
    每个模态的 Query 会查询一个由所有模态融合而成的 Key/Value。
    输出的 weightedX 会保持与输入 valueX 相同的序列长度和嵌入维度。
    """

    def __init__(self, embed_dim, dropout=0.1):
        """
        初始化 DenseCoAttn 模块。

        Args:
            embed_dim (int): 输入数据的嵌入维度（所有模态统一）。
            dropout (float): Dropout 概率。
        """
        super(DenseCoAttn, self).__init__()
        self.embed_dim = embed_dim
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(3)])
        self.query1_linear = nn.Linear(embed_dim, embed_dim)
        self.query2_linear = nn.Linear(embed_dim, embed_dim)
        self.query3_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ELU()

    def forward(self, value1, value2, value3):
        joint_kv_source = torch.cat((value1, value2, value3), dim=1)
        k_fused = self.key_linear(joint_kv_source)
        v_fused = self.value_linear(joint_kv_source)
        q1 = self.query1_linear(value1)
        q2 = self.query2_linear(value2)
        q3 = self.query3_linear(value3)
        weighted1, attn1 = self._scaled_dot_product_attention(q1, k_fused, v_fused, dropout=self.dropouts[0])
        weighted2, attn2 = self._scaled_dot_product_attention(q2, k_fused, v_fused, dropout=self.dropouts[1])
        weighted3, attn3 = self._scaled_dot_product_attention(q3, k_fused, v_fused, dropout=self.dropouts[2])
        return (weighted1, weighted2, weighted3)

    def _scaled_dot_product_attention(self, query, key, value, dropout=None):
        """
        标准的 Scaled Dot-Product Attention 计算流程。
        Args:
            query (torch.Tensor): 查询张量 (Batch, Seq_Q, Dim)。
            key (torch.Tensor): 键张量 (Batch, Seq_K, Dim)。
            value (torch.Tensor): 值张量 (Batch, Seq_K, Dim)。
            dropout (nn.Dropout, optional): Dropout 层。Defaults to None.

        Returns:
            tuple: (加权后的值 (Batch, Seq_Q, Dim), 注意力分数 (Batch, Seq_Q, Seq_K))。
        """
        d_k = query.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_k)
        scores = torch.tanh(scores)
        if dropout:
            scores = dropout(scores)
        weighted = torch.bmm(scores, value)
        weighted = torch.tanh(weighted)
        return (self.act(weighted), scores)

class NormalSubLayer1(nn.Module):
    """
    一个子层，包含 DenseCoAttn 模块和用于残差连接的线性层。
    """

    def __init__(self, embed_dim, dropout):
        """
        初始化 NormalSubLayer。

        Args:
            embed_dim (int): 嵌入维度。
            dropout (float): Dropout 概率。
        """
        super(NormalSubLayer1, self).__init__()
        self.dense_coattn = DenseCoAttn(embed_dim, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])
        self.residual_gates = nn.Parameter(torch.full((3,), 0.01))
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(p=dropout)) for _ in range(3)])

    def forward(self, data1, data2, data3):
        originals = (data1, data2, data3)
        normalized = [norm(x) for norm, x in zip(self.norms, originals)]
        weighted = self.dense_coattn(*normalized)
        outputs = []
        for i, (original, attended) in enumerate(zip(originals, weighted)):
            delta = self.linears[i](attended)
            scale = original.std(dim=-1, keepdim=True, unbiased=False)
            output = original + self.residual_gates[i] * scale * delta
            outputs.append(output)
        return tuple(outputs)

class NormalSubLayer2(nn.Module):
    """
    一个子层，包含 DenseCoAttn 模块和用于残差连接的线性层。
    """

    def __init__(self, embed_dim, dropout):
        """
        初始化 NormalSubLayer。

        Args:
            embed_dim (int): 嵌入维度。
            dropout (float): Dropout 概率。
        """
        super(NormalSubLayer2, self).__init__()
        self.dense_coattn = DenseCoAttn(embed_dim, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(p=dropout)) for _ in range(3)])

    def forward(self, data1, data2, data3):
        data1, data2, data3 = (self.norms[0](data1), self.norms[1](data2), self.norms[2](data3))
        weighted1, weighted2, weighted3 = self.dense_coattn(data1, data2, data3)
        data1 = data1 + self.linears[0](weighted1)
        data2 = data2 + self.linears[1](weighted2)
        data3 = data3 + self.linears[2](weighted3)
        return (data1, data2, data3)

class DCNLayer(nn.Module):
    """
    一个堆叠了多个 NormalSubLayer 的模块，形成一个深度交叉注意力网络。
    """

    def __init__(self, embed_dim, num_layers, dropout):
        super(DCNLayer, self).__init__()
        self.dcn_layers = nn.ModuleList([NormalSubLayer1(embed_dim, dropout) for _ in range(num_layers)])

    def fft_data(self, data):
        """Return spectral amplitudes and the original unit-complex phase."""
        bz, patch_num, patch_size = data.shape
        data = data.contiguous().view(bz * patch_num, patch_size)
        spectral = torch.fft.rfft(data, dim=-1, norm='forward')
        spectral = spectral.contiguous().view(bz, patch_num, 101)
        amplitude = spectral.abs()
        unit_phase = spectral / amplitude.clamp_min(1e-12)
        unit_phase = torch.where(amplitude > 1e-12, unit_phase, torch.ones_like(unit_phase))
        return (amplitude, unit_phase)

    def irfft_data(self, data):
        reconstructed_data = torch.fft.irfft(data, n=200, dim=-1, norm='forward')
        return reconstructed_data

    def forward(self, data1, data2, data3):
        data1 = data1.reshape(data1.shape[0], -1, data1.shape[-1])
        data2 = data2.reshape(data2.shape[0], -1, data2.shape[-1])
        data3 = data3.reshape(data3.shape[0], -1, data3.shape[-1])
        amplitude1, phase1 = self.fft_data(data1)
        amplitude2, phase2 = self.fft_data(data2)
        amplitude3, phase3 = self.fft_data(data3)
        combined1, combined2, combined3 = (amplitude1, amplitude2, amplitude3)
        for sub_layer in self.dcn_layers:
            combined1, combined2, combined3 = sub_layer(combined1, combined2, combined3)
        combined1 = combined1.clamp_min(0.0)
        combined2 = combined2.clamp_min(0.0)
        combined3 = combined3.clamp_min(0.0)
        resulting_spectral1 = self.irfft_data(phase1 * combined1)
        resulting_spectral2 = self.irfft_data(phase2 * combined2)
        resulting_spectral3 = self.irfft_data(phase3 * combined3)
        return (resulting_spectral1, resulting_spectral2, resulting_spectral3)

class DCNLayer2(nn.Module):
    """
    一个堆叠了多个 NormalSubLayer 的模块，形成一个深度交叉注意力网络。
    """

    def __init__(self, embed_dim, num_layers, dropout):
        super(DCNLayer2, self).__init__()
        self.dcn_layers = nn.ModuleList([NormalSubLayer2(embed_dim, dropout) for _ in range(num_layers)])

    def forward(self, data1, data2, data3):
        data1 = data1.reshape(data1.shape[0], -1, data1.shape[-1])
        data2 = data2.reshape(data2.shape[0], -1, data2.shape[-1])
        data3 = data3.reshape(data3.shape[0], -1, data3.shape[-1])
        for sub_layer in self.dcn_layers:
            data1, data2, data3 = sub_layer(data1, data2, data3)
        return (data1, data2, data3)

class Custom_LaBraM(nn.Module):

    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()
        self.t_a_patch_block = load_model_dict(args, 18, num_t, True, 'patch_labram_base_patch200_200')
        self.t_b_patch_block = load_model_dict(args, 1, num_t, False, 'patch_labram_base_patch200_200')
        self.t_c_patch_block = load_model_dict(args, 1, num_t, False, 'patch_labram_base_patch200_200')
        self.t_a_front_model = load_model_dict(args, 18, num_t, True, 'front_labram_base_patch200_200')
        self.t_b_front_model = load_model_dict(args, 1, num_t, False, 'front_labram_base_patch200_200')
        self.t_c_front_model = load_model_dict(args, 1, num_t, False, 'front_labram_base_patch200_200')
        self.t_a_mid_model = load_model_dict(args, 18, num_t, True, 'mid_labram_base_patch200_200')
        self.t_b_mid_model = load_model_dict(args, 1, num_t, False, 'mid_labram_base_patch200_200')
        self.t_c_mid_model = load_model_dict(args, 1, num_t, False, 'mid_labram_base_patch200_200')
        self.t_a_rear_model = load_model_dict(args, 18, num_t, True, 'rear_labram_base_patch200_200')
        self.t_b_rear_model = load_model_dict(args, 1, num_t, False, 'rear_labram_base_patch200_200')
        self.t_c_rear_model = load_model_dict(args, 1, num_t, False, 'rear_labram_base_patch200_200')
        self.t_a_rear_model.head = nn.Identity()
        self.t_b_rear_model.head = nn.Identity()
        self.t_c_rear_model.head = nn.Identity()
        self.task_head = None
        self.ch_names = ch_names
        self.fusion1 = DCNLayer(embed_dim=101, num_layers=1, dropout=0.5)
        self.fusion2 = DCNLayer2(embed_dim=200, num_layers=2, dropout=0.7)

    def psd(self, data):
        reshaped_data = data.reshape(data.shape[0], -1, data.shape[-1])
        fft_result = torch.fft.rfft(reshaped_data, dim=-1)
        power_spectrum = torch.abs(fft_result) ** 2
        normalized_psd = (power_spectrum - power_spectrum.mean(dim=-1, keepdim=True)) / (power_spectrum.std(dim=-1, keepdim=True) + 1e-08)
        return normalized_psd

    def forward(self, x):
        b, total_channels, total_time = x.shape
        patch_size = 200
        patch_num = total_time // patch_size
        x = x.reshape(b, total_channels, patch_num, patch_size)
        x_a = x[:, :18, :, :]
        x_b = x[:, 18:19, :, :]
        x_c = x[:, 19:, :, :]
        input_chans = utils.get_input_chans(self.ch_names[:18])
        t_a, t_b, t_c = self.fusion1(x_a, x_b, x_c)
        t_a = t_a.reshape(b, -1, patch_num, patch_size)
        t_b = t_b.reshape(b, -1, patch_num, patch_size)
        t_c = t_c.reshape(b, -1, patch_num, patch_size)
        t_c = self.t_c_patch_block(t_c, None, return_all_tokens=True, num_ch=1)
        t_b = self.t_b_patch_block(t_b, None, return_all_tokens=True, num_ch=1)
        t_a = self.t_a_patch_block(t_a, input_chans, return_all_tokens=True, num_ch=18)
        psd_c = self.psd(x[:, 19:20, :, :])
        psd_b = self.psd(x[:, 18:19, :, :])
        psd_a = self.psd(x[:, :18, :, :])
        t_c = self.t_c_front_model(t_c, None, shape_x=x[:, 19:20, :, :].shape, psd=psd_c, return_all_tokens=True, num_ch=1)
        t_b = self.t_b_front_model(t_b, None, shape_x=x[:, 18:19, :, :].shape, psd=psd_b, return_all_tokens=True, num_ch=1)
        t_a = self.t_a_front_model(t_a, input_chans, shape_x=x[:, :18, :, :].shape, psd=psd_a, return_all_tokens=True, num_ch=18)
        for blk_t_a, blk_t_b, blk_t_c in zip(self.t_a_mid_model.blocks, self.t_b_mid_model.blocks, self.t_c_mid_model.blocks):
            t_a = blk_t_a(t_a, rel_pos_bias=None)
            t_b = blk_t_b(t_b, rel_pos_bias=None)
            t_c = blk_t_c(t_c, rel_pos_bias=None)
        t_a = self.t_a_rear_model(t_a, return_all_tokens=True)
        t_b = self.t_b_rear_model(t_b, return_all_tokens=True)
        t_c = self.t_c_rear_model(t_c, return_all_tokens=True)
        t_a, t_b, t_c = self.fusion2(t_a, t_b, t_c)
        y = torch.cat((t_a, t_b, t_c), dim=1)
        return self.task_head(y)
_FACTORY = {'front_labram_base_patch200_200': front_model_front_labram_base_patch200_200, 'mid_labram_base_patch200_200': mid_model_mid_labram_base_patch200_200, 'rear_labram_base_patch200_200': rear_model_rear_labram_base_patch200_200, 'patch_labram_base_patch200_200': patch_block_patch_labram_base_patch200_200}

def create_model(name, **kwargs):
    kwargs.pop('drop_block_rate', None)
    return _FACTORY[name](**kwargs)

def build_model(args, ch_names, num_t):
    model = Custom_LaBraM(args, ch_names, num_t, not args.from_scratch)
    model.task_head = LinearWithConstraint(42200, args.nb_classes, max_norm=1, flatten=1)
    return model
