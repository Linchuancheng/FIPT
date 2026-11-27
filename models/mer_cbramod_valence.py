# Derived from AdaBrain-Bench, LaBraM and CBraMod; see THIRD_PARTY_NOTICES.md.
from util.model_storage import load_pretrained
'Complete FIPT (CBraMod) model for MER valence.\nAll model components are defined in this file; parameter names retain source compatibility.\n'
from pyexpat import model
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class front_model_CBraMod_Patch(nn.Module):

    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8):
        super().__init__()
        self.patch_embedding = front_model_PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        self.apply(front_model__weights_init)

    def forward(self, x, mask=None):
        patch_emb = self.patch_embedding(x)
        return patch_emb

class front_model_PatchEmbedding(nn.Module):

    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)), nn.GroupNorm(5, 25), nn.GELU(), nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)), nn.GroupNorm(5, 25), nn.GELU(), nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)), nn.GroupNorm(5, 25), nn.GELU())

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding
        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)
        return patch_emb

def front_model__weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class front_model_fft_CBraMod_Patch_fft(nn.Module):

    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8):
        super().__init__()
        self.patch_embedding = front_model_fft_PatchEmbedding_fft(in_dim, out_dim, d_model, seq_len)
        self.apply(front_model_fft__weights_init)

    def forward(self, x, patch_emb, mask=None):
        patch_emb = self.patch_embedding(x, patch_emb)
        return patch_emb

class front_model_fft_PatchEmbedding_fft(nn.Module):

    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3), groups=d_model))
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)), nn.GroupNorm(5, 25), nn.GELU(), nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)), nn.GroupNorm(5, 25), nn.GELU(), nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)), nn.GroupNorm(5, 25), nn.GELU())
        self.spectral_proj = nn.Sequential(nn.Linear(101, d_model), nn.Dropout(0.1))

    def forward(self, x, patch_emb, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding
        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb
        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)
        patch_emb = patch_emb + positional_embedding
        return patch_emb

def front_model_fft__weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
import copy
from typing import Optional, Any, Union, Callable
import warnings
from torch import Tensor
from torch.nn import functional as F
import pdb

class criss_cross_transformer_TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f'torch.nn.modules.{self.__class__.__name__}')
        self.layers = criss_cross_transformer__get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, is_causal: Optional[bool]=None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class criss_cross_transformer_TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1, activation: Union[str, Callable[[Tensor], Tensor]]=F.relu, layer_norm_eps: float=1e-05, batch_first: bool=False, norm_first: bool=False, bias: bool=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs)
        self.self_attn_t = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(activation, str):
            activation = criss_cross_transformer__get_activation_fn(activation)
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, is_causal: bool=False) -> Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool=False) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, :patch_size // 2]
        xt = x[:, :, :, patch_size // 2:]
        xs = xs.transpose(1, 2).contiguous().view(bz * patch_num, ch_num, patch_size // 2)
        xt = xt.contiguous().view(bz * ch_num, patch_num, patch_size // 2)
        xs = self.self_attn_s(xs, xs, xs, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size // 2).transpose(1, 2)
        xt = self.self_attn_t(xt, xt, xt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size // 2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def criss_cross_transformer__get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}')

def criss_cross_transformer__get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class mid_model_CBraMod_Transformer(nn.Module):

    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8):
        super().__init__()
        self.patch_embedding = mid_model_PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        encoder_layer = criss_cross_transformer_TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True, activation=F.gelu)
        self.encoder = criss_cross_transformer_TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        self.apply(mid_model__weights_init)
        self.psd_proj = nn.Linear(101, out_dim)
        self.ctx = nn.Parameter(torch.zeros(2, 4, out_dim))

    def forward(self, x, psd, mask=None):
        psd = self.psd_proj(psd)
        ctx = self.ctx.expand(x.shape[0], -1, -1, -1)
        x = torch.cat((x, psd, ctx), dim=1)
        feats = self.encoder(x)
        out = self.proj_out(feats)
        return out

class mid_model_PatchEmbedding(nn.Module):

    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3), groups=d_model))
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)), nn.GroupNorm(5, 25), nn.GELU(), nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)), nn.GroupNorm(5, 25), nn.GELU(), nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)), nn.GroupNorm(5, 25), nn.GELU())
        self.spectral_proj = nn.Sequential(nn.Linear(101, d_model), nn.Dropout(0.1))

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding
        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)
        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb
        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)
        patch_emb = patch_emb + positional_embedding
        return patch_emb

def mid_model__weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

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
    if hasattr(model, 'patch_embedding'):
        patch_params = sum((p.numel() for p in model.patch_embedding.parameters()))
        print(f'patch_embedding参数量: {patch_params / 1000000.0:.2f} M')
    if hasattr(model, 'encoder'):
        patch_params = sum((p.numel() for p in model.encoder.parameters()))
        print(f'encoder参数量: {patch_params / 1000000.0:.2f} M')
    if hasattr(model, 'psd_proj'):
        for param in model.psd_proj.parameters():
            param.requires_grad = True
        unfrozen_params.append('psd_proj.*')
    if hasattr(model, 'proj_out'):
        for param in model.proj_out.parameters():
            param.requires_grad = True
        unfrozen_params.append('proj_out.*')
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

class NormalSubLayer(nn.Module):
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
        super(NormalSubLayer, self).__init__()
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
        self.dcn_layers = nn.ModuleList([NormalSubLayer(embed_dim, dropout) for _ in range(num_layers)])

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
        self.dcn_layers = nn.ModuleList([NormalSubLayer(embed_dim, dropout) for _ in range(num_layers)])

    def forward(self, data1, data2, data3):
        data1 = data1.reshape(data1.shape[0], -1, data1.shape[-1])
        data2 = data2.reshape(data2.shape[0], -1, data2.shape[-1])
        data3 = data3.reshape(data3.shape[0], -1, data3.shape[-1])
        for sub_layer in self.dcn_layers:
            data1, data2, data3 = sub_layer(data1, data2, data3)
        return (data1, data2, data3)

class Custom_CBraMod(nn.Module):

    def __init__(self, args, from_pretrain=False):
        super().__init__()
        model_patch_t_a = front_model_CBraMod_Patch()
        model_patch_t_a = setup_finetuning_parameters(model_patch_t_a)
        model_patch_fft_t_a = front_model_fft_CBraMod_Patch_fft()
        model_patch_fft_t_a = setup_finetuning_parameters(model_patch_fft_t_a)
        model_trans_t_a = mid_model_CBraMod_Transformer()
        model_trans_t_a = setup_finetuning_parameters(model_trans_t_a)
        if not getattr(args, 'from_scratch', False):
            model_patch_t_a.load_state_dict(load_pretrained(args), strict=False)
        if not getattr(args, 'from_scratch', False):
            model_trans_t_a.load_state_dict(load_pretrained(args), strict=False)
        if not getattr(args, 'from_scratch', False):
            model_patch_fft_t_a.load_state_dict(load_pretrained(args), strict=False)
        model_trans_t_a.proj_out = nn.Identity()
        model_patch_t_b = front_model_CBraMod_Patch()
        model_patch_t_b = setup_finetuning_parameters(model_patch_t_b)
        model_patch_fft_t_b = front_model_fft_CBraMod_Patch_fft()
        model_patch_fft_t_b = setup_finetuning_parameters(model_patch_fft_t_b)
        model_trans_t_b = mid_model_CBraMod_Transformer()
        model_trans_t_b = setup_finetuning_parameters(model_trans_t_b)
        if not getattr(args, 'from_scratch', False):
            model_patch_t_b.load_state_dict(load_pretrained(args), strict=False)
        if not getattr(args, 'from_scratch', False):
            model_trans_t_b.load_state_dict(load_pretrained(args), strict=False)
        if not getattr(args, 'from_scratch', False):
            model_patch_fft_t_b.load_state_dict(load_pretrained(args), strict=False)
        model_trans_t_b.proj_out = nn.Identity()
        model_patch_t_c = front_model_CBraMod_Patch()
        model_patch_t_c = setup_finetuning_parameters(model_patch_t_c)
        model_patch_fft_t_c = front_model_fft_CBraMod_Patch_fft()
        model_patch_fft_t_c = setup_finetuning_parameters(model_patch_fft_t_c)
        model_trans_t_c = mid_model_CBraMod_Transformer()
        model_trans_t_c = setup_finetuning_parameters(model_trans_t_c)
        if not getattr(args, 'from_scratch', False):
            model_patch_t_c.load_state_dict(load_pretrained(args), strict=False)
        if not getattr(args, 'from_scratch', False):
            model_trans_t_c.load_state_dict(load_pretrained(args), strict=False)
        if not getattr(args, 'from_scratch', False):
            model_patch_fft_t_c.load_state_dict(load_pretrained(args), strict=False)
        model_trans_t_c.proj_out = nn.Identity()
        self.task_head = None
        self.model_patch_t_a = model_patch_t_a
        self.model_trans_t_a = model_trans_t_a
        self.model_patch_fft_t_a = model_patch_fft_t_a
        self.model_patch_t_b = model_patch_t_b
        self.model_trans_t_b = model_trans_t_b
        self.model_patch_fft_t_b = model_patch_fft_t_b
        self.model_patch_t_c = model_patch_t_c
        self.model_trans_t_c = model_trans_t_c
        self.model_patch_fft_t_c = model_patch_fft_t_c
        self.fusion1 = DCNLayer(embed_dim=101, num_layers=1, dropout=0.5)
        self.fusion2 = DCNLayer2(embed_dim=200, num_layers=2, dropout=0.7)

    def psd(self, data):
        fft_result = torch.fft.rfft(data, dim=-1)
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
        t_a, t_b, t_c = self.fusion1(x_a, x_b, x_c)
        b, n, t = t_a.shape
        t_a = t_a.reshape(b, -1, patch_num, patch_size)
        t_b = t_b.reshape(b, -1, patch_num, patch_size)
        t_c = t_c.reshape(b, -1, patch_num, patch_size)
        t_a = self.model_patch_t_a(t_a)
        t_b = self.model_patch_t_b(t_b)
        t_c = self.model_patch_t_c(t_c)
        t_a = t_a.reshape(x[:, :18, :, :].shape)
        t_b = t_b.reshape(x[:, 18:19, :, :].shape)
        t_c = t_c.reshape(x[:, 19:, :, :].shape)
        t_a = self.model_patch_fft_t_a(x_a, t_a)
        t_b = self.model_patch_fft_t_b(x_b, t_b)
        t_c = self.model_patch_fft_t_c(x_c, t_c)
        psd_a = self.psd(x[:, :18, :, :])
        psd_b = self.psd(x[:, 18:19, :, :])
        psd_c = self.psd(x[:, 19:, :, :])
        t_a = self.model_trans_t_a(t_a, psd_a)
        t_b = self.model_trans_t_b(t_b, psd_b)
        t_c = self.model_trans_t_c(t_c, psd_c)
        t_a, t_b, t_c = self.fusion2(t_a, t_b, t_c)
        y = torch.cat((t_a, t_b, t_c), dim=1)
        return self.task_head(y)

def build_model(args, ch_names, num_t):
    model = Custom_CBraMod(args, not args.from_scratch)
    model.task_head = LinearWithConstraint(36800, args.nb_classes, max_norm=1, flatten=1)
    return model
