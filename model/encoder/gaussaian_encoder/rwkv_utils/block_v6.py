import torch
import torch.nn as nn
from mmengine.model import BaseModule
from .drop import DropPath
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import spconv.pytorch as spconv

from ..token_shift import OmniSpatialShift
from ..utils import cartesian

T_MAX = 12800
HEAD_SIZE = 16

wkv6_cuda = load(name="wkv6",
                 sources=["/model/encoder/gaussian_encoder/rwkv_utils/cuda_v6/wkv6_op.cpp",
                          "/model/encoder/gaussian_encoder/rwkv_utils/cuda_v6/wkv6_cuda.cu"],
                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math",
                                                  "-O3", "-Xptxas=-O3",
                                                  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}",
                                                  f"-D_T_={T_MAX}"])

class Block(BaseModule):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1 / 4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False,
                 with_cp=False, conv_embed_channels=128, kernel_size=[1, 5], dilation=[1, 5],
                 pc_range=None, grid_size=None):
        super().__init__()
        self.layer_id = layer_id
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.att = RWKV6_SpatialMix(n_embd, n_head, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, init_mode, key_norm=key_norm,
                                    conv_embed_channels=conv_embed_channels, kernel_size=kernel_size, dilation=dilation,
                                    pc_range=pc_range, grid_size=grid_size)

        self.ffn = RWKV6_ChannelMix(n_embd, n_head, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, hidden_rate, init_mode, key_norm=key_norm,
                                    conv_embed_channels=conv_embed_channels, kernel_size=kernel_size, dilation=dilation,
                                    pc_range=pc_range, grid_size=grid_size)

        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None, means3d=None, pc_range=None, grid_size=None):

        att_x = x + self.drop_path(self.att(x, patch_resolution, means3d, pc_range, grid_size))
        ffn_x = att_x + self.drop_path(self.ffn(x, patch_resolution, means3d, pc_range, grid_size))

        return ffn_x

class RWKV6_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 channel_gamma=1 / 6, shift_pixel=1, init_mode='fancy', key_norm=False, with_cls_token=False,
                 with_cp=False, patch_resolution=None, conv_embed_channels=128, kernel_size=[1, 5], dilation=[1, 5],
                 pc_range=None, grid_size=None):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd

        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.device = None
        self._init_weights(init_mode)

        self.shift_func = OmniSpatialShift(n_embd, conv_embed_channels, kernel_size, dilation,
                                        pc_range=pc_range,
                                        grid_size=grid_size)

        self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.gate = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(self.attn_sz, n_embd, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, self.attn_sz, eps=1e-5)
        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    ddd[0, 0, i] = i / self.n_embd

                # fancy time_mix
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                TIME_MIX_EXTRA_DIM = 32  # generate TIME_MIX for w,k,v,r,g
                self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-1e-4, 1e-4))
                self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(-1e-4, 1e-4))

                # fancy time_decay
                decay_speed = torch.ones(self.attn_sz)
                for n in range(self.attn_sz):
                    decay_speed[n] = -6 + 5 * (n / (self.attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, self.attn_sz))

                TIME_DECAY_EXTRA_DIM = 64
                self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
                self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, self.attn_sz).uniform_(-1e-4, 1e-4))

                tmp = torch.zeros(self.attn_sz)
                for n in range(self.attn_sz):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (self.attn_sz - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution=None, means3d=None, pc_range=None, grid_size=None):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()

        xx = self.shift_func(x, means3d, pc_range, grid_size) - x

        xxx = x + xx * self.time_maa_x  # [B, T, C]
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        # [5, B*T, TIME_MIX_EXTRA_DIM]
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        # [5, B, T, C]
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        # [B, T, C]
        w = self.time_decay + ww

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x, patch_resolution=None, means3d=None, pc_range=None, grid_size=None):

        B, T, C = x.size()
        self.device = x.device

        r, k, v, g, w = self.jit_func(x, patch_resolution, means3d, pc_range, grid_size)
        x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)

        out = self.jit_func_2(x, g)

        return out

class RWKV6_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 channel_gamma=1 / 6, shift_pixel=1, hidden_rate=4, init_mode='fancy', key_norm=False,
                 with_cp=False, patch_resolution=None, conv_embed_channels=128, kernel_size=[1, 5], dilation=[1, 5],
                 pc_range=None, grid_size=None):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.with_cp = with_cp

        self.shift_func = OmniSpatialShift(n_embd, conv_embed_channels, kernel_size, dilation,
                                        pc_range=pc_range, grid_size=grid_size)

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.attn_sz, self.n_embd, bias=False)

    def forward(self, x, patch_resolution=None, means3d=None, pc_range=None, grid_size=None):

        xx = self.shift_func(x, means3d, pc_range, grid_size) - x

        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2 # torch.square
        kv = self.value(k)

        out = torch.sigmoid(self.receptance(xr)) * kv

        return out

class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.float32,
                            memory_format=torch.contiguous_format)  #.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)  #.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)  #.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)  #.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)  #.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)  #.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


def anchor_shift(input, shift_anchor=1, gamma=1 / 6, spatial_shape=None, means3d=None, pc_range=None, grid_size=None):
    assert gamma <= 1 / 6
    B, N, C = input.shape
    xyz = cartesian(means3d[..., :3], pc_range).flatten(0, 1)
    indices = xyz - pc_range[None, :3]
    indices = indices / grid_size[None, :]
    indices = indices.to(torch.int32)
    batched_indices = torch.cat([
        torch.arange(B, device=indices.device, dtype=torch.int32).reshape(
            B, 1, 1).expand(-1, N, -1).flatten(0, 1), indices], dim=-1)
    spatial_shape = (pc_range[3:] - pc_range[:3]) / grid_size
    spatial_shape = spatial_shape.to(torch.int32)

    input = spconv.SparseConvTensor(
        input.flatten(0, 1),
        indices=batched_indices,
        spatial_shape=spatial_shape,
        batch_size=B
    ).dense()

    B, C, H, W, D = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_anchor:W, :] = input[:, 0:int(C * gamma), :, 0:W - shift_anchor, :]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_anchor, :] = input[:, int(C * gamma):int(C * gamma * 2),
                                                                             :, shift_anchor:W, :]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_anchor:H, :, :] = input[:,
                                                                             int(C * gamma * 2):int(C * gamma * 3),
                                                                             0:H - shift_anchor, :, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_anchor, :, :] = input[:,
                                                                                 int(C * gamma * 3):int(C * gamma * 4),
                                                                                 shift_anchor:H, :, :]
    output[:, int(C * gamma * 2):int(C * gamma * 5), :, :, shift_anchor:D] = input[:,
                                                                             int(C * gamma * 2):int(C * gamma * 5), :,
                                                                             :, shift_anchor:D]
    output[:, int(C * gamma * 3):int(C * gamma * 6), :, :, 0:D - shift_anchor] = input[:,
                                                                                 int(C * gamma * 3):int(C * gamma * 6),
                                                                                 :, :, 0:D - shift_anchor]

    output = output[..., indices[:, 0], indices[:, 1], indices[:, 2]].transpose(1, 2)
    return output


class GroupedShift(nn.Module):
    def __init__(self, 
                 embed_dim,
                 group_size=100,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert embed_dim % 4 == 0

        self.embed_dim = embed_dim
        self.group_size = group_size
        self.split_size = embed_dim // 4

    def forward(self, x):
        b, n, c = x.shape

        if n % self.group_size !=0:
            raise ValueError(f"Sequence length ({n}) must be divisible by group_size ({self.group_size})")

        x_grouped = x.view(b, -1, self.group_size, c)

        x1, x2, x3, x4 = torch.split(x_grouped, self.split_size, dim=3)
        
        prev_x1 = F.pad(x1, (0, 0, 1, 0), mode='replicate')[..., :self.group_size, :]
        prev_x3 = F.pad(x3, (0, 0, 1, 0), mode='replicate')[..., :self.group_size, :]

        next_x2 = F.pad(x2, (0, 0, 0, 1), mode='replicate')[..., 1:, :]
        next_x4 = F.pad(x4, (0, 0, 0, 1), mode='replicate')[..., 1:, :]

        x_prime_grouped = torch.cat([prev_x1, next_x2, prev_x3, next_x4], dim=3)

        return x_prime_grouped.view(b, n, c)
    
class AnisotropicModulator(nn.Module):
    def __init__(self, 
                 embed_dim,
                 scale_dim=3,
                 rotation_dim=4,
                 mlp_ratio=0.5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        input_dim = scale_dim + rotation_dim
        hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        scales, rotations = x[..., 3:6], x[..., 6:10]
        geo_att = torch.cat([scales, rotations], dim=-1)
        modu_vec = self.mlp(geo_att)
        gate_vec = torch.sigmoid(modu_vec)


        return gate_vec
