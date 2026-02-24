from mmseg.registry import MODELS
import torch
from mmengine.model import BaseModule
from ...utils.safe_ops import safe_sigmoid
import torch.nn as nn
import math
import torch.nn.functional as F
from pointops import farthest_point_sampling, query_and_group, aggregation, interpolation, subtraction


@MODELS.register_module()
class CascadeLanguageGuidance(BaseModule):
    def __init__(self, 
                 embed_dim,
                 downsample_points,
                 text_dim=128,
                 num_heads=4,
                 sa_k=16,
                 sa_channel_multiplier=2,
                 drop_out=0.0,
                 init_cfg = None):
        super().__init__(init_cfg)

        assert isinstance(downsample_points, list)

        self.downsample_points = downsample_points
        self.encoder = nn.ModuleList()
        self.text_peojections = nn.ModuleList()
        self.funsion_blocks = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.output = nn.Linear(embed_dim, embed_dim)

        encoder_dims = [embed_dim]
        in_channel = embed_dim

        for i, npoint in enumerate(downsample_points):
            out_channel = int(in_channel * sa_channel_multiplier)
            current_k = sa_k[i]
            self.encoder.append(
                SetAbstraction(npoint, current_k, in_channel, [out_channel])
            )

            self.text_peojections.append(nn.Linear(text_dim, out_channel))
            self.funsion_blocks.append(
                nn.ModuleDict({
                    'cross_attn': nn.MultiheadAttention(
                        out_channel, num_heads=num_heads, batch_first=True, dropout=0.1
                    ),
                    'norm': nn.LayerNorm(out_channel),
                    'gate': nn.Sequential(
                        nn.Linear(out_channel * 2, out_channel),
                        nn.Sigmoid()
                    )
                }),
            )

            in_channel = out_channel
            encoder_dims.append(in_channel)

        for i in range(len(downsample_points)-1, -1, -1):
            in_channel_coarse = encoder_dims[i+1]
            in_channel_fine = encoder_dims[i]
            out_channel = encoder_dims[i]

            self.decoder.append(
                nn.ModuleDict({
                    'up_sample': FeaturePropagationV2(
                    in_channel_fine,
                    in_channel_coarse,
                    out_channels=[out_channel]
                ),
                # 'gate':AttentionGate(out_channel, in_channel_fine, out_channel),
                'norm': nn.LayerNorm(out_channel)
                })
            )

    def forward(self, instance_feature, anchor, language_features):

        B, N, _ = instance_feature.shape
        _, L, _ = language_features.shape
        f_3d = instance_feature
        f_text = language_features

        xyz = anchor[..., :3]
        norm_xyz = 2.0 * safe_sigmoid(xyz) - 1.0
        norm_xyz = norm_xyz.view(B*N, -1)
        f_3d = f_3d.view(B*N, -1)
        f_text = f_text.view(B*L, -1)

        skip_connections = []
        current_p, current_f = norm_xyz, f_3d
        current_o = torch.tensor([current_p.shape[0]], device=norm_xyz.device, dtype=torch.int)
        skip_connections.append((current_p, current_f, current_o))

        for i in range(len(self.encoder)):
            sa_layer = self.encoder[i]
            text_proj = self.text_peojections[i]
            fusion_block = self.funsion_blocks[i]

            p_down, f_down, o_down = sa_layer(current_p, current_f, current_o)

            f_text_proj = text_proj(f_text)
            f_coarse_fused, _ = fusion_block['cross_attn'](
                query=f_down, key=f_text_proj, value=f_text_proj
            )
            gate_input = torch.cat([f_down, f_coarse_fused], dim=-1)
            g = fusion_block['gate'](gate_input)
            f_gated = (1 - g) * f_down + g * f_coarse_fused
            f_coarse_fused_normed = fusion_block['norm'](f_gated)

            current_p, current_f, current_o = p_down, f_coarse_fused_normed, o_down
            skip_connections.append((current_p, current_f, current_o))

        for i in range(len(self.decoder)):
            fp_layer = self.decoder[i]
            p_fine, f_fine, o_fine = skip_connections[-(i+2)]
            f_up = fp_layer['up_sample'](
                p_fine, current_p, f_fine, current_f, o_fine, current_o
            )
            f_up = fp_layer['norm'](f_fine + f_up)
            current_p, current_f, current_o = p_fine, f_up, o_fine

        current_f = current_f.view(B, N, -1)
        output = self.output(current_f)

        return output

class SetAbstraction(nn.Module):
    def __init__(self, 
                 npoint, 
                 nsample,
                 in_channel,
                 out_channels,
                 group_all=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.npoint = npoint
        self.nsample = nsample
        self.group_all = group_all

        self.mlps = nn.ModuleList()
        temp_channel = in_channel + 3 # D + xyz
        for out_channel in out_channels:
            self.mlps.append(nn.Linear(temp_channel, out_channel))
            temp_channel = out_channel

        self.norms = nn.ModuleList([nn.LayerNorm(out_channel) for out_channel in out_channels])
        self.act = nn.ReLU(inplace=True)

        # self.pool = nn.MaxPool1d(self.nsample)

        last_channels = out_channels[-1]
        self.aggregator = AttentionAggregator(last_channels, last_channels)


    def forward(self, xyz, features, offset):
        '''
            xyz: N, 3
            feature: N, D
            offset: B
        '''
        
        new_offset = torch.tensor([self.npoint], device=xyz.device, dtype=torch.int)
        scanidx = farthest_point_sampling(xyz, offset, new_offset)
        new_xyz = xyz[scanidx.long(), :]

        grouped_xyz_feats = query_and_group(self.nsample, xyz, new_xyz, features, None, offset, 
                                            new_offset, use_xyz=True)
        
        grouped_xyz, grouped_feat = grouped_xyz_feats[..., :3], grouped_xyz_feats[..., 3:]

        relative_xyz = grouped_xyz - new_xyz.unsqueeze(1)

        mlp_input_feats = torch.cat([relative_xyz, grouped_feat], dim=-1)
        current_feats = mlp_input_feats
        for linear, norm in zip(self.mlps, self.norms):
            current_feats = self.act(norm(linear(current_feats)))

        # new_features = self.pool(grouped_features.transpose(1, 2).contiguous()).squeeze(-1)
        new_features = self.aggregator(current_feats, grouped_xyz, new_xyz)

        return new_xyz, new_features, new_offset

class FeaturePropagation(nn.Module):
    def __init__(self, 
                 in_channel_fine,
                 in_channel_coarse,
                 out_channels,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlps = nn.ModuleList()
        temp_channel = in_channel_fine + in_channel_coarse

        for out_channel in out_channels:
            self.mlps.append(nn.Linear(temp_channel, out_channel))
            temp_channel = out_channel

        self.act = nn.ReLU(inplace=True)
        self.norms = nn.ModuleList([nn.LayerNorm(out_channel) for out_channel in out_channels])

    def forward(self, xyz1, xyz2, feature1, feature2, offset1, offset2):
        '''
            xyz1:   xyz_fine
            xyz2:   xyz_coarse
        '''
        interpolated_features = interpolation(xyz2, xyz1, feature2, offset2, offset1)

        if feature1 is not None:
            new_features = torch.cat([feature1, interpolated_features], dim=1)
        else:
            new_features = interpolated_features

        for mlp, norm in zip(self.mlps, self.norms):
            out_1 = mlp(new_features)
            out_2 = norm(out_1)
            new_features = self.act(out_2)
            # new_features = self.act(norm(mlp(new_features)))

        return new_features
    
class FeaturePropagationV2(nn.Module):
    def __init__(self, 
                 in_channel_fine,
                 in_channel_coarse,
                 out_channels,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(out_channels, list):
            out_channels = out_channels[0]

        self.proj = nn.Linear(in_channel_coarse, out_channels)

    def forward(self, xyz1, xyz2, feature1, feature2, offset1, offset2):
        '''
            xyz1:   xyz_fine
            xyz2:   xyz_coarse
        '''

        interpolated_features = interpolation(xyz2, xyz1, feature2, offset2, offset1)

        interpolated_features = self.proj(interpolated_features)

        return interpolated_features

        # if feature1 is not None:
        #     new_features = torch.cat([feature1, interpolated_features], dim=-1)
        # else:
        #     new_features = interpolated_features

        # g = self.gate(new_features)

        # f_fused = (1 - g) * feature1 + g * interpolated_features

        # new_features = self.mlps(f_fused)

        # return new_features

class AttentionAggregator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.attention_mlp = nn.Sequential(
            nn.Linear(in_channels + 3, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 1)
        )

        self.out_proj = nn.Linear(in_channels, out_channels)

    def forward(self, grouped_feats, grouped_xyz, new_xyz):

        relative_xyz = grouped_xyz - new_xyz.unsqueeze(1)

        atten_input = torch.cat([relative_xyz, grouped_feats], dim=-1)

        atten_score = self.attention_mlp(atten_input) # m k 1
        atten_weight = F.softmax(atten_score, dim=1)

        weighted_feats = torch.sum(grouped_feats * atten_weight, dim=1) # m k d -> m d

        output = self.out_proj(weighted_feats)

        return output

class AttentionGate(nn.Module):
    def __init__(self, 
                 up_channels,
                 skip_channels,
                 out_channels,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.proj_up = nn.Linear(up_channels, out_channels) if up_channels != out_channels else nn.Identity()
        self.proj_skip = nn.Linear(skip_channels, out_channels) if skip_channels != out_channels else nn.Identity()

        self.gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, f_skip, f_up):

        f_up_proj = self.proj_up(f_up)
        f_skip_proj = self.proj_skip(f_skip)

        attention_map = self.gate(f_up_proj)

        f_skip_refined = f_skip_proj * attention_map

        return f_skip_refined