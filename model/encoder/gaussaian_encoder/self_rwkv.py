# from functools import partial
#
# from mmseg.registry import MODELS
# import torch
# from mmengine.model import BaseModule
# # from .rwkv_utils.block import Block
# from .rwkv_utils.block_v6 import Block
# from .utils import cartesian, reverse_cartesian
# from ...utils.safe_ops import safe_sigmoid, safe_inverse_sigmoid
# from .sparse_reorder import SparseReorder
# import torch.nn as nn
#
# @MODELS.register_module()
# class Self_RWVK(BaseModule):
#     def __init__(self,
#                  pc_range,
#                  grid_size,
#                  n_embd,
#                  n_head,
#                  n_layer,
#                  num_layers=2,
#                  shift_model='q_shift',
#                  channel_gamma=1 / 4,
#                  shift_pixel=1,
#                  drop_path=0.,
#                  hidden_rate=4,
#                  conv_embed_channels=128, kernel_size=[1, 5], dilation=[1, 5],
#                  **kwargs):
#         super().__init__()
#         self.n_embd = n_embd
#         self.rwkv_block = nn.ModuleList([Block(
#             n_embd=n_embd,
#             n_head=n_head,
#             n_layer=n_layer,
#             layer_id=0,
#             shift_mode=shift_model,
#             channel_gamma=channel_gamma,
#             shift_pixel=shift_pixel,
#             drop_path=drop_path,
#             hidden_rate=hidden_rate,
#             init_mode='fancy',
#             post_norm=False,
#             key_norm=False,
#             init_values=None,
#             with_cp=False,
#             conv_embed_channels=conv_embed_channels, kernel_size=kernel_size, dilation=dilation,
#             pc_range=pc_range, grid_size=grid_size,
#         ) for _ in range(num_layers)])
#
#         self.get_xyz = partial(cartesian, pc_range=pc_range)
#         self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
#         self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))
#
#         # self.output_proj = nn.Linear(n_embd * 2, n_embd)
#         # self.pos_encoder = nn.Sequential(
#         #     nn.Linear(3, n_embd),
#         #     nn.GELU(),
#         #     nn.Linear(n_embd, n_embd)
#         # )
#
#     def forward(self, instance_feature, means3d, metas=None, **kwargs):
#         device = instance_feature.device
#         bs, num_anchor, _ = means3d.shape
#
#         anchor_xyz = cartesian(means3d, pc_range=self.pc_range)
#         # norm_anchor_xyz = reverse_cartesian(anchor_xyz, pc_range=self.pc_range)
#
#         indices = anchor_xyz - self.pc_range[None, :3]
#         indices = indices / self.grid_size[None, :]
#         coords = indices.to(torch.int32).to(device)
#         coords = coords.view(bs*num_anchor, -1)
#         # norm_coords = 2.0 * safe_sigmoid(means3d[..., :3]) - 1.0
#         # positional_embedding = self.pos_encoder(norm_coords)
#
#
#         spatial_shape = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
#         spatial_shape = spatial_shape.to(torch.int32)
#
#         x = instance_feature
#         for layer in self.rwkv_block:
#             x = layer(x, spatial_shape, means3d, self.pc_range, self.grid_size)
#
#         return x
#
#
# @MODELS.register_module()
# class Self_RWVK_v2(BaseModule):
#     def __init__(self,
#                  pc_range,
#                  grid_size,
#                  n_embd,
#                  n_head,
#                  n_layer,
#                  num_layers=2,
#                  shift_model='q_shift',
#                  channel_gamma=1 / 4,
#                  shift_pixel=1,
#                  drop_path=0.,
#                  hidden_rate=4,
#                  **kwargs):
#         super().__init__()
#         self.n_embd = n_embd
#         self.rwkv_block = nn.ModuleList([Block(
#             n_embd=n_embd,
#             n_head=n_head,
#             n_layer=n_layer,
#             layer_id=0,
#             shift_mode=shift_model,
#             channel_gamma=channel_gamma,
#             shift_pixel=shift_pixel,
#             drop_path=drop_path,
#             hidden_rate=hidden_rate,
#             init_mode='fancy',
#             post_norm=False,
#             key_norm=False,
#             init_values=None,
#             with_cp=False,
#             pc_range=pc_range, grid_size=grid_size,
#         ) for _ in range(num_layers)])
#
#         self.get_xyz = partial(cartesian, pc_range=pc_range)
#         self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
#         self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))
#
#         self.pos_encoder = nn.Sequential(
#             nn.Linear(3, n_embd),
#             nn.GELU(),
#             nn.Linear(n_embd, n_embd)
#         )
#
#     def forward(self, instance_feature, means3d, scale=None, rot=None, metas=None, **kwargs):
#         device = instance_feature.device
#         bs = 1
#         num_anchor, _ = means3d.shape
#
#         anchor_xyz = cartesian(means3d, pc_range=self.pc_range)
#
#         positional_embedding = self.pos_encoder(anchor_xyz)
#         instance_feature = instance_feature + positional_embedding
#
#         indices = anchor_xyz - self.pc_range[None, :3]
#         indices = indices / self.grid_size[None, :]
#         coords = indices.to(torch.int32).to(device)
#         coords = coords.view(bs*num_anchor, -1)
#
#
#         spatial_shape = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
#         spatial_shape = spatial_shape.to(torch.int32)
#
#         reorder_manager = SparseReorder(coords, (200, 200, 16))
#
#         reordered_features = reorder_manager.reorder(instance_feature.view(bs*num_anchor, -1))
#         reordered_coords = reorder_manager.reorder(coords)
#
#         x = reordered_features.reshape(bs, num_anchor, -1)
#         for layer in self.rwkv_block:
#             x = layer(x, spatial_shape, means3d, self.pc_range, self.grid_size)
#
#         restored_features = reorder_manager.restore(x.view(bs*num_anchor, -1))
#
#         return restored_features.reshape(bs*num_anchor, -1)