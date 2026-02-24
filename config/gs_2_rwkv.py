data_root = "/mnt/ikcdir/nu/dataset/ns_data/"
anno_root = "/mnt/ikcdir/nu/dataset/ns_cam/"
occ_path = "/mnt/ikcdir/nu/dataset/ns_occ/nuscenes_occ/samples/"
batch_size = 1
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# =========== data config ==============
input_shape = (1600, 864)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

train_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train'
)

val_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val'
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=2,
    shuffle=True
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=2
)

# =========== misc config ==============
max_epochs = 30
print_freq = 50
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=4e-4, weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)}
    )
)
grad_max_norm = 35
# ========= model config ===============
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=17,
            num_classes=18,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],
            ignore_empty=False,
            lovasz_use_softmax=False),
        dict(
            type="PixelDistributionLoss",
            weight=1.0,
            use_sigmoid=False),
        # dict(
        #     type="CosineSimilarityLoss",
        #     weight=1.0,)
        ])

loss_input_convertion = dict(
    pred_occ="pred_occ",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_mask",
    bin_logits="bin_logits",
    density="density",
    pixel_logits="pixel_logits",
    pixel_gt="pixel_gt",
    # T_orig_features='T_orig_features',
    # T_proj_features='T_proj_features'
)

# ========= model config ===============
embed_dims = 128
num_decoder = 4
num_levels = 4
drop_out = 0.1
num_groups = 4
num_single_frame_decoder = 1
use_deformable_func = True
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
scale_range = [0.01, 2.5]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
load_from = '/home/dgg/haiyang/code/r101_dcn_fcos3d_pretrain.pth'
semantics = True
semantic_dim = 17

label = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation', 'empty']

model = dict(
    type='BEVSegmentor',
    use_text=True,
    label=label,
    freeze_lifter=True,
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=1,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    lifter=dict(
        type='GaussianLifterV2',
        num_anchor=6400,
        embed_dims=embed_dims,
        anchor_grad=False,
        feat_grad=False,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        num_samples=128,
        anchors_per_pixel=1,
        random_sampling=False,
        projection_in=None,
        initializer=dict(
            type="ResNetSecondFPN",
            img_backbone_out_indices=[0, 1, 2, 3],
            img_backbone_config=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                # original DCNv2 will print log when perform load_state_dict
                stage_with_dcn=(False, False, True, True)),
            neck_confifg=dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=[embed_dims] * 4,
                upsample_strides=[0.5, 1, 2, 4])),
        initializer_img_downsample=None,
        deterministic=False,
        random_samples=6400),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            ffn_drop=0.1,
            add_identity=False,
            pre_norm=dict(type="LN"),
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="none",
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                pc_range=pc_range,
                scale_range=scale_range,
                learnable_fixed_scale=6.0,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModuleV2',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            unit_xyz=[4.0, 4.0, 1.0],
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='identity',
        ),
        # spconv_layer=dict(
        #     type="SparseConv3D",
        #     in_channels=embed_dims,
        #     embed_channels=embed_dims,
        #     pc_range=pc_range,
        #     grid_size=[1.0, 1.0, 1.0],
        #     phi_activation=phi_activation,
        #     xyz_coordinate=xyz_coordinate,
        #     use_out_proj=True,
        #     use_multi_layer=True,
        # ),
        self_rwkv_layer=dict(
            type="Self_RWVK",
            pc_range=pc_range,
            grid_size=[0.5, 0.5, 0.5],
            n_embd=128,
            n_head=8,
            n_layer=8,
            conv_embed_channels=128,
            num_layers=1,
            kernel_size=[3, 5],
            dilation=[3, 5]
        ),
        hierarchical_cross_attn=dict(
            type='MultiLevelCrossAttention',
            embed_dim=128,
            downsample_points=[6400, 3200, 1600],
            # downsample_points=[9600, 7200, 5400],
            sa_k=[16, 8, 4],
            # sa_k=[16, 16, 16],
            # sa_k=[4, 8, 16],
            text_dim=512,
        ),
        num_decoder=num_decoder,
        operation_order=[
                            "identity",
                            "deformable",
                            "add",
                            "norm",

                            "identity",
                            "ffn",
                            "add",
                            "norm",

                            "identity",
                            "self_rwkv",
                            "add",
                            "norm",

                            # "identity",
                            # "ffn",
                            # "add",
                            # "norm",

                            # "identity",
                            # "hierarchical_cross_attn",
                            # "add",
                            # "norm",

                            "identity",
                            "ffn",
                            "add",
                            "norm",

                            "refine",
                        ] * num_decoder,
    ),
    # encoder=dict(
    #     type='GaussianOccEncoder',
    #     anchor_encoder=dict(
    #         type='SparseGaussian3DEncoder',
    #         embed_dims=embed_dims,
    #         include_opa=include_opa,
    #         semantics=semantics,
    #         semantic_dim=semantic_dim
    #     ),
    #     norm_layer=dict(type="LN", normalized_shape=embed_dims),
    #     ffn=dict(
    #         type="AsymmetricFFN",
    #         in_channels=embed_dims,
    #         embed_dims=embed_dims,
    #         feedforward_channels=embed_dims * 4,
    #         ffn_drop=0.1,
    #         add_identity=False,
    #         pre_norm=dict(type="LN"),
    #         num_fcs=2,
    #         act_cfg=dict(type="ReLU", inplace=True),
    #     ),
    #     deformable_model=dict(
    #         type='DeformableFeatureAggregation',
    #         embed_dims=embed_dims,
    #         num_groups=num_groups,
    #         num_levels=num_levels,
    #         num_cams=6,
    #         attn_drop=0.15,
    #         use_deformable_func=use_deformable_func,
    #         use_camera_embed=True,
    #         residual_mode="none",
    #         kps_generator=dict(
    #             type="SparseGaussian3DKeyPointsGenerator",
    #             embed_dims=embed_dims,
    #             phi_activation=phi_activation,
    #             xyz_coordinate=xyz_coordinate,
    #             num_learnable_pts=6,
    #             pc_range=pc_range,
    #             scale_range=scale_range,
    #             learnable_fixed_scale=6.0,
    #             fix_scale=[
    #                 [0, 0, 0],
    #                 [0.45, 0, 0],
    #                 [-0.45, 0, 0],
    #                 [0, 0.45, 0],
    #                 [0, -0.45, 0],
    #                 [0, 0, 0.45],
    #                 [0, 0, -0.45],
    #             ],
    #         ),
    #     ),
    #     refine_layer=dict(
    #         type='SparseGaussian3DRefinementModuleV2',
    #         embed_dims=embed_dims,
    #         pc_range=pc_range,
    #         scale_range=scale_range,
    #         unit_xyz=[4.0, 4.0, 1.0],
    #         semantics=semantics,
    #         semantic_dim=semantic_dim,
    #         include_opa=include_opa,
    #         xyz_coordinate=xyz_coordinate,
    #         semantics_activation='identity',
    #     ),
    #     lgp_layer=dict(
    #         type='Local_Geometry_Perception_Block',
    #         input_dim=128,
    #         conv_embed_channels=128, 
    #         kernel_size=3, 
    #         dilation=[1, 2, 4],
    #         pc_range=pc_range, 
    #         grid_size=[0.5, 0.5, 0.5],
    #     ),
    #     sas_rwkv_layer=dict(
    #         type='Semantic_aware_Spatial_RWKV_Block',
    #         embed_dim=128,
    #         downsample_points=[6400, 3200, 1600],
    #         sa_k=[16, 8, 4],
    #         text_dim=512,
    #         n_head=[16, 32, 64],
    #         self_rwkv_cfg=dict(
    #             type="Self_RWVK_v2",
    #             pc_range=pc_range,
    #             grid_size=[0.5, 0.5, 0.5],
    #             n_embd=128,
    #             n_head=8,
    #             n_layer=8,
    #             num_layers=1,
    #         ),
    #     ),
    #     num_decoder=num_decoder,
    #     operation_order=[
    #                         "identity",
    #                         "deformable",
    #                         "add",
    #                         "norm",

    #                         "identity",
    #                         "ffn",
    #                         "add",
    #                         "norm",

    #                         "identity",
    #                         "lgp",
    #                         "add",
    #                         "norm",

    #                         "identity",
    #                         "sas_rwkv",
    #                         "add",
    #                         "norm",

    #                         "identity",
    #                         "ffn",
    #                         "add",
    #                         "norm",

    #                         "refine",
    #                     ] * num_decoder,
    # ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0],
        ),
        with_empty=False,
        use_localaggprob=True,
        use_localaggprob_fast=False,
        combine_geosem=True,
        cuda_kwargs=dict(
            scale_multiplier=5,
            H=200, W=200, D=16,
            pc_min=[-50.0, -50.0, -5.0],
            grid_size=0.5),
    )
)