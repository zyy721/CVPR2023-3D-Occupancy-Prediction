_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.2, 0.2, 8]



img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
# queue_length = 4 # each sequence contains `queue_length` frames.
queue_length = 1 # each sequence contains `queue_length` frames.


# occ_vae
occ_vae_dim_ = 16
expansion = 8
base_channel = 64
# base_channel = 96

n_e_ = 512


model = dict(
    # type='BEVFormerOcc',
    type='CustomBEVFormerOcc',

    # use_grid_mask=True,
    use_grid_mask=False,

    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        # type='BEVFormerOccHead',
        type='CustomBEVFormerOccHead',

        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=18,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        # loss_occ=dict(
        #     type='FocalLoss',
        #     use_sigmoid=False,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=10.0),
        use_mask=False,
        loss_occ= dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),


        multi_loss = dict(
            type='MultiLoss',
            loss_cfgs=[
                dict(
                    type='ReconLoss',
                    weight=10.0,
                    ignore_label=-100,
                    use_weight=False,
                    cls_weight=None,
                    input_dict={
                        'logits': 'logits',
                        'labels': 'inputs'}),
                dict(
                    type='LovaszLoss',
                    weight=1.0,
                    input_dict={
                        'logits': 'logits',
                        'labels': 'inputs'}),
                # dict(
                #     type='VQVAEEmbedLoss',
                #     weight=1.0),

                dict(
                    type='KldLoss',
                    # weight=0.002,
                    weight=0.00005,
                    input_dict={
                        'z_mu': 'z_mu',
                        'logvar': 'logvar'}),
                ]),

        loss_input_convertion = dict(
            logits='logits',
            # embed_loss='embed_loss'
            z_mu='z_mu',
            logvar='logvar'
        ),


        transformer=dict(
            type='CustomTransformerOcc',
            pillar_h=16,
            num_classes=18,
            norm_cfg=dict(type='BN', ),
            norm_cfg_3d=dict(type='BN3d', ),
            use_3d=True,
            use_conv=False,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=4,
                pc_range=point_cloud_range,
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,

        ),


        occ_vae = dict(
            # type = 'VAERes2D',
            type = 'CustomVAERes2D',
            encoder_cfg=dict(
                # type='Encoder2D',
                type='CustomEncoder2D',

                ch = base_channel, 
                out_ch = base_channel, 
                # ch_mult = (1,2,4), 
                ch_mult = (1,2,4,8), 

                # num_res_blocks = 2,
                num_res_blocks = 4,
                # attn_resolutions = (50,), 
                attn_resolutions = (25,), 
                dropout = 0.0, 
                resamp_with_conv = True, 
                in_channels = occ_vae_dim_ * expansion,
                resolution = 200, 
                # z_channels = base_channel * 2, 
                # double_z = False,
                z_channels = 32, 
                double_z = True,

            ), 
            # decoder_cfg=dict(
            #     # type='Decoder2D',
            #     type='CustomDecoder2D',

            #     ch = base_channel, 
            #     out_ch = occ_vae_dim_ * expansion, 
            #     # ch_mult = (1,2,4), 
            #     ch_mult = (1,2,4,8), 
            #     num_res_blocks = 2,
            #     # attn_resolutions = (50,), 
            #     attn_resolutions = (25,), 
            #     dropout = 0.0, 
            #     resamp_with_conv = True, 
            #     in_channels = occ_vae_dim_ * expansion,
            #     resolution = 200, 
            #     # z_channels = base_channel * 2, 
            #     # z_channels = base_channel, 
            #     z_channels = 32, 
            #     give_pre_end = False
            # ),
            decoder_cfg=dict(
                # type='Decoder2D',
                type='CustomTemporalDecoder',

                ch = base_channel, 
                out_ch = occ_vae_dim_ * expansion, 
                # ch_mult = (1,2,4), 
                ch_mult = (1,2,4,8), 
                num_res_blocks = 2,
                # attn_resolutions = (50,), 
                attn_resolutions = (25,), 
                dropout = 0.0, 
                resamp_with_conv = True, 
                in_channels = occ_vae_dim_ * expansion,
                resolution = 200, 
                # z_channels = base_channel * 2, 
                # z_channels = base_channel, 
                z_channels = 32, 
                give_pre_end = False
            ),
            num_classes=18,
            expansion=expansion, 
            # vqvae_cfg=dict(
            #     type='VectorQuantizer',
            #     n_e = n_e_, 
            #     e_dim = base_channel * 2, 
            #     beta = 1., 
            #     z_channels = base_channel * 2, 
            #     use_voxel=False)
        ),




    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range)))))

dataset_type = 'NuSceneOcc'
data_root = 'data/occ3d-nus/'
file_client_args = dict(backend='disk')
occ_gt_data_root='data/occ3d-nus'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile',data_root=occ_gt_data_root),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=[ 'img','voxel_semantics','mask_lidar','mask_camera'] )
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile',data_root=occ_gt_data_root),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            # dict(type='CustomCollect3D', keys=['img'])
            dict(type='CustomCollect3D', keys=['img','voxel_semantics'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'occ_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,

              ann_file=data_root + 'occ_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
