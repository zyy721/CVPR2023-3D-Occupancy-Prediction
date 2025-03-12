'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-07-10 11:08:49
Email: haimingzhang@link.cuhk.edu.cn
Description: The Pretraining head.
'''
import os
import os.path as osp
import numpy as np
from einops import rearrange
import torch
from torch import nn
from random import randint
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from mmdet3d.models import builder
from mmcv.runner import get_dist_info

from .nerf_utils import (visualize_image_semantic_depth_pair, 
                         visualize_image_pairs)
from .. import utils
from .depth_ssl import *
from .loss_utils import (l1_loss, loss_depth_smoothness, patch_norm_mse_loss, 
                         patch_norm_mse_loss_global, ssim)


OCC3D_PALETTE = torch.Tensor([
    [0, 0, 0],
    [255, 120, 50],  # barrier              orangey
    [255, 192, 203],  # bicycle              pink
    [255, 255, 0],  # bus                  yellow
    [0, 150, 245],  # car                  blue
    [0, 255, 255],  # construction_vehicle cyan
    [200, 180, 0],  # motorcycle           dark orange
    [255, 0, 0],  # pedestrian           red
    [255, 240, 150],  # traffic_cone         light yellow
    [135, 60, 0],  # trailer              brown
    [160, 32, 240],  # truck                purple
    [255, 0, 255],  # driveable_surface    dark pink
    [139, 137, 137], # other_flat           dark grey
    [75, 0, 75],  # sidewalk             dard purple
    [150, 240, 80],  # terrain              light green
    [230, 230, 250],  # manmade              white
    [0, 175, 0],  # vegetation           green
    [0, 255, 127],  # ego car              dark cyan
    [255, 99, 71],
    [0, 191, 255],
    [125, 125, 125]
])


def warp_bev_features(voxel_feats, 
                      voxel_flow,
                      voxel_size, 
                      occ_size,
                      curr_ego_to_future_ego=None):
    """Warp the given voxel features using the predicted voxel flow.

    Args:
        voxel_feats (Tensor): _description_
        voxel_flow (Tensor): (bs, f, H, W, 2)
        voxel_size (Tensor): the voxel size for each voxel, for example torch.Tensor([0.4, 0.4])
        occ_size (Tensor): the size of the occupancy map, for example torch.Tensor([200, 200])
        extrinsic_matrix (_type_, optional): global to ego transformation matrix. Defaults to None.

    Returns:
        _type_: _description_
    """
    device = voxel_feats.device
    bs, num_pred, x_size, y_size, c = voxel_flow.shape

    if curr_ego_to_future_ego is not None:
        for i in range(bs):
            _extrinsic_matrix = curr_ego_to_future_ego[i]
            _voxel_flow = voxel_flow[i].reshape(num_pred, -1, 2)
            ## padding the zero flow for z axis
            _voxel_flow = torch.cat([_voxel_flow, torch.zeros(num_pred, _voxel_flow.shape[1], 1).to(device)], dim=-1)
            trans_flow = torch.matmul(_extrinsic_matrix[..., :3, :3], _voxel_flow.permute(0, 2, 1))
            trans_flow = trans_flow + _extrinsic_matrix[..., :3, 3][:, :, None]
            trans_flow = trans_flow.permute(0, 2, 1)[..., :2]
            voxel_flow[i] = trans_flow.reshape(num_pred, *voxel_flow.shape[2:])

    voxel_flow = rearrange(voxel_flow, 'b f h w dim2 -> (b f) h w dim2')
    new_bs = voxel_flow.shape[0]

    # normalize the flow in m/s unit to voxel unit and then to [-1, 1]
    voxel_size = voxel_size.to(device)
    occ_size = occ_size.to(device)

    voxel_flow = voxel_flow / voxel_size / occ_size

    # generate normalized grid
    x = torch.linspace(-1.0, 1.0, x_size).view(-1, 1).repeat(1, y_size).to(device)
    y = torch.linspace(-1.0, 1.0, y_size).view(1, -1).repeat(x_size, 1).to(device)
    grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)  # (h, w, 2)
    
    # add flow to grid
    grid = grid.unsqueeze(0).expand(new_bs, -1, -1, -1).flip(-1) + voxel_flow

    # perform the voxel feature warping
    voxel_feats = torch.repeat_interleave(voxel_feats, num_pred, dim=0)
    warped_voxel_feats = F.grid_sample(voxel_feats, 
                                       grid.float(), 
                                       mode='bilinear', 
                                       padding_mode='border')
    warped_voxel_feats = rearrange(warped_voxel_feats, '(b f) c h w -> b f c h w', b=bs)

    return warped_voxel_feats


def warp_voxel_features(voxel_feats, 
                        voxel_flow,
                        voxel_size, 
                        occ_size,
                        curr_ego_to_future_ego=None):
    """Warping the voxel features using the predicted voxel flow.

    Args:
        voxel_feats (Tensor): [bs, c, h, w, d]
        voxel_flow (_type_): torch.Size([bs, f, h, w, d, 2])
        voxel_size (_type_): voxel resolution in meters
        occ_size (_type_): voxel size in grid numbers
        curr_ego_to_future_ego (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    device = voxel_flow.device
    bs, num_pred, x_size, y_size, z_size, c = voxel_flow.shape

    if curr_ego_to_future_ego is not None:
        for i in range(bs):
            _extrinsic_matrix = curr_ego_to_future_ego[i]
            _voxel_flow = voxel_flow[i].reshape(num_pred, -1, 2)
            _voxel_flow = torch.cat([_voxel_flow, torch.zeros(num_pred, _voxel_flow.shape[1], 1).to(device)], dim=-1)
            trans_flow = torch.matmul(_extrinsic_matrix[:, :3, :3], _voxel_flow.permute(0, 2, 1))
            trans_flow = trans_flow + _extrinsic_matrix[..., :3, 3][:, :, None]
            trans_flow = trans_flow.permute(0, 2, 1)[..., :2]
            voxel_flow[i] = trans_flow.reshape(num_pred, *voxel_flow.shape[2:])
    
    ## padding the zero flow for z axis
    voxel_flow = torch.cat([voxel_flow, torch.zeros(bs, num_pred, x_size, y_size, z_size, 1).to(device)], dim=-1)

    voxel_flow = rearrange(voxel_flow, 'b f h w d dim3 -> (b f) h w d dim3')
    
    # normalize the flow in m/s unit to voxel unit and then to [-1, 1]
    voxel_size = voxel_size.to(device)
    occ_size = occ_size.to(device)

    voxel_flow = voxel_flow / voxel_size / occ_size

    # generate normalized grid
    x = torch.linspace(-1.0, 1.0, x_size).view(-1, 1, 1).repeat(1, y_size, z_size).to(device)
    y = torch.linspace(-1.0, 1.0, y_size).view(1, -1, 1).repeat(x_size, 1, z_size).to(device)
    z = torch.linspace(-1.0, 1.0, z_size).view(1, 1, -1).repeat(x_size, y_size, 1).to(device)
    grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
    
    # add flow to grid
    grid = grid.unsqueeze(0).expand(bs, -1, -1, -1, -1).flip(-1) + voxel_flow

    if not isinstance(voxel_feats, list):
        voxel_feats = [voxel_feats]

    outputs = []
    for _feat in voxel_feats:
        if _feat is None:
            outputs.append(None)
            continue

        # perform the voxel feature warping
        _feat = _feat.unsqueeze(1).expand(-1, num_pred, -1, -1, -1, -1)
        _feat = rearrange(_feat, 'b f c h w d -> (b f) c h w d')
        warped_voxel_feats = F.grid_sample(_feat, 
                                           grid.float(), 
                                           mode='nearest', 
                                           padding_mode='border', 
                                           align_corners=False)
        warped_voxel_feats = rearrange(warped_voxel_feats, '(b f) c h w d -> b f c h w d', b=bs)
        warped_voxel_feats = warped_voxel_feats.squeeze(1)
        outputs.append(warped_voxel_feats)

    return outputs


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1, 
                 transpose=False, act_norm=False):
        super(BasicConv3d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv3d(in_channels, out_channels, 
                                  kernel_size=kernel_size, 
                                  stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


@HEADS.register_module()
class PretrainHead(BaseModule):
    def __init__(
        self,
        in_channels=128,
        view_cfg=None,
        uni_conv_cfg=None,
        render_head_cfg=None,
        render_scale=(1, 1),
        use_semantic=False,
        semantic_class=17,
        vis_gt=False,
        vis_pred=False,
        use_depth_consistency=False,
        render_view_indices=list(range(6)),
        depth_ssl_size=None,
        depth_loss_weight=1.0,
        rgb_loss_weight=1.0,
        use_depth_gt_loss=False,
        use_semantic_gt_loss=False,
        depth_gt_loss_weight=1.0,
        opt=None,
        save_dir=None,
        pred_flow=False,
        voxel_shape=None,
        voxel_size=None,
        use_flow_ssl=False,
        use_flow_photometric_loss=False,
        flow_depth_loss_weight=0.15,
        use_flow_rgb=False,
        rgb_future_loss_weight=1.0,
        use_flow_refine_layer=False,
        use_sperate_render_head=False,
        use_pseudo_depth_loss=False,
        pseudo_depth_loss_weight=1.0,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels

        self.use_semantic = use_semantic
        self.vis_gt = vis_gt
        self.vis_pred = vis_pred

        self.pred_flow = pred_flow
        self.use_flow_ssl = use_flow_ssl
        self.use_flow_photometric_loss = use_flow_photometric_loss # whether to use the photometric loss for the flow SSL
        self.flow_depth_loss_weight = flow_depth_loss_weight
        self.use_flow_rgb = use_flow_rgb  # use the RGB to supervise the flow
        self.rgb_future_loss_weight = rgb_future_loss_weight
        self.use_flow_refine_layer = use_flow_refine_layer
        self.use_sperate_render_head = use_sperate_render_head

        self.voxel_shape = voxel_shape
        self.voxel_size = voxel_size

        self.steps = -1

        ## use the depth self-supervised consistency loss
        self.use_depth_consistency = use_depth_consistency
        self.render_view_indices = render_view_indices
        self.depth_ssl_size = depth_ssl_size
        self.opt = opt  # options for the depth consistency loss
        self.depth_loss_weight = depth_loss_weight

        self.rgb_loss_weight = rgb_loss_weight
        
        self.use_depth_gt_loss = use_depth_gt_loss
        self.depth_gt_loss_weight = depth_gt_loss_weight

        self.use_semantic_gt_loss = use_semantic_gt_loss

        self.use_pseudo_depth_loss = use_pseudo_depth_loss
        self.pseudo_depth_loss_weight = pseudo_depth_loss_weight

        self.save_dir = save_dir

        if self.use_depth_consistency:
            h = depth_ssl_size[0]
            w = depth_ssl_size[1]
            num_cam = len(self.render_view_indices)
            self.backproject_depth = BackprojectDepth(num_cam, h, w)
            self.project_3d = Project3D(num_cam, h, w)

            self.ssim = SSIM()

        if view_cfg is not None:
            vtrans_type = view_cfg.pop('type', 'Uni3DViewTrans')
            self.view_trans = getattr(utils, vtrans_type)(**view_cfg)

        if uni_conv_cfg is not None:
            self.uni_conv = nn.Sequential(
                nn.Conv3d(
                    uni_conv_cfg["in_channels"],
                    uni_conv_cfg["out_channels"],
                    kernel_size=uni_conv_cfg["kernel_size"],
                    padding=uni_conv_cfg["padding"],
                    stride=1,
                ),
                nn.BatchNorm3d(uni_conv_cfg["out_channels"]),
                nn.ReLU(inplace=True),
            )

        if render_head_cfg is not None:
            self.render_head = builder.build_head(render_head_cfg)

        self.render_head_cfg = render_head_cfg

        self.render_scale = render_scale

        out_dim = uni_conv_cfg["out_channels"]
        self.out_dim = out_dim

        if use_semantic:
            self.semantic_head = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, semantic_class),
            )

        self.occupancy_head = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, 1),
            )

        if self.pred_flow:
            self.flow_head = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, 2),
            )
        
        if self.use_flow_refine_layer:
            self.flow_refine_layer = nn.Sequential(
                nn.Conv3d(
                    32,
                    32,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
            )

        if self.use_sperate_render_head:
            # use a sperate render head the future volume feature
            self.render_head_future = builder.build_head(render_head_cfg)

            self.occupancy_head_future = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, 1),
            )
        
    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, 
                pts_feats, 
                img_feats, 
                img_metas, 
                img_depth,
                img_inputs,
                **kwargs):

        output = dict()

        # Prepare the projection parameters
        lidar2cam, intrinsics = [], []
        for img_meta in img_metas:
            lidar2cam.append(img_meta["lidar2cam"])
            intrinsics.append(img_meta["cam_intrinsic"])
        lidar2cam = np.asarray(lidar2cam)  # (bs, 6, 1, 4, 4)
        intrinsics = np.asarray(intrinsics)

        ref_tensor = img_feats[0].float()

        intrinsics = ref_tensor.new_tensor(intrinsics)
        pose_spatial = torch.inverse(
            ref_tensor.new_tensor(lidar2cam)
        )

        output['pose_spatial'] = pose_spatial[:, :, 0]
        output['intrinsics'] = intrinsics[:, :, 0]  # (bs, 6, 4, 4)
        output['intrinsics'][:, :, 0] *= self.render_scale[1]
        output['intrinsics'][:, :, 1] *= self.render_scale[0]

        if self.vis_gt:
            ## NOTE: due to the occ gt is labelled in the ego coordinate, we need to
            # use the cam2ego matrix as ego matrix
            cam2camego = []
            for img_meta in img_metas:
                cam2camego.append(img_meta["cam2camego"])
            cam2camego = np.asarray(cam2camego)  # (bs, 6, 1, 4, 4)
            output['pose_spatial'] = ref_tensor.new_tensor(cam2camego)

            gt_data_dict = self.prepare_gt_data(**kwargs)
            output.update(gt_data_dict)
            render_results = self.render_head(output, vis_gt=True)
            
            ## visualiza the results
            render_depth, rgb_pred, semantic_pred = render_results
            # current_frame_img = torch.zeros_like(rgb_pred).cpu().numpy()
            current_frame_img = img_inputs.cpu().numpy()
            visualize_image_semantic_depth_pair(
                current_frame_img[0],
                rgb_pred[0].permute(0, 2, 3, 1),
                render_depth[0],
                save_dir="results/vis/3dgs_baseline_gt"
            )
            exit()

        ## 1. Prepare the volume feature from the pts features and img features
        uni_feats = []
        if img_feats is not None:
            uni_feats.append(
                self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
            )
        if pts_feats is not None:
            uni_feats.append(pts_feats)

        uni_feats = sum(uni_feats)
        uni_feats = self.uni_conv(uni_feats)  # (bs, c, z, y, x)

        ## 2. Prepare the features for rendering
        _uni_feats = rearrange(uni_feats, 'b c z y x -> b x y z c')

        output['volume_feat'] = _uni_feats

        occupancy_output = self.occupancy_head(_uni_feats)
        occupancy_output = rearrange(occupancy_output, 'b x y z dim1 -> b dim1 x y z')
        output['density_prob'] = occupancy_output  # density score

        ## 3. Prepare the semantic features
        output['semantic'] = None
        if self.use_semantic:
            semantic_output = self.semantic_head(_uni_feats)
            semantic_output = rearrange(semantic_output, 'b x y z C -> b C x y z')
            output['semantic'] = semantic_output

        if self.pred_flow:
            flow_output = self.flow_head(_uni_feats)
            flow_output = rearrange(flow_output, 'b x y z dim1 -> b () x y z dim1')
            output['flow'] = flow_output

        ## 2. Start rendering, including neural rendering or 3DGS
        render_results = self.render_head(output)

        if self.use_flow_ssl:
            # the Flow-based SSL
            assert self.pred_flow, "The flow prediction is required for the flow self-supervised loss!"

            # density_prob = output['density_prob']
            # semantic = output['semantic']
            volume_feature = output['volume_feat']
            voxel_flow_pred = output['flow']

            warped_results = warp_voxel_features(
                rearrange(volume_feature, 'b x y z c -> b c x y z'), 
                voxel_flow_pred, 
                voxel_size=torch.Tensor(self.voxel_size), 
                occ_size=torch.Tensor(self.voxel_shape),
                curr_ego_to_future_ego=kwargs.get('curr_lidar_T_future_lidar', None))
            
            future_volume_feat = warped_results[0]  # (bs, c, x, y, z)
            if self.use_flow_refine_layer:
                future_volume_feat = self.flow_refine_layer(future_volume_feat)
            future_volume_feat = rearrange(future_volume_feat, 'b c x y z -> b x y z c')

            future_output = dict()
            future_output['volume_feat'] = future_volume_feat

            future_output['pose_spatial'] = kwargs['pose_spatial_future']
            future_output['intrinsics'] = kwargs['cam_intrinsic_future']
            future_output['intrinsics'][:, :, 0] *= self.render_scale[1]
            future_output['intrinsics'][:, :, 1] *= self.render_scale[0]

            # start rendering future volume feature
            if self.use_sperate_render_head:
                occupancy_output = self.occupancy_head_future(future_volume_feat)
                future_output['density_prob'] = rearrange(
                    occupancy_output, 'b x y z dim1 -> b dim1 x y z') # density score
                # TODO: add semantic head
                future_output['semantic'] = None
                future_render_results = self.render_head_future(future_output, suffix='_future')
            else:
                occupancy_output = self.occupancy_head(future_volume_feat)
                future_output['density_prob'] = rearrange(
                    occupancy_output, 'b x y z dim1 -> b dim1 x y z') # density score

                future_output['semantic'] = None
                if self.use_semantic:
                    semantic_output = self.semantic_head(future_volume_feat)
                    future_output['semantic'] = rearrange(semantic_output, 'b x y z C -> b C x y z')

                future_render_results = self.render_head(future_output, suffix='_future')

            render_results.update(future_render_results)

        ## Visualize the results
        if self.vis_pred:
            from .nerf_utils import VisElement, visualize_elements

            save_dir = self.save_dir
            os.makedirs(save_dir, exist_ok=True)

            ## save the occupancy offline for visualization
            # torch.save(semantic_output.detach().cpu(), f'{save_dir}/semantic_pred.pth')
            # torch.save(occupancy_output.detach().cpu(), f'{save_dir}/occupancy_pred.pth')

            render_depth = render_results['render_depth']
            rgb_pred = render_results['render_rgb']
            semantic_pred = render_results['render_semantic']

            render_gt_semantic = kwargs.get('render_gt_semantic', None)
            render_gt_depth = kwargs.get('render_gt_depth', None)

            semantic = semantic_pred.argmax(2)
            semantic = OCC3D_PALETTE[semantic].to(semantic_pred)
            # visualize_image_pairs(
            #     img_inputs[0],
            #     semantic[0], # rgb_pred[0].permute(0, 2, 3, 1),
            #     render_depth[0],
            #     semantic_is_sparse=False,
            #     depth_is_sparse=False,
            #     save_dir=save_dir
            # )

            target_size = (semantic.shape[2], semantic.shape[3])  # (H, W)
            target_size = (180, 320)
            visualize_elements(
                [
                    # VisElement(
                    #     img_inputs[0],
                    #     type='rgb'
                    # ),
                    # VisElement(
                    #     rgb_pred[0],
                    #     type='rgb',
                    #     need_denormalize=False,
                    # ),
                    VisElement(
                        render_depth[0],
                        type='depth',
                        is_sparse=False,
                    ),
                    # VisElement(
                    #     semantic[0],
                    #     type='semantic',
                    #     is_sparse=False,
                    # ),
                ],
                target_size=target_size,
                save_dir=save_dir
            )
            # exit()
        return render_results
    
    def compute_rgb_loss(self, 
                         pred_img, 
                         target_img, 
                         target_size=None,
                         use_ssim=False):
        if target_size is None:
            target_size = (target_img.shape[-2], target_img.shape[-1])
        
        _pred_img = rearrange(
            pred_img, 'b num_view dim3 h w -> (b num_view) dim3 h w')
        _pred_img = F.interpolate(
            _pred_img, target_size, mode="bilinear", align_corners=False)
        _pred_img = rearrange(
            _pred_img, '(b num_view) dim3 h w -> b num_view dim3 h w', 
            b=pred_img.shape[0])
        rgb_loss = F.l1_loss(_pred_img, target_img)
        return rgb_loss
    
    def loss(self, preds_dict, targets):
        if self.use_depth_consistency:
            ## Visualize the input data
            DEBUG = False
            if DEBUG:
                from .nerf_utils import VisElement, visualize_elements
                save_dir = "results/vis/3dgs_depth_ssl"
                os.makedirs(save_dir, exist_ok=True)

                source_imgs = targets['source_imgs'][0]  # to (2, N, 3, h, w)
                target_size = (source_imgs.shape[-2], source_imgs.shape[-1])  # (H, W)

                curr_imgs = targets['target_imgs'][0]  # to (N, 3, h, w)
                visualize_elements(
                    [
                        VisElement(
                            source_imgs[0],
                            type='rgb'
                        ),
                        VisElement(
                            curr_imgs,
                            type='rgb',
                        ),
                        VisElement(
                            source_imgs[1],
                            type='rgb',
                        ),
                    ],
                    target_size=target_size,
                    save_dir=save_dir
                )
                exit()
            
            # preds_dict['render_depth_raw'] = preds_dict['render_depth'].clone()

            device = preds_dict['render_depth'].device

            loss_dict = {}

            if self.depth_loss_weight > 0.0:
                ## 1) Compute the reprojected rgb images based on the rendered depth
                self.generate_image_pred(targets, preds_dict)

                ## 2) Compute the depth consistency loss
                loss_depth_ssl = self.compute_self_supervised_losses(targets, preds_dict)
                loss_dict.update(loss_depth_ssl)

            if self.use_flow_ssl:
                if not self.use_flow_photometric_loss:
                    ## Compute the flow loss with GT
                    loss_flow_ssl = self.render_head.loss(
                        preds_dict, targets, 
                        suffix='_future', weight=self.flow_depth_loss_weight)
                    loss_dict.update(loss_flow_ssl)
                else:
                    if self.use_flow_rgb:
                        render_rgb = preds_dict['render_rgb_future']  # (bs, num_cam, 3, h, w)
                        target_img = targets['target_imgs_future']  # (bs, num_cam, 3, h, w)

                        render_rgb = rearrange(
                            render_rgb, 'b num_view dim3 h w -> (b num_view) dim3 h w')
                        render_rgb = F.interpolate(
                            render_rgb, self.depth_ssl_size, 
                            mode="bilinear", align_corners=False)
                        render_rgb = rearrange(
                            render_rgb, '(b num_view) dim3 h w -> b num_view dim3 h w', b=target_img.shape[0])
                        rgb_loss = self.rgb_future_loss_weight * F.l1_loss(render_rgb, target_img)
                        loss_rgb = {'loss_rgb_future' : rgb_loss}
                        loss_dict.update(loss_rgb)
                    else:
                        ## Compute the photometric loss for the flow SSL
                        self.generate_image_pred(targets, preds_dict, suffix='_future')

                        loss_flow_ssl = self.compute_self_supervised_losses(
                            targets, preds_dict, suffix='_future')
                        loss_dict.update(loss_flow_ssl)

            ## 3) Compute the RGB reconstruction loss
            if self.rgb_loss_weight > 0.0:
                render_curr_img = preds_dict['render_rgb']  # (bs, num_cam, 3, h, w)
                target_img = targets['target_imgs']  # (bs, num_cam, 3, h, w)

                rgb_loss = self.rgb_loss_weight * self.compute_rgb_loss(
                    render_curr_img, target_img, target_size=self.depth_ssl_size)
                loss_rgb = {'loss_rgb' : rgb_loss}
                loss_dict.update(loss_rgb)

            if self.use_pseudo_depth_loss:
                pred_depth = preds_dict['render_depth'] # (bs, num_cam, h, w)
                target_depth = targets['pseudo_depth']
                patch_range = (5, 17)
                error_tolerance = 0.00025

                loss_pseudo_depth = patch_norm_mse_loss(
                    pred_depth, target_depth, 
                    randint(patch_range[0], patch_range[1]), error_tolerance)
                loss_dict['loss_pseudo_depth'] = self.pseudo_depth_loss_weight * loss_pseudo_depth

            ## 4) Compute the depth gt loss
            if self.use_depth_gt_loss:
                render_depth = preds_dict['render_depth']
                gt_depth = targets['render_gt_depth']

                mask = gt_depth > 0.0
                loss_render_depth = F.l1_loss(render_depth[mask], gt_depth[mask])
                if torch.isnan(loss_render_depth):
                    print('NaN in render depth loss!')
                    loss_render_depth = torch.Tensor([0.0]).to(device)
                loss_dict['loss_render_depth'] = self.depth_gt_loss_weight * loss_render_depth

            if self.use_semantic_gt_loss:
                assert 'render_gt_semantic' in targets.keys()

                semantic_gt = targets['render_gt_semantic']
                semantic_pred = preds_dict['render_semantic']
                
                loss_render_sem = self.compute_semantic_loss(
                    semantic_pred, semantic_gt, ignore_index=255)
                if torch.isnan(loss_render_sem):
                    print('NaN in render semantic loss!')
                    loss_render_sem = torch.Tensor([0.0]).to(device)
                loss_dict['loss_render_sem'] = 0.1 * loss_render_sem
        else:
            loss_dict = self.render_head.loss(preds_dict, targets)

            if self.use_flow_ssl:
                # compute the loss for the future frame
                future_loss_dict = self.render_head.loss(
                    preds_dict, targets, suffix='_future', weight=0.15)
                
                loss_dict.update(future_loss_dict)

            if self.save_dir is not None:
                rank, _ = get_dist_info()
                
                if rank == 0:
                    self.steps += 1
                    if self.steps % 200 == 0:
                        from .nerf_utils import VisElement, visualize_elements

                        save_dir = self.save_dir

                        render_depth = preds_dict['render_depth'][0]  # (M, h, w)
                        render_gt_depth = targets['render_gt_depth'][0]
                        input_img = targets['input_img'][0]  # (num_cam, 3, h, w)

                        print(f"Render depth: min={render_depth.min().item()}, max={render_depth.max().item()}")

                        target_size = (render_depth.shape[-2], render_depth.shape[-1])  # (H, W)
                        visualize_elements(
                            [
                                VisElement(
                                    input_img,
                                    type='rgb',
                                ),
                                VisElement(
                                    render_gt_depth,
                                    type='depth',
                                    is_sparse=True,
                                ),
                                VisElement(
                                    render_depth,
                                    type='depth',
                                    is_sparse=False,
                                ),
                            ],
                            target_size=target_size,
                            save_dir=save_dir,
                            prefix=f"{self.steps:05d}"
                        )

        ## Visualization
        if self.use_depth_consistency and self.save_dir is not None:
            rank, _ = get_dist_info()
            
            if rank == 0:
                self.steps += 1
                if self.steps % 200 == 0:
                    from .nerf_utils import VisElement, visualize_elements

                    save_dir = self.save_dir

                    render_depth = preds_dict['render_depth']  # (1, M, h, w)

                    print(f"Render depth: min={render_depth.min().item()}, max={render_depth.max().item()}")

                    render_rgb = preds_dict['render_rgb']  # (M, num_cam, 3, h, w)

                    prev_img = targets['source_imgs'][:, 0]  # (1, num_cam, 3, h, w)
                    next_img = targets['source_imgs'][:, 1]  # (1, num_cam, 3, h, w)
                    target_img = targets['target_imgs']  # (1, num_cam, 3, h, w)

                    target_size = (render_depth.shape[-2], render_depth.shape[-1])  # (H, W)
                    visualize_elements(
                        [
                            VisElement(
                                prev_img[0],
                                type='rgb'
                            ),
                            VisElement(
                                target_img[0],
                                type='rgb'
                            ),
                            VisElement(
                                next_img[0],
                                type='rgb',
                            ),
                            VisElement(
                                render_depth[0],
                                type='depth',
                                is_sparse=False,
                            ),
                            VisElement(
                                render_rgb[0],
                                type='rgb',
                            ),
                        ],
                        target_size=target_size,
                        save_dir=save_dir,
                        prefix=f"{self.steps:05d}"
                    )

        return loss_dict

    def compute_semantic_loss(self, sem_est, sem_gt, ignore_index=-100):
        '''
        Args:
            sem_est: B, N, C, H, W, predicted unnormalized logits
            sem_gt: B, N, H, W
        '''
        B, N, C, H, W = sem_est.shape
        sem_est = sem_est.view(B * N, -1, H, W)
        sem_gt = sem_gt.view(B * N, H, W)
        loss = F.cross_entropy(sem_est, sem_gt.long(), ignore_index=ignore_index)

        return loss
    
    def compute_self_supervised_losses(self, inputs, outputs, suffix=''):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        loss = 0

        depth = outputs["render_depth_rescaled" + suffix]  # (M, 1, h, w)
        disp = 1.0 / (depth + 1e-7)
        color = outputs["target_imgs" + suffix]
        target = outputs["target_imgs" + suffix]

        reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection' + suffix])):
            pred = outputs['color_reprojection' + suffix][frame_id]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)  # (M, 2, h, w)

        ## automasking
        identity_reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection' + suffix])):
            pred = inputs["color_source_imgs" + suffix][frame_id]
            identity_reprojection_losses.append(
                self.compute_reprojection_loss(pred, target))

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        if self.opt.avg_reprojection:
            identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
        else:
            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not self.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        loss += self.opt.disparity_smoothness * smooth_loss
        
        total_loss += loss
        losses["loss_depth_ct" + suffix] = self.depth_loss_weight * total_loss  # depth consistency loss
        return losses
    
    def generate_image_pred(self, inputs, outputs, suffix=''):
        color_source_imgs_list = []
        for idx in range(inputs['source_imgs' + suffix].shape[1]):
            color_source = inputs['source_imgs' + suffix][:, idx]  # prev and next images
            color_source = rearrange(color_source, 'b num_view c h w -> (b num_view) c h w')
            color_source_imgs_list.append(color_source)
        inputs['color_source_imgs' + suffix] = color_source_imgs_list

        inv_K = inputs['inv_K' + suffix][:, self.render_view_indices]
        K = inputs['K' + suffix][:, self.render_view_indices]
        inv_K = rearrange(inv_K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')
        K = rearrange(K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')

        # rescale the rendered depth
        depth = outputs['render_depth' + suffix][:, self.render_view_indices]
        depth = rearrange(depth, 'b num_view h w -> (b num_view) () h w')
        depth = F.interpolate(
            depth, self.depth_ssl_size, mode="bilinear", align_corners=False)
        outputs['render_depth_rescaled' + suffix] = depth

        cam_T_cam = inputs["cam_T_cam" + suffix][:, :, self.render_view_indices]

        ## 1) Depth to camera points
        cam_points = self.backproject_depth(depth, inv_K)  # (M, 4, h*w)
        len_temporal = cam_T_cam.shape[1]
        color_reprojection_list = []
        for frame_id in range(len_temporal):
            T = cam_T_cam[:, frame_id]
            T = rearrange(T, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')
            ## 2) Camera points to adjacent image points
            pix_coords = self.project_3d(cam_points, K, T)  # (M, h, w, 2)

            ## 3) Reproject the adjacent image
            color_source = inputs['color_source_imgs' + suffix][frame_id]  # (M, 3, h, w)
            color_reprojection = F.grid_sample(
                color_source,
                pix_coords,
                padding_mode="border", align_corners=True)
            color_reprojection_list.append(color_reprojection)

        outputs['color_reprojection' + suffix] = color_reprojection_list
        outputs['target_imgs' + suffix] = rearrange(
            inputs['target_imgs' + suffix], 'b num_view c h w -> (b num_view) c h w')

    def compute_reprojection_loss(self, pred, target, no_ssim=False):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    def prepare_gt_data(self, **kwargs):
        # Prepare the ground truth volume data for visualization
        voxel_semantics = kwargs['voxel_semantics']
        density_prob = rearrange(voxel_semantics, 'b x y z -> b () x y z')
        density_prob = density_prob != 17
        density_prob = density_prob.float()
        density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
        density_prob[density_prob == 1] = 10

        output = dict()
        output['density_prob'] = density_prob

        semantic = OCC3D_PALETTE[voxel_semantics.long()].to(density_prob)
        semantic = semantic.permute(0, 4, 1, 2, 3)  # to (b, 3, 200, 200, 16)
        output['semantic'] = semantic
        return output
        

@HEADS.register_module()
class PretrainHeadWithFlowGuidance(PretrainHead):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        # self.refine_flow = nn.Sequential(
        #     nn.Conv3d(
        #         self.in_channels,
        #         self.in_channels,
        #         kernel_size=3,
        #         padding=1,
        #         stride=1
        #     ),
        #     nn.BatchNorm3d(self.in_channels),
        #     nn.ReLU(inplace=True),
        # )

        if self.use_flow_refine_layer:
            self.flow_refine_layer_prev = nn.Sequential(
                nn.Conv3d(
                    32,
                    32,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
            )

        if self.use_sperate_render_head:
            # use a sperate render head the previous volume feature
            self.render_head_prev = builder.build_head(self.render_head_cfg)

            self.occupancy_head_prev = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, 1),
            )
    
    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, 
                pts_feats, 
                img_feats, 
                img_metas, 
                img_depth,
                img_inputs,
                **kwargs):
        input_dict = dict()

        # Prepare the projection parameters
        lidar2cam, intrinsics = [], []
        for img_meta in img_metas:
            lidar2cam.append(img_meta["lidar2cam"])
            intrinsics.append(img_meta["cam_intrinsic"])
        lidar2cam = np.asarray(lidar2cam)  # (bs, 6, 1, 4, 4)
        intrinsics = np.asarray(intrinsics)

        ref_tensor = img_feats[0].float()

        intrinsics = ref_tensor.new_tensor(intrinsics)
        pose_spatial = torch.inverse(
            ref_tensor.new_tensor(lidar2cam)
        )

        input_dict['pose_spatial'] = pose_spatial[:, :, 0]
        input_dict['intrinsics'] = intrinsics[:, :, 0]  # (bs, 6, 4, 4)
        input_dict['intrinsics'][:, :, 0] *= self.render_scale[1]
        input_dict['intrinsics'][:, :, 1] *= self.render_scale[0]

        ## 1. Prepare the volume feature from the pts features and img features
        uni_feats = []
        if img_feats is not None:
            uni_feats.append(
                self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
            )
        if pts_feats is not None:
            uni_feats.append(pts_feats)

        uni_feats = sum(uni_feats)
        uni_feats = self.uni_conv(uni_feats)  # (bs, c, z, y, x)

        ## 2. Prepare the features for rendering
        _uni_feats = rearrange(uni_feats, 'b c z y x -> b x y z c')

        input_dict['volume_feat'] = _uni_feats

        flow_output = self.flow_head(_uni_feats)
        flow_output = rearrange(flow_output, 'b x y z dim2 -> b () x y z dim2')
        input_dict['flow'] = flow_output

        volume_feature = input_dict['volume_feat']
        voxel_flow_pred = input_dict['flow']

        render_results = dict()

        ## Warp to the next frame
        next_warped_results = warp_voxel_features(
            rearrange(volume_feature, 'b x y z c -> b c x y z'), 
            voxel_flow_pred.clone(), 
            voxel_size=torch.Tensor(self.voxel_size), 
            occ_size=torch.Tensor(self.voxel_shape),
            curr_ego_to_future_ego=kwargs['curr_lidar_T_future_lidar'])
        
        future_volume_feat = next_warped_results[0]  # (bs, c, x, y, z)
        if self.use_flow_refine_layer:
            future_volume_feat = self.flow_refine_layer(future_volume_feat)
        future_volume_feat = rearrange(future_volume_feat, 'b c x y z -> b x y z c')

        future_output = dict()
        future_output['volume_feat'] = future_volume_feat

        ## Predict the 3DGS by using the predicted future volume feat
        future_output['pose_spatial'] = kwargs['pose_spatial_future']
        future_output['intrinsics'] = kwargs['cam_intrinsic_future']
        future_output['intrinsics'][:, :, 0] *= self.render_scale[1]
        future_output['intrinsics'][:, :, 1] *= self.render_scale[0]

        if self.use_sperate_render_head:
            occupancy_output = self.occupancy_head_future(future_volume_feat)
            future_output['density_prob'] = rearrange(
                occupancy_output, 'b x y z dim1 -> b dim1 x y z') # density score
            # TODO: add semantic head
            future_output['semantic'] = None
            future_render_results, gaussians = self.render_head_future(
                future_output, return_gaussians=True, suffix='_future')
        else:
            occupancy_output = self.occupancy_head(future_volume_feat)
            future_output['density_prob'] = rearrange(
                occupancy_output, 'b x y z dim1 -> b dim1 x y z') # density score

            future_output['semantic'] = None
            if self.use_semantic:
                semantic_output = self.semantic_head(future_volume_feat)
                future_output['semantic'] = rearrange(semantic_output, 'b x y z C -> b C x y z')
            future_render_results, gaussians = self.render_head(
                future_output, return_gaussians=True, suffix='_future')

        render_results.update(future_render_results)
        
        ## Use the future volume feat to render the current frame depth
        future_output['pose_spatial'] = input_dict['pose_spatial'].clone()
        future_output['intrinsics'] = input_dict['intrinsics'].clone()

        if self.use_sperate_render_head:
            curr_render_results_w_future = self.render_head_future.render_forward(
                future_output, gaussians, suffix='_future')
        else:
            curr_render_results_w_future = self.render_head.render_forward(
                future_output, gaussians, suffix='_next')
        render_results.update(curr_render_results_w_future)

        ## ============== Warp to the prev frame ==============
        prev_warped_results = warp_voxel_features(
            rearrange(volume_feature, 'b x y z c -> b c x y z'), 
            -voxel_flow_pred.clone(), 
            voxel_size=torch.Tensor(self.voxel_size), 
            occ_size=torch.Tensor(self.voxel_shape),
            curr_ego_to_future_ego=kwargs['curr_lidar_T_prev_lidar'])
        
        prev_volume_feat = prev_warped_results[0]  # (bs, c, x, y, z)
        if self.use_flow_refine_layer:
            prev_volume_feat = self.flow_refine_layer_prev(prev_volume_feat)
        prev_volume_feat = rearrange(prev_volume_feat, 'b c x y z -> b x y z c')
        
        prev_output = dict()
        prev_output['volume_feat'] = prev_volume_feat

        ## Predict the 3DGS by using the predicted previous volume feat
        prev_output['pose_spatial'] = kwargs['pose_spatial_prev']
        prev_output['intrinsics'] = kwargs['cam_intrinsic_prev']
        prev_output['intrinsics'][:, :, 0] *= self.render_scale[1]
        prev_output['intrinsics'][:, :, 1] *= self.render_scale[0]

        if self.use_sperate_render_head:
            occupancy_output = self.occupancy_head_prev(prev_volume_feat)
            prev_output['density_prob'] = rearrange(
                occupancy_output, 'b x y z dim1 -> b dim1 x y z')
            # TODO: add semantic head
            prev_output['semantic'] = None
            prev_render_results, gaussians = self.render_head_prev(
                prev_output, return_gaussians=True, suffix='_prev')
        else:
            occupancy_output = self.occupancy_head(prev_volume_feat)
            prev_output['density_prob'] = rearrange(occupancy_output, 'b x y z dim1 -> b dim1 x y z') # density score

            prev_output['semantic'] = None
            if self.use_semantic:
                semantic_output = self.semantic_head(prev_volume_feat)
                prev_output['semantic'] = rearrange(semantic_output, 'b x y z C -> b C x y z')
        
            prev_render_results, gaussians = self.render_head(
                prev_output, return_gaussians=True, suffix='_prev')  # previous frame
        
        render_results.update(prev_render_results)

        ## Use the previous volume feat to render the current frame depth
        prev_output['pose_spatial'] = input_dict['pose_spatial'].clone()
        prev_output['intrinsics'] = input_dict['intrinsics'].clone()
        if self.use_sperate_render_head:
            curr_render_results_w_prev = self.render_head_prev.render_forward(
                prev_output, gaussians, suffix='_last')
        else:
            curr_render_results_w_prev = self.render_head.render_forward(
                prev_output, gaussians, suffix='_last')
        render_results.update(curr_render_results_w_prev)
        
        return render_results

    def loss(self, preds_dict, targets):
        
        loss_dict = {}

        if self.depth_loss_weight > 0.0:
            ## 1) Warp the future image by using the rendered next depth
            self.generate_image_pred(targets, preds_dict)

            ## 2) Compute the depth consistency loss
            loss_depth_ssl = self.compute_self_supervised_losses(targets, preds_dict)
            loss_dict.update(loss_depth_ssl)

        loss_rgb_future = self.compute_rgb_loss(
            preds_dict['render_rgb_future'], targets['target_imgs_future'])
        loss_dict['loss_rgb_future'] = 0.1 * loss_rgb_future

        loss_rgb_prev = self.compute_rgb_loss(
            preds_dict['render_rgb_prev'], targets['target_imgs_prev'])
        loss_dict['loss_rgb_prev'] = 0.1 * loss_rgb_prev
        
        return loss_dict
    
    def generate_image_pred(self, inputs, outputs, suffix=''):
        color_source_imgs_list = []
        for idx in range(inputs['source_imgs' + suffix].shape[1]):
            color_source = inputs['source_imgs' + suffix][:, idx]  # prev and next images
            color_source = rearrange(color_source, 'b num_view c h w -> (b num_view) c h w')
            color_source_imgs_list.append(color_source)
        inputs['color_source_imgs' + suffix] = color_source_imgs_list  # the adjacent frames

        inv_K = inputs['inv_K' + suffix][:, self.render_view_indices]
        K = inputs['K' + suffix][:, self.render_view_indices]
        inv_K = rearrange(inv_K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')
        K = rearrange(K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')

        # rescale the rendered depth
        depth_list = []
        for tmp in ['last', 'future']:
            _depth = outputs[f'render_depth_{tmp}'][:, self.render_view_indices]
            _depth = rearrange(_depth, 'b num_view h w -> (b num_view) () h w')
            _depth = F.interpolate(
                _depth, self.depth_ssl_size, mode="bilinear", align_corners=False)
            depth_list.append(_depth)
        outputs['render_depth_rescaled' + suffix] = depth_list

        cam_T_cam = inputs["cam_T_cam" + suffix][:, :, self.render_view_indices]

        ## 1) Depth to camera points
        len_temporal = cam_T_cam.shape[1]
        color_reprojection_list = []
        for frame_id in range(len_temporal):
            T = cam_T_cam[:, frame_id]
            T = rearrange(T, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')

            cam_points = self.backproject_depth(depth_list[frame_id], inv_K)  # (M, 4, h*w)
            ## 2) Camera points to adjacent image points
            pix_coords = self.project_3d(cam_points, K, T)  # (M, h, w, 2)

            ## 3) Reproject the adjacent image
            color_source = inputs['color_source_imgs' + suffix][frame_id]  # (M, 3, h, w)
            color_reprojection = F.grid_sample(
                color_source,
                pix_coords,
                padding_mode="border", align_corners=True)
            color_reprojection_list.append(color_reprojection)

        outputs['color_reprojection' + suffix] = color_reprojection_list
        outputs['target_imgs' + suffix] = rearrange(
            inputs['target_imgs' + suffix], 'b num_view c h w -> (b num_view) c h w')

    def compute_self_supervised_losses(self, inputs, outputs, suffix=''):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        loss = 0

        color = outputs["target_imgs" + suffix]
        target = outputs["target_imgs" + suffix]

        reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection' + suffix])):
            pred = outputs['color_reprojection' + suffix][frame_id]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)  # (M, 2, h, w)

        ## automasking
        identity_reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection' + suffix])):
            pred = inputs["color_source_imgs" + suffix][frame_id]
            identity_reprojection_losses.append(
                self.compute_reprojection_loss(pred, target))

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        if self.opt.avg_reprojection:
            identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
        else:
            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not self.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        loss += to_optimise.mean()

        smooth_loss = 0
        num_depth_maps = len(outputs["render_depth_rescaled" + suffix])
        for i in range(num_depth_maps):
            depth = outputs["render_depth_rescaled" + suffix][i]  # (M, 1, h, w)
            disp = 1.0 / (depth + 1e-7)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            _smooth_loss = get_smooth_loss(norm_disp, color)
            smooth_loss += _smooth_loss

        loss += self.opt.disparity_smoothness * (smooth_loss / num_depth_maps)
        
        total_loss += loss
        losses["loss_depth_ct" + suffix] = self.depth_loss_weight * total_loss  # depth consistency loss
        return losses
    
@HEADS.register_module()
class PretrainHeadV2(BaseModule):
    def __init__(
        self,
        in_channels=128,
        view_cfg=None,
        uni_conv_cfg=None,
        render_head_cfg=None,
        render_scale=(1, 1),
        use_semantic=False,
        vis_gt=False,
        vis_pred=False,
        render_view_indices=list(range(6)),
        depth_ssl_size=None,
        depth_loss_weight=1.0,
        rgb_loss_weight=1.0,
        use_depth_gt_loss=False,
        use_semantic_gt_loss=False,
        depth_gt_loss_weight=1.0,
        opt=None,
        save_dir=None,
        use_flow_ssl=False,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels

        self.use_semantic = use_semantic
        self.vis_gt = vis_gt
        self.vis_pred = vis_pred

        self.steps = -1

        self.render_opt = render_head_cfg.render_opt

        ## use the depth self-supervised consistency loss
        self.use_depth_consistency = self.render_opt.use_depth_consistency
        self.render_view_indices = render_view_indices
        self.depth_ssl_size = depth_ssl_size
        self.opt = opt  # options for the depth consistency loss
        self.depth_loss_weight = depth_loss_weight

        self.rgb_loss_weight = rgb_loss_weight
        
        self.use_depth_gt_loss = use_depth_gt_loss
        self.depth_gt_loss_weight = depth_gt_loss_weight

        self.use_semantic_gt_loss = use_semantic_gt_loss

        self.save_dir = save_dir

        if self.use_depth_consistency:
            h = depth_ssl_size[0]
            w = depth_ssl_size[1]
            num_cam = len(self.render_view_indices)
            self.backproject_depth = BackprojectDepth(num_cam, h, w)
            self.project_3d = Project3D(num_cam, h, w)

            self.ssim = SSIM()

        if view_cfg is not None:
            vtrans_type = view_cfg.pop('type', 'Uni3DViewTrans')
            self.view_trans = getattr(utils, vtrans_type)(**view_cfg)

        if uni_conv_cfg is not None:
            self.uni_conv = nn.Sequential(
                nn.Conv3d(
                    uni_conv_cfg["in_channels"],
                    uni_conv_cfg["out_channels"],
                    kernel_size=uni_conv_cfg["kernel_size"],
                    padding=uni_conv_cfg["padding"],
                    stride=1,
                ),
                nn.BatchNorm3d(uni_conv_cfg["out_channels"]),
                nn.ReLU(inplace=True),
            )

        if render_head_cfg is not None:
            self.render_head = builder.build_head(render_head_cfg)

        self.render_scale = render_scale
        
    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, 
                pts_feats, 
                img_feats, 
                img_metas, 
                img_depth,
                img_inputs,
                **kwargs):

        output = dict()

        # Prepare the projection parameters
        lidar2cam, intrinsics = [], []
        for img_meta in img_metas:
            lidar2cam.append(img_meta["lidar2cam"])
            intrinsics.append(img_meta["cam_intrinsic"])
        lidar2cam = np.asarray(lidar2cam)  # (bs, 6, 1, 4, 4)
        intrinsics = np.asarray(intrinsics)

        ref_tensor = img_feats[0].float()

        intrinsics = ref_tensor.new_tensor(intrinsics)
        pose_spatial = torch.inverse(
            ref_tensor.new_tensor(lidar2cam)
        )

        output['pose_spatial'] = pose_spatial[:, :, 0]
        output['intrinsics'] = intrinsics[:, :, 0]  # (bs, 6, 4, 4)
        output['intrinsics'][:, :, 0] *= self.render_scale[1]
        output['intrinsics'][:, :, 1] *= self.render_scale[0]

        if self.vis_gt:
            ## NOTE: due to the occ gt is labelled in the ego coordinate, we need to
            # use the cam2ego matrix as ego matrix
            cam2camego = []
            for img_meta in img_metas:
                cam2camego.append(img_meta["cam2camego"])
            cam2camego = np.asarray(cam2camego)  # (bs, 6, 1, 4, 4)
            output['pose_spatial'] = ref_tensor.new_tensor(cam2camego)

            gt_data_dict = self.prepare_gt_data(**kwargs)
            output.update(gt_data_dict)
            render_results = self.render_head(output, vis_gt=True)
            
            ## visualiza the results
            render_depth, rgb_pred, semantic_pred = render_results
            # current_frame_img = torch.zeros_like(rgb_pred).cpu().numpy()
            current_frame_img = img_inputs.cpu().numpy()
            visualize_image_semantic_depth_pair(
                current_frame_img[0],
                rgb_pred[0].permute(0, 2, 3, 1),
                render_depth[0],
                save_dir="results/vis/3dgs_baseline_gt"
            )
            exit()

        ## 1. Prepare the volume feature from the pts features and img features
        uni_feats = []
        if img_feats is not None:
            uni_feats.append(
                self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
            )
        if pts_feats is not None:
            uni_feats.append(pts_feats)

        uni_feats = sum(uni_feats)
        uni_feats = self.uni_conv(uni_feats)  # (bs, c, z, y, x)

        ## 2. Prepare the features for rendering
        _uni_feats = rearrange(uni_feats, 'b c z y x -> b x y z c')

        output['volume_feat'] = _uni_feats


        ## 2. Start rendering, including neural rendering or 3DGS
        render_results = self.render_head(output)


        ## Visualize the results
        if self.vis_pred:
            from .nerf_utils import VisElement, visualize_elements

            save_dir = self.save_dir
            os.makedirs(save_dir, exist_ok=True)

            ## save the occupancy offline for visualization
            # torch.save(semantic_output.detach().cpu(), f'{save_dir}/semantic_pred.pth')
            # torch.save(occupancy_output.detach().cpu(), f'{save_dir}/occupancy_pred.pth')

            render_depth = render_results['render_depth']
            rgb_pred = render_results['render_rgb']
            semantic_pred = render_results['render_semantic']

            render_gt_semantic = kwargs.get('render_gt_semantic', None)
            render_gt_depth = kwargs.get('render_gt_depth', None)

            semantic = semantic_pred.argmax(2)
            semantic = OCC3D_PALETTE[semantic].to(semantic_pred)
            # visualize_image_pairs(
            #     img_inputs[0],
            #     semantic[0], # rgb_pred[0].permute(0, 2, 3, 1),
            #     render_depth[0],
            #     semantic_is_sparse=False,
            #     depth_is_sparse=False,
            #     save_dir=save_dir
            # )

            target_size = (semantic.shape[2], semantic.shape[3])  # (H, W)
            target_size = (180, 320)
            visualize_elements(
                [
                    # VisElement(
                    #     img_inputs[0],
                    #     type='rgb'
                    # ),
                    # VisElement(
                    #     rgb_pred[0],
                    #     type='rgb',
                    #     need_denormalize=False,
                    # ),
                    VisElement(
                        render_depth[0],
                        type='depth',
                        is_sparse=False,
                    ),
                    # VisElement(
                    #     semantic[0],
                    #     type='semantic',
                    #     is_sparse=False,
                    # ),
                ],
                target_size=target_size,
                save_dir=save_dir
            )
            exit()
        return render_results
    
    def loss(self, preds_dict, targets):
        if self.use_depth_consistency:
            ## Visualize the input data
            DEBUG = False
            if DEBUG:
                from .nerf_utils import VisElement, visualize_elements
                save_dir = "results/vis/3dgs_depth_ssl"
                os.makedirs(save_dir, exist_ok=True)

                source_imgs = targets['source_imgs'][0]  # to (2, N, 3, h, w)
                target_size = (source_imgs.shape[-2], source_imgs.shape[-1])  # (H, W)

                curr_imgs = targets['target_imgs'][0]  # to (N, 3, h, w)
                visualize_elements(
                    [
                        VisElement(
                            source_imgs[0],
                            type='rgb'
                        ),
                        VisElement(
                            curr_imgs,
                            type='rgb',
                        ),
                        VisElement(
                            source_imgs[1],
                            type='rgb',
                        ),
                    ],
                    target_size=target_size,
                    save_dir=save_dir
                )
                exit()
            
            preds_dict['render_depth_raw'] = preds_dict['render_depth'].clone()

            loss_dict = {}

            if self.depth_loss_weight > 0.0:
                ## 1) Compute the reprojected rgb images based on the rendered depth
                self.generate_image_pred(targets, preds_dict)

                ## 2) Compute the depth consistency loss
                loss_depth_ssl = self.compute_self_supervised_losses(targets, preds_dict)
                loss_dict.update(loss_depth_ssl)
            else:
                preds_dict['render_depth'] = rearrange(
                    preds_dict['render_depth'], 'b num_view h w -> (b num_view) () h w')

            ## 3) Compute the RGB reconstruction loss
            if self.rgb_loss_weight > 0.0:
                render_rgb = preds_dict['render_rgb']  # (bs, num_cam, 3, h, w)
                target_img = targets['target_imgs']  # (bs, num_cam, 3, h, w)
                delta = render_rgb - target_img
                rgb_loss = self.rgb_loss_weight * (delta**2).mean()
                loss_rgb = {'loss_rgb' : rgb_loss}
                loss_dict.update(loss_rgb)

            ## 4) Compute the depth gt loss
            if self.use_depth_gt_loss:
                render_depth = preds_dict['render_depth_raw']
                gt_depth = targets['render_gt_depth']

                mask = gt_depth > 0.0
                loss_render_depth = F.l1_loss(render_depth[mask], gt_depth[mask])
                if torch.isnan(loss_render_depth):
                    print('NaN in render depth loss!')
                    loss_render_depth = torch.Tensor([0.0]).to(render_depth.device)
                loss_dict['loss_render_depth'] = self.depth_gt_loss_weight * loss_render_depth

            if self.use_semantic_gt_loss:
                assert 'render_gt_semantic' in targets.keys()

                semantic_gt = targets['render_gt_semantic']
                semantic_pred = preds_dict['render_semantic']
                
                loss_render_sem = self.compute_semantic_loss(
                    semantic_pred, semantic_gt, ignore_index=255)
                if torch.isnan(loss_render_sem):
                    print('NaN in render semantic loss!')
                    loss_render_sem = torch.Tensor([0.0]).to(preds_dict['render_depth'].device)
                loss_dict['loss_render_sem'] = 0.1 * loss_render_sem
        else:
            loss_dict = self.render_head.loss(preds_dict, targets)

            if self.save_dir is not None:
                rank, _ = get_dist_info()
                
                if rank == 0:
                    self.steps += 1
                    if self.steps % 200 == 0:
                        from .nerf_utils import VisElement, visualize_elements

                        save_dir = self.save_dir

                        render_depth = preds_dict['render_depth'][0]  # (M, h, w)
                        render_gt_depth = targets['render_gt_depth'][0]

                        print(f"Render depth: min={render_depth.min().item()}, max={render_depth.max().item()}")

                        target_size = (render_depth.shape[-2], render_depth.shape[-1])  # (H, W)
                        visualize_elements(
                            [
                                VisElement(
                                    render_gt_depth,
                                    type='depth',
                                    is_sparse=True,
                                ),
                                VisElement(
                                    render_depth,
                                    type='depth',
                                    is_sparse=False,
                                ),
                            ],
                            target_size=target_size,
                            save_dir=save_dir,
                            prefix=f"{self.steps:05d}"
                        )

            # if self.use_flow_ssl:
            #     # compute the loss for the future frame
            #     future_loss_dict = self.render_head.loss(
            #         preds_dict, targets, suffix='_future', weight=0.15)
                
            #     loss_dict.update(future_loss_dict)

        ## Visualization
        if self.use_depth_consistency and self.save_dir is not None:
            rank, _ = get_dist_info()
            
            if rank == 0:
                self.steps += 1
                if self.steps % 200 == 0:
                    from .nerf_utils import VisElement, visualize_elements

                    save_dir = self.save_dir

                    render_depth = preds_dict['render_depth']  # (M, 1, h, w)

                    print(f"Render depth: min={render_depth.min().item()}, max={render_depth.max().item()}")

                    render_rgb = preds_dict['render_rgb']  # (M, num_cam, 3, h, w)

                    prev_img = targets['source_imgs'][:, 0]  # (1, num_cam, 3, h, w)
                    next_img = targets['source_imgs'][:, 1]  # (1, num_cam, 3, h, w)
                    target_img = targets['target_imgs']  # (1, num_cam, 3, h, w)

                    target_size = (render_depth.shape[2], render_depth.shape[3])  # (H, W)
                    visualize_elements(
                        [
                            VisElement(
                                prev_img[0],
                                type='rgb'
                            ),
                            VisElement(
                                target_img[0],
                                type='rgb'
                            ),
                            VisElement(
                                next_img[0],
                                type='rgb',
                            ),
                            VisElement(
                                render_depth[:, 0],
                                type='depth',
                                is_sparse=False,
                            ),
                            VisElement(
                                render_rgb[0],
                                type='rgb',
                            ),
                        ],
                        target_size=target_size,
                        save_dir=save_dir,
                        prefix=f"{self.steps:05d}"
                    )

        return loss_dict

    def compute_semantic_loss(self, sem_est, sem_gt, ignore_index=-100):
        '''
        Args:
            sem_est: B, N, C, H, W, predicted unnormalized logits
            sem_gt: B, N, H, W
        '''
        B, N, C, H, W = sem_est.shape
        sem_est = sem_est.view(B * N, -1, H, W)
        sem_gt = sem_gt.view(B * N, H, W)
        loss = F.cross_entropy(sem_est, sem_gt.long(), ignore_index=ignore_index)

        return loss
    
    def compute_self_supervised_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        loss = 0

        depth = outputs["render_depth"]  # (M, 1, h, w)
        disp = 1.0 / (depth + 1e-7)
        color = outputs["target_imgs"]
        target = outputs["target_imgs"]

        reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection'])):
            pred = outputs['color_reprojection'][frame_id]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)  # (M, 2, h, w)

        ## automasking
        identity_reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection'])):
            pred = inputs["color_source_imgs"][frame_id]
            identity_reprojection_losses.append(
                self.compute_reprojection_loss(pred, target))

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        if self.opt.avg_reprojection:
            identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
        else:
            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not self.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        loss += self.opt.disparity_smoothness * smooth_loss
        
        total_loss += loss
        losses["loss_depth_ct"] = self.depth_loss_weight * total_loss  # depth consistency loss
        return losses
    
    def generate_image_pred(self, inputs, outputs):
        color_source_imgs_list = []
        for idx in range(inputs['source_imgs'].shape[1]):
            color_source = inputs['source_imgs'][:, idx]
            color_source = rearrange(color_source, 'b num_view c h w -> (b num_view) c h w')
            color_source_imgs_list.append(color_source)
        inputs['color_source_imgs'] = color_source_imgs_list

        inv_K = inputs['inv_K'][:, self.render_view_indices]
        K = inputs['K'][:, self.render_view_indices]
        inv_K = rearrange(inv_K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')
        K = rearrange(K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')

        # rescale the rendered depth
        depth = outputs['render_depth'][:, self.render_view_indices]
        depth = rearrange(depth, 'b num_view h w -> (b num_view) () h w')
        depth = F.interpolate(
            depth, self.depth_ssl_size, mode="bilinear", align_corners=False)
        outputs['render_depth'] = depth

        cam_T_cam = inputs["cam_T_cam"][:, :, self.render_view_indices]

        ## 1) Depth to camera points
        cam_points = self.backproject_depth(depth, inv_K)  # (M, 4, h*w)
        len_temporal = cam_T_cam.shape[1]
        color_reprojection_list = []
        for frame_id in range(len_temporal):
            T = cam_T_cam[:, frame_id]
            T = rearrange(T, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')
            ## 2) Camera points to adjacent image points
            pix_coords = self.project_3d(cam_points, K, T)  # (M, h, w, 2)

            ## 3) Reproject the adjacent image
            color_source = inputs['color_source_imgs'][frame_id]  # (M, 3, h, w)
            color_reprojection = F.grid_sample(
                color_source,
                pix_coords,
                padding_mode="border", align_corners=True)
            color_reprojection_list.append(color_reprojection)

        outputs['color_reprojection'] = color_reprojection_list
        outputs['target_imgs'] = rearrange(
            inputs['target_imgs'], 'b num_view c h w -> (b num_view) c h w')

    def compute_reprojection_loss(self, pred, target, no_ssim=False):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    def prepare_gt_data(self, **kwargs):
        # Prepare the ground truth volume data for visualization
        voxel_semantics = kwargs['voxel_semantics']
        density_prob = rearrange(voxel_semantics, 'b x y z -> b () x y z')
        density_prob = density_prob != 17
        density_prob = density_prob.float()
        density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
        density_prob[density_prob == 1] = 10

        output = dict()
        output['density_prob'] = density_prob

        semantic = OCC3D_PALETTE[voxel_semantics.long()].to(density_prob)
        semantic = semantic.permute(0, 4, 1, 2, 3)  # to (b, 3, 200, 200, 16)
        output['semantic'] = semantic
        return output
        