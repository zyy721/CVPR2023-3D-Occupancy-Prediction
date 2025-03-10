# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaoyu Tian
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32


from mmengine.registry import MODELS
from projects.mmdet3d_plugin.OccWorld.modules.vae_2d_resnet import CustomVAERes2D
from einops import rearrange  
from projects.mmdet3d_plugin.OccWorld.loss import OPENOCC_LOSS
from copy import deepcopy


@HEADS.register_module()
class CustomBEVFormerOccHead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 bev_h=30,
                 bev_w=30,
                 loss_occ=None,
                 use_mask=False,
                 positional_encoding=None,

                 occ_vae=None,
                 multi_loss=None,
                 loss_input_convertion=None,

                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.num_classes=kwargs['num_classes']
        self.use_mask=use_mask

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage


        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        super(CustomBEVFormerOccHead, self).__init__()

        self.loss_occ = build_loss(loss_occ)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims


        self.occ_vae = MODELS.build(occ_vae)
        self.multi_loss = OPENOCC_LOSS.build(multi_loss)
        self.loss_input_convertion = loss_input_convertion


        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        # if self.loss_cls.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     for m in self.cls_branches:
        #         nn.init.constant_(m[-1].bias, bias_init)

        self.occ_vae.init_weights()

    # @auto_fp16(apply_to=('mlvl_feats'))
    @auto_fp16(apply_to=('mlvl_feats', 'voxel_semantics'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, voxel_semantics=None, only_bev=False, test=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        output_dict = {}

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = None
        # bev_queries = self.bev_embedding.weight.to(dtype)

        voxel_semantics = voxel_semantics.unsqueeze(0)
        h, temb = self.occ_vae.forward_encoder_as_query(voxel_semantics.to(torch.long))
        bev_queries = h.to(dtype)
        bev_queries = rearrange(bev_queries, 'b c h w -> (h w) b c')

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=None,  # noqa:E501
                cls_branches=None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )
        # bev_embed, occ_outs = outputs
        bev_embed = outputs
        bev_embed = rearrange(bev_embed, 'b (h w) c -> b c h w', h=self.bev_h, w=self.bev_w)
        z, shapes = self.occ_vae.forward_encoder_downsample(bev_embed, temb)

        z_sampled, z_mu, z_sigma, logvar = self.occ_vae.sample_z(z)

        output_dict.update({
            'z_mu': z_mu,
            'z_sigma': z_sigma,
            'logvar': logvar})

        logits = self.occ_vae.forward_decoder(z_sampled, shapes, voxel_semantics.shape)
    
        output_dict.update({'logits': logits})

        if test:
            pred = logits.argmax(dim=-1).detach().cuda()
            output_dict['sem_pred'] = pred
            pred_iou = deepcopy(pred)
            
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou


        # outs = {
        #     'bev_embed': bev_embed,
        #     'occ':occ_outs,
        # }

        # return outs

        return output_dict


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             # gt_bboxes_list,
             # gt_labels_list,
             voxel_semantics,
             mask_camera,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        # loss_dict=dict()
        # occ=preds_dicts['occ']
        # assert voxel_semantics.min()>=0 and voxel_semantics.max()<=17
        # losses = self.loss_single(voxel_semantics,mask_camera,occ)
        # loss_dict['loss_occ']=losses
        # return loss_dict
    

        input_occs = voxel_semantics.unsqueeze(1).to(torch.long)
        target_occs = voxel_semantics.unsqueeze(1).to(torch.long)

        loss_input = {
            'inputs': input_occs,
            'target_occs': target_occs,
            # 'metas': metas
        }
        
        for loss_input_key, loss_input_val in self.loss_input_convertion.items():
            loss_input.update({
                loss_input_key: preds_dicts[loss_input_val]})
        loss, loss_dict = self.multi_loss(loss_input)

        return loss_dict


    def loss_single(self,voxel_semantics,mask_camera,preds):
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
        return loss_occ

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # return self.transformer.get_occ(
        #     preds_dicts, img_metas, rescale=rescale)
        # print(img_metas[0].keys())
        occ_out=preds_dicts['occ']
        occ_score=occ_out.softmax(-1)
        occ_score=occ_score.argmax(-1)


        return occ_score
