import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16, BaseModule
from mmdet.models.utils import build_transformer
from mmdet.core import multi_apply, multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet.core import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
    build_bbox_coder,
)
from mmcv.cnn import xavier_init, constant_init
from .. import utils
from projects.mmdet3d_plugin.core.bbox.iou_calculators import PairedBboxOverlaps3D

from projects.mmdet3d_plugin.models.dense_heads.uvtr_dn_head import UVTRDNHead


@HEADS.register_module()
class UVTRDNHead_MAP(UVTRDNHead):
    """Head of UVTR.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super(UVTRDNHead_MAP, self).__init__(**kwargs)


    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(
        self,
        pts_feats,
        img_feats,
        img_metas,
        img_depth,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        with_image, with_point = True, True
        if img_feats is None:
            with_image = False
        elif isinstance(img_feats, dict) and img_feats["key"] is None:
            with_image = False

        if pts_feats is None:
            with_point = False
        elif isinstance(pts_feats, dict) and pts_feats["key"] is None:
            with_point = False
            pts_feats = None

        # transfer to voxel level
        if with_image:
            img_feats = self.view_trans(
                img_feats, img_metas=img_metas, img_depth=img_depth
            )
        # shape: (N, L, C, D, H, W)
        if with_point:
            if len(pts_feats.shape) == 5:
                pts_feats = pts_feats.unsqueeze(1)

        if self.unified_conv is not None:
            raw_shape = pts_feats.shape
            if self.unified_conv["fusion"] == "sum":
                unified_feats = pts_feats.flatten(1, 2) + img_feats.flatten(1, 2)
            else:
                unified_feats = torch.cat(
                    [pts_feats.flatten(1, 2), img_feats.flatten(1, 2)], dim=1
                )
            for layer in self.conv_layer:
                unified_feats = layer(unified_feats)
            unified_feats = unified_feats.reshape(*raw_shape)
        else:
            unified_feats = pts_feats if pts_feats is not None else img_feats

        unified_feats = unified_feats.squeeze(1)  # (B, C, Z, Y, X)

        return unified_feats

        # bs = unified_feats.shape[0]

        # reference_points = self.reference_points.weight
        # reference_points, attn_mask, mask_dict = self.prepare_for_dn(
        #     bs, reference_points, gt_bboxes_3d, gt_labels_3d
        # )

        # reference_points = inverse_sigmoid(reference_points)
        # query_pos = self.query_embedding(reference_points)
        # hs, inter_references = self.transformer(
        #     query=torch.zeros_like(query_pos).permute(1, 0, 2),
        #     value=unified_feats,
        #     query_pos=query_pos.permute(1, 0, 2),
        #     key_pos=None,
        #     reference_points=reference_points,
        #     reg_branches=(
        #         self.reg_branches if self.with_box_refine else None
        #     ),  # noqa:E501
        #     attn_masks=[attn_mask, None],
        # )
        # hs = hs.permute(0, 2, 1, 3)  # (L, N, B, C) -> (L, B, N, C)

        # outputs_classes, outputs_coords, outputs_ious = [], [], []
        # for lvl in range(hs.shape[0]):
        #     # only backward for init_reference_points
        #     reference = (
        #         reference_points if lvl == 0 else inter_references[lvl - 1]
        #     )  # (B, N, 3)

        #     outputs_class = self.cls_branches[lvl](hs[lvl])
        #     outputs_iou = self.iou_branches[lvl](hs[lvl])
        #     outputs_coord = self.reg_branches[lvl](hs[lvl])

        #     # TODO: check the shape of reference
        #     assert reference.shape[-1] == 3
        #     outputs_coord[..., 0:2] = (
        #         outputs_coord[..., 0:2] + reference[..., 0:2]
        #     ).sigmoid()
        #     outputs_coord[..., 4:5] = (
        #         outputs_coord[..., 4:5] + reference[..., 2:3]
        #     ).sigmoid()

        #     # transfer to lidar system
        #     outputs_coord[..., 0:1] = (
        #         outputs_coord[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
        #         + self.pc_range[0]
        #     )
        #     outputs_coord[..., 1:2] = (
        #         outputs_coord[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
        #         + self.pc_range[1]
        #     )
        #     outputs_coord[..., 4:5] = (
        #         outputs_coord[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
        #         + self.pc_range[2]
        #     )

        #     # TODO: check if using sigmoid
        #     outputs_classes.append(outputs_class)
        #     outputs_ious.append(outputs_iou)
        #     outputs_coords.append(outputs_coord)

        # outputs_classes = torch.stack(outputs_classes)  # (L, B, N, num_class)
        # outputs_ious = torch.stack(outputs_ious)
        # outputs_coords = torch.stack(outputs_coords)

        # outs = {
        #     "all_cls_scores": outputs_classes,
        #     "all_iou_preds": outputs_ious,
        #     "all_bbox_preds": outputs_coords,
        # }

        # if mask_dict is not None and mask_dict["pad_size"] > 0:
        #     for key in list(outs.keys()):
        #         outs["dn_" + key] = outs[key][:, :, : mask_dict["pad_size"], :]
        #         outs[key] = outs[key][:, :, mask_dict["pad_size"] :, :]
        #     outs["dn_mask_dicts"] = mask_dict

        # return outs
