""" adopted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py """
# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import numpy as np
from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch.nn.functional as F
from copy import deepcopy

from typing import Optional, Tuple, Union

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

# def Normalize(in_channels):
#     if in_channels <= 32:
#         num_groups = in_channels // 4
#     else:
#         num_groups = 32
#     return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def Normalize(in_channels, eps=1e-6):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
    
    def forward(self, x, shape):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        diffY = shape[0] - x.size()[2]
        diffX = shape[1] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])

        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
    
    def forward(self, x):
        if self.with_conv:
            #pad = (0, 1, 0, 1, 0, 1)
            #x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x
    


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class TemporalResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=None, temb_channels=512, eps: float = 1e-6,):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, eps=eps)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels, eps=eps)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AlphaBlender(nn.Module):
    r"""
    A module to blend spatial and temporal features.

    Parameters:
        alpha (`float`): The initial value of the blending factor.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix  # For TemporalVAE

        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor, ndims: int) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor

        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)

        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                raise ValueError("Please provide image_only_indicator to use learned_with_images merge strategy")

            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                torch.sigmoid(self.mix_factor)[..., None],
            )

            # (batch, channel, frames, height, width)
            if ndims == 5:
                alpha = alpha[:, None, :, None, None]
            # (batch*frames, height*width, channels)
            elif ndims == 3:
                alpha = alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(f"Unexpected ndims {ndims}. Dimensions should be 3 or 5")

        else:
            raise NotImplementedError

        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
        alpha = alpha.to(x_spatial.dtype)

        if self.switch_spatial_to_temporal_mix:
            alpha = 1.0 - alpha

        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x



class SpatioTemporalResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512,
                 temporal_eps: Optional[float] = None,
                 merge_factor: float = 0.5,
                 merge_strategy="learned_with_images",
                 switch_spatial_to_temporal_mix: bool = False,
                 ):
        super().__init__()

        self.spatial_res_block = ResnetBlock(in_channels=in_channels,
                                             out_channels=out_channels,
                                             temb_channels=temb_channels,
                                             dropout=dropout)

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            dropout=dropout,
            temb_channels=temb_channels,
            eps=temporal_eps,
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

    def forward(self, x, temb=None, image_only_indicator: Optional[torch.Tensor] = None,):
        hidden_states = x
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class SpatioTemporalAttnBlock(nn.Module):
    def __init__(self, block_in):
        super().__init__()

        self.SpatioAttnBlock = AttnBlock(block_in)
        self.TemporalAttnBlock = AttnBlock(block_in)
    
    def forward(self, x):
        T, C, H, W = x.shape
        x = self.SpatioAttnBlock(x)
        x = x.permute(2, 3, 1, 0)
        x = x.contiguous().view(-1, C, T, 1)
        x = self.TemporalAttnBlock(x)
        x = x.view(H, W, C, T)
        x = x.permute(3, 2, 0, 1)

        return x

    
@MODELS.register_module()
class VAERes2D(BaseModule):
    def __init__(
            self, 
            encoder_cfg, 
            decoder_cfg,
            num_classes=18,
            expansion=8, 
            vqvae_cfg=None,
            init_cfg=None):
        super().__init__(init_cfg)

        self.expansion = expansion
        self.num_cls = num_classes

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)
        self.class_embeds = nn.Embedding(num_classes, expansion)

        if vqvae_cfg:
            self.vqvae = MODELS.build(vqvae_cfg)
        self.use_vq = vqvae_cfg is not None
    
    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c
        x = x.reshape(bs*F, H, W, D * self.expansion).permute(0, 3, 1, 2)

        z, shapes = self.encoder(x)
        return z, shapes
        
    def forward_decoder(self, z, shapes, input_shape):
        logits = self.decoder(z, shapes)

        bs, F, H, W, D = input_shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
        template = self.class_embeds.weight.T.unsqueeze(0) # 1, expansion, cls
        similarity = torch.matmul(logits, template) # -1, D, cls
        # pred = similarity.argmax(dim=-1) # -1, D
        # pred = pred.reshape(bs, F, H, W, D)
        return similarity.reshape(bs, F, H, W, D, self.num_cls)

    def forward(self, x, **kwargs):
        # xs = self.forward_encoder(x)
        # logits = self.forward_decoder(xs)
        # return logits, xs[-1]
        
        output_dict = {}
        z, shapes = self.forward_encoder(x)
        if self.use_vq:
            z_sampled, loss, info = self.vqvae(z, is_voxel=False)
            output_dict.update({'embed_loss': loss})
        else:
            z_sampled, z_mu, z_sigma = self.sample_z(z)
            output_dict.update({
                'z_mu': z_mu,
                'z_sigma': z_sigma})
        
        logits = self.forward_decoder(z_sampled, shapes, x.shape)
        
        output_dict.update({'logits': logits})
    
        if not self.training:
            pred = logits.argmax(dim=-1).detach().cuda()
            output_dict['sem_pred'] = pred
            pred_iou = deepcopy(pred)
            
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou
            
        return output_dict
        # loss, kl, rec = self.loss(logits, x, z_mu, z_sigma)
        # return loss, kl, rec
        
    def generate(self, z, shapes, input_shape):
        logits = self.forward_decoder(z, shapes, input_shape)
        return {'logits': logits}



@MODELS.register_module()
class Encoder2D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Enc has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        # x: bs, F, H, W, D
        shapes = []
        temb = None
                                                                                                                                                                             
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions-1:
                shapes.append(h.shape[-2:])
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        #
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, shapes

@MODELS.register_module()
class Decoder2D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align with encoder
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Dec has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, shapes):
        # z: bs*F, C, H, W
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align encoder
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, shapes.pop())

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h




@MODELS.register_module()
class CustomEncoder2D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Enc has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        self.proj_token_to_query = torch.nn.Conv2d(ch*ch_mult[0],
                                       256,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.proj_query_to_token = torch.nn.Conv2d(256,
                                       ch*ch_mult[0],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)


    def encode_as_query(self, x):
        # x: bs, F, H, W, D
        # shapes = []
        temb = None

        h = self.conv_in(x)

        i_level = 0

        for i_block in range(self.num_res_blocks):
            # h = self.down[i_level].block[i_block](hs[-1], temb)
            h = self.down[i_level].block[i_block](h, temb)

            if len(self.down[i_level].attn) > 0:
                h = self.down[i_level].attn[i_block](h)
            # hs.append(h)

        h = self.proj_token_to_query(h)

        return h, temb


    def encode_downsample(self, h, temb):
        h = self.proj_query_to_token(h)

        shapes = []

        i_level = 0
        if i_level != self.num_resolutions-1:
            shapes.append(h.shape[-2:])
            # hs.append(self.down[i_level].downsample(hs[-1]))
            h = self.down[i_level].downsample(h)

        for i_level in range(1, self.num_resolutions):
            
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions-1:
                shapes.append(h.shape[-2:])
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        #
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, shapes      



    def forward(self, x):
        # x: bs, F, H, W, D
        shapes = []
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions-1:
                shapes.append(h.shape[-2:])
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        #
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, shapes


@MODELS.register_module()
class CustomDecoder2D(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align with encoder
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Dec has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, shapes, image_only_indicator=None):
        # z: bs*F, C, H, W
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align encoder
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, shapes.pop())

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


@MODELS.register_module()
class CustomTemporalDecoder(BaseModule):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        # self.mid = nn.Module()
        # self.mid.block_1 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        # self.mid.block_2 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)

        self.mid = nn.Module()
        self.mid.block_1 = SpatioTemporalResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       temporal_eps=1e-5,
                                       merge_factor=0.0,
                                       merge_strategy="learned",
                                       switch_spatial_to_temporal_mix=True,
                                       )
        self.mid.attn_1 = SpatioTemporalAttnBlock(block_in)
        self.mid.block_2 = SpatioTemporalResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       temporal_eps=1e-5,
                                       merge_factor=0.0,
                                       merge_strategy="learned",
                                       switch_spatial_to_temporal_mix=True,
                                       )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align with encoder
                block.append(SpatioTemporalResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         temporal_eps=1e-5,
                                         merge_factor=0.0,
                                         merge_strategy="learned",
                                         switch_spatial_to_temporal_mix=True,
                                         ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Dec has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(SpatioTemporalAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, shapes, image_only_indicator: Optional[torch.Tensor] = None):
        # z: bs*F, C, H, W
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, image_only_indicator)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, image_only_indicator)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align encoder
                h = self.up[i_level].block[i_block](h, temb, image_only_indicator)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, shapes.pop())

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


    # def forward_give_pre_end(self, z, shapes, input_shape):
    #     image_only_indicator = torch.zeros(input_shape[0], input_shape[1], dtype=z.dtype, device=z.device)

    #     # z: bs*F, C, H, W
    #     self.last_z_shape = z.shape

    #     # timestep embedding
    #     temb = None

    #     # z to block_in
    #     h = self.conv_in(z)

    #     # middle
    #     h = self.mid.block_1(h, temb, image_only_indicator)
    #     h = self.mid.attn_1(h)
    #     h = self.mid.block_2(h, temb, image_only_indicator)

    #     # upsampling
    #     for i_level in reversed(range(self.num_resolutions)):
    #         # for i_block in range(self.num_res_blocks+1):
    #         for i_block in range(self.num_res_blocks): # change this to align encoder
    #             h = self.up[i_level].block[i_block](h, temb, image_only_indicator)
    #             if len(self.up[i_level].attn) > 0:
    #                 h = self.up[i_level].attn[i_block](h)
    #         if i_level != 0:
    #             h = self.up[i_level].upsample(h, shapes.pop())

    #     return h
    

    # def forward_give_after_end(self, h):
    #     h = self.norm_out(h)
    #     h = nonlinearity(h)
    #     h = self.conv_out(h)
    #     return h



@MODELS.register_module()
class CustomVAERes2D(BaseModule):
    def __init__(
            self, 
            encoder_cfg, 
            decoder_cfg,
            num_classes=18,
            expansion=8, 
            vqvae_cfg=None,
            init_cfg=None):
        super().__init__(init_cfg)

        self.expansion = expansion
        self.num_cls = num_classes

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)
        self.class_embeds = nn.Embedding(num_classes, expansion)

        if vqvae_cfg:
            self.vqvae = MODELS.build(vqvae_cfg)
        self.use_vq = vqvae_cfg is not None

        if not self.use_vq:
            self.quant_conv = nn.Conv2d(encoder_cfg['z_channels'] * 2, encoder_cfg['z_channels'] * 2, 1)
            # self.quant_conv = nn.Conv2d(encoder_cfg['z_channels'], encoder_cfg['z_channels'], 1)

    def forward_encoder_as_query(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c
        x = x.reshape(bs*F, H, W, D * self.expansion).permute(0, 3, 1, 2)

        h, temb = self.encoder.encode_as_query(x)

        return h, temb


    def forward_encoder_downsample(self, h, temb):
        z, shapes = self.encoder.encode_downsample(h, temb)
        z = self.quant_conv(z)
        return z, shapes


    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)

        logvar = z[:, dim:]

        return mu + sigma * eps, mu, sigma, logvar

    def forward_encoder(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c
        x = x.reshape(bs*F, H, W, D * self.expansion).permute(0, 3, 1, 2)

        z, shapes = self.encoder(x)
        z = self.quant_conv(z)
        return z, shapes
        
    def forward_decoder(self, z, shapes, input_shape):
        image_only_indicator = torch.zeros(input_shape[0], input_shape[1], dtype=z.dtype, device=z.device)
        logits = self.decoder(z, shapes, image_only_indicator)

        bs, F, H, W, D = input_shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
        template = self.class_embeds.weight.T.unsqueeze(0) # 1, expansion, cls
        similarity = torch.matmul(logits, template) # -1, D, cls
        # pred = similarity.argmax(dim=-1) # -1, D
        # pred = pred.reshape(bs, F, H, W, D)
        return similarity.reshape(bs, F, H, W, D, self.num_cls)


    def forward_decoder_give_logits(self, z, shapes, input_shape):
        image_only_indicator = torch.zeros(input_shape[0], input_shape[1], dtype=z.dtype, device=z.device)
        logits = self.decoder(z, shapes, image_only_indicator)

        bs, F, H, W, D = input_shape
        # logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
        logits = logits.permute(0, 2, 3, 1).reshape(bs*F, H, W, D, self.expansion)
        return logits
    

    def forward_decoder_w_logits(self, logits, shapes, input_shape):
        # image_only_indicator = torch.zeros(input_shape[0], input_shape[1], dtype=z.dtype, device=z.device)
        # logits = self.decoder(z, shapes, image_only_indicator)

        bs, F, H, W, D = input_shape
        # logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
        logits = logits.reshape(-1, D, self.expansion)
        template = self.class_embeds.weight.T.unsqueeze(0) # 1, expansion, cls
        similarity = torch.matmul(logits, template) # -1, D, cls
        # pred = similarity.argmax(dim=-1) # -1, D
        # pred = pred.reshape(bs, F, H, W, D)
        return similarity.reshape(bs, F, H, W, D, self.num_cls)


    # def forward_decoder_w_logits(self, logits, shapes, input_shape):
    #     bs, F, H, W, D = input_shape
    #     logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.expansion)
    #     template = self.class_embeds.weight.T.unsqueeze(0) # 1, expansion, cls
    #     similarity = torch.matmul(logits, template) # -1, D, cls
    #     # pred = similarity.argmax(dim=-1) # -1, D
    #     # pred = pred.reshape(bs, F, H, W, D)
    #     return similarity.reshape(bs, F, H, W, D, self.num_cls)


    def forward(self, x, **kwargs):
        # xs = self.forward_encoder(x)
        # logits = self.forward_decoder(xs)
        # return logits, xs[-1]
        
        output_dict = {}
        z, shapes = self.forward_encoder(x)
        if self.use_vq:
            z_sampled, loss, info = self.vqvae(z, is_voxel=False)
            output_dict.update({'embed_loss': loss})
        else:
            z_sampled, z_mu, z_sigma, logvar = self.sample_z(z)
            output_dict.update({
                'z_mu': z_mu,
                'z_sigma': z_sigma,
                'logvar': logvar})
        
        logits = self.forward_decoder(z_sampled, shapes, x.shape)
        
        output_dict.update({'logits': logits})
    
        if not self.training:
            pred = logits.argmax(dim=-1).detach().cuda()
            output_dict['sem_pred'] = pred
            pred_iou = deepcopy(pred)
            
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou
            
        return output_dict
        # loss, kl, rec = self.loss(logits, x, z_mu, z_sigma)
        # return loss, kl, rec
        
    def generate(self, z, shapes, input_shape):
        logits = self.forward_decoder(z, shapes, input_shape)
        return {'logits': logits}


if __name__ == "__main__":
    # test encoder
    import torch
    encoder = Encoder2D(in_channels=3, ch=64, out_ch=64, ch_mult=(1,2,4,8), num_res_blocks=2, resolution=200,attn_resolutions=(100,50), z_channels=64, double_z=True)
    #decoder = Decoder3D()
    decoder = Decoder2D(in_channels=3, ch=64, out_ch=3, ch_mult=(1,2,4,8), num_res_blocks=2, resolution=200,attn_resolutions=(100,50), z_channels=64, give_pre_end=False)
    
    import pdb; pdb.set_trace()