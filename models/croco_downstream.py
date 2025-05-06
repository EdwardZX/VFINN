# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# CroCo model for downstream tasks
# --------------------------------------------------------

import torch
import torch.nn as nn
from .croco import CroCoNet
from models.diffloss import DiffLoss


def croco_args_from_ckpt(ckpt):
    if 'croco_kwargs' in ckpt: # CroCo v2 released models
        return ckpt['croco_kwargs']
    elif 'args' in ckpt and hasattr(ckpt['args'], 'model'): # pretrained using the official code release
        s = ckpt['args'].model # eg "CroCoNet(enc_embed_dim=1024, enc_num_heads=16, enc_depth=24)"
        assert s.startswith('CroCoNet(')
        return eval('dict'+s[len('CroCoNet'):]) # transform it into the string of a dictionary and evaluate it
    else: # CroCo v1 released models
        return dict()

class CroCoDownstreamMonocularEncoder(CroCoNet):

    def __init__(self,
                 head,
                 **kwargs):
        """ Build network for monocular downstream task, only using the encoder.
        It takes an extra argument head, that is called with the features 
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        NOTE: It works by *calling super().__init__() but with redefined setters
        
        """
        super(CroCoDownstreamMonocularEncoder, self).__init__(**kwargs)
        head.setup(self)
        self.head = head

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return

    def _set_decoder(self, *args, **kwargs):
        """ No decoder """
        return

    def _set_prediction_head(self, *args, **kwargs):
        """ No 'prediction head' for downstream tasks."""
        return

    def forward(self, img):
        """
        img if of size batch_size x 3 x h x w
        """
        B, C, H, W = img.size()
        img_info = {'height': H, 'width': W}
        need_all_layers = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        out, _, _ = self._encode_image(img, do_mask=False, return_all_blocks=need_all_layers)
        return self.head(out, img_info)
        
        
class CroCoDownstreamBinocular(CroCoNet):

    def __init__(self,
                 head,
                 **kwargs):
        """ Build network for binocular downstream task
        It takes an extra argument head, that is called with the features 
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        """
        super(CroCoDownstreamBinocular, self).__init__(**kwargs)
        head.setup(self)
        self.head = head

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head for downstream tasks, define your own head """
        return
        
    def encode_image_pairs(self, img1, img2, return_all_blocks=False):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension 
             than to encode them separately
        """
        ## the two commented lines below is the naive version with separate encoding
        #out, pos, _ = self._encode_image(img1, do_mask=False, return_all_blocks=return_all_blocks)
        #out2, pos2, _ = self._encode_image(img2, do_mask=False, return_all_blocks=False)
        ## and now the faster version
        out, pos, _ = self._encode_image( torch.cat( (img1,img2), dim=0), do_mask=False, return_all_blocks=return_all_blocks )
        if return_all_blocks:
            out,out2 = list(map(list, zip(*[o.chunk(2, dim=0) for o in out])))
            out2 = out2[-1]
        else:
            out,out2 = out.chunk(2, dim=0)
        pos,pos2 = pos.chunk(2, dim=0)            
        return out, out2, pos, pos2

    def forward(self, img1, img2):
        B, C, H, W = img1.size()
        img_info = {'height': H, 'width': W}
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        out, out2, pos, pos2 = self.encode_image_pairs(img1, img2, return_all_blocks=return_all_blocks)
        if return_all_blocks:
            decout = self._decoder(out[-1], pos, None, out2, pos2, return_all_blocks=return_all_blocks)
            decout = out+decout
        else:
            decout = self._decoder(out, pos, None, out2, pos2, return_all_blocks=return_all_blocks)
        
        print(self.head(decout, img_info).shape)
        return self.head(decout, img_info)
    
class CroCoDownstreamPIV(CroCoNet):

    def __init__(self,
                 head,
                 **kwargs):
        """ Build network for binocular downstream task
        It takes an extra argument head, that is called with the features 
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        """
        super(CroCoDownstreamPIV, self).__init__(**kwargs)
        head.setup(self)
        self.head = head

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return
        
    def encode_image_pairs(self, img1, img2, return_all_blocks=False):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension 
             than to encode them separately
        """
        ## the two commented lines below is the naive version with separate encoding
        out, pos, _ = self._encode_image(img1, do_mask=False, return_all_blocks=return_all_blocks)
        out2, pos2, _ = self._encode_image(img2, do_mask=False, return_all_blocks=False)
        ## and now the faster version           
        return out, out2, pos, pos2

    def forward(self, img1, img2):
        B, C, H, W = img1.size()
        img_info = {'height': H, 'width': W}
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        out, out2, pos, pos2 = self.encode_image_pairs(img1, img2, return_all_blocks=return_all_blocks)
        if return_all_blocks:
            decout = self._decoder(out[-1], pos, None, out2, pos2, return_all_blocks=return_all_blocks)
            decout = out+decout
        else:
            decout = self._decoder(out, pos, None, out2, pos2, return_all_blocks=return_all_blocks)
        
        # out = self.prediction_head(decfeat)
        # print(self.head(decout, img_info).shape)
        out1 = out[-1]
        pos1 = pos
        IntensityPatch_I = self.prediction_head(self._decoder(out1, pos1, None, out1, pos1))
        IntensityPatch_II = self.prediction_head(self._decoder(out2, pos2, None, out2, pos2))
        return self.head(decout, img_info), IntensityPatch_I, IntensityPatch_II

    def decode_to_image(self,dec1, img1, dec2, img2,
                   imagenet_mean_tensor, imagenet_std_tensor):
        
        patchified1 = self.patchify(img1)
        mean1 = patchified1.mean(dim=-1, keepdim=True)
        var1 = patchified1.var(dim=-1, keepdim=True)
        out1 = self.unpatchify(dec1 * (var1 + 1.e-6)**.5 + mean1)
        out1 = out1 * imagenet_std_tensor + imagenet_mean_tensor

        # the paired
        patchified2 = self.patchify(img2)
        mean2 = patchified2.mean(dim=-1, keepdim=True)
        var2 = patchified2.var(dim=-1, keepdim=True)
        out2 = self.unpatchify(dec2 * (var2 + 1.e-6)**.5 + mean2)
    # undo imagenet normalization, prepare masked image
        out2 = out2 * imagenet_std_tensor + imagenet_mean_tensor

        return out1, out2
    
    # def patchify_PIV_pairs(self, ref, target):
    def patchify_warpForward(self,tensorTarget, tensorInput, tensorFlow): # Im1, Im2, Flow
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1,
                                                    tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1,
                                                    tensorFlow.size(3))

        Forward_tensorGrid = torch.cat(
                [tensorHorizontal, tensorVertical], 1).to(tensorFlow.device)

        tensorFlow = torch.cat([
            tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
        ], 1)
        
        tensorTarget_pred =  torch.nn.functional.grid_sample(
            input=tensorInput,
            grid= (Forward_tensorGrid + tensorFlow).permute(0, 2, 3, 1), # B, [vx,vy], H, W -> B , H, W, [p + vx, p + vy]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        
        return self.patchify(tensorTarget), self.patchify(tensorTarget_pred)
        

class PIVDiffL(nn.Module):
    def __init__(self,
                 img_channel,
                 patch_size,
                 enc_embed_dim,
                 img_w = 224,
                 depth = 3,
                 num_sampling_steps = '100',):
        super(PIVDiffL, self).__init__()
        self.img_channel = img_channel # RGB
        self.patch_size = patch_size
        self.enc_embed_dim = enc_embed_dim
        self.num_sampling_steps = num_sampling_steps
        self.depth = depth
        self.img_w = img_w

        #adding diffusion loss
        self.diffloss = DiffLoss(
        target_channels= self.img_channel * patch_size**2, # B x patchified x patch**2 x channel
        z_channels=self.enc_embed_dim,
        width=self.img_w, # width == height
        depth=3,
        num_sampling_steps=self.num_sampling_steps,
        )

    def forward(self, patch_target, patch_target_pred, z):
        # target with pachified aligned with encoder z: B x 196 x [patch ** 2]
        B, L, _ = patch_target.shape
        patch_target = patch_target.reshape(B*L,-1)
        patch_target_pred = patch_target_pred.reshape(B*L,-1)
        z = z.reshape(B*L,-1)
        loss = self.diffloss(patch_target, patch_target_pred, z)
        return loss 
    
