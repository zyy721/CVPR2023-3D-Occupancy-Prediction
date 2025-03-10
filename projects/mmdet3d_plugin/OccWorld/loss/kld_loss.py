from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F
import torch

@OPENOCC_LOSS.register_module()
class KldLoss(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'z_mu': 'z_mu',
                'logvar': 'logvar'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.kld_loss
    
    def kld_loss(self, z_mu, logvar):
        kld_loss = torch.mean(-0.5 * (1 + logvar - z_mu.pow(2) - logvar.exp()))
        return kld_loss