import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
import torch.nn.functional as F

class BN(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(BN, self).__init__(cfg, model, optimizer)
        return

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        outputs = model(batch_data)
        return outputs

    def configure_model(self, model: nn.Module):

        model.requires_grad_(False)

        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

        return model

