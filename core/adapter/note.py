import torch
import torch.nn as nn
from ..utils import memory
from .base_adapter import BaseAdapter
from .base_adapter import softmax_entropy
from ..utils.iabn import convert_iabn, InstanceAwareBatchNorm2d, InstanceAwareBatchNorm1d

class NOTE(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(NOTE, self).__init__(cfg, model, optimizer)
        self.mem = memory.PBRS(capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, num_class=cfg.CORRUPTION.NUM_CLASS)
        self.current_instance = 0
        return

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            pred_out = self.model(batch_data)
            pseudo_label = torch.argmax(pred_out, dim=1)
        model.train()
        
        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            current_instance = (data, p_l, -1, -1, 0)
            self.mem.add_instance(current_instance)
            self.current_instance += 1
        self.update_model(model, optimizer)

        return pred_out

    def update_model(self, model, optimizer, batch_data=None):
        batch_data, _, _ = self.mem.get_memory()
        batch_data = torch.stack(batch_data)
        # adapt
        model.train()
        outputs = model(batch_data)
        loss = softmax_entropy(outputs, outputs).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()   
        return

    def configure_model(self, model: nn.Module):
        convert_iabn(model)

        for param in model.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in model.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                module.track_running_stats = True
                module.momentum = 0.01

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            if isinstance(module, InstanceAwareBatchNorm2d) or isinstance(module, InstanceAwareBatchNorm1d):
                for param in module.parameters():
                    param.requires_grad = True

        return model
    
   

