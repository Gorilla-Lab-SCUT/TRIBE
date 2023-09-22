import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from .base_adapter import softmax_entropy
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torchvision import transforms
from torch.utils.data import DataLoader
import math
import os


class EATA(BaseAdapter):
    def __init__(self, cfg, model, optimizer, e_margin = math.log(1000)*0.40, d_margin=0.05, fisher_alpha=2000.0):
        super(EATA, self).__init__(cfg, model, optimizer)
        self.fishers = self.collect_source_statistics()
        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0

        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin

        self.current_model_probs = None

        self.fisher_alpha = fisher_alpha
        return

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(batch_data, self.model, self.optimizer, self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin)
        self.num_samples_update_2 += num_counts_2
        self.num_samples_update_1 += num_counts_1
        self.reset_model_probs(updated_probs)
        return outputs
    
    def reset_model_probs(self, probs):
        self.current_model_probs = probs
    

    def collect_source_statistics(self):
        transform = transforms.Compose([transforms.ToTensor()])
        if self.cfg.CORRUPTION.DATASET == "cifar10":
            tset = CIFAR10(self.cfg.DATA_DIR, True, transform)
        elif self.cfg.CORRUPTION.DATASET == "cifar100":
            tset = CIFAR100(self.cfg.DATA_DIR, True, transform)
        elif self.cfg.CORRUPTION.DATASET == "imagenet":
            te_transforms = transforms.Compose([transforms.Resize(256),
									transforms.CenterCrop(224),
									transforms.ToTensor()])
            tset = ImageNet(os.path.join(self.cfg.DATA_DIR, 'ImageNet'), 'val', transform=te_transforms)
        else:
            raise Exception("Not Implemented in collect_source_statistics")
        loader = DataLoader(tset, self.cfg.TEST.BATCH_SIZE, num_workers=4)
        self.model.eval()

        fishers = {}
        ewc_optimizer = torch.optim.SGD(self.collect_params(self.model)[0], 0.001)
        for batch_idx, (inputs, _) in enumerate(loader):
            outputs = self.model(inputs.cuda())
            _, targets = outputs.max(1)

            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if batch_idx > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if batch_idx == len(loader):
                        fisher = fisher / batch_idx
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()

        del ewc_optimizer
        return fishers


    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)

        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
        return model



@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    Return: 
    1. model outputs; 
    2. the number of reliable and non-redundant samples; 
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward
    outputs = model(x)
    # adapt
    entropys = softmax_entropy(outputs, outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0]>-0.1)
    entropys = entropys[filter_ids_1] 
    # filter redundant samples
    if current_model_probs is not None: 
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
    if x[ids1][ids2].size(0) != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
