from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
from .my_transforms import Clip, ColorJitterPro, GaussianNoise
from time import time
# import logging

import torch.optim as optim


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        Clip(0.0, 1.0), 
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=transforms.InterpolationMode.BILINEAR,
            # fillcolor=None
            fill=0
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        GaussianNoise(0, gaussian_std),
        Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def find_quantile(arr, perc):
    arr_sorted = torch.sort(arr).values
    frac_idx = perc*(len(arr_sorted)-1)
    frac_part = frac_idx - int(frac_idx)
    low_idx = int(frac_idx)
    high_idx = low_idx + 1
    quant = arr_sorted[low_idx] + (arr_sorted[high_idx]-arr_sorted[low_idx]) * frac_part # linear interpolation

    return quant


class PETALFim(nn.Module):
    """PETALFim adapts a model by using PETAL during testing and restoring based on Fisher Matrix.

    A model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, optimizer):
        super().__init__()
        model, mean_model, cov_model = model
        model = configure_model(model)
        mean_model = configure_model(mean_model)
        cov_model = configure_model(cov_model)
        params, param_names = collect_params(model)
        optimizer = optimizer(params)
        
        self.cfg = cfg
        self.model = model
        self.mean_model = mean_model
        self.cov_model = cov_model
        self.optimizer = optimizer
        self.steps = self.cfg.ADAPTER.PETAL.STEPS
        assert self.steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = self.cfg.ADAPTER.PETAL.EPISODIC
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    
        self.mt = self.cfg.ADAPTER.PETAL.MT_ALPHA
        self.rst = self.cfg.ADAPTER.PETAL.RST_M
        self.ap = self.cfg.ADAPTER.PETAL.AP
        self.spw = self.cfg.ADAPTER.PETAL.SPW
        self.perc = self.cfg.ADAPTER.PETAL.PERC

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        outputs = self.model(x)
        # Teacher Prediction, see line 3 in the main paper
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0].detach()
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction, see line 3-6 in Algorithm 2 in the main paper 
        N = 32 
        outputs_emas = []
        for i in range(N):
            x_tx = self.transform(x)
            outputs_  = self.model_ema(x_tx).detach()
            outputs_emas.append(outputs_)

        # Threshold choice discussed in CoTTA paper's supplementary
        if anchor_prob.mean(0)<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema


        # Student update, line 10-11 in Algorithm 2 in the main paper
        loss_H = (softmax_entropy(outputs, outputs_ema)).mean(0)
        para_loss = weighted_parameter_loss(self.model, self.mean_model, self.cov_model)

        loss = loss_H + self.spw * para_loss  # Equation 12 in the Appendix

        loss.backward()
        
        # Fisher Information, line 13 in Algorithm 2 in the main paper
        fisher_dict = {}
        for nm, m  in self.model.named_modules():  ## previously used model, but now using self.model
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    fisher_dict[f"{nm}.{npp}"] = p.grad.data.clone().pow(2)
        fisher_list = []
        for name in fisher_dict:
            fisher_list.append(fisher_dict[name].reshape(-1))
        fisher_flat = torch.cat(fisher_list)
        threshold = find_quantile(fisher_flat, self.perc)

        optimizer.step()
        optimizer.zero_grad()
        # Teacher update, see line 12 in Algorithm 2 in the main paper
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)

        # FIM based restore, line 13-15 in Algorithm 2 in the main paper
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask_fish = (fisher_dict[f"{nm}.{npp}"]<threshold).float().cuda() # masking makes it restore candidate
                        mask = mask_fish
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema

def weighted_parameter_loss(params, means, variances, damp=1e-6):
    """
    Uses a quadratic regularizer around the given means with provided diagional variance
    """
    para_loss = 0.0
    for (name_b, param_b), (name_m, param_m), (name_c, param_c) in zip(params.named_parameters(), means.named_parameters(), variances.named_parameters()):
        assert name_b == name_m == name_c
        para_loss += torch.sum(torch.square(param_b - param_m) / (param_c + damp))
    para_loss = 0.5*para_loss
    return para_loss



@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model"""
    # train mode
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"