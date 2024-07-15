import torch
import torch.nn as nn
from copy import deepcopy
try:
    import balanced_bn
except:
    from . import balanced_bn_pyv as balanced_bn

class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

        self.current_mu = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.source_mean.view(1, -1), self.source_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias


class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias


class BalancedBNV5(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, num_classes=1, momentum_a=1e-01, gamma=0.0):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.num_classes = num_classes
        self.eps = bn_layer.eps
        self.momentum = momentum_a
        self.gamma = gamma

        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("global_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("global_var", deepcopy(bn_layer.running_var))
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)
        local_mean = deepcopy(bn_layer.running_mean)[None, ...].expand(num_classes, -1).clone()
        local_var = deepcopy(bn_layer.running_var)[None, ...].expand(num_classes, -1).clone()
        self.register_buffer("local_mean", local_mean)
        self.register_buffer("local_var", local_var)

        self.label = None

    def forward(self, x):
        self.global_mean = self.global_mean.detach()
        self.global_var = self.global_var.detach()
        self.local_mean = self.local_mean.detach()
        self.local_var = self.local_var.detach()

   
class BalancedRobustBN1dV5(BalancedBNV5):
    def forward(self, x):
        super().forward(x)
        # B, N, C = x.shape
        # x = x.reshape(B * N, C, 1)
        x = x.permute(0, 2, 1)
        label = self.label
        if label is not None:
            balanced_bn.update_statistics_1d_v5(self.local_mean, self.local_var, self.global_mean, self.global_var, self.momentum, x, label, self.gamma, self.training)
            self.label = None
        else:
            if self.training:
                b_var, b_mean = torch.var_mean(x, dim=[0, 2], unbiased=False, keepdim=False)  # (C,)
                self.global_mean = (1 - self.momentum) * self.global_mean + self.momentum * b_mean
                self.global_var = (1 - self.momentum) * self.global_var + self.momentum * b_var
        x = (x - self.global_mean[None, :, None]) / torch.sqrt(self.global_var[None, :, None] + self.eps)
        x = self.weight[None, :, None] * x + self.bias[None, :, None]
        return x.permute(0, 2, 1)


class BalancedRobustBN2dV5(BalancedBNV5):
    def forward(self, x):
        super().forward(x)
        label = self.label
        if label is not None:
            balanced_bn.update_statistics_2d_v5(self.local_mean, self.local_var, self.global_mean, self.global_var, self.momentum, x, label, self.gamma, self.training)
            self.label = None
        else:
            if self.training:
                b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
                self.global_mean = (1 - self.momentum) * self.global_mean + self.momentum * b_mean
                self.global_var = (1 - self.momentum) * self.global_var + self.momentum * b_var
        x = (x - self.global_mean[None, :, None, None]) / torch.sqrt(self.global_var[None, :, None, None] + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None] 



class BalancedBNEMA(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, num_classes=1, momentum_a=1e-01, gamma=0.0):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.num_classes = num_classes
        self.eps = bn_layer.eps
        self.momentum = momentum_a
        self.gamma = gamma

        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("global_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("global_var", deepcopy(bn_layer.running_var))
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)
        local_mean = deepcopy(bn_layer.running_mean)[None, ...].expand(num_classes, -1).clone()
        local_var = deepcopy(bn_layer.running_var)[None, ...].expand(num_classes, -1).clone()
        self.register_buffer("local_mean", local_mean)
        self.register_buffer("local_var", local_var)

        self.label = None

    def forward(self, x):
        self.global_mean = self.global_mean.detach()
        self.global_var = self.global_var.detach()
        self.local_mean = self.local_mean.detach()
        self.local_var = self.local_var.detach()
        
class BalancedRobustBN2dEMA(BalancedBNEMA):
    def forward(self, x):
        super().forward(x)
        label = self.label
        if label is not None:
            balanced_bn.update_statistics_2d_ema(self.local_mean, self.local_var, self.global_mean, self.global_var, self.momentum, x, label, self.gamma, self.training)
            self.label = None
        else:
            if self.training:
                b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
                self.global_mean = (1 - self.momentum) * self.global_mean + self.momentum * b_mean
                self.global_var = (1 - self.momentum) * self.global_var + self.momentum * b_var
        x = (x - self.global_mean[None, :, None, None]) / torch.sqrt(self.global_var[None, :, None, None] + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None] 

