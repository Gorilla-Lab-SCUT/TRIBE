import torch
import torch.nn as nn
from ..utils.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, RobustBN2d, BalancedRobustBN1dV5



class Toy(nn.Module):
    def __init__(self, ):
        super().__init__()
        bn = nn.BatchNorm1d(num_features=1, affine=True, momentum=0.1)
        bn.running_mean.copy_(torch.tensor([4.5]))
        bn.running_var.copy_(torch.tensor([8.25]))
        # self.bn = bn
        self.bn = BalancedRobustBN1dV5(bn, num_classes=10, momentum_a=0.1, gamma=0.)
        pass

    def forward(self, x, y):
        x = x[:, :, None]
        self.bn.label = y
        out = self.bn(x)
        print(self.bn.global_mean, self.bn.global_var)
        # print(self.bn.running_mean, self.bn.running_var)
        return self.bn(x)