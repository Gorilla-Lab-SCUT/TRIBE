#!/usr/bin/env python
# coding: utf-8

import math
import os
import itertools
from copy import deepcopy


os.environ["CUDA_VISIBLE_DEVICES"]="0"


from robustbench.utils import load_model
from robustbench.data import load_cifar100
from core.configs.defaults import cfg
from robustbench.model_zoo.enums import ThreatModel


import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from typing import Callable, Dict, Optional, Sequence, Set, Tuple


class SquaredAverageModel(nn.Module):
    def __init__(self, model, device=None, avg_fn=None, use_buffers=False):
        super(SquaredAverageModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter +                     (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers else model.parameters()
        )
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            squared_p_model_ = (p_model_**2) # squaring here
            if self.n_averaged == 0:
                p_swa.detach().copy_(squared_p_model_) 
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), squared_p_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1


cfg.MODEL.ARCH = "Hendrycks2020AugMix_ResNeXt"
cfg.CORRUPTION.DATASET = "cifar100"
cfg.CKPT_DIR = "./ckpt"


base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()


PREPROCESSINGS = {
    'Res256Crop224': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()]),
    'Crop288': transforms.Compose([transforms.CenterCrop(288),
                                   transforms.ToTensor()]),
    'none': transforms.Compose([transforms.ToTensor()]),
}


def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_cifar100_train(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR100(root=data_dir,
                               train=True,
                               transform=PREPROCESSINGS["none"],
                               download=True)
    return _load_dataset(dataset, n_examples)


cfg.CORRUPTION.NUM_EX = 50000


batch_size = 200


x_train, y_train = load_cifar100_train(cfg.CORRUPTION.NUM_EX, cfg.DATA_DIR)
x_train, y_train = x_train.cuda(), y_train.cuda()


import torch.utils.data as data_utils

train = data_utils.TensorDataset(x_train, y_train)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)


def cf_learning_rate(init):#, epoch):
#     optim_factor = 0
#     if(epoch > 160):
#         optim_factor = 3
#     elif(epoch > 120):
#         optim_factor = 2
#     elif(epoch > 60):
#         optim_factor = 1
    optim_factor = 3

    return init*math.pow(0.2, optim_factor)


from torch.optim.swa_utils import SWALR, AveragedModel
from tqdm import trange
from tqdm import tqdm


args_lr = 0.1
n_batches = math.ceil(x_train.shape[0] / batch_size)
extra_epochs = 5


optimizer = torch.optim.SGD(base_model.parameters(), lr=cf_learning_rate(args_lr), momentum=0.9, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()
swa_model = AveragedModel(base_model)
sqa_model = SquaredAverageModel(base_model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
swa_start = 200
swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=3, swa_lr=0.01)

for epoch in range(200, 200+extra_epochs):
    for x_curr, y_curr in tqdm(train_loader):
        optimizer.zero_grad()
        loss = loss_fn(base_model(x_curr), y_curr)
        loss.backward()
        optimizer.step()
    if epoch > swa_start:
        swa_model.update_parameters(base_model)
        sqa_model.update_parameters(base_model)
        swa_scheduler.step()
    else:
        scheduler.step()


# Update bn statistics for the swa_model at the end
torch.optim.swa_utils.update_bn(train_loader, swa_model)


# Update bn statistics for the sqa_model at the end
torch.optim.swa_utils.update_bn(train_loader, sqa_model)


model_path = "./ckpt/cifar100/corruptions/Hendrycks2020AugMix_ResNeXt_swa.pt"


torch.save(swa_model.module.state_dict(), model_path)


def covar(sqa_model, swa_model):
    cov_model = deepcopy(sqa_model)
    sqa_model = sqa_model
    swa_model = swa_model
    for p_cov, p_sqa, p_swa in zip(cov_model.parameters(), sqa_model.parameters(), swa_model.parameters()):
        p_sqa_ = p_sqa.detach()
        p_swa_ = p_swa.detach()
        p_cov.detach().copy_(p_sqa_ - (p_swa_**2))
    return cov_model


cov_model = covar(sqa_model, swa_model)


cov_model_path = "./ckpt/cifar100/corruptions/Hendrycks2020AugMix_ResNeXt_cov.pt"
torch.save(cov_model.module.state_dict(), cov_model_path)
print("Training completed.")
print("Files Hendrycks2020AugMix_ResNeXt_cov.pt and Hendrycks2020AugMix_ResNeXt_swa.pt created inside the directory ckpt/cifar100/corruptions/")