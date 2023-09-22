from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model, load_model_bayes
from torchvision.models import resnet18
import torch.nn as nn
import os
import torch
from .toy import Toy
import torch.nn.functional as F


def build_model(cfg):
    if cfg.MODEL.ARCH == "Toy":
        base_model = Toy()
        base_model.cuda()
        return base_model
    
    if cfg.CORRUPTION.DATASET in ["cifar10", "cifar100", "imagenet", "gradualCifar10", "gradualCifar100"]:
        if "gradual" in cfg.CORRUPTION.DATASET:
            dataset = cfg.CORRUPTION.DATASET.replace("gradualC", "c")
        else:
            dataset = cfg.CORRUPTION.DATASET
        if cfg.ADAPTER.NAME == "petal":
            base_model, base_mean_model, base_cov_model = load_model_bayes(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                    dataset, ThreatModel.corruptions)
            base_model, base_mean_model, base_cov_model = base_model.cuda(), base_mean_model.cuda(), base_cov_model.cuda()
            base_model = (base_model, base_mean_model, base_cov_model)
        else:
            base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                                    dataset, ThreatModel.corruptions).cuda()
    elif cfg.CORRUPTION.DATASET == "mnist":
        base_model = resnet18(pretrained=True)
        base_model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=base_model.conv1.kernel_size,
            stride=base_model.conv1.stride,
            padding=base_model.conv1.padding,
            bias=False
        )
        base_model.fc = nn.Linear(
            in_features=base_model.fc.in_features,
            out_features=10
        )

        if os.path.exists(os.path.join(cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, 'source', 'resnet18.pth')):
            base_model.load_state_dict(torch.load(os.path.join(cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, 'source', 'resnet18.pth')))
        base_model.cuda()
    else:
        raise NotImplementedError()

    return base_model
