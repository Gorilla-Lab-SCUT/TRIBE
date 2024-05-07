import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from ..utils.wrapper_model import WrapperModel
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

class TTAC(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(TTAC, self).__init__(cfg, model, optimizer)
        self.model = WrapperModel(self.model, self.cfg.ADAPTER.TTAC.CLASSIFIER)
        self.categorical_mu, self.categorical_cov, self.global_mu, self.global_cov = self.collect_source_statistics()

        self.ema_categorical_mu = self.categorical_mu.clone()
        self.ema_categorical_cov = self.categorical_cov.clone()
        self.ema_global_mu = self.global_mu.clone().fill_(0.)
        self.ema_global_cov = self.global_cov.clone().fill_(0.)

        bias = self.global_cov.max().item() / 30.
        self.template_ext_cov = torch.eye(self.global_mu.shape[0]).cuda() * bias

        self.class_num = self.categorical_cov.shape[0]
        self.ema_n = torch.zeros(self.class_num).cuda()
        self.ema_global_n = 0. 

        if self.class_num == 10:
            self.ema_length = 128
        else:
            self.ema_length = 64
        return

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        logit = model(batch_data)
        feat = model.backbone_out
        # adapt
        softmax_logit = logit.softmax(dim=-1)
        pro, pseudo_label = softmax_logit.max(dim=-1)
        pseudo_label_mask = (pro > 0.9)
        
        feat2 = feat[pseudo_label_mask]
        pseudo_label2 = pseudo_label[pseudo_label_mask]

        loss = 0.

        # Mixture Gaussian
        b, d = feat2.shape
        feat_ext2_categories = torch.zeros(self.class_num, b, d).cuda() # K, N, D
        feat_ext2_categories.scatter_add_(dim=0, index=pseudo_label2[None, :, None].expand(-1, -1, d), src=feat2[None, :, :])

        num_categories = torch.zeros(self.class_num, b, dtype=torch.int).cuda() # K, N
        num_categories.scatter_add_(dim=0, index=pseudo_label2[None, :], src=torch.ones_like(pseudo_label2[None, :], dtype=torch.int))

        self.ema_n += num_categories.sum(dim=1) # K
        alpha = torch.where(self.ema_n > self.ema_length, torch.ones(self.class_num, dtype=torch.float).cuda() / self.ema_length, 1. / (self.ema_n + 1e-10))

        delta_pre = (feat_ext2_categories - self.ema_categorical_mu[:, None, :]) * num_categories[:, :, None] # K, N, D
        delta = alpha[:, None] * delta_pre.sum(dim=1) # K, D
        new_component_mean = self.ema_categorical_mu + delta
        new_component_cov =  self.ema_categorical_cov \
                            + alpha[:, None, None] * ((delta_pre.permute(0, 2, 1) @ delta_pre) - num_categories.sum(dim=1)[:, None, None] * self.ema_categorical_cov) \
                            - delta[:, :, None] @ delta[:, None, :]

        with torch.no_grad():
            self.ema_categorical_mu = new_component_mean.detach()
            self.ema_categorical_cov = new_component_cov.detach()
        
        for label in pseudo_label2.unique():
            if self.ema_n[label] >= 16:
                source_domain = torch.distributions.MultivariateNormal(self.categorical_mu[label, :], self.categorical_cov[label, :, :] + self.template_ext_cov)
                target_domain = torch.distributions.MultivariateNormal(new_component_mean[label, :], new_component_cov[label, :, :] + self.template_ext_cov)
                loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) / self.class_num * 0.5

        b = feat.shape[0]
        self.ema_global_n += b
        alpha = 1. / 1280 if self.ema_global_n > 1280 else 1. / self.ema_global_n
        delta_pre = (feat - self.ema_global_mu.cuda())
        delta = alpha * delta_pre.sum(dim=0)
        tmp_mu = self.ema_global_mu.cuda() + delta
        tmp_cov = self.ema_global_cov.cuda() + alpha * (delta_pre.t() @ delta_pre - b * self.ema_global_cov.cuda()) - delta[:, None] @ delta[None, :]
        with torch.no_grad():
            self.ema_global_mu = tmp_mu.detach().cpu()
            self.ema_global_cov = tmp_cov.detach().cpu()

        source_domain = torch.distributions.MultivariateNormal(self.global_mu, self.global_cov + self.template_ext_cov)
        target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + self.template_ext_cov)
        loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * 0.5
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return logit
    
    def collect_source_statistics(self):
        transform = transforms.Compose([transforms.ToTensor()])
        if self.cfg.CORRUPTION.DATASET == "cifar10":
            tset = CIFAR10(self.cfg.DATA_DIR, True, transform)
            class_num = 10
        elif self.cfg.CORRUPTION.DATASET == "cifar100":
            tset = CIFAR100(self.cfg.DATA_DIR, True, transform)
            class_num = 100
        elif self.cfg.CORRUPTION.DATASET == "mnist":
            tset = MNIST(self.cfg.DATA_DIR, True, transform)
            class_num = 10
        else:
            raise Exception("Not Implemented in collect_source_statistics")
        loader = DataLoader(tset, self.cfg.TEST.BATCH_SIZE)
        self.model.eval()

        feat_stack = [[] for i in range(class_num)]
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(loader):
                p_l = self.model(inputs.cuda()).argmax(dim=-1)
                feat = self.model.backbone_out
                for label in p_l.unique():
                    label_mask = p_l == label
                    feat_stack[label].extend(feat[label_mask, :])

            ext_mu = []
            ext_cov = []
            ext_all = []

            for feat in feat_stack:
                ext_mu.append(torch.stack(feat).mean(dim=0))
                ext_cov.append(covariance(torch.stack(feat)))
                ext_all.extend(feat)

            ext_all = torch.stack(ext_all)
            ext_all_mu = ext_all.mean(dim=0)
            ext_all_cov = covariance(ext_all)

            ext_mu = torch.stack(ext_mu)
            ext_cov = torch.stack(ext_cov)

        return ext_mu, ext_cov, ext_all_mu, ext_all_cov


    def configure_model(self, model: nn.Module):
        """Configure model."""
        model.eval()   # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        model.requires_grad_(True)  # disable grad, to (re-)enable only necessary parts
        # re-enable gradient for normalization layers
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
            elif isinstance(m, nn.Linear):
                m.requires_grad_(False)
        return model



def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / n
    return cov
