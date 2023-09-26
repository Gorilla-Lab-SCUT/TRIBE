from .base_dataset import TTADatasetBase, DatumRaw, DatumList
from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc
from .mnist_c import load_mnistc
import numpy as np
import torch

class ToyDataset(TTADatasetBase):
    def __init__(self, cfg, *args, **kwargs):
       
        self.domain_id_to_name = {0: 'target'}
        data_source = []
        
        nums = [1000, 599, 359, 215, 129, 77, 46, 27, 16, 10]

        for label, num in enumerate(nums):
            data = np.random.randn(num) * 0.01 + label
            for i in data:
                data_item = DatumRaw(torch.Tensor([i]), label, 0)
                data_source.append(data_item)
        
        super().__init__(cfg, data_source)


class GradualCorruptionCIFAR(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        all_severity = [all_severity] if not isinstance(all_severity, list) else all_severity

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None
        if cfg.CORRUPTION.DATASET == "gradualCifar10":
            self.load_image = load_cifar10c
        elif cfg.CORRUPTION.DATASET == "gradualCifar100":
            self.load_image = load_cifar100c
        self.domain_id_to_name = {}
        data_source = []
        for i_c, corruption in enumerate(self.corruptions):
            if i_c == 0:
                severities = [5,4,3,2,1] 
            else:
                severities = [1,2,3,4,5,4,3,2,1] 
            for i_s, severity in enumerate(severities):
                d_name = f"{corruption}_{severity}"
                d_id = i_s + i_c * 9
                self.domain_id_to_name[d_id] = d_name

                x, y = self.load_image(cfg.CORRUPTION.NUM_EX,
                                    severity,
                                    cfg.DATA_DIR,
                                    False,
                                    [corruption])
                for i in range(len(y)):
                    data_item = DatumRaw(x[i], y[i].item(), d_id)
                    data_source.append(data_item)
        super().__init__(cfg, data_source)


class CorruptionImageNet(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        all_severity = [all_severity] if not isinstance(all_severity, list) else all_severity

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None
        if cfg.CORRUPTION.DATASET == "imagenet":
            self.load_image = load_imagenetc
        self.domain_id_to_name = {}
        data_source = []
        for i_s, severity in enumerate(self.severity):
            for i_c, corruption in enumerate(self.corruptions):
                d_name = f"{corruption}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name
                x, y = self.load_image(cfg.CORRUPTION.NUM_EX,
                                       severity,
                                       cfg.DATA_DIR,
                                       False,
                                       [corruption],
                                       prepr=lambda x: x)
                for i in range(len(y)):
                    data_item = DatumList(x[i], y[i].item(), d_id)
                    data_source.append(data_item)

        super().__init__(cfg, data_source)


class CorruptionMNIST(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        all_severity = [all_severity] if not isinstance(all_severity, list) else all_severity

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None

        if cfg.CORRUPTION.DATASET == "mnist":
            self.load_image = load_mnistc
        
        self.domain_id_to_name = {}
        data_source = []
        for i_s, severity in enumerate(self.severity):
            for i_c, corruption in enumerate(self.corruptions):
                d_name = f"{corruption}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name

                x, y = self.load_image(
                                       cfg.DATA_DIR,
                                       False,
                                       [corruption],
                                       )
                for i in range(len(y)):
                    data_item = DatumRaw(x[i], y[i].item(), d_id)
                    data_source.append(data_item)
        
        super().__init__(cfg, data_source)






        