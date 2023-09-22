import numpy as np
from torch.utils.data.sampler import Sampler
from .datasets.base_dataset import DatumBase
from typing import Iterator, List
from collections import defaultdict
from numpy.random import dirichlet

class GradualDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase]):
        self.data_source = data_source
    
    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(range(len(self.data_source)))



class LabelDirichletDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase], gamma, batch_size, slots=None):

        self.domain_dict = defaultdict(list)
        self.classes = set()
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
            self.classes.add(item.label)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_class = len(self.classes)
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []
        for domain in self.domains:
            indices = np.array(self.domain_dict[domain])
            labels = np.array([self.data_source[i].label for i in indices])

            class_indices = [np.argwhere(labels == y).flatten() for y in range(self.num_class)]
            slot_indices = [[] for _ in range(self.num_slots)]

            label_distribution = dirichlet([self.gamma] * self.num_slots, self.num_class)

            for c_ids, partition in zip(class_indices, label_distribution):
                for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int))):
                    slot_indices[s].append(ids)

            for s_ids in slot_indices:
                permutation = np.random.permutation(range(len(s_ids)))
                ids = []
                for i in permutation:
                    ids.extend(s_ids[i])
                final_indices.extend(indices[ids])

        return iter(final_indices)


class LabelDirichletDomainSequenceLongTailed(Sampler):
    def __init__(self, data_source: List[DatumBase], gamma, batch_size, imb_factor, class_ratio="constant", slots=None):
        assert class_ratio in ["constant", "random"]

        self.domain_dict = defaultdict(list)
        self.classes = set()
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
            self.classes.add(item.label)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source
        self.gamma = gamma
        self.batch_size = batch_size
        self.imb_factor = imb_factor
        self.class_ratio = class_ratio
        self.num_class = len(self.classes)
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100
        self._prepare_for_iter()

    def _prepare_for_iter(self):
        final_indices = []
        for domain in self.domains:
            indices = np.array(self.domain_dict[domain])
            labels = np.array([self.data_source[i].label for i in indices])

            class_indices = [np.argwhere(labels == y).flatten() for y in range(self.num_class)]
            imb_class_indices = self.gen_imbalanced_data(class_indices)

            slot_indices = [[] for _ in range(self.num_slots)]

            label_distribution = dirichlet([self.gamma] * self.num_slots, self.num_class)

            for c_ids, partition in zip(imb_class_indices, label_distribution):
                for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int))):
                    slot_indices[s].append(ids)

            for s_ids in slot_indices:
                permutation = np.random.permutation(range(len(s_ids)))
                ids = []
                for i in permutation:
                    ids.extend(s_ids[i])
                final_indices.extend(indices[ids])
        self.final_indices = final_indices
        return

    def __len__(self):
        return len(self.final_indices)

    def __iter__(self):
        return iter(self.final_indices)
    
    def gen_imbalanced_data(self, class_indices):
        gamma = 1. / self.imb_factor
        img_max = class_indices[0].shape[0]
        if img_max == 980:
            # MNIST
            img_max = 1000
        imb_class_indices = []

        nums = []
        for i in range(self.num_class):
            nums.append(int(img_max * (gamma ** (i / (len(class_indices) - 1.0)))))
        
        if self.class_ratio == "random":
            np.random.shuffle(nums)
        print(nums)

        for cls_idx, (c_ids, num) in enumerate(zip(class_indices, nums)):
            idx = np.arange(c_ids.shape[0])
            np.random.shuffle(idx)
            imb_class_indices.append(c_ids[idx[:num]])
        return imb_class_indices


def build_sampler(
        cfg,
        data_source: List[DatumBase],
        **kwargs
):
    if cfg.LOADER.SAMPLER.TYPE == "temporal":
        return LabelDirichletDomainSequence(data_source, cfg.LOADER.SAMPLER.GAMMA, cfg.TEST.BATCH_SIZE, **kwargs)
    elif cfg.LOADER.SAMPLER.TYPE == "gli_tta":
        return LabelDirichletDomainSequenceLongTailed(data_source, cfg.LOADER.SAMPLER.GAMMA, cfg.TEST.BATCH_SIZE, cfg.LOADER.SAMPLER.IMB_FACTOR, cfg.LOADER.SAMPLER.CLASS_RATIO, **kwargs)
    elif cfg.LOADER.SAMPLER.TYPE == "gradual":
        return GradualDomainSequence(data_source)
    else:
        raise NotImplementedError()
