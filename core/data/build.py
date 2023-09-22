from .datasets.common_corruption import CorruptionCIFAR
from .datasets.common_corruption import CorruptionImageNet
from .datasets.common_corruption import CorruptionMNIST
from .datasets.common_corruption import GradualCorruptionCIFAR
from .datasets.toy_dataset import ToyDataset
from .ttasampler import build_sampler
from torch.utils.data import DataLoader
from ..utils.result_precess import AvgResultProcessor


def build_loader(cfg, ds_name, all_corruptions, all_severity):
    if ds_name == "cifar10" or ds_name == "cifar100":
        dataset_class = CorruptionCIFAR
    elif ds_name == "imagenet":
        dataset_class = CorruptionImageNet
    elif ds_name == "mnist":
        dataset_class = CorruptionMNIST
    elif ds_name == "gradualCifar10" or ds_name == "gradualCifar100":
        dataset_class = GradualCorruptionCIFAR
    elif ds_name == "toy":
        dataset_class = ToyDataset
    else:
        raise NotImplementedError(f"Not Implement for dataset: {cfg.CORRUPTION.DATASET}")

    ds = dataset_class(cfg, all_corruptions, all_severity)
    sampler = build_sampler(cfg, ds.data_source)

    loader = DataLoader(ds, cfg.TEST.BATCH_SIZE, sampler=sampler, num_workers=cfg.LOADER.NUM_WORKS)

    result_processor = AvgResultProcessor(ds.domain_id_to_name)

    return loader, result_processor
