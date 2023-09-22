from tqdm import tqdm
import os
import logging
import torch
import argparse
import torch.utils as utils
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.datasets import MNIST
from torchvision import transforms
from setproctitle import setproctitle

from core.configs import cfg
from core.utils import *
from core.model import build_model
from core.optim import build_optimizer


def train(train_loader, model, optimizer, epoch, device, logger):
    train_accuracy, train_loss = 0., 0.
    model.train()
    logger.info('\nTrain start')
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        out = model(images)

        loss = F.cross_entropy(out, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        preds = out.argmax(dim=-1)

        train_accuracy += torch.sum(preds==labels).item() / len(labels)
    
    logger.info(f"epoch: {epoch+1}")
    logger.info(f"train_loss: {train_loss / len(train_loader)}")
    logger.info(f"train_accuracy: {train_accuracy / len(train_loader)}")

    return train_loss, train_accuracy

def val(test_loader, model, epoch, device, logger):
    correct_list = []
    
    model.eval()
    logger.info('\nEval start')
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            out = model(images)

            correct_list.append(out.argmax(dim=-1).eq(labels))
        
    logger.info(f"epoch: {epoch+1}")
    logger.info(f"validation_accuracy: {torch.concat(correct_list).float().mean()}")
    return torch.concat(correct_list).float().mean().item()



def Trainer(cfg):
    logger = logging.getLogger("TRAINER.train_time")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(cfg)
    model.to(device)
    
    optimizer = build_optimizer(cfg)(model.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(root=cfg.DATA_DIR, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=cfg.DATA_DIR, train=False, transform=transform, download=False)

    train_dataloader = data.DataLoader(train_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE)

    train_epoch = 100

    best_val = 0.
    for epoch in range(train_epoch):
        train(train_dataloader, model, optimizer, epoch, device, logger)
        scheduler.step(epoch=epoch)

        val_res = val(test_dataloader, model, epoch, device, logger)
        
        if val_res >= best_val:
            best_val = val_res
            if not os.path.exists('./ckpt/mnist/source'):
                os.makedirs('./ckpt/mnist/source/', exist_ok=True)
            torch.save(model.state_dict(), './ckpt/mnist/source/resnet18.pth')
        







def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)

    args = parser.parse_args()

    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.dataset_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ds = cfg.CORRUPTION.DATASET

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('TRAINER', cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)

    set_random_seed(cfg.SEED)

    Trainer(cfg)



if __name__ == "__main__":
    main()
