import torch
import numpy as np
from pathlib import Path
from torchvision import transforms

PREPROCESSINGS = {
    'Res256Crop224':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'Crop288':
    transforms.Compose([transforms.CenterCrop(288),
                        transforms.ToTensor()]),
    None:
    transforms.Compose([transforms.ToTensor()]),
    'BicubicRes256Crop224':
    transforms.Compose([
        transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode("bicubic")),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
}

CORRUPTIONS = ("shot_noise", "impulse_noise", "glass_blur", "motion_blur",
               "shear", "scale", "rotate", "brightness",
               "translate", "stripe", "fog", "spatter", "dotted_line",
               "zigzag", "canny_edges")

def load_mnistc(
    data_dir = './data',
    shuffle = False,
    corruptions = CORRUPTIONS
):
    data_dir = Path(data_dir)
    data_root_dir = data_dir / "mnist_c"

    x_test_list, y_test_list = [], []
    for corruption in corruptions:
        corruption_image_path = data_root_dir / corruption / 'test_images.npy'
        corruption_label_path = data_root_dir / corruption / 'test_labels.npy'

        images_all = np.load(corruption_image_path)
        labels_all = np.load(corruption_label_path)

        x_test_list.append(images_all)
        y_test_list.append(labels_all)

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255.
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)

    return x_test, y_test


