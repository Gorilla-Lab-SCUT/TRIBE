# The released code of TRIBE

# Preparation

## Installation
```bash
conda create -n tribe python=3.9.0
conda activate tribe

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

# this installs required packages
pip install -r .

# build Balanced BN
cd cpp_wrapper/balanced_bn
pip install .
```

## Datasets Preparation

Download [CIFAR-10-C](https://zenodo.org/record/2535967#.ZDETTHZBxhF) and [CIFAR-100-C](https://zenodo.org/record/3555552#.ZDES-XZBxhE). (Running the code directly also works, since it automatically downloads the data set at the first running, but it's too slow to tolerate and has high requirements on internet stability)

Symlink dataset by
```bash
ln -s path_to_cifar10_c datasets/CIFAR-10-C
ln -s path_to_cifar100_c datasets/CIFAR-100-C
```

## Code Running
Run TRIBE by
```bash
python GLI_TTA.py \
      -acfg configs/adapter/TRIBE.yaml \
      -dcfg configs/dataset/cifar10.yaml \
      -pcfg configs/protocol/gli_tta.yaml \
      OUTPUT_DIR TRIBE/cifar10

python GLI_TTA.py \
      -acfg configs/adapter/TRIBE.yaml \
      -dcfg configs/dataset/cifar100.yaml \
      -pcfg configs/protocol/gli_tta.yaml \
      OUTPUT_DIR TRIBE/cifar100
```

Hint: The hyper-parameters may be modified in `./configs/adapter/TRIBE.yaml`, and please modify them according to the suggestions written into the file.

## Acknowledgements
This project is based on the following open-source projects: [rotta](https://github.com/BIT-DA/RoTTA). We thank their authors for making the source code publicly available.

