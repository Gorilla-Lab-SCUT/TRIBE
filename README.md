<h1 style='text-align:center'>Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization</h1>

<h3 style='text-align:center'>
<a href='https://yysu.site'>Yongyi Su<sup>1</sup></a>,
<a href='https://alex-xun-xu.github.io'>Xun Xu<sup>2</sup></a>, 
<a href='http://kuijia.site'>Kui Jia<sup>1</sup></a>
</h3>

<h3 style='text-align:center'>
<p>[1] South China Univeristy of Technology</p>
<p>[2] Institute for Infocomm Research, A*STAR</p>
</h3>

<h3 style='text-align:left'>
<p>Paper: <a href=''>ArXiv version</a>.
</h3>

## Preparation

### Installation
```bash
conda create -n tribe python=3.9.0
conda activate tribe

# install pip and dependencies for the fresh python
conda install -y ipython pip

# install required packages
pip install -r .

# install robustbench
cd robustbench
pip install .
cd -

# build Balanced BN
cd cpp_wrapper/balanced_bn
pip install .
cd -
```

### Datasets Preparation

Download [CIFAR-10-C](https://zenodo.org/record/2535967#.ZDETTHZBxhF), [CIFAR-100-C](https://zenodo.org/record/3555552#.ZDES-XZBxhE) and [ImageNet-C](https://zenodo.org/record/2235448). (Running the code directly also works, since it automatically downloads the data set at the first running, but it's too slow to tolerate and has high requirements on internet stability)

Symlink dataset by
```bash
ln -s path_to_cifar10_c datasets/CIFAR-10-C
ln -s path_to_cifar100_c datasets/CIFAR-100-C
ln -s path_to_imagenet_c datasets/ImageNet-C
```

## Code Running

### Evaluate TRIBE on three datasets under GLI-TTA protocols
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

python GLI_TTA.py \
      -acfg configs/adapter/TRIBE.yaml \
      -dcfg configs/dataset/imagenet.yaml \
      -pcfg configs/protocol/gli_tta.yaml \
      OUTPUT_DIR TRIBE/imagenet
```

Hint: The hyper-parameters may be modified in `./configs/adapter/TRIBE.yaml`, and please modify them according to the suggestions written into the file.


### More Implementations

Apart from the TRIBE implementation, this repo has also implemented multiple mainstream TTA algorithms and TTA protocols so that you can reproduce their results simply by modifying the running command. Algorithms include `BN`, `PL`, `TENT`, `LAME`, `EATA`, `NOTE`, `TTAC` (without queue), `COTTA`, `PETAL` and `ROTTA`. TTA protocols include `Single Domain TTA`, `Continual TTA`, `Gradual Changing Continual TTA`, `PTTA` (proposed in ROTTA) and `GLI TTA` (proposed in this paper).

For example:
if we want to run `ROTTA` under `Continual TTA` protocol, we can run:

```
python GLI_TTA.py \
      -acfg configs/adapter/rotta.yaml \
      -dcfg configs/dataset/cifar10.yaml \
      -pcfg configs/protocol/continual_tta.yaml \
      OUTPUT_DIR ROTTA/cifar10
```

Or run `TRIBE` under `Gradual Changing Continual TTA` protocol, as

```
python GLI_TTA.py \
      -acfg configs/adapter/TRIBE.yaml \
      -dcfg configs/dataset/gradualCifar10.yaml \
      -pcfg configs/protocol/continual_tta.yaml \
      OUTPUT_DIR TRIBE/cifar10
```

Or under `Single Domain TTA` protocol, here need to modify the CORRUPTION.TYPE to one specific domain in `./configs/dataset/cifar10.yaml` and run:

```
python GLI_TTA.py \
      -acfg configs/adapter/TRIBE.yaml \
      -dcfg configs/dataset/cifar10.yaml \
      -pcfg configs/protocol/continual_tta.yaml \
      OUTPUT_DIR TRIBE/cifar10
```

Or under `PTTA` protocol, as

```
python GLI_TTA.py \
      -acfg configs/adapter/TRIBE.yaml \
      -dcfg configs/dataset/cifar10.yaml \
      -pcfg configs/protocol/ptta.yaml \
      OUTPUT_DIR TRIBE/cifar10
```

In addition to the above simple switching configurations, we can also make fine adjustments in different profiles, such as adjusting different category imbalance ratios in GLI-TTA protocols, as

```
LOADER:
  SAMPLER:
    TYPE: "gli_tta"
    IMB_FACTOR: 10           # global imbalance factor: 10, 100, 200
    CLASS_RATIO: "constant"  # "constant" for GLI-TTA-F or "random" for GLI-TTA-V
    GAMMA: 0.1               # local imbalance factor: 10, 1.0, 0.1, 0.01, 0.001
```

## Acknowledgements
This project is based on the following open-source projects: [rotta](https://github.com/BIT-DA/RoTTA). We thank their authors for making the source code publicly available.


## Citation
If you find our work useful in your research, please consider citing:

```
```
