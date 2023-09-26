CUDA_VISIBLE_DEVICES=0 python GLI_TTA.py \
      -acfg configs/adapter/TRIBE.yaml \
      -dcfg configs/dataset/cifar10.yaml \
      -pcfg configs/protocol/gli_tta.yaml \
      OUTPUT_DIR TRIBE/cifar10