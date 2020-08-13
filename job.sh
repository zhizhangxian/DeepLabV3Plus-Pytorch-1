#!/bin/bash
#BSUB -q gpu_v100
#BSUB -m gpu07
#BSUB -gpu num=1
CUDA_VISIBLE_DEVICES=0 python main.py --model deeplabv3plus_resnet101 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
