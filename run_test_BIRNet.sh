#!/usr/bin/env bash


# Inference
CUDA_VISIBLE_DEVICES=0
python main.py  --model BIRU_SC_SA_MAIN \
                --eval \
                --batch-size 64 \
                --input-size 224 \
                --data-set IMNET \
                --num_workers 4 \
                --data-path /root/autodl-tmp/imagenet/ \
                --epochs 100 \
               --resume './output/BIRNet/checkpoint_best.pth'
