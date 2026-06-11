#!/usr/bin/env bash

# Train
# export CUDA_VISIBLE_DEVICES=0
# OUTPUT='./output/DCNN_flowers102'
# python main.py \
#        --model DCU_SC_SA_MAIN \
#        --data-set OxfordFlowers \
#        --batch-size 128 \
#        --lr 0.0001 \
#        --num_workers 4 \
#        --data-path /root/autodl-tmp/flowers102/ \
#        --output_dir ${OUTPUT} \
#        --epochs 300 \
#        --warmup-epochs 5 \
#        --decay-epochs 20 \
#        --warmup-lr 0.000002 \
#        --finetune './output/DCNN/checkpoint_best.pth'




# Inference
CUDA_VISIBLE_DEVICES=0
python main.py --model DCU_SC_SA_MAIN \
               --eval \
               --batch-size 64 \
               --input-size 224 \
               --data-set OxfordFlowers \
               --num_workers 4 \
               --data-path /root/autodl-tmp/flowers102/ \
               --epochs 100 \
               --resume ./output/DCNN_flowers102/checkpoint_best.pth