#!/usr/bin/env bash

# Train
# export CUDA_VISIBLE_DEVICES=0

# OUTPUT='./output/DCNN_Food101'
# python main.py \
#        --model DCU_SC_SA_MAIN \
#        --data-set Food101 \
#        --batch-size 128 \
#        --lr 0.0001 \
#        --min-lr 0.000001 \
#        --num_workers 4 \
#        --data-path /root/autodl-tmp/Food101/ \
#        --output_dir ${OUTPUT} \
#        --epochs 200 \
#        --warmup-epochs 3 \
#        --decay-epochs 20 \
#        --warmup-lr 0.000002 \
#        --finetune './output/DCNN/checkpoint_best.pth'




# Inference
CUDA_VISIBLE_DEVICES=0
python main.py --model DCU_SC_SA_MAIN \
               --eval \
               --batch-size 64 \
               --input-size 224 \
               --data-set Food101 \
               --num_workers 4 \
               --data-path /root/autodl-tmp/Food101/ \
               --epochs 100 \
               --resume ./output/DCNN_Food101/checkpoint_best.pth