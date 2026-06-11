#!/usr/bin/env bash

# Train
# export CUDA_VISIBLE_DEVICES=0
# OUTPUT='./output/DCNN_StanfordDogs'
# python main.py \
#        --model DCU_SC_SA_MAIN \
#        --data-set StanfordDogs \
#        --batch-size 128 \
#        --lr 0.0001 \
#        --min-lr 0.000001 \
#        --num_workers 4 \
#        --data-path /root/autodl-tmp/StanfordDogs/data/dataset \
#        --output_dir ${OUTPUT} \
#        --epochs 100 \
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
               --data-set StanfordDogs \
               --num_workers 4 \
               --data-path /root/autodl-tmp/StanfordDogs/data/dataset \
               --epochs 100 \
               --resume ./output/DCNN_StanfordDogs/checkpoint_best.pth