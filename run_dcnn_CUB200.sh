#!/usr/bin/env bash

### Train
# export CUDA_VISIBLE_DEVICES=0

# OUTPUT='./output/DCNN_CUB200'
# python main.py \
#        --model DCU_SC_SA_MAIN \
#        --data-set CUB_200_2011 \
#        --batch-size 128 \
#        --lr 0.0001 \
#        --min-lr 0.000001 \
#        --num_workers 4 \
#        --data-path /root/autodl-tmp/CUB200-2011/CUB_200_2011/dataset \
#        --output_dir ${OUTPUT} \
#        --epochs 100 \
#        --warmup-epochs 3 \
#        --decay-epochs 20 \
#        --warmup-lr 0.000002 \
#        --mix_loss True \
#        --finetune './output/DCNN/checkpoint_best.pth'




### Inference
CUDA_VISIBLE_DEVICES=0
python main.py --model DCU_SC_SA_MAIN \
               --eval \
               --batch-size 64 \
               --input-size 224 \
               --data-set CUB_200_2011 \
               --num_workers 4 \
               --data-path /root/autodl-tmp/CUB200-2011/CUB_200_2011/dataset \
               --epochs 100 \
               --resume ./output/DCNN_CUB200/checkpoint_best.pth