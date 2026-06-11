#!/usr/bin/env bash


# Train
export CUDA_VISIBLE_DEVICES=0

OUTPUT='./output/DCNN'
python -m torch.distributed.launch --master_port 50137 --nproc_per_node=1 main.py \
       --model DCU_SC_SA_MAIN \
       --data-set IMNET \
       --batch-size 128 \
       --lr 0.001 \
       --num_workers 4 \
       --data-path /root/autodl-tmp/imagenet/ \
       --output_dir ${OUTPUT} \
       --epochs 300 \
       --warmup-epochs 3 \
       --decay-epochs 20 \
       --warmup-lr 0.000002
