#!/bin/bash
for i in 2
do
  save_path=exp/etth1//$i
  python main.py \
  --save_path $save_path \
  --data_path data/ETT-small/ETTh1/ \
  --dataset ett \
  --mask_ratio 0.6 \
  --vocab_size 192 \
  --device cuda \
  --wave_length 8 \
  --beta 1.0 \
  --alpha 5.0 \
  --layers 8 \
  --d_model 64 \
  --num_epoch_pretrain 100 \
  --num_epoch 100 \
  --load_pretrained_model 1
done

