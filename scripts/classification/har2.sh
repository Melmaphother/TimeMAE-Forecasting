#!/bin/bash

cd ../../

python -u run.py \
  --data_dir data_provider \
  --save_dir results/har/4 \
  --train_batch_size 128 \
  --val_batch_size 128 \
  --test_batch_size 128 \
  --scale False \
  --seq_len 128 \
  --num_features 9 \
  --vocab_size 192 \
  --d_model 64 \
  --nhead 4 \
  --dim_feedforward 256 \
  --dropout 0.4 \
  --num_layers 8 \
  --enable_res_param True \
  --kernel_size 8 \
  --num_layers_decoupled 4 \
  --momentum 0.99 \
  --task classification \
  --mask_ratio 0.6 \
  --verbose True \
  --alpha 1.0 \
  --beta 10.0 \
  --num_epochs_pretrain 100 \
  --eval_per_epochs_pretrain 5 \
  --lr 0.0005 \
  --weight_decay 0.001 \
  --lr_decay 0.99 \
  --finetune_mode fine_all \
  --num_epochs_finetune 100 \
  --eval_per_epochs_finetune 5 \
  --num_classes 6 \
  --pred_len 96 \
  --device cuda:5
