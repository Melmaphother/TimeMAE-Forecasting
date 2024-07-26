#!/bin/bash

cd ../../

python -u run.py \
  --data_dir data_provider \
  --save_dir results/har/1 \
  --train_batch_size 128 \
  --val_batch_size 128 \
  --test_batch_size 128 \
  --num_workers 4 \
  --seq_len 128 \
  --num_features 9 \
  --vocab_size 192 \
  --d_model 64 \
  --nhead 4 \
  --dim_feedforward 256 \
  --dropout 0.2 \
  --num_layers 8 \
  --enable_res_param True \
  --kernel_size 8 \
  --num_layers_decoupled 4 \
  --momentum 0.99 \
  --task classification \
  --mask_ratio 0.6 \
  --verbose True \
  --alpha 0.5 \
  --beta 0.1 \
  --num_epochs_pretrain 100 \
  --eval_per_epochs_pretrain 1 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --finetune_mode fine_all \
  --num_epochs_finetune 100 \
  --eval_per_epochs_finetune 1 \
  --num_classes 6 \
  --pred_len 96 \
  --device cuda
