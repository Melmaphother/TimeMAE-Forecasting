#!/bin/bash

cd ../../

# 参数配置
pred_lens=(96 192 336 720)
configs=(
  "0 0"  # need_pretrain=1, fine_zero=0
  "0 1"
)

# 循环遍历所有参数组合

for pred_len in "${pred_lens[@]}"; do
  i=17
  for config in "${configs[@]}"; do
    need_pretrain=$(echo $config | cut -d ' ' -f 1)
    fine_zero=$(echo $config | cut -d ' ' -f 2)

    python -u run.py \
      --data_dir data_provider \
      --dataset ETTh2 \
      --save_dir results/ETTh2/${pred_len}/${i} \
      --train_batch_size 16 \
      --val_batch_size 16 \
      --test_batch_size 16 \
      --scale 1 \
      --seq_len 336 \
      --num_features 7 \
      --vocab_size 64 \
      --d_model 8 \
      --nhead 8 \
      --dim_feedforward 32 \
      --dropout 0.4 \
      --num_layers 2 \
      --enable_res_param 1 \
      --kernel_size 16 \
      --stride 16 \
      --num_layers_decoupled 1 \
      --momentum 0.99 \
      --need_pretrain ${need_pretrain} \
      --task forecasting \
      --mask_ratio 0.5 \
      --verbose 1 \
      --alpha 1.0 \
      --beta 5.0 \
      --num_epochs_pretrain 50 \
      --eval_per_epochs_pretrain 5 \
      --pretrain_lr 0.001 \
      --finetune_lr 0.0001 \
      --weight_decay 0.001 \
      --lr_decay 0.99 \
      --fine_zero ${fine_zero} \
      --finetune_mode fine_all \
      --num_epochs_finetune 10 \
      --eval_per_epochs_finetune 1 \
      --num_classes 6 \
      --pred_len ${pred_len} \
      --device cuda:1
    
    i=$((i + 1))
  done
done