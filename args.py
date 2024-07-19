import argparse
import torch
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='TimeMAE Args')

# Input/Output Directory
parser.add_argument('--data_dir', type=str, default='data_provider', help='data directory')
parser.add_argument('--save_dir', type=str, default='results', help='save results or models directory')

# DataLoader
parser.add_argument('--train_batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--val_batch_size', type=int, default=32, help='batch size for validation')
parser.add_argument('--test_batch_size', type=int, default=32, help='batch size for testing')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for DataLoader')

# Model Hyperparameters
# Transformer Encoder
parser.add_argument('--vocab_size', type=int, default=512, help='vocab size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--nhead', type=int, default=8, help='number of heads')
parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of feedforward')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--num_layers', type=int, default=6, help='number of layers')
parser.add_argument('--enable_res_param', type=bool, default=False, help='enable residual parameter')
# Conv1D
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size')
parser.add_argument('--num_features', type=int, default=7, help='number of features')
# Decoupled Transformer Encoder
parser.add_argument('--decouple_num_layers', type=int, default=2, help='number of decoupled layers')
# Momentum Transformer Encoder
parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')

# Pretrain
parser.add_argument('--task', type=str, default='classification', help='task type')
parser.add_argument('--mask_ratio', type=float, default=0.6, help='mask ratio')
parser.add_argument('--verbose', type=bool, default=True, help='verbose')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
parser.add_argument('--beta', type=float, default=0.1, help='beta')
parser.add_argument('--num_epochs_pretrain', type=int, default=100, help='number of epochs for pretrain')
parser.add_argument('--eval_per_epochs_pretrain', type=int, default=1, help='evaluation per epochs for pretrain')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

# Finetune
parser.add_argument('--finetune_mode', type=str, default='fine_all', help='finetune mode')
parser.add_argument('--num_epochs_finetune', type=int, default=100, help='number of epochs for finetune')
parser.add_argument('--eval_per_epochs_finetune', type=int, default=1, help='evaluation per epochs for finetune')
# Classification
parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
# Forecasting
parser.add_argument('--seq_len', type=int, default=128, help='sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction length')

# Device
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')

args = parser.parse_args()

args.eval_per_epochs_pretrain = args.eval_per_epochs_pretrain \
    if 0 < args.eval_per_epochs_pretrain < args.num_epochs_pretrain else 1
args.eval_per_epochs_finetune = args.eval_per_epochs_finetune \
    if 0 < args.eval_per_epochs_finetune < args.num_epochs_finetune else 1

# mkdir save_dir
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# save args
with open(save_dir / 'args.json', 'w') as f:
    json.dump(vars(args), f, indent=4)

# mkdir pretrain, finetune save_dir
pretrain_save_dir = save_dir / 'pretrain'
pretrain_save_dir.mkdir(parents=True, exist_ok=True)

if args.task == 'classification':
    finetune_save_dir = save_dir / 'classification_finetune'
    finetune_save_dir.mkdir(parents=True, exist_ok=True)
elif args.task == 'forecasting':
    finetune_save_dir = save_dir / 'forecasting_finetune'
    finetune_save_dir.mkdir(parents=True, exist_ok=True)

# Device
args.device = torch.device(args.device)
