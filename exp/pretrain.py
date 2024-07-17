import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from argparse import Namespace
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class Metrics:
    loss_sum: float = 0
    loss_mcc: float = 0
    loss_mrr: float = 0
    hits: float = 0
    ndcg: float = 0


class TimeMAEPretrain:
    def __init__(
            self,
            args: Namespace,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
    ):
        self.args = args
        self.model = model.to(args.device)
        self.train_loader = tqdm(train_loader, desc="Training") if args.verbose else train_loader
        self.val_loader = tqdm(val_loader, desc="Validation") if args.verbose else val_loader

        # Loss Function
        self.loss_mcc_fn = nn.CrossEntropyLoss()  # MCC means masked codeword classification
        self.loss_mrr_fn = nn.MSELoss()  # MRR means masked representation regression
        self.alpha = args.alpha
        self.beta = args.beta

        # Training Metrics
        self.num_epochs_pretrain = args.num_epochs_pretrain
        self.eval_per_epochs = args.eval_per_epochs
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: args.lr_decay ** step,
            verbose=args.verbose
        )

        # Save result
        self.train_result_save_path = Path(args.save_dir) / 'pretrain_train_result.csv'
        self.val_result_save_path = Path(args.save_dir) / 'pretrain_val_result.csv'
        self.model_save_path = Path(args.save_dir) / 'pretrain_model.pth'
        self.df = pd.DataFrame(columns=[
            'Epoch',
            'Loss Sum',  # Loss Sum = alpha * Loss MCC (CE) + beta * Loss MRR (MSE)
            'Loss MCC (CE)',
            'Loss MRR (MSE)',
            'Hits',
            'NDCG@10'
        ])
        self.df.to_csv(self.train_result_save_path, index=False)
        self.df.to_csv(self.val_result_save_path, index=False)

    def __append_df(self, epoch, metrics: Metrics, mode: str = 'train'):
        self.df = self.df.append({
            'Epoch': epoch,
            'Loss Sum': metrics.loss_sum,
            'Loss MCC (CE)': metrics.loss_mcc,
            'Loss MRR (MSE)': metrics.loss_mrr,
            'Hits': metrics.hits,
            'NDCG@10': metrics.ndcg
        }, ignore_index=True)
        if mode == 'train':
            self.df.to_csv(self.train_result_save_path, index=False)
        elif mode == 'val':
            self.df.to_csv(self.val_result_save_path, index=False)
        else:
            raise ValueError(f"Invalid mode: {mode}, mode should be 'train' or 'val'.")

    def __print_metrics(self, epoch, metrics):
        if self.args.verbose:
            _metrics = f"""Pretrain Epoch {epoch} | \
                        Loss Sum: {metrics.loss_sum:.4f} | \
                        Loss MCC (CE): {metrics.loss_mcc:.4f} | \
                        Loss MRR (MSE): {metrics.loss_mrr:.4f} | \
                        Hits: {metrics.hits:.4f} | \
                        NDCG@10: {metrics.ndcg:.4f}"""
            print(_metrics)

    def pretrain(self):
        self.model.copy_weight()  # align the weights of the model and momentum model

        best_val_loss = float('inf')
        for epoch in range(self.num_epochs_pretrain):
            train_metrics = self.__train_one_epoch()
            self.__append_df(epoch + 1, train_metrics, mode='train')  # Save result to csv file
            self.__print_metrics(epoch + 1, train_metrics)  # Print result if verbose
            if (epoch + 1) % self.eval_per_epochs == 0:
                val_metrics = self.__validate_one_epoch()
                self.__append_df(epoch + 1, val_metrics, mode='val')
                self.__print_metrics(epoch + 1, val_metrics)
                if val_metrics.loss_sum < best_val_loss:
                    best_val_loss = val_metrics.loss_sum
                    torch.save(self.model.state_dict(), self.model_save_path)

    def __train_one_epoch(self) -> Metrics:
        self.model.train()
        metrics = Metrics()
        for (data, _) in self.train_loader:
            self.optimizer.zero_grad()

            ([rep_mask, rep_mask_prediction],
             [mask_words, mask_words_prediction]) = self.model.pretrain_forward(data)

            loss_mcc = self.loss_mcc_fn(mask_words_prediction, mask_words)
            loss_mrr = self.loss_mrr_fn(rep_mask, rep_mask_prediction)
            loss_sum = self.alpha * loss_mcc + self.beta * loss_mrr
            metrics.loss_mcc += loss_mcc.item()
            metrics.loss_mrr += loss_mrr.item()
            metrics.loss_sum += loss_sum.item()

            loss_sum.backward()
            self.optimizer.step()

            self.model.momentum_update()  # update momentum model

        metrics.loss_mcc /= len(self.train_loader)
        metrics.loss_mrr /= len(self.train_loader)
        metrics.loss_sum /= len(self.train_loader)

        self.scheduler.step()

        return metrics

    @torch.no_grad()
    def __validate_one_epoch(self) -> Metrics:
        self.model.eval()
        metrics = Metrics()
        for (data, _) in self.val_loader:
            ([rep_mask, rep_mask_prediction],
             [mask_words, mask_words_prediction]) = self.model.pretrain_forward(data)

            loss_mcc = self.loss_mcc_fn(mask_words_prediction, mask_words)
            loss_mrr = self.loss_mrr_fn(rep_mask, rep_mask_prediction)
            loss_sum = self.alpha * loss_mcc + self.beta * loss_mrr
            metrics.loss_mcc += loss_mcc.item()
            metrics.loss_mrr += loss_mrr.item()
            metrics.loss_sum += loss_sum.item()

        metrics.loss_mcc /= len(self.val_loader)
        metrics.loss_mrr /= len(self.val_loader)
        metrics.loss_sum /= len(self.val_loader)

        return metrics
