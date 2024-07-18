import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from argparse import Namespace
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR
from models.TimeMAE import TimeMAE
from layers.TimeMAE_downstream import ClassificationHead
from torcheval.metrics import MulticlassAccuracy


@dataclass
class Metrics:
    loss_sum: float = 0.0
    loss_mcc: float = 0.0
    loss_mrr: float = 0.0
    hits: float = 0.0
    ndcg: float = 0.0
    accuracy: float = -1.0  # -1 when pretraining, >= 0 when pretrain validation

    def __repr__(self):
        _repr = (
            f"Loss Sum: {self.loss_sum:.4f} | "
            f"Loss MCC (CE): {self.loss_mcc:.4f} | "
            f"Loss MRR (MSE): {self.loss_mrr:.4f} | "
            f"Hits: {self.hits:.4f} | "
            f"NDCG@10: {self.ndcg:.4f}"
        )
        if self.accuracy != -1:  # When Validation
            _repr += f" | Accuracy: {self.accuracy:.4f}"
        return _repr


class TimeMAEPretrain:
    def __init__(
            self,
            args: Namespace,
            model: TimeMAE,
            train_loader: DataLoader,
            val_loader: DataLoader,
            task: str = 'classification',
    ):
        self.args = args
        self.model = model.to(args.device)
        self.train_loader = tqdm(train_loader, desc="Training") if args.verbose else train_loader
        self.val_loader = tqdm(val_loader, desc="Validation") if args.verbose else val_loader
        self.task = task

        # Loss Function
        self.mcc_criterion = nn.CrossEntropyLoss()  # MCC means masked codeword classification
        self.mrr_criterion = nn.MSELoss()  # MRR means masked representation regression
        self.alpha = args.alpha
        self.beta = args.beta

        # Training Metrics
        self.num_epochs_pretrain = args.num_epochs_pretrain
        self.eval_per_epochs_pretrain = args.eval_per_epochs_pretrain
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
        self.train_result_save_path = Path(args.save_dir) / 'pretrain_train.csv'
        self.val_result_save_path = Path(args.save_dir) / 'pretrain_val.csv'
        self.model_save_path = Path(args.save_dir) / 'pretrain_model.pth'
        self.train_df = pd.DataFrame(columns=[
            'Epoch',
            'Train Loss',  # Loss = alpha * Loss MCC (CE) + beta * Loss MRR (MSE)
            'Loss MCC (CE)',
            'Loss MRR (MSE)',
            'Hits',
            'NDCG@10'
        ])
        self.train_df.to_csv(self.train_result_save_path, index=False)
        self.val_df = pd.DataFrame(columns=[
            'Epoch',
            'Val Loss',
            'Loss MCC (CE)',
            'Loss MRR (MSE)',
            'Hits',
            'NDCG@10',
            'Accuracy'
        ])
        self.val_df.to_csv(self.val_result_save_path, index=False)

    def __append_to_csv(self, epoch: int, metrics: Metrics, mode: str = 'train'):
        if mode == 'train':
            self.train_df = self.train_df.append({
                'Epoch': epoch,
                'Train Loss': metrics.loss_sum,
                'Loss MCC (CE)': metrics.loss_mcc,
                'Loss MRR (MSE)': metrics.loss_mrr,
                'Hits': metrics.hits,
                'NDCG@10': metrics.ndcg
            }, ignore_index=True)
            self.train_df.to_csv(self.train_result_save_path, index=False)
        elif mode == 'val':
            self.val_df = self.val_df.append({
                'Epoch': epoch,
                'Val Loss': metrics.loss_sum,
                'Loss MCC (CE)': metrics.loss_mcc,
                'Loss MRR (MSE)': metrics.loss_mrr,
                'Hits': metrics.hits,
                'NDCG@10': metrics.ndcg,
                'Accuracy': metrics.accuracy
            }, ignore_index=True)
            self.val_df.to_csv(self.val_result_save_path, index=False)
        else:
            raise ValueError(f"Invalid mode: {mode}, mode should be 'train' or 'val'.")

    def pretrain(self):
        self.model.copy_weight()  # align the weights of the model and momentum model

        best_val_loss = float('inf')
        for epoch in range(self.num_epochs_pretrain):
            train_metrics = self.__train_one_epoch()
            self.__append_to_csv(epoch + 1, train_metrics, mode='train')  # Save result to csv file
            if self.args.verbose:
                print(f"Training Epoch {epoch + 1} | {train_metrics}")
            if (epoch + 1) % self.eval_per_epochs_pretrain == 0:
                val_metrics = self.__val_one_epoch()
                self.__append_to_csv(epoch + 1, val_metrics, mode='val')
                if self.args.verbose:
                    print(f"Validating Epoch {epoch + 1} | {val_metrics}")
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

            loss_mcc = self.mcc_criterion(mask_words_prediction, mask_words)
            loss_mrr = self.mrr_criterion(rep_mask, rep_mask_prediction)
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
    def __val_one_epoch(self) -> Metrics:
        self.model.eval()
        metrics = Metrics()
        for (val_data, labels) in self.val_loader:
            # Pretrain Validation
            ([rep_mask, rep_mask_prediction],
             [mask_words, mask_words_prediction]) = self.model.pretrain_forward(val_data)

            loss_mcc = self.mcc_criterion(mask_words_prediction, mask_words)
            loss_mrr = self.mrr_criterion(rep_mask, rep_mask_prediction)
            loss_sum = self.alpha * loss_mcc + self.beta * loss_mrr
            metrics.loss_mcc += loss_mcc.item()
            metrics.loss_mrr += loss_mrr.item()
            metrics.loss_sum += loss_sum.item()

            # Classification Supervised Validation when pretraining
            if self.task == 'classification':
                pretrain_eval = TimeMAEClassificationForPretrainEval(
                    args=self.args,
                    TimeMAE_encoder=self.model
                )
                pretrain_eval.fit(self.train_loader)
                accuracy = pretrain_eval.score(self.val_loader)
                metrics.accuracy = accuracy

        metrics.loss_mcc /= len(self.val_loader)
        metrics.loss_mrr /= len(self.val_loader)
        metrics.loss_sum /= len(self.val_loader)

        return metrics


@torch.no_grad()
def pretrain_test(
        args: Namespace,
        model: nn.Module,
        test_loader: DataLoader,
):
    model.eval()
    mcc_criterion = nn.CrossEntropyLoss()
    mrr_criterion = nn.MSELoss()
    alpha = args.alpha
    beta = args.beta

    test_loader = tqdm(test_loader, desc="Testing") if args.verbose else test_loader

    metrics = Metrics()
    test_result_save_path = Path(args.save_dir) / 'pretrain_test.csv'
    df = pd.DataFrame(columns=[
        'Test Loss',  # Loss Sum = alpha * Loss MCC (CE) + beta * Loss MRR (MSE)
        'Loss MCC (CE)',
        'Loss MRR (MSE)',
        'Hits',
        'NDCG@10'
    ])
    df.to_csv(test_result_save_path, index=False)

    for (data, _) in test_loader:
        ([rep_mask, rep_mask_prediction],
         [mask_words, mask_words_prediction]) = model.pretrain_forward(data)

        loss_mcc = mcc_criterion(mask_words_prediction, mask_words)
        loss_mrr = mrr_criterion(rep_mask, rep_mask_prediction)
        loss_sum = alpha * loss_mcc + beta * loss_mrr
        metrics.loss_mcc += loss_mcc.item()
        metrics.loss_mrr += loss_mrr.item()
        metrics.loss_sum += loss_sum.item()

    metrics.loss_mcc /= len(test_loader)
    metrics.loss_mrr /= len(test_loader)
    metrics.loss_sum /= len(test_loader)

    df = df.append({
        'Test Loss': metrics.loss_sum,
        'Loss MCC (CE)': metrics.loss_mcc,
        'Loss MRR (MSE)': metrics.loss_mrr,
        'Hits': metrics.hits,
        'NDCG@10': metrics.ndcg
    }, ignore_index=True)
    df.to_csv(test_result_save_path, index=False)

    if args.verbose:
        print(f"Pretrain Test | {metrics}")


class TimeMAEClassificationForPretrainEval(nn.Module):
    def __init__(
            self,
            args: Namespace,
            TimeMAE_encoder: TimeMAE,
    ):
        super(TimeMAEClassificationForPretrainEval, self).__init__()
        self.args = args
        self.TimeMAE_encoder = TimeMAE_encoder
        self.classify_head = ClassificationHead(
            d_model=args.d_model,
            num_classes=args.num_classes
        ).to(args.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.TimeMAE_encoder.parameters(),
            lr=args.lr
        )

        self.accuracy = MulticlassAccuracy()

    def fit(self, train_loader):
        self.model.train()
        for (data, labels) in train_loader:
            self.optimizer.zero_grad()
            with torch.no_grad():
                x = self.TimeMAE_encoder(data, task='classification')  # Don't update TimeMAE_encoder
            outputs = self.classify_head(x)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def score(self, val_loader):
        self.model.eval()
        self.accuracy.reset()
        for (data, labels) in val_loader:
            x = self.TimeMAE_encoder(data, task='classification')
            outputs = self.classify_head(x)
            _, predicted = torch.max(outputs, -1)
            self.accuracy.update(predicted, labels)

        return self.accuracy.compute()
