import pandas as pd
import torch
import torch.nn as nn
from argparse import Namespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from models.TimeMAE import TimeMAEForecastingForFinetune


@dataclass
class Metrics:
    loss_mse: float = 0.0
    loss_mae: float = 0.0

    def __repr__(self):
        _repr = (
            f"MSE: {self.loss_mse:.4f} | "
            f"MAE: {self.loss_mae:.4f}"
        )
        return _repr


class ForecastingFinetune:
    def __init__(
            self,
            args: Namespace,
            model: TimeMAEForecastingForFinetune,
            train_loader: DataLoader,
            val_loader: DataLoader,
            save_dir: Path,
    ):
        self.args = args
        self.verbose = args.verbose
        self.model = model.to(args.device)
        self.train_loader = tqdm(train_loader, desc="Finetune Training") if self.verbose else train_loader
        self.val_loader = tqdm(val_loader, desc="Finetune Validation") if self.verbose else val_loader
        self.finetune_mode = args.finetune_mode

        # Training Metrics
        self.num_epochs_finetune = args.num_epochs_finetune
        self.eval_per_epochs_finetune = args.eval_per_epochs_finetune
        self.mse_criterion = nn.MSELoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: args.lr_decay ** step
        )

        # Evaluation Metrics
        self.mae_criterion = nn.L1Loss()
        # Save Path
        self.train_result_save_path = save_dir / "finetune_train.csv"
        self.val_result_save_path = save_dir / "finetune_val.csv"
        self.model_save_path = save_dir / "finetune_model.pth"
        self.train_df = pd.DataFrame(columns=[
            'Epoch',
            'Loss MSE (Train)',
            'MAE'
        ])
        self.train_df.to_csv(self.train_result_save_path, index=False)
        self.val_df = pd.DataFrame(columns=[
            'Epoch',
            'Loss MSE (Val)',
            'MAE'
        ])
        self.val_df.to_csv(self.val_result_save_path, index=False)

    def __append_to_csv(self, epoch: int, metrics: Metrics, mode: str = 'train'):
        if mode == 'train':
            self.train_df = self.train_df.append({
                'Epoch': epoch,
                'Loss MSE (Train)': metrics.loss_mse,
                'MAE': metrics.loss_mae,
            }, ignore_index=True)
            self.train_df.to_csv(self.train_result_save_path, index=False)
        elif mode == 'val':
            self.val_df = self.val_df.append({
                'Epoch': epoch,
                'Loss MSE (Val)': metrics.loss_mse,
                'MAE': metrics.loss_mae,
            }, ignore_index=True)
            self.val_df.to_csv(self.val_result_save_path, index=False)
        else:
            raise ValueError(f"Invalid mode: {mode}, mode should be 'train' or 'val'.")

    def finetune(self):
        best_val_loss_mse = float('inf')  # Use forecasting MSE as the metric to select the best model
        for epoch in range(self.num_epochs_finetune):
            train_metrics = self.__train_one_epoch()
            self.__append_to_csv(epoch, train_metrics, mode='train')
            if self.verbose:
                print(f"Forecasting Finetune Training Epoch {epoch + 1} | {train_metrics}")
            if (epoch + 1) % self.eval_per_epochs_finetune == 0:
                val_metrics = self.__val_one_epoch()
                self.__append_to_csv(epoch, val_metrics, mode='val')
                if self.verbose:
                    print(f"Forecasting Finetune Validation Epoch {epoch + 1} | {val_metrics}")
                if val_metrics.loss_mse < best_val_loss_mse:
                    best_val_loss_mse = val_metrics.loss_mse
                    torch.save(self.model.state_dict(), self.model_save_path)

    def __train_one_epoch(self) -> Metrics:
        self.model.train()
        metrics = Metrics()

        for (data, labels) in self.train_loader:
            self.optimizer.zero_grad()

            outputs = self.model(data, finetune_mode=self.finetune_mode)
            loss_mse = self.mse_criterion(outputs, labels)
            loss_mae = self.mae_criterion(outputs, labels)
            metrics.loss_mse += loss_mse.item()
            metrics.loss_mae += loss_mae.item()

            loss_mse.backward()
            self.optimizer.step()

        metrics.loss_mse /= len(self.train_loader)
        metrics.loss_mae /= len(self.train_loader)

        self.scheduler.step()

        return metrics

    @torch.no_grad()
    def __val_one_epoch(self) -> Metrics:
        self.model.eval()
        metrics = Metrics()

        for (data, labels) in self.val_loader:
            outputs = self.model(data, finetune_mode=self.finetune_mode)
            loss_mse = self.mse_criterion(outputs, labels)
            loss_mae = self.mae_criterion(outputs, labels)
            metrics.loss_mse += loss_mse.item()
            metrics.loss_mae += loss_mae.item()

        metrics.loss_mse /= len(self.val_loader)
        metrics.loss_mae /= len(self.val_loader)

        return metrics


@torch.no_grad()
def forecasting_finetune_test(
        args: Namespace,
        model: TimeMAEForecastingForFinetune,
        test_loader: DataLoader,
        save_dir: Path
):
    model.to(args.device)
    model.eval()
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    test_loader = tqdm(test_loader, desc="Finetune Testing") if args.verbose else test_loader

    metrics = Metrics()
    test_result_save_path = save_dir / "finetune_test.csv"
    df = pd.DataFrame(columns=[
        'Loss MSE (Test)',
        'MAE'
    ])
    df.to_csv(test_result_save_path, index=False)

    for (data, labels) in test_loader:
        outputs = model(data, finetune_mode=args.finetune_mode)
        loss_mse = mse_criterion(outputs, labels)
        loss_mae = mae_criterion(outputs, labels)
        metrics.loss_mse += loss_mse.item()
        metrics.loss_mae += loss_mae.item()

    metrics.loss_mse /= len(test_loader)
    metrics.loss_mae /= len(test_loader)

    df.append({
        'Loss MSE (Test)': metrics.loss_mse,
        'MAE': metrics.loss_mae
    }, ignore_index=True)
    df.to_csv(test_result_save_path, index=False)

    if args.verbose:
        print(f"Forecasting Finetune Test | {metrics}")
