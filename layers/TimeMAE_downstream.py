import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_classes: int,
    ):
        super(ClassificationHead, self).__init__()
        self.classify_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
        Returns:
            output tensor, shape (batch_size, num_classes)
        """
        x = x.mean(dim=1)
        x = self.classify_head(x)  # (batch_size, num_classes)
        return x


class ForecastingHead(nn.Module):
    def __init__(
            self,
            seq_len: int,
            d_model: int,
            pred_len: int,
            num_features: int,
    ):
        super(ForecastingHead, self).__init__()
        self.pred_len = pred_len
        self.num_features = num_features
        self.flatten = nn.Flatten()
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len * num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
        Returns:
            output tensor, shape (batch_size, pred_len, num_features)
        """
        x = self.flatten(x)  # (batch_size, seq_len * d_model)
        x = self.forecast_head(x)  # (batch_size, pred_len * num_features)
        x = x.view(-1, self.pred_len, self.num_features)  # (batch_size, pred_len, num_features)
        return x
