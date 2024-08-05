import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from argparse import Namespace
from layers.RevIN import RevIN
from layers.TimeMAE_backbone import (
    FeatureExtractor,
    CodeBook,
    PositionalEncoding,
    TransformerEncoder,
    TransformerDecoupledEncoder,
)
from layers.TimeMAE_downstream import (
    ClassificationHead,
    ForecastingHead,
)


class TimeMAE(nn.Module):
    def __init__(
        self,
        args: Namespace,
        origin_seq_len: int,
        num_features: int,
    ):
        super(TimeMAE, self).__init__()

        # For Model Hyperparameters
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.num_layers_decoupled = args.num_layers_decoupled
        self.enable_res_param = args.enable_res_param
        self.device = args.device

        # For Loss Hyperparameters
        self.momentum = args.momentum

        # Feature Extractor: Conv1D
        self.origin_seq_len = origin_seq_len
        self.num_features = num_features
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.feature_extractor = FeatureExtractor(
            num_features=1,
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.seq_len = int((self.origin_seq_len - self.kernel_size) / self.stride) + 1

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            max_len=self.seq_len, d_model=self.d_model, dropout=self.dropout
        )

        # TimeMAE Encoder
        self.mask_len = int(args.mask_ratio * self.seq_len)
        self.replaced_mask = nn.Parameter(
            torch.randn(
                self.d_model,
            )
        )
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            num_layers=self.num_layers,
            enable_res_param=self.enable_res_param,
            dropout=self.dropout,
        )

        # TimeMAE Decoupled Encoder
        self.decoupled_encoder = TransformerDecoupledEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            num_layers=self.num_layers_decoupled,
            enable_res_param=self.enable_res_param,
            dropout=self.dropout,
        )

        # For Calculate MRR Loss
        self.momentum_encoder = TransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            num_layers=self.num_layers,
            enable_res_param=self.enable_res_param,
            dropout=self.dropout,
        )

        # For Calculate MCC Loss
        self.code_book = CodeBook(d_model=self.d_model, vocab_size=self.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for param_a, param_b in zip(
                self.encoder.parameters(), self.momentum_encoder.parameters()
            ):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for param_a, param_b in zip(
                self.encoder.parameters(), self.momentum_encoder.parameters()
            ):
                param_b.data = (
                    self.momentum * param_b.data + (1 - self.momentum) * param_a.data
                )

    def pretrain_forward(self, x: torch.Tensor):
        batch_size, original_seq_len, num_features = x.size()  # (batch_size, original_seq_len, num_features)
        x = x.permute(0, 2, 1)  # (batch_size, num_features, original_seq_len)
        x = x.reshape(-1, original_seq_len, 1)  # (batch_size * num_features, original_seq_len, 1)
        x = self.feature_extractor(x)  # (batch_size * num_features, seq_len, d_model)
        x_words = self.code_book(x)  # (batch_size * num_features, seq_len)
        rep_replaced_mask = self.replaced_mask.repeat(
            x.shape[0], x.shape[1], 1
        )  # random init mask representation

        # Positional Encoding
        x_position = self.positional_encoding(x)
        x += x_position
        rep_replaced_mask += x_position

        # Split visible and mask
        indices = torch.randperm(
            x.shape[1], device=self.device
        )  # random permutation between 0 ~ seq_len - 1
        visible_indices = indices[: -self.mask_len]
        mask_indices = indices[-self.mask_len :]
        x_visible = x[:, visible_indices, :]
        x_mask = x[:, mask_indices, :]
        mask_words = x_words[:, mask_indices]
        rep_replaced_mask = rep_replaced_mask[:, mask_indices, :]

        # TimeMAE Encoder
        rep_visible = self.encoder(x_visible)

        # Momentum Encoder
        with torch.no_grad():
            rep_mask = self.momentum_encoder(x_mask)

        # TimeMAE Decoupled Encoder
        rep_mask_prediction = self.decoupled_encoder(rep_visible, rep_replaced_mask)

        # Mapping to Words
        mask_words_prediction = self.code_book.get_code_word(rep_mask_prediction)

        return [rep_mask, rep_mask_prediction], [mask_words, mask_words_prediction]

    def forward(self, x, task: str = "linear_probability"):
        if task == "linear_probability":
            with torch.no_grad():
                x = self.feature_extractor(x)
                x += self.positional_encoding(x)
                x = self.encoder(x)
                return torch.mean(x, dim=1)
        elif task == "classification":
            x = self.feature_extractor(x)
            x += self.positional_encoding(x)
            x = self.encoder(x)
            return x
        elif task == "forecasting":
            batch_size, original_seq_len, num_features = x.size()
            x = x.permute(0, 2, 1)  # (batch_size, num_features, original_seq_len)
            x = x.reshape(-1, original_seq_len, 1)  # (batch_size * num_features, original_seq_len, 1)
            x = self.feature_extractor(x)  # (batch_size * num_features, seq_len, d_model)
            x += self.positional_encoding(x)
            x = self.encoder(x)  # (batch_size * num_features, seq_len, d_model)
            x = x.reshape(batch_size, num_features, self.seq_len, self.d_model)  # (batch_size, num_features, seq_len, d_model)
            return x
        else:
            raise ValueError(
                "mode should be one of ['linear_probability', 'classification', 'forecasting']"
            )


class TimeMAEClassificationForFinetune(nn.Module):
    def __init__(
        self,
        args: Namespace,
        TimeMAE_encoder: TimeMAE,
    ):
        super(TimeMAEClassificationForFinetune, self).__init__()

        self.TimeMAE_encoder = TimeMAE_encoder

        self.classify_head = ClassificationHead(
            d_model=args.d_model, num_classes=args.num_classes
        ).to(args.device)

    def forward(self, x, finetune_mode: str = "fine_all"):
        if finetune_mode == "fine_all":
            x = self.TimeMAE_encoder(x, task="classification")
            x = self.classify_head(x)  # (batch_size, num_classes)
        elif finetune_mode == "fine_last":
            with torch.no_grad():
                x = self.TimeMAE_encoder(x, task="classification")
            x = self.classify_head(x)
        else:
            raise ValueError(
                "fine_tuning_mode should be one of ['fine_all', 'fine_last']"
            )
        return x


class TimeMAEForecastingForFinetune(nn.Module):
    def __init__(
        self,
        args: Namespace,
        TimeMAE_encoder: TimeMAE,
        origin_seq_len: int,
        num_features: int,
    ):
        super(TimeMAEForecastingForFinetune, self).__init__()

        self.TimeMAE_encoder = TimeMAE_encoder

        self.revin_layer = RevIN(
            num_features=num_features,
        ).to(args.device)

        self.seq_len = int((origin_seq_len - args.kernel_size) / args.stride) + 1
        self.forecast_head = ForecastingHead(
            seq_len=self.seq_len,
            d_model=args.d_model,
            pred_len=args.pred_len,
            num_features=num_features,
            dropout=args.dropout,
        ).to(args.device)

    def forward(self, x, finetune_mode: str = "fine_all"):
        if finetune_mode == "fine_all":
            x = self.revin_layer(x, "norm")
            x = self.TimeMAE_encoder(x, task="forecasting")  # (batch_size, num_features, seq_len, d_model)
            x = self.forecast_head(x)  # (batch_size, pred_len, num_features)
            x = self.revin_layer(x, "denorm")
        elif finetune_mode == "fine_last":
            x = self.revin_layer(x, "norm")
            with torch.no_grad():
                x = self.TimeMAE_encoder(x, task="forecasting")
            x = self.forecast_head(x)
            x = self.revin_layer(x, "denorm")
        else:
            raise ValueError(
                "fine_tuning_mode should be one of ['fine_all', 'fine_last']"
            )
        return x

    def freeze_encoder(self):
        for param in self.TimeMAE_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.TimeMAE_encoder.parameters():
            param.requires_grad = True