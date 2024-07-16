import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            num_features: int,
            d_model: int,
            kernel_size: int,
            stride: int,
    ):
        super(FeatureExtractor, self).__init__()
        self.input_projection = nn.Conv1d(
            in_channels=num_features,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch_size, seq_len，n_features)
        Returns:
            output tensor, shape (batch_size, seq_len, d_model)
        """
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        return x.transpose(1, 2).contiguous()


class CodeBook(nn.Module):
    def __init__(
            self,
            d_model: int,
            vocab_size: int,
    ):
        super(CodeBook, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.code_word = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
        Returns:
            output tensor, shape (batch_size, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        probs = self.code_word(x.view(-1, d_model))
        probs = F.gumbel_softmax(probs)
        idx = probs.max(-1, keepdim=True)[1]
        return idx.view(batch_size, seq_len)

    def get_code_word(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the code word of the input tensor.
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
        Returns:
            output tensor, shape (batch_size, seq_len, vocab_size)
        """
        return self.code_word(x)


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            max_len: int,
            d_model: int,
            dropout: float,
    ):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.embedding.weight)  # initialize the positional encoding, using 'xavier_uniform_'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
        Returns:
            output tensor, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        pos = torch.arange(seq_len).repeat(batch_size, 1).to(x.device)
        x = self.embedding(pos)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(
            self,
            d_model: int,
            enable_res_param: bool,
            dropout: float,
    ):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.enable_res_param = enable_res_param
        if self.enable_res_param:
            self.res_param = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
            sublayer: sublayer module
        Returns:
            output tensor, shape (batch_size, seq_len, d_model)
        """
        if isinstance(x, list):
            # For TimeMAEDecoupledEncoderLayer
            assert len(x) == 2
            return x[1] + self.res_param * self.dropout(sublayer(self.norm(x[0])))

        if self.enable_res_param:
            return x + self.res_param * self.dropout(sublayer(self.norm(x)))
        else:
            return x + self.dropout(sublayer(self.norm(x)))


class TimeMAEEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            enable_res_param: bool,
    ):
        super(TimeMAEEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.res_connection1 = ResidualConnection(d_model, enable_res_param=enable_res_param, dropout=dropout)
        self.res_connection2 = ResidualConnection(d_model, enable_res_param=enable_res_param, dropout=dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
            mask: mask tensor, shape (seq_len, seq_len)
        Returns:
            output tensor, shape (batch_size, seq_len, d_model)
        """
        # Multi-head self-attention
        x = self.res_connection1(x, lambda _x: self.self_attn(_x, _x, _x, attn_mask=mask)[0])
        # Feedforward
        x = self.res_connection2(x, self.feedforward)
        return x


class TimeMAEDecoupledEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            enable_res_param: bool,
    ):
        super(TimeMAEDecoupledEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.res_connection1 = ResidualConnection(d_model, enable_res_param=enable_res_param, dropout=dropout)
        self.res_connection2 = ResidualConnection(d_model, enable_res_param=enable_res_param, dropout=dropout)

    def forward(self, x_visible, x_mask_token, mask=None):
        """
        Args:
            x_visible: input tensor, shape (batch_size, visible_seq_len, d_model)
            x_mask_token: input tensor, shape (batch_size, seq_len, d_model)
            mask: mask tensor, shape (visible_seq_len, visible_seq_len)
        Returns:
            output tensor, shape (batch_size, visible_seq_len, d_model)
        """
        x = [x_visible, x_mask_token]
        # Multi-head self-attention
        x = self.res_connection1(x, lambda _x: self.self_attn(_x[1], _x[0], _x[0], attn_mask=mask)[0])
        # Feedforward
        x = self.res_connection2(x, self.feedforward)
        return x


class TimeMAEEncoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            num_layers: int,
            enable_res_param: bool,
    ):
        super(TimeMAEEncoder, self).__init__()
        self.encoder = nn.ModuleList([
            TimeMAEEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                enable_res_param=enable_res_param,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
            mask: mask tensor, shape (seq_len, seq_len)
        Returns:
            output tensor, shape (batch_size, seq_len, d_model)
        """
        for layer in self.encoder:
            x = layer(x, mask=mask)
        return x


class TimeMAEDecoupledEncoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            num_layers: int,
            enable_res_param: bool,
    ):
        super(TimeMAEDecoupledEncoder, self).__init__()
        self.decoupled_encoder = nn.ModuleList([
            TimeMAEDecoupledEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                enable_res_param=enable_res_param,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x_visible: torch.Tensor, x_mask_token: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x_visible: input tensor, shape (batch_size, visible_seq_len, d_model)
            x_mask_token: input tensor, shape (batch_size, seq_len, d_model)
            mask: mask tensor, shape (visible_seq_len, visible_seq_len)
        Returns:
            output tensor, shape (batch_size, visible_seq_len, d_model)
        """
        for layer in self.decoupled_encoder:
            x_mask_token = layer(x_visible, x_mask_token, mask=mask)
        return x_mask_token


class TimeMAEClassifyHead(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_classes: int,
    ):
        super(TimeMAEClassifyHead, self).__init__()
        self.classify_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
        Returns:
            output tensor, shape (batch_size, num_classes)
        """
        x = x.mean(dim=1)
        return self.classify_head(x)


class TimeMAEForecastHead(nn.Module):
    def __init__(
            self,
            seq_len: int,
            d_model: int,
            pred_len: int,
            num_features: int,
    ):
        super(TimeMAEForecastHead, self).__init__()
        self.pred_len = pred_len
        self.num_features = num_features
        self.flatten = nn.Flatten()
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len * num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch_size, seq_len, d_model)
        Returns:
            output tensor, shape (batch_size, pred_len. channels)
        """
        x = self.flatten(x)  # (batch_size, seq_len * d_model)
        x = self.forecast_head(x)  # (batch_size, pred_len * channels)
        x = x.view(-1, self.pred_len, self.num_features)  # (batch_size, pred_len, channels)
        return x
