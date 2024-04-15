import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_
from .layers import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock
from .RevIN import RevIN

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


class Tokenizer(nn.Module):
    def __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        probs = self.center(x.view(-1, dim))
        ret = F.gumbel_softmax(probs)
        indexes = ret.max(-1, keepdim=True)[1]
        return indexes.view(bs, length)


class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


class TimeMAE(nn.Module):
    def __init__(self, args):
        super(TimeMAE, self).__init__()
        d_model = args.d_model

        self.momentum = args.momentum
        self.linear_proba = "linear_proba"
        self.device = args.device
        self.data_shape = args.data_shape
        self.max_len = self.data_shape[0]  # 原始数据的长度
        self.channels = self.data_shape[1] # 原始数据的通道数
        self.max_len_conv = int(self.data_shape[0] / args.wave_length) # 卷积后的长度
        print(self.max_len_conv)
        self.mask_len = int(args.mask_ratio * self.max_len_conv)
        self.position = PositionalEmbedding(self.max_len_conv, d_model)

        self.mask_token = nn.Parameter(torch.randn(d_model, ))
        self.input_projection = nn.Conv1d(args.data_shape[1], d_model, kernel_size=args.wave_length,
                                          stride=args.wave_length)
        self.linear_projection = nn.Linear(self.max_len_conv * d_model, self.max_len * d_model) # 将卷积的结果映射到原始数据的长度
        self.encoder = Encoder(args)
        self.momentum_encoder = Encoder(args)
        self.tokenizer = Tokenizer(d_model, args.vocab_size)
        self.reg = Regressor(d_model, args.attn_heads, 4 * d_model, 1, args.reg_layers)
        self.predict_head = nn.Linear(d_model, args.num_class)
        self.apply(self._init_weights)

    def init_forecasting(self, args, pred_len):
        self.flatten = nn.Flatten().to(self.device)
        self.forecasting_head = nn.Linear(self.max_len_conv * args.d_model, pred_len * self.channels).to(self.device)
        self.norm = nn.BatchNorm1d(self.channels).to(self.device)
        self.dropout = nn.Dropout(args.dropout).to(self.device)

        self.revin_layer = RevIN(self.channels)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def pretrain_forward(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        # x = self.linear_projection(x.view(x.size(0), -1)).view(x.size(0), self.max_len, -1)
        tokens = self.tokenizer(x)
        x += self.position(x)
        rep_mask_token = self.mask_token.repeat(x.shape[0], x.shape[1], 1) + self.position(x)

        index = np.arange(x.shape[1])
        random.shuffle(index)
        v_index = index[:-self.mask_len]
        m_index = index[-self.mask_len:]
        visible = x[:, v_index, :]
        mask = x[:, m_index, :]
        tokens = tokens[:, m_index]
        rep_mask_token = rep_mask_token[:, m_index, :]

        rep_visible = self.encoder(visible)
        with torch.no_grad():
            # rep_mask = self.encoder(mask)
            rep_mask = self.momentum_encoder(mask)
        rep_mask_prediction = self.reg(rep_visible, rep_mask_token)
        token_prediction_prob = self.tokenizer.center(rep_mask_prediction)

        return [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens]

    def forward(self, x, pred_len=1):
        if self.linear_proba == "linear_proba":
            with torch.no_grad():
                x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
                # x = self.linear_projection(x.view(x.size(0), -1)).view(x.size(0), self.max_len, -1)
                x += self.position(x)
                x = self.encoder(x)
                return torch.mean(x, dim=1)
        elif self.linear_proba == "classification":
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            # x = self.linear_projection(x.view(x.size(0), -1)).view(x.size(0), self.max_len, -1)
            x += self.position(x)
            x = self.encoder(x)
            return self.predict_head(torch.mean(x, dim=1))
        elif self.linear_proba == "forecasting":  # prediction
            x = self.revin_layer(x, 'norm')

            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            # x = self.linear_projection(x.view(x.size(0), -1)).view(x.size(0), self.max_len, -1)
            x += self.position(x)
            x = self.encoder(x)  # [bs, len_conv, d_model]
            
            x = self.flatten(x) # [bs, len_conv * d_model]
            x = self.forecasting_head(x) # [bs, pred_len * channels]
            x = x.view(-1, pred_len, self.channels) # [bs, pred_len, channels]
            x = self.dropout(x)
        
            x = self.revin_layer(x, 'denorm')
            return x
        else:
            raise ValueError("linear_proba should be one of ['linear_proba', 'classification', 'forecasting']")

    def get_tokens(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)
        return tokens
