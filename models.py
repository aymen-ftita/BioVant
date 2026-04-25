import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class SleepTransformer(nn.Module):
    def __init__(self, n_channels=5, patch_len=50,
                 d_model=128, nhead=8, num_layers=4, num_classes=3):
        super().__init__()
        self.patch_len   = patch_len
        self.n_patches   = 3000 // patch_len    # 60
        self.patch_embed = nn.Linear(n_channels * patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.n_patches+1)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        patches = x.unfold(-1, self.patch_len, self.patch_len)
        patches = patches.permute(0,2,1,3).reshape(B, self.n_patches, -1)
        x   = self.patch_embed(patches)
        cls = self.cls_token.expand(B,-1,-1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_encoder(x)
        x   = self.transformer(x)
        return self.classifier(x[:, 0])

class SleepCNN(nn.Module):
    def __init__(self, n_channels=5, num_classes=3):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=200, stride=25, padding=100),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*16*2, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64),      nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        b1 = self.branch1(x).flatten(1)
        b2 = self.branch2(x).flatten(1)
        return self.classifier(torch.cat([b1, b2], dim=1))


class SleepLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128,
                 num_layers=2, num_classes=3):
        super(SleepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        lstm_out = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, 1),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        B = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, B, self.hidden_size,
                         dtype=x.dtype, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, B, self.hidden_size,
                         dtype=x.dtype, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        attn   = self.attention(out)
        ctx    = torch.sum(attn * out, dim=1)
        return self.fc(ctx)
