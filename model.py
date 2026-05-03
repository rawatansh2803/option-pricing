import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SqueezeExcitation(out_ch) if use_se else nn.Identity()
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        res = self.residual(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + res)

class SpatioTemporalPricingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = config.n_channels
        self.input_bn = nn.BatchNorm2d(in_ch)
        
        blocks = []
        curr_ch = in_ch
        for out_ch in config.conv_channels:
            blocks.append(ResidualConvBlock(curr_ch, out_ch, config.use_se))
            curr_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.target_embed = nn.Sequential(
            nn.Linear(config.target_input_dim, config.target_embed_dim),
            nn.LayerNorm(config.target_embed_dim),
            nn.GELU(),
            nn.Linear(config.target_embed_dim, config.target_embed_dim)
        )
        
        head_in = config.conv_channels[-1] + config.target_embed_dim
        layers = []
        for dim in config.fc_dims:
            layers.extend([nn.Linear(head_in, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(config.dropout)])
            head_in = dim
        layers.append(nn.Linear(head_in, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x, meta):
        # x: [B, T, K, C] -> Convert to image format [B, C, T, K]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.input_bn(x)
        x = self.backbone(x)
        x = self.pool(x).view(x.size(0), -1)
        
        m = self.target_embed(meta)
        combined = torch.cat([x, m], dim=1)
        raw = self.head(combined).squeeze(-1)
        return torch.sigmoid(raw) * config.max_time_value_ratio
