import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import hashlib

# ==========================================
# FILE: config.py
# ==========================================
class Config:
    n_trading_days = 500
    n_strikes = 11
    temporal_window = 20
    n_channels = 12
    feature_channels = [
        'close_norm', 'high_low_range', 'open_close_return', 'volume_norm',
        'oi_change', 'implied_vol', 'realised_vol', 'bid_ask_spread',
        'moneyness', 'log_moneyness', 'time_to_expiry', 'historical_premium'
    ]
    conv_channels = [32, 64, 128]
    fc_dims = [128, 64]
    dropout = 0.15
    use_se = True
    target_input_dim = 5
    target_embed_dim = 32
    max_time_value_ratio = 0.25
    batch_size = 64
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'checkpoints/final_weights.pth'

config = Config()
