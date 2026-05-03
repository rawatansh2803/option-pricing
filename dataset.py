# ==========================================
# FILE: dataset.py
# ==========================================
class SyntheticDataGenerator:
    def __init__(self, index_name='NIFTY'):
        self.rng = np.random.RandomState(int(hashlib.md5(index_name.encode()).hexdigest()[:8], 16) % 2**31)
    def generate(self):
        n = config.n_trading_days
        S = np.zeros(n); S[0] = 20000.0
        for i in range(1, n):
            S[i] = S[i-1] * np.exp((0.05 - 0.5 * 0.2**2) * (1/252) + 0.2 * np.sqrt(1/252) * self.rng.randn())
        df = pd.DataFrame({'date': range(n), 'open': S*0.99, 'high': S*1.01, 'low': S*0.98, 'close': S, 'vol_norm': 1.0})
        recs = []
        for i in range(n):
            for off in np.arange(-5, 6):
                strike = round(S[i]*(1+off*0.02)/50)*50
                recs.append({'day_idx': i, 'spot': S[i], 'strike': strike, 'tte_years': 30/365, 'call_price': max(S[i]-strike, 0) + S[i]*0.01, 'iv': 0.2, 'oi': 1e5, 'oi_change': 0, 'vol_norm': 1.0, 'open': S[i]*0.99, 'high': S[i]*1.01, 'low': S[i]*0.98})
        return df, pd.DataFrame(recs)

class TheDataset(Dataset):
    def __init__(self, X, y, meta, spot, intrinsic):
        self.X, self.y, self.meta = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(meta, dtype=torch.float32)
        self.spot, self.intrinsic = torch.tensor(spot, dtype=torch.float32), torch.tensor(intrinsic, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return {'X': self.X[i], 'y': self.y[i], 'meta': self.meta[i], 'spot': self.spot[i], 'intrinsic': self.intrinsic[i]}

def prepare_tensors(options):
    T, K, C = config.temporal_window, config.n_strikes, config.n_channels
    X, y, meta, spots, intrinsics = [], [], [], [], []
    groups = options.groupby('day_idx')
    days = sorted(options['day_idx'].unique())
    for i in range(T, len(days)):
        d = days[i]
        tgt = groups.get_group(d).nsmallest(K, 'strike')
        if len(tgt) < K: continue
        win = np.zeros((T, K, C))
        for t in range(T):
            prev = groups.get_group(days[i-T+t]).nsmallest(K, 'strike')
            win[t] = np.random.randn(K, C) # Placeholder for feature vector extraction
        for k in range(K):
            row = tgt.iloc[k]
            X.append(win); y.append((row['call_price'] - max(row['spot']-row['strike'], 0))/row['spot'])
            meta.append([row['strike']/row['spot'], row['tte_years'], 0.0, max(row['spot']-row['strike'], 0)/row['spot'], 0.0])
            spots.append(row['spot']); intrinsics.append(max(row['spot']-row['strike'], 0))
    return np.array(X), np.array(y), np.array(meta), np.array(spots), np.array(intrinsics)
