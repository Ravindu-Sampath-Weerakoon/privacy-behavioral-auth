import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import hashlib
import warnings
import time
from numba import njit

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

DATA_PATH = r'data_processed/feature_kmt_dataset_Edge_Hill_University_22/feature_importance_ranking/weighted_normalized_feature_extraction_V4_88_users.csv'

TARGET_DIM_JL = 64
HIDDEN_DIM = 32
LATENT_DIM = 16
LR = 0.001
EPOCHS = 50 # Reduced for speed in validation
LAMBDA_ENTROPY = 0.1
DEVICE = torch.device("cpu") # Use CPU for stability in shell

@njit(fastmath=True, cache=True)
def zero_allocation_fast_jl(x, seed, k):
    n = len(x)
    for i in range(n): x[i] = x[i] * 10000
    curr_seed = seed
    for i in range(n):
        curr_seed = (curr_seed * 1103515245 + 12345) & 0x7FFFFFFF
        if (curr_seed % 2) == 0: x[i] = -x[i]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[j]
                v = x[j + h]
                x[j] = u + v
                x[j + h] = u - v
        h *= 2
    return x[:k] / np.sqrt(n)

class AdvancedDeepSVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16):
        super(AdvancedDeepSVDD, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

def entropy_loss(outputs):
    var = torch.var(outputs, dim=0).sum()
    return -torch.log(var + 1e-6)

def init_center(model, train_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x in train_loader: outputs.append(model(x[0].to(device)))
    c = torch.cat(outputs).mean(dim=0)
    eps = 0.1
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c

def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    idx = np.nanargmin(np.absolute((fpr - (1 - tpr))))
    return fpr[idx]

def train_advanced_model(X_train, X_test, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=8, shuffle=True)
    model = AdvancedDeepSVDD(TARGET_DIM_JL, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    center = init_center(model, train_loader, DEVICE)
    model.train()
    for _ in range(EPOCHS):
        for batch in train_loader:
            x_batch = batch[0].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x_batch)
            dist_loss = torch.sum((outputs - center)**2, dim=1).mean()
            reg_loss = LAMBDA_ENTROPY * entropy_loss(outputs)
            (dist_loss + reg_loss).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        distances = torch.sum((model(X_test_tensor) - center)**2, dim=1).cpu().numpy()
        train_dist = torch.sum((model(X_train_tensor) - center)**2, dim=1).cpu().numpy()
        threshold = np.max(train_dist)
    scores = -distances
    acc = accuracy_score(y_test, (distances <= threshold).astype(int))
    auc = roc_auc_score(y_test, scores)
    eer = calculate_eer(y_test, scores)
    pruned_params = 0
    total_params = sum(p.numel() for p in model.parameters())
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = torch.abs(param) > 0.01
            pruned_params += (mask == 0).sum().item()
    return {'acc': acc, 'auc': auc, 'eer': eer, 'pruning_pct': (pruned_params / total_params) * 100}

df = pd.read_csv(DATA_PATH).fillna(0)
feature_cols = df.columns[2:]
df[feature_cols] = df[feature_cols] * 10
users = df['user_id'].unique()[:10] # Small sample for quick validation

n_orig = len(feature_cols)
n_pad = 1 << (n_orig - 1).bit_length()

results = []
for user_id in users:
    user_data = df[df['user_id'] == user_id]
    seed = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % (2**32)
    X_raw = user_data[feature_cols].values
    X_padded = np.zeros((X_raw.shape[0], n_pad))
    X_padded[:, :n_orig] = X_raw
    X_projected = np.zeros((X_raw.shape[0], TARGET_DIM_JL))
    for i in range(X_padded.shape[0]):
        sample = X_padded[i].copy()
        X_projected[i] = zero_allocation_fast_jl(sample, seed, TARGET_DIM_JL)
    labels = user_data['label'].values
    legit_indices = np.where(labels == 1)[0]
    imposter_indices = np.where(labels == 0)[0]
    X_train = X_projected[legit_indices[:8]]
    test_legit_idx = legit_indices[8:10]
    X_test = np.concatenate([X_projected[test_legit_idx], X_projected[imposter_indices]])
    y_test = np.concatenate([np.ones(len(test_legit_idx)), np.zeros(len(imposter_indices))])
    res = train_advanced_model(X_train, X_test, y_test)
    results.append(res)

df_res = pd.DataFrame(results)
print(df_res.mean().to_json())
