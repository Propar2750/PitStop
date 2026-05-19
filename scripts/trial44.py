"""Trial 44: MLP on baseline raw numerics (no Driver/Race), ensembled with T43.

Goal: add architectural + featural diversity to the T43 best (0.95018 OOF
3-way rank-avg of XGB + CB + Transformer). The user's intuition: Driver and
Race are proxies, not causal drivers of a per-row pit decision — drop them.
Combined with a feed-forward MLP (vs trees + sequence transformer), this is
the most orthogonal 4th base model we can add without inventing features.

Feature set: 10 baseline raw numerics + one-hot Compound (5 dims) = 15 inputs.
No engineered features (clean ablation), no LapNumber, no Driver, no Race.

Standalone MLP is expected to be weak (likely 0.90-0.93) because the
features are sparse without sequence/categorical context. The value, if any,
is purely in the blend.

Reuses T35 XGB/CB OOFs and T43 Transformer OOFs from /tmp if present;
otherwise just reports MLP standalone and saves OOFs for later ensembling.
"""
import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from scipy.stats import rankdata
import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()
torch.set_num_threads(8)
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
if USE_CUDA:
    print(f'device={torch.cuda.get_device_name(0)}', flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    print('device=cpu', flush=True)

DATA_DIR = Path('/home/propar/Documents/Projects/F1-pitstop-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
TARGET, ID_COL = 'PitNextLap', 'id'

# Baseline raw numerics, no LapNumber (matches README baseline).
RAW_NUM = ['Year', 'PitStop', 'Stint', 'TyreLife', 'Position',
           'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
           'RaceProgress', 'Position_Change']
# Compound as one-hot (only categorical kept; Driver and Race dropped per user spec).
COMPOUND_VALS = sorted(pd.concat([train['Compound'], test['Compound']]).astype(str).unique().tolist())
print(f'Compound vocab: {COMPOUND_VALS}', flush=True)

def build_X(df):
    Xn = df[RAW_NUM].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cmp_oh = pd.get_dummies(df['Compound'].astype(str), prefix='cmp')
    for v in COMPOUND_VALS:
        col = f'cmp_{v}'
        if col not in cmp_oh.columns:
            cmp_oh[col] = 0.0
    cmp_oh = cmp_oh[[f'cmp_{v}' for v in COMPOUND_VALS]].astype(np.float32)
    return np.concatenate([Xn.values, cmp_oh.values], axis=1).astype(np.float32)

X_train = build_X(train)
X_test  = build_X(test)
y_full  = train[TARGET].astype(np.int8).values
N_RAW = len(RAW_NUM)
N_FEAT = X_train.shape[1]
print(f'features: {N_FEAT} ({N_RAW} raw numerics + {N_FEAT - N_RAW} compound one-hot)', flush=True)

groups = (train['Race'].astype(str) + '_' + train['Year'].astype(str) + '_' + train['Driver'].astype(str)).values
N_FOLDS = 5
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
splits = list(cv.split(np.arange(len(train)), y_full, groups))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            # BN only on layers >0 helped in early trials; here use BN on first two.
            if i < len(hidden) - 1:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


SEEDS = [42, 2024, 7, 31337]
EPOCHS = 40
BATCH = 1024
LR = 1e-3
WD = 1e-4
PATIENCE = 8

mlp_oof  = np.zeros(len(train), dtype=np.float64)
mlp_test = np.zeros(len(test),  dtype=np.float64)

print('\n=== T44: MLP (raw numerics + Compound, no Driver/Race) ===', flush=True)
for fold, (tr_idx, va_idx) in enumerate(splits):
    fstart = time.time()
    # Standardize raw numerics on train fold only; pass one-hot through untouched.
    scaler = StandardScaler()
    Xtr_num = scaler.fit_transform(X_train[tr_idx, :N_RAW])
    Xva_num = scaler.transform(X_train[va_idx, :N_RAW])
    Xte_num = scaler.transform(X_test[:, :N_RAW])
    Xtr = np.concatenate([Xtr_num, X_train[tr_idx, N_RAW:]], axis=1).astype(np.float32)
    Xva = np.concatenate([Xva_num, X_train[va_idx, N_RAW:]], axis=1).astype(np.float32)
    Xte = np.concatenate([Xte_num, X_test[:, N_RAW:]], axis=1).astype(np.float32)
    ytr = y_full[tr_idx].astype(np.float32)
    yva = y_full[va_idx].astype(np.float32)

    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Xte_t = torch.from_numpy(Xte).to(DEVICE)

    fold_va_pred = np.zeros(len(va_idx), dtype=np.float64)
    fold_te_pred = np.zeros(len(test),   dtype=np.float64)
    for seed in SEEDS:
        torch.manual_seed(seed + fold); np.random.seed(seed + fold)
        model = MLP(N_FEAT).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        best_va = 0.0; best_state = None; best_ep = -1; pat = 0
        n_tr = Xtr_t.shape[0]
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(n_tr, device=DEVICE)
            for i in range(0, n_tr, BATCH):
                idx = perm[i:i+BATCH]
                xb = Xtr_t.index_select(0, idx)
                yb = ytr_t.index_select(0, idx)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sch.step()
            model.eval()
            with torch.no_grad():
                va_logits = []
                for i in range(0, Xva_t.shape[0], 8192):
                    va_logits.append(model(Xva_t[i:i+8192]).cpu().numpy())
                va_p = 1.0 / (1.0 + np.exp(-np.concatenate(va_logits)))
            va_auc = roc_auc_score(yva, va_p)
            if va_auc > best_va:
                best_va = va_auc; best_ep = ep; pat = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= PATIENCE:
                    break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            va_logits = []
            for i in range(0, Xva_t.shape[0], 8192):
                va_logits.append(model(Xva_t[i:i+8192]).cpu().numpy())
            va_p = 1.0 / (1.0 + np.exp(-np.concatenate(va_logits)))
            te_logits = []
            for i in range(0, Xte_t.shape[0], 8192):
                te_logits.append(model(Xte_t[i:i+8192]).cpu().numpy())
            te_p = 1.0 / (1.0 + np.exp(-np.concatenate(te_logits)))
        fold_va_pred += va_p
        fold_te_pred += te_p
        print(f'  fold{fold} seed={seed} best ep{best_ep} va_auc={best_va:.5f}', flush=True)
        del model, best_state
        if USE_CUDA: torch.cuda.empty_cache()
    fold_va_pred /= len(SEEDS)
    fold_te_pred /= len(SEEDS)
    fold_auc = roc_auc_score(yva, fold_va_pred)
    mlp_oof[va_idx] = fold_va_pred
    mlp_test += fold_te_pred / N_FOLDS
    print(f'  fold{fold} bagged AUC={fold_auc:.5f}  ({time.time()-fstart:.0f}s)', flush=True)

mlp_auc = roc_auc_score(y_full, mlp_oof)
mlp_ap  = average_precision_score(y_full, mlp_oof)
mlp_ll  = log_loss(y_full, np.clip(mlp_oof, 1e-7, 1 - 1e-7))
print(f'\n[MLP standalone] OOF AUC = {mlp_auc:.5f}  AP = {mlp_ap:.5f}  LL = {mlp_ll:.5f}', flush=True)
np.save('/tmp/trial44_mlp_oof.npy',  mlp_oof.astype(np.float32))
np.save('/tmp/trial44_mlp_test.npy', mlp_test.astype(np.float32))


def rank_avg(*arrs):
    L = len(arrs[0]); s = np.zeros(L)
    for a in arrs: s += rankdata(a)
    return s / (len(arrs) * L)

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


cached = {}
for tag in ['trial35_xgb', 'trial35_cb', 'trial43_tfm']:
    op = Path(f'/tmp/{tag}_oof.npy')
    tp = Path(f'/tmp/{tag}_test.npy')
    if op.exists() and tp.exists():
        cached[tag] = (np.load(op).astype(np.float64), np.load(tp).astype(np.float64))

print(f'\ncached OOF arrays found: {list(cached.keys())}', flush=True)

final_test, final_name, final_auc = mlp_test, 'mlp_standalone', mlp_auc

if {'trial35_xgb', 'trial35_cb', 'trial43_tfm'} <= set(cached):
    xgb_oof, xgb_test = cached['trial35_xgb']
    cb_oof,  cb_test  = cached['trial35_cb']
    tfm_oof, tfm_test = cached['trial43_tfm']

    three_oof  = rank_avg(xgb_oof, cb_oof, tfm_oof)
    three_test = rank_avg(xgb_test, cb_test, tfm_test)
    three_auc  = roc_auc_score(y_full, three_oof)

    four_oof  = rank_avg(xgb_oof, cb_oof, tfm_oof, mlp_oof)
    four_test = rank_avg(xgb_test, cb_test, tfm_test, mlp_test)
    four_auc  = roc_auc_score(y_full, four_oof)

    # LR-meta over the 4 base logits.
    oof_stack  = np.column_stack([logit(xgb_oof),  logit(cb_oof),  logit(tfm_oof),  logit(mlp_oof)])
    test_stack = np.column_stack([logit(xgb_test), logit(cb_test), logit(tfm_test), logit(mlp_test)])
    meta_oof = np.zeros(len(train), dtype=np.float64)
    for tr_idx, va_idx in splits:
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(oof_stack[tr_idx], y_full[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(oof_stack[va_idx])[:, 1]
    meta_auc = roc_auc_score(y_full, meta_oof)
    meta_full = LogisticRegression(C=1.0, max_iter=1000).fit(oof_stack, y_full)
    meta_test = meta_full.predict_proba(test_stack)[:, 1]

    print(f'\n==== T44 ENSEMBLE RESULTS ====', flush=True)
    print(f'corr(MLP, XGB) = {np.corrcoef(xgb_oof, mlp_oof)[0,1]:.4f}', flush=True)
    print(f'corr(MLP, CB ) = {np.corrcoef(cb_oof,  mlp_oof)[0,1]:.4f}', flush=True)
    print(f'corr(MLP, TFM) = {np.corrcoef(tfm_oof, mlp_oof)[0,1]:.4f}', flush=True)
    print(f'3-way (T43)     OOF AUC = {three_auc:.5f}', flush=True)
    print(f'4-way (T43+MLP) OOF AUC = {four_auc:.5f}  (delta {four_auc-three_auc:+.5f})', flush=True)
    print(f'LR-meta 4-way   OOF AUC = {meta_auc:.5f}  (delta {meta_auc-three_auc:+.5f})', flush=True)
    print(f'MLP standalone  OOF AUC = {mlp_auc:.5f}', flush=True)

    candidates = {
        '3way_rank_avg':         (three_auc, three_test),
        '4way_rank_avg':         (four_auc,  four_test),
        'lr_meta_4way':          (meta_auc,  meta_test),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0])
    final_auc, final_test = candidates[best_name]
    final_name = best_name
    print(f'\nBest blend: {best_name} @ {final_auc:.5f}', flush=True)
else:
    missing = {'trial35_xgb','trial35_cb','trial43_tfm'} - set(cached)
    print(f'\n[warn] missing cached OOFs: {missing}', flush=True)
    print('       run scripts/trial35.py (XGB+CB) and scripts/trial43.py (TFM) first to enable ensembling.', flush=True)
    print('       saved MLP OOF/test to /tmp/trial44_mlp_*.npy for later use.', flush=True)

sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: final_test})
sub.to_csv(DATA_DIR / 'submission.csv', index=False)
print(f'\nwrote submission.csv from {final_name} @ {final_auc:.5f}', flush=True)
print(f'TOTAL : {time.time()-t0:.0f}s', flush=True)
