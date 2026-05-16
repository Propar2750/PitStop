"""Trial 33: per-driver-race GRU (causal by construction).

Same setup as T31 but uni-directional GRU encoder instead of transformer.
Different sequential inductive bias (recurrence vs attention) — expected to
correlate less with T31 transformer than another transformer would.

CV: StratifiedGroupKFold on (Race,Year,Driver) — same as GBDT pipeline.
Output: OOF on train rows, test predictions averaged across 5 folds.
"""
import os, time, math, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()
torch.set_num_threads(8)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={DEVICE} torch={torch.__version__}', flush=True)

DATA_DIR = Path('/home/propar/Documents/Projects/F1-pitstop-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
TARGET, ID_COL = 'PitNextLap', 'id'

# Concatenate; keep is_train flag and original target (NaN for test)
train['_is_train'] = 1
test['_is_train']  = 0
test[TARGET]       = -1  # placeholder, masked in loss
df = pd.concat([train, test], ignore_index=True)
df = df.sort_values(['Year','Race','Driver','LapNumber']).reset_index(drop=True)
print(f'combined {len(df):,} rows', flush=True)

# Engineered numeric extras (from T21 set, lightweight subset)
eps = 1e-6
df['RemainingRace'] = 1.0 - df['RaceProgress']
df['TyreStress']    = df['TyreLife'] * df['Cumulative_Degradation']
df['LapTime_per_TyreLife'] = df['LapTime (s)'] / (df['TyreLife'] + eps)
df['PitWindow']     = df['RaceProgress'] * (1 - df['RaceProgress'])

NUM_COLS = ['TyreLife','Position','LapTime (s)','LapTime_Delta','Cumulative_Degradation',
            'RaceProgress','Position_Change','Stint','PitStop',
            'RemainingRace','TyreStress','LapTime_per_TyreLife','PitWindow']

# Z-score normalise numerics globally (OK; based on full data, no target leak)
for c in NUM_COLS:
    mu, sd = df[c].mean(), df[c].std() + 1e-6
    df[c] = ((df[c] - mu) / sd).astype(np.float32)

# Categorical encoding -> int ids
CAT_COLS = ['Driver','Compound','Race','Year']
cat_vocab = {}
for c in CAT_COLS:
    cats = pd.Categorical(df[c].astype(str))
    df[c+'_id'] = cats.codes.astype(np.int64)
    cat_vocab[c] = len(cats.categories)
print(f'vocab sizes: {cat_vocab}', flush=True)

# Sequence groups
df['_seq'] = df.groupby(['Year','Race','Driver']).ngroup()
n_seq = df['_seq'].max() + 1
seq_lens = df.groupby('_seq').size()
MAX_LEN = int(seq_lens.max())
print(f'{n_seq} sequences; max len {MAX_LEN}; mean len {seq_lens.mean():.1f}', flush=True)

# Build padded tensors per sequence
N = n_seq
num_arr   = np.zeros((N, MAX_LEN, len(NUM_COLS)), dtype=np.float32)
cat_arr   = np.zeros((N, MAX_LEN, len(CAT_COLS)), dtype=np.int64)
pos_arr   = np.zeros((N, MAX_LEN), dtype=np.int64)  # LapNumber as positional id
y_arr     = np.zeros((N, MAX_LEN), dtype=np.float32)
mask_arr  = np.zeros((N, MAX_LEN), dtype=np.float32)   # 1 = real row
loss_mask = np.zeros((N, MAX_LEN), dtype=np.float32)   # 1 = train row (counts in loss)
row_seq   = np.zeros(len(df), dtype=np.int64)
row_pos   = np.zeros(len(df), dtype=np.int64)

cat_id_cols = [c+'_id' for c in CAT_COLS]
for sid, g in df.groupby('_seq'):
    L = len(g)
    num_arr[sid, :L]  = g[NUM_COLS].values
    cat_arr[sid, :L]  = g[cat_id_cols].values
    pos_arr[sid, :L]  = g['LapNumber'].values.clip(0, 199)
    y_arr[sid, :L]    = np.where(g['_is_train'].values==1, g[TARGET].values, 0.0)
    mask_arr[sid, :L] = 1.0
    loss_mask[sid, :L]= g['_is_train'].values.astype(np.float32)
    row_seq[g.index]  = sid
    row_pos[g.index]  = np.arange(L)

# Precompute per-seq lookup arrays: positions of train rows + their orig ids,
# positions of test rows + their orig ids. Avoids per-row df slicing in CV loop.
seq_train_pos = [None] * N
seq_train_idx = [None] * N  # -> idx in train.csv ordering
seq_test_pos  = [None] * N
seq_test_idx  = [None] * N  # -> idx in test.csv ordering

train_id_to_pos = {tid: i for i, tid in enumerate(train[ID_COL].values)}
test_id_to_pos  = {tid: i for i, tid in enumerate(test[ID_COL].values)}

for sid, g in df.groupby('_seq'):
    g_reset = g.reset_index(drop=True)
    is_tr = g_reset['_is_train'].values == 1
    tr_pos = np.where(is_tr)[0]
    te_pos = np.where(~is_tr)[0]
    seq_train_pos[sid] = tr_pos.astype(np.int64)
    seq_test_pos[sid]  = te_pos.astype(np.int64)
    seq_train_idx[sid] = np.array([train_id_to_pos[t] for t in g_reset.loc[is_tr, ID_COL].values], dtype=np.int64)
    seq_test_idx[sid]  = np.array([test_id_to_pos[t]  for t in g_reset.loc[~is_tr, ID_COL].values], dtype=np.int64)

print(f'tensors built in {time.time()-t0:.1f}s', flush=True)

# Original ordering arrays for OOF / test
df_train_idx = df.index[df['_is_train']==1].values
df_test_idx  = df.index[df['_is_train']==0].values
train_orig_id = df.loc[df_train_idx, ID_COL].values
test_orig_id  = df.loc[df_test_idx, ID_COL].values

# --- model: unidirectional GRU (already causal) ---
class PitGRU(nn.Module):
    def __init__(self, n_num, cat_vocab, d_model=128, n_layers=2, dropout=0.15, max_pos=200):
        super().__init__()
        self.num_proj = nn.Linear(n_num, d_model)
        self.cat_emb = nn.ModuleList([nn.Embedding(v, d_model) for v in cat_vocab.values()])
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, num, cats, pos, key_padding_mask, attn_mask):
        h = self.num_proj(num)
        for i, emb in enumerate(self.cat_emb):
            h = h + emb(cats[..., i])
        h = h + self.pos_emb(pos)
        h = self.norm_in(self.drop(h))
        out, _ = self.gru(h)
        out = self.norm_out(out)
        return self.head(out).squeeze(-1)

# Group fold split (use first row of each sequence as the group representative)
seq_first_idx = df.groupby('_seq').head(1).reset_index(drop=True)
seq_groups = (seq_first_idx['Race'].astype(str) + '_' + seq_first_idx['Year'].astype(str)
              + '_' + seq_first_idx['Driver'].astype(str)).values
# Per-sequence label: any positive in that sequence's train rows
seq_y = np.zeros(N, dtype=int)
for sid in range(N):
    seq_y[sid] = int(y_arr[sid].sum() > 0)

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_train = np.zeros(len(train), dtype=np.float32)  # in original train order
test_pred = np.zeros(len(test),  dtype=np.float32)

# Map train/test row idx (in df) back to original train.csv / test.csv order
train_id_to_pos = {tid: i for i, tid in enumerate(train[ID_COL].values)}
test_id_to_pos  = {tid: i for i, tid in enumerate(test[ID_COL].values)}

def make_causal_mask(L, device):
    return torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)

EPOCHS = 35
BATCH = 128
LR = 2e-3

for fold, (tr_seq, va_seq) in enumerate(cv.split(np.arange(N), seq_y, seq_groups)):
    fstart = time.time()
    print(f'\n==== FOLD {fold}  train_seq={len(tr_seq)} val_seq={len(va_seq)} ====', flush=True)
    torch.manual_seed(42 + fold); np.random.seed(42 + fold)

    model = PitGRU(len(NUM_COLS), cat_vocab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    num_t = torch.from_numpy(num_arr)
    cat_t = torch.from_numpy(cat_arr)
    pos_t = torch.from_numpy(pos_arr)
    y_t   = torch.from_numpy(y_arr)
    m_t   = torch.from_numpy(mask_arr)
    lm_t  = torch.from_numpy(loss_mask)

    causal = make_causal_mask(MAX_LEN, DEVICE)

    best_va_auc = 0.0
    best_state = None
    best_epoch = -1
    patience = 0

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr_seq)
        ep_loss = 0.0; ep_count = 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i+BATCH]
            num_b = num_t[b].to(DEVICE)
            cat_b = cat_t[b].to(DEVICE)
            pos_b = pos_t[b].to(DEVICE)
            y_b   = y_t[b].to(DEVICE)
            m_b   = m_t[b].to(DEVICE)         # 1 = real
            lm_b  = lm_t[b].to(DEVICE)        # 1 = train row (counts in loss)
            kpm   = (m_b == 0)                # True = pad → ignore in attention
            logits = model(num_b, cat_b, pos_b, kpm, causal)
            bce = F.binary_cross_entropy_with_logits(logits, y_b, reduction='none')
            loss = (bce * lm_b).sum() / lm_b.sum().clamp_min(1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += loss.item() * lm_b.sum().item(); ep_count += lm_b.sum().item()
        sch.step()
        # validation AUC on val sequences' train rows
        model.eval()
        with torch.no_grad():
            preds_all = []; ys_all = []
            for i in range(0, len(va_seq), BATCH):
                b = va_seq[i:i+BATCH]
                num_b = num_t[b].to(DEVICE); cat_b = cat_t[b].to(DEVICE)
                pos_b = pos_t[b].to(DEVICE); m_b = m_t[b].to(DEVICE)
                kpm = (m_b == 0)
                logits = model(num_b, cat_b, pos_b, kpm, causal)
                p = torch.sigmoid(logits).cpu().numpy()
                lm_b = lm_t[b].numpy(); y_b = y_t[b].numpy()
                sel = lm_b == 1
                preds_all.append(p[sel]); ys_all.append(y_b[sel])
            va_p = np.concatenate(preds_all); va_y = np.concatenate(ys_all)
            va_auc = roc_auc_score(va_y, va_p)
        avg_loss = ep_loss / max(ep_count, 1)
        print(f'  ep {ep:02d} loss={avg_loss:.4f} va_auc={va_auc:.5f}'
              f' lr={sch.get_last_lr()[0]:.2e}', flush=True)
        if va_auc > best_va_auc:
            best_va_auc = va_auc; best_epoch = ep; patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                print(f'  early-stop at ep {ep} (best ep {best_epoch} auc {best_va_auc:.5f})', flush=True)
                break

    # Restore best
    model.load_state_dict(best_state)
    model.eval()
    # Predict on val sequences (OOF on train rows) and ALL sequences (test rows)
    with torch.no_grad():
        # OOF: val seqs only
        for i in range(0, len(va_seq), BATCH):
            b = va_seq[i:i+BATCH]
            num_b = num_t[b].to(DEVICE); cat_b = cat_t[b].to(DEVICE)
            pos_b = pos_t[b].to(DEVICE); m_b = m_t[b].to(DEVICE)
            kpm = (m_b == 0)
            logits = model(num_b, cat_b, pos_b, kpm, causal)
            p = torch.sigmoid(logits).cpu().numpy()
            for k, sid in enumerate(b):
                tp = seq_train_pos[sid]; ti = seq_train_idx[sid]
                if len(tp): oof_train[ti] = p[k, tp]
        # Test: predict on all seqs, average across folds
        for i in range(0, N, BATCH):
            b = np.arange(i, min(i+BATCH, N))
            num_b = num_t[b].to(DEVICE); cat_b = cat_t[b].to(DEVICE)
            pos_b = pos_t[b].to(DEVICE); m_b = m_t[b].to(DEVICE)
            kpm = (m_b == 0)
            logits = model(num_b, cat_b, pos_b, kpm, causal)
            p = torch.sigmoid(logits).cpu().numpy()
            for k, sid in enumerate(b):
                tp = seq_test_pos[sid]; ti = seq_test_idx[sid]
                if len(tp): test_pred[ti] += p[k, tp] / 5.0

    print(f'  fold time {time.time()-fstart:.1f}s  best_va_auc={best_va_auc:.5f} (ep {best_epoch})', flush=True)

# Final
y_full = train[TARGET].values
oof_auc = roc_auc_score(y_full, oof_train)
print('\n==== FINAL ====', flush=True)
print(f'OOF AUC : {oof_auc:.5f}', flush=True)
print(f'AP      : {average_precision_score(y_full, oof_train):.5f}', flush=True)
print(f'LL      : {log_loss(y_full, np.clip(oof_train,1e-6,1-1e-6)):.5f}', flush=True)
print(f'TOTAL: {time.time()-t0:.1f}s', flush=True)

np.save('/tmp/trial33_oof.npy', oof_train)
np.save('/tmp/trial33_test.npy', test_pred)
print('saved', flush=True)
