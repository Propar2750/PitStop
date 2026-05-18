"""Trial 38: T37 + stint-relative positional encoding.

Single change vs T37: add a SECOND positional embedding indexed by
"laps since the start of the current stint" alongside the absolute
LapNumber positional embedding. F1 pit decisions are stint-relative;
T37 (and all earlier transformers) only saw absolute lap position.

Computation: within (Year, Race, Driver, Stint) groups, cumcount() gives
the within-stint lap index (0, 1, 2, ...). Clipped to max_stint_pos=80.
Embedded into d_model and added to the existing token representation.

All other knobs frozen at T37 values.
"""
import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from scipy.stats import rankdata
import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()
torch.set_num_threads(8)
assert torch.cuda.is_available(), 'CUDA required'
DEVICE = torch.device('cuda')
print(f'device={torch.cuda.get_device_name(0)}  torch={torch.__version__}', flush=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DATA_DIR = Path('/home/propar/Documents/Projects/F1-pitstop-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
TARGET, ID_COL = 'PitNextLap', 'id'

NUM_COLS = ['Year','PitStop','Stint','TyreLife','Position','LapTime (s)',
            'LapTime_Delta','Cumulative_Degradation','RaceProgress','Position_Change']
CAT_COLS = ['Driver','Compound','Race']

y_full = train[TARGET].astype(np.int8).values

xgb_oof  = np.load('/tmp/trial35_xgb_oof.npy')
xgb_test = np.load('/tmp/trial35_xgb_test.npy')
cb_oof   = np.load('/tmp/trial35_cb_oof.npy')
cb_test  = np.load('/tmp/trial35_cb_test.npy')

train_tag = train.copy(); train_tag['_is_train'] = 1
test_tag  = test.copy();  test_tag['_is_train']  = 0
test_tag[TARGET] = -1
df = pd.concat([train_tag, test_tag], ignore_index=True)
df = df.sort_values(['Year','Race','Driver','LapNumber']).reset_index(drop=True)

# Stint-relative lap index (the new feature for positional encoding).
df['_stint_pos'] = df.groupby(['Year','Race','Driver','Stint']).cumcount()
print(f'stint_pos range: [{df["_stint_pos"].min()}, {df["_stint_pos"].max()}]', flush=True)

for c in NUM_COLS:
    mu = df[c].mean(); sd = df[c].std() + 1e-6
    df[c] = ((df[c] - mu) / sd).astype(np.float32)

cat_vocab = {}
for c in CAT_COLS:
    cats = pd.Categorical(df[c].astype(str))
    df[c+'_id'] = cats.codes.astype(np.int64)
    cat_vocab[c] = len(cats.categories)
print(f'vocab: {cat_vocab}', flush=True)

df['_seq'] = df.groupby(['Year','Race','Driver']).ngroup()
n_seq = int(df['_seq'].max() + 1)
seq_lens = df.groupby('_seq').size()
MAX_LEN = int(seq_lens.max())
MAX_STINT_POS = 80
print(f'{n_seq} sequences; max len {MAX_LEN}; max stint_pos {df["_stint_pos"].max()}', flush=True)

N = n_seq
num_arr       = np.zeros((N, MAX_LEN, len(NUM_COLS)), dtype=np.float32)
cat_arr       = np.zeros((N, MAX_LEN, len(CAT_COLS)), dtype=np.int64)
pos_arr       = np.zeros((N, MAX_LEN), dtype=np.int64)
stint_pos_arr = np.zeros((N, MAX_LEN), dtype=np.int64)
y_arr     = np.zeros((N, MAX_LEN), dtype=np.float32)
mask_arr  = np.zeros((N, MAX_LEN), dtype=np.float32)
loss_mask = np.zeros((N, MAX_LEN), dtype=np.float32)

cat_id_cols = [c+'_id' for c in CAT_COLS]
seq_train_pos = [None] * N
seq_train_idx = [None] * N
seq_test_pos  = [None] * N
seq_test_idx  = [None] * N
train_id_to_pos = {tid: i for i, tid in enumerate(train[ID_COL].values)}
test_id_to_pos  = {tid: i for i, tid in enumerate(test[ID_COL].values)}

for sid, g in df.groupby('_seq'):
    L = len(g)
    num_arr[sid, :L]       = g[NUM_COLS].values
    cat_arr[sid, :L]       = g[cat_id_cols].values
    pos_arr[sid, :L]       = g['LapNumber'].values.clip(0, 199)
    stint_pos_arr[sid, :L] = g['_stint_pos'].values.clip(0, MAX_STINT_POS - 1)
    y_arr[sid, :L]     = np.where(g['_is_train'].values == 1, g[TARGET].values, 0.0)
    mask_arr[sid, :L]  = 1.0
    loss_mask[sid, :L] = g['_is_train'].values.astype(np.float32)
    g_reset = g.reset_index(drop=True)
    is_tr = g_reset['_is_train'].values == 1
    seq_train_pos[sid] = np.where(is_tr)[0].astype(np.int64)
    seq_test_pos[sid]  = np.where(~is_tr)[0].astype(np.int64)
    seq_train_idx[sid] = np.array(
        [train_id_to_pos[t] for t in g_reset.loc[is_tr, ID_COL].values], dtype=np.int64)
    seq_test_idx[sid]  = np.array(
        [test_id_to_pos[t]  for t in g_reset.loc[~is_tr, ID_COL].values], dtype=np.int64)

num_g       = torch.from_numpy(num_arr).to(DEVICE)
cat_g       = torch.from_numpy(cat_arr).to(DEVICE)
pos_g       = torch.from_numpy(pos_arr).to(DEVICE)
stint_pos_g = torch.from_numpy(stint_pos_arr).to(DEVICE)
y_g         = torch.from_numpy(y_arr).to(DEVICE)
m_g         = torch.from_numpy(mask_arr).to(DEVICE)
lm_g        = torch.from_numpy(loss_mask).to(DEVICE)
print(f'GPU resident: {torch.cuda.memory_allocated()/1e6:.1f} MB', flush=True)


class PitTransformerV3Stint(nn.Module):
    """T37 architecture + stint-relative positional embedding."""
    def __init__(self, num_cols, cat_vocab,
                 num_dim_per_col=6, cat_dims=None,
                 d_model=128, n_heads=4, n_layers=4, dim_ff=256,
                 dropout=0.1, max_pos=200, max_stint_pos=MAX_STINT_POS):
        super().__init__()
        if cat_dims is None:
            cat_dims = {'Driver': 24, 'Compound': 6, 'Race': 12}
        self.cat_keys = list(cat_vocab.keys())
        self.num_proj = nn.Linear(len(num_cols), len(num_cols) * num_dim_per_col)
        self.num_total = len(num_cols) * num_dim_per_col
        self.cat_emb = nn.ModuleDict({
            k: nn.Embedding(cat_vocab[k], cat_dims[k]) for k in self.cat_keys
        })
        cat_total = sum(cat_dims[k] for k in self.cat_keys)
        self.fuse = nn.Linear(self.num_total + cat_total, d_model)
        self.pos_emb       = nn.Embedding(max_pos,       d_model)
        self.stint_pos_emb = nn.Embedding(max_stint_pos, d_model)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, num, cats, pos, stint_pos, key_padding_mask, attn_mask):
        n = self.num_proj(num)
        emb_list = [self.cat_emb[k](cats[..., i]) for i, k in enumerate(self.cat_keys)]
        fused = torch.cat([n] + emb_list, dim=-1)
        h = self.fuse(fused)
        h = h + self.pos_emb(pos) + self.stint_pos_emb(stint_pos)
        h = self.drop(h)
        h = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        h = self.head_norm(h)
        return self.head(h).squeeze(-1)


seq_first = df.groupby('_seq').head(1).reset_index(drop=True)
seq_groups = (seq_first['Race'].astype(str) + '_' + seq_first['Year'].astype(str)
              + '_' + seq_first['Driver'].astype(str)).values
seq_y = (y_arr.sum(axis=1) > 0).astype(int)
N_FOLDS = 5
cv_seq = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
seq_splits = list(cv_seq.split(np.arange(N), seq_y, seq_groups))

groups_row = (train['Race'].astype(str) + '_' + train['Year'].astype(str)
              + '_' + train['Driver'].astype(str)).values
cv_row = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
row_splits = list(cv_row.split(np.arange(len(train)), y_full, groups_row))

TFM_SEEDS = [42, 2024, 7]
EPOCHS = 35
BATCH = 256
LR = 2e-3
PATIENCE = 8

def make_causal_mask(L, device):
    return torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
causal = make_causal_mask(MAX_LEN, DEVICE)

tfm_oof  = np.zeros(len(train), dtype=np.float64)
tfm_test = np.zeros(len(test),  dtype=np.float64)
t_tfm = time.time()

print('\n=== T38: T37 + stint-relative positional embedding ===', flush=True)
for fold, (tr_seq, va_seq) in enumerate(seq_splits):
    fstart = time.time()
    print(f'fold {fold}', flush=True)
    fold_oof  = np.zeros(len(train), dtype=np.float64)
    fold_test = np.zeros(len(test),  dtype=np.float64)
    for seed in TFM_SEEDS:
        torch.manual_seed(seed + fold); np.random.seed(seed + fold)
        model = PitTransformerV3Stint(NUM_COLS, cat_vocab).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        scaler = torch.amp.GradScaler('cuda')
        best_va = 0.0; best_state = None; best_ep = -1; pat = 0
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(tr_seq)
            for i in range(0, len(perm), BATCH):
                b = torch.from_numpy(perm[i:i+BATCH]).to(DEVICE)
                num_b = num_g.index_select(0, b); cat_b = cat_g.index_select(0, b)
                pos_b = pos_g.index_select(0, b); sp_b  = stint_pos_g.index_select(0, b)
                y_b   = y_g.index_select(0, b)
                m_b   = m_g.index_select(0, b);   lm_b  = lm_g.index_select(0, b)
                kpm = (m_b == 0)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(num_b, cat_b, pos_b, sp_b, kpm, causal)
                    bce = F.binary_cross_entropy_with_logits(logits, y_b, reduction='none')
                    denom = lm_b.sum().clamp_min(1.0)
                    loss = (bce * lm_b).sum() / denom
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
            sch.step()
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                preds_all = []; ys_all = []
                for i in range(0, len(va_seq), BATCH):
                    b = torch.from_numpy(va_seq[i:i+BATCH]).to(DEVICE)
                    num_b = num_g.index_select(0, b); cat_b = cat_g.index_select(0, b)
                    pos_b = pos_g.index_select(0, b); sp_b  = stint_pos_g.index_select(0, b)
                    m_b   = m_g.index_select(0, b)
                    kpm = (m_b == 0)
                    logits = model(num_b, cat_b, pos_b, sp_b, kpm, causal)
                    p = torch.sigmoid(logits.float()).cpu().numpy()
                    lm_b = lm_g.index_select(0, b).cpu().numpy()
                    y_b  = y_g.index_select(0, b).cpu().numpy()
                    sel  = lm_b == 1
                    preds_all.append(p[sel]); ys_all.append(y_b[sel])
                va_p = np.concatenate(preds_all); va_y = np.concatenate(ys_all)
                va_auc = roc_auc_score(va_y, va_p)
            if va_auc > best_va:
                best_va = va_auc; best_ep = ep; pat = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= PATIENCE:
                    print(f'    seed={seed} stop ep{ep} best ep{best_ep} auc={best_va:.5f}', flush=True)
                    break
        else:
            print(f'    seed={seed} done ep{EPOCHS-1} best ep{best_ep} auc={best_va:.5f}', flush=True)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            for i in range(0, len(va_seq), BATCH):
                b_np = va_seq[i:i+BATCH]
                b = torch.from_numpy(b_np).to(DEVICE)
                num_b = num_g.index_select(0, b); cat_b = cat_g.index_select(0, b)
                pos_b = pos_g.index_select(0, b); sp_b  = stint_pos_g.index_select(0, b)
                m_b   = m_g.index_select(0, b)
                kpm = (m_b == 0)
                logits = model(num_b, cat_b, pos_b, sp_b, kpm, causal)
                p = torch.sigmoid(logits.float()).cpu().numpy()
                for k, sid in enumerate(b_np):
                    tp = seq_train_pos[sid]; ti = seq_train_idx[sid]
                    if len(tp): fold_oof[ti] += p[k, tp]
            for i in range(0, N, BATCH):
                b_np = np.arange(i, min(i+BATCH, N))
                b = torch.from_numpy(b_np).to(DEVICE)
                num_b = num_g.index_select(0, b); cat_b = cat_g.index_select(0, b)
                pos_b = pos_g.index_select(0, b); sp_b  = stint_pos_g.index_select(0, b)
                m_b   = m_g.index_select(0, b)
                kpm = (m_b == 0)
                logits = model(num_b, cat_b, pos_b, sp_b, kpm, causal)
                p = torch.sigmoid(logits.float()).cpu().numpy()
                for k, sid in enumerate(b_np):
                    tp = seq_test_pos[sid]; ti = seq_test_idx[sid]
                    if len(tp): fold_test[ti] += p[k, tp]
        del model, best_state
        torch.cuda.empty_cache()
    fold_oof  /= len(TFM_SEEDS)
    fold_test /= len(TFM_SEEDS)
    tfm_oof  += fold_oof
    tfm_test += fold_test / N_FOLDS
    print(f'  fold time {time.time()-fstart:.0f}s', flush=True)

tfm_auc = roc_auc_score(y_full, tfm_oof)
print(f'\nT38 TFM OOF AUC = {tfm_auc:.5f}  ({time.time()-t_tfm:.0f}s total)', flush=True)
np.save('/tmp/trial38_tfm_oof.npy',  tfm_oof.astype(np.float32))
np.save('/tmp/trial38_tfm_test.npy', tfm_test.astype(np.float32))

def rank_avg(*arrs):
    L = len(arrs[0]); s = np.zeros(L)
    for a in arrs: s += rankdata(a)
    return s / (len(arrs) * L)
def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

gbdt_oof  = rank_avg(xgb_oof,  cb_oof)
gbdt_test = rank_avg(xgb_test, cb_test)
gbdt_auc = roc_auc_score(y_full, gbdt_oof)
oof_stack  = np.column_stack([logit(gbdt_oof),  logit(tfm_oof)])
test_stack = np.column_stack([logit(gbdt_test), logit(tfm_test)])
meta_oof = np.zeros(len(train), dtype=np.float64)
for tr_idx, va_idx in row_splits:
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(oof_stack[tr_idx], y_full[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(oof_stack[va_idx])[:, 1]
meta_auc = roc_auc_score(y_full, meta_oof)
three_auc = roc_auc_score(y_full, rank_avg(xgb_oof, cb_oof, tfm_oof))
print(f'\n==== T38 RESULTS ====', flush=True)
print(f'T38 TFM   OOF AUC : {tfm_auc:.5f}  (T37 0.93958 -> delta {tfm_auc-0.93958:+.5f})', flush=True)
print(f'GBDT      OOF AUC : {gbdt_auc:.5f}', flush=True)
print(f'LR-meta   OOF AUC : {meta_auc:.5f}', flush=True)
print(f'3-way     OOF AUC : {three_auc:.5f}', flush=True)
print(f'corr(T38_TFM, XGB)={np.corrcoef(xgb_oof, tfm_oof)[0,1]:.4f} '
      f'corr(T38_TFM, CB)={np.corrcoef(cb_oof, tfm_oof)[0,1]:.4f}', flush=True)
print(f'TOTAL : {time.time()-t0:.0f}s', flush=True)
