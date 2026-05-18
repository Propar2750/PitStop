"""Trial 35: T31 ensemble architecture, raw features only, full GPU.

Ablation of T31 (OOF 0.94948 — XGB + CatBoost + transformer LR-meta stack).
Strips every engineered feature and pseudo-label from the lineage; keeps the
exact architecture and hyperparameters, and runs all three base models on GPU.

Features: README baseline only.
  numeric (10): Year, PitStop, Stint, TyreLife, Position, LapTime (s),
                LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change
  categorical (3): Driver, Compound, Race
  dropped: id, PitNextLap, LapNumber

Training data: real train.csv rows only. No pseudo-labels.

CV: StratifiedGroupKFold(5, shuffle=True, random_state=42),
    group = Race + "_" + Year + "_" + Driver.

Pipeline:
  XGBoost  (depth=7, lr=0.03, 6000/200, reg_alpha=0.1, reg_lambda=1.0, mcw=20,
            tree_method=hist, device=cuda) — 2-seed bag (42, 2024)
  CatBoost (depth=8, lr=0.03, 6000/200, l2=5.0, task_type=GPU) — 2-seed bag
  Transformer (T31 spec, 3-seed bag) — GPU resident, AMP fp16
  Stack: LR-meta on [logit(rank-avg(XGB,CB)), logit(transformer)]

Outputs: prints OOF AUC/AP/LL, writes submission.csv, dumps OOF/test arrays
to /tmp/trial35_*.npy for downstream re-stacking.
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
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()
torch.set_num_threads(8)
assert torch.cuda.is_available(), 'CUDA required for T35'
DEVICE = torch.device('cuda')
print(f'device={torch.cuda.get_device_name(0)}  torch={torch.__version__}'
      f'  xgb={xgb.__version__}', flush=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DATA_DIR = Path('/home/propar/Documents/Projects/F1-pitstop-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
TARGET, ID_COL = 'PitNextLap', 'id'
print(f'train {len(train):,}  test {len(test):,}', flush=True)

# README baseline only. NO engineering.
NUM_COLS = ['Year','PitStop','Stint','TyreLife','Position','LapTime (s)',
            'LapTime_Delta','Cumulative_Degradation','RaceProgress','Position_Change']
CAT_COLS = ['Driver','Compound','Race']
FEAT_COLS = NUM_COLS + CAT_COLS
assert TARGET not in FEAT_COLS and 'LapNumber' not in FEAT_COLS and ID_COL not in FEAT_COLS

y = train[TARGET].astype(np.int8).values
groups = (train['Race'].astype(str) + '_' + train['Year'].astype(str)
          + '_' + train['Driver'].astype(str)).values

# Shared categorical codes across train+test. For XGB we use ordinal int codes
# (proven setup in fast_trial.py — partition splits over 887-cardinality Driver
# fragment more than ordinal int splits on this dataset). For CB we keep
# strings and let CB handle them natively.
df_all = pd.concat([train[FEAT_COLS], test[FEAT_COLS]], ignore_index=True)
for c in CAT_COLS:
    df_all[c] = df_all[c].astype(str).astype('category')
X_train = df_all.iloc[:len(train)].copy().reset_index(drop=True)
X_test  = df_all.iloc[len(train):].copy().reset_index(drop=True)

# XGB sees integer codes (no enable_categorical).
X_train_xgb = X_train.copy()
X_test_xgb  = X_test.copy()
for c in CAT_COLS:
    X_train_xgb[c] = X_train_xgb[c].cat.codes.astype(np.int32)
    X_test_xgb[c]  = X_test_xgb[c].cat.codes.astype(np.int32)

N_FOLDS = 5
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
splits = list(cv.split(X_train, y, groups))
print(f'CV folds: {[(len(tr), len(va)) for tr,va in splits]}', flush=True)

# ---------------------------------------------------------------------------
# XGBoost — 2-seed GPU bag (T21 hyperparams)
# ---------------------------------------------------------------------------
print('\n=== XGBoost ===', flush=True)
XGB_SEEDS = [42, 2024]
xgb_oof  = np.zeros(len(train), dtype=np.float64)
xgb_test = np.zeros(len(test),  dtype=np.float64)
t_xgb = time.time()
for fold, (tr_idx, va_idx) in enumerate(splits):
    fold_va = np.zeros(len(va_idx), dtype=np.float64)
    fold_te = np.zeros(len(test),   dtype=np.float64)
    for seed in XGB_SEEDS:
        model = xgb.XGBClassifier(
            max_depth=7, learning_rate=0.03, n_estimators=6000,
            early_stopping_rounds=200, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=20, subsample=0.9, colsample_bytree=0.9,
            tree_method='hist', device='cuda',
            eval_metric='auc',
            random_state=seed, verbosity=0,
        )
        model.fit(X_train_xgb.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_train_xgb.iloc[va_idx], y[va_idx])], verbose=False)
        fold_va += model.predict_proba(X_train_xgb.iloc[va_idx])[:, 1]
        fold_te += model.predict_proba(X_test_xgb)[:, 1]
    fold_va /= len(XGB_SEEDS)
    fold_te /= len(XGB_SEEDS)
    xgb_oof[va_idx] = fold_va
    xgb_test += fold_te / N_FOLDS
    auc = roc_auc_score(y[va_idx], fold_va)
    print(f'  fold {fold}  va_auc={auc:.5f}  t={time.time()-t_xgb:.0f}s', flush=True)
xgb_auc = roc_auc_score(y, xgb_oof)
print(f'XGB OOF AUC = {xgb_auc:.5f}  ({time.time()-t_xgb:.0f}s total)', flush=True)
np.save('/tmp/trial35_xgb_oof.npy',  xgb_oof.astype(np.float32))
np.save('/tmp/trial35_xgb_test.npy', xgb_test.astype(np.float32))

# ---------------------------------------------------------------------------
# CatBoost — 2-seed GPU bag (T21 hyperparams)
# ---------------------------------------------------------------------------
print('\n=== CatBoost ===', flush=True)
CB_SEEDS = [42, 2024]
# CB on GPU does not accept pandas 'category' dtype; pass strings.
X_train_cb = X_train.copy()
X_test_cb  = X_test.copy()
for c in CAT_COLS:
    X_train_cb[c] = X_train_cb[c].astype(str)
    X_test_cb[c]  = X_test_cb[c].astype(str)
cb_cat_idx = [X_train_cb.columns.get_loc(c) for c in CAT_COLS]

cb_oof  = np.zeros(len(train), dtype=np.float64)
cb_test = np.zeros(len(test),  dtype=np.float64)
t_cb = time.time()
for fold, (tr_idx, va_idx) in enumerate(splits):
    fold_va = np.zeros(len(va_idx), dtype=np.float64)
    fold_te = np.zeros(len(test),   dtype=np.float64)
    tr_pool = Pool(X_train_cb.iloc[tr_idx], label=y[tr_idx], cat_features=cb_cat_idx)
    va_pool = Pool(X_train_cb.iloc[va_idx], label=y[va_idx], cat_features=cb_cat_idx)
    te_pool = Pool(X_test_cb,                                  cat_features=cb_cat_idx)
    for seed in CB_SEEDS:
        model = CatBoostClassifier(
            depth=8, learning_rate=0.03, iterations=6000,
            l2_leaf_reg=5.0, eval_metric='AUC',
            bootstrap_type='Bernoulli', subsample=0.9,
            task_type='GPU', devices='0',
            random_seed=seed, od_type='Iter', od_wait=200,
            verbose=False, allow_writing_files=False,
        )
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        fold_va += model.predict_proba(va_pool)[:, 1]
        fold_te += model.predict_proba(te_pool)[:, 1]
    fold_va /= len(CB_SEEDS)
    fold_te /= len(CB_SEEDS)
    cb_oof[va_idx] = fold_va
    cb_test += fold_te / N_FOLDS
    auc = roc_auc_score(y[va_idx], fold_va)
    print(f'  fold {fold}  va_auc={auc:.5f}  t={time.time()-t_cb:.0f}s', flush=True)
cb_auc = roc_auc_score(y, cb_oof)
print(f'CB OOF AUC = {cb_auc:.5f}  ({time.time()-t_cb:.0f}s total)', flush=True)
np.save('/tmp/trial35_cb_oof.npy',  cb_oof.astype(np.float32))
np.save('/tmp/trial35_cb_test.npy', cb_test.astype(np.float32))

# ---------------------------------------------------------------------------
# Transformer — 3-seed GPU bag (T31/T34 spec, raw inputs only)
# ---------------------------------------------------------------------------
print('\n=== Transformer ===', flush=True)
TFM_SEEDS = [42, 2024, 7]

# Build sequences over train+test. Year stays numeric (baseline convention);
# only Driver, Compound, Race get embeddings.
train_tag = train.copy(); train_tag['_is_train'] = 1
test_tag  = test.copy();  test_tag['_is_train']  = 0
test_tag[TARGET] = -1
df = pd.concat([train_tag, test_tag], ignore_index=True)
df = df.sort_values(['Year','Race','Driver','LapNumber']).reset_index(drop=True)

# z-score numerics (combined train+test stats — features are not the target).
for c in NUM_COLS:
    mu = df[c].mean()
    sd = df[c].std() + 1e-6
    df[c] = ((df[c] - mu) / sd).astype(np.float32)

TFM_CAT_COLS = ['Driver','Compound','Race']  # NO Year embedding (numeric per baseline)
cat_vocab = {}
for c in TFM_CAT_COLS:
    cats = pd.Categorical(df[c].astype(str))
    df[c+'_id'] = cats.codes.astype(np.int64)
    cat_vocab[c] = len(cats.categories)
print(f'  vocab: {cat_vocab}', flush=True)

df['_seq'] = df.groupby(['Year','Race','Driver']).ngroup()
n_seq = int(df['_seq'].max() + 1)
seq_lens = df.groupby('_seq').size()
MAX_LEN = int(seq_lens.max())
print(f'  {n_seq} sequences; max len {MAX_LEN}; mean {seq_lens.mean():.1f}', flush=True)

N = n_seq
num_arr   = np.zeros((N, MAX_LEN, len(NUM_COLS)), dtype=np.float32)
cat_arr   = np.zeros((N, MAX_LEN, len(TFM_CAT_COLS)), dtype=np.int64)
pos_arr   = np.zeros((N, MAX_LEN), dtype=np.int64)
y_arr     = np.zeros((N, MAX_LEN), dtype=np.float32)
mask_arr  = np.zeros((N, MAX_LEN), dtype=np.float32)
loss_mask = np.zeros((N, MAX_LEN), dtype=np.float32)

cat_id_cols = [c+'_id' for c in TFM_CAT_COLS]
seq_train_pos = [None] * N
seq_train_idx = [None] * N
seq_test_pos  = [None] * N
seq_test_idx  = [None] * N
train_id_to_pos = {tid: i for i, tid in enumerate(train[ID_COL].values)}
test_id_to_pos  = {tid: i for i, tid in enumerate(test[ID_COL].values)}

for sid, g in df.groupby('_seq'):
    L = len(g)
    num_arr[sid, :L]   = g[NUM_COLS].values
    cat_arr[sid, :L]   = g[cat_id_cols].values
    pos_arr[sid, :L]   = g['LapNumber'].values.clip(0, 199)
    y_arr[sid, :L]     = np.where(g['_is_train'].values == 1, g[TARGET].values, 0.0)
    mask_arr[sid, :L]  = 1.0
    loss_mask[sid, :L] = g['_is_train'].values.astype(np.float32)
    g_reset = g.reset_index(drop=True)
    is_tr = g_reset['_is_train'].values == 1
    tr_pos = np.where(is_tr)[0]
    te_pos = np.where(~is_tr)[0]
    seq_train_pos[sid] = tr_pos.astype(np.int64)
    seq_test_pos[sid]  = te_pos.astype(np.int64)
    seq_train_idx[sid] = np.array(
        [train_id_to_pos[t] for t in g_reset.loc[is_tr, ID_COL].values], dtype=np.int64)
    seq_test_idx[sid]  = np.array(
        [test_id_to_pos[t]  for t in g_reset.loc[~is_tr, ID_COL].values], dtype=np.int64)

num_g = torch.from_numpy(num_arr).to(DEVICE)
cat_g = torch.from_numpy(cat_arr).to(DEVICE)
pos_g = torch.from_numpy(pos_arr).to(DEVICE)
y_g   = torch.from_numpy(y_arr).to(DEVICE)
m_g   = torch.from_numpy(mask_arr).to(DEVICE)
lm_g  = torch.from_numpy(loss_mask).to(DEVICE)
print(f'  GPU resident: {torch.cuda.memory_allocated()/1e6:.1f} MB', flush=True)

class PitTransformer(nn.Module):
    def __init__(self, n_num, cat_vocab, d_model=96, n_heads=4, n_layers=3,
                 dim_ff=192, dropout=0.1, max_pos=200):
        super().__init__()
        self.num_proj = nn.Linear(n_num, d_model)
        self.cat_emb = nn.ModuleList([nn.Embedding(v, d_model) for v in cat_vocab.values()])
        self.pos_emb = nn.Embedding(max_pos, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, num, cats, pos, key_padding_mask, attn_mask):
        h = self.num_proj(num)
        for i, emb in enumerate(self.cat_emb):
            h = h + emb(cats[..., i])
        h = h + self.pos_emb(pos)
        h = self.drop(h)
        h = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        return self.head(h).squeeze(-1)

# Sequence-level CV: same group split as row-level (one sequence per group).
seq_first = df.groupby('_seq').head(1).reset_index(drop=True)
seq_groups = (seq_first['Race'].astype(str) + '_' + seq_first['Year'].astype(str)
              + '_' + seq_first['Driver'].astype(str)).values
seq_y = (y_arr.sum(axis=1) > 0).astype(int)
cv_seq = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
seq_splits = list(cv_seq.split(np.arange(N), seq_y, seq_groups))

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
for fold, (tr_seq, va_seq) in enumerate(seq_splits):
    fstart = time.time()
    print(f'  fold {fold}  tr_seq={len(tr_seq)} va_seq={len(va_seq)}', flush=True)
    fold_oof  = np.zeros(len(train), dtype=np.float64)
    fold_test = np.zeros(len(test),  dtype=np.float64)
    for seed in TFM_SEEDS:
        torch.manual_seed(seed + fold); np.random.seed(seed + fold)
        model = PitTransformer(len(NUM_COLS), cat_vocab).to(DEVICE)
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
                pos_b = pos_g.index_select(0, b); y_b   = y_g.index_select(0, b)
                m_b   = m_g.index_select(0, b);   lm_b  = lm_g.index_select(0, b)
                kpm = (m_b == 0)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(num_b, cat_b, pos_b, kpm, causal)
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
                    pos_b = pos_g.index_select(0, b); m_b   = m_g.index_select(0, b)
                    kpm = (m_b == 0)
                    logits = model(num_b, cat_b, pos_b, kpm, causal)
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
                    print(f'    seed={seed} early-stop ep{ep} best ep{best_ep} auc={best_va:.5f}', flush=True)
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
                pos_b = pos_g.index_select(0, b); m_b   = m_g.index_select(0, b)
                kpm = (m_b == 0)
                logits = model(num_b, cat_b, pos_b, kpm, causal)
                p = torch.sigmoid(logits.float()).cpu().numpy()
                for k, sid in enumerate(b_np):
                    tp = seq_train_pos[sid]; ti = seq_train_idx[sid]
                    if len(tp): fold_oof[ti] += p[k, tp]
            for i in range(0, N, BATCH):
                b_np = np.arange(i, min(i+BATCH, N))
                b = torch.from_numpy(b_np).to(DEVICE)
                num_b = num_g.index_select(0, b); cat_b = cat_g.index_select(0, b)
                pos_b = pos_g.index_select(0, b); m_b   = m_g.index_select(0, b)
                kpm = (m_b == 0)
                logits = model(num_b, cat_b, pos_b, kpm, causal)
                p = torch.sigmoid(logits.float()).cpu().numpy()
                for k, sid in enumerate(b_np):
                    tp = seq_test_pos[sid]; ti = seq_test_idx[sid]
                    if len(tp): fold_test[ti] += p[k, tp]
        del model, best_state
        torch.cuda.empty_cache()

    fold_oof  /= len(TFM_SEEDS)
    fold_test /= len(TFM_SEEDS)
    tfm_oof  += fold_oof              # disjoint val rows
    tfm_test += fold_test / N_FOLDS
    print(f'    fold time {time.time()-fstart:.0f}s', flush=True)

tfm_auc = roc_auc_score(y, tfm_oof)
print(f'TFM OOF AUC = {tfm_auc:.5f}  ({time.time()-t_tfm:.0f}s total)', flush=True)
np.save('/tmp/trial35_tfm_oof.npy',  tfm_oof.astype(np.float32))
np.save('/tmp/trial35_tfm_test.npy', tfm_test.astype(np.float32))

# ---------------------------------------------------------------------------
# Stack: rank-avg(XGB,CB) -> GBDT_mean ; LR-meta over [GBDT_mean, TFM] logits
# ---------------------------------------------------------------------------
print('\n=== Stack ===', flush=True)

def rank_avg(a, b):
    return (rankdata(a) + rankdata(b)) / (2.0 * len(a))

gbdt_oof_rank  = rank_avg(xgb_oof,  cb_oof)
gbdt_test_rank = rank_avg(xgb_test, cb_test)
gbdt_auc = roc_auc_score(y, gbdt_oof_rank)
print(f'GBDT rank-avg OOF AUC = {gbdt_auc:.5f}', flush=True)

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

oof_stack  = np.column_stack([logit(gbdt_oof_rank),  logit(tfm_oof)])
test_stack = np.column_stack([logit(gbdt_test_rank), logit(tfm_test)])

# OOF score from cross-validated LR-meta to avoid using full-OOF refit on itself.
meta_oof = np.zeros(len(train), dtype=np.float64)
for tr_idx, va_idx in splits:
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(oof_stack[tr_idx], y[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(oof_stack[va_idx])[:, 1]
meta_oof_auc = roc_auc_score(y, meta_oof)
meta_oof_ap  = average_precision_score(y, meta_oof)
meta_oof_ll  = log_loss(y, np.clip(meta_oof, 1e-6, 1-1e-6))

# Final stacker on full OOF, applied to test for submission.
meta_full = LogisticRegression(C=1.0, max_iter=1000).fit(oof_stack, y)
test_pred = meta_full.predict_proba(test_stack)[:, 1]

print(f'\n==== T35 RESULTS ====', flush=True)
print(f'XGB   OOF AUC : {xgb_auc:.5f}', flush=True)
print(f'CB    OOF AUC : {cb_auc:.5f}', flush=True)
print(f'TFM   OOF AUC : {tfm_auc:.5f}', flush=True)
print(f'GBDT  OOF AUC : {gbdt_auc:.5f}  (rank-avg XGB+CB)', flush=True)
print(f'OOF AUC : {meta_oof_auc:.5f}', flush=True)
print(f'AP      : {meta_oof_ap:.5f}', flush=True)
print(f'LL      : {meta_oof_ll:.5f}', flush=True)
print(f'TOTAL   : {time.time()-t0:.0f}s', flush=True)

sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_pred})
sub.to_csv(DATA_DIR / 'submission.csv', index=False)
print('wrote submission.csv', flush=True)
