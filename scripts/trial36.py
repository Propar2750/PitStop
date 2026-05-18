"""Trial 36: Transformer v2 (T31 architecture, beefed-up training recipe).

Goal: push the transformer's standalone OOF from T35's 0.93542 toward ~0.94
by stacking known regularization wins on top of T31's frozen architecture.
Then re-stack with T35's saved XGB/CB OOFs via LR-meta and 3-way rank-avg.

Frozen (matches T31/T35 exactly):
  - 3-layer encoder, d_model=96, n_heads=4, dim_ff=192, dropout=0.1,
    GELU, pre-norm, causal mask
  - Raw baseline features only: 10 numerics + 3 cat embeddings (Driver, Compound, Race)
  - StratifiedGroupKFold(5, shuffle=True, random_state=42), group=Race_Year_Driver
  - 3-seed bag (42, 2024, 7), AMP fp16, GPU-resident sequence tensors

New training recipe (the bundle being tested):
  1. Stochastic depth (DropPath) on attention + FFN residuals, linear schedule p in [0, 0.1]
  2. Label smoothing 0.05 (y_smooth = 0.95*y + 0.025)
  3. Feature-column dropout p=0.1 per batch per numeric column
  4. EMA of model weights (decay 0.999), eval/checkpoint on shadow weights
  5. Cosine warm restarts (T_0=20, T_mult=2 -> 60 epochs total)
  6. Peak LR 1.5e-3 (was 2e-3) + patience 20

Ensemble: rank-avg(T35 XGB, T35 CB) -> GBDT_mean; LR-meta on logits
[GBDT_mean, T36_TFM]; also report 3-way rank-avg(XGB, CB, T36_TFM).
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
assert torch.cuda.is_available(), 'CUDA required for T36'
DEVICE = torch.device('cuda')
print(f'device={torch.cuda.get_device_name(0)}  torch={torch.__version__}', flush=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DATA_DIR = Path('/home/propar/Documents/Projects/F1-pitstop-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
TARGET, ID_COL = 'PitNextLap', 'id'
print(f'train {len(train):,}  test {len(test):,}', flush=True)

NUM_COLS = ['Year','PitStop','Stint','TyreLife','Position','LapTime (s)',
            'LapTime_Delta','Cumulative_Degradation','RaceProgress','Position_Change']
CAT_COLS = ['Driver','Compound','Race']

y_full = train[TARGET].astype(np.int8).values

# Reuse T35 GBDT predictions.
xgb_oof  = np.load('/tmp/trial35_xgb_oof.npy')
xgb_test = np.load('/tmp/trial35_xgb_test.npy')
cb_oof   = np.load('/tmp/trial35_cb_oof.npy')
cb_test  = np.load('/tmp/trial35_cb_test.npy')
assert xgb_oof.shape == (len(train),) and cb_oof.shape == (len(train),)
print(f'reused T35 GBDT OOFs: XGB={roc_auc_score(y_full, xgb_oof):.5f}  '
      f'CB={roc_auc_score(y_full, cb_oof):.5f}', flush=True)

# Build sequences over train+test.
train_tag = train.copy(); train_tag['_is_train'] = 1
test_tag  = test.copy();  test_tag['_is_train']  = 0
test_tag[TARGET] = -1
df = pd.concat([train_tag, test_tag], ignore_index=True)
df = df.sort_values(['Year','Race','Driver','LapNumber']).reset_index(drop=True)

for c in NUM_COLS:
    mu = df[c].mean()
    sd = df[c].std() + 1e-6
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
print(f'{n_seq} sequences; max len {MAX_LEN}; mean {seq_lens.mean():.1f}', flush=True)

N = n_seq
num_arr   = np.zeros((N, MAX_LEN, len(NUM_COLS)), dtype=np.float32)
cat_arr   = np.zeros((N, MAX_LEN, len(CAT_COLS)), dtype=np.int64)
pos_arr   = np.zeros((N, MAX_LEN), dtype=np.int64)
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
    num_arr[sid, :L]   = g[NUM_COLS].values
    cat_arr[sid, :L]   = g[cat_id_cols].values
    pos_arr[sid, :L]   = g['LapNumber'].values.clip(0, 199)
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

num_g = torch.from_numpy(num_arr).to(DEVICE)
cat_g = torch.from_numpy(cat_arr).to(DEVICE)
pos_g = torch.from_numpy(pos_arr).to(DEVICE)
y_g   = torch.from_numpy(y_arr).to(DEVICE)
m_g   = torch.from_numpy(mask_arr).to(DEVICE)
lm_g  = torch.from_numpy(loss_mask).to(DEVICE)
print(f'GPU resident: {torch.cuda.memory_allocated()/1e6:.1f} MB', flush=True)


# ---------------------------------------------------------------------------
# Transformer v2 components
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth: scaled Bernoulli mask on a residual branch, per-sample."""
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        # one mask per (batch_item) — same mask across seq + features in that item
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class PreNormEncoderLayer(nn.Module):
    """Custom pre-norm transformer encoder layer with DropPath on each residual."""
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop  = nn.Dropout(dropout)
        self.dp1   = DropPath(drop_path)
        self.dp2   = DropPath(drop_path)

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dp1(self.drop(a))
        h = self.norm2(x)
        f = self.ff(h)
        x = x + self.dp2(self.drop(f))
        return x


class PitTransformerV2(nn.Module):
    def __init__(self, n_num, cat_vocab, d_model=96, n_heads=4, n_layers=3,
                 dim_ff=192, dropout=0.1, max_pos=200,
                 drop_path_max=0.1, feat_dropout_p=0.1):
        super().__init__()
        self.num_proj = nn.Linear(n_num, d_model)
        self.cat_emb = nn.ModuleList([nn.Embedding(v, d_model) for v in cat_vocab.values()])
        self.pos_emb = nn.Embedding(max_pos, d_model)
        # Linear DropPath schedule across layers.
        dp_rates = [drop_path_max * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            PreNormEncoderLayer(d_model, n_heads, dim_ff, dropout, dp)
            for dp in dp_rates
        ])
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        self.feat_dropout_p = feat_dropout_p
        self.n_num = n_num

    def forward(self, num, cats, pos, key_padding_mask, attn_mask):
        if self.training and self.feat_dropout_p > 0:
            # one mask per (batch_item, feature) — same across the sequence
            keep = 1.0 - self.feat_dropout_p
            B = num.shape[0]
            mask = num.new_empty((B, 1, self.n_num)).bernoulli_(keep).div_(keep)
            num = num * mask
        h = self.num_proj(num)
        for i, emb in enumerate(self.cat_emb):
            h = h + emb(cats[..., i])
        h = h + self.pos_emb(pos)
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        h = self.head_norm(h)
        return self.head(h).squeeze(-1)


class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if k in self.shadow}
        sd = model.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.shadow[k])

    def restore(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = None

    def state(self):
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}


# ---------------------------------------------------------------------------
# CV setup (same as T35)
# ---------------------------------------------------------------------------
seq_first = df.groupby('_seq').head(1).reset_index(drop=True)
seq_groups = (seq_first['Race'].astype(str) + '_' + seq_first['Year'].astype(str)
              + '_' + seq_first['Driver'].astype(str)).values
seq_y = (y_arr.sum(axis=1) > 0).astype(int)
N_FOLDS = 5
cv_seq = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
seq_splits = list(cv_seq.split(np.arange(N), seq_y, seq_groups))
print(f'fold sizes: {[(len(t), len(v)) for t,v in seq_splits]}', flush=True)
print(f'fold 0 first 5 val seq IDs: {seq_splits[0][1][:5].tolist()}', flush=True)

# Row-level split (for LR-meta CV later).
groups_row = (train['Race'].astype(str) + '_' + train['Year'].astype(str)
              + '_' + train['Driver'].astype(str)).values
cv_row = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
row_splits = list(cv_row.split(np.arange(len(train)), y_full, groups_row))

# Recipe knobs.
TFM_SEEDS = [42, 2024, 7]
EPOCHS = 60          # 20 + 40 = T_0 + T_0*T_mult
BATCH = 256
LR = 1.5e-3
PATIENCE = 20
LABEL_SMOOTH = 0.05
EMA_DECAY = 0.999

def make_causal_mask(L, device):
    return torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
causal = make_causal_mask(MAX_LEN, DEVICE)

tfm_oof  = np.zeros(len(train), dtype=np.float64)
tfm_test = np.zeros(len(test),  dtype=np.float64)
t_tfm = time.time()

print('\n=== Transformer v2 ===', flush=True)
for fold, (tr_seq, va_seq) in enumerate(seq_splits):
    fstart = time.time()
    print(f'fold {fold}  tr_seq={len(tr_seq)} va_seq={len(va_seq)}', flush=True)
    fold_oof  = np.zeros(len(train), dtype=np.float64)
    fold_test = np.zeros(len(test),  dtype=np.float64)
    for seed in TFM_SEEDS:
        torch.manual_seed(seed + fold); np.random.seed(seed + fold)
        model = PitTransformerV2(
            len(NUM_COLS), cat_vocab,
            d_model=96, n_heads=4, n_layers=3, dim_ff=192, dropout=0.1,
            drop_path_max=0.1, feat_dropout_p=0.1,
        ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
        scaler = torch.amp.GradScaler('cuda')
        ema = EMA(model, decay=EMA_DECAY)

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
                # Label smoothing only on the loss target.
                y_smooth = y_b * (1.0 - LABEL_SMOOTH) + (LABEL_SMOOTH / 2.0)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(num_b, cat_b, pos_b, kpm, causal)
                    bce = F.binary_cross_entropy_with_logits(logits, y_smooth, reduction='none')
                    denom = lm_b.sum().clamp_min(1.0)
                    loss = (bce * lm_b).sum() / denom
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
                ema.update(model)
            sch.step()

            # Eval on EMA weights.
            ema.apply_shadow(model)
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
            ema.restore(model)

            if va_auc > best_va:
                best_va = va_auc; best_ep = ep; pat = 0
                best_state = ema.state()
            else:
                pat += 1
            if ep in (0, 19, 20, 39, 40, EPOCHS-1):
                print(f'    seed={seed} ep{ep:02d} ema_va_auc={va_auc:.5f} '
                      f'best={best_va:.5f}@ep{best_ep} lr={sch.get_last_lr()[0]:.2e}', flush=True)
            if pat >= PATIENCE:
                print(f'    seed={seed} early-stop ep{ep} best ep{best_ep} auc={best_va:.5f}', flush=True)
                break
        else:
            print(f'    seed={seed} done ep{EPOCHS-1} best ep{best_ep} auc={best_va:.5f}', flush=True)

        # Load best EMA weights and predict.
        sd = model.state_dict()
        for k, v in best_state.items():
            sd[k].copy_(v.to(DEVICE))
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
        del model, best_state, ema
        torch.cuda.empty_cache()

    fold_oof  /= len(TFM_SEEDS)
    fold_test /= len(TFM_SEEDS)
    tfm_oof  += fold_oof
    tfm_test += fold_test / N_FOLDS
    print(f'  fold time {time.time()-fstart:.0f}s', flush=True)

tfm_auc = roc_auc_score(y_full, tfm_oof)
print(f'\nT36 TFM OOF AUC = {tfm_auc:.5f}  ({time.time()-t_tfm:.0f}s total)', flush=True)
np.save('/tmp/trial36_tfm_oof.npy',  tfm_oof.astype(np.float32))
np.save('/tmp/trial36_tfm_test.npy', tfm_test.astype(np.float32))

# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------
print('\n=== Stack ===', flush=True)

def rank_avg(*arrs):
    L = len(arrs[0])
    s = np.zeros(L)
    for a in arrs:
        s += rankdata(a)
    return s / (len(arrs) * L)

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

gbdt_oof  = rank_avg(xgb_oof,  cb_oof)
gbdt_test = rank_avg(xgb_test, cb_test)
gbdt_auc = roc_auc_score(y_full, gbdt_oof)
print(f'GBDT rank-avg OOF AUC = {gbdt_auc:.5f}', flush=True)

# LR-meta cross-validated.
oof_stack  = np.column_stack([logit(gbdt_oof),  logit(tfm_oof)])
test_stack = np.column_stack([logit(gbdt_test), logit(tfm_test)])
meta_oof = np.zeros(len(train), dtype=np.float64)
for tr_idx, va_idx in row_splits:
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(oof_stack[tr_idx], y_full[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(oof_stack[va_idx])[:, 1]
meta_auc = roc_auc_score(y_full, meta_oof)
print(f'LR-meta(GBDT, T36_TFM) OOF AUC = {meta_auc:.5f}', flush=True)

# 3-way rank-avg.
three_oof  = rank_avg(xgb_oof,  cb_oof,  tfm_oof)
three_test = rank_avg(xgb_test, cb_test, tfm_test)
three_auc  = roc_auc_score(y_full, three_oof)
print(f'3-way rank-avg(XGB, CB, T36_TFM) OOF AUC = {three_auc:.5f}', flush=True)

# Final stacker on full OOF, applied to test.
meta_full = LogisticRegression(C=1.0, max_iter=1000).fit(oof_stack, y_full)
meta_test = meta_full.predict_proba(test_stack)[:, 1]

# Pick best blend for submission.
candidates = {
    'gbdt_rank_avg': (gbdt_auc, gbdt_test),
    'lr_meta':       (meta_auc, meta_test),
    '3way_rank_avg': (three_auc, three_test),
}
best_name = max(candidates, key=lambda k: candidates[k][0])
best_auc, best_test = candidates[best_name]

print(f'\n==== T36 RESULTS ====', flush=True)
print(f'T36 TFM   OOF AUC : {tfm_auc:.5f}  (vs T35 TFM 0.93542 -> delta {tfm_auc-0.93542:+.5f})', flush=True)
print(f'GBDT      OOF AUC : {gbdt_auc:.5f}', flush=True)
print(f'LR-meta   OOF AUC : {meta_auc:.5f}  (vs T35 LR-meta 0.94913 -> delta {meta_auc-0.94913:+.5f})', flush=True)
print(f'3-way     OOF AUC : {three_auc:.5f}', flush=True)
print(f'Best blend        : {best_name} @ {best_auc:.5f}', flush=True)
print(f'AP                : {average_precision_score(y_full, candidates[best_name][1] if False else meta_oof if best_name=="lr_meta" else (three_oof if best_name=="3way_rank_avg" else gbdt_oof)):.5f}', flush=True)
oof_for_metrics = meta_oof if best_name == 'lr_meta' else (three_oof if best_name == '3way_rank_avg' else gbdt_oof)
print(f'AP                : {average_precision_score(y_full, oof_for_metrics):.5f}', flush=True)
print(f'LL                : {log_loss(y_full, np.clip(oof_for_metrics, 1e-6, 1-1e-6)):.5f}', flush=True)
print(f'TOTAL             : {time.time()-t0:.0f}s', flush=True)

sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: best_test})
sub.to_csv(DATA_DIR / 'submission.csv', index=False)
print(f'wrote submission.csv from {best_name}', flush=True)
