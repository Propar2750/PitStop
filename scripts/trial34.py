"""Trial 34: GPU multi-seed bag of the T31 transformer.

Same architecture and inputs as T31 (3 layers, d_model=96, 4 heads, dim_ff=192,
causal-masked encoder over per-(Year,Race,Driver) lap sequences). The only
deliberate change vs T31 is a 3-seed bag (42, 2024, 7) averaged at sigmoid
output, run on RTX 5060 with tensors resident on GPU.

Rationale:
  - T31 standalone OOF 0.93430 added +0.00024 to the GBDT ensemble via
    LR-meta. Per-fold best epochs landed at 8-12 with patience 8 — single
    seed, noisy. Seed-bagging variance-reduces the transformer the same
    way T25 reduced CB variance (~+0.00017 per extra pair of seeds in CB).
  - T32 confirmed scaling up overfits. So we keep T31's scale and spend
    GPU budget on seeds instead of capacity.
  - GPU residency + AMP makes 3 seeds × 5 folds tractable; CPU made even
    1 seed take 50 min.

CV: StratifiedGroupKFold on (Race,Year,Driver) — identical split to T31.
Outputs /tmp/trial34_oof.npy and /tmp/trial34_test.npy for ensembling.
"""
import os, time, warnings
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
assert torch.cuda.is_available(), 'CUDA required for T34'
DEVICE = torch.device('cuda')
print(f'device={torch.cuda.get_device_name(0)}  torch={torch.__version__}', flush=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DATA_DIR = Path('/home/propar/Documents/Projects/F1-pitstop-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
TARGET, ID_COL = 'PitNextLap', 'id'

train['_is_train'] = 1
test['_is_train']  = 0
test[TARGET]       = -1
df = pd.concat([train, test], ignore_index=True)
df = df.sort_values(['Year','Race','Driver','LapNumber']).reset_index(drop=True)
print(f'combined {len(df):,} rows', flush=True)

eps = 1e-6
df['RemainingRace'] = 1.0 - df['RaceProgress']
df['TyreStress']    = df['TyreLife'] * df['Cumulative_Degradation']
df['LapTime_per_TyreLife'] = df['LapTime (s)'] / (df['TyreLife'] + eps)
df['PitWindow']     = df['RaceProgress'] * (1 - df['RaceProgress'])

NUM_COLS = ['TyreLife','Position','LapTime (s)','LapTime_Delta','Cumulative_Degradation',
            'RaceProgress','Position_Change','Stint','PitStop',
            'RemainingRace','TyreStress','LapTime_per_TyreLife','PitWindow']

for c in NUM_COLS:
    mu, sd = df[c].mean(), df[c].std() + 1e-6
    df[c] = ((df[c] - mu) / sd).astype(np.float32)

CAT_COLS = ['Driver','Compound','Race','Year']
cat_vocab = {}
for c in CAT_COLS:
    cats = pd.Categorical(df[c].astype(str))
    df[c+'_id'] = cats.codes.astype(np.int64)
    cat_vocab[c] = len(cats.categories)
print(f'vocab sizes: {cat_vocab}', flush=True)

df['_seq'] = df.groupby(['Year','Race','Driver']).ngroup()
n_seq = df['_seq'].max() + 1
seq_lens = df.groupby('_seq').size()
MAX_LEN = int(seq_lens.max())
print(f'{n_seq} sequences; max len {MAX_LEN}; mean len {seq_lens.mean():.1f}', flush=True)

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
    num_arr[sid, :L]  = g[NUM_COLS].values
    cat_arr[sid, :L]  = g[cat_id_cols].values
    pos_arr[sid, :L]  = g['LapNumber'].values.clip(0, 199)
    y_arr[sid, :L]    = np.where(g['_is_train'].values==1, g[TARGET].values, 0.0)
    mask_arr[sid, :L] = 1.0
    loss_mask[sid, :L]= g['_is_train'].values.astype(np.float32)
    g_reset = g.reset_index(drop=True)
    is_tr = g_reset['_is_train'].values == 1
    tr_pos = np.where(is_tr)[0]
    te_pos = np.where(~is_tr)[0]
    seq_train_pos[sid] = tr_pos.astype(np.int64)
    seq_test_pos[sid]  = te_pos.astype(np.int64)
    seq_train_idx[sid] = np.array([train_id_to_pos[t] for t in g_reset.loc[is_tr, ID_COL].values], dtype=np.int64)
    seq_test_idx[sid]  = np.array([test_id_to_pos[t]  for t in g_reset.loc[~is_tr, ID_COL].values], dtype=np.int64)

print(f'tensors built in {time.time()-t0:.1f}s', flush=True)

# Pin all sequence tensors to GPU once (fits easily on 8GB).
num_g = torch.from_numpy(num_arr).to(DEVICE)
cat_g = torch.from_numpy(cat_arr).to(DEVICE)
pos_g = torch.from_numpy(pos_arr).to(DEVICE)
y_g   = torch.from_numpy(y_arr).to(DEVICE)
m_g   = torch.from_numpy(mask_arr).to(DEVICE)
lm_g  = torch.from_numpy(loss_mask).to(DEVICE)
print(f'resident GPU mem: {torch.cuda.memory_allocated()/1e6:.1f} MB', flush=True)

class PitTransformer(nn.Module):
    def __init__(self, n_num, cat_vocab, d_model=96, n_heads=4, n_layers=3, dim_ff=192,
                 dropout=0.1, max_pos=200):
        super().__init__()
        self.num_proj = nn.Linear(n_num, d_model)
        self.cat_emb = nn.ModuleList([nn.Embedding(v, d_model) for v in cat_vocab.values()])
        self.pos_emb = nn.Embedding(max_pos, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=dim_ff, dropout=dropout,
                                               batch_first=True, activation='gelu',
                                               norm_first=True)
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

seq_first_idx = df.groupby('_seq').head(1).reset_index(drop=True)
seq_groups = (seq_first_idx['Race'].astype(str) + '_' + seq_first_idx['Year'].astype(str)
              + '_' + seq_first_idx['Driver'].astype(str)).values
seq_y = (y_arr.sum(axis=1) > 0).astype(int)

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

SEEDS = [42, 2024, 7]
EPOCHS = 35
BATCH = 256
LR = 2e-3
PATIENCE = 8

oof_train = np.zeros(len(train), dtype=np.float64)  # sum across seeds, divide at end
test_pred = np.zeros(len(test),  dtype=np.float64)  # sum across seeds and folds

def make_causal_mask(L, device):
    return torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)

causal = make_causal_mask(MAX_LEN, DEVICE)

for fold, (tr_seq, va_seq) in enumerate(cv.split(np.arange(N), seq_y, seq_groups)):
    fstart = time.time()
    print(f'\n==== FOLD {fold}  train_seq={len(tr_seq)} val_seq={len(va_seq)} ====', flush=True)

    fold_oof = np.zeros(len(train), dtype=np.float64)
    fold_test = np.zeros(len(test), dtype=np.float64)

    for seed in SEEDS:
        torch.manual_seed(seed + fold); np.random.seed(seed + fold)
        model = PitTransformer(len(NUM_COLS), cat_vocab).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        scaler = torch.amp.GradScaler('cuda')

        best_va_auc = 0.0; best_state = None; best_epoch = -1; patience = 0
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(tr_seq)
            ep_loss = 0.0; ep_count = 0.0
            for i in range(0, len(perm), BATCH):
                b = torch.from_numpy(perm[i:i+BATCH]).to(DEVICE)
                num_b = num_g.index_select(0, b)
                cat_b = cat_g.index_select(0, b)
                pos_b = pos_g.index_select(0, b)
                y_b   = y_g.index_select(0, b)
                m_b   = m_g.index_select(0, b)
                lm_b  = lm_g.index_select(0, b)
                kpm   = (m_b == 0)
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
                ep_loss += loss.item() * denom.item(); ep_count += denom.item()
            sch.step()
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                preds_all = []; ys_all = []
                for i in range(0, len(va_seq), BATCH):
                    b = torch.from_numpy(va_seq[i:i+BATCH]).to(DEVICE)
                    num_b = num_g.index_select(0, b); cat_b = cat_g.index_select(0, b)
                    pos_b = pos_g.index_select(0, b); m_b = m_g.index_select(0, b)
                    kpm = (m_b == 0)
                    logits = model(num_b, cat_b, pos_b, kpm, causal)
                    p = torch.sigmoid(logits.float()).cpu().numpy()
                    lm_b = lm_g.index_select(0, b).cpu().numpy()
                    y_b = y_g.index_select(0, b).cpu().numpy()
                    sel = lm_b == 1
                    preds_all.append(p[sel]); ys_all.append(y_b[sel])
                va_p = np.concatenate(preds_all); va_y = np.concatenate(ys_all)
                va_auc = roc_auc_score(va_y, va_p)
            avg_loss = ep_loss / max(ep_count, 1)
            print(f'  seed={seed} ep {ep:02d} loss={avg_loss:.4f} va_auc={va_auc:.5f}'
                  f' lr={sch.get_last_lr()[0]:.2e}', flush=True)
            if va_auc > best_va_auc:
                best_va_auc = va_auc; best_epoch = ep; patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f'  seed={seed} early-stop ep {ep} best ep {best_epoch} auc {best_va_auc:.5f}', flush=True)
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            for i in range(0, len(va_seq), BATCH):
                b_np = va_seq[i:i+BATCH]
                b = torch.from_numpy(b_np).to(DEVICE)
                num_b = num_g.index_select(0, b); cat_b = cat_g.index_select(0, b)
                pos_b = pos_g.index_select(0, b); m_b = m_g.index_select(0, b)
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
                pos_b = pos_g.index_select(0, b); m_b = m_g.index_select(0, b)
                kpm = (m_b == 0)
                logits = model(num_b, cat_b, pos_b, kpm, causal)
                p = torch.sigmoid(logits.float()).cpu().numpy()
                for k, sid in enumerate(b_np):
                    tp = seq_test_pos[sid]; ti = seq_test_idx[sid]
                    if len(tp): fold_test[ti] += p[k, tp]

        del model, best_state
        torch.cuda.empty_cache()

    fold_oof /= len(SEEDS)
    fold_test /= len(SEEDS)
    oof_train += fold_oof  # disjoint val rows across folds — additive is fine
    test_pred += fold_test / 5.0
    print(f'  fold time {time.time()-fstart:.1f}s', flush=True)

y_full = train[TARGET].values
oof_auc = roc_auc_score(y_full, oof_train)
print('\n==== FINAL ====', flush=True)
print(f'OOF AUC : {oof_auc:.5f}', flush=True)
print(f'AP      : {average_precision_score(y_full, oof_train):.5f}', flush=True)
print(f'LL      : {log_loss(y_full, np.clip(oof_train,1e-6,1-1e-6)):.5f}', flush=True)
print(f'TOTAL: {time.time()-t0:.1f}s', flush=True)

np.save('/tmp/trial34_oof.npy',  oof_train.astype(np.float32))
np.save('/tmp/trial34_test.npy', test_pred.astype(np.float32))
print('saved /tmp/trial34_oof.npy /tmp/trial34_test.npy', flush=True)
