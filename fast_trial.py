"""Fast trial harness: XGBoost-only, lightly subsampled, for quick feature experiments.

Goal: score as close as possible to the best full pipeline (trial 30 XGB+CB ensemble,
OOF ~0.9492) while training in a fraction of the time. We drop CatBoost, drop the
2-seed average, and lightly subsample by group — but keep the same XGB hyperparams
and CV scheme as the best run so deltas from feature changes are informative.

NOT for trials.csv (not directly comparable to baseline). Use to screen feature
ideas, then promote winners to a full trial script.

Speed knobs:
  SAMPLE_FRAC : fraction of train *groups* (Race+Year+Driver stints) kept.
                Sampling by group preserves the no-leak CV property.
  N_SPLITS    : keep at 5 to match baseline; drop to 3 for extra speed.
  Single seed, single model (XGB).

Edit `add_features` to test feature ideas.
"""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import xgboost as xgb

# ------------------------- speed knobs -------------------------
SAMPLE_FRAC = 1      # fraction of stints kept; 1.0 = no subsampling
N_SPLITS    = 5
SEED        = 42
# ---------------------------------------------------------------

t0 = time.time()
DATA_DIR = Path('/home/propar/Documents/Projects/F1-pitstop-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
TARGET, ID_COL = 'PitNextLap', 'id'

# === feature engineering (edit me to test ideas) ===
def add_features(df):
    df = df.copy(); eps = 1e-6
    return df

# =============================================================
# DATA CLEANING — Modified Z-score (MAD-based) outlier flagging
# =============================================================
# Produce a boolean keep-mask on train; do NOT drop rows. The CV loop fits
# on the clean subset of each fold's train split, but validates on every row
# of the val split -- so OOF AUC remains directly comparable to baseline,
# which is computed over the entire train set.
#
# Test is never touched.
TARGET_RATE = 0.000  # ~1% of train rows flagged

# Dtype coercions used downstream.
if 'Year' in train.columns:
    train['Year'] = train['Year'].astype('int32')
if 'Stint' in train.columns:
    train['Stint'] = pd.to_numeric(train['Stint'], errors='coerce').astype('Int32')

def _modified_z_outliers(df, group_col, value_col, target_rate):
    g = df.groupby(group_col)[value_col]
    med = g.transform('median')
    abs_dev = (df[value_col] - med).abs()
    mad = abs_dev.groupby(df[group_col]).transform('median')
    eps_floor = 1e-9 * med.clip(lower=1.0)
    mad_safe = mad.where(mad > 0, eps_floor)
    mod_z = 0.6745 * (df[value_col] - med) / mad_safe
    thresh = float(mod_z.abs().quantile(1 - target_rate))
    return mod_z.abs() > thresh, thresh

train_clean_mask = pd.Series(True, index=train.index)
if {'Race', 'LapTime (s)'}.issubset(train.columns):
    sub = train.dropna(subset=['LapTime (s)'])
    outlier_mask, thresh = _modified_z_outliers(
        sub, 'Race', 'LapTime (s)', TARGET_RATE,
    )
    drop_idx = outlier_mask[outlier_mask].index
    train_clean_mask.loc[drop_idx] = False
    print(f'  Modified-Z threshold: |Z| > {thresh:.2f}  '
          f'-> flagged {(~train_clean_mask).sum():,} of {len(train):,} train rows '
          f'({(~train_clean_mask).mean()*100:.2f}%) — excluded from fitting, kept for validation')
print('train shape (unchanged):', train.shape, '| test shape (unchanged):', test.shape)

train_fe = add_features(train)
# Carry the clean-mask alongside train_fe so subsample re-indexing keeps them aligned.
train_fe['_clean'] = train_clean_mask.reindex(train_fe.index).fillna(True).to_numpy()

CAT_COLS = ['Driver', 'Compound', 'Race']
for c in CAT_COLS:
    train_fe[c] = train_fe[c].astype('category')

groups_full = (train_fe['Race'].astype(str) + '_' + train_fe['Year'].astype(str) + '_' + train_fe['Driver'].astype(str))

# === subsample by group (preserves no-leak CV) ===
if SAMPLE_FRAC < 1.0:
    rng = np.random.RandomState(SEED)
    uniq_groups = groups_full.unique()
    keep_groups = set(rng.choice(uniq_groups, size=int(len(uniq_groups) * SAMPLE_FRAC), replace=False))
    mask = groups_full.isin(keep_groups)
    train_fe = train_fe.loc[mask].reset_index(drop=True)
    groups = groups_full.loc[mask].reset_index(drop=True)
    print(f'sampled {len(train_fe):,} rows ({len(keep_groups):,} groups, frac={SAMPLE_FRAC})', flush=True)
else:
    groups = groups_full
    print(f'no subsampling: {len(train_fe):,} rows', flush=True)

clean_mask_arr = train_fe['_clean'].to_numpy()
train_fe = train_fe.drop(columns=['_clean'])

DROP_COLS = [ID_COL, TARGET]
FEATURES = [c for c in train_fe.columns if c not in DROP_COLS]
y = train_fe[TARGET]
X = train_fe[FEATURES].copy()
for c in CAT_COLS:
    if c in X.columns:
        X[c] = X[c].cat.codes
print(f'features ({len(FEATURES)}): {FEATURES}', flush=True)

# XGB params matched to trial-30 best config (lr=0.03, depth=7) — keeps quality.
xgb_params = dict(n_estimators=6000, learning_rate=0.03, max_depth=7,
                  subsample=0.9, colsample_bytree=0.9, min_child_weight=20,
                  reg_alpha=0.1, reg_lambda=1.0,
                  objective='binary:logistic', eval_metric='auc', seed=SEED,
                  tree_method='hist', n_jobs=8, early_stopping_rounds=200)

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train_fe))

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
    fs = time.time()
    # Fit on CLEAN rows of the train split; validate on ALL rows of the val
    # split (incl. outliers) so OOF AUC stays comparable to the full-data baseline.
    tr_idx_fit = tr_idx[clean_mask_arr[tr_idx]]
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X.iloc[tr_idx_fit], y.iloc[tr_idx_fit],
          eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
    oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
    n_dropped = len(tr_idx) - len(tr_idx_fit)
    print(f'  fold {fold} AUC={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f} '
          f'best_iter={m.best_iteration} fit_rows={len(tr_idx_fit):,} '
          f'(skipped {n_dropped:,} outliers) t={time.time()-fs:.1f}s', flush=True)

print('\n==== RESULT ====', flush=True)
print(f'OOF AUC : {roc_auc_score(y, oof):.5f}', flush=True)
print(f'AP      : {average_precision_score(y, oof):.5f}', flush=True)
print(f'LogLoss : {log_loss(y, oof):.5f}', flush=True)
print(f'TOTAL   : {time.time()-t0:.1f}s', flush=True)