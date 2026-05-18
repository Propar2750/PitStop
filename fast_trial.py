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
TARGET, ID_COL = 'PitNextLap', 'id'

# === feature engineering (edit me to test ideas) ===
def add_features(df):
    df = df.copy(); eps = 1e-6
    df['TyreCliff'] = df['TyreLife'] * df['LapTime_Delta']
    return df

train_fe = add_features(train)
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
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
    oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
    print(f'  fold {fold} AUC={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f} '
          f'best_iter={m.best_iteration} t={time.time()-fs:.1f}s', flush=True)

print('\n==== RESULT ====', flush=True)
print(f'OOF AUC : {roc_auc_score(y, oof):.5f}', flush=True)
print(f'AP      : {average_precision_score(y, oof):.5f}', flush=True)
print(f'LogLoss : {log_loss(y, oof):.5f}', flush=True)
print(f'TOTAL   : {time.time()-t0:.1f}s', flush=True)