"""Trial 57: Pseudo-labeled XGBoost (T35 raw-features recipe + pseudo from best7way_v2).

Stack contribution from GBDT family has been the limiting factor since T54
won the standalone slot. Pseudo-labels worked +0.00029 standalone on
RealMLP (T46->T54). Same lever applied to a GBDT base, using the strong
T55-7way pseudo source, should similarly lift the GBDT slot in the meta.

Recipe: T35 XGBoost only (GPU, raw 10 numerics + 3 cats label-encoded),
2-seed bag, + pseudo-positives (best7way_v2_test > 0.80, ~17k rows)
appended to each training fold. CV scheme + groups identical to T35.
"""
import time, warnings, os
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent.parent
print(f'xgb={xgb.__version__}', flush=True)

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
print(f'train {len(train):,}  test {len(test):,}', flush=True)

ID = 'id'; TARGET = 'PitNextLap'
NUM = ['Year', 'PitStop', 'Stint', 'TyreLife', 'Position',
       'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
       'RaceProgress', 'Position_Change']
CATS = ['Driver', 'Compound', 'Race']
FEATS = NUM + CATS

# label-encode cats consistently
combined = pd.concat([train[FEATS], test[FEATS]], axis=0).reset_index(drop=True)
for c in CATS:
    combined[c] = combined[c].astype('category').cat.codes.astype(np.int32)
for c in NUM:
    combined[c] = combined[c].astype(np.float32)
    if combined[c].isna().any():
        combined[c] = combined[c].fillna(combined[c].median())

X_train = combined.iloc[:len(train)].values.astype(np.float32)
X_test = combined.iloc[len(train):].values.astype(np.float32)
y_train = train[TARGET].values
groups = (train['Race'].astype(str)+'_'+train['Year'].astype(str)+'_'+train['Driver'].astype(str)).values

# Pseudo
PSEUDO_THR = 0.80
best7way = np.load('/tmp/best7way_v2_test.npy')
pmask = best7way > PSEUDO_THR
X_pseudo = X_test[pmask]
y_pseudo = np.ones(pmask.sum(), dtype=np.int64)
print(f'pseudo-positives: {pmask.sum()} ({100*pmask.mean():.1f}% of test)', flush=True)

# CV
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
SEEDS = [42, 2024]
oof = np.zeros(len(y_train), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)

t0 = time.time()
for fold, (tr, va) in enumerate(cv.split(X_train, y_train, groups=groups)):
    Xa = np.vstack([X_train[tr], X_pseudo])
    ya = np.concatenate([y_train[tr], y_pseudo])
    Xv = X_train[va]; yv = y_train[va]
    for seed in SEEDS:
        clf = xgb.XGBClassifier(
            n_estimators=6000, max_depth=7, learning_rate=0.03,
            min_child_weight=20, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method='hist', device='cuda', eval_metric='auc',
            early_stopping_rounds=200, random_state=seed, verbosity=0,
        )
        clf.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        bi = clf.best_iteration
        oof[va] += clf.predict_proba(Xv)[:,1] / len(SEEDS)
        test_pred += clf.predict_proba(X_test)[:,1] / (5 * len(SEEDS))
        print(f'  fold {fold} seed={seed} best_iter={bi}', flush=True)
    print(f'  fold {fold} done  va_auc={roc_auc_score(yv, oof[va]):.5f}  t={time.time()-t0:.0f}s', flush=True)

print(f'\n==== Trial 57 ====')
print(f'OOF AUC : {roc_auc_score(y_train, oof):.5f}')
print(f'OOF AP  : {average_precision_score(y_train, oof):.5f}')
print(f'OOF LL  : {log_loss(y_train, np.clip(oof, 1e-7, 1-1e-7)):.5f}')

np.save('/tmp/trial57_oof.npy', oof)
np.save('/tmp/trial57_test.npy', test_pred)
print('saved', flush=True)
