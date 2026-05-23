"""Trial 58: Pseudo-labeled CatBoost (parallel to T57 pseudo-XGB).

T57 confirmed pseudo-labels lift the GBDT slot too (+0.00038 standalone on
XGBoost). Apply same lever to CatBoost (T35's GPU CB recipe, depth=8 lr=0.03
6000/200 l2=5.0 task_type=GPU, 2-seed bag), with pseudo-positives from the
strong best7way_v2 source (>0.80 threshold, 17125 rows).
"""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from catboost import CatBoostClassifier, Pool

DATA_DIR = Path(__file__).resolve().parent.parent
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
print(f'train {len(train):,}  test {len(test):,}', flush=True)

ID = 'id'; TARGET = 'PitNextLap'
NUM = ['Year','PitStop','Stint','TyreLife','Position',
       'LapTime (s)','LapTime_Delta','Cumulative_Degradation',
       'RaceProgress','Position_Change']
CATS = ['Driver','Compound','Race']
FEATS = NUM + CATS

# CatBoost: keep cats as int dtype in a DataFrame (Pool handles mixed dtypes)
combined = pd.concat([train[FEATS], test[FEATS]], axis=0).reset_index(drop=True)
for c in CATS:
    combined[c] = combined[c].astype('category').cat.codes.astype(np.int32)
for c in NUM:
    combined[c] = combined[c].astype(np.float32)
    if combined[c].isna().any():
        combined[c] = combined[c].fillna(combined[c].median())

X_train_df = combined.iloc[:len(train)].reset_index(drop=True)
X_test_df  = combined.iloc[len(train):].reset_index(drop=True)
y_train = train[TARGET].values
groups = (train['Race'].astype(str)+'_'+train['Year'].astype(str)+'_'+train['Driver'].astype(str)).values

cat_idx = [FEATS.index(c) for c in CATS]

# Pseudo
PSEUDO_THR = 0.80
best7way = np.load('/tmp/best7way_v2_test.npy')
pmask = best7way > PSEUDO_THR
X_pseudo_df = X_test_df.iloc[pmask].reset_index(drop=True)
y_pseudo = np.ones(pmask.sum(), dtype=np.int64)
print(f'pseudo-positives: {pmask.sum()}', flush=True)

cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
SEEDS = [42, 2024]
oof = np.zeros(len(y_train), dtype=np.float32)
test_pred = np.zeros(len(X_test_df), dtype=np.float32)

t0 = time.time()
for fold, (tr, va) in enumerate(cv.split(X_train_df, y_train, groups=groups)):
    Xa = pd.concat([X_train_df.iloc[tr], X_pseudo_df], axis=0).reset_index(drop=True)
    ya = np.concatenate([y_train[tr], y_pseudo])
    # restore int dtype after concat
    for c in CATS:
        Xa[c] = Xa[c].astype(np.int32)
    Xv = X_train_df.iloc[va]; yv = y_train[va]
    for seed in SEEDS:
        clf = CatBoostClassifier(
            iterations=6000, depth=8, learning_rate=0.03,
            l2_leaf_reg=5.0, bootstrap_type='Bernoulli', subsample=0.9,
            od_type='Iter', od_wait=200, task_type='GPU',
            eval_metric='AUC', random_seed=seed, verbose=0,
            cat_features=cat_idx,
        )
        train_pool = Pool(Xa, ya, cat_features=cat_idx)
        val_pool   = Pool(Xv, yv, cat_features=cat_idx)
        clf.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va] += clf.predict_proba(Xv)[:,1] / len(SEEDS)
        test_pred += clf.predict_proba(X_test_df)[:,1] / (5 * len(SEEDS))
        print(f'  fold {fold} seed={seed} best_iter={clf.best_iteration_}', flush=True)
    print(f'  fold {fold} done  va_auc={roc_auc_score(yv, oof[va]):.5f}  t={time.time()-t0:.0f}s', flush=True)

print(f'\nOOF AUC : {roc_auc_score(y_train, oof):.5f}')
print(f'OOF AP  : {average_precision_score(y_train, oof):.5f}')
print(f'OOF LL  : {log_loss(y_train, np.clip(oof, 1e-7, 1-1e-7)):.5f}')

np.save('/tmp/trial58_oof.npy', oof)
np.save('/tmp/trial58_test.npy', test_pred)
print('saved', flush=True)
