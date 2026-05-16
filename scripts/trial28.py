"""Trial 28: stacked features — add T26's OOF preds as 3 input features (xgb_logit, cb_logit, ens_logit).
T21 base setup. OOF preds are leak-free under StratifiedGroupKFold.
"""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import catboost as cb
from scipy.stats import rankdata
from scipy.special import logit

t0 = time.time()
DATA_DIR = Path('/home/propar/Documents/Projects/F1-pitstop-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
TARGET, ID_COL = 'PitNextLap', 'id'

def add_features(df):
    df = df.copy(); eps = 1e-6
    df['RemainingRace'] = 1.0 - df['RaceProgress']
    df['PitWindow'] = df['RaceProgress'] * (1 - df['RaceProgress'])
    df['IsLateRace'] = (df['RaceProgress'] > 0.7).astype(int)
    df['LapTime_per_TyreLife'] = df['LapTime (s)'] / (df['TyreLife'] + eps)
    df['Deg_per_TyreLife'] = df['Cumulative_Degradation'] / (df['TyreLife'] + eps)
    df['TyreStress'] = df['TyreLife'] * df['Cumulative_Degradation']
    df['StrategicUrgency'] = df['TyreStress'] * df['RemainingRace']
    df['TyreExhaustion'] = (df['TyreLife'] ** 2) * df['RemainingRace']
    df['TyreCliff'] = df['TyreLife'] * df['LapTime_Delta']
    df['PositionPressure'] = df['Position'] * df['RemainingRace']
    df['RecoveryPressure'] = abs(df['Position_Change']) * df['Cumulative_Degradation']
    compound_map = {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3}
    df['CompoundCode'] = df['Compound'].astype(str).str.upper().map(compound_map).fillna(2)
    df['CompoundTyreInteraction'] = df['CompoundCode'] * df['TyreLife']
    df['StrategyPressure'] = df['TyreStress'] * df['PitWindow']
    df['PitOffsetPotential'] = df['LapTime_Delta'] * df['RemainingRace'] * 10
    df['UndercutPotential'] = df['LapTime_Delta'] * abs(df['Position_Change']) * df['RemainingRace']
    df['StintSurvivalPressure'] = df['TyreLife'] * df['RemainingRace'] * df['LapTime_Delta']
    df['PaceCollapse'] = df['LapTime_Delta'] * df['Cumulative_Degradation']
    df['LateRaceTyreRisk'] = df['IsLateRace'] * df['TyreLife']
    return df

train_fe = add_features(train); test_fe = add_features(test)
CAT_COLS = ['Driver', 'Compound', 'Race']
for c in CAT_COLS:
    train_fe[c] = train_fe[c].astype('category')
    test_fe[c]  = test_fe[c].astype('category')

# Load T26 OOF + test preds, derive logits and ensemble logit, add as features
oof_xgb_t26 = np.load('/tmp/trial26_oof_xgb.npy')
oof_cb_t26  = np.load('/tmp/trial26_oof_cb.npy')
test_xgb_t26 = np.load('/tmp/trial26_test_xgb.npy')
test_cb_t26  = np.load('/tmp/trial26_test_cb.npy')

def safe_logit(p, eps=1e-6): return logit(np.clip(p, eps, 1-eps))

# Ensemble logit via LR-meta fit on full OOF (this is leak-free for use as a feature
# at inference; for OOF use it'd be slightly leaky if refit on full y. Instead:
# compute OOF ensemble via per-fold LR-meta, and full-train LR-meta for test only.)
y = train_fe[TARGET]
groups = (train_fe['Race'].astype(str) + '_' + train_fe['Year'].astype(str) + '_' + train_fe['Driver'].astype(str))
cv_meta = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
Z_train = np.column_stack([safe_logit(oof_xgb_t26), safe_logit(oof_cb_t26)])
ens_oof = np.zeros(len(y))
for tr, va in cv_meta.split(Z_train, y, groups):
    lr = LogisticRegression(C=1.0, max_iter=1000); lr.fit(Z_train[tr], y.iloc[tr])
    ens_oof[va] = lr.predict_proba(Z_train[va])[:, 1]
lr_full = LogisticRegression(C=1.0, max_iter=1000); lr_full.fit(Z_train, y)
ens_test = lr_full.predict_proba(np.column_stack([safe_logit(test_xgb_t26), safe_logit(test_cb_t26)]))[:, 1]

train_fe['stk_xgb_logit'] = safe_logit(oof_xgb_t26).astype(np.float32)
train_fe['stk_cb_logit']  = safe_logit(oof_cb_t26).astype(np.float32)
train_fe['stk_ens_logit'] = safe_logit(ens_oof).astype(np.float32)
test_fe['stk_xgb_logit']  = safe_logit(test_xgb_t26).astype(np.float32)
test_fe['stk_cb_logit']   = safe_logit(test_cb_t26).astype(np.float32)
test_fe['stk_ens_logit']  = safe_logit(ens_test).astype(np.float32)

DROP_COLS = [ID_COL, TARGET, 'LapNumber']
FEATURES = [c for c in train_fe.columns if c not in DROP_COLS]
print(f'{len(FEATURES)} features (T21 base + 3 stacked)', flush=True)

X = train_fe[FEATURES]; X_test = test_fe[FEATURES]

SEEDS = [42, 2024]; N_SPLITS = 5

def make_xgb_params(seed):
    return dict(n_estimators=6000, learning_rate=0.03, max_depth=7,
                subsample=0.9, colsample_bytree=0.9, min_child_weight=20,
                reg_alpha=0.1, reg_lambda=1.0,
                objective='binary:logistic', eval_metric='auc', seed=seed,
                tree_method='hist', n_jobs=8, early_stopping_rounds=200)
def make_cb_params(seed):
    return dict(iterations=6000, learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
                loss_function='Logloss', eval_metric='AUC', random_seed=seed,
                early_stopping_rounds=200, thread_count=8, verbose=0)

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(train_fe)); test_xgb = np.zeros(len(test_fe))
oof_cb  = np.zeros(len(train_fe)); test_cb  = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
    fstart = time.time()
    print(f'\n==== FOLD {fold} ====', flush=True)
    X_tr = X.iloc[tr_idx]; X_va = X.iloc[va_idx]
    y_tr = y.iloc[tr_idx]; y_va = y.iloc[va_idx]
    X_te_f = X_test

    X_tr_xgb = X_tr.copy(); X_va_xgb = X_va.copy(); X_te_xgb = X_te_f.copy()
    for c in CAT_COLS:
        X_tr_xgb[c] = X_tr_xgb[c].cat.codes
        X_va_xgb[c] = X_va_xgb[c].cat.codes
        X_te_xgb[c] = X_te_xgb[c].cat.codes
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]

    fold_xgb_va = np.zeros(len(va_idx)); fold_xgb_te = np.zeros(len(test_fe))
    fold_cb_va  = np.zeros(len(va_idx)); fold_cb_te  = np.zeros(len(test_fe))

    for s in SEEDS:
        ss = time.time()
        m = xgb.XGBClassifier(**make_xgb_params(s))
        m.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
        fold_xgb_va += m.predict_proba(X_va_xgb)[:, 1] / len(SEEDS)
        fold_xgb_te += m.predict_proba(X_te_xgb)[:, 1] / len(SEEDS)
        print(f'  XGB seed={s} best_iter={m.best_iteration} t={time.time()-ss:.1f}s', flush=True)

        ss = time.time()
        cm = cb.CatBoostClassifier(**make_cb_params(s))
        cm.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_idx, verbose=False)
        fold_cb_va += cm.predict_proba(X_va)[:, 1] / len(SEEDS)
        fold_cb_te += cm.predict_proba(X_te_f)[:, 1] / len(SEEDS)
        print(f'  CB  seed={s} best_iter={cm.get_best_iteration()} t={time.time()-ss:.1f}s', flush=True)

    oof_xgb[va_idx] = fold_xgb_va; oof_cb[va_idx] = fold_cb_va
    test_xgb += fold_xgb_te / N_SPLITS; test_cb += fold_cb_te / N_SPLITS
    print(f'  fold XGB AUC: {roc_auc_score(y_va, fold_xgb_va):.5f}', flush=True)
    print(f'  fold CB  AUC: {roc_auc_score(y_va, fold_cb_va):.5f}', flush=True)
    print(f'  fold time: {time.time()-fstart:.1f}s', flush=True)

print('\n==== FINAL ====', flush=True)
print(f'XGB OOF AUC : {roc_auc_score(y, oof_xgb):.5f}', flush=True)
print(f'CB  OOF AUC : {roc_auc_score(y, oof_cb):.5f}', flush=True)

# Ensemble: rank-avg, LR-meta
def ranked(p): return rankdata(p) / len(p)
oof_rank = (ranked(oof_xgb) + ranked(oof_cb)) / 2
test_rank = (ranked(test_xgb) + ranked(test_cb)) / 2
Z2 = np.column_stack([safe_logit(oof_xgb), safe_logit(oof_cb)])
oof_meta = np.zeros(len(y))
for tr, va in cv.split(Z2, y, groups):
    lrm = LogisticRegression(C=1.0, max_iter=1000); lrm.fit(Z2[tr], y.iloc[tr])
    oof_meta[va] = lrm.predict_proba(Z2[va])[:, 1]
lrf = LogisticRegression(C=1.0, max_iter=1000); lrf.fit(Z2, y)
test_meta = lrf.predict_proba(np.column_stack([safe_logit(test_xgb), safe_logit(test_cb)]))[:, 1]
print(f'rank-avg : {roc_auc_score(y, oof_rank):.5f}', flush=True)
print(f'LR-meta  : {roc_auc_score(y, oof_meta):.5f}', flush=True)

if roc_auc_score(y, oof_meta) >= roc_auc_score(y, oof_rank):
    chosen = 'lr-meta'; oof_chosen = oof_meta; test_chosen = test_meta
else:
    chosen = 'rank-avg'; oof_chosen = oof_rank; test_chosen = test_rank
print(f'CHOSEN: {chosen} OOF AUC {roc_auc_score(y, oof_chosen):.5f}', flush=True)
print(f'  AP: {average_precision_score(y, oof_chosen):.5f}', flush=True)
print(f'  LL: {log_loss(y, oof_chosen):.5f}', flush=True)
print(f'TOTAL: {time.time()-t0:.1f}s', flush=True)

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_chosen})
sub.to_csv(DATA_DIR / 'submission.csv', index=False)
np.save('/tmp/trial28_oof_xgb.npy', oof_xgb)
np.save('/tmp/trial28_oof_cb.npy', oof_cb)
np.save('/tmp/trial28_test_xgb.npy', test_xgb)
np.save('/tmp/trial28_test_cb.npy', test_cb)
print('submission saved', flush=True)
