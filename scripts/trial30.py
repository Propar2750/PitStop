"""Trial 30: simplification probe.

Hypothesis: 32-feature set has redundancy. Train one XGB to get gain importances,
keep top-K features, then run full XGB+CB 2-seed pipeline. Goal: same OOF AUC
(~0.94919 T26) with materially fewer features → faster trial budget for downstream.
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

DROP_COLS = [ID_COL, TARGET, 'LapNumber']
ALL_FEATS = [c for c in train_fe.columns if c not in DROP_COLS]
y = train_fe[TARGET]
groups = (train_fe['Race'].astype(str) + '_' + train_fe['Year'].astype(str) + '_' + train_fe['Driver'].astype(str))

# === Step 1: single XGB feature-importance pass on full 32 ===
print(f'=== importance pass: {len(ALL_FEATS)} feats ===', flush=True)
X_all = train_fe[ALL_FEATS].copy()
for c in CAT_COLS: X_all[c] = X_all[c].cat.codes
cv_imp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr_imp, va_imp = next(iter(cv_imp.split(X_all, y, groups)))
ti = time.time()
m_imp = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=7,
                          subsample=0.9, colsample_bytree=0.9, min_child_weight=20,
                          reg_alpha=0.1, reg_lambda=1.0, tree_method='hist',
                          n_jobs=8, eval_metric='auc', early_stopping_rounds=100)
m_imp.fit(X_all.iloc[tr_imp], y.iloc[tr_imp],
          eval_set=[(X_all.iloc[va_imp], y.iloc[va_imp])], verbose=False)
print(f'  importance fit: {time.time()-ti:.1f}s, best_iter={m_imp.best_iteration}', flush=True)
raw_gain = m_imp.get_booster().get_score(importance_type='gain')
gain_arr = np.zeros(len(ALL_FEATS))
for k, v in raw_gain.items():
    idx = int(k[1:]) if k.startswith('f') else -1
    if 0 <= idx < len(ALL_FEATS):
        gain_arr[idx] = float(v)
imp_df = pd.DataFrame({'feat': ALL_FEATS, 'gain': gain_arr}).sort_values('gain', ascending=False).reset_index(drop=True)
print(imp_df.to_string(), flush=True)

# === Step 2: select top-K. Force-include cat cols (they're informative even at low gain). ===
K = 20
top_feats = list(imp_df.head(K)['feat'])
for c in CAT_COLS:
    if c not in top_feats:
        top_feats.append(c)
FEATURES = top_feats
print(f'\n=== chosen {len(FEATURES)} feats: {FEATURES} ===', flush=True)

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

    X_tr_xgb = X_tr.copy(); X_va_xgb = X_va.copy(); X_te_xgb = X_test.copy()
    cat_in_use = [c for c in CAT_COLS if c in FEATURES]
    for c in cat_in_use:
        X_tr_xgb[c] = X_tr_xgb[c].cat.codes
        X_va_xgb[c] = X_va_xgb[c].cat.codes
        X_te_xgb[c] = X_te_xgb[c].cat.codes
    cat_idx = [X.columns.get_loc(c) for c in cat_in_use]

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
        fold_cb_te += cm.predict_proba(X_test)[:, 1] / len(SEEDS)
        print(f'  CB  seed={s} best_iter={cm.get_best_iteration()} t={time.time()-ss:.1f}s', flush=True)

    oof_xgb[va_idx] = fold_xgb_va; oof_cb[va_idx] = fold_cb_va
    test_xgb += fold_xgb_te / N_SPLITS; test_cb += fold_cb_te / N_SPLITS
    print(f'  fold XGB AUC: {roc_auc_score(y_va, fold_xgb_va):.5f}', flush=True)
    print(f'  fold CB  AUC: {roc_auc_score(y_va, fold_cb_va):.5f}', flush=True)
    print(f'  fold time: {time.time()-fstart:.1f}s', flush=True)

print('\n==== FINAL ====', flush=True)
print(f'XGB OOF AUC : {roc_auc_score(y, oof_xgb):.5f}', flush=True)
print(f'CB  OOF AUC : {roc_auc_score(y, oof_cb):.5f}', flush=True)

def safe_logit(p, eps=1e-6): return logit(np.clip(p, eps, 1-eps))
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

np.save('/tmp/trial30_oof_xgb.npy', oof_xgb)
np.save('/tmp/trial30_oof_cb.npy', oof_cb)
np.save('/tmp/trial30_test_xgb.npy', test_xgb)
np.save('/tmp/trial30_test_cb.npy', test_cb)
np.save('/tmp/trial30_oof_chosen.npy', oof_chosen)
np.save('/tmp/trial30_test_chosen.npy', test_chosen)
print('saved', flush=True)
