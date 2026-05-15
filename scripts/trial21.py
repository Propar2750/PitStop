"""Trial 21: tuned XGB+CB with 2-seed averaging."""
import os, sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import xgboost as xgb
import catboost as cb
from scipy.stats import rankdata

t0 = time.time()
DATA_DIR = Path('/home/propar/Documents/Projects/F1-pitstop-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
TARGET, ID_COL = 'PitNextLap', 'id'

def add_features(df):
    df = df.copy()
    eps = 1e-6
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

train_fe = add_features(train)
test_fe  = add_features(test)

CAT_COLS = ['Driver', 'Compound', 'Race']
for c in CAT_COLS:
    train_fe[c] = train_fe[c].astype('category')
    test_fe[c]  = test_fe[c].astype('category')

DROP_COLS = [ID_COL, TARGET, 'LapNumber']
FEATURES = [c for c in train_fe.columns if c not in DROP_COLS]
print(f'{len(FEATURES)} features', flush=True)

X = train_fe.drop(columns=[c for c in DROP_COLS if c in train_fe.columns])
y = train_fe[TARGET]
groups = (train_fe['Race'].astype(str) + '_' +
          train_fe['Year'].astype(str)  + '_' +
          train_fe['Driver'].astype(str))
X_test = test_fe.drop(columns=[c for c in DROP_COLS if c in test_fe.columns])

SEEDS = [42, 2024]
N_SPLITS = 5

def make_xgb_params(seed):
    return dict(
        n_estimators=6000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='auc',
        seed=seed,
        tree_method='hist',
        n_jobs=8,
        early_stopping_rounds=200,
    )

def make_cb_params(seed):
    return dict(
        iterations=6000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5.0,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=seed,
        early_stopping_rounds=200,
        thread_count=8,
        verbose=0,
    )

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_fe))
oof_cb  = np.zeros(len(train_fe))
test_xgb = np.zeros(len(test_fe))
test_cb  = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
    fstart = time.time()
    print(f'\n==== FOLD {fold} ====', flush=True)
    X_tr = X.iloc[tr_idx][FEATURES]; X_va = X.iloc[va_idx][FEATURES]
    y_tr = y.iloc[tr_idx];           y_va = y.iloc[va_idx]
    X_te_f = X_test[FEATURES]

    # XGB needs ints for cats
    X_tr_xgb = X_tr.copy(); X_va_xgb = X_va.copy(); X_te_xgb = X_te_f.copy()
    for c in CAT_COLS:
        X_tr_xgb[c] = X_tr_xgb[c].cat.codes
        X_va_xgb[c] = X_va_xgb[c].cat.codes
        X_te_xgb[c] = X_te_xgb[c].cat.codes

    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]

    fold_xgb_va = np.zeros(len(va_idx))
    fold_xgb_te = np.zeros(len(test_fe))
    fold_cb_va  = np.zeros(len(va_idx))
    fold_cb_te  = np.zeros(len(test_fe))

    for s in SEEDS:
        sstart = time.time()
        m = xgb.XGBClassifier(**make_xgb_params(s))
        m.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
        fold_xgb_va += m.predict_proba(X_va_xgb)[:, 1] / len(SEEDS)
        fold_xgb_te += m.predict_proba(X_te_xgb)[:, 1] / len(SEEDS)
        print(f'  XGB seed={s} best_iter={m.best_iteration} t={time.time()-sstart:.1f}s', flush=True)

        sstart = time.time()
        cm = cb.CatBoostClassifier(**make_cb_params(s))
        cm.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_idx, verbose=False)
        fold_cb_va += cm.predict_proba(X_va)[:, 1] / len(SEEDS)
        fold_cb_te += cm.predict_proba(X_te_f)[:, 1] / len(SEEDS)
        print(f'  CB  seed={s} best_iter={cm.get_best_iteration()} t={time.time()-sstart:.1f}s', flush=True)

    oof_xgb[va_idx] = fold_xgb_va
    oof_cb[va_idx]  = fold_cb_va
    test_xgb += fold_xgb_te / N_SPLITS
    test_cb  += fold_cb_te  / N_SPLITS

    print(f'  fold XGB AUC: {roc_auc_score(y_va, fold_xgb_va):.5f}', flush=True)
    print(f'  fold CB  AUC: {roc_auc_score(y_va, fold_cb_va):.5f}', flush=True)
    print(f'  fold time: {time.time()-fstart:.1f}s', flush=True)

def rank_avg(*preds):
    n = len(preds[0])
    return sum(rankdata(p) / n for p in preds) / len(preds)

oof_ens = rank_avg(oof_xgb, oof_cb)
test_ens = rank_avg(test_xgb, test_cb)

print('\n==== FINAL ====', flush=True)
print(f'XGB OOF AUC : {roc_auc_score(y, oof_xgb):.5f}', flush=True)
print(f'CB  OOF AUC : {roc_auc_score(y, oof_cb):.5f}', flush=True)
print(f'ENS OOF AUC : {roc_auc_score(y, oof_ens):.5f}', flush=True)
print(f'ENS OOF AP  : {average_precision_score(y, oof_ens):.5f}', flush=True)
print(f'ENS OOF LL  : {log_loss(y, oof_ens):.5f}', flush=True)
print(f'TOTAL: {time.time()-t0:.1f}s', flush=True)

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_ens})
sub.to_csv(DATA_DIR / 'submission.csv', index=False)
np.save('/tmp/trial21_oof.npy', oof_ens)
np.save('/tmp/trial21_test.npy', test_ens)
print('submission saved')
