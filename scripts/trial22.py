"""Trial 22: T21 + OOF target encoding (Driver, Race+Compound, Driver+Compound)."""
import time
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

# Helper string keys for compound TE groups (so we can groupby easily without categorical issues)
def _key(df, cols):
    return df[cols[0]].astype(str).str.cat(
        [df[c].astype(str) for c in cols[1:]], sep='|'
    ) if len(cols) > 1 else df[cols[0]].astype(str)

train_fe['_k_driver']    = _key(train_fe, ['Driver'])
train_fe['_k_race_cmp']  = _key(train_fe, ['Race', 'Compound'])
train_fe['_k_drv_cmp']   = _key(train_fe, ['Driver', 'Compound'])
test_fe['_k_driver']     = _key(test_fe,  ['Driver'])
test_fe['_k_race_cmp']   = _key(test_fe,  ['Race', 'Compound'])
test_fe['_k_drv_cmp']    = _key(test_fe,  ['Driver', 'Compound'])

TE_GROUPS = [('_k_driver',   'TE_Driver'),
             ('_k_race_cmp', 'TE_Race_Compound'),
             ('_k_drv_cmp',  'TE_Driver_Compound')]
SMOOTHING = 50.0

def compute_te(train_df, y_tr, val_df, test_df, key_col, smoothing):
    """OOF TE: fit means on train fold only, apply to val and test."""
    global_mean = y_tr.mean()
    tmp = pd.DataFrame({'k': train_df[key_col].values, 'y': y_tr.values})
    agg = tmp.groupby('k')['y'].agg(['sum', 'count'])
    smoothed = (agg['sum'] + smoothing * global_mean) / (agg['count'] + smoothing)
    smoothed_dict = smoothed.to_dict()
    tr_te  = train_df[key_col].map(smoothed_dict).fillna(global_mean).astype(np.float32)
    va_te  = val_df[key_col].map(smoothed_dict).fillna(global_mean).astype(np.float32)
    te_te  = test_df[key_col].map(smoothed_dict).fillna(global_mean).astype(np.float32)
    return tr_te.values, va_te.values, te_te.values

# Base feature list (excludes _k_* keys; those are only for TE)
BASE_FEATURES = [c for c in train_fe.columns
                 if c not in DROP_COLS and not c.startswith('_k_')]
TE_FEATURES = [name for _, name in TE_GROUPS]
FEATURES = BASE_FEATURES + TE_FEATURES
print(f'{len(BASE_FEATURES)} base features + {len(TE_FEATURES)} TE = {len(FEATURES)} total', flush=True)

y = train_fe[TARGET]
groups = (train_fe['Race'].astype(str) + '_' +
          train_fe['Year'].astype(str)  + '_' +
          train_fe['Driver'].astype(str))

SEEDS = [42, 2024]
N_SPLITS = 5

def make_xgb_params(seed):
    return dict(
        n_estimators=6000, learning_rate=0.03, max_depth=7,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=20,
        reg_alpha=0.1, reg_lambda=1.0,
        objective='binary:logistic', eval_metric='auc', seed=seed,
        tree_method='hist', n_jobs=8, early_stopping_rounds=200,
    )

def make_cb_params(seed):
    return dict(
        iterations=6000, learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
        loss_function='Logloss', eval_metric='AUC', random_seed=seed,
        early_stopping_rounds=200, thread_count=8, verbose=0,
    )

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_fe))
oof_cb  = np.zeros(len(train_fe))
test_xgb = np.zeros(len(test_fe))
test_cb  = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(cv.split(train_fe, y, groups)):
    fstart = time.time()
    print(f'\n==== FOLD {fold} ====', flush=True)

    tr_df = train_fe.iloc[tr_idx].copy()
    va_df = train_fe.iloc[va_idx].copy()
    te_df = test_fe.copy()
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    # ---- OOF target encoding (fit on train fold only) ----
    for key_col, te_name in TE_GROUPS:
        tr_te, va_te, te_te = compute_te(tr_df, y_tr, va_df, te_df, key_col, SMOOTHING)
        tr_df[te_name] = tr_te
        va_df[te_name] = va_te
        te_df[te_name] = te_te

    X_tr = tr_df[FEATURES]
    X_va = va_df[FEATURES]
    X_te = te_df[FEATURES]

    # XGB: int-encode cat cols
    X_tr_xgb = X_tr.copy(); X_va_xgb = X_va.copy(); X_te_xgb = X_te.copy()
    for c in CAT_COLS:
        X_tr_xgb[c] = X_tr_xgb[c].cat.codes
        X_va_xgb[c] = X_va_xgb[c].cat.codes
        X_te_xgb[c] = X_te_xgb[c].cat.codes

    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]

    fold_xgb_va = np.zeros(len(va_idx)); fold_xgb_te = np.zeros(len(test_fe))
    fold_cb_va  = np.zeros(len(va_idx)); fold_cb_te  = np.zeros(len(test_fe))

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
        fold_cb_te += cm.predict_proba(X_te)[:, 1] / len(SEEDS)
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
np.save('/tmp/trial22_oof_xgb.npy', oof_xgb)
np.save('/tmp/trial22_oof_cb.npy',  oof_cb)
np.save('/tmp/trial22_test_xgb.npy', test_xgb)
np.save('/tmp/trial22_test_cb.npy',  test_cb)
print('submission saved')
