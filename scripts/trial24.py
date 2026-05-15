"""Trial 24: T21 (XGB+CB, no TE) + within-stint temporal features (LapNumber-aware)."""
import time, warnings
warnings.filterwarnings('ignore')
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

def add_temporal_features(df):
    """Within-stint, LapNumber-aware temporal features. No leakage (uses only past laps in same stint).
    Computed per (Driver,Race,Year,Stint). Note: train and test rows belong to different driver-races
    (group-CV setup), so we can compute these globally on train+test then split."""
    df = df.copy()
    sort_cols = ['Driver','Race','Year','Stint','LapNumber']
    df = df.sort_values(sort_cols).reset_index(drop=False).rename(columns={'index':'_orig_idx'})
    gk = ['Driver','Race','Year','Stint']
    g = df.groupby(gk, sort=False)

    # laps sampled in stint so far (0-indexed)
    df['sampled_idx_in_stint'] = g.cumcount()
    # gap to previous sampled lap
    df['LapNumber_prev'] = g['LapNumber'].shift(1)
    df['LapGap_prev']    = (df['LapNumber'] - df['LapNumber_prev']).fillna(0.0)

    # per-lap-normalized deltas (defensive divide)
    safe_gap = df['LapGap_prev'].clip(lower=1.0)
    df['LapTime_prev']     = g['LapTime (s)'].shift(1)
    df['LapTime_delta_pl'] = ((df['LapTime (s)'] - df['LapTime_prev']) / safe_gap).fillna(0.0)
    df['CumDeg_prev']      = g['Cumulative_Degradation'].shift(1)
    df['CumDeg_delta_pl']  = ((df['Cumulative_Degradation'] - df['CumDeg_prev']) / safe_gap).fillna(0.0)
    df['Position_prev']    = g['Position'].shift(1)
    df['Position_delta_pl']= ((df['Position'] - df['Position_prev']) / safe_gap).fillna(0.0)

    # rolling stats over last sampled laps within stint
    # NOTE: cummin/cumsum/expanding within groupby is the only safe method (rolling with groups + min_periods is leaky on shift order)
    # Use expanding from start of stint
    df['LapTime_stint_min']    = g['LapTime (s)'].cummin()
    df['LapTime_above_min']    = df['LapTime (s)'] - df['LapTime_stint_min']
    df['LapTime_stint_max']    = g['LapTime (s)'].cummax()
    df['LapTime_range_stint']  = df['LapTime_stint_max'] - df['LapTime_stint_min']

    # expanding mean/std (slow but correct under group)
    df['LapTime_stint_mean']   = g['LapTime (s)'].expanding().mean().reset_index(level=list(range(len(gk))), drop=True)
    df['LapTime_above_mean']   = df['LapTime (s)'] - df['LapTime_stint_mean']

    # EWM of LapTime_Delta within stint (alpha=0.5 ≈ half-life 1 sample)
    df['LapTimeDelta_ewm']     = g['LapTime_Delta'].transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())
    df['CumDeg_ewm']           = g['Cumulative_Degradation'].transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())

    # Acceleration via second finite difference (fast, no Python loop)
    df['LapTime_accel_pl'] = (df['LapTime_delta_pl'] - g['LapTime_delta_pl'].shift(1)).fillna(0.0)
    df['CumDeg_accel_pl']  = (df['CumDeg_delta_pl']  - g['CumDeg_delta_pl'].shift(1)).fillna(0.0)

    # fraction of stint completed in TyreLife units relative to global compound max
    cmp_max = df.groupby('Compound')['TyreLife'].transform('max').replace(0, np.nan)
    df['TyreLife_frac_cmpmax'] = (df['TyreLife'] / cmp_max).fillna(0.0)

    # restore original row order
    df = df.sort_values('_orig_idx').drop(columns=['_orig_idx','LapNumber_prev','LapTime_prev','CumDeg_prev','Position_prev']).reset_index(drop=True)
    return df

# Compute features on combined train+test for stint context (NO target leak — only uses raw columns)
print('Computing features...', flush=True)
combined = pd.concat([train.assign(__is_test=0), test.assign(__is_test=1, **{TARGET: 0})], ignore_index=True)
combined = add_features(combined)
combined = add_temporal_features(combined)
train_fe = combined[combined['__is_test']==0].drop(columns=['__is_test']).reset_index(drop=True)
test_fe  = combined[combined['__is_test']==1].drop(columns=['__is_test', TARGET]).reset_index(drop=True)
print('Done.', flush=True)

# Re-attach target (was zeroed for test in combined; train kept it)
train_fe[TARGET] = train[TARGET].values

CAT_COLS = ['Driver', 'Compound', 'Race']
for c in CAT_COLS:
    train_fe[c] = train_fe[c].astype('category')
    test_fe[c]  = test_fe[c].astype('category')

DROP_COLS = [ID_COL, TARGET, 'LapNumber']
GBDT_FEATURES = [c for c in train_fe.columns if c not in DROP_COLS]
print(f'GBDT features: {len(GBDT_FEATURES)}', flush=True)
print(GBDT_FEATURES, flush=True)

y = train_fe[TARGET]
groups = (train_fe['Race'].astype(str) + '_' +
          train_fe['Year'].astype(str)  + '_' +
          train_fe['Driver'].astype(str))

SEEDS = [42, 2024]
N_SPLITS = 5

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

for fold, (tr_idx, va_idx) in enumerate(cv.split(train_fe, y, groups)):
    fstart = time.time()
    print(f'\n==== FOLD {fold} ====', flush=True)
    X_tr = train_fe.iloc[tr_idx][GBDT_FEATURES]
    X_va = train_fe.iloc[va_idx][GBDT_FEATURES]
    X_te = test_fe[GBDT_FEATURES]
    y_tr = y.iloc[tr_idx]; y_va = y.iloc[va_idx]

    X_tr_xgb = X_tr.copy(); X_va_xgb = X_va.copy(); X_te_xgb = X_te.copy()
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
        fold_cb_te += cm.predict_proba(X_te)[:, 1] / len(SEEDS)
        print(f'  CB  seed={s} best_iter={cm.get_best_iteration()} t={time.time()-ss:.1f}s', flush=True)

    oof_xgb[va_idx] = fold_xgb_va; test_xgb += fold_xgb_te / N_SPLITS
    oof_cb[va_idx]  = fold_cb_va;  test_cb  += fold_cb_te  / N_SPLITS
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
np.save('/tmp/trial24_oof_xgb.npy', oof_xgb)
np.save('/tmp/trial24_oof_cb.npy',  oof_cb)
np.save('/tmp/trial24_test_xgb.npy', test_xgb)
np.save('/tmp/trial24_test_cb.npy',  test_cb)
print('submission saved', flush=True)
