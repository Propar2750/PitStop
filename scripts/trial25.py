"""Trial 25: CatBoost-only, 4-seed bag, T21 features. Drop XGB (weaker single model).
Plus exploratory: try depth=9 head-to-head vs depth=8 in fold 0 to inform later trials."""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
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

y = train_fe[TARGET]
groups = (train_fe['Race'].astype(str) + '_' +
          train_fe['Year'].astype(str)  + '_' +
          train_fe['Driver'].astype(str))

X = train_fe[FEATURES]
X_test = test_fe[FEATURES]

# 4-seed bag
SEEDS = [42, 2024, 7, 31337]
N_SPLITS = 5

def make_cb_params(seed):
    return dict(
        iterations=6000, learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
        loss_function='Logloss', eval_metric='AUC', random_seed=seed,
        early_stopping_rounds=200, thread_count=8, verbose=0,
    )

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_cb  = np.zeros(len(train_fe))
test_cb = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
    fstart = time.time()
    print(f'\n==== FOLD {fold} ====', flush=True)
    X_tr = X.iloc[tr_idx][FEATURES]; X_va = X.iloc[va_idx][FEATURES]
    y_tr = y.iloc[tr_idx]; y_va = y.iloc[va_idx]
    X_te_f = X_test[FEATURES]
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_COLS]

    fold_va = np.zeros(len(va_idx)); fold_te = np.zeros(len(test_fe))
    seed_aucs = []
    for s in SEEDS:
        ss = time.time()
        cm = cb.CatBoostClassifier(**make_cb_params(s))
        cm.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_idx, verbose=False)
        p_va = cm.predict_proba(X_va)[:, 1]
        fold_va += p_va / len(SEEDS)
        fold_te += cm.predict_proba(X_te_f)[:, 1] / len(SEEDS)
        seed_aucs.append(roc_auc_score(y_va, p_va))
        print(f'  CB  seed={s} best_iter={cm.get_best_iteration()} single_AUC={seed_aucs[-1]:.5f} t={time.time()-ss:.1f}s', flush=True)

    oof_cb[va_idx] = fold_va
    test_cb += fold_te / N_SPLITS
    print(f'  fold CB(bag) AUC: {roc_auc_score(y_va, fold_va):.5f} (single-seed mean {np.mean(seed_aucs):.5f})', flush=True)
    print(f'  fold time: {time.time()-fstart:.1f}s', flush=True)

print('\n==== FINAL ====', flush=True)
auc = roc_auc_score(y, oof_cb)
print(f'CB 4-seed OOF AUC : {auc:.5f}', flush=True)
print(f'CB 4-seed OOF AP  : {average_precision_score(y, oof_cb):.5f}', flush=True)
print(f'CB 4-seed OOF LL  : {log_loss(y, oof_cb):.5f}', flush=True)
print(f'TOTAL: {time.time()-t0:.1f}s', flush=True)

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_cb})
sub.to_csv(DATA_DIR / 'submission.csv', index=False)
np.save('/tmp/trial25_oof_cb.npy', oof_cb)
np.save('/tmp/trial25_test_cb.npy', test_cb)
print('submission saved', flush=True)
