"""Trial 26: pseudo-labeling using T25 ensemble's confident test predictions.
Single-seed CB to keep runtime tractable; if it helps, expand to multi-seed in T27."""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import catboost as cb
import xgboost as xgb
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

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

# T27 = iterative pseudo-labeling: use T26's STRONGER preds (OOF 0.94919) as source.
oof_xgb_2 = np.load('/tmp/trial26_oof_xgb.npy')
oof_cb_4  = np.load('/tmp/trial26_oof_cb.npy')
test_xgb_2 = np.load('/tmp/trial26_test_xgb.npy')
test_cb_4  = np.load('/tmp/trial26_test_cb.npy')
print('Pseudo source: T26 OOF/test (iterative round 2)', flush=True)

def safe_logit(p, eps=1e-6): return logit(np.clip(p, eps, 1-eps))
y = train_fe[TARGET]
groups = (train_fe['Race'].astype(str) + '_' +
          train_fe['Year'].astype(str)  + '_' +
          train_fe['Driver'].astype(str))

# Refit LR-meta on full train OOFs to get calibrated test probabilities
Z_train = np.column_stack([safe_logit(oof_xgb_2), safe_logit(oof_cb_4)])
Z_test  = np.column_stack([safe_logit(test_xgb_2), safe_logit(test_cb_4)])
lrf = LogisticRegression(C=1.0, max_iter=1000)
lrf.fit(Z_train, y)
test_probs = lrf.predict_proba(Z_test)[:, 1]
print(f'Test pred distribution: min={test_probs.min():.4f} median={np.median(test_probs):.4f} max={test_probs.max():.4f}', flush=True)
print(f'  pred>0.5  : {(test_probs>0.5).sum():>7d} ({(test_probs>0.5).mean():.3%})', flush=True)
print(f'  pred>0.9  : {(test_probs>0.9).sum():>7d}', flush=True)
print(f'  pred>0.95 : {(test_probs>0.95).sum():>7d}', flush=True)
print(f'  pred<0.05 : {(test_probs<0.05).sum():>7d}', flush=True)
print(f'  pred<0.02 : {(test_probs<0.02).sum():>7d}', flush=True)

# Pseudo-label strategy: add ONLY high-confidence positives (rare class needs more signal,
# negatives are already abundant in train). Confidence threshold tuned to get a substantial
# but high-precision pos set.
HI = 0.80
pos_mask = test_probs > HI
n_pos = int(pos_mask.sum())
print(f'\nPseudo-labels: {n_pos} positives (thr>{HI}, no negatives) = {n_pos/len(test):.3%} of test', flush=True)

pseudo_X = test_fe.loc[pos_mask, FEATURES].reset_index(drop=True)
pseudo_y = pd.Series(np.ones(n_pos, dtype=int), name=TARGET)
print(f'Pseudo class balance: pos rate = {pseudo_y.mean():.4f} (train pos rate = {y.mean():.4f})', flush=True)

X_train_real = train_fe[FEATURES]
y_real = y

SEEDS = [42, 2024]  # 2 seeds for both XGB and CB to balance compute
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
oof_xgb = np.zeros(len(train_fe)); oof_cb = np.zeros(len(train_fe))
test_xgb_new = np.zeros(len(test_fe)); test_cb_new = np.zeros(len(test_fe))

# Pre-encode pseudo X for XGB (cat → int codes consistent with full encoding)
def encode_xgb_cats(df):
    out = df.copy()
    for c in CAT_COLS:
        out[c] = out[c].cat.codes
    return out

pseudo_X_xgb = encode_xgb_cats(pseudo_X)
X_test_full = test_fe[FEATURES]
X_test_xgb_full = encode_xgb_cats(X_test_full)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train_real, y_real, groups)):
    fstart = time.time()
    print(f'\n==== FOLD {fold} ====', flush=True)
    X_tr = X_train_real.iloc[tr_idx]
    X_va = X_train_real.iloc[va_idx]
    y_tr = y_real.iloc[tr_idx]
    y_va = y_real.iloc[va_idx]

    # Augment train with pseudo rows. concat downcasts categoricals — re-cast.
    X_tr_aug = pd.concat([X_tr, pseudo_X], ignore_index=True)
    y_tr_aug = pd.concat([y_tr.reset_index(drop=True), pseudo_y], ignore_index=True)
    for c in CAT_COLS:
        X_tr_aug[c] = X_tr_aug[c].astype('category')
    print(f'  augmented train: {len(X_tr)} real + {len(pseudo_X)} pseudo = {len(X_tr_aug)} (pos rate {y_tr_aug.mean():.4f})', flush=True)

    X_tr_xgb = encode_xgb_cats(X_tr_aug)
    X_va_xgb = encode_xgb_cats(X_va)
    cat_idx = [X_tr_aug.columns.get_loc(c) for c in CAT_COLS]

    fold_xgb_va = np.zeros(len(va_idx)); fold_xgb_te = np.zeros(len(test_fe))
    fold_cb_va  = np.zeros(len(va_idx)); fold_cb_te  = np.zeros(len(test_fe))

    for s in SEEDS:
        ss = time.time()
        m = xgb.XGBClassifier(**make_xgb_params(s))
        m.fit(X_tr_xgb, y_tr_aug, eval_set=[(X_va_xgb, y_va)], verbose=False)
        fold_xgb_va += m.predict_proba(X_va_xgb)[:, 1] / len(SEEDS)
        fold_xgb_te += m.predict_proba(X_test_xgb_full)[:, 1] / len(SEEDS)
        print(f'  XGB seed={s} best_iter={m.best_iteration} t={time.time()-ss:.1f}s', flush=True)

        ss = time.time()
        cm = cb.CatBoostClassifier(**make_cb_params(s))
        cm.fit(X_tr_aug, y_tr_aug, eval_set=(X_va, y_va), cat_features=cat_idx, verbose=False)
        fold_cb_va += cm.predict_proba(X_va)[:, 1] / len(SEEDS)
        fold_cb_te += cm.predict_proba(X_test_full)[:, 1] / len(SEEDS)
        print(f'  CB  seed={s} best_iter={cm.get_best_iteration()} t={time.time()-ss:.1f}s', flush=True)

    oof_xgb[va_idx] = fold_xgb_va
    oof_cb[va_idx]  = fold_cb_va
    test_xgb_new += fold_xgb_te / N_SPLITS
    test_cb_new  += fold_cb_te  / N_SPLITS
    print(f'  fold XGB AUC: {roc_auc_score(y_va, fold_xgb_va):.5f} (T21 was {0.94878 if fold==0 else 0.94767 if fold==1 else 0.94679 if fold==2 else 0.94805 if fold==3 else 0.94854:.5f})', flush=True)
    print(f'  fold CB  AUC: {roc_auc_score(y_va, fold_cb_va):.5f} (T21 was {0.94926 if fold==0 else 0.94819 if fold==1 else 0.94728 if fold==2 else 0.94832 if fold==3 else 0.94893:.5f})', flush=True)
    print(f'  fold time: {time.time()-fstart:.1f}s', flush=True)

# Ensemble (rank-avg + LR-meta)
def ranked(p): return rankdata(p) / len(p)
oof_rank = (ranked(oof_xgb) + ranked(oof_cb)) / 2
test_rank = (ranked(test_xgb_new) + ranked(test_cb_new)) / 2

Z2 = np.column_stack([safe_logit(oof_xgb), safe_logit(oof_cb)])
oof_meta = np.zeros(len(y))
for tr, va in cv.split(Z2, y_real, groups):
    lr = LogisticRegression(C=1.0, max_iter=1000); lr.fit(Z2[tr], y_real.iloc[tr])
    oof_meta[va] = lr.predict_proba(Z2[va])[:, 1]
lrf2 = LogisticRegression(C=1.0, max_iter=1000); lrf2.fit(Z2, y_real)
Z2_test = np.column_stack([safe_logit(test_xgb_new), safe_logit(test_cb_new)])
test_meta = lrf2.predict_proba(Z2_test)[:, 1]

print('\n==== FINAL ====', flush=True)
print(f'XGB OOF AUC : {roc_auc_score(y, oof_xgb):.5f}', flush=True)
print(f'CB  OOF AUC : {roc_auc_score(y, oof_cb):.5f}', flush=True)
print(f'rank-avg    : {roc_auc_score(y, oof_rank):.5f}', flush=True)
print(f'LR-meta     : {roc_auc_score(y, oof_meta):.5f}', flush=True)

# Pick best
best_score = max(roc_auc_score(y, oof_rank), roc_auc_score(y, oof_meta))
if roc_auc_score(y, oof_meta) >= roc_auc_score(y, oof_rank):
    best_name = 'lr-meta'; oof_chosen = oof_meta; test_chosen = test_meta
else:
    best_name = 'rank-avg'; oof_chosen = oof_rank; test_chosen = test_rank
print(f'CHOSEN: {best_name} OOF AUC {best_score:.5f}', flush=True)
print(f'  AP: {average_precision_score(y, oof_chosen):.5f}', flush=True)
print(f'  LL: {log_loss(y, oof_chosen):.5f}', flush=True)
print(f'TOTAL: {time.time()-t0:.1f}s', flush=True)

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_chosen})
sub.to_csv(DATA_DIR / 'submission.csv', index=False)
np.save('/tmp/trial27_oof_xgb.npy', oof_xgb)
np.save('/tmp/trial27_oof_cb.npy',  oof_cb)
np.save('/tmp/trial27_test_xgb.npy', test_xgb_new)
np.save('/tmp/trial27_test_cb.npy',  test_cb_new)
print('submission saved', flush=True)
