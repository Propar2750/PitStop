"""Trial 23: T21 XGB+CB (no TE) + LR base model (standardized + OOF TE) — 3-way ensemble."""
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

def _key(df, cols):
    if len(cols) == 1: return df[cols[0]].astype(str)
    return df[cols[0]].astype(str).str.cat([df[c].astype(str) for c in cols[1:]], sep='|')

train_fe['_k_driver']    = _key(train_fe, ['Driver'])
train_fe['_k_race_cmp']  = _key(train_fe, ['Race', 'Compound'])
train_fe['_k_drv_cmp']   = _key(train_fe, ['Driver', 'Compound'])
test_fe['_k_driver']     = _key(test_fe,  ['Driver'])
test_fe['_k_race_cmp']   = _key(test_fe,  ['Race', 'Compound'])
test_fe['_k_drv_cmp']    = _key(test_fe,  ['Driver', 'Compound'])

TE_GROUPS = [('_k_driver','TE_Driver'),
             ('_k_race_cmp','TE_Race_Compound'),
             ('_k_drv_cmp','TE_Driver_Compound')]
SMOOTHING = 50.0

def compute_te(train_df, y_tr, val_df, test_df, key_col, smoothing):
    g = float(y_tr.mean())
    tmp = pd.DataFrame({'k': train_df[key_col].values, 'y': y_tr.values})
    agg = tmp.groupby('k')['y'].agg(['sum','count'])
    smoothed = (agg['sum'] + smoothing * g) / (agg['count'] + smoothing)
    d = smoothed.to_dict()
    return (train_df[key_col].map(d).fillna(g).astype(np.float32).values,
            val_df[key_col].map(d).fillna(g).astype(np.float32).values,
            test_df[key_col].map(d).fillna(g).astype(np.float32).values)

GBDT_FEATURES = [c for c in train_fe.columns
                 if c not in DROP_COLS and not c.startswith('_k_')]
TE_FEATURES = [n for _, n in TE_GROUPS]
print(f'GBDT features: {len(GBDT_FEATURES)}', flush=True)

# LR features: numeric base feats only (drop raw cats; replace with TE), plus Compound one-hots
NUMERIC_BASE = [c for c in GBDT_FEATURES if c not in CAT_COLS]
print(f'LR numeric base: {len(NUMERIC_BASE)} + 3 TE + 3 Compound one-hot', flush=True)

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
oof_lr  = np.zeros(len(train_fe))
test_xgb = np.zeros(len(test_fe))
test_cb  = np.zeros(len(test_fe))
test_lr  = np.zeros(len(test_fe))

# precompute compound one-hots once (same scheme across folds)
def compound_ohe(df):
    s = df['Compound'].astype(str).str.upper()
    return np.column_stack([(s == 'HARD').astype(float),
                            (s == 'MEDIUM').astype(float),
                            (s == 'SOFT').astype(float)])

for fold, (tr_idx, va_idx) in enumerate(cv.split(train_fe, y, groups)):
    fstart = time.time()
    print(f'\n==== FOLD {fold} ====', flush=True)

    tr_df = train_fe.iloc[tr_idx].copy()
    va_df = train_fe.iloc[va_idx].copy()
    te_df = test_fe.copy()
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    # OOF TE (LR will use; GBDTs don't get TE per T22 lesson)
    te_tr = {}; te_va = {}; te_te = {}
    for key_col, te_name in TE_GROUPS:
        a, b, c = compute_te(tr_df, y_tr, va_df, te_df, key_col, SMOOTHING)
        te_tr[te_name] = a; te_va[te_name] = b; te_te[te_name] = c

    # --- GBDT features (no TE) ---
    X_tr = tr_df[GBDT_FEATURES]
    X_va = va_df[GBDT_FEATURES]
    X_te = te_df[GBDT_FEATURES]
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

    # --- LR features ---
    ss = time.time()
    Xtr_num = tr_df[NUMERIC_BASE].values.astype(np.float64)
    Xva_num = va_df[NUMERIC_BASE].values.astype(np.float64)
    Xte_num = te_df[NUMERIC_BASE].values.astype(np.float64)
    # add TE
    Xtr_te = np.column_stack([te_tr[n] for n in TE_FEATURES])
    Xva_te = np.column_stack([te_va[n] for n in TE_FEATURES])
    Xte_te = np.column_stack([te_te[n] for n in TE_FEATURES])
    # add compound OHE
    Xtr_ohe = compound_ohe(tr_df)
    Xva_ohe = compound_ohe(va_df)
    Xte_ohe = compound_ohe(te_df)

    Xtr_lr = np.hstack([Xtr_num, Xtr_te, Xtr_ohe])
    Xva_lr = np.hstack([Xva_num, Xva_te, Xva_ohe])
    Xte_lr = np.hstack([Xte_num, Xte_te, Xte_ohe])

    # standardize on train fold
    mu = Xtr_lr.mean(0); sd = Xtr_lr.std(0) + 1e-9
    Xtr_lr_s = (Xtr_lr - mu) / sd
    Xva_lr_s = (Xva_lr - mu) / sd
    Xte_lr_s = (Xte_lr - mu) / sd

    lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    lr.fit(Xtr_lr_s, y_tr)
    fold_lr_va = lr.predict_proba(Xva_lr_s)[:, 1]
    fold_lr_te = lr.predict_proba(Xte_lr_s)[:, 1]
    print(f'  LR  t={time.time()-ss:.1f}s', flush=True)

    oof_xgb[va_idx] = fold_xgb_va
    oof_cb[va_idx]  = fold_cb_va
    oof_lr[va_idx]  = fold_lr_va
    test_xgb += fold_xgb_te / N_SPLITS
    test_cb  += fold_cb_te  / N_SPLITS
    test_lr  += fold_lr_te  / N_SPLITS

    print(f'  fold XGB AUC: {roc_auc_score(y_va, fold_xgb_va):.5f}', flush=True)
    print(f'  fold CB  AUC: {roc_auc_score(y_va, fold_cb_va):.5f}', flush=True)
    print(f'  fold LR  AUC: {roc_auc_score(y_va, fold_lr_va):.5f}', flush=True)
    print(f'  fold time: {time.time()-fstart:.1f}s', flush=True)

def rank_avg(*preds):
    n = len(preds[0])
    return sum(rankdata(p) / n for p in preds) / len(preds)

print('\n==== INDIVIDUAL OOF ====', flush=True)
auc_xgb = roc_auc_score(y, oof_xgb); print(f'XGB OOF AUC : {auc_xgb:.5f}', flush=True)
auc_cb  = roc_auc_score(y, oof_cb);  print(f'CB  OOF AUC : {auc_cb:.5f}', flush=True)
auc_lr  = roc_auc_score(y, oof_lr);  print(f'LR  OOF AUC : {auc_lr:.5f}', flush=True)

# Pairs and triples for ensembling
oof_xc  = rank_avg(oof_xgb, oof_cb)
oof_xl  = rank_avg(oof_xgb, oof_lr)
oof_cl  = rank_avg(oof_cb,  oof_lr)
oof_3   = rank_avg(oof_xgb, oof_cb, oof_lr)
test_3  = rank_avg(test_xgb, test_cb, test_lr)

print('\n==== ENSEMBLES (rank-avg) ====', flush=True)
print(f'XGB+CB     : {roc_auc_score(y, oof_xc):.5f}', flush=True)
print(f'XGB+LR     : {roc_auc_score(y, oof_xl):.5f}', flush=True)
print(f'CB+LR      : {roc_auc_score(y, oof_cl):.5f}', flush=True)
print(f'XGB+CB+LR  : {roc_auc_score(y, oof_3):.5f}', flush=True)

# Weighted ensemble: grid search over weights on OOF
print('\n==== WEIGHTED RANK-AVG SEARCH ====', flush=True)
def wrank(weights, *preds):
    n = len(preds[0])
    return sum(w * (rankdata(p) / n) for w, p in zip(weights, preds))
best_w = None; best_auc = 0.0
for wx in np.arange(0.0, 1.05, 0.05):
    for wc in np.arange(0.0, 1.05 - wx, 0.05):
        wl = 1.0 - wx - wc
        if wl < -1e-9: continue
        oof_w = wrank((wx, wc, wl), oof_xgb, oof_cb, oof_lr)
        a = roc_auc_score(y, oof_w)
        if a > best_auc:
            best_auc = a; best_w = (wx, wc, wl)
print(f'best weights (xgb,cb,lr)={best_w} AUC={best_auc:.5f}', flush=True)

# LR meta-stacker on logits
print('\n==== LR META STACKER ====', flush=True)
def safe_logit(p, eps=1e-6): return logit(np.clip(p, eps, 1-eps))
Z = np.column_stack([safe_logit(oof_xgb), safe_logit(oof_cb), safe_logit(oof_lr)])
oof_meta = np.zeros(len(y))
for fold, (tr, va) in enumerate(cv.split(Z, y, groups)):
    metalr = LogisticRegression(C=1.0, max_iter=1000)
    metalr.fit(Z[tr], y[tr])
    oof_meta[va] = metalr.predict_proba(Z[va])[:, 1]
print(f'LR-meta(logits): {roc_auc_score(y, oof_meta):.5f}', flush=True)

# Save preds + use best of {3-way rank-avg, weighted rank-avg, LR-meta} for submission
candidates = {
    '3way_rank': (roc_auc_score(y, oof_3), test_3),
    'weighted':  (best_auc, wrank(best_w, test_xgb, test_cb, test_lr)),
    'lr_meta':   (roc_auc_score(y, oof_meta), None),  # need test logits
}
# Build test for LR-meta using same per-fold trained metalrs... simpler: refit metalr on full train
Z_train_full = Z
metalr_full = LogisticRegression(C=1.0, max_iter=1000)
metalr_full.fit(Z_train_full, y)
Z_test = np.column_stack([safe_logit(test_xgb), safe_logit(test_cb), safe_logit(test_lr)])
test_meta = metalr_full.predict_proba(Z_test)[:, 1]
candidates['lr_meta'] = (roc_auc_score(y, oof_meta), test_meta)

best_name = max(candidates, key=lambda k: candidates[k][0])
best_score, test_best = candidates[best_name]
print(f'\nFINAL CHOICE: {best_name} (OOF AUC {best_score:.5f})', flush=True)

# Final metrics on chosen
chosen_oof = {'3way_rank': oof_3, 'weighted': wrank(best_w, oof_xgb, oof_cb, oof_lr), 'lr_meta': oof_meta}[best_name]
print(f'  AP: {average_precision_score(y, chosen_oof):.5f}', flush=True)
print(f'  LL: {log_loss(y, chosen_oof):.5f}', flush=True)
print(f'TOTAL: {time.time()-t0:.1f}s', flush=True)

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_best})
sub.to_csv(DATA_DIR / 'submission.csv', index=False)
np.save('/tmp/trial23_oof_xgb.npy', oof_xgb)
np.save('/tmp/trial23_oof_cb.npy',  oof_cb)
np.save('/tmp/trial23_oof_lr.npy',  oof_lr)
np.save('/tmp/trial23_test_xgb.npy', test_xgb)
np.save('/tmp/trial23_test_cb.npy',  test_cb)
np.save('/tmp/trial23_test_lr.npy',  test_lr)
print('submission saved', flush=True)
