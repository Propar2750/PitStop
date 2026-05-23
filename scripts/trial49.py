"""Trial 49: TabNet (longer training). Trial48 hit best_epoch=39/40 every
fold — under-converged. Same recipe with max_epochs=120, patience=15, and
OneCycleLR span re-aligned to 120 so the warmup/anneal tracks the new
horizon. Otherwise identical to T48.

The T47 ensemble probe showed all our existing models (RealMLP, XGB, CB, seq
transformer) live in a tight rank-correlation cluster (0.95-0.98). To break
out we need a model with genuinely different inductive bias. TabNet uses
sequential attention for *instance-wise* feature selection — completely
different from gradient boosting (greedy splits) or MLP (dense mixing) or
transformer (sequential context).

Bet: TabNet standalone will be weaker than T46 (RealMLP) but its OOF will
sit in a different region of model-space, giving real ensemble lift.

Setup:
  * Feature set: T35-style raw baseline + LapNumber (= 11 numeric: LapNumber,
    Year, PitStop, Stint, TyreLife, Position, LapTime, LapTime_Delta,
    Cumulative_Degradation, RaceProgress, Position_Change) + 3 categoricals
    (Driver, Compound, Race) passed via cat_idxs/cat_dims.
    No engineered features — TabNet's attention is supposed to discover its
    own interactions. Mirrors what T35 used for GBDTs (the proven raw set).
  * CV: StratifiedGroupKFold by Race_Year_Driver, random_state=42 — matches
    every other group-CV trial in the log.
  * Hyperparams: pytorch-tabnet's published "tabnet-paper-ish" defaults with
    sensible adjustments for 440k-row binary task. Single seed first; if it
    works, can bag later.

Saves:
  * /tmp/trial49_oof.npy, /tmp/trial49_test.npy
  * Re-runs T47's full 7-way LR-meta probe with TabNet added.
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from scipy.stats import rankdata
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

DATA_DIR = Path(__file__).resolve().parent.parent
SEED = 42
N_SPLITS = 5

assert torch.cuda.is_available(), "CUDA required for TabNet GPU"
DEVICE = "cuda"
print(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}", flush=True)

# ── Load + features ──────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
print(f"train {len(train):,}  test {len(test):,}", flush=True)

y = train["PitNextLap"].values
groups = (train["Race"].astype(str) + "_" + train["Year"].astype(str)
          + "_" + train["Driver"].astype(str)).values

NUM_COLS = ["LapNumber", "Year", "PitStop", "Stint", "TyreLife", "Position",
            "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
            "RaceProgress", "Position_Change"]
CAT_COLS = ["Driver", "Compound", "Race"]
FEATS = NUM_COLS + CAT_COLS

# Factorize cats consistently across train+test so codes match
combined = pd.concat([train[FEATS], test[FEATS]], axis=0).reset_index(drop=True)
for c in CAT_COLS:
    combined[c] = combined[c].astype("category").cat.codes.astype(np.int64)
for c in NUM_COLS:
    combined[c] = combined[c].astype(np.float32)
# Median-impute any NaN (LapTime_Delta has a few)
for c in NUM_COLS:
    if combined[c].isna().any():
        med = combined[c].median()
        combined[c] = combined[c].fillna(med)

X_train = combined.iloc[:len(train)].values.astype(np.float32)
X_test  = combined.iloc[len(train):].values.astype(np.float32)
cat_idxs = [FEATS.index(c) for c in CAT_COLS]
cat_dims = [int(combined[c].max() + 1) for c in CAT_COLS]
cat_emb_dim = [min(50, max(4, (d + 1) // 2)) for d in cat_dims]  # rule-of-thumb
print(f"X_train {X_train.shape}  X_test {X_test.shape}")
print(f"cat_dims={cat_dims}  cat_emb_dim={cat_emb_dim}", flush=True)

# ── CV loop ──────────────────────────────────────────────────────────────────
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = np.zeros(len(y), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)

t0 = time.time()
for fold, (tr, va) in enumerate(cv.split(X_train, y, groups=groups)):
    print(f"\n=== Fold {fold} ===", flush=True)
    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        lambda_sparse=1e-4,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
        scheduler_params=dict(
            max_lr=2e-2,
            steps_per_epoch=max(1, len(tr) // 4096),
            epochs=120,
        ),
        mask_type="entmax",
        seed=SEED,
        device_name=DEVICE,
        verbose=0,
    )
    t_fold = time.time()
    clf.fit(
        X_train=X_train[tr], y_train=y[tr],
        eval_set=[(X_train[va], y[va])],
        eval_metric=["auc"],
        max_epochs=120,
        patience=15,
        batch_size=4096,
        virtual_batch_size=512,
        num_workers=0,
        drop_last=False,
    )
    va_pred = clf.predict_proba(X_train[va])[:, 1]
    oof[va] = va_pred
    test_pred += clf.predict_proba(X_test)[:, 1] / N_SPLITS
    auc = roc_auc_score(y[va], va_pred)
    print(f"  fold {fold} va_auc={auc:.5f}  t={time.time()-t_fold:.0f}s",
          flush=True)
    del clf
    torch.cuda.empty_cache()

oof_auc = roc_auc_score(y, oof)
oof_ap  = average_precision_score(y, oof)
oof_ll  = log_loss(y, np.clip(oof, 1e-7, 1 - 1e-7))
print(f"\n==== TabNet OOF ====")
print(f"AUC : {oof_auc:.5f}")
print(f"AP  : {oof_ap:.5f}")
print(f"LL  : {oof_ll:.5f}")
print(f"time: {(time.time()-t0)/60:.1f} min", flush=True)

np.save("/tmp/trial49_oof.npy", oof)
np.save("/tmp/trial49_test.npy", test_pred)

# ── Ensemble probe ───────────────────────────────────────────────────────────
print("\n==== Ensemble probe with TabNet ====", flush=True)
realmlp_oof  = np.load("/tmp/trial46_oof.npy");   realmlp_test = np.load("/tmp/trial46_test.npy")
gbdt_oof     = np.load("/tmp/trial30_oof_chosen.npy"); gbdt_test = np.load("/tmp/trial30_test_chosen.npy")
xgb_oof      = np.load("/tmp/trial30_oof_xgb.npy"); xgb_test   = np.load("/tmp/trial30_test_xgb.npy")
cb_oof       = np.load("/tmp/trial30_oof_cb.npy");  cb_test    = np.load("/tmp/trial30_test_cb.npy")
t35_xgb_oof  = np.load("/tmp/trial35_xgb_oof.npy"); t35_xgb_test = np.load("/tmp/trial35_xgb_test.npy")
t35_cb_oof   = np.load("/tmp/trial35_cb_oof.npy");  t35_cb_test  = np.load("/tmp/trial35_cb_test.npy")
t43_tfm_oof  = np.load("/tmp/trial43_tfm_oof.npy"); t43_tfm_test = np.load("/tmp/trial43_tfm_test.npy")

def rcorr(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]

print("\nrank-corrs vs TabNet:")
for name, arr in [("T46_realmlp", realmlp_oof), ("T30_chosen", gbdt_oof),
                  ("T30_xgb", xgb_oof), ("T30_cb", cb_oof),
                  ("T35_xgb", t35_xgb_oof), ("T35_cb", t35_cb_oof),
                  ("T43_tfm", t43_tfm_oof)]:
    print(f"  tabnet vs {name:14s}: {rcorr(oof, arr):.4f}")

def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

def lr_meta_oof(features, y, groups, n_splits=5, seed=42):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.zeros(len(y))
    for tr, va in cv.split(features, y, groups=groups):
        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(features[tr], y[tr])
        out[va] = lr.predict_proba(features[va])[:, 1]
    return out

def lr_full(oof_feats, test_feats, y):
    lr = LogisticRegression(C=1.0, max_iter=200); lr.fit(oof_feats, y)
    print(f"   coefs={np.round(lr.coef_.ravel(),3)}  b={lr.intercept_[0]:.3f}")
    return lr.predict_proba(test_feats)[:, 1]

print("\nEnsembles (OOF AUC):")
print(f"  TabNet alone               : {oof_auc:.5f}")

# 2-way: TabNet + T46
X2 = np.column_stack([logit(oof), logit(realmlp_oof)])
m2 = lr_meta_oof(X2, y, groups)
print(f"  LR-meta(TabNet, T46)       : {roc_auc_score(y, m2):.5f}")

# T47's best 6-way + TabNet = 7-way
X7 = np.column_stack([logit(realmlp_oof), logit(xgb_oof), logit(cb_oof),
                      logit(t35_xgb_oof), logit(t35_cb_oof),
                      logit(t43_tfm_oof), logit(oof)])
m7 = lr_meta_oof(X7, y, groups)
print(f"  LR-meta 7-way (+ TabNet)   : {roc_auc_score(y, m7):.5f}")

# T46 + T30_chosen + T43 + TabNet (compact 4-way)
X4 = np.column_stack([logit(realmlp_oof), logit(gbdt_oof),
                      logit(t43_tfm_oof), logit(oof)])
m4 = lr_meta_oof(X4, y, groups)
print(f"  LR-meta 4-way (T46+GBDT+T43+TabNet): {roc_auc_score(y, m4):.5f}")

# Reference: T47's 6-way (no TabNet)
X6 = np.column_stack([logit(realmlp_oof), logit(xgb_oof), logit(cb_oof),
                      logit(t35_xgb_oof), logit(t35_cb_oof), logit(t43_tfm_oof)])
m6 = lr_meta_oof(X6, y, groups)
print(f"  LR-meta 6-way (T47, no TabNet): {roc_auc_score(y, m6):.5f}")

# Write best submission
print("\nRefit best on full OOF, predict test...")
best_oof_auc = max(roc_auc_score(y, m7), roc_auc_score(y, m4),
                   roc_auc_score(y, m2), roc_auc_score(y, m6), oof_auc)
print(f"Best OOF AUC across probes: {best_oof_auc:.5f}")
test_blend = lr_full(X7,
    np.column_stack([logit(realmlp_test), logit(xgb_test), logit(cb_test),
                     logit(t35_xgb_test), logit(t35_cb_test),
                     logit(t43_tfm_test), logit(test_pred)]),
    y)
sub = pd.DataFrame({"id": test["id"].values, "PitNextLap": test_blend})
sub.to_csv(DATA_DIR / "submission_trial49.csv", index=False)
print(f"Wrote submission_trial49.csv (7-way LR-meta incl TabNet)")
