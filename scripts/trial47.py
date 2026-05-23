"""Trial 47: Ensemble probe — T46 RealMLP + T30 XGB+CB.

Loads /tmp/trial46_oof.npy + /tmp/trial46_test.npy (RealMLP under
StratifiedGroupKFold, OOF 0.95231) and the T30 GBDT OOFs (XGB 0.94818,
CB 0.94863, chosen 0.94917). Compares per-model standalone, correlations,
rank-avg blends, weighted rank-avg, and LR-meta on logits.

The bet: T46's PBLD-embedded MLP is architecturally orthogonal to GBDTs.
If correlation < ~0.95 we should see a meaningful ensemble lift.

Reads pre-computed OOFs only — does not retrain. Writes
``submission_trial47.csv`` if any ensemble beats T46's 0.95231.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from scipy.stats import rankdata

DATA_DIR = Path(__file__).resolve().parent.parent
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
y = train["PitNextLap"].values
test_id = test["id"].values
groups = (
    train["Race"].astype(str) + "_" + train["Year"].astype(str)
    + "_" + train["Driver"].astype(str)
).values

# Load OOFs / test preds
realmlp_oof   = np.load("/tmp/trial46_oof.npy")
realmlp_test  = np.load("/tmp/trial46_test.npy")
xgb_oof       = np.load("/tmp/trial30_oof_xgb.npy")
xgb_test      = np.load("/tmp/trial30_test_xgb.npy")
cb_oof        = np.load("/tmp/trial30_oof_cb.npy")
cb_test       = np.load("/tmp/trial30_test_cb.npy")
gbdt_chosen_oof  = np.load("/tmp/trial30_oof_chosen.npy")
gbdt_chosen_test = np.load("/tmp/trial30_test_chosen.npy")
# T43 transformer + its underlying T35 XGB/CB (different feature set, GPU)
t35_xgb_oof   = np.load("/tmp/trial35_xgb_oof.npy")
t35_xgb_test  = np.load("/tmp/trial35_xgb_test.npy")
t35_cb_oof    = np.load("/tmp/trial35_cb_oof.npy")
t35_cb_test   = np.load("/tmp/trial35_cb_test.npy")
t43_tfm_oof   = np.load("/tmp/trial43_tfm_oof.npy")
t43_tfm_test  = np.load("/tmp/trial43_tfm_test.npy")

print(f"len(y) = {len(y)}, len(test) = {len(test_id)}")

def report(name, oof):
    auc = roc_auc_score(y, oof)
    ap  = average_precision_score(y, oof)
    print(f"  {name:36s}  AUC={auc:.5f}  AP={ap:.5f}")
    return auc

print("\n--- Standalone ---")
report("T46 RealMLP",         realmlp_oof)
report("T30 XGB",             xgb_oof)
report("T30 CB",              cb_oof)
report("T30 chosen (LR-meta)", gbdt_chosen_oof)
report("T35 XGB (raw, GPU)",  t35_xgb_oof)
report("T35 CB  (raw, GPU)",  t35_cb_oof)
report("T43 Transformer",     t43_tfm_oof)

# Correlations
print("\n--- Correlations on OOF ranks ---")
def rcorr(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]
print(f"  realmlp vs xgb (T30)     : {rcorr(realmlp_oof, xgb_oof):.4f}")
print(f"  realmlp vs cb  (T30)     : {rcorr(realmlp_oof, cb_oof):.4f}")
print(f"  realmlp vs gbdt_chosen   : {rcorr(realmlp_oof, gbdt_chosen_oof):.4f}")
print(f"  realmlp vs t43_tfm       : {rcorr(realmlp_oof, t43_tfm_oof):.4f}")
print(f"  realmlp vs t35_xgb (raw) : {rcorr(realmlp_oof, t35_xgb_oof):.4f}")
print(f"  realmlp vs t35_cb  (raw) : {rcorr(realmlp_oof, t35_cb_oof):.4f}")
print(f"  t43_tfm vs gbdt_chosen   : {rcorr(t43_tfm_oof, gbdt_chosen_oof):.4f}")
print(f"  xgb vs cb (T30)          : {rcorr(xgb_oof, cb_oof):.4f}")

def rank_avg(*arrs, weights=None):
    if weights is None:
        weights = [1.0] * len(arrs)
    n = len(arrs[0])
    ranks = [rankdata(a) / n for a in arrs]
    out = np.zeros(n)
    wsum = 0.0
    for r, w in zip(ranks, weights):
        out += w * r
        wsum += w
    return out / wsum

def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

print("\n--- 2-way rank-avg blends ---")
b1 = rank_avg(realmlp_oof, gbdt_chosen_oof);                    report("rankavg(T46, T30_chosen)",     b1)
b2 = rank_avg(realmlp_oof, xgb_oof);                            report("rankavg(T46, T30_xgb)",        b2)
b3 = rank_avg(realmlp_oof, cb_oof);                             report("rankavg(T46, T30_cb)",         b3)
b4 = rank_avg(realmlp_oof, xgb_oof, cb_oof);                    report("rankavg 3-way (T46,XGB,CB)",   b4)

print("\n--- Weighted rank-avg (T46, T30_chosen) grid ---")
best = (0, 0, 0)
for w in np.arange(0.30, 0.86, 0.05):
    blend = rank_avg(realmlp_oof, gbdt_chosen_oof, weights=[w, 1 - w])
    auc = roc_auc_score(y, blend)
    if auc > best[0]:
        best = (auc, w, blend)
    print(f"  w_realmlp={w:.2f}  AUC={auc:.5f}")
print(f"  best: w_realmlp={best[1]:.2f}  AUC={best[0]:.5f}")
best_w_oof = best[2]

print("\n--- LR-meta on logits (OOF inputs, 5-fold group CV stacker) ---")
def lr_meta_oof(features, y, groups, n_splits=5, seed=42):
    """Out-of-fold LR meta-stack across the same group CV used in base trials."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.zeros(len(y))
    for tr, va in cv.split(features, y, groups=groups):
        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(features[tr], y[tr])
        out[va] = lr.predict_proba(features[va])[:, 1]
    return out

X2 = np.column_stack([logit(realmlp_oof), logit(gbdt_chosen_oof)])
m2 = lr_meta_oof(X2, y, groups);             report("LR-meta(T46, T30_chosen)",   m2)
X3 = np.column_stack([logit(realmlp_oof), logit(xgb_oof), logit(cb_oof)])
m3 = lr_meta_oof(X3, y, groups);             report("LR-meta(T46, T30_xgb, T30_cb)", m3)
X3b = np.column_stack([logit(realmlp_oof), logit(gbdt_chosen_oof), logit(t43_tfm_oof)])
m3b = lr_meta_oof(X3b, y, groups);           report("LR-meta(T46, T30_chosen, T43_tfm)", m3b)
X4 = np.column_stack([logit(realmlp_oof), logit(xgb_oof), logit(cb_oof), logit(t43_tfm_oof)])
m4 = lr_meta_oof(X4, y, groups);             report("LR-meta(T46, T30_xgb, T30_cb, T43_tfm)", m4)
X5 = np.column_stack([logit(realmlp_oof), logit(xgb_oof), logit(cb_oof),
                      logit(t35_xgb_oof), logit(t35_cb_oof), logit(t43_tfm_oof)])
m5 = lr_meta_oof(X5, y, groups);             report("LR-meta(T46, T30xgb, T30cb, T35xgb, T35cb, T43tfm)", m5)

# Pure rank-avg with T43
b5 = rank_avg(realmlp_oof, gbdt_chosen_oof, t43_tfm_oof);  report("rankavg 3-way (T46, T30_chosen, T43_tfm)", b5)
b6 = rank_avg(realmlp_oof, xgb_oof, cb_oof, t43_tfm_oof);  report("rankavg 4-way (T46, XGB, CB, T43_tfm)",    b6)

# Refit on full data for test mapping
print("\n--- Refit best LR-meta on full OOF, predict test ---")
def lr_full_predict_test(oof_feats, test_feats, y):
    lr = LogisticRegression(C=1.0, max_iter=200)
    lr.fit(oof_feats, y)
    print(f"   coefs={lr.coef_.ravel()}  intercept={lr.intercept_[0]:.4f}")
    return lr.predict_proba(test_feats)[:, 1]

# Use best blend (decide which to submit)
candidates = {
    "T46_realmlp_only": (realmlp_oof, realmlp_test),
    "rankavg_T46_T30chosen": (b1, rank_avg(realmlp_test, gbdt_chosen_test)),
    "rankavg_3way": (b4, rank_avg(realmlp_test, xgb_test, cb_test)),
    "weighted_rankavg": (best_w_oof,
        rank_avg(realmlp_test, gbdt_chosen_test, weights=[best[1], 1 - best[1]])),
    "lrmeta_2way": (m2,
        lr_full_predict_test(
            np.column_stack([logit(realmlp_oof), logit(gbdt_chosen_oof)]),
            np.column_stack([logit(realmlp_test), logit(gbdt_chosen_test)]),
            y)),
    "lrmeta_3way": (m3,
        lr_full_predict_test(
            np.column_stack([logit(realmlp_oof), logit(xgb_oof), logit(cb_oof)]),
            np.column_stack([logit(realmlp_test), logit(xgb_test), logit(cb_test)]),
            y)),
    "lrmeta_3way_T43": (m3b,
        lr_full_predict_test(
            np.column_stack([logit(realmlp_oof), logit(gbdt_chosen_oof), logit(t43_tfm_oof)]),
            np.column_stack([logit(realmlp_test), logit(gbdt_chosen_test), logit(t43_tfm_test)]),
            y)),
    "lrmeta_4way": (m4,
        lr_full_predict_test(
            np.column_stack([logit(realmlp_oof), logit(xgb_oof), logit(cb_oof), logit(t43_tfm_oof)]),
            np.column_stack([logit(realmlp_test), logit(xgb_test), logit(cb_test), logit(t43_tfm_test)]),
            y)),
    "lrmeta_6way": (m5,
        lr_full_predict_test(
            np.column_stack([logit(realmlp_oof), logit(xgb_oof), logit(cb_oof),
                             logit(t35_xgb_oof), logit(t35_cb_oof), logit(t43_tfm_oof)]),
            np.column_stack([logit(realmlp_test), logit(xgb_test), logit(cb_test),
                             logit(t35_xgb_test), logit(t35_cb_test), logit(t43_tfm_test)]),
            y)),
    "rankavg_T46_T30chosen_T43": (b5, rank_avg(realmlp_test, gbdt_chosen_test, t43_tfm_test)),
    "rankavg_4way_T43": (b6, rank_avg(realmlp_test, xgb_test, cb_test, t43_tfm_test)),
}
print("\n--- Candidates (OOF AUC) ---")
ranked = sorted(
    ((roc_auc_score(y, o), name, o, t) for name, (o, t) in candidates.items()),
    key=lambda x: -x[0],
)
for auc, name, _, _ in ranked:
    print(f"  {name:24s}  AUC={auc:.5f}")
best_auc, best_name, best_oof, best_test = ranked[0]
print(f"\nBest candidate: {best_name}  AUC={best_auc:.5f}")
print(f"  AP={average_precision_score(y, best_oof):.5f}  "
      f"LL={log_loss(y, np.clip(best_oof,1e-7,1-1e-7)):.5f}")

# Write submission
sub = pd.DataFrame({"id": test_id, "PitNextLap": best_test})
sub.to_csv(DATA_DIR / "submission_trial47.csv", index=False)
np.save("/tmp/trial47_oof.npy", best_oof)
np.save("/tmp/trial47_test.npy", best_test)
print(f"\nWrote submission_trial47.csv ({best_name})")
