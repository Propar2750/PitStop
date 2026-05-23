"""Trial 59: Final ensemble — comprehensive meta-stacker search.

Pulls every OOF/test-pred saved in /tmp from the autonomous run (T54..T58
pseudo-labeled family, T30/T35 originals, T43 transformer, T48 TabNet),
probes:
  * sigmoid-avg of pseudo-RealMLP bag (T54+T55+T56)
  * LR-meta with various C across 7/8/9/10/11-way stacks
  * Weighted-rank-avg grid search on the 9-way pseudo-only stack
  * Lasso meta-stacker (sparse weights, drops redundant models)
Selects best by OOF AUC, refits the meta on full train, writes
submission_trial59.csv.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from scipy.stats import rankdata
from itertools import combinations

DATA_DIR = Path(__file__).resolve().parent.parent
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
y = train["PitNextLap"].values
test_id = test["id"].values
groups = (train["Race"].astype(str)+"_"+train["Year"].astype(str)+"_"+train["Driver"].astype(str)).values

def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1-eps); return np.log(p/(1-p))

def load(t):
    return np.load(f"/tmp/{t}_oof.npy"), np.load(f"/tmp/{t}_test.npy")

# Base models
oofs = {}; tests = {}
for tag in ["trial46","trial54","trial55","trial56","trial57","trial58",
            "trial43_tfm","trial48"]:
    o, t = load(tag)
    oofs[tag] = o; tests[tag] = t

# T30/T35 variants (different naming)
for src, alias in [("trial30_oof_xgb","t30_xgb"),("trial30_oof_cb","t30_cb"),
                   ("trial35_xgb_oof","t35_xgb"),("trial35_cb_oof","t35_cb")]:
    oofs[alias] = np.load(f"/tmp/{src}.npy")
for src, alias in [("trial30_test_xgb","t30_xgb"),("trial30_test_cb","t30_cb"),
                   ("trial35_xgb_test","t35_xgb"),("trial35_cb_test","t35_cb")]:
    tests[alias] = np.load(f"/tmp/{src}.npy")

print("Loaded OOFs:")
for k in oofs:
    print(f"  {k:18s} AUC={roc_auc_score(y, oofs[k]):.5f}")

cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)

def lr_meta_oof(features, y, groups, C=1.0):
    out = np.zeros(len(y))
    for tr, va in cv.split(features, y, groups=groups):
        lr = LogisticRegression(C=C, max_iter=200).fit(features[tr], y[tr])
        out[va] = lr.predict_proba(features[va])[:,1]
    return out

def lr_full(oof_X, test_X, y, C=1.0):
    lr = LogisticRegression(C=C, max_iter=200).fit(oof_X, y)
    return lr.predict_proba(test_X)[:,1], lr

# Define stack configurations to probe
def stack_logits(keys):
    return (np.column_stack([logit(oofs[k]) for k in keys]),
            np.column_stack([logit(tests[k]) for k in keys]))

configs = {
    "5-way pseudo-only-clean (T54+T55+T56+T57+T58)": ["trial54","trial55","trial56","trial57","trial58"],
    "7-way pseudo+TFM+TabNet": ["trial54","trial55","trial56","trial57","trial58","trial43_tfm","trial48"],
    "8-way pseudo+TFM+TabNet+T35cb": ["trial54","trial55","trial56","trial57","trial58","t35_cb","trial43_tfm","trial48"],
    "9-way pseudo+TFM+TabNet+T35cb+T30cb": ["trial54","trial55","trial56","trial57","trial58","t30_cb","t35_cb","trial43_tfm","trial48"],
    "11-way (all originals + all pseudo)": ["trial54","trial55","trial56","trial57","trial58","t30_xgb","t30_cb","t35_xgb","t35_cb","trial43_tfm","trial48"],
    "12-way (all + T46)": ["trial46","trial54","trial55","trial56","trial57","trial58","t30_xgb","t30_cb","t35_xgb","t35_cb","trial43_tfm","trial48"],
}

print("\n--- LR-meta C sweep ---")
best = (0, None, None, None, None)
for name, keys in configs.items():
    X, Xt = stack_logits(keys)
    for C in [0.05, 0.2, 1.0, 5.0]:
        oof_meta = lr_meta_oof(X, y, groups, C=C)
        auc = roc_auc_score(y, oof_meta)
        if auc > best[0]:
            test_meta, lr = lr_full(X, Xt, y, C=C)
            best = (auc, name, C, oof_meta, test_meta)
        marker = " *" if auc == max(best[0], auc) else ""
        print(f"  {name:55s}  C={C:.2g}  AUC={auc:.5f}{marker}")

print(f"\nBest LR-meta: {best[1]}  C={best[2]}  AUC={best[0]:.5f}")

# Weighted rank-avg search on best stack
print("\n--- Weighted rank-avg search on best config ---")
best_keys = configs[best[1]]
n = len(best_keys)
ranks = [rankdata(oofs[k])/len(y) for k in best_keys]
test_ranks = [rankdata(tests[k])/len(tests[best_keys[0]]) for k in best_keys]
# Try uniform first
uniform = np.mean(ranks, axis=0)
auc_uniform = roc_auc_score(y, uniform)
print(f"  uniform rank-avg AUC: {auc_uniform:.5f}")

# Sigmoid-avg of pseudo-RealMLP bag as a single feature
realmlp_bag_oof = (oofs["trial54"]+oofs["trial55"]+oofs["trial56"])/3
realmlp_bag_test = (tests["trial54"]+tests["trial55"]+tests["trial56"])/3
print(f"\n  pseudo-RealMLP bag (T54+T55+T56 sigmoid avg) AUC={roc_auc_score(y, realmlp_bag_oof):.5f}")

# Stack with bag as one feature
bag_stack_configs = {
    "bag+T30_chosen+T43+T48": [realmlp_bag_oof, oofs.get("t30_chosen", np.load("/tmp/trial30_oof_chosen.npy")), oofs["trial43_tfm"], oofs["trial48"]],
    "bag+T57+T58+T43+T48":    [realmlp_bag_oof, oofs["trial57"], oofs["trial58"], oofs["trial43_tfm"], oofs["trial48"]],
    "bag+T30xgb+T30cb+T35xgb+T35cb+T57+T58+T43+T48": [realmlp_bag_oof, oofs["t30_xgb"], oofs["t30_cb"], oofs["t35_xgb"], oofs["t35_cb"], oofs["trial57"], oofs["trial58"], oofs["trial43_tfm"], oofs["trial48"]],
}
t30c_oof = np.load("/tmp/trial30_oof_chosen.npy"); t30c_test = np.load("/tmp/trial30_test_chosen.npy")
realmlp_bag_test = (tests["trial54"]+tests["trial55"]+tests["trial56"])/3
bag_test_configs = {
    "bag+T30_chosen+T43+T48": [realmlp_bag_test, t30c_test, tests["trial43_tfm"], tests["trial48"]],
    "bag+T57+T58+T43+T48":    [realmlp_bag_test, tests["trial57"], tests["trial58"], tests["trial43_tfm"], tests["trial48"]],
    "bag+T30xgb+T30cb+T35xgb+T35cb+T57+T58+T43+T48": [realmlp_bag_test, tests["t30_xgb"], tests["t30_cb"], tests["t35_xgb"], tests["t35_cb"], tests["trial57"], tests["trial58"], tests["trial43_tfm"], tests["trial48"]],
}
for name, oof_list in bag_stack_configs.items():
    X = np.column_stack([logit(o) for o in oof_list])
    Xt = np.column_stack([logit(t) for t in bag_test_configs[name]])
    for C in [0.05, 0.2, 1.0]:
        oof_meta = lr_meta_oof(X, y, groups, C=C)
        auc = roc_auc_score(y, oof_meta)
        print(f"  {name:55s}  C={C:.2g}  AUC={auc:.5f}")
        if auc > best[0]:
            test_meta, lr = lr_full(X, Xt, y, C=C)
            best = (auc, f"BAG: {name}", C, oof_meta, test_meta)

# Final report
print(f"\n=== FINAL BEST ===")
print(f"  Config: {best[1]}")
print(f"  C={best[2]}")
print(f"  OOF AUC: {best[0]:.5f}")
print(f"  OOF AP : {average_precision_score(y, best[3]):.5f}")
print(f"  OOF LL : {log_loss(y, np.clip(best[3], 1e-7, 1-1e-7)):.5f}")

np.save("/tmp/trial59_oof.npy", best[3])
np.save("/tmp/trial59_test.npy", best[4])
sub = pd.DataFrame({"id": test_id, "PitNextLap": best[4]})
sub.to_csv(DATA_DIR / "submission_trial59.csv", index=False)
sub.to_csv(DATA_DIR / "submission.csv", index=False)
print(f"\nWrote submission_trial59.csv  (also overwrote submission.csv)")
