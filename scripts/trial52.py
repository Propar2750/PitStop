"""Trial 52: XGBoost meta-stacker (vs LR-meta in T47/T48).

LR-meta is a linear blend on logits. If there's any nonlinearity in how the
base models combine (e.g. "trust RealMLP more on early-race rows, trust GBDT
more on late-race"), an XGB meta-stacker can capture it. Same 7-way inputs
as T48 (T46 RealMLP + T30 XGB/CB + T35 XGB/CB + T43 TFM + T48 TabNet).

Risk: XGB on 7 logit features with 440k rows can overfit the stacker even
with shallow trees; use very conservative hparams (max_depth=3, n_est=200,
reg_alpha=0.5, reg_lambda=2.0). 5-fold group-CV OOF to get a fair score.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent.parent
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
y = train["PitNextLap"].values
groups = (train["Race"].astype(str)+"_"+train["Year"].astype(str)
          +"_"+train["Driver"].astype(str)).values

def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1-eps); return np.log(p/(1-p))

names = ["T46_realmlp", "T30_xgb", "T30_cb", "T35_xgb", "T35_cb", "T43_tfm", "T48_tabnet"]
oofs = [np.load(f"/tmp/trial46_oof.npy"),
        np.load("/tmp/trial30_oof_xgb.npy"),
        np.load("/tmp/trial30_oof_cb.npy"),
        np.load("/tmp/trial35_xgb_oof.npy"),
        np.load("/tmp/trial35_cb_oof.npy"),
        np.load("/tmp/trial43_tfm_oof.npy"),
        np.load("/tmp/trial48_oof.npy")]
tests = [np.load(f"/tmp/trial46_test.npy"),
        np.load("/tmp/trial30_test_xgb.npy"),
        np.load("/tmp/trial30_test_cb.npy"),
        np.load("/tmp/trial35_xgb_test.npy"),
        np.load("/tmp/trial35_cb_test.npy"),
        np.load("/tmp/trial43_tfm_test.npy"),
        np.load("/tmp/trial48_test.npy")]

X = np.column_stack([logit(o) for o in oofs])
Xt = np.column_stack([logit(t) for t in tests])

# Configs to probe
configs = [
    dict(max_depth=2, n_estimators=200, learning_rate=0.05, reg_alpha=1.0, reg_lambda=2.0, subsample=0.8, min_child_weight=20),
    dict(max_depth=3, n_estimators=200, learning_rate=0.05, reg_alpha=1.0, reg_lambda=2.0, subsample=0.8, min_child_weight=20),
    dict(max_depth=3, n_estimators=400, learning_rate=0.03, reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, min_child_weight=20),
    dict(max_depth=4, n_estimators=200, learning_rate=0.03, reg_alpha=1.0, reg_lambda=2.0, subsample=0.8, min_child_weight=50),
]

best = (0, None, None, None)
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
for ci, cfg in enumerate(configs):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(Xt))
    for tr, va in cv.split(X, y, groups=groups):
        clf = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", device="cuda", verbosity=0,
            **cfg,
        )
        clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = clf.predict_proba(X[va])[:,1]
        test_pred += clf.predict_proba(Xt)[:,1] / 5
    auc = roc_auc_score(y, oof)
    print(f"cfg{ci}: {cfg}  AUC={auc:.5f}")
    if auc > best[0]:
        best = (auc, cfg, oof, test_pred)

print(f"\nBest XGB meta: AUC={best[0]:.5f}")
print(f"  AP={average_precision_score(y, best[2]):.5f}  LL={log_loss(y, np.clip(best[2],1e-7,1-1e-7)):.5f}")
print(f"vs LR-meta 7-way (T48): 0.95284")

np.save("/tmp/trial52_oof.npy", best[2])
np.save("/tmp/trial52_test.npy", best[3])
if best[0] > 0.95284:
    sub = pd.DataFrame({"id": test["id"].values, "PitNextLap": best[3]})
    sub.to_csv(DATA_DIR / "submission_trial52.csv", index=False)
    print("Wrote submission_trial52.csv (improvement over T48)")
else:
    print("No improvement over T48; skipping submission write.")
