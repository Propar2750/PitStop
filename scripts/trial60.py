"""Trial 60: Logit rank-remap blend (blender notebook §9).

Anchor = T59 7-way LR-meta OOF (0.95313 — current best).
Support = sweep over every saved OOF + every 2-OOF rank-avg pair we have.
Weight  = sweep {0.01, 0.02, 0.05, 0.10, 0.20}.
Best (support, weight) by OOF AUC -> apply same transform on test -> submission_trial60.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _blender_methods import logit_rank_blend, clip_pred

DATA_DIR = Path(__file__).resolve().parent.parent
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
y = train["PitNextLap"].values
test_id = test["id"].values

ANCHOR = "trial59"
anchor_oof = np.load(f"/tmp/{ANCHOR}_oof.npy")
anchor_test = np.load(f"/tmp/{ANCHOR}_test.npy")
anchor_auc = roc_auc_score(y, anchor_oof)
print(f"Anchor T59 OOF AUC: {anchor_auc:.5f}")

SUPPORT_KEYS = {
    "T46": ("trial46_oof.npy", "trial46_test.npy"),
    "T54": ("trial54_oof.npy", "trial54_test.npy"),
    "T55": ("trial55_oof.npy", "trial55_test.npy"),
    "T56": ("trial56_oof.npy", "trial56_test.npy"),
    "T57": ("trial57_oof.npy", "trial57_test.npy"),
    "T58": ("trial58_oof.npy", "trial58_test.npy"),
    "T43_tfm": ("trial43_tfm_oof.npy", "trial43_tfm_test.npy"),
    "T48_tab": ("trial48_oof.npy", "trial48_test.npy"),
    "T30_xgb": ("trial30_oof_xgb.npy", "trial30_test_xgb.npy"),
    "T30_cb":  ("trial30_oof_cb.npy",  "trial30_test_cb.npy"),
    "T35_xgb": ("trial35_xgb_oof.npy", "trial35_xgb_test.npy"),
    "T35_cb":  ("trial35_cb_oof.npy",  "trial35_cb_test.npy"),
}

supports = {}
supports_test = {}
for tag, (oof_f, test_f) in SUPPORT_KEYS.items():
    supports[tag] = np.load(f"/tmp/{oof_f}")
    supports_test[tag] = np.load(f"/tmp/{test_f}")

WEIGHTS = [0.01, 0.02, 0.05, 0.10, 0.20]

print(f"\n{'support':<10s} {'w':>5s} {'OOF AUC':>10s} {'delta':>9s}")
best = (anchor_auc, None, None, None, None)
for sup_tag, sup_oof in supports.items():
    for w in WEIGHTS:
        blended = logit_rank_blend(anchor_oof, sup_oof, w)
        auc = roc_auc_score(y, blended)
        marker = " *" if auc > best[0] else ""
        print(f"{sup_tag:<10s} {w:5.2f} {auc:10.5f} {auc - anchor_auc:+9.5f}{marker}")
        if auc > best[0]:
            test_blend = logit_rank_blend(anchor_test, supports_test[sup_tag], w)
            best = (auc, sup_tag, w, blended, test_blend)

print(f"\n=== BEST ===")
print(f"  support={best[1]}  w={best[2]}  OOF AUC={best[0]:.5f}  delta={best[0]-anchor_auc:+.5f}")
if best[3] is not None:
    print(f"  AP={average_precision_score(y, best[3]):.5f}  LL={log_loss(y, np.clip(best[3],1e-7,1-1e-7)):.5f}")
    np.save("/tmp/trial60_oof.npy", best[3])
    np.save("/tmp/trial60_test.npy", best[4])
    pd.DataFrame({"id": test_id, "PitNextLap": best[4]}).to_csv(DATA_DIR / "submission_trial60.csv", index=False)
    print(f"  Wrote submission_trial60.csv")
else:
    print(f"  No (support,w) improved over anchor.")
