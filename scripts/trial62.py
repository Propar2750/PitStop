"""Trial 62: Asymmetric extremum (max/min) rank blends (blender §11).

Two operators:
  rank_max_blend  : elementwise max(rank_anchor, rank_support * w)  — logical OR
  rank_min_blend  : elementwise min(rank_anchor, rank_support / w)  — logical AND
Sweep support OOFs and w. Note the blender notebook treats w as a *multiplier*
on the support rank (or divisor for min); we follow that convention.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _blender_methods import rank_max_blend, rank_min_blend

DATA_DIR = Path(__file__).resolve().parent.parent
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
y = train["PitNextLap"].values
test_id = test["id"].values

anchor_oof = np.load("/tmp/trial59_oof.npy")
anchor_test = np.load("/tmp/trial59_test.npy")
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
    "T35_xgb": ("trial35_xgb_oof.npy", "trial35_xgb_test.npy"),
    "T35_cb":  ("trial35_cb_oof.npy",  "trial35_cb_test.npy"),
}
supports = {k: np.load(f"/tmp/{v[0]}") for k, v in SUPPORT_KEYS.items()}
supports_t = {k: np.load(f"/tmp/{v[1]}") for k, v in SUPPORT_KEYS.items()}

# For max: w is multiplier (<1 shrinks support); for min: w divides support (<1 grows).
W_GRID = [0.80, 0.90, 0.95, 0.98, 1.00]

print(f"\n{'op':<4s} {'support':<10s} {'w':>5s} {'OOF AUC':>10s} {'delta':>9s}")
best = (anchor_auc, None, None, None, None, None)
for sup_tag, sup_oof in supports.items():
    for w in W_GRID:
        for op_name, op in [("MAX", rank_max_blend), ("MIN", rank_min_blend)]:
            blended = op(anchor_oof, sup_oof, w)
            auc = roc_auc_score(y, blended)
            marker = " *" if auc > best[0] else ""
            print(f"{op_name:<4s} {sup_tag:<10s} {w:5.2f} {auc:10.5f} {auc - anchor_auc:+9.5f}{marker}")
            if auc > best[0]:
                test_b = op(anchor_test, supports_t[sup_tag], w)
                best = (auc, op_name, sup_tag, w, blended, test_b)

print(f"\n=== BEST ===")
print(f"  op={best[1]}  support={best[2]}  w={best[3]}  OOF AUC={best[0]:.5f}  delta={best[0]-anchor_auc:+.5f}")
if best[4] is not None:
    print(f"  AP={average_precision_score(y, best[4]):.5f}  LL={log_loss(y, np.clip(best[4],1e-7,1-1e-7)):.5f}")
    np.save("/tmp/trial62_oof.npy", best[4])
    np.save("/tmp/trial62_test.npy", best[5])
    pd.DataFrame({"id": test_id, "PitNextLap": best[5]}).to_csv(DATA_DIR / "submission_trial62.csv", index=False)
    print("  Wrote submission_trial62.csv")
else:
    print("  No (op, support, w) improved over anchor.")
