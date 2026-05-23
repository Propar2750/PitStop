"""Trial 61: Multi-tier confidence gate (blender notebook §10).

Anchor = T59 OOF. Apply logit-rank-blend only inside the ambiguous middle
(default 0.15-0.85 core, 0.02-0.15 / 0.85-0.98 edge), leaving absolute
extremes untouched. Sweep support OOF and (core_w, edge_w).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _blender_methods import multi_tiered_gate

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
    "T35_cb":  ("trial35_cb_oof.npy",  "trial35_cb_test.npy"),
}
supports = {k: np.load(f"/tmp/{v[0]}") for k, v in SUPPORT_KEYS.items()}
supports_t = {k: np.load(f"/tmp/{v[1]}") for k, v in SUPPORT_KEYS.items()}

# (core_w, edge_w) pairs incl. notebook defaults (0.04, 0.01) and harder pushes
WEIGHT_PAIRS = [(0.02, 0.005), (0.04, 0.01), (0.08, 0.02), (0.15, 0.04), (0.25, 0.08)]

print(f"\n{'support':<10s} {'core_w':>7s} {'edge_w':>7s} {'OOF AUC':>10s} {'delta':>9s}")
best = (anchor_auc, None, None, None, None, None)
for sup_tag, sup_oof in supports.items():
    for cw, ew in WEIGHT_PAIRS:
        blended = multi_tiered_gate(anchor_oof, sup_oof, core_w=cw, edge_w=ew)
        auc = roc_auc_score(y, blended)
        marker = " *" if auc > best[0] else ""
        print(f"{sup_tag:<10s} {cw:7.3f} {ew:7.3f} {auc:10.5f} {auc - anchor_auc:+9.5f}{marker}")
        if auc > best[0]:
            test_blend = multi_tiered_gate(anchor_test, supports_t[sup_tag], core_w=cw, edge_w=ew)
            best = (auc, sup_tag, cw, ew, blended, test_blend)

print(f"\n=== BEST ===")
print(f"  support={best[1]}  core_w={best[2]}  edge_w={best[3]}  OOF AUC={best[0]:.5f}  delta={best[0]-anchor_auc:+.5f}")
if best[4] is not None:
    print(f"  AP={average_precision_score(y, best[4]):.5f}  LL={log_loss(y, np.clip(best[4],1e-7,1-1e-7)):.5f}")
    np.save("/tmp/trial61_oof.npy", best[4])
    np.save("/tmp/trial61_test.npy", best[5])
    pd.DataFrame({"id": test_id, "PitNextLap": best[5]}).to_csv(DATA_DIR / "submission_trial61.csv", index=False)
    print(f"  Wrote submission_trial61.csv")
else:
    print("  No (support, weights) combination improved over anchor.")
