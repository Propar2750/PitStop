"""Trial 63: Piecewise / micro-band rescaling (blender notebook §8).

Within `bins` quantile bins of the anchor, multiply by support_mean/anchor_mean.
Within each bin the rank is unchanged; only adjacent-bin ordering can shift.
This is primarily a CALIBRATION operator (affects log-loss strongly, AUC weakly).
Sweep bins ∈ {20, 100} and supports.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _blender_methods import piecewise_rescale

DATA_DIR = Path(__file__).resolve().parent.parent
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
y = train["PitNextLap"].values
test_id = test["id"].values

anchor_oof = np.load("/tmp/trial59_oof.npy")
anchor_test = np.load("/tmp/trial59_test.npy")
anchor_auc = roc_auc_score(y, anchor_oof)
anchor_ll  = log_loss(y, np.clip(anchor_oof, 1e-7, 1-1e-7))
print(f"Anchor T59: AUC={anchor_auc:.5f}  LL={anchor_ll:.5f}")

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

CONFIGS = [
    (20,  None),
    (20,  (0.95, 1.05)),
    (100, None),
    (100, (0.95, 1.05)),
]

print(f"\n{'support':<10s} {'bins':>5s} {'clip':<12s} {'OOF AUC':>10s} {'delta':>9s} {'LL':>9s}")
best_auc = (anchor_auc, None, None, None, None, None)
best_ll  = (anchor_ll,  None, None, None, None, None)
for sup_tag, sup_oof in supports.items():
    for bins, clip in CONFIGS:
        blended = piecewise_rescale(anchor_oof, sup_oof, bins=bins, scalar_clip=clip)
        auc = roc_auc_score(y, blended)
        ll  = log_loss(y, np.clip(blended, 1e-7, 1-1e-7))
        clip_s = f"{clip}" if clip is not None else "none"
        marker = " *" if auc > best_auc[0] else ""
        print(f"{sup_tag:<10s} {bins:>5d} {clip_s:<12s} {auc:10.5f} {auc-anchor_auc:+9.5f} {ll:9.5f}{marker}")
        if auc > best_auc[0]:
            test_b = piecewise_rescale(anchor_test, supports_t[sup_tag], bins=bins, scalar_clip=clip)
            best_auc = (auc, sup_tag, bins, clip, blended, test_b)
        if ll < best_ll[0]:
            test_b = piecewise_rescale(anchor_test, supports_t[sup_tag], bins=bins, scalar_clip=clip)
            best_ll = (ll, sup_tag, bins, clip, blended, test_b)

print(f"\n=== BEST AUC ===")
print(f"  support={best_auc[1]}  bins={best_auc[2]}  clip={best_auc[3]}  OOF AUC={best_auc[0]:.5f}  delta={best_auc[0]-anchor_auc:+.5f}")
print(f"\n=== BEST LL ===")
print(f"  support={best_ll[1]}  bins={best_ll[2]}  clip={best_ll[3]}  OOF LL={best_ll[0]:.5f}  delta={best_ll[0]-anchor_ll:+.5f}")

# Write the better-AUC variant (matches trial-log convention; LL gain is bonus).
out = best_auc if best_auc[4] is not None else best_ll
if out[4] is not None:
    blended, test_b = out[4], out[5]
    print(f"  AP={average_precision_score(y, blended):.5f}  LL={log_loss(y, np.clip(blended,1e-7,1-1e-7)):.5f}")
    np.save("/tmp/trial63_oof.npy", blended)
    np.save("/tmp/trial63_test.npy", test_b)
    pd.DataFrame({"id": test_id, "PitNextLap": test_b}).to_csv(DATA_DIR / "submission_trial63.csv", index=False)
    print("  Wrote submission_trial63.csv")
else:
    print("  Nothing improved over anchor.")
