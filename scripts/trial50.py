"""Trial 50: T46 RealMLP + 19 T21 engineered multiplicative features.

Mechanism the same as T42 -> T43 for the transformer: GBDTs are insensitive
to these features (12/19 had zero gain in T30 importance pass) but MLPs see
multiplicative interactions differently. T43 saw +0.00259 standalone from
adding these. RealMLP currently doesn't have them — adding gives more
information without changing architecture.

Single change vs T46: 19 extra engineered numerics appended to the
NumericalPreprocessor input. All other knobs identical.
"""
"""Trial 46: RealMLP (PyTorch), trial45 recipe under our baseline CV.

Same RealMLP architecture, features, and hyperparams as trial45, with two
changes to make the score directly comparable to T1-T44:

  * CV scheme changed from StratifiedKFold (row-level) to
    StratifiedGroupKFold with group = Race + "_" + Year + "_" + Driver
    so same-stint rows can't leak across folds.

  * External data f1_strategy_dataset_v4.csv (101k rows) is NOT
    concatenated to the training fold. trial45 used it; no other trial in
    the log does.

Everything else (feature engineering, NumericalPreprocessor, RealMLP module,
CONFIG dict, target encoding of combo features, 4 epochs / bs=256 / GPU) is
identical to trial45.
"""
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent

print("PyTorch version:", torch.__version__)


def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ────────────────────────────────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print("Train shape:", train.shape)
print("Test  shape:", test.shape)


# ────────────────────────────────────────────────────────────────────────────
# Preprocess Features
# ────────────────────────────────────────────────────────────────────────────
ID = "id"
TARGET = "PitNextLap"
# Build group key BEFORE feature_engineering categorizes Race/Year/Driver
groups = (
    train["Race"].astype(str) + "_" + train["Year"].astype(str)
    + "_" + train["Driver"].astype(str)
).values
X = train.drop([ID, TARGET], axis=1)
train_id = train[ID]
y = train[TARGET]
X_test = test.drop([ID], axis=1)
test_id = test[ID]
del train, test

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
print("init len(cat_cols):", len(cat_cols))
print("init len(num_cols):", len(num_cols))

category_map = {}
important_combos = [
    ("Race", "Compound"),
    ("Race", "Year"),
]


def feature_engineering(df, fit=False):
    df["_LapNumber_/_RaceProgress"] = (
        df["LapNumber"] / (df["RaceProgress"] + 1e-6)
    ).astype("float32")
    df["_TyreLife_/_LapNumber"] = (
        df["TyreLife"] / df["LapNumber"].clip(lower=1)
    ).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation"] = (
        df["LapTime (s)"] * df["Cumulative_Degradation"]
    ).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (
        df["LapTime (s)"] * df["Cumulative_Degradation"].abs()
    ).astype("float32")
    df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (
        df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)
    ).astype("float32")

    # T21 / T42 engineered multiplicative features (19) — prefix with _ so
    # they get auto-grouped into new_num_cols by the bookkeeping below.
    eps = 1e-6
    df["_RemainingRace"]           = (1.0 - df["RaceProgress"]).astype("float32")
    df["_PitWindow"]               = (df["RaceProgress"] * (1.0 - df["RaceProgress"])).astype("float32")
    df["_IsLateRace"]              = (df["RaceProgress"] > 0.75).astype("float32")
    df["_LapTime_per_TyreLife"]    = (df["LapTime (s)"] / (df["TyreLife"] + eps)).astype("float32")
    df["_Deg_per_TyreLife"]        = (df["Cumulative_Degradation"] / (df["TyreLife"] + eps)).astype("float32")
    df["_TyreStress"]              = (df["TyreLife"] * df["Cumulative_Degradation"]).astype("float32")
    df["_StrategicUrgency"]        = (df["_TyreStress"] * df["_RemainingRace"]).astype("float32")
    df["_TyreExhaustion"]          = ((df["TyreLife"] ** 2) * df["_RemainingRace"]).astype("float32")
    df["_TyreCliff"]               = (df["TyreLife"] * df["LapTime_Delta"]).astype("float32")
    df["_PositionPressure"]        = (df["Position"] * df["_RemainingRace"]).astype("float32")
    df["_RecoveryPressure"]        = (df["Position_Change"].abs() * df["Cumulative_Degradation"]).astype("float32")
    df["_CompoundCode"]            = pd.Categorical(df["Compound"].astype(str)).codes.astype("float32")
    df["_CompoundTyreInteraction"] = (df["_CompoundCode"] * df["TyreLife"]).astype("float32")
    df["_StrategyPressure"]        = (df["_TyreStress"] * df["_PitWindow"]).astype("float32")
    df["_PitOffsetPotential"]      = (df["LapTime_Delta"] * df["_RemainingRace"] * 10.0).astype("float32")
    df["_UndercutPotential"]       = (df["LapTime_Delta"] * df["Position_Change"].abs() * df["_RemainingRace"]).astype("float32")
    df["_StintSurvivalPressure"]   = (df["TyreLife"] * df["_RemainingRace"] * df["LapTime_Delta"]).astype("float32")
    df["_PaceCollapse"]            = (df["LapTime_Delta"] * df["Cumulative_Degradation"]).astype("float32")
    df["_LateRaceTyreRisk"]        = (df["_IsLateRace"] * df["TyreLife"]).astype("float32")

    for col in cat_cols:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype("int32")
        df[col] = codes
        df[col] = df[col].astype("category")

    for col in num_cols + ["_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"]:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype("category")

    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].astype(object).map(count_map).fillna(0).astype("int32")

    bin_config = {"RaceProgress": [200], "LapTime (s)": [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ["quantile"]:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode="ordinal",
                        strategy=strategy,
                        subsample=None,
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype("int32")
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype("category")

    combo_names = []
    for cols in important_combos:
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + "_" + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype("category")

    new_cat_cols = [c for c in df.columns if c.endswith("_")]
    new_num_cols = [c for c in df.columns if c.startswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names


X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
X_test, *_ = feature_engineering(X_test, fit=False)
cat_cols += new_cat_cols
num_cols += new_num_cols
print("prep len(cat_cols):", len(cat_cols))
print("prep len(num_cols):", len(num_cols))
print("X      prep shape:", X.shape)
print("X_test prep shape:", X_test.shape)


# ────────────────────────────────────────────────────────────────────────────
# Model components (verbatim from notebook)
# ────────────────────────────────────────────────────────────────────────────
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [
            t for t in tfms
            if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")
        ]

    def fit(self, X, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X, y=None):
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8, device=None):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []
        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList(
                    [nn.Embedding(dim, embed_dim) for _ in range(n_ens)]
                )
                self.embed_layers.append(emb)
                self._embed_feature_indices.append(i)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx : idx + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)
        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx : feat_idx + 1].long()
                feat_embs.append(emb_list[model_idx](indices))
            feat_combined = torch.cat(feat_embs, dim=1)
            features.append(feat_combined)
        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4,
                 freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(
            torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim)
        )
        self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(
            2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0))
        )
        transformed = self.act(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0)
        )
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"],
        )
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens, n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"], out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"], activation=cfg["pbld_activation"],
        )
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(
            c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims
        )
        total_dim = num_emb_dim + cat_emb_dim
        hidden_dims = cfg["hidden_dims"]
        act = cfg["activation"]
        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(hidden_dims):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0:
                self.first_linear = linear
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim_h
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined)
        x = self.output_layer(x)
        return x


def apply_schedule(init_value, progress, sched, flat_ratio=0.3):
    if sched == "constant":
        return init_value
    elif sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    elif sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    elif sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    elif sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    elif sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    else:
        raise ValueError(f"Unknown schedule: '{sched}'")


def get_parameter_groups(model, p):
    first_linear_weight_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_linear_weight_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    LR = p["lr"]
    WD = p["weight_decay"]
    return [
        {"params": scale_p,   "lr": LR * p["lr_scale_mult"],         "weight_decay": WD * p["wd_scale_mult"], "group": "scale"},
        {"params": pbld_p,    "lr": LR * p["pbld_lr_factor"],        "weight_decay": WD,                      "group": "pbld"},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD,                      "group": "first_w"},
        {"params": other_w_p, "lr": LR,                              "weight_decay": WD,                      "group": "other_w"},
        {"params": bias_p,    "lr": LR * p["lr_bias_mult"],          "weight_decay": WD * p["wd_bias_mult"],  "group": "bias"},
    ]


def binary_bce_loss(y_true, logits, ls=0.0, pos_weight=None):
    if ls > 0.0:
        y_true = y_true * (1.0 - ls) + 0.5 * ls
    if pos_weight is None:
        loss = (1.0 - y_true) * logits + F.softplus(-logits)
    else:
        loss = (1.0 - y_true) * logits + (1.0 + (pos_weight - 1.0) * y_true) * F.softplus(-logits)
    return loss.mean()


class RealMLP_TD_Classifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None,
            ckpt_path="realmlp_ckpt.pth", X_test=None):
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]
        X_tr_num = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr = np.asarray(y_train)
        y_v = np.asarray(y_val)

        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names
        if cat_col_names:
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None:
                all_cat.append(X_test[cat_col_names].values.astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims
        if cat_dims:
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        pos_weight = torch.tensor(weights_np[1], dtype=torch.float32, device=dev)

        self.model_ = RealMLP(
            output_dim=1, cat_dims=cat_dims,
            n_numerical=X_tr_num.shape[1], cfg=p,
        ).to(dev)

        param_groups = get_parameter_groups(self.model_, p)
        for g in param_groups:
            g["lr_base"] = g["lr"]
        optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.float32, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

        n_ens = p["n_ens"]
        train_bs = p["train_bs"]
        eval_bs = p["eval_bs"]
        epochs = p["epochs"]
        lr_sched = p["lr_sched"]
        flat_ratio = p["flat_ratio"]
        total_steps = epochs * len(y_tr)
        train_order = np.arange(len(y_tr))

        best_score = -np.inf
        best_epoch = 0
        best_val_probs = None
        self.ckpt_path_ = ckpt_path

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx_batch = train_order[start : start + train_bs]
                for g in optimizer.param_groups:
                    g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)
                optimizer.zero_grad()
                y_pred = self.model_(Xtn[idx_batch], Xtc[idx_batch])
                ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val
                loss = binary_bce_loss(
                    ytt[idx_batch].repeat_interleave(n_ens),
                    y_pred.reshape(-1),
                    ls=ls_val,
                    pos_weight=pos_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()
            np.random.shuffle(train_order)

            self.model_.eval()
            with torch.no_grad():
                val_probs_pos = np.concatenate([
                    torch.sigmoid(self.model_(Xvn[s : s + eval_bs], Xvc[s : s + eval_bs]))
                        .mean(dim=1).squeeze(-1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)
                ], axis=0)
                val_probs = np.stack([1.0 - val_probs_pos, val_probs_pos], axis=1)
            val_pred = val_probs[:, 1]
            epoch_score = roc_auc_score(y_v, val_pred)
            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_val_probs = val_probs.copy()
                torch.save(self.model_.state_dict(), ckpt_path)
            if verbose >= 2:
                print(
                    f"  epoch {epoch+1}/{epochs}  score={epoch_score:.5f}  "
                    f"best={best_score:.5f}  ls={ls_val:.4f}  drop={drop_val:.4f}"
                    + ("  *" if improved else "")
                )
            if p["use_early_stopping"]:
                patience = (best_epoch * p["early_stopping_multiplicative_patience"]
                            + p["early_stopping_additive_patience"])
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        print(f"  Early stopping at epoch {epoch+1} (best {best_epoch})")
                    break

        self.model_.load_state_dict(torch.load(ckpt_path))
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            print(f"  -> best score: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X):
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(
            X[self.num_col_names_].values.astype(np.float32)
        )
        X_cat = X[self.cat_col_names_].values.astype(np.int64)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            probs_pos = np.concatenate([
                torch.sigmoid(self.model_(Xn[s : s + eval_bs], Xc[s : s + eval_bs]))
                    .mean(dim=1).squeeze(-1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return self.classes_[(probs >= 0.5).astype(np.int64)]


# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Model architecture
    "n_ens": 16,
    "embed_dim": 6,
    "onehot_thresh": 4,
    "hidden_dims": [512, 64, 128],
    "dropout": 0.05,
    "p_drop_sched": "expm4t",
    "activation": nn.SiLU,
    "add_front_scale": True,
    # PBLD
    "pbld_hidden_dim": 20,
    "pbld_out_dim": 5,
    "pbld_freq_scale": 5.0,
    "pbld_activation": nn.PReLU,
    "pbld_lr_factor": 0.093,
    # Optimizer
    "lr": 0.008,
    "mom": 0.9,
    "sq_mom": 0.98,
    "lr_sched": "flat_cos",
    "flat_ratio": 0.3,
    "first_layer_lr_factor": 1.0,
    "lr_scale_mult": 10.0,
    "lr_bias_mult": 0.1,
    "weight_decay": 0.005,
    "wd_scale_mult": 0.1,
    "wd_bias_mult": 0.5,
    "grad_clip": 1.0,
    # Label smoothing
    "ls_eps": 0.04,
    "ls_eps_sched": "cos",
    # Preprocessing
    "tfms": ["median_center", "robust_scale", "smooth_clip"],
    # Training loop
    "epochs": 4,
    "train_bs": 256,
    "eval_bs": 10240,
    "verbosity": 2,
    # Early stopping
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    # Device
    "device": "cuda",
    "random_state": 42,
}

FOLDS = 5
SEED = 42


# ────────────────────────────────────────────────────────────────────────────
# Train K-Fold
# ────────────────────────────────────────────────────────────────────────────
skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

t0 = time.time()
TE_FLAG = True
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y, groups=groups), 1):
    X_tr = X.iloc[tr_idx].copy().reset_index(drop=True)
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].copy()
    y_val = y.iloc[val_idx]
    X_tst = X_test.copy()

    if TE_FLAG:
        te_cols = combo_names
        te = TargetEncoder(cv=FOLDS, smooth="auto", shuffle=True, random_state=SEED)
        tr_enc = te.fit_transform(X_tr[te_cols], y_tr)
        val_enc = te.transform(X_val[te_cols])
        tst_enc = te.transform(X_tst[te_cols])
        te_names = [f"_{col}TE" for col in te_cols]
        X_tr[te_names] = tr_enc
        X_val[te_names] = val_enc
        X_tst[te_names] = tst_enc

    if fold == 1:
        print("len(FEATURES):", len(X_tr.columns.tolist()))
    print("#" * 16)
    print(f"### Fold {fold}/{FOLDS}")
    print("#" * 16)

    model = RealMLP_TD_Classifier(**CONFIG)
    model.fit(
        X_tr, y_tr,
        X_val, y_val,
        cat_col_names=cat_cols,
        ckpt_path=f"/tmp/realmlp50_fold{fold}.pth",
    )

    val_preds = model.best_val_probs_[:, 1]
    fold_test_preds = model.predict_proba(X_tst)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += fold_test_preds / FOLDS

    fold_score = roc_auc_score(y_val, val_preds)
    print(f"\nFold {fold} | AUC: {fold_score:.5f}\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

oof_auc = roc_auc_score(y, oof_preds)
oof_ap = average_precision_score(y, oof_preds)
oof_ll = log_loss(y, np.clip(oof_preds, 1e-7, 1 - 1e-7))
print("=" * 26)
print(f"OOF AUC : {oof_auc:.5f}")
print(f"OOF AP  : {oof_ap:.5f}")
print(f"OOF LL  : {oof_ll:.5f}")
print(f"Time    : {(time.time() - t0)/60:.1f} min")
print("=" * 26)

# Save OOF + submission
np.save("/tmp/trial50_oof.npy", oof_preds)
np.save("/tmp/trial50_test.npy", test_preds)
sub = pd.DataFrame({ID: test_id, TARGET: test_preds})
sub.to_csv(DATA_DIR / "submission_trial50.csv", index=False)
print("Wrote submission_trial50.csv")
