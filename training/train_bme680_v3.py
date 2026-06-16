#!/usr/bin/env python3
"""
train_bme680_v3.py  —  baseline-invariant classifier

The core idea: every feature is computed relative to the window's OWN
starting point, so absolute gas level never enters the model.
This means the model works even if the sensor boots to a completely
different baseline next power-cycle.

Features used (all baseline-invariant):
  - delta_log_gas: log_gas[-1] - log_gas[0]   (total excursion in window)
  - slope of log_gas (linear fit)
  - slope of ema_diff (is ema_diff rising or falling?)
  - ema_diff at end of window (steady-state direction indicator)
  - ema_diff min, max, mean (all relative within window)
  - std(dlog_dt), max|dlog_dt|  (volatility)
  - sign_changes of dlog_dt     (turbulence oscillates)
  - early/late delta            (direction of curve in window)
  - ema_diff range = max - min  (peak swing, turbulence has wide swings)
  - |slope| of log_gas          (turbulence = high magnitude slope)
  - hum_delta                   (alcohol drops humidity slightly)
  - temp features (not baseline-dependent, just environmental context)

NOT used: gas_ohm mean/level, log_gas mean/level, ema_fast/slow mean/level

Usage:
  python train_bme680_v3.py --csv wAlcohol_bme680_log.csv
"""

from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import joblib

COL_ALIASES = {
    "ms":         ["ms", "time_ms", "timestamp_ms"],
    "label":      ["label", "label_name", "y", "class"],
    "tempC":      ["tempC", "temp_c", "temperature"],
    "humPct":     ["humPct", "hum_pct", "humidity", "rh"],
    "press_hPa":  ["press_hPa", "pressure_hpa", "press"],
    "gas_ohm":    ["gas_ohm", "gas_ohms", "gas", "gas_resistance"],
    "log_gas":    ["log_gas", "loggas"],
    "dloggas_dt": ["dloggas_dt", "dlog_dt"],
    "ema_fast":   ["ema_fast"],
    "ema_slow":   ["ema_slow"],
    "ema_diff":   ["ema_diff"],
}

def resolve(df, canonical):
    for c in COL_ALIASES.get(canonical, []):
        if c in df.columns: return c
    return None

def coerce(s):
    return pd.to_numeric(s, errors="coerce")

def segment_ids(labels):
    seg = np.zeros(len(labels), dtype=np.int32)
    cur = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]: cur += 1
        seg[i] = cur
    return seg

def compute_derived(df, col_ms, col_gas, af=0.25, as_=0.05):
    out = df.copy()
    gas = coerce(out[col_gas]).astype(np.float64).clip(lower=1.0).to_numpy()
    log_gas = np.log(gas)
    ms = coerce(out[col_ms]).astype(np.float64).to_numpy()
    dt_s = np.diff(ms, prepend=ms[0]) / 1000.0
    dt_s[0] = np.nan
    dlog = np.diff(log_gas, prepend=log_gas[0])
    dlog_dt = np.where(np.isfinite(dt_s) & (dt_s > 0), dlog / dt_s, 0.0)

    ema_f = np.empty_like(gas); ema_s = np.empty_like(gas)
    ema_f[0] = gas[0]; ema_s[0] = gas[0]
    for i in range(1, len(gas)):
        ema_f[i] = af * gas[i] + (1-af) * ema_f[i-1]
        ema_s[i] = as_ * gas[i] + (1-as_) * ema_s[i-1]

    if resolve(out, "log_gas")   is None: out["log_gas"]    = log_gas
    if resolve(out, "dloggas_dt") is None: out["dloggas_dt"] = dlog_dt
    if resolve(out, "ema_fast")  is None: out["ema_fast"]   = ema_f
    if resolve(out, "ema_slow")  is None: out["ema_slow"]   = ema_s
    if resolve(out, "ema_diff")  is None: out["ema_diff"]   = ema_f - ema_s
    return out

def linslope(t, x):
    """Return (slope, r2) of linear fit of x vs t."""
    mask = np.isfinite(t) & np.isfinite(x)
    t, x = t[mask], x[mask]
    if len(t) < 3: return 0.0, 0.0
    t0 = t - t[0]
    A = np.vstack([t0, np.ones_like(t0)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    slope = coef[0]
    pred = t0 * slope + coef[1]
    ss_res = np.sum((x - pred)**2)
    ss_tot = np.sum((x - x.mean())**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r2)

def sign_changes(x):
    """Count sign changes in array (ignores zeros)."""
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) < 2: return 0
    signs = np.sign(x)
    return int(np.sum(signs[1:] != signs[:-1]))


@dataclass
class Config:
    window_s: float
    stride_s: float
    warmup_s: float
    label_map: Dict[str, str]
    drop_labels: List[str]


def build_windows(df, cfg):
    col_ms  = resolve(df, "ms")
    col_lab = resolve(df, "label")
    col_gas = resolve(df, "gas_ohm")
    col_tmp = resolve(df, "tempC")
    col_hum = resolve(df, "humPct")

    df = df.copy()
    for c in [col_ms, col_gas, col_tmp, col_hum]:
        df[c] = coerce(df[c])

    lab = df[col_lab].astype(str).str.strip().map(cfg.label_map).fillna(df[col_lab].astype(str).str.strip())
    df["label_name"] = lab
    if cfg.drop_labels:
        df = df[~df["label_name"].isin(cfg.drop_labels)].reset_index(drop=True)

    df = compute_derived(df, col_ms=col_ms, col_gas=col_gas)
    df["_seg"] = segment_ids(df["label_name"].to_numpy())

    ms  = df[col_ms].to_numpy(dtype=np.float64)
    t_s = (ms - ms[0]) / 1000.0

    col_lg = resolve(df, "log_gas")   or "log_gas"
    col_dl = resolve(df, "dloggas_dt") or "dloggas_dt"
    col_ed = resolve(df, "ema_diff")  or "ema_diff"

    seg_arr = df["_seg"].to_numpy()
    seg_start = {}
    for seg, ts in zip(seg_arr, t_s):
        if seg not in seg_start: seg_start[seg] = ts

    win, stride = cfg.window_s, cfg.stride_s
    feats = []
    feature_names: List[str] = []

    def push(row, name, val):
        row[name] = float(val) if np.isfinite(val) else 0.0
        if name not in feature_names: feature_names.append(name)

    t0 = t_s[0]
    while t0 + win <= t_s[-1]:
        t1 = t0 + win
        idx = np.where((t_s >= t0) & (t_s < t1))[0]
        if idx.size < 6:
            t0 += stride; continue

        # dominant label
        vals, cnts = np.unique(df["label_name"].to_numpy()[idx], return_counts=True)
        y = vals[np.argmax(cnts)]

        # dominant segment
        sv, sc = np.unique(seg_arr[idx], return_counts=True)
        dom_seg = int(sv[np.argmax(sc)])

        # warmup skip
        if (t0 - seg_start.get(dom_seg, t0)) < cfg.warmup_s:
            t0 += stride; continue

        logg = df[col_lg].to_numpy(dtype=np.float64)[idx]
        dlog = df[col_dl].to_numpy(dtype=np.float64)[idx]
        ed   = df[col_ed].to_numpy(dtype=np.float64)[idx]
        hum  = df[col_hum].to_numpy(dtype=np.float64)[idx]
        temp = df[col_tmp].to_numpy(dtype=np.float64)[idx]
        tt   = t_s[idx] - t_s[idx[0]]

        row = {"t0_s": float(t0), "t1_s": float(t1), "n": len(idx),
               "group_id": dom_seg, "label_name": y}

        # ── BASELINE-INVARIANT FEATURES ───────────────────────────────────────

        # 1. Total log_gas excursion across the window
        #    (+) = gas went up, (-) = gas went down (alcohol!)
        lg_start = float(np.nanmean(logg[:max(1, len(logg)//5)]))
        lg_end   = float(np.nanmean(logg[-max(1, len(logg)//5):]))
        push(row, "delta_log",        lg_end - lg_start)
        push(row, "delta_log_abs",    abs(lg_end - lg_start))

        # 2. Linear slope of log_gas (sign is the class fingerprint)
        slope_lg, r2_lg = linslope(tt, logg)
        push(row, "slope_logg",       slope_lg)
        push(row, "slope_logg_abs",   abs(slope_lg))
        push(row, "slope_logg_r2",    r2_lg)

        # 3. ema_diff at the END of the window
        #    (the "steady state" signal: negative=alcohol, positive=coffee/garlic, ~0=air)
        k = max(1, len(ed)//4)
        push(row, "emad_end",         float(np.nanmean(ed[-k:])))
        push(row, "emad_start",       float(np.nanmean(ed[:k])))
        push(row, "emad_delta",       float(np.nanmean(ed[-k:])) - float(np.nanmean(ed[:k])))

        # 4. ema_diff statistics (all relative within window, no absolute level)
        ed_finite = ed[np.isfinite(ed)]
        if ed_finite.size:
            push(row, "emad_mean",    float(np.nanmean(ed)))
            push(row, "emad_std",     float(np.nanstd(ed)))
            push(row, "emad_min",     float(np.nanmin(ed)))
            push(row, "emad_max",     float(np.nanmax(ed)))
            push(row, "emad_range",   float(np.nanmax(ed) - np.nanmin(ed)))
            push(row, "emad_sign",    float(np.sign(np.nanmean(ed))))
        else:
            for k_ in ["emad_mean","emad_std","emad_min","emad_max","emad_range","emad_sign"]:
                push(row, k_, 0.0)

        # 5. Slope of ema_diff (is the signal still responding or leveling off?)
        slope_ed, r2_ed = linslope(tt, ed)
        push(row, "slope_emad",       slope_ed)
        push(row, "slope_emad_abs",   abs(slope_ed))
        push(row, "slope_emad_r2",    r2_ed)

        # 6. dlog/dt volatility features (turbulence = noisy, scents = smooth)
        dlog_f = dlog[np.isfinite(dlog)]
        if dlog_f.size:
            push(row, "dlogdt_std",      float(np.nanstd(dlog_f)))
            push(row, "dlogdt_maxabs",   float(np.nanmax(np.abs(dlog_f))))
            push(row, "dlogdt_mean",     float(np.nanmean(dlog_f)))
            push(row, "dlogdt_rms",      float(np.sqrt(np.nanmean(dlog_f**2))))
        else:
            for k_ in ["dlogdt_std","dlogdt_maxabs","dlogdt_mean","dlogdt_rms"]:
                push(row, k_, 0.0)

        # 7. Sign changes in dlog/dt  (turbulence flips sign, scents are monotone)
        push(row, "dlogdt_sign_changes", sign_changes(dlog))

        # 8. Curvature: is the response decelerating (settling onto a new plateau)?
        #    Positive curvature = decelerating rise (coffee, garlic settling)
        #    Negative curvature = decelerating fall (alcohol settling)
        #    High abs curvature = still responding
        if len(tt) >= 6:
            mid = len(tt) // 2
            slope_early, _ = linslope(tt[:mid], logg[:mid])
            slope_late,  _ = linslope(tt[mid:], logg[mid:])
            push(row, "logg_curvature",    slope_late - slope_early)   # neg = decelerating fall
            push(row, "logg_curv_abs",     abs(slope_late - slope_early))
        else:
            push(row, "logg_curvature", 0.0)
            push(row, "logg_curv_abs",  0.0)

        # 9. Humidity delta (alcohol evaporation slightly changes humidity)
        hum_f = hum[np.isfinite(hum)]
        if hum_f.size > 1:
            push(row, "hum_delta", float(hum_f[-1] - hum_f[0]))
            push(row, "hum_std",   float(np.std(hum_f)))
        else:
            push(row, "hum_delta", 0.0)
            push(row, "hum_std",   0.0)

        # 10. Temperature (not baseline-dep, just context)
        push(row, "temp_mean",  float(np.nanmean(temp)))
        push(row, "temp_delta", float(temp[-1] - temp[0]) if len(temp) > 1 else 0.0)

        feats.append(row)
        t0 += stride

    return pd.DataFrame(feats), feature_names


def export_logreg_header(pipe, feature_names, label_names, out_path):
    scaler = pipe.named_steps.get("scaler")
    clf    = pipe.named_steps["model"]
    coef   = clf.coef_.astype(np.float32)
    bias   = clf.intercept_.astype(np.float32)
    mean_  = (scaler.mean_.astype(np.float32)  if scaler else np.zeros(len(feature_names), np.float32))
    std_   = (scaler.scale_.astype(np.float32) if scaler else np.ones(len(feature_names), np.float32))

    def arr(a, name):
        vals = ", ".join(f"{float(v):.8f}f" for v in a.flatten())
        return f"const float {name}[] PROGMEM = {{{vals}}};\n"

    lines = [
        "// Auto-generated by train_bme680_v3.py\n",
        "// BASELINE-INVARIANT features — works across power cycles\n",
        "#pragma once\n",
        "#include <avr/pgmspace.h>\n\n",
        f"#define N_FEATURES  {len(feature_names)}\n",
        f"#define N_CLASSES   {len(label_names)}\n\n",
        "const char* const CLASS_NAMES[] = {" + ", ".join(f'"{l}"' for l in label_names) + "};\n\n",
        "// Feature names (for debugging)\n",
        "const char* const FEATURE_NAMES[] = {" + ", ".join(f'"{f}"' for f in feature_names) + "};\n\n",
        arr(mean_, "SCALER_MEAN"),
        arr(std_,  "SCALER_STD"),
        "\n",
        arr(coef.flatten(), "LR_COEF"),
        "\n",
        arr(bias, "LR_BIAS"),
    ]
    Path(out_path).write_text("".join(lines))
    print(f"Saved: {out_path}  ({len(feature_names)} features, {len(label_names)} classes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       required=True)
    ap.add_argument("--window_s",  type=float, default=5.0)
    ap.add_argument("--stride_s",  type=float, default=0.5)
    ap.add_argument("--warmup_s",  type=float, default=1.5)
    ap.add_argument("--out_dir",   default="bme680_v3_out")
    ap.add_argument("--seed",      type=int,   default=7)
    ap.add_argument("--keep_air",  action="store_true")
    ap.add_argument("--turb_boost",type=float, default=2.0,
                    help="Extra weight on turbulence class (1.0 = no boost)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, comment="#")
    label_map = {
        "c":"coffee","g":"garlic","a":"alcohol","l":"alcohol","t":"turbulence","u":"air",
        "coffee":"coffee","garlic":"garlic","alcohol":"alcohol","turbulence":"turbulence","air":"air",
    }
    drop_labels = [] if args.keep_air else ["air"]

    cfg = Config(window_s=args.window_s, stride_s=args.stride_s, warmup_s=args.warmup_s,
                 label_map=label_map, drop_labels=drop_labels)

    feat_df, feature_names = build_windows(df, cfg)
    if feat_df.empty:
        raise SystemExit("No windows generated.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Window counts per class:")
    print(feat_df["label_name"].value_counts().to_string())
    print(f"Total: {len(feat_df)} windows  |  {len(feature_names)} features\n")
    print("Features:", feature_names)
    print()

    feat_df.to_csv(out_dir / "windows_features.csv", index=False)

    X      = feat_df[feature_names].to_numpy(dtype=np.float32)
    y      = feat_df["label_name"].astype(str).to_numpy()
    groups = feat_df["group_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))
    X_tr, y_tr, g_tr = X[tr_idx], y[tr_idx], groups[tr_idx]
    X_te, y_te       = X[te_idx], y[te_idx]

    # Class weights with turbulence boost
    classes = np.unique(y_tr)
    base_w  = compute_class_weight("balanced", classes=classes, y=y_tr)
    w_dict  = dict(zip(classes, base_w))
    if "turbulence" in w_dict:
        w_dict["turbulence"] *= args.turb_boost
    print(f"Class weights (turb_boost={args.turb_boost}x): {w_dict}\n")

    cv = GroupKFold(n_splits=5)
    seed = args.seed

    candidates = [
        ("LogReg",
         Pipeline([("scaler", StandardScaler()),
                   ("model", LogisticRegression(max_iter=5000, class_weight=w_dict,
                                                solver="lbfgs"))]),
         {"model__C": [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]}),

        ("ExtraTrees",
         Pipeline([("model", ExtraTreesClassifier(n_estimators=800, random_state=seed,
                                                   n_jobs=-1, class_weight="balanced_subsample"))]),
         {"model__max_depth":        [None, 6, 10],
          "model__min_samples_leaf": [1, 2, 4],
          "model__max_features":     ["sqrt", 0.6]}),

        ("RandomForest",
         Pipeline([("model", RandomForestClassifier(n_estimators=800, random_state=seed,
                                                     n_jobs=-1, class_weight="balanced_subsample"))]),
         {"model__max_depth":        [None, 6, 10],
          "model__min_samples_leaf": [1, 2, 4],
          "model__max_features":     ["sqrt", 0.6]}),
    ]

    trained = {}
    leaderboard = []
    best = None

    for name, pipe, grid in candidates:
        search = RandomizedSearchCV(
            pipe, param_distributions=grid,
            n_iter=min(20, sum(len(v) for v in grid.values())),
            scoring="f1_macro",
            cv=cv.split(X_tr, y_tr, g_tr),
            random_state=seed, n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        est   = search.best_estimator_
        pred  = est.predict(X_te)
        score = f1_score(y_te, pred, average="macro")
        leaderboard.append((name, float(score)))
        trained[name] = est
        print(f"{name:12s} F1={score:.4f}  {search.best_params_}")
        if best is None or score > best[0]:
            best = (score, name, est, search.best_params_)

    # Soft-vote ensemble
    # VotingClassifier encodes labels as integers, so a string-keyed
    # class_weight dict breaks. Swap LogReg to "balanced" inside ensemble.
    import copy
    vote_est = []
    for n, e in trained.items():
        if not hasattr(e.named_steps["model"], "predict_proba"):
            continue
        e2 = copy.deepcopy(e)
        if isinstance(e2.named_steps["model"].class_weight, dict):
            e2.named_steps["model"].class_weight = "balanced"
        vote_est.append((n, e2))
    if len(vote_est) >= 2:
        from sklearn.ensemble import VotingClassifier
        vc = VotingClassifier(vote_est, voting="soft", n_jobs=-1)
        vc.fit(X_tr, y_tr)
        vp = vc.predict(X_te)
        vs = f1_score(y_te, vp, average="macro")
        leaderboard.append(("Ensemble", vs))
        print(f"{'Ensemble':12s} F1={vs:.4f}")
        if vs > best[0]:
            best = (vs, "Ensemble", vc, {})
            trained["Ensemble"] = vc

    leaderboard.sort(key=lambda x: x[1], reverse=True)
    print("\n== Leaderboard ==")
    for i, (n, s) in enumerate(leaderboard, 1):
        print(f"  {i}. {n:14s} F1={s:.4f}")

    best_score, best_name, best_est, best_params = best
    labels_sorted = sorted(np.unique(y))
    pred = best_est.predict(X_te)
    cm   = confusion_matrix(y_te, pred, labels=labels_sorted)
    rep  = classification_report(y_te, pred, digits=3)

    print(f"\n== Best: {best_name}  F1={best_score:.4f} ==")
    print(f"Labels: {labels_sorted}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(rep)

    joblib.dump(best_est, out_dir / "best_model.joblib")

    cfg_out = {
        "version": "v3_baseline_invariant",
        "window_s": args.window_s, "stride_s": args.stride_s, "warmup_s": args.warmup_s,
        "feature_names": feature_names, "label_map": label_map,
        "drop_labels": drop_labels, "best_model_name": best_name,
        "best_params": best_params, "labels": labels_sorted,
        "turb_boost": args.turb_boost,
    }
    (out_dir / "feature_config.json").write_text(json.dumps(cfg_out, indent=2))

    # Export Arduino header (LogReg only — smallest and fastest on M4)
    logreg = trained.get("LogReg")
    if logreg:
        export_logreg_header(logreg, feature_names, labels_sorted,
                             out_dir / "model_weights.h")
        print("\n[Deploy] Copy bme680_v3_out/model_weights.h to your sketch folder.")
        print(f"         N_FEATURES={len(feature_names)}  N_CLASSES={len(labels_sorted)}")
    else:
        print("[Deploy] No LogReg model trained — cannot export header.")


if __name__ == "__main__":
    main()
