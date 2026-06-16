#!/usr/bin/env python3
"""
relabel_turbulence.py
─────────────────────
Removes the 'turbulence' label by splitting each turbulence segment
at the dominant slope-change point and reassigning samples to the
adjacent scent labels.

Split logic (per your instruction):
  1. Find the slope-reversal point: first index where |dlog_dt| crosses a
     significance threshold and the sign has clearly changed direction
     (relative to the leading quiet region).
  2. If a clear reversal exists → split there.
     Samples before split → previous label (fading scent).
     Samples from split onward → next label (arriving scent).
  3. If no clear reversal (flat noisy transition with no dramatic slope) →
     split in half.

The result is a new CSV with identical structure but 't' labels replaced.

Usage:
  python relabel_turbulence.py --csv wAlcohol_bme680_log.csv
  → writes  wAlcohol_bme680_log_relabeled.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Tuning knobs ──────────────────────────────────────────────────────────────
# A sample is "dramatically sloping" when |dlog_dt| exceeds this value.
# From your data, quiet plateau noise is ±0.05, dramatic slopes are >0.2
SLOPE_THRESHOLD = 0.15

# Minimum number of consecutive dramatic samples to count as "real" slope onset
MIN_CONSECUTIVE = 2
# ─────────────────────────────────────────────────────────────────────────────


def find_split_index(dlog: np.ndarray) -> int:
    """
    Find the index within a turbulence segment where the sensor starts
    responding to the arriving scent.

    Returns the first index of the dramatic slope region (all samples
    from this index onward → next label). Returns -1 if no clear split
    found (use midpoint instead).
    """
    n = len(dlog)

    # Smooth out single-sample noise spikes with a tiny median filter
    smoothed = dlog.copy().astype(float)
    for i in range(1, n - 1):
        smoothed[i] = np.median(dlog[max(0, i-1):min(n, i+2)])

    # Find first run of MIN_CONSECUTIVE samples all above SLOPE_THRESHOLD
    # in absolute value — that's where the sensor starts responding
    for i in range(n - MIN_CONSECUTIVE + 1):
        window = smoothed[i:i + MIN_CONSECUTIVE]
        if np.all(np.abs(window) >= SLOPE_THRESHOLD):
            # Back up to the actual onset (first sample > half threshold)
            onset = i
            for j in range(i - 1, -1, -1):
                if abs(smoothed[j]) >= SLOPE_THRESHOLD * 0.5:
                    onset = j
                else:
                    break
            return onset

    return -1  # no dramatic slope found → use midpoint


def relabel_segment(df: pd.DataFrame, seg_indices: list[int],
                    prev_label: str, next_label: str) -> pd.DataFrame:
    """
    Relabel one turbulence segment. Modifies df in-place.
    Returns df for chaining.
    """
    n = len(seg_indices)
    dlog = df.loc[seg_indices, 'dloggas_dt'].to_numpy(dtype=float)

    split_at = find_split_index(dlog)

    if split_at == -1 or split_at == 0 or split_at >= n:
        # No clear reversal — split in half
        split_at = n // 2
        method = 'midpoint'
    else:
        method = f'slope_onset@[{split_at}]'

    before = seg_indices[:split_at]
    after  = seg_indices[split_at:]

    if before:
        df.loc[before, 'label'] = prev_label
    if after:
        df.loc[after, 'label'] = next_label

    t_start = df.loc[seg_indices[0], 't_s'] if 't_s' in df.columns else '?'
    print(f"  Seg t={t_start:.1f}s  "
          f"prev={prev_label} next={next_label}  "
          f"n={n}  split={split_at}/{n}  method={method}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default=None,
                    help='Output CSV path (default: <input>_relabeled.csv)')
    ap.add_argument('--threshold', type=float, default=0.15,
                    help='Slope significance threshold (default 0.15)')
    ap.add_argument('--min_consec', type=int, default=2,
                    help='Min consecutive dramatic samples (default 2)')
    args = ap.parse_args()

    SLOPE_THRESHOLD = args.threshold
    MIN_CONSECUTIVE = args.min_consec

    in_path = Path(args.csv)
    df = pd.read_csv(in_path, comment='#')
    df['t_s'] = (df['ms'] - df['ms'].iloc[0]) / 1000.0

    # ── Identify contiguous segments ─────────────────────────────────────────
    segments = []
    cur_label = df['label'].iloc[0]
    cur_start = 0
    for i in range(1, len(df)):
        if df['label'].iloc[i] != cur_label:
            segments.append({
                'label':   cur_label,
                'indices': list(range(cur_start, i)),
            })
            cur_label = df['label'].iloc[i]
            cur_start = i
    segments.append({'label': cur_label, 'indices': list(range(cur_start, len(df)))})

    print(f"Total segments: {len(segments)}")
    print(f"Turbulence segments: {sum(1 for s in segments if s['label'] == 't')}")
    print()

    # ── Process each turbulence segment ──────────────────────────────────────
    label_map = {
        'u': 'air', 'c': 'coffee', 'l': 'alcohol',
        'g': 'garlic', 't': 'turbulence', 'a': 'air',
    }

    for i, seg in enumerate(segments):
        if seg['label'] != 't':
            continue

        prev_raw = segments[i-1]['label'] if i > 0        else 'u'
        next_raw = segments[i+1]['label'] if i < len(segments)-1 else 'u'

        # Skip if neighbor is also turbulence or air (rare, just keep as-is → relabel to air)
        if prev_raw == 't': prev_raw = 'u'
        if next_raw == 't': next_raw = 'u'

        # Use full label names for clarity
        prev_label = label_map.get(prev_raw, prev_raw)
        next_label = label_map.get(next_raw, next_raw)

        relabel_segment(df, seg['indices'], prev_label, next_label)

    # ── Map all remaining single-char labels to full names ───────────────────
    df['label'] = df['label'].map(label_map).fillna(df['label'])

    # Drop helper column
    df = df.drop(columns=['t_s'])

    # ── Print final distribution ──────────────────────────────────────────────
    print()
    print("Label distribution after relabeling:")
    print(df['label'].value_counts().to_string())
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = args.out or str(in_path.with_name(in_path.stem + '_relabeled.csv'))
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print()
    print("Next step:")
    print(f"  python train_bme680_v3.py --csv {out_path}")


if __name__ == '__main__':
    main()
