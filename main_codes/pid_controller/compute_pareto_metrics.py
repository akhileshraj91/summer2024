#!/usr/bin/env python3
"""
Compute Sparsity and Hypervolume of the PID sweep Pareto front for each application.

Methodology mirrors the DDMORL notebook (cell 28):
  - Median (exec_time, energy) per setpoint from PID_Control_sweep tars
  - Filter to Pareto-optimal subset (minimising both objectives)
  - Sparsity  : Hayes et al. ICLR 2023 Definition 2  (unit: s²+kJ²)
  - Hypervolume: 2D sweep-line, reference = worst + 10% range (unit: s·kJ)

Usage:
  python3 compute_pareto_metrics.py
  python3 compute_pareto_metrics.py --sweep-dir /path/to/PID_Control_sweep
"""

import argparse
import glob
import io
import os
import tarfile
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument('--sweep-dir', default=None)
_args = _parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_ROOT = os.path.normpath(
    _args.sweep_dir or
    os.path.join(SCRIPT_DIR, '..', 'experiment_data', 'PID_Control_sweep')
)

APP_LABELS = {
    'ones-npb-bt':      'NPB-BT',
    'ones-npb-cg':      'NPB-CG',
    'ones-npb-ep':      'NPB-EP',
    'ones-npb-ft':      'NPB-FT',
    'ones-npb-is':      'NPB-IS',
    'ones-npb-mg':      'NPB-MG',
    'ones-stream-add':  'Stream-Add',
    'ones-stream-copy': 'Stream-Copy',
    'ones-stream-full': 'Stream-Full',
    'ones-stream-scale':'Stream-Scale',
    'ones-stream-triad':'Stream-Triad',
}

# ── Data loading ───────────────────────────────────────────────────────────────

def _read_tar_csvs(tar_path):
    out = {}
    try:
        with tarfile.open(tar_path, 'r:*') as tf:
            for name in tf.getnames():
                if name.endswith('.csv'):
                    fobj = tf.extractfile(name)
                    if fobj:
                        out[os.path.basename(name)] = pd.read_csv(
                            io.BytesIO(fobj.read()))
    except Exception:
        pass
    return out


def _exec_time(csvs):
    prog = csvs.get('progress.csv')
    if prog is None or len(prog) < 2:
        return None
    return float(prog['time'].iloc[-1] - prog['time'].iloc[0])


def _energy_kj(csvs):
    pw = csvs.get('measured_power.csv')
    if pw is None or len(pw) < 2:
        return None
    pw  = pw.sort_values('time').copy()
    avg = pw.groupby('time')['value'].mean().reset_index()
    dt  = avg['time'].diff().fillna(0).values
    return float(np.sum(avg['value'].values * dt)) / 1000.0


def load_sweep_runs(app_dir):
    """Return list of (exec_time_s, energy_kj, target_Hz) for all sweep tars."""
    runs = []
    for tp in sorted(glob.glob(os.path.join(app_dir, '*.tar'))):
        csvs = _read_tar_csvs(tp)
        et   = _exec_time(csvs)
        en   = _energy_kj(csvs)
        pl   = csvs.get('pid_log.csv')
        if et and en and pl is not None and len(pl):
            runs.append((et, en, float(pl['target_perf_Hz'].iloc[0])))
    return runs


# ── Pareto metrics (matching notebook cell 28) ────────────────────────────────

def _pareto_dominated_local(xi, yi, xs_all, ys_all):
    for xj, yj in zip(xs_all, ys_all):
        if xj == xi and yj == yi:
            continue
        if xj <= xi and yj <= yi and (xj < xi or yj < yi):
            return True
    return False


def pareto_front(points):
    """Non-dominated subset of (x, y) points, both minimised."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [(x, y) for x, y in points
            if not _pareto_dominated_local(x, y, xs, ys)]


def hypervolume_2d(points, ref):
    """2D sweep-line HV (minimisation). Unit = x_unit × y_unit."""
    rx, ry = ref
    pts = sorted([(x, y) for x, y in points if x <= rx and y <= ry],
                 key=lambda p: p[0])
    hv = 0.0
    for i, (xi, yi) in enumerate(pts):
        x_next = pts[i + 1][0] if i + 1 < len(pts) else rx
        hv += (x_next - xi) * (ry - yi)
    return hv


def sparsity(points):
    """Hayes et al. (ICLR 2023) Definition 2:
    Sp(P) = 1/(N-1) · Σ_j Σ_i (P_ij - P_{i+1,j})²
    Objectives sorted independently; raw units (s² + kJ²).
    """
    pts = np.array(points)   # shape (N, L)
    N, L = pts.shape
    if N < 2:
        return float('nan')
    total = 0.0
    for j in range(L):
        sorted_j = np.sort(pts[:, j])
        total += float(np.sum(np.diff(sorted_j) ** 2))
    return total / (N - 1)


# ── Main ───────────────────────────────────────────────────────────────────────

print("# ----------------------------------------------------------")
print("# SPARSITY & HYPERVOLUME OF PID SWEEP PARETO FRONT")
print("# Computed in Execution Time [s] vs Energy [kJ] space.")
print("# Source: PID_Control_sweep ONLY (teal/green dots).")
print("# Static PCAP background dots are NOT included.")
print("# Median (exec_time, energy) per setpoint; Pareto-filtered.")
print("# HV unit: s·kJ   |   Sparsity unit: s²+kJ²")
print("# ----------------------------------------------------------")
print()

sweep_apps = sorted(
    os.path.basename(d.rstrip('/'))
    for d in glob.glob(os.path.join(SWEEP_ROOT, '*/'))
)

if not sweep_apps:
    print(f"No sweep data found in {SWEEP_ROOT}")
    raise SystemExit(1)

records = []

for app in sweep_apps:
    app_dir = os.path.join(SWEEP_ROOT, app)
    runs    = load_sweep_runs(app_dir)
    if not runs:
        continue

    # Median (exec_time, energy) per setpoint
    by_target = defaultdict(list)
    for et, en, tgt in runs:
        by_target[round(tgt, 4)].append((et, en))

    median_pts = []
    for tgt in sorted(by_target.keys()):
        pts    = by_target[tgt]
        et_med = float(np.median([p[0] for p in pts]))
        en_med = float(np.median([p[1] for p in pts]))
        median_pts.append((et_med, en_med))

    # Pareto front (non-dominated)
    pf   = pareto_front(median_pts)
    pf_t = np.array([p[0] for p in pf])
    pf_e = np.array([p[1] for p in pf])

    if len(pf) < 2:
        records.append({
            'App'                : APP_LABELS.get(app, app),
            'N setpoints'        : len(median_pts),
            'N Pareto pts'       : len(pf),
            'Sparsity (s²+kJ²)'  : float('nan'),
            'Hypervolume (s·kJ)' : float('nan'),
        })
        continue

    # Reference: 10% beyond worst point in each axis (matches notebook)
    tr  = float(pf_t.max() - pf_t.min()) or 1.0
    er  = float(pf_e.max() - pf_e.min()) or 1.0
    ref = (float(pf_t.max() + 0.1 * tr),
           float(pf_e.max() + 0.1 * er))

    sp = sparsity(list(zip(pf_t, pf_e)))
    hv = hypervolume_2d(list(zip(pf_t, pf_e)), ref)

    records.append({
        'App'                : APP_LABELS.get(app, app),
        'N setpoints'        : len(median_pts),
        'N Pareto pts'       : len(pf),
        'Sparsity (s²+kJ²)'  : round(sp, 4),
        'Hypervolume (s·kJ)' : round(hv, 3),
    })

df_metrics = pd.DataFrame(records).set_index('App')
print(df_metrics.to_string())
print()
print(df_metrics)
