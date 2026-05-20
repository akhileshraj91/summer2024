#!/usr/bin/env python3
"""
Plot PID controller experiment results.

Reads compressed iteration tars from experiment_data/PID_Control/<app>/
and generates:
  pid_results_per_app.pdf              — one page per app (performance | PCAP | power)
  pid_results_summary.pdf             — cross-app comparison (MAE %, mean PCAP, mean power)
  Pareto_Energy_vs_time_<app>.pdf     — Energy vs Execution Time Pareto scatter (one per app)

The Pareto plots show static-PCAP experiments (YlOrRd background) plus PID sweep
operating points (teal dots) annotated with the normalized preference weight
(w_efficiency, w_performance) = (1-target_frac, target_frac).

Run from the pid_controller/ directory:
  python plot_pid_results.py
  python plot_pid_results.py --pareto
  python plot_pid_results.py --pareto --sweep-dir /path/to/PID_Control_sweep
  python plot_pid_results.py --data-dir /path/to/PID_Control --out-dir ./figures
"""

import argparse
import glob
import io
import os
import tarfile
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Argument parsing ──────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description='Plot PID experiment results')
_parser.add_argument('--data-dir', default=None,
                     help='Path to experiment_data/PID_Control (auto-detected if omitted)')
_parser.add_argument('--out-dir', default=None,
                     help='Output directory for PDFs (default: script directory)')
_parser.add_argument('--pareto', action='store_true',
                     help='Also generate Pareto Energy-vs-Time scatter plots from sweep data')
_parser.add_argument('--sweep-dir', default=None,
                     help='Path to experiment_data/PID_Control_sweep (auto-detected if omitted)')
_parser.add_argument('--static-dir', default=None,
                     help='Path to experiment_data/plotting_data for static PCAP background')
args = _parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.normpath(
    args.data_dir or os.path.join(SCRIPT_DIR, '..', 'experiment_data', 'PID_Control')
)
SWEEP_ROOT = os.path.normpath(
    args.sweep_dir or os.path.join(SCRIPT_DIR, '..', 'experiment_data', 'PID_Control_sweep')
)
STATIC_ROOT = os.path.normpath(
    args.static_dir or os.path.join(SCRIPT_DIR, '..', 'experiment_data', 'plotting_data')
)
OUT_DIR = args.out_dir or SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'figure.dpi': 120,
})

C_PERF  = '#1976D2'   # blue   — performance
C_PCAP  = '#E65100'   # orange — PCAP
C_POWER = '#2E7D32'   # green  — node power
C_TARGET = '#D32F2F'  # red    — setpoint / target

# ── Data loading ──────────────────────────────────────────────────────────────

def _read_pid_log_from_tar(tar_path):
    """Extract and parse pid_log.csv from a single tar file. Returns DataFrame or None."""
    try:
        with tarfile.open(tar_path, 'r:*') as tar:
            if 'pid_log.csv' not in tar.getnames():
                return None
            f = tar.extractfile('pid_log.csv')
            if f is None:
                return None
            df = pd.read_csv(io.BytesIO(f.read()))
    except Exception as exc:
        print(f"  [WARN] {os.path.basename(tar_path)}: {exc}")
        return None
    if len(df) < 3:
        return None
    df['t_rel'] = df['time'] - df['time'].iloc[0]
    df['measured_power_W'] = pd.to_numeric(df['measured_power_W'], errors='coerce')
    return df


def _read_pid_log_loose(app_dir):
    """Read a loose (uncompressed) pid_log.csv directly in app_dir, if present."""
    path = os.path.join(app_dir, 'pid_log.csv')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if len(df) < 3:
        return None
    df['t_rel'] = df['time'] - df['time'].iloc[0]
    df['measured_power_W'] = pd.to_numeric(df['measured_power_W'], errors='coerce')
    return df


def load_app_iterations(app_dir):
    """Return a list of DataFrames (one per completed iteration) for an app directory."""
    dfs = []
    for tf in sorted(glob.glob(os.path.join(app_dir, '*.tar'))):
        df = _read_pid_log_from_tar(tf)
        if df is not None:
            dfs.append(df)
    # Also pick up any in-progress (uncompressed) run
    loose = _read_pid_log_loose(app_dir)
    if loose is not None:
        dfs.append(loose)
    return dfs


def load_all_apps(data_root):
    apps = {}
    for d in sorted(glob.glob(os.path.join(data_root, '*/'))):
        app = os.path.basename(d.rstrip('/'))
        dfs = load_app_iterations(d)
        if dfs:
            apps[app] = dfs
            print(f"  {app:<30} {len(dfs):>3} iterations")
        else:
            print(f"  {app:<30}   0 iterations (no data)")
    return apps


# ── Resampling / aggregation ──────────────────────────────────────────────────

def _resample(df, t_grid, cols):
    """Interpolate selected columns from df onto t_grid."""
    out = {}
    for col in cols:
        if col not in df.columns:
            out[col] = np.full(len(t_grid), np.nan)
            continue
        vals = df[col].ffill().bfill().values.astype(float)
        out[col] = np.interp(t_grid, df['t_rel'].values, vals,
                             left=vals[0], right=vals[-1])
    return out


def aggregate(dfs, n_points=300):
    """
    Resample all iterations onto a common time grid and compute
    median + IQR for each key signal.
    Returns (t_grid, stats_dict) where stats_dict[col] = (median, p25, p75).
    """
    t_max = min(df['t_rel'].max() for df in dfs)
    t_grid = np.linspace(0, t_max, n_points)
    cols = ['measured_perf_Hz', 'pcap_cmd_W', 'measured_power_W', 'error_Hz']
    matrix = {c: [] for c in cols}
    for df in dfs:
        r = _resample(df, t_grid, cols)
        for c in cols:
            matrix[c].append(r[c])
    stats = {}
    for c in cols:
        arr = np.array(matrix[c])           # shape (n_iters, n_points)
        stats[c] = (
            np.nanmedian(arr, axis=0),
            np.nanpercentile(arr, 25, axis=0),
            np.nanpercentile(arr, 75, axis=0),
        )
    return t_grid, stats


# ── Per-app page ──────────────────────────────────────────────────────────────

def _shade(ax, t, med, p25, p75, color, alpha=0.18):
    ax.fill_between(t, p25, p75, color=color, alpha=alpha)
    return ax.plot(t, med, color=color, lw=1.8)[0]


def plot_app_page(fig, app, dfs):
    target = float(dfs[0]['target_perf_Hz'].iloc[0])
    u_ff   = float(dfs[0]['u_ff_W'].iloc[0])
    n      = len(dfs)

    t_grid, stats = aggregate(dfs)

    gs = fig.add_gridspec(3, 1, hspace=0.50, top=0.92, bottom=0.07)
    ax_perf  = fig.add_subplot(gs[0])
    ax_pcap  = fig.add_subplot(gs[1], sharex=ax_perf)
    ax_power = fig.add_subplot(gs[2], sharex=ax_perf)

    fig.suptitle(f'{app}  ({n} iteration{"s" if n != 1 else ""})',
                 fontsize=11, fontweight='bold')

    # ── Panel 1: Performance ─────────────────────────────────────────────────
    for df in dfs:
        ax_perf.plot(df['t_rel'], df['measured_perf_Hz'],
                     color=C_PERF, alpha=0.15, lw=0.7)
    med, p25, p75 = stats['measured_perf_Hz']
    ln_med = _shade(ax_perf, t_grid, med, p25, p75, C_PERF)
    ln_tgt = ax_perf.axhline(target, color=C_TARGET, ls='--', lw=1.5)
    ax_perf.set_ylabel('Progress Rate [Hz]')
    ax_perf.legend([ln_med, ln_tgt],
                   [f'measured (median ± IQR)', f'target = {target:.4f} Hz'],
                   loc='upper right')
    ax_perf.grid(True, alpha=0.25)

    # ── Panel 2: PCAP ────────────────────────────────────────────────────────
    for df in dfs:
        ax_pcap.step(df['t_rel'], df['pcap_cmd_W'],
                     color=C_PCAP, alpha=0.15, lw=0.7, where='post')
    med_p, p25_p, p75_p = stats['pcap_cmd_W']
    ax_pcap.fill_between(t_grid, p25_p, p75_p,
                         step='post', color=C_PCAP, alpha=0.18)
    ln_pcap = ax_pcap.step(t_grid, med_p, color=C_PCAP, lw=1.8,
                           where='post', label='PCAP cmd (median ± IQR)')[0]
    ln_uff  = ax_pcap.axhline(u_ff, color='black', ls=':', lw=1.2,
                              label=f'feedforward u_ff = {u_ff:.1f} W')
    ax_pcap.set_ylabel('Power Cap [W]')
    ax_pcap.set_ylim(70, 175)
    ax_pcap.set_yticks([78, 95, 112, 124, 141, 159, 165])
    ax_pcap.legend([ln_pcap, ln_uff],
                   [f'PCAP cmd (median ± IQR)', f'u_ff = {u_ff:.1f} W'],
                   loc='upper right')
    ax_pcap.grid(True, alpha=0.25)

    # ── Panel 3: Node power ──────────────────────────────────────────────────
    med_w, p25_w, p75_w = stats['measured_power_W']
    has_power = not np.all(np.isnan(med_w))
    if has_power:
        for df in dfs:
            valid = df['measured_power_W'].notna()
            ax_power.plot(df.loc[valid, 't_rel'], df.loc[valid, 'measured_power_W'],
                          color=C_POWER, alpha=0.15, lw=0.7)
        ln_pw = _shade(ax_power, t_grid, med_w, p25_w, p75_w, C_POWER)
        ax_power.legend([ln_pw], ['node power (median ± IQR)'], loc='upper right')
    else:
        ax_power.text(0.5, 0.5, 'no power data', transform=ax_power.transAxes,
                      ha='center', va='center', color='gray')
    ax_power.set_ylabel('Node Power [W]')
    ax_power.set_xlabel('Time [s]')
    ax_power.grid(True, alpha=0.25)

    for ax in [ax_perf, ax_pcap, ax_power]:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        plt.setp(ax.get_xticklabels(), visible=(ax is ax_power))


# ── Summary figure ────────────────────────────────────────────────────────────

def compute_summary(apps_data):
    rows = []
    for app, dfs in apps_data.items():
        target = float(dfs[0]['target_perf_Hz'].iloc[0])
        u_ff   = float(dfs[0]['u_ff_W'].iloc[0])
        perf   = pd.concat([df['measured_perf_Hz'] for df in dfs], ignore_index=True)
        pcap   = pd.concat([df['pcap_cmd_W']        for df in dfs], ignore_index=True)
        err    = pd.concat([df['error_Hz']           for df in dfs], ignore_index=True)
        power_parts = [df['measured_power_W'].dropna() for df in dfs]
        power  = pd.concat(power_parts, ignore_index=True) if power_parts else pd.Series(dtype=float)
        rows.append({
            'app':          app,
            'n_iters':      len(dfs),
            'target_Hz':    target,
            'u_ff_W':       u_ff,
            'mean_perf_Hz': perf.mean(),
            'std_perf_Hz':  perf.std(),
            'mae_Hz':       err.abs().mean(),
            'mae_pct':      err.abs().mean() / target * 100 if target != 0 else np.nan,
            'mean_pcap_W':  pcap.mean(),
            'mean_power_W': power.mean() if len(power) else np.nan,
        })
    df = pd.DataFrame(rows).set_index('app')
    df.sort_values('mae_pct', inplace=True)
    return df


def plot_summary(fig, summary):
    apps = summary.index.tolist()
    y = np.arange(len(apps))
    h = 0.6

    gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.22, right=0.97,
                          top=0.88, bottom=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── MAE % ────────────────────────────────────────────────────────────────
    bars = ax1.barh(y, summary['mae_pct'], height=h, color=C_TARGET, alpha=0.80)
    ax1.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=7.5)
    ax1.set_yticks(y); ax1.set_yticklabels(apps, fontsize=8)
    ax1.set_xlabel('MAE  [% of target]')
    ax1.set_title('Performance Error')
    ax1.grid(axis='x', alpha=0.25)
    ax1.set_xlim(0, summary['mae_pct'].max() * 1.25)

    # ── Mean PCAP ────────────────────────────────────────────────────────────
    bars2 = ax2.barh(y, summary['mean_pcap_W'], height=h, color=C_PCAP, alpha=0.80)
    ax2.bar_label(bars2, fmt='%.0fW', padding=3, fontsize=7.5)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel('Mean PCAP  [W]')
    ax2.set_title('Avg PCAP Commanded')
    ax2.axvline(78,  color='gray', ls=':', lw=1.0, label='min 78 W')
    ax2.axvline(165, color='gray', ls=':', lw=1.0, label='max 165 W')
    ax2.legend(fontsize=7)
    ax2.grid(axis='x', alpha=0.25)
    ax2.set_xlim(0, 185)

    # ── Mean power ───────────────────────────────────────────────────────────
    valid = summary['mean_power_W'].notna()
    bars3 = ax3.barh(y[valid.values], summary.loc[valid, 'mean_power_W'],
                     height=h, color=C_POWER, alpha=0.80)
    ax3.bar_label(bars3, fmt='%.0fW', padding=3, fontsize=7.5)
    ax3.set_yticks(y); ax3.set_yticklabels([])
    ax3.set_xlabel('Mean Node Power  [W]')
    ax3.set_title('Avg Measured Power')
    ax3.grid(axis='x', alpha=0.25)
    ax3.set_xlim(0, 185)

    fig.suptitle('PID Controller — Cross-App Summary', fontsize=12, fontweight='bold')


# ── Console summary table ─────────────────────────────────────────────────────

def print_summary(summary):
    cols = ['n_iters', 'target_Hz', 'mean_perf_Hz', 'mae_Hz', 'mae_pct',
            'mean_pcap_W', 'mean_power_W']
    fmt = {
        'n_iters':      '{:>7.0f}',
        'target_Hz':    '{:>10.4f}',
        'mean_perf_Hz': '{:>13.4f}',
        'mae_Hz':       '{:>10.4f}',
        'mae_pct':      '{:>9.2f}',
        'mean_pcap_W':  '{:>12.1f}',
        'mean_power_W': '{:>13.1f}',
    }
    header = (f"{'App':<28} {'Iters':>7} {'Target(Hz)':>10} {'MeasPerf(Hz)':>13} "
              f"{'MAE(Hz)':>10} {'MAE(%)':>9} {'PCAP(W)':>12} {'Power(W)':>13}")
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for app, row in summary.iterrows():
        line = f"{app:<28}"
        for c in cols:
            v = row[c]
            if pd.isna(v):
                line += f"{'—':>13}"
            else:
                line += fmt[c].format(v)
        print(line)
    print("─" * len(header))


# ══════════════════════════════════════════════════════════════════════════════
# Pareto Energy-vs-Time scatter plots (PID sweep + static PCAP background)
# ══════════════════════════════════════════════════════════════════════════════

ACTIONS     = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0,
               124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
CMAP_NAME   = 'YlOrRd'
PCAP_NORM   = Normalize(vmin=min(ACTIONS), vmax=max(ACTIONS))
STATIC_CMAP = plt.cm.get_cmap(CMAP_NAME)
PID_COLOR   = '#00796B'   # teal

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

plt.rcParams.update({
    'font.size':        14,
    'font.weight':      'bold',
    'axes.labelsize':   'large',
    'axes.labelweight': 'bold',
    'axes.titlesize':   16,
    'axes.titleweight': 'bold',
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'figure.dpi':       150,
})


def _read_tar_csvs(tar_path):
    """Return {basename: DataFrame} for every CSV inside a tar archive."""
    out = {}
    try:
        with tarfile.open(tar_path, 'r:*') as tf:
            for name in tf.getnames():
                if name.endswith('.csv'):
                    fobj = tf.extractfile(name)
                    if fobj:
                        out[os.path.basename(name)] = pd.read_csv(
                            io.BytesIO(fobj.read()))
    except Exception as exc:
        print(f"  [WARN] {os.path.basename(tar_path)}: {exc}")
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
    pw = pw.sort_values('time').copy()
    avg = pw.groupby('time')['value'].mean().reset_index()
    dt  = avg['time'].diff().fillna(0).values
    return float(np.sum(avg['value'].values * dt)) / 1000.0


def load_static_runs(app_dir):
    """Return list of (exec_time_s, energy_kj, pcap_W) for all static tars."""
    runs = []
    for tp in sorted(glob.glob(os.path.join(app_dir, '*.tar'))):
        csvs = _read_tar_csvs(tp)
        et   = _exec_time(csvs)
        en   = _energy_kj(csvs)
        pc   = csvs.get('PCAP_file.csv')
        if et and en and pc is not None and len(pc):
            runs.append((et, en, float(pc['value'].mode().iloc[0])))
    return runs


def load_sweep_runs(app_dir):
    """
    Return list of (exec_time_s, energy_kj, target_Hz) for all sweep tars.
    """
    runs = []
    for tp in sorted(glob.glob(os.path.join(app_dir, '*.tar'))):
        csvs = _read_tar_csvs(tp)
        et   = _exec_time(csvs)
        en   = _energy_kj(csvs)
        pl   = csvs.get('pid_log.csv')
        if et and en and pl is not None and len(pl):
            runs.append((et, en, float(pl['target_perf_Hz'].iloc[0])))
    return runs


def plot_pareto_app(app, static_runs, sweep_runs, out_path):
    """
    Energy vs Execution Time Pareto scatter for one application.

    Background  : static-PCAP dots coloured by PCAP value (YlOrRd)
    Foreground  : PID-sweep median dots (teal, one per setpoint)
                  annotated with the target performance value in Hz
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # ── Static PCAP background (one averaged dot per PCAP level) ─────────────
    if static_runs:
        by_pcap = defaultdict(list)
        for et, en, pc in static_runs:
            nearest = min(ACTIONS, key=lambda a: abs(a - pc))
            by_pcap[nearest].append((et, en))
        for pcap_val in sorted(by_pcap.keys()):
            color  = STATIC_CMAP(PCAP_NORM(pcap_val))
            et_avg = float(np.mean([p[0] for p in by_pcap[pcap_val]]))
            en_avg = float(np.mean([p[1] for p in by_pcap[pcap_val]]))
            ax.scatter(et_avg, en_avg, color=color, s=120, zorder=2)
        sm = ScalarMappable(cmap=CMAP_NAME, norm=PCAP_NORM)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Static PCAP [W]', fontweight='bold')
        cbar.set_ticks(ACTIONS[::3])

    # ── PID sweep foreground ──────────────────────────────────────────────────
    if sweep_runs:
        # Group by target Hz, compute median (exec_time, energy)
        by_target = defaultdict(list)
        for et, en, tgt in sweep_runs:
            by_target[round(tgt, 4)].append((et, en))

        targets_sorted = sorted(by_target.keys())

        median_pts = []   # (et_med, en_med, target_Hz)
        for tgt in targets_sorted:
            pts    = by_target[tgt]
            et_med = float(np.median([p[0] for p in pts]))
            en_med = float(np.median([p[1] for p in pts]))
            median_pts.append((et_med, en_med, tgt))

        # Plot all median dots first
        et_vals = [p[0] for p in median_pts]
        en_vals = [p[1] for p in median_pts]
        ax.scatter(et_vals, en_vals,
                   color=PID_COLOR, s=160, zorder=5,
                   edgecolors='black', linewidths=0.8)

        # Annotate a subset of dots — keep spacing so labels don't pile up.
        # Keep every point whose (et, en) is far enough from all already-labelled ones.
        x_range  = max(et_vals) - min(et_vals) or 1.0
        y_range  = max(en_vals) - min(en_vals) or 0.1
        x_mid    = (max(et_vals) + min(et_vals)) / 2

        MIN_DX = x_range * 0.10   # minimum time separation between labelled dots
        MIN_DY = y_range * 0.08   # minimum energy separation

        labelled = []   # list of (et, en) already annotated
        for (et_med, en_med, tgt) in median_pts:
            too_close = any(
                abs(et_med - px) < MIN_DX and abs(en_med - py) < MIN_DY
                for px, py in labelled
            )
            if too_close:
                continue
            labelled.append((et_med, en_med))

            label_text = f'{tgt:.2f} Hz'

            # Place label left of dot when in the right half to avoid edge clipping
            if et_med >= x_mid:
                off_x = -x_range * 0.18
            else:
                off_x = x_range * 0.12
            off_y = y_range * 0.10

            ax.annotate(
                label_text,
                xy=(et_med, en_med),
                xytext=(et_med + off_x, en_med + off_y),
                fontsize=9,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='black', lw=1.2),
                zorder=6,
            )

    ax.set_xlabel('Execution Time [s]')
    ax.set_ylabel('Consumed Energy [kJ]')
    ax.set_title(APP_LABELS.get(app, app))
    ax.grid(True, alpha=0.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def main_pareto():
    """Generate Pareto Energy-vs-Time scatter plots for all apps with sweep data."""
    print(f"\nPareto plots")
    print(f"  Sweep data  : {SWEEP_ROOT}")
    print(f"  Static data : {STATIC_ROOT}")
    print(f"  Output dir  : {OUT_DIR}\n")

    sweep_apps = sorted(
        os.path.basename(d.rstrip('/'))
        for d in glob.glob(os.path.join(SWEEP_ROOT, '*/'))
    )
    if not sweep_apps:
        print("  No sweep data found — skipping Pareto plots.")
        return

    for app in sweep_apps:
        sweep_dir  = os.path.join(SWEEP_ROOT, app)
        static_dir = os.path.join(STATIC_ROOT, app)

        sweep_runs  = load_sweep_runs(sweep_dir)
        static_runs = load_static_runs(static_dir) if os.path.isdir(static_dir) else []

        if not sweep_runs:
            print(f"  {app}: no sweep data — skipping")
            continue

        n_setpoints = len(set(round(r[2], 4) for r in sweep_runs))
        print(f"  {app}: {len(sweep_runs)} sweep runs ({n_setpoints} setpoints), "
              f"{len(static_runs)} static runs")

        out_path = os.path.join(OUT_DIR, f'Pareto_Energy_vs_time_{app}.pdf')
        plot_pareto_app(app, static_runs, sweep_runs, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Data root : {DATA_ROOT}")
    print(f"Output dir: {OUT_DIR}\n")

    # ── Per-app time-series plots (if PID_Control data exists) ────────────────
    if os.path.isdir(DATA_ROOT):
        print("Loading iterations:")
        apps_data = load_all_apps(DATA_ROOT)
        if apps_data:
            # Per-app PDF
            per_app_out = os.path.join(OUT_DIR, 'pid_results_per_app.pdf')
            print(f"\nGenerating per-app plots → {per_app_out}")
            with PdfPages(per_app_out) as pdf:
                for app, dfs in sorted(apps_data.items()):
                    print(f"  {app}...")
                    fig = plt.figure(figsize=(10, 10))
                    plot_app_page(fig, app, dfs)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            print(f"  Saved {len(apps_data)} pages.")

            # Summary PDF
            summary = compute_summary(apps_data)
            print_summary(summary)
            summary_out = os.path.join(OUT_DIR, 'pid_results_summary.pdf')
            print(f"\nGenerating summary plot → {summary_out}")
            fig = plt.figure(figsize=(14, max(5, len(apps_data) * 0.55 + 2)))
            plot_summary(fig, summary)
            fig.savefig(summary_out, bbox_inches='tight')
            plt.close(fig)
        else:
            print("No pid_log.csv data found in PID_Control — skipping time-series plots.")
    else:
        print(f"PID_Control directory not found ({DATA_ROOT}) — skipping time-series plots.")

    # ── Pareto Energy-vs-Time scatter plots (from sweep data) ─────────────────
    if args.pareto or os.path.isdir(SWEEP_ROOT):
        main_pareto()

    print("Done.")


if __name__ == '__main__':
    main()
