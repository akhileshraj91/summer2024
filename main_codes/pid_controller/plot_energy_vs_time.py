#!/usr/bin/env python3
"""
Energy vs Execution Time scatter plots for the PID controller.

Replicates the style of figures/Energy_vs_time_<app>.pdf:
  - Background (YlOrRd): static-PCAP experiments from experiment_data/plotting_data
  - Foreground (teal, annotated): PID controller runs from experiment_data/PID_Control

One PDF per application is saved to pid_controller/ (or --out-dir).

Usage:
  python plot_energy_vs_time.py
  python plot_energy_vs_time.py --out-dir ../figures
"""

import argparse
import glob
import io
import os
import tarfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings('ignore')

# ── Argument parsing ──────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument('--out-dir',      default=None)
_p.add_argument('--static-dir',   default=None,
                help='Path to experiment_data/plotting_data')
_p.add_argument('--pid-dir',      default=None,
                help='Path to experiment_data/PID_Control')
args = _p.parse_args()

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT      = os.path.join(SCRIPT_DIR, '..')
STATIC_DIR  = os.path.normpath(args.static_dir or
              os.path.join(PARENT, 'experiment_data', 'plotting_data'))
PID_DIR     = os.path.normpath(args.pid_dir    or
              os.path.join(PARENT, 'experiment_data', 'PID_Control'))
OUT_DIR     = args.out_dir or SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style (matching the reference figures exactly) ────────────────────────────
plt.rcParams.update({
    'font.size':          14,
    'font.weight':        'bold',
    'axes.labelsize':     'large',
    'axes.labelweight':   'bold',
    'axes.titlesize':     16,
    'axes.titleweight':   'bold',
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'figure.dpi':         150,
})

ACTIONS      = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0,
                124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
CMAP_NAME    = 'YlOrRd'
NORM         = Normalize(vmin=min(ACTIONS), vmax=max(ACTIONS))
STATIC_CMAP  = plt.cm.get_cmap(CMAP_NAME)
PID_COLOR    = '#00796B'   # teal — matches RL controller style
MARKER_BG    = 120         # marker size for static PCAP dots
MARKER_PID   = 160         # marker size for PID dots

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

# ── Data helpers ──────────────────────────────────────────────────────────────

def _read_csvs_from_tar(tar_path):
    """Return dict of {filename: DataFrame} for CSV files in a tar."""
    out = {}
    try:
        with tarfile.open(tar_path, 'r:*') as tar:
            for name in tar.getnames():
                if name.endswith('.csv'):
                    f = tar.extractfile(name)
                    if f:
                        out[os.path.basename(name)] = pd.read_csv(io.BytesIO(f.read()))
    except Exception as exc:
        print(f"  [WARN] {os.path.basename(tar_path)}: {exc}")
    return out


def _read_csvs_loose(directory):
    """Return dict of {filename: DataFrame} for CSV files directly in directory."""
    out = {}
    for path in glob.glob(os.path.join(directory, '*.csv')):
        try:
            out[os.path.basename(path)] = pd.read_csv(path)
        except Exception:
            pass
    return out


def _execution_time(csvs):
    """Execution time in seconds = span of progress timestamps."""
    prog = csvs.get('progress.csv')
    if prog is None or len(prog) < 2:
        return None
    return float(prog['time'].iloc[-1] - prog['time'].iloc[0])


def _energy_kj(csvs):
    """
    Consumed energy in kJ by integrating measured_power.csv.
    Averages the two CPU socket readings and integrates over time.
    """
    pw = csvs.get('measured_power.csv')
    if pw is None or len(pw) < 2:
        return None
    pw = pw.sort_values('time').copy()
    # Average both sockets at each timestamp
    avg = pw.groupby('time')['value'].mean().reset_index()
    avg.columns = ['time', 'power']
    dt = avg['time'].diff().fillna(0).values
    energy_j = float(np.sum(avg['power'].values * dt))
    return energy_j / 1000.0


def _static_pcap(csvs):
    """Return the (constant) PCAP value used in a static experiment."""
    pc = csvs.get('PCAP_file.csv')
    if pc is None or len(pc) == 0:
        return None
    return float(pc['value'].mode().iloc[0])


def load_static_runs(app_dir):
    """
    Load (execution_time, energy_kj, pcap) for all static-PCAP runs in app_dir.
    Handles both compressed tars and loose CSVs.
    """
    runs = []
    for tar_path in sorted(glob.glob(os.path.join(app_dir, '*.tar'))):
        csvs = _read_csvs_from_tar(tar_path)
        et = _execution_time(csvs)
        en = _energy_kj(csvs)
        pc = _static_pcap(csvs)
        if et and en and pc:
            runs.append((et, en, pc))
    # Also pick up any uncompressed run sitting in the directory
    loose = _read_csvs_loose(app_dir)
    if loose:
        et = _execution_time(loose)
        en = _energy_kj(loose)
        pc = _static_pcap(loose)
        if et and en and pc:
            runs.append((et, en, pc))
    return runs


def load_pid_runs(app_dir):
    """
    Load (execution_time, energy_kj, mean_pcap, target_perf) for all PID runs.
    Falls back to measured_power.csv energy if pid_log is present but power is NaN.
    """
    runs = []
    for tar_path in sorted(glob.glob(os.path.join(app_dir, '*.tar'))):
        csvs = _read_csvs_from_tar(tar_path)
        et = _execution_time(csvs)
        en = _energy_kj(csvs)
        if not et or not en:
            continue
        # Target perf and mean PCAP from pid_log if available
        target = None
        mean_pcap = None
        pl = csvs.get('pid_log.csv')
        if pl is not None and len(pl) > 0:
            target    = float(pl['target_perf_Hz'].iloc[0])
            mean_pcap = float(pl['pcap_cmd_W'].mean())
        runs.append((et, en, mean_pcap, target))
    # Loose files
    loose = _read_csvs_loose(app_dir)
    if loose:
        et = _execution_time(loose)
        en = _energy_kj(loose)
        if et and en:
            target = mean_pcap = None
            pl = loose.get('pid_log.csv')
            if pl is not None and len(pl) > 0:
                target    = float(pl['target_perf_Hz'].iloc[0])
                mean_pcap = float(pl['pcap_cmd_W'].mean())
            runs.append((et, en, mean_pcap, target))
    return runs


# ── Per-app plot ──────────────────────────────────────────────────────────────

def plot_app(app, static_runs, pid_runs, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    # ── Background: static PCAP dots ─────────────────────────────────────────
    if static_runs:
        # Group by PCAP so we can assign the right colour index
        from collections import defaultdict
        by_pcap = defaultdict(list)
        for et, en, pc in static_runs:
            # Map to nearest action to get a consistent colour index
            nearest = min(ACTIONS, key=lambda a: abs(a - pc))
            by_pcap[nearest].append((et, en))

        for pcap_val in sorted(by_pcap.keys()):
            color = STATIC_CMAP(NORM(pcap_val))
            for et, en in by_pcap[pcap_val]:
                ax.scatter(et, en, color=color, s=MARKER_BG, zorder=2)

        # Colour bar
        sm = ScalarMappable(cmap=CMAP_NAME, norm=NORM)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Static PCAP [W]', fontweight='bold')
        cbar.set_ticks(ACTIONS[::3])

    # ── Foreground: PID dots + annotation ────────────────────────────────────
    if pid_runs:
        et_vals  = [r[0] for r in pid_runs]
        en_vals  = [r[1] for r in pid_runs]
        target   = pid_runs[0][3]   # same target for all runs of this app

        ax.scatter(et_vals, en_vals,
                   color=PID_COLOR, s=MARKER_PID, zorder=5,
                   edgecolors='black', linewidths=0.8,
                   label='PID Controller')

        # Annotate cluster mean with a boxed label
        et_mean = float(np.mean(et_vals))
        en_mean = float(np.mean(en_vals))

        label_text = ('PID Controller' if target is None
                      else f'PID\n(target={target:.3f} Hz)')

        # Offset label so it doesn't overlap the dots
        x_range = ax.get_xlim()
        offset_x = (max(et_vals) - min(et_vals)) * 0.5 + 1.5
        offset_y = (max(en_vals) - min(en_vals)) * 0.4

        ax.annotate(
            label_text,
            xy=(et_mean, en_mean),
            xytext=(et_mean + offset_x, en_mean + offset_y),
            fontsize=10,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='black', lw=1.2),
            zorder=6,
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Static PCAP data : {STATIC_DIR}")
    print(f"PID data         : {PID_DIR}")
    print(f"Output dir       : {OUT_DIR}\n")

    # Collect all apps that have PID data
    pid_apps = sorted(
        os.path.basename(d.rstrip('/'))
        for d in glob.glob(os.path.join(PID_DIR, '*/'))
    )
    if not pid_apps:
        print("No PID experiment directories found.")
        return

    for app in pid_apps:
        pid_app_dir    = os.path.join(PID_DIR,    app)
        static_app_dir = os.path.join(STATIC_DIR, app)

        pid_runs    = load_pid_runs(pid_app_dir)
        static_runs = (load_static_runs(static_app_dir)
                       if os.path.isdir(static_app_dir) else [])

        if not pid_runs:
            print(f"  {app}: no PID data — skipping")
            continue

        print(f"  {app}: {len(pid_runs)} PID runs, "
              f"{len(static_runs)} static runs")

        out_path = os.path.join(OUT_DIR, f'Energy_vs_time_PID_{app}.pdf')
        plot_app(app, static_runs, pid_runs, out_path)

    print("\nDone.")


if __name__ == '__main__':
    main()
