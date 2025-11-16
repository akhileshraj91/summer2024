#!/usr/bin/env python3
import os
import csv
import sys
import tarfile
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------- config / args -----------------------------
def get_args():
    p = argparse.ArgumentParser(description="Build cumulative-counter data.csv for pcap policy training")
    p.add_argument("--data-root", default=None,
                   help="Root folder containing per-app subfolders (default: /home/cc/comparison/training_data)")
    p.add_argument("--out-name", default="data.csv", help="Output csv filename (default: data.csv)")
    return p.parse_args()

def default_data_root():
    return "/home/cc/comparison/experiment_data/training_data"

# Fixed PCAP action grid (for snapping)
ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0,
           124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]

# Reward configuration: 'multiply' -> r = Power * (norm_IPC ** EXPONENT)
# 'divide' -> r = Power / (norm_IPC ** EXPONENT)
REWARD_MODE = "multiply"
REWARD_EXPONENT = 3

# ----------------------------- helpers -----------------------------
def _snap_action(a, actions):
    a = float(a)
    if not actions:
        return a
    return min(actions, key=lambda v: abs(v - a))

def get_roi_data(df, time_column, start_time, end_time):
    """Select rows in (start_time, end_time]."""
    return df[(df[time_column] > start_time) & (df[time_column] <= end_time)]

def _last_in_window(df, start_t, end_t, time_col='time', value_col='value'):
    """Return the last value within (start_t, end_t], NaN if none."""
    if df is None or len(df) == 0:
        return np.nan
    roi = get_roi_data(df, time_column=time_col, start_time=start_t, end_time=end_t)
    if roi is None or roi.empty:
        return np.nan
    return float(roi.iloc[-1][value_col])

def _mean_in_window(df, start_t, end_t, time_col='timestamp', value_col='value'):
    """Return mean value within (start_t, end_t], NaN if none."""
    if df is None or len(df) == 0:
        return np.nan
    roi = get_roi_data(df, time_column=time_col, start_time=start_t, end_time=end_t)
    if roi is None or roi.empty:
        return np.nan
    return float(np.nanmean(roi[value_col].to_numpy(dtype=float)))

# ----------------------------- loaders / derivations -----------------------------
def generate_PCAP(df):
    """From PCAP_file.csv -> keep rows after the initial 0 time and add elapsed_time, rename time->timestamp."""
    # Expected columns: ['time','actuator','value']
    pc = df.copy()
    if "time" not in pc.columns or "value" not in pc.columns:
        raise ValueError("PCAP_file.csv must have columns: time, actuator, value")
    pc = pc.sort_values("time").reset_index(drop=True)
    # Drop any row where time == 0 (as per your earlier behavior)
    pc = pc[pc["time"] != 0].reset_index(drop=True)
    if pc.empty:
        return pc.rename(columns={"time": "timestamp"})
    pc["elapsed_time"] = pc["time"] - pc["time"].iloc[0]
    pc = pc.rename(columns={"time": "timestamp"})
    return pc

def compute_measured_power(power_df, pcap_df):
    """
    Align power measurements to start at first PCAP timestamp and rename time->timestamp.
    Input measured_power.csv columns: ['time','scope','value'].
    Output columns: ['timestamp','scope','value','elapsed_time'].
    """
    if power_df is None or power_df.empty:
        return power_df
    if "time" not in power_df.columns or "value" not in power_df.columns:
        raise ValueError("measured_power.csv must have columns: time, scope, value")
    if pcap_df is None or pcap_df.empty:
        # still normalize timestamps
        newp = power_df.copy().rename(columns={"time": "timestamp"})
        newp["elapsed_time"] = newp["timestamp"] - newp["timestamp"].iloc[0]
        return newp
    first_pcap_ts = float(pcap_df["timestamp"].iloc[0])
    newp = power_df[power_df["time"] >= first_pcap_ts].copy()
    if newp.empty:
        return pd.DataFrame(columns=["timestamp","scope","value","elapsed_time"])
    newp["elapsed_time"] = newp["time"] - newp["time"].iloc[0]
    newp = newp.rename(columns={"time": "timestamp"})
    return newp

def collect_papi(papi_df, pcap_df):
    """
    Build dict of cumulative PAPI counters as DataFrames with ['time','value','scope','elapsed_time'].
    Input papi.csv columns: ['time','scope','value'] where 'value' is cumulative.
    We return only the three we need: PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L3_TCM
    """
    if papi_df is None or papi_df.empty:
        return {}
    if "time" not in papi_df.columns or "scope" not in papi_df.columns or "value" not in papi_df.columns:
        raise ValueError("papi.csv must have columns: time, scope, value")

    # restrict to >= first PCAP timestamp
    if pcap_df is not None and not pcap_df.empty:
        first_pcap = float(pcap_df["timestamp"].iloc[0])
        papi_df = papi_df[papi_df["time"] >= first_pcap].copy()

    out = {}
    required = ["PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_L3_TCM", "PAPI_RES_STL"]

    # Scopes look like: nrm.papi.<COUNTER>.<something> – we extract the third token as key
    for scope_name in papi_df["scope"].unique():
        parts = str(scope_name).split(".")
        if len(parts) > 3:
            key = parts[3]
            if key in required:
                dfk = papi_df[papi_df["scope"] == scope_name][["time","scope","value"]].copy()
                dfk = dfk.sort_values("time").reset_index(drop=True)
                dfk["elapsed_time"] = dfk["time"] - dfk["time"].iloc[0]
                out[key] = dfk
    

    # Sanity: ensure all required present (may be missing in some traces)
    for r in required:
        if r not in out:
            out[r] = pd.DataFrame(columns=["time","scope","value","elapsed_time"])

    return out


def collect_frequency(freq_df, pcap_df):
    """
    Parse frequency.csv and return a dict with:
      - 'raw': DataFrame with columns ['time','core','value','elapsed_time'] (one row per core sample)
      - 'avg': DataFrame with columns ['time','value','elapsed_time'] representing mean frequency across cores at each timestamp

    Accepts files that may or may not have headers and will try common column names.
    If pcap_df is provided the series will be restricted to timestamps >= first PCAP timestamp.
    """
    if freq_df is None or freq_df.empty:
        return {"raw": pd.DataFrame(columns=["time", "core", "value", "elapsed_time"]),
                "avg": pd.DataFrame(columns=["time", "value", "elapsed_time"])}

    df = freq_df.copy()

    # Heuristically determine column names: prefer explicit names, else positional
    cols = [c.lower() for c in df.columns]
    # time col
    if "time" in cols:
        time_col = df.columns[cols.index("time")]
    else:
        time_col = df.columns[0]
    # core col
    if "core" in cols:
        core_col = df.columns[cols.index("core")]
    elif "cpu" in cols:
        core_col = df.columns[cols.index("cpu")]
    else:
        core_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    # value col
    if "value" in cols:
        val_col = df.columns[cols.index("value")]
    elif "freq" in cols or "frequency" in cols:
        idx = cols.index("freq") if "freq" in cols else cols.index("frequency")
        val_col = df.columns[idx]
    else:
        # fallback to third column if present
        val_col = df.columns[2] if len(df.columns) > 2 else df.columns[-1]

    # Normalize to known names
    df = df.rename(columns={time_col: "time", core_col: "core", val_col: "value"})

    # Ensure numeric types
    try:
        df["time"] = df["time"].astype(float)
        df["core"] = df["core"].astype(int)
        df["value"] = df["value"].astype(float)
    except Exception:
        # best-effort: coerce errors to NaN then drop
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["core"] = pd.to_numeric(df["core"], errors="coerce").fillna(-1).astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)

    # Restrict to >= first PCAP timestamp if provided
    if pcap_df is not None and not pcap_df.empty:
        first_pcap = float(pcap_df["timestamp"].iloc[0])
        df = df[df["time"] >= first_pcap].copy()

    if df.empty:
        return {"raw": pd.DataFrame(columns=["time", "core", "value", "elapsed_time"]),
                "avg": pd.DataFrame(columns=["time", "value", "elapsed_time"]) }

    df["elapsed_time"] = df["time"] - df["time"].iloc[0]

    # Compute average frequency across cores per timestamp
    # Some traces record each core in sequence for the same timestamp; we group by time.
    grp = df.groupby("time")["value"].mean().reset_index()
    grp = grp.rename(columns={"value": "value"})
    grp["elapsed_time"] = grp["time"] - grp["time"].iloc[0]

    return {"raw": df.reset_index(drop=True), "avg": grp.reset_index(drop=True)}

def build_training_data(root_dir):
    """
    Walk root_dir/<APP>/<TRACE>, load CSVs and build:
      training_data[APP][TRACE] = {
         'PCAP': DataFrame(['timestamp','actuator','value','elapsed_time']),
         'measured_power': DataFrame(['timestamp','scope','value','elapsed_time']),
         'papi': { 'PAPI_TOT_INS': df, 'PAPI_TOT_CYC': df, 'PAPI_L3_TCM': df }
      }
    """
    training_data = {}

    # root_dir/<APP>/
    for app in sorted(os.listdir(root_dir)):
        app_dir = os.path.join(root_dir, app)
        if not os.path.isdir(app_dir):
            continue
        training_data[app] = {}

        # Within <APP>, we may see either trace folders (like compressed_iteration_...) or tars
        for item in sorted(os.listdir(app_dir)):
            item_path = os.path.join(app_dir, item)

            extract_dir = None
            if os.path.isdir(item_path):
                # a trace directory with CSVs
                extract_dir = item_path
            elif item.endswith(".tar"):
                # extract tar into side-by-side folder (item without .tar)
                extract_dir = os.path.join(app_dir, item[:-4])
                if not os.path.exists(extract_dir):
                    os.makedirs(extract_dir, exist_ok=True)
                    with tarfile.open(item_path, "r") as tar:
                        tar.extractall(path=extract_dir)
            else:
                # skip unknown files
                continue

            # Expect these CSVs inside extract_dir
            pcap_csv = os.path.join(extract_dir, "PCAP_file.csv")
            # New action file produced by frequency-based runner
            freq_action_csv = os.path.join(extract_dir, "FREQ_file.csv")
            power_csv = os.path.join(extract_dir, "measured_power.csv")
            papi_csv  = os.path.join(extract_dir, "papi.csv")
            freq_csv  = os.path.join(extract_dir, "frequency.csv")

            # Accept either the old PCAP_file.csv or the new FREQ_file.csv as the action timeline
            action_csv = None
            if os.path.exists(freq_action_csv):
                action_csv = freq_action_csv
            elif os.path.exists(pcap_csv):
                action_csv = pcap_csv

            if action_csv is None or not os.path.exists(power_csv) or not os.path.exists(papi_csv):
                # Skip if any required file missing
                # (progress.csv, energy.csv, frequency.csv are not required for cumulative CSV)
                continue

            try:
                pc = pd.read_csv(action_csv)
                pw = pd.read_csv(power_csv)
                pa = pd.read_csv(papi_csv)

                PCAP = generate_PCAP(pc)
                measured_power = compute_measured_power(pw, PCAP)
                papi = collect_papi(pa, PCAP)

                # optional frequency.csv
                freq = None
                try:
                    if os.path.exists(freq_csv):
                        fq = pd.read_csv(freq_csv)
                        freq = collect_frequency(fq, PCAP)
                except Exception:
                    freq = None

                training_data[app][item] = {
                    "PCAP": PCAP,
                    "measured_power": measured_power,
                    "papi": papi,
                    "frequency": freq,
                }
            except Exception as e:
                print(f"[WARN] Skipping trace due to parse error: {item_path} -> {e}")

    return training_data

# ----------------------------- builder for cumulative CSV -----------------------------
def build_cumulative_csv_from_training_data(training_data, actions, out_dir, out_name="data.csv"):
    """
    Write CSV with columns:
      App,EndTime,PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_L3_TCM,Power,Action
    Each row corresponds to a PCAP window end time (t2).
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    rows = []
    for app in training_data:
        for trace in training_data[app]:
            td = training_data[app][trace]
            pcap_df = td.get("PCAP")
            if pcap_df is None or pcap_df.empty:
                continue
            pcap_df = pcap_df.sort_values("timestamp").reset_index(drop=True)

            # cumulative PAPI series (with 'time','value')
            papi = td.get("papi", {})
            PAPI_INS = papi.get("PAPI_TOT_INS", pd.DataFrame(columns=["time","value"]))
            PAPI_CYC = papi.get("PAPI_TOT_CYC", pd.DataFrame(columns=["time","value"]))
            PAPI_L3M = papi.get("PAPI_L3_TCM", pd.DataFrame(columns=["time","value"]))
            PAPI_RES = papi.get("PAPI_RES_STL", pd.DataFrame(columns=["time","value"]))

            power_df = td.get("measured_power")

            # We build raw per-window metrics first (rates and IPC), then normalize IPC and compute reward
            t1 = float("-inf")
            for _, row in pcap_df.iterrows():
                t2 = float(row["timestamp"])
                # Determine action: if the recorded action values are large (e.g., Hz),
                # avoid snapping to the legacy power grid. Preserve frequency actions as-is.
                raw_action = float(row["value"])
                if abs(raw_action) > 1e6:
                    # frequency-like action (Hz) — keep original numeric value
                    action = raw_action
                else:
                    action = _snap_action(raw_action, actions)

                # cumulative values at window end
                ins_end = _last_in_window(PAPI_INS, t1, t2, time_col="time", value_col="value")
                cyc_end = _last_in_window(PAPI_CYC, t1, t2, time_col="time", value_col="value")
                l3m_end = _last_in_window(PAPI_L3M, t1, t2, time_col="time", value_col="value")
                res_end = _last_in_window(PAPI_RES, t1, t2, time_col="time", value_col="value")

                # cumulative values at window start (last sample <= t1)
                ins_start = _last_in_window(PAPI_INS, float("-inf"), t1, time_col="time", value_col="value")
                cyc_start = _last_in_window(PAPI_CYC, float("-inf"), t1, time_col="time", value_col="value")
                l3m_start = _last_in_window(PAPI_L3M, float("-inf"), t1, time_col="time", value_col="value")
                res_start = _last_in_window(PAPI_RES, float("-inf"), t1, time_col="time", value_col="value")

                pwr_mean = _mean_in_window(power_df, t1, t2, time_col="timestamp", value_col="value")
                # average core frequency in this window (optional)
                freq_entry = td.get("frequency")
                freq_mean = np.nan
                if freq_entry and isinstance(freq_entry, dict):
                    freq_avg_df = freq_entry.get("avg")
                    freq_mean = _mean_in_window(freq_avg_df, t1, t2, time_col="time", value_col="value")

                # compute dt and per-window rates (instantaneous rates)
                dt = t2 - t1 if np.isfinite(t1) else np.nan
                ins_rate = np.nan
                cyc_rate = np.nan
                l3_rate = np.nan
                res_rate = np.nan
                ipc = np.nan
                if np.isfinite(dt) and dt > 0:
                    try:
                        ins_rate = (float(ins_end) - float(ins_start)) / dt if not np.isnan(ins_end) and not np.isnan(ins_start) else np.nan
                    except Exception:
                        ins_rate = np.nan
                    try:
                        cyc_rate = (float(cyc_end) - float(cyc_start)) / dt if not np.isnan(cyc_end) and not np.isnan(cyc_start) else np.nan
                    except Exception:
                        cyc_rate = np.nan
                    try:
                        l3_rate = (float(l3m_end) - float(l3m_start)) / dt if not np.isnan(l3m_end) and not np.isnan(l3m_start) else np.nan
                    except Exception:
                        l3_rate = np.nan
                    try:
                        res_rate = (float(res_end) - float(res_start)) / dt if not np.isnan(res_end) and not np.isnan(res_start) else np.nan
                    except Exception:
                        res_rate = np.nan

                    if not np.isnan(ins_rate) and not np.isnan(cyc_rate) and cyc_rate != 0:
                        ipc = ins_rate / cyc_rate

                rows.append({
                    "App": app,
                    "EndTime": t2,
                    "PAPI_TOT_INS": ins_end,
                    "PAPI_TOT_CYC": cyc_end,
                    "PAPI_L3_TCM": l3m_end,
                    "Power": pwr_mean,
                    "AvgFreq": freq_mean,
                    "Action": float(action),
                    # raw per-window derived features
                    "INS_RATE": ins_rate,
                    "CYC_RATE": cyc_rate,
                    "L3M_RATE": l3_rate,
                    "RES_STL_RATE": res_rate,
                    "IPC": ipc,
                })

                t1 = t2

    # build dataframe from collected rows
    df = pd.DataFrame(rows)

    # compute normalized IPC across the dataset (avoid divide-by-zero)
    if df.empty or ("IPC" not in df.columns):
        df["IPC_NORM"] = np.nan
    else:
        ipc_vals = df["IPC"].replace([np.inf, -np.inf], np.nan).dropna()
        max_ipc = float(ipc_vals.max()) if not ipc_vals.empty else 1.0
        if max_ipc == 0:
            max_ipc = 1.0
        df["IPC_NORM"] = df["IPC"].apply(lambda x: float(x) / max_ipc if np.isfinite(x) else np.nan)

    # compute reward per window using configured mode
    def compute_reward(pwr, ipc_norm):
        if pwr is None or (isinstance(pwr, float) and np.isnan(pwr)):
            return np.nan
        if ipc_norm is None or (isinstance(ipc_norm, float) and np.isnan(ipc_norm)):
            return np.nan
        try:
            if REWARD_MODE == "multiply":
                return float(pwr) * (float(ipc_norm) ** REWARD_EXPONENT)
            elif REWARD_MODE == "divide":
                return float(pwr) / (max(float(ipc_norm) ** REWARD_EXPONENT, 1e-12))
            else:
                # default
                return float(pwr) * (float(ipc_norm) ** REWARD_EXPONENT)
        except Exception:
            return np.nan

    df["Reward"] = df.apply(lambda r: compute_reward(r.get("Power"), r.get("IPC_NORM")), axis=1)

    # keep AvgFreq optional (don't drop rows missing it)
    before = len(df)
    subset = [c for c in ["Power", "Action"] if c in df.columns]
    if subset:
        df = df.dropna(subset=subset)
    after = len(df)

    # Select and order columns for output
    out_cols = [
        "App", "EndTime", "PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_L3_TCM",
        "Power", "AvgFreq", "Action",
        # derived rates / state
        "INS_RATE", "CYC_RATE", "L3M_RATE", "RES_STL_RATE", "IPC", "IPC_NORM",
        # reward
        "Reward",
    ]
    # ensure columns exist
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    out_df = df[out_cols]
    out_path = os.path.join(out_dir, out_name)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {after} rows (dropped {before - after} with NaNs) -> {out_path}")
    return out_path

# ----------------------------- main -----------------------------
def main():
    args = get_args()
    data_root = args.data_root or default_data_root()

    if not os.path.isdir(data_root):
        raise SystemExit(f"Data root not found: {data_root}")

    print(f"[INFO] Scanning data root: {data_root}")
    training_data = build_training_data(data_root)

    # Basic stats
    num_apps = len(training_data)
    num_traces = sum(len(training_data[a]) for a in training_data)
    print(f"[INFO] Loaded apps: {num_apps}, traces: {num_traces}")

    out_csv = build_cumulative_csv_from_training_data(
        training_data=training_data,
        actions=ACTIONS,
        out_dir=data_root,
        out_name=args.out_name,
    )
    print(f"[DONE] data.csv ready: {out_csv}")

if __name__ == "__main__":
    main()
