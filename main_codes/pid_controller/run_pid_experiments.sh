#!/bin/bash
# Run PID performance controller experiments.
#
# Two modes:
#   default  — one target per app (auto 80% of max perf), 10 repetitions
#   --sweep  — 20 setpoints from 5% to 100% of the performance range,
#              3 repetitions each, results in experiment_data/PID_Control_sweep/
#
# Usage:
#   ./run_pid_experiments.sh                          # all apps, default mode
#   ./run_pid_experiments.sh --sweep                  # all apps, full sweep
#   ./run_pid_experiments.sh -a ones-npb-bt           # single app, default
#   ./run_pid_experiments.sh -a ones-npb-bt --sweep   # single app, sweep
#   ./run_pid_experiments.sh -a ones-npb-bt -t 6.0    # explicit target (default mode)

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_SCRIPT="$SCRIPT_DIR/pid_control.py"
COEFF_DIR="$SCRIPT_DIR/.."
LOG_DIR="$SCRIPT_DIR/../experiment_logs/pid"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

DEFAULT_REPS=10
SWEEP_REPS=3
SWEEP_STEPS=20          # number of setpoints: 5%,10%,15%,...,100%
TIMEOUT=3600            # seconds per single pid_control.py call

TARGET_PERF=""          # empty → auto (80% of polynomial max)
FILTER_APP=""           # empty → all apps
SWEEP_MODE=false

declare -a ALL_APPS=(
    "ones-stream-triad"
    "ones-stream-add"
    "ones-stream-scale"
    "ones-npb-is"
    "ones-npb-bt"
    "ones-npb-ep"
    "ones-npb-ft"
    "ones-npb-mg"
    "ones-stream-copy"
    "ones-stream-full"
    "ones-npb-cg"
)

# ── Argument parsing ────────────────────────────────────────────────────────────
show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -a APP        Run only this application (default: all apps)
  -t HZ         Target performance in Hz (default mode only; default: 80% of max)
  -n N          Repetitions per setpoint (default: $DEFAULT_REPS normal, $SWEEP_REPS sweep)
  --sweep       Sweep 5%–100% of each app's performance range (20 setpoints)
  --steps N     Number of sweep steps (default: $SWEEP_STEPS)
  --timeout S   Timeout per run in seconds (default: $TIMEOUT)
  -h            Show this help

Examples:
  $0                              # all apps, auto target, 10 reps
  $0 --sweep                      # all apps, 10 setpoints × 3 reps each
  $0 -a ones-npb-bt --sweep       # single app sweep
  $0 -a ones-npb-bt -t 6.0 -n 5  # explicit target
EOF
}

REPS=""   # resolved after parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a)         FILTER_APP="$2";  shift 2 ;;
        -t)         TARGET_PERF="$2"; shift 2 ;;
        -n)         REPS="$2";        shift 2 ;;
        --sweep)    SWEEP_MODE=true;  shift   ;;
        --steps)    SWEEP_STEPS="$2"; shift 2 ;;
        --timeout)  TIMEOUT="$2";     shift 2 ;;
        -h|--help)  show_usage; exit 0 ;;
        *)          echo "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

# Default repetitions depend on mode
if [[ -z "$REPS" ]]; then
    REPS=$( $SWEEP_MODE && echo "$SWEEP_REPS" || echo "$DEFAULT_REPS" )
fi

# Build app list
if [[ -n "$FILTER_APP" ]]; then
    APPS=("$FILTER_APP")
else
    APPS=("${ALL_APPS[@]}")
fi

mkdir -p "$LOG_DIR"

# ── Helpers ─────────────────────────────────────────────────────────────────────

# Compute sweep setpoints for one app using the static characteristics polynomial.
# Prints SWEEP_STEPS space-separated Hz values (10% … 100% of achievable range).
compute_setpoints() {
    local app="$1"
    python3 - <<PYEOF
import numpy as np, yaml, glob, os, sys

coeff_dir = '$COEFF_DIR'
app       = '$app'
steps     = $SWEEP_STEPS

fp = glob.glob(os.path.join(coeff_dir, f'static_characteristics_{app}_coeffs.yaml'))
if not fp:
    sys.exit(f"[ERROR] No static characteristics YAML for {app}")

with open(fp[0]) as f:
    d = yaml.safe_load(f)

cp = d['coefficients_perf']
a2, a1, a0 = cp
PCAP_MIN, PCAP_MAX = 78.0, 165.0

p_at_min = float(np.polyval(cp, PCAP_MIN))
p_at_max = float(np.polyval(cp, PCAP_MAX))

# Vertex of parabola (optimal PCAP for max/min performance)
if abs(a2) > 1e-12:
    v = float(np.clip(-a1 / (2.0 * a2), PCAP_MIN, PCAP_MAX))
    p_vertex = float(np.polyval(cp, v))
else:
    p_vertex = max(p_at_min, p_at_max)

perf_max = max(p_at_min, p_at_max, p_vertex)
perf_min = float(np.polyval(cp, PCAP_MIN))   # performance at most restrictive cap

# Setpoints: 5%, 10%, 15%, ..., 100% of the achievable range
setpoints = [perf_min + (k / steps) * (perf_max - perf_min)
             for k in range(1, steps + 1)]

print(' '.join(f'{s:.6f}' for s in setpoints))
PYEOF
}

# Run one pid_control.py call, log output, handle timeout/failure.
run_one() {
    local app="$1"
    local target="$2"      # Hz or empty
    local reps="$3"
    local exp_name="$4"
    local label="$5"       # for logging

    local log_file="$LOG_DIR/${app}_${label}_${TIMESTAMP}.log"
    local cmd=(python3 "$PID_SCRIPT" -a "$app" -n "$reps" --exp-name "$exp_name")
    [[ -n "$target" ]] && cmd+=(-t "$target")

    echo "    target=${target:-auto}  reps=$reps  log=$(basename "$log_file")"

    if timeout "$TIMEOUT" "${cmd[@]}" > "$log_file" 2>&1; then
        echo "    OK"
    else
        local ec=$?
        [[ $ec -eq 124 ]] && echo "    TIMED OUT (${TIMEOUT}s)" \
                           || echo "    FAILED (exit $ec)"
        pkill -f "pid_control.py" 2>/dev/null || true
        return 1
    fi
}

# ── Header ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  PID Experiment Runner"
echo "  Mode      : $( $SWEEP_MODE && echo "SWEEP (${SWEEP_STEPS} setpoints × ${REPS} reps)" \
                                   || echo "DEFAULT (${REPS} reps)" )"
echo "  Apps      : ${APPS[*]}"
echo "  Started   : $(date)"
echo "  Log dir   : $LOG_DIR"
echo "============================================================"

TOTAL_APPS=${#APPS[@]}
PASSED=0; FAILED=0; IDX=0

# ── Main loop ────────────────────────────────────────────────────────────────────
for APP in "${APPS[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "[$IDX/$TOTAL_APPS] $APP"

    APP_OK=true

    if $SWEEP_MODE; then
        # ── Sweep mode: 10 setpoints per app ─────────────────────────────────
        SETPOINTS_STR=$(compute_setpoints "$APP") || {
            echo "  [ERROR] Could not compute setpoints for $APP — skipping"
            FAILED=$((FAILED + 1))
            continue
        }
        read -r -a SETPOINTS <<< "$SETPOINTS_STR"

        step_size=$((100 / SWEEP_STEPS))
        pct=$step_size
        for TARGET_HZ in "${SETPOINTS[@]}"; do
            echo "  [${pct}%] target=${TARGET_HZ} Hz"
            if ! run_one "$APP" "$TARGET_HZ" "$REPS" \
                         "PID_Control_sweep" \
                         "sweep_${pct}pct"; then
                APP_OK=false
            fi
            pct=$((pct + step_size))
            sleep 1
        done

    else
        # ── Default mode: single target ───────────────────────────────────────
        if ! run_one "$APP" "$TARGET_PERF" "$REPS" \
                     "PID_Control" \
                     "default"; then
            APP_OK=false
        fi
    fi

    $APP_OK && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))
    sleep 2
done

# ── Summary ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Batch complete : $(date)"
echo "  Passed : $PASSED / $TOTAL_APPS"
echo "  Failed : $FAILED / $TOTAL_APPS"
if $SWEEP_MODE; then
    echo "  Results: experiment_data/PID_Control_sweep/<app>/"
else
    echo "  Results: experiment_data/PID_Control/<app>/"
fi
echo "  Logs   : $LOG_DIR"
echo "============================================================"

[[ $FAILED -eq 0 ]]
