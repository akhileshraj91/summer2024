#!/usr/bin/env python3
"""
PID Performance Controller for HPC benchmarks.

Regulates application progress rate (Hz) to a target setpoint by actuating
the node Power Cap (PCAP).  For each application the static characteristics
polynomial  Perf(PCAP) = a2*PCAP^2 + a1*PCAP + a0  (from coefficients_perf)
is used to:
  1. Compute a feedforward PCAP by inverting the polynomial at the target
     performance.
  2. Derive PID gains from the static plant gain  K = dPerf/dPCAP at the
     operating point.

Node power is collected only as telemetry; it is NOT the controlled variable.

Usage:
  python pid_control.py -a <app> [<app> ...] -t <target_Hz> [-n <steps>]
                        [--kp KP] [--ki KI] [--kd KD]

Example:
  python pid_control.py -a ones-npb-bt -t 6.0 -n 3
  python pid_control.py -a ones-stream-copy -t 250.0 -n 1
"""

import csv
import glob
import os
import signal
import subprocess
import tarfile
import time
import yaml
from datetime import datetime

import numpy as np
import nrm
import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0,
           124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
PCAP_MIN, PCAP_MAX = ACTIONS[0], ACTIONS[-1]
PCAP_NOMINAL = (PCAP_MIN + PCAP_MAX) / 2.0   # 121.5 W — nominal operating point
CONTROL_PERIOD = 2.0                          # seconds between actuations

# ──────────────────────────────────────────────────────────────────────────────
# Signal handling — terminate benchmark and flush files on SIGTERM / Ctrl-C
# ──────────────────────────────────────────────────────────────────────────────
_active_process = None


def _sigterm_handler(signum, frame):
    global _active_process
    if _active_process is not None and _active_process.poll() is None:
        _active_process.terminate()
        try:
            _active_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _active_process.kill()
    raise SystemExit(1)


signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT,  _sigterm_handler)

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="PID performance controller for HPC benchmarks"
)
parser.add_argument('-a', '--application', nargs='+', required=True,
                    help='Application(s) to evaluate (e.g. ones-npb-bt)')
parser.add_argument('-t', '--target_perf', type=float, default=None,
                    help='Target progress rate setpoint in Hz. '
                         'If omitted, defaults to 80%% of max performance '
                         'predicted by the static characteristics polynomial.')
parser.add_argument('-n', '--num_steps', type=int, default=3,
                    help='Repetitions per application (default: 3)')
parser.add_argument('--kp', type=float, default=None,
                    help='Override proportional gain (default: derived from '
                         'static characteristics)')
parser.add_argument('--ki', type=float, default=None,
                    help='Override integral gain (default: derived from '
                         'static characteristics)')
parser.add_argument('--kd', type=float, default=0.0,
                    help='Derivative gain (default: 0)')
parser.add_argument('--exp-name', type=str, default='PID_Control',
                    help='Sub-directory name under experiment_data/ for results '
                         '(default: PID_Control)')
args = parser.parse_args()

APPLICATIONS = args.application

# ──────────────────────────────────────────────────────────────────────────────
# Static characteristics helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_static_coeffs(app_short, coeff_dir):
    """Load the static characteristics YAML for an application."""
    pattern = os.path.join(coeff_dir, f'static_characteristics_{app_short}_coeffs.yaml')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No static characteristics file for '{app_short}' in {coeff_dir}"
        )
    with open(matches[0]) as f:
        return yaml.safe_load(f)


def perf_at_pcap(coeffs_perf, pcap):
    """Evaluate the performance polynomial at a given PCAP."""
    return float(np.polyval(coeffs_perf, pcap))


def max_perf_pcap(coeffs_perf):
    """
    Return (pcap_opt, perf_max): the PCAP that maximises performance and the
    corresponding performance value.  The degree-2 polynomial has its vertex
    at  PCAP_opt = -a1 / (2*a2).  Clamped to [PCAP_MIN, PCAP_MAX].
    """
    a2, a1, _a0 = coeffs_perf
    if abs(a2) < 1e-12:          # linear — maximum is at one boundary
        pcap_opt = PCAP_MAX if a1 > 0 else PCAP_MIN
    else:
        pcap_opt = -a1 / (2.0 * a2)
        pcap_opt = float(np.clip(pcap_opt, PCAP_MIN, PCAP_MAX))
        # For a concave-down parabola (a2 < 0) this is the true maximum.
        # For a concave-up parabola (a2 > 0) the max within range is at a boundary.
        if a2 > 0:
            pcap_opt = max([PCAP_MIN, PCAP_MAX],
                           key=lambda p: perf_at_pcap(coeffs_perf, p))
    return pcap_opt, perf_at_pcap(coeffs_perf, pcap_opt)


def feedforward_pcap(coeffs_perf, target_perf):
    """
    Compute the feedforward PCAP for a target performance by inverting
    Perf(PCAP) = a2*PCAP^2 + a1*PCAP + a0.

    Solves  a2*u^2 + a1*u + (a0 - target_perf) = 0  for u in [PCAP_MIN, PCAP_MAX].
    Among valid roots picks the one closest to PCAP_NOMINAL.
    Falls back to PCAP_NOMINAL if no valid root exists.
    """
    a2, a1, a0 = coeffs_perf
    if abs(a2) < 1e-12:
        u = (target_perf - a0) / a1 if abs(a1) > 1e-12 else PCAP_NOMINAL
    else:
        disc = a1 ** 2 - 4.0 * a2 * (a0 - target_perf)
        if disc < 0.0:
            return PCAP_NOMINAL
        sq = np.sqrt(disc)
        roots = [(-a1 + sq) / (2.0 * a2), (-a1 - sq) / (2.0 * a2)]
        valid = [r for r in roots if PCAP_MIN <= r <= PCAP_MAX]
        if not valid:
            u = min(roots, key=lambda r: abs(r - PCAP_NOMINAL))
        else:
            # Pick the root closest to PCAP_NOMINAL (prefer operating in the
            # middle of the range rather than at extremes)
            u = min(valid, key=lambda r: abs(r - PCAP_NOMINAL))
    return float(np.clip(u, PCAP_MIN, PCAP_MAX))


def compute_pid_gains(coeffs_perf, nominal_pcap=PCAP_NOMINAL):
    """
    Derive PID gains from the performance polynomial
    Perf(u) = a2*u^2 + a1*u + a0.

    Static plant gain at the nominal operating point:
        K = dPerf/dPCAP |_{u=nominal} = 2*a2*nominal + a1   [Hz / W]

    Conservative IMC-inspired tuning:
        Kp = 0.5 / K    [W / Hz]   — half the inverse plant gain
        Ki = Kp / 10    [W / (Hz·s)] — integral time constant = 10 control periods
        Kd = 0

    Anti-windup is applied directly on the PCAP correction contributed by the
    integral term (see PIDController.step), so the limit is always in Watts
    regardless of the performance scale.
    """
    a2, a1, _a0 = coeffs_perf
    K = 2.0 * a2 * nominal_pcap + a1     # [Hz / W]
    K = max(abs(K), 1e-3)                 # guard against near-zero gain
    Kp = 0.5 / K
    Ki = Kp / 10.0
    return Kp, Ki


def snap_to_actions(pcap_continuous):
    """Return the nearest discrete PCAP from the 16-value action set."""
    return min(ACTIONS, key=lambda a: abs(a - pcap_continuous))


# ──────────────────────────────────────────────────────────────────────────────
# PID controller
# ──────────────────────────────────────────────────────────────────────────────

class PIDController:
    """
    Discrete-time PID with per-term anti-windup.

    Units:
      setpoint / measured  : Hz  (progress rate)
      correction output    : W   (PCAP adjustment on top of feedforward)

    Anti-windup is enforced on the integral's PCAP contribution (Ki * integral),
    not on the raw integral accumulator, so the limit is always in Watts and
    remains meaningful regardless of how performance is scaled.
    """

    def __init__(self, Kp, Ki, Kd, setpoint, Ts=CONTROL_PERIOD,
                 integral_pcap_limit=40.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint        # Hz
        self.Ts = Ts
        self.integral_pcap_limit = integral_pcap_limit   # max |Ki * integral| in W
        self._integral = 0.0            # accumulated error * time  [Hz·s]
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, measured):
        """
        Compute PCAP correction for the current measurement.

        Returns (correction [W], error [Hz]) where:
          correction > 0  →  raise PCAP  (performance too low, need more power)
          correction < 0  →  lower PCAP  (performance too high, can save power)
        """
        error = self.setpoint - measured          # Hz
        self._integral += error * self.Ts         # Hz·s

        # Anti-windup: clamp integral term's PCAP contribution, then back-compute
        # the allowed integral accumulation so the integrator doesn't drift further.
        i_contribution = self.Ki * self._integral
        i_clamped = float(np.clip(i_contribution,
                                  -self.integral_pcap_limit,
                                  self.integral_pcap_limit))
        # Conditionally integrate: only accumulate if not saturated, or if the
        # new error would drive the integrator back inside the limits.
        if abs(i_contribution) > self.integral_pcap_limit:
            if np.sign(error) != np.sign(i_contribution):
                # error is pushing the integrator back → allow accumulation
                pass
            else:
                # saturated and error keeps pushing → freeze the integrator
                self._integral -= error * self.Ts

        derivative = (error - self._prev_error) / self.Ts
        self._prev_error = error

        correction = (self.Kp * error
                      + i_clamped
                      + self.Kd * derivative)
        return correction, error


# ──────────────────────────────────────────────────────────────────────────────
# NRM client
# ──────────────────────────────────────────────────────────────────────────────
client = nrm.Client()
actuators = client.list_actuators()

# ──────────────────────────────────────────────────────────────────────────────
# Measurement helpers (identical to RL controller)
# ──────────────────────────────────────────────────────────────────────────────

def initialize_state_dict():
    return {
        'progress': [],
        'energy_0': [], 'energy_1': [],
        'PAPI_L3_TCA': [], 'PAPI_TOT_INS': [], 'PAPI_TOT_CYC': [],
        'PAPI_RES_STL': [], 'PAPI_L3_TCM': [],
        'measured_power_0': [], 'measured_power_1': [],
    }


_empty_ref = initialize_state_dict()


def measure_progress(progress_data):
    """Compute median progress rate in Hz from a list of [timestamp, value] pairs."""
    freq = [
        progress_data[k][1] / (progress_data[k][0] - progress_data[k - 1][0])
        for k in range(1, len(progress_data))
    ]
    return float(np.nanmedian([0.0] + freq))


def measure_power(P0, P1):
    """Average measured power over both CPU sockets (telemetry only)."""
    pwr0 = [P0[i][1] for i in range(1, len(P0))]
    pwr1 = [P1[i][1] for i in range(1, len(P1))]
    min_len = min(len(pwr0), len(pwr1))
    avg = [(pwr0[i] + pwr1[i]) / 2.0 for i in range(min_len)]
    return float(np.mean(avg)) if avg else float('nan')


def compress_files(timestamp_str, exp_dir):
    tar_file = os.path.join(exp_dir, f'compressed_iteration_{timestamp_str}.tar')
    with tarfile.open(tar_file, 'w:gz') as tarf:
        for root, _dirs, files in os.walk(exp_dir):
            if root != exp_dir:
                continue
            for fname in files:
                if fname.endswith(('.csv', '.yaml', '.log')):
                    fpath = os.path.join(root, fname)
                    if os.path.exists(fpath):
                        tarf.add(fpath, arcname=fname)
                        os.remove(fpath)
    print(f'Compressed into {tar_file}')


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark launch command (mirrors mild_RL_controller.py)
# ──────────────────────────────────────────────────────────────────────────────
_PAPI = '-e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM'


def build_command(application):
    if 'stream' in application:
        N, IT = 33554432, 10000
    else:
        N, IT = 26, 1000

    wrap = f'time nrm-papiwrapper -i {_PAPI} --'
    if 'ones-npb-ft'  in application: return f'{wrap} {application} 500'
    if 'ones-npb-mg'  in application: return f'{wrap} {application} 1000'
    if 'ones-npb-bt'  in application: return f'{wrap} {application} 1000'
    if 'ones-npb-cg'  in application: return f'{wrap} {application} 10000'
    if 'ones-npb-is'  in application: return f'{wrap} {application} 26 1000'
    if 'phases'       in application: return f'{wrap} {application} {N} 5 1000'
    if 'solvers'      in application: return f'{wrap} {application} {N} poor 0 {IT}'
    return f'{wrap} {application} {N} {IT}'


# ──────────────────────────────────────────────────────────────────────────────
# Core experiment loop
# ──────────────────────────────────────────────────────────────────────────────

def experiment_for(application, exp_dir, pid: PIDController, u_ff: float):
    """Run one benchmark iteration under PID performance control."""
    global _active_process

    state_dict = initialize_state_dict()
    first_progress = [False]
    pid.reset()
    current_pcap = snap_to_actions(u_ff)

    with (open(f'{exp_dir}/{application}_output.log',   'w')            as log_file,
          open(f'{exp_dir}/measured_power.csv', 'w', newline='')        as power_file,
          open(f'{exp_dir}/progress.csv',       'w', newline='')        as progress_file,
          open(f'{exp_dir}/energy.csv',          'w', newline='')        as energy_file,
          open(f'{exp_dir}/PCAP_file.csv',       'w', newline='')        as PCAP_file,
          open(f'{exp_dir}/papi.csv',            'w', newline='')        as papi_file,
          open(f'{exp_dir}/pid_log.csv',         'w', newline='')        as pid_log_file):

        pw = csv.writer(power_file);    pw.writerow(['time', 'scope', 'value'])
        pr = csv.writer(progress_file); pr.writerow(['time', 'value'])
        en = csv.writer(energy_file);   en.writerow(['time', 'scope', 'value'])
        pc = csv.writer(PCAP_file);     pc.writerow(['time', 'actuator', 'value'])
        pa = csv.writer(papi_file);     pa.writerow(['time', 'scope', 'value'])
        pl = csv.writer(pid_log_file)
        pl.writerow(['time', 'target_perf_Hz', 'measured_perf_Hz', 'error_Hz',
                     'u_ff_W', 'correction_W', 'pcap_cmd_W',
                     'measured_power_W', 'Kp', 'Ki', 'Kd'])

        def cb(*args_):
            (sensor, time_ns, scope, value) = args_
            scope_uuid = scope.get_uuid()
            sensor = sensor.decode("UTF-8")
            ts = time_ns / 1e9

            if sensor == "nrm.benchmarks.progress":
                first_progress[0] = True
                pr.writerow([ts, value])
                state_dict['progress'].append([ts, value])
            elif first_progress[0]:
                if sensor == "nrm.geopm.CPU_POWER":
                    pw.writerow([ts, scope_uuid[-1], value])
                    state_dict[f'measured_power_{scope_uuid[-1]}'].append([ts, value])
                elif sensor == "nrm.geopm.CPU_ENERGY":
                    en.writerow([ts, scope_uuid[-1], value])
                    state_dict[f'energy_{scope_uuid[-1]}'].append((ts, value))
                elif "PAPI" in sensor:
                    pa.writerow([ts, sensor, value])
                    state_dict[sensor.split('.')[3]].append((ts, value))

        client.set_event_listener(cb)
        client.start_event_listener("")

        my_env = os.environ.copy()
        my_env["OMP_NUM_THREADS"] = "95"
        process = subprocess.Popen(
            ['bash', '-c', build_command(application)],
            stdout=log_file, stderr=log_file, env=my_env
        )
        _active_process = process
        time.sleep(0.5)

        last_control_t = 0.0

        try:
            while True:
                now = time.time()

                if now - last_control_t >= CONTROL_PERIOD:
                    # Need at least 2 progress events to compute a frequency
                    has_data = (
                        state_dict != _empty_ref
                        and len(state_dict['progress']) >= 2
                    )

                    if has_data:
                        try:
                            measured_perf = measure_progress(state_dict['progress'])

                            # Power is telemetry only
                            measured_power = (
                                measure_power(state_dict['measured_power_0'],
                                              state_dict['measured_power_1'])
                                if (len(state_dict['measured_power_0']) >= 2 and
                                    len(state_dict['measured_power_1']) >= 2)
                                else float('nan')
                            )

                            correction, error = pid.step(measured_perf)

                            pcap_cont    = float(np.clip(u_ff + correction,
                                                         PCAP_MIN, PCAP_MAX))
                            current_pcap = snap_to_actions(pcap_cont)

                            pl.writerow([now, pid.setpoint, measured_perf, error,
                                         u_ff, correction, current_pcap,
                                         measured_power, pid.Kp, pid.Ki, pid.Kd])
                            print(
                                f"[PID] perf={measured_perf:.3f}Hz  "
                                f"target={pid.setpoint:.3f}Hz  "
                                f"err={error:+.3f}Hz  corr={correction:+.2f}W  "
                                f"u_ff={u_ff:.1f}W  PCAP→{current_pcap:.0f}W  "
                                f"power={measured_power:.1f}W"
                            )
                        except Exception as exc:
                            print(f"[WARN] PID step failed ({exc}); "
                                  f"holding PCAP={current_pcap:.0f}W")
                    else:
                        current_pcap = snap_to_actions(u_ff)
                        print(f"[PID] Awaiting data — "
                              f"feedforward PCAP={current_pcap:.0f}W")

                    client.actuate(actuators[0], current_pcap)
                    pc.writerow([time.time(), actuators[0], current_pcap])
                    last_control_t = now
                    state_dict = initialize_state_dict()

                if process.poll() is not None:
                    print("Process completed.")
                    break

        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            for fh in [power_file, progress_file, energy_file,
                       PCAP_file, papi_file, pid_log_file]:
                fh.flush()

    compress_files(datetime.now().strftime("%Y%m%d_%H%M%S"), exp_dir)
    print("----------------------------------")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    coeff_dir   = os.path.join(current_dir, '..')

    for step in range(args.num_steps):
        print(f"\n>>> Step {step + 1}/{args.num_steps}")
        for APPLICATION in APPLICATIONS:
            app_short = APPLICATION.split('/')[-1]

            # ── Load static characteristics ──────────────────────────────────
            try:
                static_data = load_static_coeffs(app_short, coeff_dir)
            except FileNotFoundError as exc:
                print(f"[ERROR] {exc}  — skipping {APPLICATION}")
                continue

            coeffs_perf = static_data['coefficients_perf']
            coeffs_pow  = static_data['coefficients_pow']   # telemetry reference

            # ── Resolve target performance ───────────────────────────────────
            if args.target_perf is not None:
                target_perf = args.target_perf
            else:
                _pcap_opt, _perf_max = max_perf_pcap(coeffs_perf)
                target_perf = 0.80 * _perf_max
                print(f"  [auto target] max perf @ PCAP={_pcap_opt:.1f}W "
                      f"= {_perf_max:.3f}Hz  →  80% target = {target_perf:.3f}Hz")

            # ── PID gains from performance polynomial ────────────────────────
            Kp, Ki = compute_pid_gains(coeffs_perf)
            if args.kp is not None: Kp = args.kp
            if args.ki is not None: Ki = args.ki
            Kd = args.kd

            # ── Feedforward: nominal PCAP for the performance target ─────────
            u_ff   = feedforward_pcap(coeffs_perf, target_perf)
            u_ff_s = snap_to_actions(u_ff)
            perf_check = perf_at_pcap(coeffs_perf, u_ff)

            print(f"\n[{APPLICATION}]")
            print(f"  target perf      : {target_perf:.4f} Hz")
            print(f"  coefficients_perf: {coeffs_perf}")
            print(f"  feedforward PCAP : {u_ff:.2f} W  →  {u_ff_s:.0f} W (snapped)")
            print(f"  Perf(u_ff) check : {perf_check:.4f} Hz  (should ≈ target)")
            print(f"  PID gains        : Kp={Kp:.4f} W/Hz  Ki={Ki:.5f} W/(Hz·s)  Kd={Kd:.4f}")

            pid = PIDController(
                Kp=Kp, Ki=Ki, Kd=Kd,
                setpoint=target_perf,
                Ts=CONTROL_PERIOD,
            )

            exp_dir = os.path.join(current_dir, '..', 'experiment_data',
                                   args.exp_name, APPLICATION)
            os.makedirs(exp_dir, exist_ok=True)

            experiment_for(APPLICATION, exp_dir, pid, u_ff)
