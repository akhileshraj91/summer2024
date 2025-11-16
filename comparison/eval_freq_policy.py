import time
import numpy as np
import sys
import csv
import subprocess
import os
import tarfile
from datetime import datetime
import torch
import argparse
import nrm

"""
Runtime evaluation script that uses the trained continuous-frequency policy
(`freq_policy.pth`) and scaler (`freq_policy_scaler.npz`) to read runtime
telemetry, compute the state vector, predict a frequency (Hz), quantize to the
nearest supported actuator frequency and actuate the CPU frequency actuator.

This follows the same telemetry collection style used to generate the training
data: it listens to NRM events, accumulates short windows of samples, and at
each control interval computes rates and acts.
"""

# Supported actuator frequency grid (Hz) used for quantization
ACTIONS_FREQ = np.array([
    1000000000.0, 1100000000.0, 1200000000.0, 1300000000.0, 1400000000.0,
    1500000000.0, 1600000000.0, 1700000000.0, 1800000000.0, 1900000000.0,
    2000000000.0, 2100000000.0, 2200000000.0, 2300000000.0, 2400000000.0,
    2500000000.0, 2600000000.0, 2700000000.0, 2800000000.0, 2900000000.0,
    3000000000.0, 3100000000.0, 3200000000.0, 3300000000.0, 3400000000.0,
    3500000000.0, 3600000000.0, 3700000000.0, 3800000000.0, 4000000000.0,
])


def load_policy(model_path, scaler_path):
    data = np.load(scaler_path)
    input_mean = data['input_mean']
    input_std = data['input_std']
    target_scale = float(data['target_scale'])

    # create model instance matching train_models.FCNetwork
    from train_models import FCNetwork
    input_dim = int(input_mean.shape[0])
    model = FCNetwork(input_dim=input_dim, layers=[64, 64])
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model, input_mean, input_std, target_scale


def quantize_to_grid(hz):
    idx = np.argmin(np.abs(ACTIONS_FREQ - hz))
    return float(ACTIONS_FREQ[idx])


def compute_rate_from_cumulative(series):
    """Given a list of (timestamp, value) cumulative samples, compute rate = delta / dt.
    Returns np.nan if insufficient data.
    """
    if series is None or len(series) < 2:
        return np.nan
    try:
        t0, v0 = series[0]
        t1, v1 = series[-1]
        dt = float(t1) - float(t0)
        if dt <= 0:
            return np.nan
        return float(v1 - v0) / dt
    except Exception:
        return np.nan


def compute_power_from_energy(energy_list):
    # energy_list: list of (timestamp, energy)
    if energy_list is None or len(energy_list) < 2:
        return np.nan
    vals = []
    for i in range(1, len(energy_list)):
        t0, e0 = energy_list[i-1]
        t1, e1 = energy_list[i]
        dt = t1 - t0
        if dt <= 0:
            continue
        # handle wraparound (best-effort); assume large wrap value if needed
        diff = e1 - e0
        if diff < 0:
            # guess a large wrap value (shouldn't happen often)
            diff = e1
        vals.append(diff / dt)
    return float(np.nanmean(vals)) if vals else np.nan


def process_state(state_dict):
    # Compute rates from collected series
    ins_rate = compute_rate_from_cumulative(state_dict.get('PAPI_TOT_INS', []))
    cyc_rate = compute_rate_from_cumulative(state_dict.get('PAPI_TOT_CYC', []))
    l3m_rate = compute_rate_from_cumulative(state_dict.get('PAPI_L3_TCM', []))
    res_rate = compute_rate_from_cumulative(state_dict.get('PAPI_RES_STL', []))

    # compute power from energy or measured_power arrays
    power0 = compute_power_from_energy(state_dict.get('energy_0', []))
    power1 = compute_power_from_energy(state_dict.get('energy_1', []))
    # if energy not present, try measured_power lists which are (timestamp,value)
    if np.isnan(power0) and 'measured_power_0' in state_dict:
        power0 = np.nanmean([v for (_, v) in state_dict.get('measured_power_0', [])])
    if np.isnan(power1) and 'measured_power_1' in state_dict:
        power1 = np.nanmean([v for (_, v) in state_dict.get('measured_power_1', [])])
    power = np.nanmean([p for p in (power0, power1) if not np.isnan(p)])

    # Build state vector in the same order used for training
    # preferred_state_cols = ["INS_RATE", "CYC_RATE", "L3M_RATE", "RES_STL_RATE", "Power"]
    state_vec = np.array([ins_rate, cyc_rate, l3m_rate, res_rate, power], dtype=np.float32)
    return state_vec


def initialize_state_dict():
    sd = {}
    sd['progress'] = []
    sd['energy_0'] = []
    sd['energy_1'] = []
    sd['PAPI_L3_TCA'] = []
    sd['PAPI_TOT_INS'] = []
    sd['PAPI_TOT_CYC'] = []
    sd['PAPI_RES_STL'] = []
    sd['PAPI_L3_TCM'] = []
    sd['measured_power_0'] = []
    sd['measured_power_1'] = []
    return sd


def compress_files(EXP_DIR, iteration):
    tar_file = os.path.join(EXP_DIR, f'compressed_iteration_{iteration}.tar.gz')
    with tarfile.open(tar_file, 'w:gz') as tarf:
        for root, dirs, files in os.walk(EXP_DIR):
            for file in files:
                if file.endswith('.csv') or file.endswith('.yaml') or file.endswith('.log'):
                    file_path = os.path.join(root, file)
                    tarf.add(file_path, arcname=os.path.basename(file_path))
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
    print(f'Compressed files into {tar_file}')


def find_freq_actuator(actuators):
    # Prefer actuator with uuid containing 'cpu.freq' else fallback to index 1 or 0
    for a in actuators:
        try:
            s = str(a)
            if 'freq' in s or 'cpu.freq' in s:
                return a
        except Exception:
            continue
    if len(actuators) > 1:
        return actuators[1]
    return actuators[0]


def experiment_for(APPLICATION, EXP_DIR, ACTION=None):
    """Return sensible default PROBLEM_SIZE and ITERATIONS for known applications.
    Caller may override these with CLI args. Returns (problem_size, iterations) as ints.
    """
    # Defaults chosen from common patterns in this workspace
    if "stream" in APPLICATION:
        return 33554432, 10000
    if "npb" in APPLICATION or "ones-npb" in APPLICATION:
        # many NPB entries use small integer problem sizes; choose conservative defaults
        return 26, 1000
    if "solvers" in APPLICATION:
        return 1000, 10
    if "phases" in APPLICATION:
        return 5, 1000
    # generic default
    return 0, 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='freq_policy.pth')
    p.add_argument('--scaler', default='freq_policy_scaler.npz')
    p.add_argument('-a', '--application', nargs='+', help='Applications to run')
    p.add_argument('--problem-size', dest='problem_size', type=int, default=None, help='Problem size argument to pass to the application')
    p.add_argument('--iterations', dest='iterations', type=int, default=None, help='Iteration count argument to pass to the application')
    p.add_argument('--experiment-dir', default='experiment_data/eval', help='Base output directory')
    args = p.parse_args()

    apps = args.application or []
    if not apps:
        print('No applications supplied; use -a APPNAME')
        return

    # load policy
    model, input_mean, input_std, target_scale = load_policy(args.model, args.scaler)

    client = nrm.Client()
    actuators = client.list_actuators()
    freq_act = find_freq_actuator(actuators)

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    for APPLICATION in apps:
        # Use script directory as base (match data_gen behaviour) so EXP_DIR is predictable
        EXP_DIR = os.path.join(current_dir, args.experiment_dir, APPLICATION)
        os.makedirs(EXP_DIR, exist_ok=True)
        # open CSV writers
        with open(f'{EXP_DIR}/{APPLICATION}_output.log','w') as log_file, open(f'{EXP_DIR}/FREQ_file.csv', mode='w', newline='') as freq_file, open(f'{EXP_DIR}/measured_power.csv', mode='w', newline='') as power_file, open(f'{EXP_DIR}/progress.csv', mode='w', newline='') as progress_file, open(f'{EXP_DIR}/energy.csv', mode='w', newline='') as energy_file, open(f'{EXP_DIR}/papi.csv', mode='w', newline='') as papi_file:
            freq_writer = csv.writer(freq_file)
            progress_writer = csv.writer(progress_file)
            energy_writer = csv.writer(energy_file)
            power_writer = csv.writer(power_file)
            papi_writer = csv.writer(papi_file)
            power_writer.writerow(['time','scope','value'])
            progress_writer.writerow(['time','value'])
            energy_writer.writerow(['time','scope','value'])
            freq_writer.writerow(['time','actuator','value'])
            papi_writer.writerow(['time','scope','value'])
            # flush headers immediately so an early crash doesn't leave empty files
            try:
                power_file.flush(); os.fsync(power_file.fileno())
            except Exception:
                pass
            try:
                progress_file.flush(); os.fsync(progress_file.fileno())
            except Exception:
                pass
            try:
                energy_file.flush(); os.fsync(energy_file.fileno())
            except Exception:
                pass
            try:
                freq_file.flush(); os.fsync(freq_file.fileno())
            except Exception:
                pass
            try:
                papi_file.flush(); os.fsync(papi_file.fileno())
            except Exception:
                pass

            state_dict = initialize_state_dict()

            def cb(*args):
                # print(args)
                (sensor, time_ns, scope, value) = args
                sensor = sensor.decode('UTF-8') if isinstance(sensor, bytes) else str(sensor)
                timestamp = float(time_ns) / 1e9
                # push into state_dict based on sensor
                if sensor == 'nrm.benchmarks.progress':
                    progress_writer.writerow([timestamp, value])
                    state_dict['progress'].append((timestamp, value))
                    try:
                        progress_file.flush(); os.fsync(progress_file.fileno())
                    except Exception:
                        pass
                elif sensor == 'nrm.geopm.CPU_ENERGY':
                    # write energy row and flush so it's persisted
                    scope_uuid = scope.get_uuid() if hasattr(scope, 'get_uuid') else scope
                    energy_writer.writerow([timestamp, scope_uuid, value])
                    try:
                        energy_file.flush(); os.fsync(energy_file.fileno())
                    except Exception:
                        pass
                    # scope may be complex; extract trailing integer CPU index robustly
                    try:
                        su = str(scope_uuid)
                        import re
                        m = re.search(r"(\d+)$", su)
                        cpu_idx = int(m.group(1)) if m else 0
                    except Exception:
                        cpu_idx = 0
                    state_dict.setdefault(f'energy_{cpu_idx}', []).append((timestamp, value))
                elif sensor == 'nrm.geopm.CPU_POWER':
                    scope_uuid = scope.get_uuid() if hasattr(scope, 'get_uuid') else scope
                    power_writer.writerow([timestamp, scope_uuid, value])
                    try:
                        power_file.flush(); os.fsync(power_file.fileno())
                    except Exception:
                        pass
                    try:
                        su = str(scope_uuid)
                        import re
                        m = re.search(r"(\d+)$", su)
                        cpu_idx = int(m.group(1)) if m else 0
                    except Exception:
                        cpu_idx = 0
                    state_dict.setdefault(f'measured_power_{cpu_idx}', []).append((timestamp, value))
                elif 'PAPI' in sensor:
                    papi_writer.writerow([timestamp, sensor, value])
                    parts = sensor.split('.')
                    if len(parts) > 3:
                        key = parts[3]
                        state_dict.setdefault(key, []).append((timestamp, value))

            client.set_event_listener(cb)
            client.start_event_listener("")

            # start application under nrm-papiwrapper similar to training data generator
            # Launch the parsed APPLICATION under nrm-papiwrapper and redirect output to log_file
            # Determine default problem size and iterations for this application, allow CLI override
            default_ps, default_it = experiment_for(APPLICATION, EXP_DIR)
            PROBLEM_SIZE = str(args.problem_size) if args.problem_size is not None else str(default_ps)
            ITERATIONS = str(args.iterations) if args.iterations is not None else str(default_it)
            process = None
            if "solvers" in APPLICATION:
                cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} {PROBLEM_SIZE} poor 0 {ITERATIONS}'
                print(f'Launching: {cmd}')
                process = subprocess.Popen(['bash', '-c', cmd], stdout=log_file, stderr=log_file)
            elif "phases" in APPLICATION:
                print(f"Starting Execution of phases {APPLICATION, PROBLEM_SIZE, ITERATIONS}")
                process = subprocess.Popen(
                    ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} {PROBLEM_SIZE} 5 1000'],
                    stdout=log_file,
                    stderr=log_file
                )
            elif "ones-npb-ft" in APPLICATION:
                process = subprocess.Popen(
                    ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 500'],
                    stdout=log_file,
                    stderr=log_file
                )
            elif "ones-npb-mg" in APPLICATION:
                process = subprocess.Popen(
                    ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 1000'],
                    stdout=log_file,
                    stderr=log_file
                )
            elif "ones-npb-bt" in APPLICATION:
                process = subprocess.Popen(
                    ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 1000'],
                    stdout=log_file,
                    stderr=log_file
                )
            elif "ones-npb-cg" in APPLICATION:
                process = subprocess.Popen(
                    ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 1000'],
                    stdout=log_file,
                    stderr=log_file
                )
            elif "ones-npb-is" in APPLICATION:
                process = subprocess.Popen(
                    ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 26 1000'],
                    stdout=log_file,
                    stderr=log_file
                )
            else:
                cmd = 'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {} {} {}'.format(APPLICATION, PROBLEM_SIZE, ITERATIONS)
                print(f'Launching: {cmd}')
                process = subprocess.Popen(['bash', '-c', cmd], stdout=log_file, stderr=log_file)

            # give the wrapper a moment to start and detect immediate failure
            time.sleep(0.5)
            try:
                if process.poll() is not None:
                    print(f'Process exited immediately with code {process.returncode}')
                    # try to show the beginning of the log to diagnose
                    try:
                        log_path = log_file.name
                        with open(log_path, 'r') as lf:
                            lines = lf.readlines()
                        print('--- process log head ---')
                        for ln in lines[:50]:
                            print(ln.rstrip())
                        print('--- end log head ---')
                    except Exception as e:
                        print(f'Unable to read log file {log_file.name}: {e}')
            except Exception:
                pass

            last_actuate = 0.0
            try:
                while True:
                    now = time.time()
                    if now - last_actuate >= 2.0:  # control interval
                        # compute state and act if we have some telemetry
                        state_vec = process_state(state_dict)
                        if not np.any(np.isfinite(state_vec)):
                            # no telemetry yet
                            print('No telemetry yet, skipping actuation')
                        else:
                            # scale and predict
                            x = (state_vec - input_mean) / input_std
                            xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
                            with torch.no_grad():
                                pred_scaled = model(xt).cpu().numpy().squeeze()
                            pred_hz = float(pred_scaled * target_scale)
                            q_hz = quantize_to_grid(pred_hz)
                            print(f'Actuating frequency: pred={pred_hz:.1f} Hz, quantized={q_hz:.1f} Hz')
                            # actuate
                            try:
                                client.actuate(freq_act, q_hz)
                            except Exception as e:
                                print(f'Actuation failed: {e}')
                            freq_writer.writerow([time.time(), freq_act, q_hz])
                        last_actuate = now
                        # reset buffers for next window
                        state_dict = initialize_state_dict()
                    time.sleep(0.1)
                    # If the application process has finished, stop the controller for this app
                    try:
                        if process is not None and process.poll() is not None:
                            print(f'Application process exited with code {process.returncode}, stopping controller for {APPLICATION}')
                            # compress and rotate logs for this experiment run
                            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                            compress_files(EXP_DIR, current_time)
                            break
                    except Exception:
                        # ignore polling errors and continue
                        pass
            except KeyboardInterrupt:
                print('Interrupted, compressing logs')
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                compress_files(EXP_DIR, current_time)


if __name__ == '__main__':
    main()
