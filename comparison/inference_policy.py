import argparse
import torch
import numpy as np
import pandas as pd
import os

from train_models import FCNetwork

# Frequency action grid (supported actuator frequencies in Hz)
ACTIONS_FREQ = np.array([
    1000000000.0, 1100000000.0, 1200000000.0, 1300000000.0, 1400000000.0,
    1500000000.0, 1600000000.0, 1700000000.0, 1800000000.0, 1900000000.0,
    2000000000.0, 2100000000.0, 2200000000.0, 2300000000.0, 2400000000.0,
    2500000000.0, 2600000000.0, 2700000000.0, 2800000000.0, 2900000000.0,
    3000000000.0, 3100000000.0, 3200000000.0, 3300000000.0, 3400000000.0,
    3500000000.0, 3600000000.0, 3700000000.0, 3800000000.0, 4000000000.0,
])


def load_model(model_path, scaler_path):
    data = np.load(scaler_path)
    input_mean = data['input_mean']
    input_std = data['input_std']
    target_scale = float(data['target_scale'])

    input_dim = input_mean.shape[0]
    model = FCNetwork(input_dim=input_dim, layers=[64, 64])
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model, input_mean, input_std, target_scale


def predict_and_quantize(model, input_mean, input_std, target_scale, state_vec):
    # state_vec: 1d numpy array
    x = (state_vec - input_mean) / input_std
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(xt).cpu().numpy().squeeze()
    pred_hz = float(pred_scaled * target_scale)
    # quantize to nearest supported frequency
    idx = np.argmin(np.abs(ACTIONS_FREQ - pred_hz))
    return pred_hz, float(ACTIONS_FREQ[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='freq_policy.pth')
    p.add_argument('--scaler', default='freq_policy_scaler.npz')
    p.add_argument('--csv', default=None, help='CSV file with state columns or single row')
    p.add_argument('--cols', default=None, help='Comma-separated state column names to pick from CSV')
    args = p.parse_args()

    if not os.path.exists(args.model) or not os.path.exists(args.scaler):
        raise SystemExit('Model or scaler file not found; run training first')

    model, input_mean, input_std, target_scale = load_model(args.model, args.scaler)

    if args.csv is None:
        # interactive: read values from stdin
        print('Enter state values as comma-separated numbers:')
        line = input().strip()
        vals = np.array([float(x) for x in line.split(',')], dtype=np.float32)
        pred_hz, q_hz = predict_and_quantize(model, input_mean, input_std, target_scale, vals)
        print(f'predicted (Hz): {pred_hz}, quantized (Hz): {q_hz}')
        return

    df = pd.read_csv(args.csv)
    if args.cols is None:
        raise SystemExit('Please provide --cols to indicate state column names present in CSV')
    cols = [c.strip() for c in args.cols.split(',')]
    for c in cols:
        if c not in df.columns:
            raise SystemExit(f"Column {c} not found in CSV")

    out_rows = []
    for _, r in df.iterrows():
        state_vec = r[cols].to_numpy(dtype=np.float32)
        pred_hz, q_hz = predict_and_quantize(model, input_mean, input_std, target_scale, state_vec)
        out_rows.append({'pred_hz': pred_hz, 'quantized_hz': q_hz})

    out_df = pd.DataFrame(out_rows)
    print(out_df.head())


if __name__ == '__main__':
    main()
