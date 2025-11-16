# train_pcap_policy_pth.py
import argparse
import torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np
import sys

# ---------------- Model: 4 -> 16 (matches pcap_policy.py) ----------------
class FCNetwork(nn.Module):
    """
    Fully-connected network that maps state -> scalar action (frequency in Hz).
    Output is a single continuous value (no final activation).
    """
    def __init__(self, input_dim, layers=[32, 32]):
        super().__init__()
        dim_input = input_dim
        dim_output = 1
        net = []
        d = dim_input
        for h in layers:
            net += [nn.Linear(d, h), nn.ReLU()]
            d = h
        net += [nn.Linear(d, dim_output)]
        self.network = nn.Sequential(*net)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return self.network(x).squeeze(-1)

def main():
    parser = argparse.ArgumentParser(description="Train pcap policy (supervised).")
    parser.add_argument("csv", nargs="?", default="data.csv", help="input CSV path")
    parser.add_argument("--max-action", dest="max_action", action="store_true",
                        help="Train policy to always return the maximum action")
    args = parser.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)

    # Choose state columns (fall back to best-known names if missing)
    preferred_state_cols = ["INS_RATE", "CYC_RATE", "L3M_RATE", "RES_STL_RATE", "Power"]
    state_cols = [c for c in preferred_state_cols if c in df.columns]
    if len(state_cols) < 3:
        raise ValueError(f"Not enough state columns found in CSV; found: {list(df.columns)}")

    if "Action" not in df.columns:
        raise ValueError("CSV must contain 'Action' column with the observed frequency action (Hz)")

    # Input matrix
    X = df[state_cols].to_numpy(dtype=np.float32)

    # Reward: prefer explicit Reward, else try to derive from Next_TOT_INS_PER_CYC
    if "Reward" in df.columns:
        reward = df["Reward"].to_numpy(dtype=np.float32)
    elif "Next_TOT_INS_PER_CYC" in df.columns:
        next_ipc = df["Next_TOT_INS_PER_CYC"].to_numpy(dtype=np.float32)
        reward = 0.5 * (next_ipc ** 3)
    else:
        # if missing, fall back to uniform weights
        reward = np.zeros(len(df), dtype=np.float32)

    # Targets: continuous frequency actions (Hz)
    actions = df["Action"].to_numpy(dtype=np.float32)

    # Optionally force max-action mode: choose the maximum observed action as target
    if args.max_action:
        max_act = float(np.nanmax(actions))
        print(f"Training mode: --max-action — forcing target to action {max_act}")
        y = np.full_like(actions, max_act, dtype=np.float32)
    else:
        y = actions

    # Build sample weights from reward (normalize to [0,1] then add 1.0 to keep baseline)
    reward_safe = np.nan_to_num(reward, nan=0.0)
    # normalize rewards to [0,1] using percentile clipping to reduce outliers
    if reward_safe.size == 0:
        weights = np.ones_like(reward_safe, dtype=np.float32)
    else:
        # clip to [p1, p99] to avoid extreme outliers
        p1 = np.percentile(reward_safe, 1)
        p99 = np.percentile(reward_safe, 99)
        if p99 <= p1:
            reward_clipped = np.clip(reward_safe, None, None)
        else:
            reward_clipped = np.clip(reward_safe, p1, p99)
        # min-max scale to [0,1]
        rr_min = float(np.min(reward_clipped))
        rr_max = float(np.max(reward_clipped))
        if rr_max - rr_min <= 0:
            norm_r = np.zeros_like(reward_clipped, dtype=np.float32)
        else:
            norm_r = (reward_clipped - rr_min) / (rr_max - rr_min)
        # base weight 1.0 plus normalized reward
        weights = 1.0 + norm_r.astype(np.float32)

    # ---------------- Train with scaling ----------------
    # Feature scaling: compute mean/std on training data and save scaler for inference.
    X_np = np.array(X, dtype=np.float32)
    Y_np = np.array(y, dtype=np.float32)

    # compute input mean/std
    input_mean = np.nanmean(X_np, axis=0)
    input_std = np.nanstd(X_np, axis=0)
    input_std[input_std == 0] = 1.0

    # scale targets (Hz -> GHz) to bring values into ~[1,4] range
    target_scale = 1e9
    Y_scaled = Y_np / target_scale

    # prepare tensors and drop rows with NaNs
    X_scaled = (X_np - input_mean) / input_std
    valid_mask = np.isfinite(X_scaled).all(axis=1) & np.isfinite(Y_scaled)
    X_scaled = X_scaled[valid_mask]
    Y_scaled = Y_scaled[valid_mask]
    weights = weights[valid_mask]

    X_t = torch.from_numpy(X_scaled)
    y_t = torch.from_numpy(Y_scaled)
    w_t = torch.from_numpy(weights)

    input_dim = X_t.shape[1]
    model = FCNetwork(input_dim=input_dim, layers=[64, 64])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction='none')

    for epoch in range(10000):
        model.train()
        opt.zero_grad()
        pred = model(X_t)
        loss_per_sample = loss_fn(pred, y_t)
        weighted = (loss_per_sample * w_t).mean()
        weighted.backward()
        opt.step()
        if epoch % 25 == 0:
            print(f"epoch {epoch:3d} | weighted loss {weighted.item():.4e}")

    # Save model and scalers
    out_path = "freq_policy.pth"
    torch.save(model.state_dict(), out_path)
    np.savez("freq_policy_scaler.npz", input_mean=input_mean, input_std=input_std, target_scale=target_scale)
    print(f"Saved -> {out_path}")
    print("Saved scaler -> freq_policy_scaler.npz")


if __name__ == '__main__':
    main()
