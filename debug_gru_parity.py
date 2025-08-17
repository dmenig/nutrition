import os
import numpy as np
import torch
import pandas as pd

from app.np_infer import load_numpy_weights


def load_norm_features() -> pd.DataFrame:
    from build_features import main as build_features_main
    df = build_features_main(
        journal_path="data/processed_journal.csv",
        variables_path="data/variables.csv",
    )
    nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]
    for col in nutrition_cols + ["pds", "sport"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Load normalization
    params_path = os.path.join("models", "best_params.json")
    import json
    with open(params_path, "r") as f:
        params = json.load(f)
    norm = params.get("normalization", {})
    norm_df = df.copy()
    for col in nutrition_cols + ["sport"]:
        mean = norm.get(col, {}).get("mean", 0.0)
        std = norm.get(col, {}).get("std", 1.0) or 1.0
        norm_df[col] = (norm_df[col] - mean) / std
    return norm_df[nutrition_cols]


def torch_gru_layer_outputs(x_seq: np.ndarray, sd: dict, layer_idx: int) -> np.ndarray:
    # x_seq: [T, input_dim]
    x = torch.tensor(x_seq, dtype=torch.float32)
    W_ih = torch.tensor(sd[f"gru.weight_ih_l{layer_idx}"], dtype=torch.float32)
    W_hh = torch.tensor(sd[f"gru.weight_hh_l{layer_idx}"], dtype=torch.float32)
    b_ih = torch.tensor(sd.get(f"gru.bias_ih_l{layer_idx}"), dtype=torch.float32)
    b_hh = torch.tensor(sd.get(f"gru.bias_hh_l{layer_idx}"), dtype=torch.float32)
    H = W_hh.shape[1]
    W_ir, W_iz, W_in = torch.split(W_ih, H, dim=0)
    W_hr, W_hz, W_hn = torch.split(W_hh, H, dim=0)
    b_ir, b_iz, b_in = torch.split(b_ih, H, dim=0)
    b_hr, b_hz, b_hn = torch.split(b_hh, H, dim=0)
    h_t = torch.zeros((H,), dtype=torch.float32)
    outs = []
    for t in range(x.shape[0]):
        x_t = x[t]
        r = torch.sigmoid(x_t @ W_ir.T + b_ir + h_t @ W_hr.T + b_hr)
        z = torch.sigmoid(x_t @ W_iz.T + b_iz + h_t @ W_hz.T + b_hz)
        n = torch.tanh(x_t @ W_in.T + b_in + r * (h_t @ W_hn.T + b_hn))
        h_t = (1 - z) * n + z * h_t
        outs.append(h_t)
    return torch.stack(outs, dim=0).numpy()


def numpy_gru_layer_outputs(x_seq: np.ndarray, weights: dict, layer_idx: int) -> np.ndarray:
    H = weights[f"gru.weight_hh_l{layer_idx}"].shape[1]
    W_ih = weights[f"gru.weight_ih_l{layer_idx}"]
    W_hh = weights[f"gru.weight_hh_l{layer_idx}"]
    b_ih = weights.get(f"gru.bias_ih_l{layer_idx}")
    b_hh = weights.get(f"gru.bias_hh_l{layer_idx}")
    W_ir, W_iz, W_in = np.split(W_ih, 3, axis=0)
    W_hr, W_hz, W_hn = np.split(W_hh, 3, axis=0)
    b_ir = b_ih[0:H] if isinstance(b_ih, np.ndarray) else 0.0
    b_iz = b_ih[H:2 * H] if isinstance(b_ih, np.ndarray) else 0.0
    b_in = b_ih[2 * H:3 * H] if isinstance(b_ih, np.ndarray) else 0.0
    b_hr = b_hh[0:H] if isinstance(b_hh, np.ndarray) else 0.0
    b_hz = b_hh[H:2 * H] if isinstance(b_hh, np.ndarray) else 0.0
    b_hn = b_hh[2 * H:3 * H] if isinstance(b_hh, np.ndarray) else 0.0
    h_t = np.zeros((H,), dtype=np.float32)
    outs = []
    for t in range(x_seq.shape[0]):
        x_t = x_seq[t]
        r = 1.0 / (1.0 + np.exp(-(x_t @ W_ir.T + b_ir + h_t @ W_hr.T + b_hr)))
        z = 1.0 / (1.0 + np.exp(-(x_t @ W_iz.T + b_iz + h_t @ W_hz.T + b_hz)))
        n = np.tanh(x_t @ W_in.T + b_in + r * (h_t @ W_hn.T + b_hn))
        h_t = (1.0 - z) * n + z * h_t
        outs.append(h_t)
    return np.stack(outs, axis=0)


def numpy_gru_layer_outputs_with_order(x_seq: np.ndarray, weights: dict, layer_idx: int, order: str) -> np.ndarray:
    H = weights[f"gru.weight_hh_l{layer_idx}"].shape[1]
    W_ih = weights[f"gru.weight_ih_l{layer_idx}"]
    W_hh = weights[f"gru.weight_hh_l{layer_idx}"]
    b_ih = weights.get(f"gru.bias_ih_l{layer_idx}")
    b_hh = weights.get(f"gru.bias_hh_l{layer_idx}")
    # order can be 'rzn' or 'zrn'
    blocks_ih = np.split(W_ih, 3, axis=0)
    blocks_hh = np.split(W_hh, 3, axis=0)
    if order == 'rzn':
        W_ir, W_iz, W_in = blocks_ih
        W_hr, W_hz, W_hn = blocks_hh
        b_ir = b_ih[0:H] if isinstance(b_ih, np.ndarray) else 0.0
        b_iz = b_ih[H:2 * H] if isinstance(b_ih, np.ndarray) else 0.0
        b_in = b_ih[2 * H:3 * H] if isinstance(b_ih, np.ndarray) else 0.0
        b_hr = b_hh[0:H] if isinstance(b_hh, np.ndarray) else 0.0
        b_hz = b_hh[H:2 * H] if isinstance(b_hh, np.ndarray) else 0.0
        b_hn = b_hh[2 * H:3 * H] if isinstance(b_hh, np.ndarray) else 0.0
    else:  # 'zrn'
        W_iz, W_ir, W_in = blocks_ih
        W_hz, W_hr, W_hn = blocks_hh
        b_iz = b_ih[0:H] if isinstance(b_ih, np.ndarray) else 0.0
        b_ir = b_ih[H:2 * H] if isinstance(b_ih, np.ndarray) else 0.0
        b_in = b_ih[2 * H:3 * H] if isinstance(b_ih, np.ndarray) else 0.0
        b_hz = b_hh[0:H] if isinstance(b_hh, np.ndarray) else 0.0
        b_hr = b_hh[H:2 * H] if isinstance(b_hh, np.ndarray) else 0.0
        b_hn = b_hh[2 * H:3 * H] if isinstance(b_hh, np.ndarray) else 0.0
    h_t = np.zeros((H,), dtype=np.float32)
    outs = []
    for t in range(x_seq.shape[0]):
        x_t = x_seq[t]
        r = 1.0 / (1.0 + np.exp(-(x_t @ W_ir.T + b_ir + h_t @ W_hr.T + b_hr)))
        z = 1.0 / (1.0 + np.exp(-(x_t @ W_iz.T + b_iz + h_t @ W_hz.T + b_hz)))
        n = np.tanh(x_t @ W_in.T + b_in + r * (h_t @ W_hn.T + b_hn))
        h_t = (1.0 - z) * n + z * h_t
        outs.append(h_t)
    return np.stack(outs, axis=0)


def main():
    # inputs
    norm_feats = load_norm_features().values.astype("float32")
    x_seq = norm_feats[:10]  # first 10 steps
    # load weights
    sd = torch.load(os.path.join("models", "recurrent_model.pth"), map_location="cpu")
    npw = load_numpy_weights(os.path.join("models", "recurrent_model_np.npz"))
    # layer 0 compare
    t_out = torch_gru_layer_outputs(x_seq, sd, 0)
    n_out = numpy_gru_layer_outputs(x_seq, npw, 0)
    l2 = float(np.linalg.norm(t_out - n_out))
    mx = float(np.max(np.abs(t_out - n_out)))
    print(f"Layer0 hidden parity: l2={l2:.6f} max_abs={mx:.6f}")
    print("t_out[0,:5]", t_out[0][:5])
    print("n_out[0,:5]", n_out[0][:5])
    # order sweep on layer 0
    n_out_rzn = numpy_gru_layer_outputs_with_order(x_seq, npw, 0, 'rzn')
    n_out_zrn = numpy_gru_layer_outputs_with_order(x_seq, npw, 0, 'zrn')
    l2_rzn = float(np.linalg.norm(t_out - n_out_rzn))
    l2_zrn = float(np.linalg.norm(t_out - n_out_zrn))
    print(f"Layer0 order l2: rzn={l2_rzn:.6f} zrn={l2_zrn:.6f}")
    # print best first row sample
    best = 'rzn' if l2_rzn < l2_zrn else 'zrn'
    print("Best order:", best)
    n_out_best = n_out_rzn if best=='rzn' else n_out_zrn
    print("t_out[0,:5]", t_out[0][:5])
    print("n_best[0,:5]", n_out_best[0][:5])
    # If 2 layers, compare layer 1 by feeding layer0 outputs as inputs
    if f"gru.weight_ih_l1" in sd:
        t_out2 = torch_gru_layer_outputs(t_out, sd, 1)
        n_out2 = numpy_gru_layer_outputs(n_out, npw, 1)
        l2_2 = float(np.linalg.norm(t_out2 - n_out2))
        mx_2 = float(np.max(np.abs(t_out2 - n_out2)))
        print(f"Layer1 hidden parity: l2={l2_2:.6f} max_abs={mx_2:.6f}")
        print("t_out2[0,:5]", t_out2[0][:5])
        print("n_out2[0,:5]", n_out2[0][:5])


if __name__ == "__main__":
    main()


