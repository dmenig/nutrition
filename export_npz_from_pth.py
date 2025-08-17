import os
import json
import argparse
from typing import Dict

import numpy as np
import torch

from train_model import FinalModel


def infer_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Infer (input_size, hidden_size, num_layers) from a GRU state_dict."""
    # Hidden size from W_hh_l0: [3H, H]
    hh0 = sd["gru.weight_hh_l0"]
    hidden_size = hh0.shape[1]
    # Input size from head.0.weight: [64, hidden+input]
    head0 = sd["metabolism_increment_head.0.weight"]
    input_plus_hidden = head0.shape[1]
    input_size = input_plus_hidden - hidden_size
    # Number of layers by presence of weight_ih_l{idx}
    num_layers = 0
    while f"gru.weight_ih_l{num_layers}" in sd:
        num_layers += 1
    if num_layers == 0:
        num_layers = 1
    return input_size, hidden_size, num_layers


def export_npz(pth_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sd = torch.load(pth_path, map_location="cpu")
    input_size, hidden_size, num_layers = infer_arch_from_state_dict(sd)

    # Construct a model to match parameter structure (initial_weight value is irrelevant)
    model = FinalModel(input_size, initial_weight_guess=70.0, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(sd)
    sd = model.state_dict()

    npz_path = os.path.join(out_dir, "recurrent_model_np.npz")
    np.savez(
        npz_path,
        **{
            "gru.weight_ih_l0": sd.get("gru.weight_ih_l0", torch.empty(0)).cpu().numpy(),
            "gru.weight_hh_l0": sd.get("gru.weight_hh_l0", torch.empty(0)).cpu().numpy(),
            "gru.bias_ih_l0": sd.get("gru.bias_ih_l0", torch.empty(0)).cpu().numpy(),
            "gru.bias_hh_l0": sd.get("gru.bias_hh_l0", torch.empty(0)).cpu().numpy(),
            # Optional second layer
            "gru.weight_ih_l1": sd.get("gru.weight_ih_l1", torch.empty(0)).cpu().numpy(),
            "gru.weight_hh_l1": sd.get("gru.weight_hh_l1", torch.empty(0)).cpu().numpy(),
            "gru.bias_ih_l1": sd.get("gru.bias_ih_l1", torch.empty(0)).cpu().numpy(),
            "gru.bias_hh_l1": sd.get("gru.bias_hh_l1", torch.empty(0)).cpu().numpy(),
            # Head
            "head.0.weight": sd.get("metabolism_increment_head.0.weight").cpu().numpy(),
            "head.0.bias": sd.get("metabolism_increment_head.0.bias").cpu().numpy(),
            "head.2.weight": sd.get("metabolism_increment_head.2.weight").cpu().numpy(),
            "head.2.bias": sd.get("metabolism_increment_head.2.bias").cpu().numpy(),
            # Scalars
            "initial_metabolism": sd.get("initial_metabolism").cpu().numpy(),
            "initial_adj_weight": sd.get("initial_adj_weight").cpu().numpy(),
        }
    )
    # Touch mtime to be >= pth so backend auto-selects NumPy
    try:
        pth_mtime = os.path.getmtime(pth_path)
        os.utime(npz_path, (pth_mtime, pth_mtime + 1))
    except Exception:
        pass
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export recurrent_model_np.npz from a .pth state dict")
    parser.add_argument("--pth", default="models/recurrent_model.pth", help="Path to .pth state dict")
    parser.add_argument("--out", default="models", help="Output directory for .npz")
    args = parser.parse_args()

    assert os.path.exists(args.pth), f"Missing .pth file: {args.pth}"
    out_path = export_npz(args.pth, args.out)
    print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()


