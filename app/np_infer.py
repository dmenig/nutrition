import numpy as np
from typing import Dict, Tuple


K_CAL_PER_KG = 7700.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class NumpyFinalModel:
    def __init__(self, weights: Dict[str, np.ndarray], input_size: int, hidden_size: int, num_layers: int):
        self.weights = weights
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Coerce scalar parameters from arrays; apply sane defaults if missing/empty
        init_met = weights.get("initial_metabolism", None)
        if isinstance(init_met, np.ndarray) and init_met.size > 0:
            self.initial_metabolism = float(init_met.reshape(-1)[0])
        elif isinstance(init_met, (float, int)):
            self.initial_metabolism = float(init_met)
        else:
            self.initial_metabolism = 2.5
        init_w = weights.get("initial_adj_weight", None)
        if isinstance(init_w, np.ndarray) and init_w.size > 0:
            self.initial_adj_weight = float(init_w.reshape(-1)[0])
        elif isinstance(init_w, (float, int)):
            self.initial_adj_weight = float(init_w)
        else:
            self.initial_adj_weight = 70.0

    def _gru_layer_forward(self, x_seq: np.ndarray, layer_idx: int) -> np.ndarray:
        # x_seq: [seq_len, input_dim]
        seq_len, input_dim = x_seq.shape
        hidden_dim = self.hidden_size
        W_ih = self.weights[f"gru.weight_ih_l{layer_idx}"]  # [3H, input_dim]
        W_hh = self.weights[f"gru.weight_hh_l{layer_idx}"]  # [3H, hidden_dim]
        b_ih = self.weights.get(f"gru.bias_ih_l{layer_idx}")  # [3H]
        b_hh = self.weights.get(f"gru.bias_hh_l{layer_idx}")  # [3H]

        # Split into gates in PyTorch's packed order: r, z, n
        W_ir, W_iz, W_in = np.split(W_ih, 3, axis=0)
        W_hr, W_hz, W_hn = np.split(W_hh, 3, axis=0)
        b_ir = b_ih[0:hidden_dim] if isinstance(b_ih, np.ndarray) else 0.0
        b_iz = b_ih[hidden_dim:2 * hidden_dim] if isinstance(b_ih, np.ndarray) else 0.0
        b_in = b_ih[2 * hidden_dim:3 * hidden_dim] if isinstance(b_ih, np.ndarray) else 0.0
        b_hr = b_hh[0:hidden_dim] if isinstance(b_hh, np.ndarray) else 0.0
        b_hz = b_hh[hidden_dim:2 * hidden_dim] if isinstance(b_hh, np.ndarray) else 0.0
        b_hn = b_hh[2 * hidden_dim:3 * hidden_dim] if isinstance(b_hh, np.ndarray) else 0.0

        h_t = np.zeros((hidden_dim,), dtype=np.float32)
        outs = []
        for t in range(seq_len):
            x_t = x_seq[t]
            r_t = _sigmoid(x_t @ W_ir.T + b_ir + h_t @ W_hr.T + b_hr)
            z_t = _sigmoid(x_t @ W_iz.T + b_iz + h_t @ W_hz.T + b_hz)
            n_t = _tanh(x_t @ W_in.T + b_in + r_t * (h_t @ W_hn.T + b_hn))
            h_t = (1.0 - z_t) * n_t + z_t * h_t
            outs.append(h_t)
        return np.stack(outs, axis=0)  # [seq_len, hidden_dim]

    def forward(self, nutrition_data: np.ndarray) -> np.ndarray:
        # nutrition_data: [1, seq_len, input_size]
        assert nutrition_data.ndim == 3 and nutrition_data.shape[0] == 1
        seq_len = nutrition_data.shape[1]

        # GRU stack
        x = nutrition_data[0]
        for layer_idx in range(self.num_layers):
            x = self._gru_layer_forward(x, layer_idx)

        gru_out = x  # [seq_len, hidden_size]
        base_metabolisms = []
        current_metabolism = float(self.initial_metabolism)

        # Head weights (PyTorch Linear uses y = x @ W.T + b)
        W0 = self.weights["head.0.weight"]  # [64, hidden+input]
        b0 = self.weights["head.0.bias"]  # [64]
        W2 = self.weights["head.2.weight"]  # [1, 64]
        b2 = self.weights["head.2.bias"]  # [1]

        for t in range(seq_len):
            current_gru_out = gru_out[t]  # [hidden]
            current_nutrition = nutrition_data[0, t, :]  # [input]
            combined = np.concatenate([current_gru_out, current_nutrition], axis=0)
            h = _relu(combined @ W0.T + b0)
            # Match Torch model: tanh head scaled to max 0.2 (~200 kcal/day)
            inc = np.tanh(h @ W2.T + b2)[0] * 0.2
            current_metabolism = current_metabolism + float(inc)
            base_metabolisms.append([current_metabolism])

        return np.asarray(base_metabolisms, dtype=np.float32).reshape(1, seq_len, 1)


def reconstruct_trajectory_numpy(
    observed_weights: np.ndarray,
    base_metabolisms: np.ndarray,
    nutrition_data: np.ndarray,
    sport_data: np.ndarray,
    initial_adj_weight: float,
    normalization_stats: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    # All arrays are numpy; observed_weights: [1, seq_len]
    seq_len = observed_weights.shape[1]

    cal_stats = normalization_stats["calories"]
    sport_stats = normalization_stats["sport"]

    calories_in_unnorm = nutrition_data[:, :, 0] * cal_stats["std"] + cal_stats["mean"]
    sport_unnorm = sport_data * sport_stats["std"] + sport_stats["mean"]

    # Clamp base metabolism to a physiologically plausible range (in thousands of kcal/day)
    base_thousands = base_metabolisms.squeeze(-1)
    base_thousands_clamped = np.clip(base_thousands, 1.5, 3.5)

    calories_delta = calories_in_unnorm - sport_unnorm - base_thousands_clamped * 1000.0

    # Anchor the trajectory using normalized observed weight at t=0 when available
    first_obs_norm = float(observed_weights[0, 0])
    start_weight = first_obs_norm if first_obs_norm != 0.0 else float(initial_adj_weight)
    w_adj = np.zeros((1, seq_len), dtype=np.float32)
    w_adj[:, 0] = start_weight
    for t in range(1, seq_len):
        weight_change = calories_delta[:, t - 1] / K_CAL_PER_KG
        w_adj[:, t] = w_adj[:, t - 1] + weight_change

    return w_adj, w_adj


def load_numpy_weights(npz_path: str) -> Dict[str, np.ndarray]:
    npz = np.load(npz_path)
    weights: Dict[str, np.ndarray] = {}

    # Map PyTorch keys => numpy names used here
    for key in npz.files:
        weights[key] = npz[key]

    return weights


