# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy tests
COPY tests /app/tests

# Copy the application code
COPY app /app/app
COPY train_model.py /app/
COPY build_features.py /app/
COPY sport_formulas.py /app/
COPY data_processor.py /app/
COPY utils.py /app/
COPY nutrition_calculator.py /app/
COPY process_nutrition_journal.py /app/
COPY plot_results.py /app/
COPY analyze_sensitivity.py /app/
COPY app/db/populate_db.py /app/app/db/
COPY app/init_plots.py /app/app/
COPY models /app/models

# Export a lightweight NumPy weights file for torch-free inference, if the PTH exists
RUN python - <<'PY'
import os, numpy as np
try:
    import torch
except Exception as e:
    print("Torch not available during build:", e)
    torch=None
pth="/app/models/recurrent_model.pth"
out="/app/models/recurrent_model_np.npz"
if os.path.exists(pth) and torch is not None:
    sd = torch.load(pth, map_location="cpu")
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    def to_np(key):
        t = sd.get(key, None)
        if t is None:
            return np.empty((0,), dtype=np.float32)
        return t.detach().cpu().numpy()
    np.savez(out, **{
        "gru.weight_ih_l0": to_np("gru.weight_ih_l0"),
        "gru.weight_hh_l0": to_np("gru.weight_hh_l0"),
        "gru.bias_ih_l0": to_np("gru.bias_ih_l0"),
        "gru.bias_hh_l0": to_np("gru.bias_hh_l0"),
        "gru.weight_ih_l1": to_np("gru.weight_ih_l1"),
        "gru.weight_hh_l1": to_np("gru.weight_hh_l1"),
        "gru.bias_ih_l1": to_np("gru.bias_ih_l1"),
        "gru.bias_hh_l1": to_np("gru.bias_hh_l1"),
        "head.0.weight": to_np("metabolism_increment_head.0.weight"),
        "head.0.bias": to_np("metabolism_increment_head.0.bias"),
        "head.2.weight": to_np("metabolism_increment_head.2.weight"),
        "head.2.bias": to_np("metabolism_increment_head.2.bias"),
        "initial_metabolism": to_np("initial_metabolism"),
        "initial_adj_weight": to_np("initial_adj_weight"),
    })
    print(f"Exported {out}")
else:
    print("No PTH found or torch unavailable; skipping export")
PY

# List the contents of the directory to debug file path issues

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.main:app when the container launches
# Only initialize DB once if needed; no CSV usage in production
CMD ["sh", "-c", "python app/db/populate_db.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]