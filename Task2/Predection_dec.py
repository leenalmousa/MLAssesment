import contextlib
import io
import os
import shutil

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.append("../Task1")
from Preprocessing import preprocess

MODEL_PATH = "simple_model.joblib"
FILE_PATH = "december-chart-inputs.csv"          # read from and written back to this same file
BACKUP_PATH = "december_chart_inputs_original.csv"

FEATURES = [
    "distance",
    "weight",
    "day_of_week",
    "day_of_month",
    "month",
    "equipment_Dry Van",
    "equipment_Flatbed",
    "equipment_Reefer",
]

# ---- one-time backup of the untouched template ----
if not os.path.exists(BACKUP_PATH):
    shutil.copy(FILE_PATH, BACKUP_PATH)

# ---- read the file for the model (normal types) ----
dec = pd.read_csv(FILE_PATH)
dec_in = dec.drop(columns="predicted_rate").copy()
dec_in["market_index"] = np.nan        # preprocess() expects this column; the model doesn't use it

with contextlib.redirect_stdout(io.StringIO()):
    data = preprocess(dec_in).reset_index(drop=True)

for col in ["equipment_Dry Van", "equipment_Flatbed", "equipment_Reefer"]:
    if col not in data.columns:
        data[col] = 0

# ---- predict ----
model = joblib.load(MODEL_PATH)
predicted_rate = model.predict(data[FEATURES]) * data["distance"].to_numpy()

assert len(predicted_rate) == 31, "expected 31 December rows"
assert (predicted_rate > 0).all(), "non-positive prediction"

# ---- write back: read the file as plain text so nothing else changes ----
out = pd.read_csv(FILE_PATH, dtype=str, keep_default_na=False)
assert len(out) == len(predicted_rate)
out["predicted_rate"] = np.round(predicted_rate, 2).astype(str)
out.to_csv(FILE_PATH, index=False)

print(f"Filled predicted_rate in {FILE_PATH} (backup: {BACKUP_PATH})")
print(out.to_string(index=False))