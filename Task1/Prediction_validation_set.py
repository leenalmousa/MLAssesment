import contextlib
import io
import joblib
import pandas as pd
import numpy as np

from Preprocessing import preprocess

MODEL_PATH = "final_mae_model.joblib"

INPUT_PATH = "validation.csv"

OUTPUT_PATH = "validation_predictions.csv"


FEATURES = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "day_of_week",
    "month",
    "equipment_Dry Van",
    "equipment_Flatbed",
    "equipment_Reefer",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]



model = joblib.load(
    MODEL_PATH
)



raw = pd.read_csv(
    INPUT_PATH
)


# Keep the original load IDs before preprocessing
load_ids = raw["load_id"].copy()



with contextlib.redirect_stdout(io.StringIO()):

    data = preprocess(
        raw
    ).reset_index(drop=True)



if len(raw) != len(data):

    raise ValueError(
        f"Preprocessing changed the number of rows. "
        f"Input has {len(raw)} rows, "
        f"but preprocessed data has {len(data)} rows."
    )



predicted_rpm = model.predict(
    data[FEATURES]
)



predicted_rate = (
    predicted_rpm
    *
    data["distance"].to_numpy()
)



predictions = pd.DataFrame({

    "load_id": load_ids.to_numpy(),

    "predicted_rate": predicted_rate
})


predictions.to_csv(
    OUTPUT_PATH,
    index=False
)



print("\nPrediction completed.")

print(
    f"Input rows:  {len(raw)}"
)

print(
    f"Output rows: {len(predictions)}"
)

print(
    f"Output file: {OUTPUT_PATH}"
)

print("\nFirst predictions:")

print(
    predictions.head(10).to_string(
        index=False
    )
)