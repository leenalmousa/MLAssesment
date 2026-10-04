import shap
import joblib
import pandas as pd
import numpy as np
import contextlib
import io
from Preprocessing import preprocess

# Load the model and Sep-Oct test data
model = joblib.load("final_mae_model.joblib")

raw = pd.read_csv("../train-test.csv")
months = pd.to_datetime(raw["date"]).dt.month

test_raw = raw[months.isin([9, 10])]

with contextlib.redirect_stdout(io.StringIO()):
    test_data = preprocess(test_raw).reset_index(drop=True)
FEATURES = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "day_of_week",
    "day_of_month",
    "month",
    "equipment_Dry Van",
    "equipment_Flatbed",
    "equipment_Reefer",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]
print("Available columns:")
print(test_data.columns.tolist())
X_test = test_data[FEATURES]

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (bar chart)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
import matplotlib.pyplot as plt
plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# Summary plot (detailed)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print("SHAP analysis complete. Saved plots.")