# Freight Rate Prediction

## Repository layout

```
.
├── train-test.csv                  labeled development data
├── score.py                        provided scorer
├── requirements.txt                updated dependencies
├── Instructions                    Information given about the task
├── scorer_results                  Results of running score.py 
├── Task1/                          validation predictions
│   ├── Preprocessing.py            cleaning pipeline (shared by both tasks)
│   ├── Model.py                    trains + compares models, saves final model
│   ├── Prediction_validation_set.py   predicts values based on validation.csv
│   └── validation.csv
└── Task2/                          December chart
    ├── Simpler_model_dec.py        trains the simplified model that only uses the featres available in the december set 
    ├── Predection_dec.py           fills predicted_rate in the December file
    └── december-chart-inputs.csv   Data for december
```

## Setup (Python 3.10+)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

requirements.txt needs: pandas, numpy, scikit-learn, joblib, matplotlib

## Run order

Run each script from inside its own folder. The scripts use relative paths
(`../train-test.csv`, `../Task1`).

### Task 1: validation predictions

```powershell
cd Task1
python Model.py                      # compares models, trains final model, saves final_mae_model.joblib
python Prediction_validation_set.py  # writes validation_predictions.csv (load_id,predicted_rate)
cd ..
```

### Task 2: December chart

```powershell
cd Task2
python Simpler_model_dec.py          # saves simple_model.joblib
python Predection_dec.py             # fills predicted_rate in december-chart-inputs.csv
cd ..
```

### Score and create the chart

From the repository root:

```powershell
python score.py --predictions Task1/validation_predictions.csv --december-predictions Task2/december-chart-inputs.csv
```

This validates both files and writes `scorer_results/candidate_december.png`.

## Notes

- `Predection_dec.py` keeps a backup of the empty template (`december_chart_inputs_original.csv`)
  the first time it runs. Re-running is safe.
- Both models predict rate per mile and multiply by distance to get `predicted_rate`.
