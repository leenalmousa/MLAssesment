import warnings
import contextlib
import io
import joblib
import json
import numpy as np
import pandas as pd
import sys
sys.path.append("../Task1")
from Preprocessing import preprocess
from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.linear_model import (
    Ridge,
    LinearRegression
)


warnings.filterwarnings("ignore")



DATA_PATH ="../train-test.csv"
SEED = 42
USE_DISTANCE_WEIGHT = True

OPTIMIZATION_MODES = [
    "mae"
]

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

HOLDOUT_MONTHS = [9, 10]


# MODELS

def get_models():

    return {

        "HistGB | absolute_error": lambda: HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=400,
            learning_rate=0.06,
            random_state=SEED
        ),

        "HistGB | squared_error": lambda: HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=400,
            learning_rate=0.06,
            random_state=SEED
        ),

        "GradientBoosting | squared_error": lambda: GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED
        ),

        "GradientBoosting | huber": lambda: GradientBoostingRegressor(
            loss="huber",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED
        ),

        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED
        ),

        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED
        ),

        "Ridge": lambda: Ridge(
            alpha=1.0
        ),

        "LinearRegression": lambda: LinearRegression(),
    }



# PREPROCESSING


def prep(raw_df):

    """
    Apply preprocessing.py to a split.
    """

    with contextlib.redirect_stdout(io.StringIO()):

        return preprocess(
            raw_df
        ).reset_index(drop=True)



# METRICS


def metrics(actual, pred):

    """
    Calculate:

    MAE  = Mean Absolute Error
    MAPE = Mean Absolute Percentage Error
    RMSE = Root Mean Squared Error
    """

    actual = np.asarray(actual)
    pred = np.asarray(pred)

    err = actual - pred

    return {

        "mae": float(
            np.mean(
                np.abs(err)
            )
        ),

        "rmse": float(
            np.sqrt(
                np.mean(
                    err ** 2
                )
            )
        ),

        "mape": float(
            np.mean(
                np.abs(err) /
                np.maximum(
                    np.abs(actual),
                    1e-6
                )
            ) * 100
        ),
    }


# TRAINING WEIGHTS

def get_sample_weights(train, optimization):

   

    # MAPE-oriented training

    if optimization == "mape":

        weights = (
            1.0 /
            np.maximum(
                train["posted_rate"].to_numpy(),
                1e-6
            )
        )

        return weights

    # MAE / normal training

    if USE_DISTANCE_WEIGHT:

        return train[
            "distance"
        ].to_numpy()

    return None


# FIT MODEL

def fit_model(
    build,
    features,
    train,
    optimization="normal"
):

    model = build()

    weights = get_sample_weights(
        train,
        optimization
    )

    if weights is not None:

        model.fit(
            train[features],
            train["rpm"],
            sample_weight=weights
        )

    else:

        model.fit(
            train[features],
            train["rpm"]
        )

    return model


# SCORE MODEL

def score_model(
    model,
    features,
    test
):

    # Predict RPM

    predicted_rpm = model.predict(
        test[features]
    )

    # Convert RPM -> total posted rate

    predicted_rate = (
        predicted_rpm
        *
        test["distance"].to_numpy()
    )

    actual_rate = test[
        "posted_rate"
    ].to_numpy()

    return metrics(
        actual_rate,
        predicted_rate
    )


# RECORD RESULTS

results = []


def record(
    experiment,
    optimization,
    model_name,
    model,
    features,
    train,
    test
):

    test_metrics = score_model(
        model,
        features,
        test
    )

    results.append({

        "experiment": experiment,

        "optimization": optimization,

        "model": model_name,

        "n_train": len(train),

        "n_test": len(test),

        "mae": test_metrics["mae"],

        "rmse": test_metrics["rmse"],

        "mape": test_metrics["mape"],
    })

    print(
        f"{experiment:30s} "
        f"{optimization:7s} "
        f"{model_name:32s} "
        f"train={len(train):5d} "
        f"test={len(test):5d} | "
        f"MAE=${test_metrics['mae']:7.1f} "
        f"MAPE={test_metrics['mape']:5.2f}% "
        f"RMSE=${test_metrics['rmse']:7.1f}"
    )



# RUN ALL MODELS

def run_models(
    experiment,
    train,
    test,
    models
):

    for optimization in OPTIMIZATION_MODES:

        for model_name, build in models.items():

            model = fit_model(
                build,
                FEATURES,
                train,
                optimization=optimization
            )

            record(
                experiment,
                optimization,
                model_name,
                model,
                FEATURES,
                train,
                test
            )


# MAIN

def main():

    global results

    results = []

    # Load data

    raw = pd.read_csv(
        DATA_PATH
    )

    months = pd.to_datetime(raw["date"]).dt.month
    models = get_models()


    # EXPERIMENT 1 :STRATIFIED RANDOM 70/30

    print("\n" + "=" * 110)

    print(
        "EXPERIMENT 1: STRATIFIED RANDOM 70/30"
    )

    print(
        "Sep/Oct are included in training"
    )

    print("=" * 110)

    strata = (
       months.astype(str)
        + "_"
        + raw["pickup"].astype(str)
    )

    train_raw, test_raw = train_test_split(
        raw,
        test_size=0.30,
        stratify=strata,
        random_state=SEED
    )

    train_random = prep(
        train_raw
    )

    test_random = prep(
        test_raw
    )

    test_random_sepoct = test_random[
        test_random["month"].isin(
            HOLDOUT_MONTHS
        )
    ]



    print(
        "\n--- RANDOM 70/30 - ALL ---"
    )

    run_models(
        "RANDOM 70/30 - ALL",
        train_random,
        test_random,
        models
    )

    # EXPERIMENT 2 : JAN-AUG -> SEP-OCT


    print("\n" + "=" * 110)

    print(
        "EXPERIMENT 2: JAN-AUG -> SEP-OCT"
    )

    print(
        "No Sep/Oct observations are used for training"
    )

    print("=" * 110)

    train_raw_time = raw[
        months< 9
    ]

    test_raw_time = raw[
        months.isin(
            HOLDOUT_MONTHS
        )
    ]

    train_time = prep(
        train_raw_time
    )

    test_time = prep(
        test_raw_time
    )

    run_models(
        "JAN-AUG -> SEP-OCT",
        train_time,
        test_time,
        models
    )


    # EXPERIMENT 3: EXPANDING-WINDOW VALIDATION

    print("\n" + "=" * 110)

    print(
        "EXPERIMENT 3: EXPANDING-WINDOW VALIDATION"
    )

    print("=" * 110)

    expanding_folds = [

        {
            "name": "JAN-FEB -> MAR-APR",

            "train_months": [1, 2],

            "test_months": [3, 4],
        },

        {
            "name": "JAN-APR -> MAY-JUN",

            "train_months": [
                1, 2, 3, 4
            ],

            "test_months": [5, 6],
        },

        {
            "name": "JAN-JUN -> JUL-AUG",

            "train_months": [
                1, 2, 3, 4, 5, 6
            ],

            "test_months": [7, 8],
        },

        {
            "name": "JAN-AUG -> SEP-OCT",

            "train_months": [
                1, 2, 3, 4,
                5, 6, 7, 8
            ],

            "test_months": [9, 10],
        },
    ]


    for fold in expanding_folds:

        train_months = fold[
            "train_months"
        ]

        test_months = fold[
            "test_months"
        ]
        train_raw = raw[
            months.isin(train_months)
        ]

        test_raw = raw[
           months.isin(test_months)
        ]

        train_fold = prep(
            train_raw
        )

        test_fold = prep(
            test_raw
        )

        print(
            f"\n--- {fold['name']} ---"
        )

        run_models(
            fold["name"],
            train_fold,
            test_fold,
            models
        )



        results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "model_comparison_optimization_results.csv",
        index=False
    )


    last_fold = results_df[
        results_df["experiment"]
        == "JAN-AUG -> SEP-OCT"
    ][
        [
            "optimization",
            "model",
            "mae",
            "mape",
            "rmse"
        ]
    ].sort_values(
        "mae"
    )

    # Select model with lowest Sep-Oct MAE

    best_row = (
        last_fold
        .sort_values("mae")
        .iloc[0]
    )

    best_model_name = best_row["model"]


    final_train = prep(
        raw
    )

    final_model = fit_model(
        get_models()[best_model_name],
        FEATURES,
        final_train,
        optimization="mae"
    )

    # Save trained model

    joblib.dump(
        final_model,
        "simple_model.joblib"
    )


    model_info = {

        "model": best_model_name,

        "optimization": "mae",

        "validation_period": "Sep-Oct",

        "validation_mae": float(
            best_row["mae"]
        ),

        "validation_mape": float(
            best_row["mape"]
        ),

        "validation_rmse": float(
            best_row["rmse"]
        ),

        "training_period": "Jan-Oct",

        "n_training_samples": int(
            len(final_train)
        ),

        "features": FEATURES,

        "parameters": final_model.get_params()
    }

    with open(
        "simple_model_parameters.json",
        "w"
    ) as f:

        json.dump(
            model_info,
            f,
            indent=4,
            default=str
        )
 



if __name__ == "__main__":
    main()