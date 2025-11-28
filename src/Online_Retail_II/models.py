from __future__ import annotations

import math

import numpy as np
import pandas as pd
import optuna

from typing import Tuple, Dict

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.Online_Retail_II.constants import COL_INVOICE_TOTAL, COL_INVOICE_NO

from xgboost import XGBRegressor

# przygotowanie modeli
def _split_features_target(
    df: pd.DataFrame,
    target_col: str = COL_INVOICE_TOTAL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    df = df.copy()

    if COL_INVOICE_NO in df.columns:
        df = df.drop(columns=[COL_INVOICE_NO])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

def _build_preprocessor(X: pd.DataFrame, modelType: str) -> ColumnTransformer:

    numeric_feats = X.select_dtypes(include=["int64", "float64", "bool"]).columns
    categorical_feats = X.select_dtypes(include=["object", "category"]).columns

    if modelType == "linear":
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_feats),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_feats),
            ]
        )
    else:
        pre = ColumnTransformer(
            transformers=[],
            remainder="passthrough"
        )
    # else:
    #     pre = ColumnTransformer(
    #         transformers=[("num", StandardScaler(), numeric_feats)]
    #     )

    return pre

# metryki
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
    }

# modele
def benchmark_model(y_train, y_test) -> Dict[str, float]:
    """ benchmark – wrzuca średnią """
    y_pred = np.repeat(y_train.mean(), len(y_test))
    return compute_metrics(y_test, y_pred)

def linear_regression_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    """ regresja liniowa """
    X_train, X_test, y_train, y_test = _split_features_target(df)

    pre = _build_preprocessor(X_train, modelType='linear')
    lr = LinearRegression()

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", lr)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    return pipe, compute_metrics(y_test, pred)

def random_forest_baseline(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    """ RandomForest – baseline."""
    X_train, X_test, y_train, y_test = _split_features_target(df)

    pre = _build_preprocessor(X_train, modelType='tree')
    model = RandomForestRegressor(
        # n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    return pipe, compute_metrics(y_test, pred)

def random_forest_optuna(df: pd.DataFrame, n_trials: int = 30):
    """Optymalizacja hiperparametrów RF przy użyciu Optuna."""

    X_train, X_test, y_train, y_test = _split_features_target(df)

    def objective(trial: optuna.trial.Trial):
        pre = _build_preprocessor(X_train, 'tree')

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.7]
            ),
        }

        model = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline([
            ("preprocess", pre),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        return math.sqrt(mean_squared_error(y_test, pred))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # final model
    best_params = study.best_params
    pre = _build_preprocessor(X_train, 'tree')

    final_model = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )

    best_pipe = Pipeline([
        ("preprocess", pre),
        ("model", final_model)
    ])

    best_pipe.fit(X_train, y_train)
    pred = best_pipe.predict(X_test)

    return study, best_pipe, compute_metrics(y_test, pred)


def xgboost_baseline(df: pd.DataFrame):

    X_train, X_test, y_train, y_test = _split_features_target(df)

    pre = _build_preprocessor(X_train, modelType='tree')

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    return pipe, compute_metrics(y_test, pred)