import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

#  FEATURE IMPORTANCES – RandomForest-style
def plot_feature_importances(model_pipe: object, X_train: object, top_n: object = 15) -> None:
    model = model_pipe["model"]

    try:
        importances = model.feature_importances_
    except AttributeError:
        raise ValueError("Model nie ma atrybutu feature_importances_. "
                         "Działa tylko dla drzew / RF / XGB.")

    # nazwy kolumn po OHE
    ohe_features = (
        model_pipe["preprocess"]
        .named_transformers_["cat"]
        .get_feature_names_out(X_train.select_dtypes("object").columns)
    )

    num_features = X_train.select_dtypes(["int64", "float64", "bool"]).columns

    all_features = np.concatenate([num_features, ohe_features])

    df_imp = (
        pd.DataFrame({"feature": all_features, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    df_imp.plot(kind="barh", x="feature", figsize=(8, 6), title="Feature importances (RandomForest/XGBoost)")
    plt.gca().invert_yaxis()
    plt.show()

    return df_imp

# PERMUTATION IMPORTANCE (najbardziej wiarygodne)
def permutation_importance_plot(model_pipe, X_test, y_test, top_n=15):
    result = permutation_importance(
        model_pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    df_perm = pd.DataFrame({
        "feature": X_test.columns,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=False).head(top_n)

    df_perm.plot(kind="barh", x="feature", figsize=(8, 6), title="Permutation Importance")
    plt.gca().invert_yaxis()
    plt.show()

    return df_perm

# LASSO — Feature Selection (tylko dla danych numerycznych)
def lasso_feature_selection(df, target="InvoiceTotal", alpha=0.001):
    df = df.copy()
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)

    coef = pd.Series(lasso.coef_, index=X.columns)
    coef_nonzero = coef[coef != 0].sort_values()

    coef_nonzero.plot(kind="barh", figsize=(8, 6), title=f"Lasso feature selection (alpha={alpha})")
    plt.show()

    return coef_nonzero


# ============================================================
#   4. SHAP – interpretacja globalna i lokalna
# ============================================================

# def shap_summary(model_pipe, X_train):
#     explainer = shap.TreeExplainer(model_pipe["model"])
#     X_transformed = model_pipe["preprocess"].transform(X_train)
#
#     shap_values = explainer.shap_values(X_transformed)
#     shap.summary_plot(shap_values, X_transformed, plot_type="dot")

def shap_summary(model_pipe, X_train, nsamples: int = 1000):
    """
    SHAP summary plot dla pipeline'u (preprocessing + model).
    SHAP sam ogarnia transformacje, my podajemy surowe X_train.
    """
    # próbka, żeby nie liczyć na całym zbiorze (przyspieszenie)
    if len(X_train) > nsamples:
        X_sample = X_train.sample(nsamples, random_state=42)
    else:
        X_sample = X_train

    explainer = shap.Explainer(model_pipe)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample, plot_type="dot")

def shap_beeswarm(model_pipe, X_train):
    explainer = shap.TreeExplainer(model_pipe["model"])
    X_transformed = model_pipe["preprocess"].transform(X_train)
    shap_values = explainer.shap_values(X_transformed)

    shap.summary_plot(shap_values, X_transformed, plot_type="violin")

def shap_waterfall(model_pipe, X_train, index=0, nsamples=500):
    """
    Waterfall plot dla pojedynczej obserwacji.
    Działa z nowym SHAP API (Explainer + __call__).
    """

    # próbka (przyśpiesza)
    if len(X_train) > nsamples:
        X_sample = X_train.sample(nsamples, random_state=42)
    else:
        X_sample = X_train

    explainer = shap.Explainer(model_pipe)
    shap_values = explainer(X_sample)

    # wybór obserwacji
    single = shap_values[index]

    shap.plots.waterfall(single)

# def shap_waterfall(model_pipe, X_train, index=0):
#     explainer = shap.TreeExplainer(model_pipe["model"])
#     X_t = model_pipe["preprocess"].transform(X_train)
#
#     shap_values = explainer.shap_values(X_t)
#
#     shap.plots._waterfall.waterfall_legacy(
#         explainer.expected_value,
#         shap_values[index],
#     )
