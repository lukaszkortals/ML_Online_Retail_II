import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from src.Online_Retail_II.constants import COL_INVOICE_TOTAL


def plot_feature_importances(model_pipe, top_n=15):
    model = model_pipe["model"]
    importances = model.feature_importances_

    pre = model_pipe["preprocess"]
    feature_names = pre.get_feature_names_out()

    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    df_imp.plot(kind="barh", x="feature", figsize=(6, 4),
                title="Najbardziej znaczące zmienne wg. Feature importances")
    plt.gca().invert_yaxis()
    plt.show()

    return df_imp

def permutation_importance_plot(model_pipe, X_test, y_test, top_n=15):
    perm_imp = permutation_importance(
        model_pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    df_perm = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm_imp.importances_mean
    }).sort_values("importance", ascending=False).head(top_n)

    df_perm.plot(kind="barh", x="feature", figsize=(6, 4), title="Najbardziej znaczące zmienne wg. Permutation Importance")
    plt.gca().invert_yaxis()
    plt.show()

    return df_perm

def lasso_feature_selection(df, target=COL_INVOICE_TOTAL, alpha=0.001):
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

    coef_nonzero.plot(kind="barh", figsize=(6, 4), title=f"Najbardziej znaczące zmienne wg Lasso (alpha={alpha})")
    plt.show()

    return coef_nonzero

#  szap
def shap_summary(model_pipe, X_train, nsamples=1000):

    model = model_pipe["model"]
    pre = model_pipe["preprocess"]

    if len(X_train) > nsamples:
        X_sample = X_train.sample(nsamples, random_state=42)
    else:
        X_sample = X_train

    X_trans = pre.transform(X_sample)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, plot_type="dot")
