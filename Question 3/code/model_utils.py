from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_ols(y: pd.Series, x: pd.DataFrame):
    x_with_const = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, x_with_const, missing="drop").fit()
    return model


def regression_diagnostics(model, y: pd.Series, x: pd.DataFrame) -> dict:
    x_with_const = sm.add_constant(x, has_constant="add")
    y_hat = model.predict(x_with_const)
    resid = y - y_hat
    return {
        "n_obs": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "alpha_monthly": float(model.params.get("const", np.nan)),
        "alpha_annualized": float((1.0 + model.params.get("const", 0.0)) ** 12 - 1.0),
        "resid_vol_monthly": float(resid.std(ddof=1)),
        "resid_vol_annualized": float(resid.std(ddof=1) * np.sqrt(12.0)),
        "corr_fitted_actual": float(np.corrcoef(y, y_hat)[0, 1]),
    }


def coef_table(model) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "coef": model.params,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
        }
    )
    return out


def compare_two_models(model_a_name: str, diag_a: dict, model_b_name: str, diag_b: dict) -> pd.DataFrame:
    rows = [
        {
            "model": model_a_name,
            "n_obs": diag_a["n_obs"],
            "adj_r2": diag_a["adj_r2"],
            "alpha_monthly": diag_a["alpha_monthly"],
            "alpha_annualized": diag_a["alpha_annualized"],
            "resid_vol_annualized": diag_a["resid_vol_annualized"],
            "corr_fitted_actual": diag_a["corr_fitted_actual"],
        },
        {
            "model": model_b_name,
            "n_obs": diag_b["n_obs"],
            "adj_r2": diag_b["adj_r2"],
            "alpha_monthly": diag_b["alpha_monthly"],
            "alpha_annualized": diag_b["alpha_annualized"],
            "resid_vol_annualized": diag_b["resid_vol_annualized"],
            "corr_fitted_actual": diag_b["corr_fitted_actual"],
        },
    ]
    return pd.DataFrame(rows)
