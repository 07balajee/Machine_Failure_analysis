"""Random Forest model: predict a shift's good output from conditions known *before*
the shift runs (machine, shift, operator, temperature, maintenance state).

Downtime, Production_Volume and OEE components are deliberately excluded — they are
measured during the shift and would leak the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Machine_Output"
CAT = ["Machine_ID", "Shift", "Operator", "Maintenance_Status", "DayOfWeek"]
NUM = ["Temperature", "Days_Since_Maint"]
FEATURES = CAT + NUM
TEST_START = pd.Timestamp("2025-10-01")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CAT:
        X[c] = X[c].astype(str)
    X["Days_Since_Maint"] = X["Days_Since_Maint"].fillna(-1)   # -1 = no service observed yet
    return X


def build_pipeline(n_estimators=300, max_depth=14, min_samples_leaf=5, seed=26) -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                            remainder="passthrough")
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=seed)
    return Pipeline([("prep", pre), ("rf", rf)])


def _metrics(y, p) -> dict:
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R²": r2_score(y, p),
            "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100)}


def _baseline(train, test):
    """Naive benchmark: machine x shift average from the training data."""
    base_map = train.groupby(["Machine_ID", "Shift"], observed=True)[TARGET].mean()
    return test.join(base_map.rename("b"), on=["Machine_ID", "Shift"])["b"].to_numpy()


def train_and_evaluate(df: pd.DataFrame, **params) -> dict:
    """Main evaluation: stratified (by machine) random 80/20 hold-out + 5-fold CV.
    Secondary: forward-in-time test (train Jan–Sep, test Oct–Dec) to show extrapolation limits."""
    df = df.sort_values("Date").reset_index(drop=True)
    X, y = _prep(df), df[TARGET]
    tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=26, stratify=df["Machine_ID"])
    train, test = df.loc[tr_idx], df.loc[te_idx]

    model = build_pipeline(**params).fit(X.loc[tr_idx], y.loc[tr_idx])
    pred = model.predict(X.loc[te_idx])

    cv = cross_val_score(build_pipeline(**params), X, y,
                         cv=KFold(5, shuffle=True, random_state=26), scoring="r2", n_jobs=-1)

    perm = permutation_importance(model, X.loc[te_idx], y.loc[te_idx], n_repeats=8, random_state=26,
                                  scoring="neg_mean_absolute_error", n_jobs=-1)
    imp = (pd.DataFrame({"Feature": FEATURES,
                         "Importance": perm.importances_mean,
                         "Std": perm.importances_std})
             .sort_values("Importance", ascending=False).reset_index(drop=True))

    # Forward-in-time check
    f_tr, f_te = df[df["Date"] < TEST_START], df[df["Date"] >= TEST_START]
    f_model = build_pipeline(**params).fit(_prep(f_tr), f_tr[TARGET])
    forward = {"rf": _metrics(f_te[TARGET].to_numpy(), f_model.predict(_prep(f_te))),
               "baseline": _metrics(f_te[TARGET].to_numpy(), _baseline(f_tr, f_te)),
               "max_age_train": f_tr["Days_Since_Maint"].max(),
               "max_age_test": f_te["Days_Since_Maint"].max()}

    return {
        "model": model,
        "rf_metrics": _metrics(y.loc[te_idx].to_numpy(), pred),
        "baseline_metrics": _metrics(y.loc[te_idx].to_numpy(), _baseline(train, test)),
        "train_metrics": _metrics(y.loc[tr_idx].to_numpy(), model.predict(X.loc[tr_idx])),
        "cv_r2": cv,
        "forward": forward,
        "importance": imp,
        "test": test.assign(Predicted=pred),
        "n_train": len(train), "n_test": len(test),
    }


def what_if(model: Pipeline, row: pd.DataFrame, days_list) -> pd.DataFrame:
    """Predicted output for a machine as its maintenance age increases."""
    out = []
    for d in days_list:
        r = row.copy()
        r["Days_Since_Maint"] = d
        r["Maintenance_Status"] = ("Up to Date" if d < 25 else "Due Soon" if d <= 35 else "Overdue")
        out.append({"Days_Since_Maint": d, "Predicted_Output": float(model.predict(_prep(r))[0])})
    return pd.DataFrame(out)
