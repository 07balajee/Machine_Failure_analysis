"""Train the Random Forest once and save the model + evaluation results to models/."""
import json
from pathlib import Path

import joblib
import pandas as pd

from src import analysis as A
from src import model as M
from src.cleaning import clean

ROOT = Path(__file__).parent

if __name__ == "__main__":
    df, _ = clean(pd.read_csv(ROOT / "data" / "production_data.csv"))
    df = A.add_features(df)
    res = M.train_and_evaluate(df)

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(res["model"], out / "random_forest.joblib")
    report = {
        "features": M.FEATURES, "target": M.TARGET,
        "n_train": res["n_train"], "n_test": res["n_test"],
        "holdout_rf": res["rf_metrics"], "holdout_baseline": res["baseline_metrics"],
        "train_rf": res["train_metrics"], "cv_r2": list(res["cv_r2"]),
        "forward_in_time": res["forward"],
        "permutation_importance": res["importance"].to_dict(orient="records"),
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["holdout_rf", "holdout_baseline", "cv_r2"]}, indent=2, default=float))
    print(f"Saved -> {out}")
