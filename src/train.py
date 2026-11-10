"""Het model trainen, vergelijken met de baseline, en wegschrijven.

python -m src.train
"""

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import alle_features, laad_config
from .data import laad_data


class VorigeWeek(RegressorMixin, BaseEstimator):
    """Baseline: morgen verkoop je wat je vorige week op dezelfde weekdag verkocht."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]  # niets te leren, maar sklearn eist een "fitted" teken
        return self

    def predict(self, X):
        return X["lag_7"].to_numpy(dtype=float)


def bouw_pijplijn(cfg: dict, soort: str | None = None) -> Pipeline:
    """Alle stappen die iets leren uit de data zitten ín de pijplijn. Zie labo 4."""
    soort = soort or cfg["model"]["soort"]
    if soort == "vorige_week":
        return Pipeline([("model", VorigeWeek())])
    if soort != "gradient_boosting":
        raise ValueError(f"Onbekende model.soort {soort!r}, kies vorige_week of gradient_boosting")

    voorbewerking = ColumnTransformer(
        [
            ("num", StandardScaler(), cfg["features"]["numeriek"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cfg["features"]["categorisch"],
            ),
        ]
    )
    model = HistGradientBoostingRegressor(
        **cfg["model"]["gradient_boosting"], random_state=cfg["split"]["seed"]
    )
    return Pipeline([("voorbewerking", voorbewerking), ("model", model)])


def splits(df: pd.DataFrame, cfg: dict):
    """Tijdsplit: train op alles vóór de laatste `test_dagen`, test op die laatste dagen."""
    datum = pd.to_datetime(df["date"])
    grens = datum.max() - pd.Timedelta(days=cfg["split"]["test_dagen"])
    train, test = df[datum <= grens], df[datum > grens]
    X = alle_features(cfg)
    return train[X], test[X], train[cfg["doel"]], test[cfg["doel"]]


def train(df: pd.DataFrame, cfg: dict, soort: str | None = None) -> tuple[Pipeline, float]:
    """Train op de trainingsdagen, geef het model en de RMSE op de testdagen terug."""
    X_train, X_test, y_train, y_test = splits(df, cfg)
    pijplijn = bouw_pijplijn(cfg, soort).fit(X_train, y_train)
    rmse = root_mean_squared_error(y_test, pijplijn.predict(X_test))
    return pijplijn, float(rmse)


def main() -> None:
    cfg = laad_config()
    df = laad_data(cfg["data"]["pad"])
    soort, baseline = cfg["model"]["soort"], cfg["model"]["baseline"]

    _, rmse_baseline = train(df, cfg, baseline)
    model, rmse = train(df, cfg, soort)
    print("RMSE op de testdagen, in stuks per product per dag:")
    print(f"  {baseline:<18} {rmse_baseline:.3f}  (baseline)")
    print(f"  {soort:<18} {rmse:.3f}  ({100 * (1 - rmse / rmse_baseline):.1f}% beter)")

    if rmse >= rmse_baseline:
        raise SystemExit(f"{soort} verslaat de baseline niet. Niet weggeschreven.")
    maximum = cfg["model"]["maximum_rmse"]
    if rmse > maximum:
        raise SystemExit(f"RMSE {rmse:.3f} ligt boven het maximum {maximum}. Niet weggeschreven.")

    cfg["model"]["pad"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg["model"]["pad"])
    print(f"Model weggeschreven naar {cfg['model']['pad']}")


if __name__ == "__main__":
    main()
