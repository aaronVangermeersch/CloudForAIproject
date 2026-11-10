"""Een opgeslagen model laden en de verkoop voorspellen.

python -m src.predict data/m5_subset.csv
python -m src.predict data/m5_subset.csv --zonder-schema
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import schema
from .config import alle_features, laad_config
from .data import laad_data


def laad_model(pad: str | Path) -> Pipeline:
    return joblib.load(pad)


def voorspel(model: Pipeline, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """De verwachte verkoop in stuks, één getal per product-dag."""
    return model.predict(df[alle_features(cfg)]).astype(float)


def main(argv: list[str]) -> None:
    cfg = laad_config()
    pad = argv[0]
    valideer = None if "--zonder-schema" in argv else schema.INVOER

    df = laad_data(pad, valideer_met=valideer)
    # Zoals de batch van 20 uur: voorspel voor de laatste dag in het bestand.
    dag = df[df["date"] == df["date"].max()].copy()
    dag["verwacht"] = voorspel(laad_model(cfg["model"]["pad"]), dag, cfg)

    print(f"Verwachte verkoop op {dag['date'].iloc[0]:%Y-%m-%d}, {len(dag)} producten:")
    print(dag[["item_id", "verwacht"]].round(1).to_string(index=False))
    print(f"Totaal: {dag['verwacht'].sum():.0f} stuks")


if __name__ == "__main__":
    main(sys.argv[1:])
