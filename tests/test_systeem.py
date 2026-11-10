"""Systeemtest: de hele keten zoals in productie, van csv tot voorspelling."""

import joblib
import numpy as np
import pytest

from src import schema
from src.data import laad_data
from src.predict import laad_model, voorspel
from src.train import train


def test_van_csv_tot_voorspelling(df, cfg, tmp_path):
    # Arrange: trainingsdata inlezen, een plek voor het model, en "morgen" zonder verkoop
    data = laad_data(cfg["data"]["pad"])
    pad = tmp_path / "model.joblib"
    morgen = data[data["date"] == data["date"].max()].drop(columns=[cfg["doel"]])

    # Act: trainen, wegschrijven, terugladen, nieuwe dag valideren, voorspellen
    model, rmse = train(data, cfg)
    joblib.dump(model, pad)
    herladen = laad_model(pad)
    verwacht = voorspel(herladen, schema.INVOER.validate(morgen), cfg)

    # Assert: geen exacte waarde, wel bovengrens, vorm, invariant en persistentie
    assert rmse <= cfg["model"]["maximum_rmse"]
    assert verwacht.shape == (len(morgen),)
    assert np.all(verwacht >= 0)
    np.testing.assert_allclose(verwacht, voorspel(model, morgen, cfg))


def test_csv_met_prijzen_in_cent_wordt_geweigerd(df, tmp_path):
    # Een nieuw kassasysteem dat in cent exporteert, mag nooit tot bij het model geraken.
    pad = tmp_path / "in_cent.csv"
    df.assign(sell_price=df["sell_price"] * 100).to_csv(pad, index=False)
    with pytest.raises(SystemExit):
        laad_data(pad, valideer_met=schema.INVOER)
