"""Test een model zonder te weten wat het exact moet voorspellen.

`assert resultaat == verwacht` werkt hier niet: er is geen verwachte waarde. Wat je wél kan
vastleggen:

1. vorm en type van de output
2. invarianten: dingen die altijd waar moeten zijn
3. de pijplijn: ruwe invoer erin, voorspelling eruit, zonder te breken
4. persistentie: opgeslagen en teruggeladen geeft hetzelfde
5. reproduceerbaarheid: zelfde seed, zelfde model
6. een ondergrens: slechter dan de baseline mag niet naar productie
7. een eerlijke split: de test ligt volledig na de training
"""

import joblib
import numpy as np
import pandas as pd

from src.predict import laad_model, voorspel
from src.train import splits, train


def test_vorm_een_getal_per_product_dag(model, df, cfg):
    verwacht = voorspel(model, df.head(50), cfg)
    assert verwacht.shape == (50,)
    assert verwacht.dtype == np.float64


def test_invariant_verkoop_is_nooit_negatief(model, df, cfg):
    # Geen bereik tussen 0 en 1 zoals bij een kans: een aantal stuks is enkel >= 0.
    assert np.all(voorspel(model, df, cfg) >= 0)


def test_pijplijn_neemt_ruwe_rijen(model, df, cfg):
    # Eén ruwe rij, met tekst in `weekday` en een kolom die het model niet gebruikt.
    ruw = df.head(1).copy()
    ruw["iets_onverwachts"] = "negeer mij"
    assert voorspel(model, ruw, cfg).shape == (1,)


def test_onbekend_product_breekt_de_pijplijn_niet(model, df, cfg):
    # Een nieuw product in het assortiment: de encoder mag niet crashen.
    nieuw = df.head(1).assign(item_id="FOODS_3_999")
    assert voorspel(model, nieuw, cfg).shape == (1,)


def test_persistentie(model, df, cfg, tmp_path):
    # Arrange: het model wegschrijven
    pad = tmp_path / "model.joblib"
    joblib.dump(model, pad)

    # Act: terugladen en voorspellen
    herladen = laad_model(pad)
    na = voorspel(herladen, df, cfg)

    # Assert: dezelfde voorspelling als ervoor
    np.testing.assert_allclose(na, voorspel(model, df, cfg))


def test_zelfde_seed_zelfde_model(df, cfg):
    eerste, _ = train(df, cfg)
    tweede, _ = train(df, cfg)
    np.testing.assert_allclose(voorspel(eerste, df, cfg), voorspel(tweede, df, cfg))


def test_beter_dan_de_baseline(model_en_rmse, df, cfg):
    # Geen exacte waarde, een ondergrens: het model moet "zelfde als vorige week" verslaan.
    _, rmse = model_en_rmse
    _, rmse_baseline = train(df, cfg, cfg["model"]["baseline"])
    assert rmse < rmse_baseline
    assert rmse <= cfg["model"]["maximum_rmse"]


def test_split_is_eerlijk_in_de_tijd(df, cfg):
    # Het grootste leakagerisico in tijdreeksen: trainen op dagen die na de testdagen liggen.
    X_train, X_test, _, _ = splits(df, cfg)
    datum = pd.to_datetime(df["date"])
    assert datum[X_train.index].max() < datum[X_test.index].min()
    assert datum[X_test.index].nunique() == cfg["split"]["test_dagen"]
