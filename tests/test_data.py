"""Test de data, niet alleen de code. Garbage in, garbage out.

De meeste tests nemen de echte data, maken er bewust iets aan stuk, en controleren dat het
schema het tegenhoudt. Een schema dat alles doorlaat, is geen schema.
"""

import pandas as pd
import pandera.errors
import pytest

from src import schema
from src.prepare import voeg_lags_toe


def test_echte_data_is_geldig(df):
    schema.TRAINING.validate(df)


def test_hernoemde_kolom_wordt_geweigerd(df):
    kapot = df.rename(columns={"sell_price": "sell_price_usd"})
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_prijs_in_cent_wordt_geweigerd(df):
    # Het stilste soort fout: de kolom bestaat nog, het type klopt nog, de eenheid niet.
    kapot = df.assign(sell_price=df["sell_price"] * 100)
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_onbekende_winkel_wordt_geweigerd(df):
    kapot = df.copy()
    kapot.loc[0, "store_id"] = "XX_9"
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_negatieve_verkoop_wordt_geweigerd(df):
    kapot = df.copy()
    kapot.loc[0, "sales"] = -3
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_dubbele_product_dag_wordt_geweigerd(df):
    kapot = pd.concat([df, df.head(1)], ignore_index=True)
    with pytest.raises(pandera.errors.SchemaError):
        schema.TRAINING.validate(kapot)


def test_voorspellen_vraagt_geen_doelkolom(df, cfg):
    # Bij het voorspellen ken je de verkoop nog niet. Het invoerschema mag ze niet eisen.
    schema.INVOER.validate(df.drop(columns=[cfg["doel"]]))


def test_lags_kijken_alleen_naar_het_verleden(cfg):
    # Draait zonder M5-data. Twee producten, verkoop 0, 1, 2, ... per dag.
    dagen = 40
    ruw = pd.DataFrame(
        {
            "item_id": ["A"] * dagen + ["B"] * dagen,
            "sales": list(range(dagen)) + list(range(100, 100 + dagen)),
        }
    )
    df = voeg_lags_toe(ruw, cfg)
    a = df[df["item_id"] == "A"].reset_index(drop=True)

    assert a.loc[10, "lag_1"] == 9  # gisteren
    assert a.loc[10, "lag_7"] == 3  # vorige week
    assert a.loc[10, "gem_7"] == sum(range(3, 10)) / 7  # dag 3 t.e.m. 9, niet dag 10 zelf
    # Een lag loopt nooit over naar een ander product.
    assert pd.isna(df[df["item_id"] == "B"].iloc[0]["lag_1"])
