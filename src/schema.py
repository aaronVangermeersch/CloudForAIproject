"""Hoe de voorbereide M5-data eruit hoort te zien, vastgelegd met Pandera.

Twee schema's, want op het moment van voorspellen ken je de verkoop van morgen nog niet:

- INVOER: wat een product-dag meebrengt, bij training én bij voorspellen
- TRAINING: hetzelfde, plus de doelkolom `sales`
"""

import pandera.pandas as pa

WEEKDAGEN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
EVENTS = ["geen", "Sporting", "Cultural", "National", "Religious"]

# Aantal stuks verkocht: nooit negatief.
_aantal = pa.Column(float, pa.Check.ge(0), coerce=True)

INVOER = pa.DataFrameSchema(
    {
        "item_id": pa.Column(str, pa.Check.str_matches(r"^[A-Z]+_\d_\d{3}$")),
        "store_id": pa.Column(str, pa.Check.str_matches(r"^(CA|TX|WI)_\d$")),
        "date": pa.Column("datetime64[ns]", coerce=True),
        "lag_1": _aantal,
        "lag_7": _aantal,
        "lag_28": _aantal,
        "gem_7": _aantal,
        "gem_28": _aantal,
        # In dollar. Voeding in M5 kost tussen 0,20 en 20 dollar: een prijs in cent valt erbuiten.
        "sell_price": pa.Column(float, pa.Check.in_range(0.01, 50)),
        "snap": pa.Column(int, pa.Check.isin([0, 1])),
        "weekday": pa.Column(str, pa.Check.isin(WEEKDAGEN)),
        "event_type_1": pa.Column(str, pa.Check.isin(EVENTS)),
    },
    # Eén rij per product per dag. Een dubbele rij telt de verkoop twee keer.
    unique=["item_id", "date"],
    # Een extra kolom is geen probleem, een ontbrekende wel.
    strict=False,
    name="M5-product-dagen",
)

TRAINING = INVOER.add_columns({"sales": pa.Column(int, pa.Check.ge(0))})
