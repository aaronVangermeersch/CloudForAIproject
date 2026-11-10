"""Van de ruwe M5-bestanden naar één tabel: één rij per product per dag, met features.

python -m src.prepare

De ruwe verkoop is breed (één kolom per dag, `d_1` tot `d_1941`) en meer dan 100 MB.
Hier wordt dat een kleine, lange tabel voor één winkel en één departement.
"""

import pandas as pd

from .config import laad_config


def voeg_lags_toe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Lags en gemiddelden per product, altijd met `shift`: alleen het verleden telt.

    `df` moet per product op datum gesorteerd zijn. Zonder de `shift(1)` voor de rolling zou
    het gemiddelde de verkoop van de dag zelf bevatten: dat is data leakage.
    """
    per_product = df.groupby("item_id")["sales"]
    for k in cfg["lags"]:
        df[f"lag_{k}"] = per_product.shift(k)
    for k in cfg["gemiddelden"]:
        df[f"gem_{k}"] = per_product.transform(lambda s, k=k: s.shift(1).rolling(k).mean())
    return df


def bereid_voor(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    # Genoeg dagen extra inlezen om de langste lag of het langste gemiddelde te kunnen vullen.
    opwarming = max(cfg["lags"] + [g + 1 for g in cfg["gemiddelden"]])
    kop = pd.read_csv(d["ruw"]["verkoop"], nrows=0).columns
    dagen = [c for c in kop if c.startswith("d_")][-(d["aantal_dagen"] + opwarming) :]

    verkoop = pd.read_csv(d["ruw"]["verkoop"], usecols=["item_id", "dept_id", "store_id"] + dagen)
    verkoop = verkoop[
        (verkoop["store_id"] == d["winkel"]) & (verkoop["dept_id"] == d["departement"])
    ]

    # De best verkopende producten, gekozen zonder naar de testdagen te kijken.
    trainingsdagen = dagen[: -cfg["split"]["test_dagen"]]
    top = verkoop.set_index("item_id")[trainingsdagen].sum(axis=1).nlargest(d["aantal_producten"])
    verkoop = verkoop[verkoop["item_id"].isin(top.index)]

    lang = verkoop.melt(
        id_vars=["item_id", "store_id"], value_vars=dagen, var_name="d", value_name="sales"
    )

    kalender = pd.read_csv(d["ruw"]["kalender"])
    kalender["snap"] = kalender[f"snap_{d['winkel'].split('_')[0]}"]
    kalender["event_type_1"] = kalender["event_type_1"].fillna("geen")
    lang = lang.merge(
        kalender[["d", "date", "wm_yr_wk", "weekday", "event_type_1", "snap"]], on="d"
    )

    prijzen = pd.read_csv(d["ruw"]["prijzen"])
    prijzen = prijzen[prijzen["store_id"] == d["winkel"]]
    lang = lang.merge(prijzen, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    lang = lang.sort_values(["item_id", "date"]).reset_index(drop=True)
    lang = voeg_lags_toe(lang, cfg)

    # Gesloten dagen (kerstdag: het hele departement verkoopt niets) tellen niet mee.
    per_dag = verkoop[dagen].sum()
    gesloten = per_dag[per_dag == 0].index
    lang = lang[~lang["d"].isin(gesloten)]
    # De opwarmdagen hebben nog geen volledige lags; zonder prijs zat het product niet in de winkel.
    lang = lang.dropna(subset=cfg["features"]["numeriek"])

    kolommen = ["item_id", "store_id", "date", cfg["doel"]] + [
        f for f in cfg["features"]["numeriek"] + cfg["features"]["categorisch"] if f != "item_id"
    ]
    return lang[kolommen].reset_index(drop=True)


def main() -> None:
    cfg = laad_config()
    df = bereid_voor(cfg)
    df.to_csv(cfg["data"]["pad"], index=False)
    print(f"{len(df)} rijen, {df['item_id'].nunique()} producten, {df['date'].nunique()} dagen")
    print(f"Weggeschreven naar {cfg['data']['pad']}")


if __name__ == "__main__":
    main()
