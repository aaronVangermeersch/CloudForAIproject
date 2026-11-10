# Verkoopvoorspelling tegen voedselverspilling

Cloud for AI, 2026-2027. Aaron Vangermeersch

Een model dat voorspelt hoeveel stuks van een voedingsproduct een winkel **morgen** verkoopt,
zodat er minder te veel besteld en weggegooid wordt. Het draait als batch: één keer per dag,
voor de bestelling van de volgende ochtend.

- **Taak:** regressie, aantal verkochte stuks per product per dag
- **Data:** [M5 Forecasting (Walmart, Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data),
  subset: winkel `CA_1`, departement `FOODS_3`, de 50 best verkopende producten, de laatste 2 jaar
- **ML-metriek:** RMSE, in stuks per product per dag. Grote missers wegen zwaarder door, en
  voor een keten kost één grote misbestelling veel

## Draaien

Je hebt Python 3.11 tot 3.13 nodig.

### 1. Omgeving

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data

De data staat niet in de repo: de bestanden zijn groter dan 100 MB, en de Kaggle-regels laten
herverdelen niet zomaar toe.

1. Log in op Kaggle en aanvaard de regels van de wedstrijd
   [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules)
2. Download de data en zet deze drie bestanden in `data/`:
   `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`
3. Maak de subset (ongeveer 10 seconden):

```bash
python -m src.prepare           # schrijft data/m5_subset.csv: ~36.000 rijen, 50 producten, 729 dagen
```

### 3. Testen, trainen, voorspellen

```bash
pytest                                  # 18 tests, allemaal groen
ruff check . ; ruff format --check .    # stijl en opmaak, allebei schoon
python -m src.train                     # vergelijkt beide modellen, schrijft models/model.joblib
python -m src.predict data/m5_subset.csv   # verwachte verkoop voor de laatste dag, per product
```

Draai alles **vanuit deze map**, niet vanuit `src/`. Anders vindt Python het pakket niet.

Zonder `data/m5_subset.csv` slaat pytest de tests over die de echte data nodig hebben. De test
op de lag-features (`test_lags_kijken_alleen_naar_het_verleden`) draait altijd.

### De notebook (optioneel)

```bash
pip install notebook matplotlib
jupyter notebook notebooks/exploratie.ipynb
```

## Modellen

| Model | `soort` in de config | RMSE op de laatste 28 dagen |
|---|---|---|
| Baseline: "zelfde als vorige week" | `vorige_week` | 8,68 |
| Gradient boosting (Poisson-loss) | `gradient_boosting` | **6,35** |

`python -m src.train` traint altijd beide en schrijft het model alleen weg als het:

1. de baseline verslaat, en
2. een RMSE van hoogstens `maximum_rmse` (7,0) haalt. Dat is ongeveer 20% beter dan de baseline:
   een model dat minder wint, is de extra complexiteit niet waard.

Een ander model kiezen doe je in `configs/default.yaml`, niet in de code.

## Wat waar staat

```
configs/default.yaml   paden, subset, features, lags, split, modelparameters. Geen code
src/
  config.py            leest de config
  prepare.py           ruwe M5-bestanden -> data/m5_subset.csv (breed naar lang, lags, opkuis)
  schema.py            hoe de data eruit hoort te zien (Pandera)
  data.py              inlezen + valideren
  train.py             baseline en model bouwen, tijdsplit, trainen, vergelijken, wegschrijven
  predict.py           model laden, verkoop voorspellen
tests/
  test_data.py         klopt de data? weigert het schema kapotte data? kijken de lags niet vooruit?
  test_model.py        vorm, invarianten, persistentie, seed, beter dan de baseline, eerlijke split
  test_systeem.py      de hele keten: csv -> trainen -> wegschrijven -> laden -> valideren -> voorspellen
notebooks/
  exploratie.ipynb     wat zit er in de data, wat is er stuk, en wat heb ik eraan gedaan
data/                  de M5-bestanden en de subset. Niet in git
models/                wordt aangemaakt door src.train. Niet in git
```

## Data leakage: waar het risico zat

- **Willekeurige split.** Dan traint het model op dagen ná de testdagen en kent het de toekomst.
  Hier is de split een tijdsplit: de laatste 28 dagen zijn test. `test_split_is_eerlijk_in_de_tijd`
  bewaakt dat.
- **Features van de dag zelf.** Een rollend gemiddelde zonder `shift` bevat de verkoop die je
  probeert te voorspellen. Alle lags en gemiddelden kijken alleen naar het verleden.
  `test_lags_kijken_alleen_naar_het_verleden` bewaakt dat.
- **Productselectie.** De 50 producten worden gekozen op de trainingsdagen, niet op de testdagen.
- Schaling en encoding zitten in de `Pipeline`, en leren dus alleen van de trainingsdagen.
