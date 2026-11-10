"""Gedeelde fixtures: één keer data inlezen en één keer trainen, voor alle tests."""

import pytest

from src.config import laad_config
from src.data import laad_data
from src.train import train


@pytest.fixture(scope="session")
def cfg():
    return laad_config()


@pytest.fixture(scope="session")
def df(cfg):
    if not cfg["data"]["pad"].exists():
        pytest.skip("Geen data/m5_subset.csv. Download M5 en draai python -m src.prepare (README)")
    return laad_data(cfg["data"]["pad"])


@pytest.fixture(scope="session")
def model_en_rmse(df, cfg):
    return train(df, cfg)


@pytest.fixture(scope="session")
def model(model_en_rmse):
    return model_en_rmse[0]
