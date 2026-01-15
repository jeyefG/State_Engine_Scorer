import pandas as pd

from state_engine.context_features import (
    compute_dist_vwap_atr,
    compute_session_bucket,
    compute_state_age,
)


def test_compute_session_bucket_uses_hour_only() -> None:
    index = pd.date_range("2024-01-01 00:00", periods=4, freq="6h")
    buckets = compute_session_bucket(index)

    assert list(buckets) == ["ASIA", "ASIA", "LONDON", "NY_PM"]


def test_compute_state_age_resets_on_state_change() -> None:
    state_hat = pd.Series([1, 1, 2, 2, 2, 1, 1], index=pd.RangeIndex(7))
    ages = compute_state_age(state_hat, max_age=12)

    assert list(ages) == [1, 2, 1, 2, 3, 1, 2]


def test_compute_dist_vwap_atr_is_absolute() -> None:
    df = pd.DataFrame(
        {
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [11.0, 9.0],
            "vwap": [10.0, 10.0],
            "atr_h2": [2.0, 2.0],
        }
    )
    dist = compute_dist_vwap_atr(df, atr_window=14, eps=1e-9)

    assert dist.iloc[0] == dist.iloc[1] == 0.5
