"""Train the State Engine model from MetaTrader 5 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]  # State_Engine/
sys.path.insert(0, str(ROOT))

from state_engine import (
    FeatureConfig,
    GatingPolicy,
    MT5Connector,
    StateEngineModel,
    StateEngineModelConfig,
)
from state_engine.pipeline import DatasetBuilder

def main() -> None:
    parser = argparse.ArgumentParser(description="Train State Engine model.")
    parser.add_argument("--symbol", required=True, help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument("--start-pos", type=int, required=True, help="Posición inicial en MT5")
    parser.add_argument("--count", type=int, required=True, help="Cantidad de velas a descargar")
    parser.add_argument("--model-out", type=Path, required=True, help="Model output path")
    args = parser.parse_args()

    connector = MT5Connector()
    try:
        ohlcv = connector.obtener_h1(args.symbol, args.start_pos, args.count)
    finally:
        connector.shutdown()

    dataset_builder = DatasetBuilder(FeatureConfig())
    artifacts = dataset_builder.build(ohlcv)

    features = artifacts.features.dropna()
    labels = artifacts.labels.loc[features.index].dropna()
    features = features.loc[labels.index]

    model = StateEngineModel(StateEngineModelConfig())
    model.fit(features, labels)
    model.save(args.model_out)

    outputs = model.predict_outputs(features)
    full_features = artifacts.full_features.loc[features.index]
    gating = GatingPolicy().apply(outputs, full_features)

    print("Training complete.")
    print(gating.tail())


if __name__ == "__main__":
    main()
