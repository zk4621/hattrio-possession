"""LightGBM possession model: train, predict, save, load."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np

from .decode import DecoderConfig
from .paths import MODELS

LGB_PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "seed": 0, "deterministic": True,
              "force_row_wise": True, "num_threads": 8, "verbose": -1}
N_ROUNDS = 400
MODEL_PATH = MODELS / "possession_lgb.txt"
DECODER_PATH = MODELS / "decoder.json"


def train(X: np.ndarray, y: np.ndarray, names: list[str], params: dict = LGB_PARAMS,
          n_rounds: int = N_ROUNDS) -> lgb.Booster:
    ds = lgb.Dataset(np.asarray(X, dtype=np.float32), label=np.asarray(y, dtype=np.int32),
                     feature_name=list(names), free_raw_data=True)
    return lgb.train(dict(params), ds, num_boost_round=n_rounds)


def predict_proba(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    P = np.asarray(booster.predict(X, num_threads=LGB_PARAMS["num_threads"]), dtype=np.float64)
    P = P.reshape(len(X), 3)
    empty = np.isnan(X).all(axis=1)
    P[empty] = 1.0 / 3.0
    return P


def save(booster: lgb.Booster, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load(path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def proba_fn(booster: lgb.Booster) -> Callable[[np.ndarray, list[str]], np.ndarray]:
    wanted = booster.feature_name()

    def fn(F: np.ndarray, names: list[str]) -> np.ndarray:
        index = {n: i for i, n in enumerate(names)}
        missing = [n for n in wanted if n not in index]
        if missing:
            raise ValueError(f"features missing for the model: {missing[:5]}")
        return predict_proba(booster, F[:, [index[n] for n in wanted]])

    return fn


def load_proba_fn(path=MODEL_PATH):
    return proba_fn(load(path))


def save_decoder(cfg: DecoderConfig, path=DECODER_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"min_len": list(cfg.min_len),
                                      "switch_penalty": cfg.switch_penalty}, indent=1))


def load_decoder(path=DECODER_PATH) -> DecoderConfig:
    d = json.loads(Path(path).read_text())
    return DecoderConfig(min_len=tuple(int(v) for v in d["min_len"]),
                         switch_penalty=int(d["switch_penalty"]))
