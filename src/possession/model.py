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
CHANGE_PATH = MODELS / "change_lgb.txt"
CHANGE_PARAMS = {k: v for k, v in LGB_PARAMS.items() if k != "num_class"} | {"objective": "binary"}
N_CHANGE_ROUNDS = 300


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


def load_change_fn(cfg: DecoderConfig, path=CHANGE_PATH):
    """The change model's scorer when the decoder uses it, else None."""
    return change_fn(load(path)) if cfg.change_weight else None


def train_change(X: np.ndarray, y: np.ndarray, names: list[str],
                 n_rounds: int = N_CHANGE_ROUNDS) -> lgb.Booster:
    ds = lgb.Dataset(np.asarray(X, dtype=np.float32), label=np.asarray(y, dtype=np.int32),
                     feature_name=list(names), free_raw_data=True)
    return lgb.train(dict(CHANGE_PARAMS), ds, num_boost_round=n_rounds)


def change_fn(booster: lgb.Booster) -> Callable[[np.ndarray, list[str]], np.ndarray]:
    """p(change near this frame) from the given columns, picked by name."""
    wanted = booster.feature_name()

    def fn(F: np.ndarray, names: list[str]) -> np.ndarray:
        index = {n: i for i, n in enumerate(names)}
        missing = [n for n in wanted if n not in index]
        if missing:
            raise ValueError(f"features missing for the change model: {missing[:5]}")
        X = np.asarray(F[:, [index[n] for n in wanted]], dtype=np.float32)
        return np.asarray(booster.predict(X, num_threads=LGB_PARAMS["num_threads"]),
                          dtype=np.float64)

    return fn


def save_decoder(cfg: DecoderConfig, path=DECODER_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"min_len": list(cfg.min_len),
                                      "switch_penalty": cfg.switch_penalty,
                                      "min_len_seen": cfg.min_len_seen,
                                      "change_weight": cfg.change_weight}, indent=1))


def load_decoder(path=DECODER_PATH) -> DecoderConfig:
    d = json.loads(Path(path).read_text())
    seen = d.get("min_len_seen")
    return DecoderConfig(min_len=tuple(int(v) for v in d["min_len"]),
                         switch_penalty=int(d["switch_penalty"]),
                         min_len_seen=None if seen is None else int(seen),
                         change_weight=float(d.get("change_weight", 0.0)))
