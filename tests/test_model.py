import numpy as np
import pytest

from possession.model import load, predict_proba, proba_fn, save, train


def synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = np.where(X[:, 0] > 0.5, 0, np.where(X[:, 0] < -0.5, 1, 2))
    return X, y, [f"f{i}" for i in range(5)]


def test_train_learns_and_predicts_probabilities():
    X, y, names = synthetic()
    booster = train(X, y, names, n_rounds=30)
    P = predict_proba(booster, X)
    assert P.shape == (len(X), 3) and np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P.argmax(axis=1) == y).mean() > 0.95


def test_training_is_deterministic():
    X, y, names = synthetic()
    assert (train(X, y, names, n_rounds=20).model_to_string()
            == train(X, y, names, n_rounds=20).model_to_string())


def test_save_load_and_all_nan_rows(tmp_path):
    X, y, names = synthetic()
    booster = train(X, y, names, n_rounds=10)
    save(booster, tmp_path / "m.txt")
    again = load(tmp_path / "m.txt")
    assert np.array_equal(predict_proba(booster, X), predict_proba(again, X))
    assert np.allclose(predict_proba(again, np.full((3, 5), np.nan, dtype=np.float32)), 1 / 3)


def test_proba_fn_selects_columns_by_name():
    X, y, names = synthetic()
    booster = train(X, y, names, n_rounds=5)
    fn = proba_fn(booster)
    extra = np.hstack([X[:, ::-1], np.zeros((len(X), 1), dtype=np.float32)])
    assert np.array_equal(fn(extra, names[::-1] + ["unused"]), predict_proba(booster, X))
    with pytest.raises(ValueError):
        fn(X[:, :4], names[:4])
