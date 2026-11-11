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


def test_change_model_learns_a_rare_event_and_selects_columns_by_name():
    from possession.model import change_fn, train_change
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 4)).astype(np.float32)
    y = (X[:, 2] > 2.0).astype(np.int8)                      # about 2 % positives
    names = ["a", "b", "c", "d"]
    booster = train_change(X, y, names, n_rounds=30)
    fn = change_fn(booster)
    p = fn(X[:, ::-1], names[::-1])
    assert p.shape == (20000,) and ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > 0.5 > p[y == 0].mean()
    assert np.array_equal(p, change_fn(train_change(X, y, names, n_rounds=30))(X[:, ::-1], names[::-1]))


def test_sample_weights_change_the_model():
    X, y, names = synthetic()
    w = np.where(X[:, 1] > 0, 5.0, 1.0)
    a = train(X, y, names, n_rounds=10).model_to_string()
    b = train(X, y, names, n_rounds=10, weight=w).model_to_string()
    assert a != b and b == train(X, y, names, n_rounds=10, weight=w).model_to_string()
    from possession.model import train_change
    yc = (X[:, 2] > 1.5).astype(np.int8)
    assert (train_change(X, yc, names, n_rounds=5).model_to_string()
            != train_change(X, yc, names, n_rounds=5, weight=w).model_to_string())
