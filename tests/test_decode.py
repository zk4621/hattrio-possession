import itertools
import time

import numpy as np

from possession.decode import DecoderConfig, decode, decode_periods, quantize
from possession.labels import A, D, H, X


def scores_for(labels, strength=3000):
    s = np.full((len(labels), 3), -strength, dtype=np.int32)
    s[np.arange(len(labels)), labels] = 0
    return s


def runs_of(y):
    y = np.asarray(y)
    b = np.flatnonzero(y[1:] != y[:-1]) + 1
    bounds = np.concatenate(([0], b, [len(y)]))
    return [(int(a), int(z), int(y[a])) for a, z in zip(bounds[:-1], bounds[1:])]


def objective(s, y, cfg):
    return int(s[np.arange(len(y)), y].sum()) - cfg.switch_penalty * int((np.diff(y) != 0).sum())


def valid(y, cfg):
    return all(z - a >= cfg.min_len[k] for a, z, k in runs_of(y))


def brute_force_best(s, cfg):
    best = None
    for y in itertools.product(range(3), repeat=len(s)):
        y = np.array(y)
        if valid(y, cfg):
            v = objective(s, y, cfg)
            best = v if best is None else max(best, v)
    return best


def test_flicker_is_removed():
    labels = np.array([H] * 100 + [A] * 5 + [H] * 100)
    assert (decode(scores_for(labels), DecoderConfig()) == H).all()


def test_real_change_is_kept_on_the_exact_frame():
    labels = np.array([H] * 100 + [A] * 100)
    out = decode(scores_for(labels), DecoderConfig())
    assert out.tolist() == labels.tolist()


def test_all_runs_respect_minimum_length():
    rng = np.random.default_rng(0)
    cfg = DecoderConfig(min_len=(25, 25, 50), switch_penalty=0)
    out = decode(rng.integers(-3000, 0, size=(2000, 3), dtype=np.int32), cfg)
    for start, stop, k in runs_of(out):
        assert stop - start >= cfg.min_len[k]


def test_matches_brute_force_optimum():
    rng = np.random.default_rng(1)
    cfg = DecoderConfig(min_len=(2, 2, 3), switch_penalty=300)
    for _ in range(20):
        s = rng.integers(-1000, 0, size=(10, 3), dtype=np.int32)
        assert objective(s, decode(s, cfg), cfg) == brute_force_best(s, cfg)


def test_short_input_returns_constant():
    s = scores_for(np.array([A] * 7))
    assert decode(s, DecoderConfig()).tolist() == [A] * 7


def test_decode_periods_marks_outside_and_forces_dead_before_live_from():
    period = np.array([0] * 10 + [1] * 300 + [0] * 10)
    proba = np.tile([0.8, 0.1, 0.1], (320, 1))
    out = decode_periods(proba, period, DecoderConfig(), live_from={1: 110})
    assert (out[:10] == X).all() and (out[-10:] == X).all()
    assert (out[10:110] == D).all() and (out[110:310] == H).all()


def test_non_finite_probabilities_do_not_crash():
    proba = np.full((200, 3), np.nan)
    out = decode_periods(proba, np.ones(200, dtype=np.int8), DecoderConfig())
    assert set(out.tolist()) <= {H, A, D}


def test_quantize_is_integer_and_clipped():
    q = quantize(np.array([[1.0, 0.0, 0.5]]))
    assert q.dtype == np.int32 and q.tolist() == [[0, -9210, -693]]


def test_long_decode_is_fast():
    rng = np.random.default_rng(2)
    s = rng.integers(-3000, 0, size=(160000, 3), dtype=np.int32)
    t = time.time()
    decode(s, DecoderConfig())
    assert time.time() - t < 5.0


def valid_per_start(y, mins):
    """Every maximal run of state k starting at a is at least mins[k][a] long."""
    return all(z - a >= mins[k][a] for a, z, k in runs_of(y))


def brute_force_best_per_start(s, mins, penalty):
    best = None
    for y in itertools.product(range(3), repeat=len(s)):
        y = np.array(y)
        if valid_per_start(y, mins):
            v = int(s[np.arange(len(y)), y].sum()) - penalty * int((np.diff(y) != 0).sum())
            best = v if best is None else max(best, v)
    return best


def test_seen_minimum_matches_brute_force_optimum():
    rng = np.random.default_rng(3)
    for _ in range(25):
        T = 9
        s = rng.integers(-1000, 0, size=(T, 3), dtype=np.int32)
        seen = rng.random(T) < 0.5
        cfg = DecoderConfig(min_len=(3, 3, 2), switch_penalty=200, min_len_seen=2)
        both = np.append(seen[:-1] & seen[1:], False)          # [s, s+2) seen
        poss = np.where(both, 2, 3)
        mins = [poss, poss, np.full(T, 2)]
        out = decode(s, cfg, seen=seen)
        assert valid_per_start(out, mins)
        v = int(s[np.arange(T), out].sum()) - 200 * int((np.diff(out) != 0).sum())
        assert v == brute_force_best_per_start(s, mins, 200)


def test_short_possession_is_kept_only_where_the_ball_is_seen():
    labels = np.array([H] * 100 + [A] * 10 + [H] * 100)
    cfg = DecoderConfig(min_len=(50, 50, 50), switch_penalty=0, min_len_seen=6)
    seen = np.ones(210, bool)
    assert decode(scores_for(labels), cfg, seen=seen).tolist() == labels.tolist()
    seen[95:115] = False                                   # the same 10 frames, ball not seen
    assert (decode(scores_for(labels), cfg, seen=seen) == H).all()


def test_without_seen_the_decoder_is_unchanged():
    rng = np.random.default_rng(4)
    s = rng.integers(-3000, 0, size=(3000, 3), dtype=np.int32)
    old = decode(s, DecoderConfig())
    assert np.array_equal(decode(s, DecoderConfig(min_len_seen=6)), old)       # no seen given
    assert np.array_equal(decode(s, DecoderConfig(), seen=np.ones(3000, bool)), old)


def test_decode_periods_passes_seen_through():
    period = np.array([0] * 10 + [1] * 210 + [0] * 10)
    labels = np.array([H] * 100 + [A] * 10 + [H] * 100)
    proba = np.full((230, 3), 0.01)
    proba[np.arange(10, 220), labels] = 0.98
    cfg = DecoderConfig(min_len=(50, 50, 50), switch_penalty=0, min_len_seen=6)
    out = decode_periods(proba, period, cfg, seen=np.ones(230, bool))
    assert out[10:220].tolist() == labels.tolist() and (out[:10] == X).all()
