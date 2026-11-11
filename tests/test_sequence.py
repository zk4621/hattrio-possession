import numpy as np
from scipy.special import erf

from helpers import scene_handover
from possession.decode import DecoderConfig
from possession.features import build_features
from possession.labels import A, D, H, X
from possession.pipeline import predict_codes
from possession.rules import rule_proba
from possession.sequence import SequenceNet, load_sequence, transform


def random_net(d_feat, ch=4, dilations=(1, 2), flags=None, seed=0):
    rng = np.random.default_rng(seed)
    flags = np.arange(d_feat) if flags is None else np.asarray(flags)
    d_in = d_feat + len(flags)
    w = {"inp.weight": rng.normal(size=(ch, d_in, 1)), "inp.bias": rng.normal(size=ch),
         "out.weight": rng.normal(size=(3, ch, 1)), "out.bias": rng.normal(size=3)}
    for i, _ in enumerate(dilations):
        w[f"blocks.{i}.conv.weight"] = rng.normal(size=(ch, ch, 3))
        w[f"blocks.{i}.conv.bias"] = rng.normal(size=ch)
        w[f"blocks.{i}.norm.weight"] = rng.normal(size=ch)
        w[f"blocks.{i}.norm.bias"] = rng.normal(size=ch)
        w[f"blocks.{i}.proj.weight"] = rng.normal(size=(ch, ch, 1))
        w[f"blocks.{i}.proj.bias"] = rng.normal(size=ch)
    w.update(mu=np.zeros(d_feat, np.float32), sd=np.ones(d_feat, np.float32), flags=flags,
             dilations=np.array(dilations), step=np.array(5))
    return SequenceNet(w), w


def direct_logits(w, x, dilations):
    """The same network written as loops over time: zero padding, dilated kernels of 3."""
    T = len(x)
    h = np.array([w["inp.weight"][:, :, 0] @ x[t] + w["inp.bias"] for t in range(T)])
    for i, d in enumerate(dilations):
        W = w[f"blocks.{i}.conv.weight"]
        new = []
        for t in range(T):
            z = w[f"blocks.{i}.conv.bias"].copy()
            for j, s in enumerate((t - d, t, t + d)):
                if 0 <= s < T:
                    z += W[:, :, j] @ h[s]
            z = (z - z.mean()) / np.sqrt(z.var() + 1e-5)
            z = z * w[f"blocks.{i}.norm.weight"] + w[f"blocks.{i}.norm.bias"]
            z = 0.5 * z * (1 + erf(z / np.sqrt(2)))
            new.append(h[t] + w[f"blocks.{i}.proj.weight"][:, :, 0] @ z + w[f"blocks.{i}.proj.bias"])
        h = np.array(new)
    return np.array([w["out.weight"][:, :, 0] @ h[t] + w["out.bias"] for t in range(T)])


def test_logits_match_a_direct_loop():
    net, w = random_net(3, dilations=(1, 2))
    x = np.random.default_rng(1).normal(size=(17, 6))
    assert np.allclose(net.logits(x), direct_logits(w, x, (1, 2)), atol=1e-9)


def test_transform_flags_missing_values():
    x = np.array([[1.0, np.nan], [3.0, 4.0]], np.float32)
    z = transform(x, np.array([2.0, 4.0], np.float32), np.array([1.0, 2.0], np.float32),
                  np.array([1]))
    assert z.shape == (2, 3)
    assert np.allclose(z, [[-1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def test_proba_is_a_distribution_inside_halves_and_uniform_outside():
    net, _ = random_net(3)
    rng = np.random.default_rng(2)
    F = rng.normal(size=(200, 3)).astype(np.float32)
    F[50:60, 1] = np.nan
    P = net.proba(F, F[:, ::-1].copy(), [(20, 120), (130, 190)])
    assert P.shape == (200, 3) and np.allclose(P.sum(axis=1), 1.0)
    assert np.allclose(P[:20], 1 / 3) and np.allclose(P[120:130], 1 / 3)
    step = net.step_proba(F[20:120:5])
    swapped = net.step_proba(F[:, ::-1][20:120:5])[:, [1, 0, 2]]
    assert np.allclose(P[20:120:5], (step + swapped) / 2)


def test_missing_network_means_lightgbm_alone(tmp_path):
    assert load_sequence(tmp_path / "none.npz") is None


def test_pipeline_with_a_network_gives_valid_codes():
    ft = scene_handover()
    F, names = build_features(ft)
    net, _ = random_net(len(names), ch=4, dilations=(1,), flags=[])
    net.mu, net.sd = np.zeros(len(names), np.float32), np.ones(len(names), np.float32)
    codes = predict_codes(ft, rule_proba, DecoderConfig(), sequence=net)
    inside = ft.period > 0
    assert set(codes[inside].tolist()) <= {H, A, D}
    assert (codes[~inside] == X).all()
