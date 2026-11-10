import pytest

from possession.labels import read_codes
from possession.paths import PUBLIC_GT, VENDOR
from possession.scoring import score


def test_vendored_scorer_reproduces_published_public_baseline():
    gt = read_codes(PUBLIC_GT)
    pred = read_codes(VENDOR / "evaluation/results/baseline_smooth50.csv")
    s = score(gt, pred)
    assert s["spell_f1"] == pytest.approx(0.3256, abs=5e-4)
    assert s["macro_f1"] == pytest.approx(0.6207, abs=5e-4)
    assert (s["n_gt_spells"], s["n_pred_spells"]) == (17, 26)
