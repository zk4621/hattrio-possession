import numpy as np

from helpers import empty_table, scene_handover, scene_with_long_gap
from possession.decode import DecoderConfig
from possession.labels import A, D, H, X
from possession.pipeline import predict_codes
from possession.rules import rule_proba


def test_pipeline_recovers_a_clean_handover():
    codes = predict_codes(scene_handover(), rule_proba, DecoderConfig())
    change = int(np.flatnonzero(np.diff(codes))[0]) + 1
    assert (codes[:change] == H).all() and (codes[change:] == A).all()
    assert abs(change - 200) <= 5


def test_pipeline_carries_state_through_a_ball_gap():
    codes = predict_codes(scene_with_long_gap(), rule_proba, DecoderConfig())
    assert (codes == H).all()


def test_pipeline_survives_empty_tracking():
    ft = empty_table(T=500, periods=[(1, 50, 450)])
    codes = predict_codes(ft, rule_proba, DecoderConfig())
    assert (codes[:50] == X).all() and (codes[450:] == X).all()
    assert set(codes[50:450].tolist()) <= {H, A, D}
