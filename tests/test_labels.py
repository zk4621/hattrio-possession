import numpy as np

from possession.labels import A, D, H, X, read_codes, swap_codes, write_submission
from possession.paths import vendor_import


def test_written_submission_passes_official_validation(tmp_path):
    pm = vendor_import("possession_metrics")
    codes = np.full(199319, X, dtype=np.int8)
    codes[11185:90652] = H
    codes[110808:190198] = D
    path = tmp_path / "team.csv"
    write_submission(path, codes)
    assert path.read_text().splitlines()[:2] == ["frame,label", "0,X"]
    assert pm.validate_submission(pm.read_label_csv(path), 199319) is None
    assert np.array_equal(read_codes(path), codes)


def test_swap_codes_exchanges_only_teams():
    assert swap_codes(np.array([H, A, D, X])).tolist() == [A, H, D, X]
