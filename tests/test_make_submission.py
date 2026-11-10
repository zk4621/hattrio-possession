import os
import subprocess
import sys

from possession.paths import ROOT


def test_make_submission_without_data_exits_cleanly(tmp_path):
    env = dict(os.environ, MTAH_DATA_DIR=str(tmp_path))
    r = subprocess.run([sys.executable, "scripts/make_submission.py", "--team", "t",
                        "--proba", "rules"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 2
    assert "match data not found" in (r.stdout + r.stderr)
    assert "Traceback" not in r.stderr
