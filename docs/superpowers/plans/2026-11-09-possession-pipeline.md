# Ball-Possession Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `outputs/<team>.csv`, a per-frame `H`/`A`/`D`/`X` possession label for all 199,319 frames of the Barcelona–Athletic tactical video, maximising the challenge's `spellF1`, from a repository that regenerates the file exactly.

**Architecture:** Tracking from any source is loaded into one `FrameTable`. Per-frame features, many of which need no ball, feed a LightGBM classifier trained on seven public DFL matches whose clean tracking has been corrupted with the long ball blackouts measured on the provided tracking. A minimum-duration Viterbi decoder turns probabilities into possession segments. Two extra ball sources (broadcast tracking, camera view centre) are measured first and built in only if they pass a gate.

**Tech Stack:** Python 3.12, uv, numpy, pandas, LightGBM, pytest. No kloppy.

**Spec:** `docs/superpowers/specs/2026-11-09-possession-design.md` — read it first, especially "What the data showed". The challenge's own documentation is vendored in Task 1 under `third_party/mtah2027/` (`CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`).

**Revision 2 (2026-11-10).** The competition data is now unpacked in `data/`. Measuring it showed that the ball is lost in 147 long blackouts (median 11 s), not in short gaps. Tasks 4, 7, 9, 10 and 11 are built around that.

## Global Constraints

These apply to every task. The first four are competition rules; breaking one disqualifies the team.

- **Competition data never enters your context.** You are a cloud-hosted LLM and the terms forbid giving it the match data or extracts. Never `Read`, `cat`, `head`, `grep` or print per-frame content of anything under `data/` or `cache/atd_*`. Scripts that touch those files may print only aggregates: counts, rates, quantiles, scores, hashes. `scripts/describe_data.py` already exists and follows this rule; use it as the model. External data (`external/`) and everything under `third_party/` is public and may be read freely.
- **No hand labels and no frame-specific constants.** Do not write any constant tied to a particular frame of the match, except the half boundaries the organisers publish in `third_party/mtah2027/games.py`. Do not hard-code the second-half kick-off frame; derive it (Task 6).
- **The public ground truth is a correctness check only.** `third_party/mtah2027/evaluation/gt/BAR-ATH_possession_gt_public.csv` may be scored against in the tests of Task 1 and in the gate of Task 11. Never train on it, never choose between two variants with it, never tune a number against it. Every hyperparameter is chosen on DFL matches.
- **Never commit** `data/`, `external/`, `cache/`, `outputs/`. Commit locally after each task. Never push and never create a remote; the human publishes the repository.
- **Determinism.** Every random draw uses an explicit seed. LightGBM runs with `deterministic=True`. The decoder works on `int32` scores. Regenerating the submission must give the same SHA-256.
- **Units.** Positions in metres, origin at the pitch centre, x along the pitch length. Speeds in m/s. 25 frames per second. Label codes `H=0, A=1, D=2, X=3`, identical to the official scorer.
- **Tests never need `data/` or `external/`.** They use synthetic fixtures and the vendored public files.
- **STOP gates.** Steps marked **STOP** have a pass condition. If it fails, stop, report the printed numbers to the human, and wait. Do not work around a gate.
- Run everything through uv: `uv run pytest`, `uv run python scripts/...`.

## Review Focus

Conditions no happy-path test exercises, most likely first. Each has a test in the task that owns the code.

1. **Team orientation.** `H` must be the first `<Team>` in the tracking metadata. A swapped mapping produces a mirrored submission that still validates. Tests: Task 2 `test_first_team_is_home`; gate in Task 11.
2. **A blackout longer than every feature window** (the longest is 5,537 frames). Gap counters must clip, windowed features must stay defined, and the model must still get player features. Test: Task 4 `test_long_blackout_keeps_player_features`.
3. **No tracked objects inside a graded half** (before the second-half kick-off, tracker dropouts). Features become all-NaN; the pipeline must still emit `H`/`A`/`D`, never `X`, never crash. Test: Task 6 `test_pipeline_survives_empty_tracking`.
4. **Competition data absent.** Every script must exit with code 2 and the vendored "match data not found" message, not a traceback. Test: Task 6 `test_make_submission_without_data_exits_cleanly`.
5. **A half shorter than the minimum segment length, or with constant evidence.** The decoder must return one constant label, not fail. Test: Task 5 `test_short_input_returns_constant`.

## File Structure

```
pyproject.toml, uv.lock, .python-version, .gitignore, README.md
third_party/mtah2027/        vendored challenge repo (MIT), unmodified
src/possession/
  paths.py        repo paths, data dir, import of vendored modules
  labels.py       label codes, CSV read/write
  scoring.py      thin wrapper over the official scorer
  frames.py       FrameTable
  io_atd.py       provided tracking (FIFA/EPTS text) -> FrameTable; kick-off tags
  io_dfl.py       DFL position XML -> FrameTable + labels
  features.py     FrameTable -> feature matrix
  decode.py       minimum-duration Viterbi
  rules.py        rule-based probabilities; the starter baseline on a FrameTable
  pipeline.py     FrameTable -> label codes
  degrade.py      blackout and player-loss profile: measure from a table, apply to a table
  model.py        LightGBM train / predict / save / load
  proxy.py        camera view centre as a coarse ball position
  sync.py         broadcast timeline -> tactical timeline; ball merge
scripts/
  describe_data.py (exists)  fetch_external.py  describe_external.py  check_atd.py
  evaluate_external.py  measure_atd_profile.py  train.py  measure_proxy.py
  sync_broadcast.py  compare_domains.py  make_submission.py
models/            committed: possession_lgb.txt, decoder.json, atd_profile.json, sources.json, cv_report.json
tests/             one test file per module
docs/technical_document.md
```

---

### Task 1: Scaffold, vendored challenge code, labels, scoring

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/possession/__init__.py`, `src/possession/paths.py`, `src/possession/labels.py`, `src/possession/scoring.py`, `third_party/mtah2027/` (vendored), `third_party/mtah2027/VENDORED.md`
- Test: `tests/test_labels.py`, `tests/test_scoring.py`

**Interfaces:**
- Produces `possession.paths`: `ROOT, VENDOR, EXTERNAL, CACHE, OUTPUTS, MODELS: Path`; `PUBLIC_GT: Path`; `data_dir() -> Path` (env `MTAH_DATA_DIR`, else `ROOT/"data"`); `vendor_import(name: str) -> module` for `"games"`, `"possession_metrics"`, `"load_smart_tagging"`, `"load_homography"` (adds `VENDOR`, `VENDOR/"evaluation"`, `VENDOR/"starter"` to `sys.path`).
- Produces `possession.labels`: `H, A, D, X = 0, 1, 2, 3`; `write_submission(path, codes: np.ndarray) -> None`; `read_codes(path) -> np.ndarray[int8]`; `swap_codes(codes) -> np.ndarray` (H↔A).
- Produces `possession.scoring`: `score(gt_codes, pred_codes) -> dict` with keys `spell_f1, macro_f1, n_gt_spells, n_pred_spells, team_acc` (from the official `evaluate()` keys `spell_f1_mean, macro_f1, n_gt_spells, n_pred_spells, team_acc_gt_alive`).

- [ ] **Step 1: Initialise the repository and environment**

`data/` and `scripts/describe_data.py` already exist; keep both.

```bash
git init -b main
```

`pyproject.toml`: project `possession`, `requires-python = ">=3.12,<3.13"`, dependencies `numpy>=2.1,<3`, `pandas>=2.2,<3`, `lightgbm>=4.5,<5`; dependency group `dev = ["pytest>=8"]`; hatchling build with `packages = ["src/possession"]`; pytest `testpaths = ["tests"]`. `.python-version` contains `3.12`. `.gitignore`: `data/`, `external/`, `cache/`, `outputs/`, `.venv/`, `__pycache__/`, `*.mp4`, `.DS_Store`.

Run: `uv sync` — Expected: creates `.venv` and `uv.lock`. Then `git status --short` must not list anything under `data/`.

- [ ] **Step 2: Vendor the challenge repository at the pinned commit**

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 third_party/mtah2027
git -C third_party/mtah2027 checkout f8f2629ff1e0985f81d7e59d8413678514ee1530
rm -rf third_party/mtah2027/.git third_party/mtah2027/starter/*.ipynb
```

Write `third_party/mtah2027/VENDORED.md`: source URL, the commit hash, "MIT licence, see LICENSE, files unmodified".

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_scoring.py
def test_vendored_scorer_reproduces_published_public_baseline():
    gt = read_codes(PUBLIC_GT)
    pred = read_codes(VENDOR / "evaluation/results/baseline_smooth50.csv")
    s = score(gt, pred)
    assert s["spell_f1"] == pytest.approx(0.3256, abs=5e-4)
    assert s["macro_f1"] == pytest.approx(0.6207, abs=5e-4)
    assert (s["n_gt_spells"], s["n_pred_spells"]) == (17, 26)

# tests/test_labels.py
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
```

- [ ] **Step 4: Run to verify they fail** — `uv run pytest -q` — Expected: import errors.
- [ ] **Step 5: Implement `paths.py`, `labels.py`, `scoring.py`** with the interfaces above. `read_codes` uses the vendored `read_label_csv` + `label_codes`.
- [ ] **Step 6: Run to verify they pass** — `uv run pytest -q` — Expected: 3 passed.
- [ ] **Step 7: Commit** — `git add -A && git commit -m "chore: scaffold, vendor challenge repo, labels and scoring"`

---

### Task 2: FrameTable and the provided-tracking loader

**Files:**
- Create: `src/possession/frames.py`, `src/possession/io_atd.py`, `scripts/check_atd.py`
- Test: `tests/test_io_atd.py`

**Interfaces:**
- Produces `possession.frames`:

```python
FPS = 25

@dataclass
class FrameTable:
    pitch: tuple[float, float]   # (length_m, width_m)
    ball: np.ndarray             # (T, 2) float32, NaN where the ball is not tracked
    home: np.ndarray             # (T, Ph, 2) float32, NaN where a slot is empty
    away: np.ndarray             # (T, Pa, 2)
    home_speed: np.ndarray       # (T, Ph) float32, m/s
    away_speed: np.ndarray       # (T, Pa)
    period: np.ndarray           # (T,) int8: 1, 2, or 0 outside play
    proxy: np.ndarray | None = None   # (T, 2) coarse ball position, NaN where unknown (Task 9)

    @property
    def n_frames(self) -> int
    def swapped(self) -> "FrameTable"                        # home <-> away, new object, proxy kept
    def period_slices(self) -> list[tuple[int, int, int]]    # (period, start, stop), half-open
```

- Produces `possession.io_atd`:
  - `load_atd(meta_path, raw_path, n_frames: int, pitch: tuple[float, float], period_blocks: list[tuple[int, int, int]]) -> FrameTable` — `period_blocks` are `(period, first_frame, last_frame)` **inclusive**, as `games.half_video_blocks` returns them. Pass `[]` for a table with no periods.
  - `load_competition(source: str = "tactical") -> FrameTable` — paths from vendored `games.require(["atd_meta", "atd_raw"], data_dir=str(data_dir()), source=source)`, `n_frames` from `games.source_frames`, pitch `games.GAMES["BAR-ATH"]["pitch"]`; `period_blocks` is `games.half_video_blocks("BAR-ATH")` for `tactical` and `[]` for `broadcast`. It calls `games.require` first, so missing data always raises, and only then uses `cache/atd_<source>.npz` if that file is newer than the raw tracking file.

**Format** (confirmed on the real files by `scripts/describe_data.py`): the metadata XML lists `<Team id=...>` elements, the first being home, and `<Player teamId=...>` elements in slot order. Each raw line is `<frame_id>:<group>;<group>;...:<ball>`; every group, the ball's included, is `x,y,speed`; coordinates are normalised 0–1 and can fall slightly outside; missing values are spelled `NaN`. `frame_id` counts from 1; row index is `frame_id - 1`. Convert `x_m = (x - 0.5) * length`, `y_m = (y - 0.5) * width`. Still parse defensively: any field that does not parse as a finite float makes that observation missing, and a ball group with two fields is accepted.

- [ ] **Step 1: Write the failing tests**

```python
META = """<main><Metadata><Teams>
<Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
<Players><Player id="a" teamId="T1"/><Player id="b" teamId="T2"/>
<Player id="c" teamId="T1"/><Player id="d" teamId="T2"/></Players></Metadata></main>"""
RAW = ("1:0.5,0.5,1.0;NaN,NaN,NaN;0.25,0.5,0.0;1.0,1.0,2.5:0.5,0.5,3.0\n"
       "2:0.5,0.5,1.0;0.0,0.0,0.5;NaN,NaN,NaN;1.0,1.0,2.5:NaN,NaN,NaN\n"
       "3:0.5,0.5,1.0;0.0,0.0,0.5;0.25,0.5,0.0;1.0,1.0,2.5:0.75,0.25\n")

def _load(tmp_path):
    (tmp_path / "m.xml").write_text(META); (tmp_path / "r.txt").write_text(RAW)
    return load_atd(tmp_path / "m.xml", tmp_path / "r.txt", n_frames=4,
                    pitch=(100.0, 60.0), period_blocks=[(1, 1, 2)])

def test_first_team_is_home(tmp_path):
    ft = _load(tmp_path)
    assert ft.home.shape == (4, 2, 2) and ft.away.shape == (4, 2, 2)
    assert ft.home[0, 0].tolist() == [0.0, 0.0]        # slot a: (0.5, 0.5) -> centre
    assert ft.home[0, 1].tolist() == [-25.0, 0.0]      # slot c
    assert ft.away[0, 1].tolist() == [50.0, 30.0]      # slot d

def test_missing_observations_and_frame_indexing(tmp_path):
    ft = _load(tmp_path)
    assert np.isnan(ft.away[0, 0]).all()               # NaN group
    assert np.isnan(ft.ball[1]).all()                  # missing ball
    assert ft.ball[2].tolist() == [25.0, -15.0]        # two-field ball group
    assert np.isnan(ft.ball[3]).all()                  # no record for frame_id 4
    assert ft.home_speed[0].tolist() == [1.0, 0.0]
    assert ft.period.tolist() == [0, 1, 1, 0]          # inclusive block (1, 1, 2)
    assert ft.proxy is None

def test_swapped_and_period_slices(tmp_path):
    ft = _load(tmp_path)
    sw = ft.swapped()
    assert np.array_equal(sw.home, ft.away, equal_nan=True)
    assert np.array_equal(sw.away_speed, ft.home_speed, equal_nan=True)
    assert ft.period_slices() == [(1, 1, 3)]
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_io_atd.py -q`
- [ ] **Step 3: Implement `frames.py` and `io_atd.py`**
- [ ] **Step 4: Run to verify they pass** — Expected: 3 passed.
- [ ] **Step 5: Write `scripts/check_atd.py`** — loads `load_competition()` and prints only: record count, slots per team, ball-visible share over all frames, ball-visible share over `period > 0`, mean tracked players per frame over `period > 0`. It asserts the values already measured: 199,319 frames; slots `(24, 22)`; ball visible 56.94% ± 0.05 overall and 59.31% ± 0.05 on graded frames; 20.49 ± 0.05 tracked players. Exit code 2 with the vendored error text if the data is absent.
- [ ] **Step 6: Run `uv run python scripts/check_atd.py`** — Expected: all assertions pass. A failure means the loader is wrong.
- [ ] **Step 7: Commit** — `git commit -m "feat: FrameTable and provided-tracking loader"`

---

### Task 3: DFL dataset: download, loader, label-definition gate

**Files:**
- Create: `src/possession/io_dfl.py`, `scripts/fetch_external.py`, `scripts/describe_external.py`
- Test: `tests/test_io_dfl.py`

**Interfaces:**
- Produces `possession.io_dfl`:
  - `DFL_MATCH_IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]`
  - `HALFTIME_GAP = 250`
  - `load_dfl(meta_path, positions_path) -> tuple[FrameTable, np.ndarray]` — the table and its label codes.
  - `load_dfl_match(match_id: str) -> tuple[FrameTable, np.ndarray]` — finds `external/dfl/DFL_02_01_matchinformation_*_DFL-MAT-<id>.xml` and `DFL_04_03_positions_raw_observed_*_DFL-MAT-<id>.xml`; caches to `cache/dfl_<id>.npz`.

**Format.** Match-information XML: `<Team TeamId=... Role="home"|"guest">`, `<Environment PitchX=... PitchY=...>` (default 105.0 × 68.0 if absent). Position XML: `<FrameSet GameSection="firstHalf"|"secondHalf" TeamId=... PersonId=...>` holding `<Frame N=... X=... Y=... S=.../>`; coordinates are already metres from the pitch centre; `S` is km/h (divide by 3.6). The ball's frame set has `TeamId="BALL"` and each of its frames also carries `BallPossession` (`1` home, `2` away) and `BallStatus` (`1` alive, otherwise dead). Ignore frame sets whose `TeamId` is neither team nor `BALL` (referees) and any other `GameSection`.

**Timeline.** For each half, the ball frame set's `N` range defines the rows. Concatenate first half, then `HALFTIME_GAP` rows with `period=0`, NaN positions and label `X`, then the second half. Labels: `D` if `BallStatus != 1`, else `H` if `BallPossession == 1`, else `A`. A player with no frame at some `N` is NaN there. Parse with `xml.etree.ElementTree.iterparse` and clear elements as you go; the files are 350–420 MB.

- [ ] **Step 1: Write the failing test** — the fixture builds the two XML files in `tmp_path`: home `CLU-H`, guest `CLU-A`, pitch 105 × 68; first half `N` 10000–10002, second half `N` 100000–100001; ball `(BallPossession, BallStatus)` per frame `(1,0) (1,1) (2,1)` then `(2,1) (1,0)`; one home player `p1` present on all frames at `X="-10.0" Y="5.0" S="7.2"`; one guest player `p2` present only at `N=10001`; one frame set with `TeamId="referee"`.

```python
def test_load_dfl_maps_labels_units_and_gap(tmp_path):
    ft, y = load_dfl(*write_fixture(tmp_path))
    assert ft.n_frames == 3 + HALFTIME_GAP + 2
    assert y[:3].tolist() == [D, H, A]
    assert set(y[3:3 + HALFTIME_GAP].tolist()) == {X}
    assert y[-2:].tolist() == [A, D]
    assert ft.period[:3].tolist() == [1, 1, 1] and ft.period[3] == 0 and ft.period[-1] == 2
    assert ft.pitch == (105.0, 68.0)
    assert ft.home.shape[1] == 1 and ft.away.shape[1] == 1      # referee ignored
    assert ft.home[0, 0].tolist() == [-10.0, 5.0]
    assert ft.home_speed[0, 0] == pytest.approx(2.0)            # 7.2 km/h
    assert np.isnan(ft.away[0, 0]).all() and not np.isnan(ft.away[1, 0]).any()
    assert np.isnan(ft.ball[3]).all()                           # inside the gap
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_io_dfl.py -q`
- [ ] **Step 3: Implement `io_dfl.py`**
- [ ] **Step 4: Run to verify it passes**
- [ ] **Step 5: Write `scripts/fetch_external.py`** — reads `https://api.figshare.com/v2/articles/28196177`, downloads each entry of `files` (`name`, `size`, `download_url`, `computed_md5`) into `external/dfl/`, skips files already present with the right size, verifies MD5. 21 files, 2.63 GB. Licence CC-BY 4.0.
- [ ] **Step 6: Run `uv run python scripts/fetch_external.py`** — Expected: 21 files, all MD5 checks pass.
- [ ] **Step 7: Write `scripts/describe_external.py`** — for each match prints: frames in play, share of `D`, number of possessions, mean possession duration in seconds, the 10th percentile of possession durations (use the vendored `possession_spells` on the labels), share of frames with the ball tracked. Last line: the median over matches of the mean possession duration.
- [ ] **Step 8: STOP gate — run `uv run python scripts/describe_external.py`**

Pass condition: all 7 matches load, each has 120,000–160,000 frames in play, and the median of the mean possession durations is between 8 and 25 seconds. For reference, the challenge documents 17 possessions of mean 14.2 s and 19.6% dead ball in its public five minutes.

If the median is outside 8–25 s, DFL's possession flag follows a different definition from the challenge's. **STOP** and report the table. Do not invent a relabelling rule.

- [ ] **Step 9: Commit** — `git commit -m "feat: DFL dataset loader and download"`

---

### Task 4: Features

**Files:**
- Create: `src/possession/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `FrameTable`.
- Produces: `build_features(ft: FrameTable) -> tuple[np.ndarray, list[str]]` — `(T, F)` float32 and the column names; `BASE_NAMES`, `WINDOWED`, `PROXY_NAMES`, `PROXY_WINDOWED: list[str]`. Rows with `period == 0` are all NaN. Everything is computed per period; no window reads across a period boundary. Proxy columns are emitted only when `ft.proxy is not None`.

**Why two families.** The ball is absent for 41% of graded frames in blackouts that last a median of 11 s. Ball features describe the visible stretches; player features and the blackout counters are all the model has inside a blackout, so they get long windows.

**Definitions.** Within a period, let `s = +1` if the mean over the period of (away centroid x − home centroid x) is positive (home attacks +x), else `−1`. Let `b` be the ball position linearly interpolated over missing runs of at most 12 frames; elsewhere `b` is the observation or NaN. `L, W = ft.pitch`. `CLIP = 3000`.

Ball features (NaN where `b` is NaN):

| name | definition |
|---|---|
| `ball_x` | `s · b_x` |
| `ball_y_abs` | `abs(b_y)` |
| `out_margin` | `max(abs(b_x) − L/2, abs(b_y) − W/2)`; positive means outside the pitch |
| `ball_speed` | `norm(b[t+1] − b[t−1]) · 12.5` |
| `ball_acc` | `norm(v[t+1] − v[t−1]) · 12.5` with `v` the velocity vector behind `ball_speed` |
| `d1_home`, `d2_home`, `d1_away`, `d2_away` | distance from `b` to the nearest / second-nearest tracked player of the team; NaN if there are not enough players |
| `d_diff` | `d1_away − d1_home` |
| `n3_home`, `n3_away` | tracked players within 3 m of `b` |

Blackout features (always defined inside a period):

| name | definition |
|---|---|
| `ball_vis` | 1 if `b` is not NaN, else 0 |
| `gap_len` | length in frames of the run of `ball_vis == 0` containing `t`; 0 if visible; clipped at `CLIP` |
| `since_seen`, `until_seen` | frames since the last / until the next visible frame; 0 if visible; clipped at `CLIP`; `CLIP` if there is none |
| `near_team` | `+1` if visible, `d1_home < 2.0` and `d1_home < d1_away`; `−1` for the mirror case; else 0 |
| `last_near`, `next_near` | the most recent / the next non-zero `near_team` in the period; 0 if none |
| `since_near`, `until_near` | frames to that frame; clipped at `CLIP`; `CLIP` if none |
| `last_out`, `next_out` | `out_margin` at the last / next visible frame (the frame itself if visible); NaN if none |

Player features (NaN only if the team has no tracked player):

| name | definition |
|---|---|
| `n_home`, `n_away` | tracked players of the team |
| `cx_home`, `cx_away` | `s ·` mean x of the team's tracked players |
| `cx_gap` | `cx_away − cx_home` |
| `cvx_home`, `cvx_away` | `(cx[t+12] − cx[t−12]) · 25/24`; NaN at period edges |
| `spread_home`, `spread_away` | `sqrt(var(x) + var(y))` over the team's tracked players |
| `len_home`, `len_away` | max − min of x over the team's tracked players |
| `wid_home`, `wid_away` | max − min of y |
| `line_home` | second-smallest `s·x` among home players (the defensive line, skipping the keeper); NaN with fewer than 2 players |
| `line_away` | second-largest `s·x` among away players |
| `speed_home`, `speed_away` | mean provided speed of the team's tracked players |
| `mean_speed` | mean provided speed over all tracked players |
| `frac_slow` | share of tracked players with speed below 1.0 m/s |

`WINDOWED = [ball_vis, d1_home, d1_away, d_diff, ball_speed, out_margin, near_team, mean_speed, frac_slow, cvx_home, cvx_away, cx_home, cx_away, spread_home, spread_away, speed_home, speed_away]`. For each, add nine NaN-ignoring means clipped at the period edges: centred `[t−w, t+w]` for `w ∈ {12, 50, 250}` named `<name>_m12`, `_m50`, `_m250`; past `[t−w, t−1]` and future `[t+1, t+w]` for `w ∈ {25, 125, 500}` named `<name>_p25`, `_f25`, `_p125`, `_f125`, `_p500`, `_f500`. Use cumulative sums, not loops over frames.

Proxy features, only when `ft.proxy is not None`, with `q = ft.proxy`: `proxy_x = s·q_x`; `proxy_y_abs = abs(q_y)`; `proxy_rel_home = proxy_x − cx_home`; `proxy_rel_away = proxy_x − cx_away`; `proxy_vx = (proxy_x[t+12] − proxy_x[t−12]) · 25/24`. `PROXY_WINDOWED = [proxy_vx, proxy_rel_home, proxy_rel_away]`, with the same nine windows.

Column order: `BASE_NAMES` (the three tables, in the order given), the windowed columns in `WINDOWED` order, then `PROXY_NAMES` and its windowed columns if present.

**Behaviour under `ft.swapped()`**, which the symmetry test pins:
- exchange: `(d1_home, d1_away)`, `(d2_home, d2_away)`, `(n3_home, n3_away)`, `(n_home, n_away)`, `(spread_home, spread_away)`, `(len_home, len_away)`, `(wid_home, wid_away)`, `(speed_home, speed_away)`;
- exchange and negate: `(cx_home, cx_away)`, `(cvx_home, cvx_away)`, `(line_home, line_away)`, `(proxy_rel_home, proxy_rel_away)`;
- negate: `ball_x`, `d_diff`, `near_team`, `last_near`, `next_near`, `proxy_x`, `proxy_vx`;
- unchanged: everything else, including `cx_gap`.

- [ ] **Step 1: Write the failing tests** — a helper `make_table(T, ball, home, away, period=None, proxy=None)` builds a `FrameTable` on a 105 × 68 pitch with zero speeds unless given; `random_table(seed, T, periods=None, proxy=False)` makes smooth random motion for 11 players per team with the ball missing on a few random stretches; `periods` are `(period, start, stop)` half-open.

```python
def col(F, names, n): return F[:, names.index(n)]

def test_nearest_player_distances():
    # home player on the ball at the centre, away player 10 m away, 30 frames
    F, names = build_features(scene_home_on_ball())
    assert np.allclose(col(F, names, "d1_home"), 0.0, atol=1e-5)
    assert np.allclose(col(F, names, "d1_away"), 10.0, atol=1e-4)
    assert np.allclose(col(F, names, "d_diff"), 10.0, atol=1e-4)
    assert (col(F, names, "near_team") == 1).all()
    assert np.isnan(col(F, names, "d2_home")).all()        # only one home player

def test_short_gap_is_interpolated_long_gap_is_not():
    # 200 frames; ball missing on frames 10..19 (10 frames) and on 60..159 (100 frames)
    F, names = build_features(scene_with_gaps([(10, 20), (60, 160)]))
    vis = col(F, names, "ball_vis")
    assert (vis[10:20] == 1).all() and (vis[60:160] == 0).all()
    assert (col(F, names, "gap_len")[60:160] == 100).all() and col(F, names, "gap_len")[59] == 0
    assert col(F, names, "since_seen")[60:160].tolist() == list(range(1, 101))
    assert col(F, names, "until_seen")[60:160].tolist() == list(range(100, 0, -1))
    assert np.isnan(col(F, names, "d1_home")[60:160]).all()

def test_long_blackout_keeps_player_features():
    # 9000 frames, ball visible only on frames 0..99 and 8900..8999
    F, names = build_features(scene_with_gaps([(100, 8900)], T=9000))
    inside = slice(100, 8900)
    assert col(F, names, "gap_len")[inside].max() == 3000          # clipped
    assert col(F, names, "since_seen")[inside].max() == 3000
    assert np.isfinite(col(F, names, "cx_home")[inside]).all()
    assert np.isfinite(col(F, names, "mean_speed_m250")[inside]).all()
    assert np.isfinite(col(F, names, "last_out")[inside]).all()
    assert np.isnan(col(F, names, "d_diff_m50")[4000:5000]).all()   # no ball anywhere near

def test_out_margin_sign():
    # ball at (0, 35) on a 105 x 68 pitch is 1 m outside the touchline
    F, names = build_features(scene_ball_at(0.0, 35.0))
    assert np.allclose(col(F, names, "out_margin"), 1.0, atol=1e-5)

def test_swap_symmetry():
    ft = random_table(seed=0, T=600, proxy=True)
    F, names = build_features(ft); G, _ = build_features(ft.swapped())
    for a, b in [("d1_home", "d1_away"), ("n3_home", "n3_away"), ("n_home", "n_away"),
                 ("spread_home", "spread_away"), ("len_home", "len_away"), ("speed_home", "speed_away")]:
        assert np.allclose(col(G, names, a), col(F, names, b), equal_nan=True, atol=1e-4)
    for a, b in [("cx_home", "cx_away"), ("cvx_home", "cvx_away"), ("line_home", "line_away"),
                 ("proxy_rel_home", "proxy_rel_away")]:
        assert np.allclose(col(G, names, a), -col(F, names, b), equal_nan=True, atol=1e-4)
    for n in ["ball_x", "d_diff", "near_team", "last_near", "next_near", "d_diff_m50", "proxy_x", "proxy_vx"]:
        assert np.allclose(col(G, names, n), -col(F, names, n), equal_nan=True, atol=1e-4)
    for n in ["ball_vis", "gap_len", "out_margin", "ball_speed", "mean_speed", "frac_slow", "cx_gap"]:
        assert np.allclose(col(G, names, n), col(F, names, n), equal_nan=True, atol=1e-4)

def test_periods_are_independent_and_outside_is_nan():
    ft = random_table(seed=1, T=600, periods=[(1, 0, 250), (2, 350, 600)])
    F, names = build_features(ft)
    assert np.isnan(F[250:350]).all()
    F2, _ = build_features(perturb_period(ft, 2))          # new random positions in period 2 only
    assert np.array_equal(F[:250], F2[:250], equal_nan=True)

def test_duplicate_and_extra_players():
    # 14 home tracks, two of them at the identical position; 11 away tracks
    F, names = build_features(scene_with_duplicates())
    assert np.isfinite(col(F, names, "d1_home")).all()
    assert (col(F, names, "n_home") == 14).all()

def test_shape_names_and_optional_proxy():
    F, names = build_features(random_table(seed=2, T=300))
    assert F.dtype == np.float32 and F.shape == (300, len(names))
    assert len(names) == len(set(names)) == len(BASE_NAMES) + 9 * len(WINDOWED)
    assert not any(n.startswith("proxy") for n in names)
    G, gnames = build_features(random_table(seed=2, T=300, proxy=True))
    assert len(gnames) == len(names) + len(PROXY_NAMES) + 9 * len(PROXY_WINDOWED)
    assert gnames[:len(names)] == names
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_features.py -q`
- [ ] **Step 3: Implement `build_features`**, vectorised over frames.
- [ ] **Step 4: Run to verify they pass** — Expected: 8 passed.
- [ ] **Step 5: Check speed** — time `build_features(load_dfl_match("J03WPY")[0])` in a one-off `uv run python -c` call. Expected: under 90 s, and no all-NaN column inside the periods.
- [ ] **Step 6: Commit** — `git commit -m "feat: per-frame features"`

---

### Task 5: Minimum-duration decoder

**Files:**
- Create: `src/possession/decode.py`
- Test: `tests/test_decode.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class DecoderConfig:
    min_len: tuple[int, int, int] = (25, 25, 50)   # H, A, D in frames
    switch_penalty: int = 5000                      # same unit as the scores

def quantize(proba: np.ndarray) -> np.ndarray       # (T,3) float -> int32: round(1000 * log(clip(p, 1e-4, 1)))
def decode(scores: np.ndarray, cfg: DecoderConfig) -> np.ndarray          # one period: (T,3) int32 -> (T,) int8 in {0,1,2}
def decode_periods(proba: np.ndarray, period: np.ndarray, cfg: DecoderConfig,
                   live_from: dict[int, int] | None = None) -> np.ndarray  # full timeline; X where period == 0
```

**Objective.** `decode` returns the labelling `y` maximising `sum_t scores[t, y_t] − switch_penalty · (number of t with y_t != y_{t−1})`, subject to every maximal constant run of state `k` having length at least `min_len[k]`. If `T < min(min_len)`, or no labelling satisfies the constraint, return the constant label with the largest column sum.

**Algorithm.** With prefix sums `C_k`, let `V[t, k]` be the best score of a valid labelling of `0..t` ending in state `k`:

```
V[t, k] = max( V[t−1, k] + scores[t, k],
               max_{j != k} V[t − m_k, j] + (C_k[t+1] − C_k[t − m_k + 1]) − switch_penalty )
V[m_k − 1, k] = C_k[m_k]          # the first run starts at frame 0
```

Keep back-pointers; ties go to staying in the same state, then to the lowest state index. All arithmetic in `int64`.

`decode_periods` replaces non-finite probability rows by `1/3`, quantises, and decodes each period separately. For `live_from = {period: frame}`, every frame of that period before `frame` gets scores `(−10**6, −10**6, 0)` before decoding, which forces `D`.

- [ ] **Step 1: Write the failing tests**

```python
def scores_for(labels, strength=3000):
    s = np.full((len(labels), 3), -strength, dtype=np.int32)
    s[np.arange(len(labels)), labels] = 0
    return s

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
        assert objective(s, decode(s, cfg), cfg) == brute_force_best(s, cfg)   # all 3**10 labellings

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
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_decode.py -q`
- [ ] **Step 3: Implement `decode.py`**
- [ ] **Step 4: Run to verify they pass** — Expected: 8 passed. A 160,000-frame decode must take under 5 s.
- [ ] **Step 5: Commit** — `git commit -m "feat: minimum-duration Viterbi decoder"`

---

### Task 6: Rule-based probabilities, pipeline, external evaluation, submission script

**Files:**
- Create: `src/possession/rules.py`, `src/possession/pipeline.py`, `scripts/evaluate_external.py`, `scripts/make_submission.py`
- Modify: `src/possession/io_atd.py` (add `kickoff_tag_frames`)
- Test: `tests/test_rules.py`, `tests/test_pipeline.py`, `tests/test_make_submission.py`

**Interfaces:**
- Consumes: `build_features`, `decode_periods`, `DecoderConfig`, `load_dfl_match`, `load_competition`, `score`, `write_submission`.
- Produces `possession.rules`:
  - `rule_proba(F: np.ndarray, names: list[str]) -> np.ndarray` — `(T, 3)` probabilities for H, A, D.
  - `baseline_codes(ft: FrameTable, window: int = 50) -> np.ndarray` — the starter baseline on a `FrameTable`: team of the nearest player on frames with the ball, nobody otherwise, then the centred majority vote of `third_party/mtah2027/starter/baseline.py::smooth` (ties to home, `D` where the window holds no vote); `X` where `period == 0`.
- Produces `possession.pipeline`: `predict_codes(ft, proba_fn, cfg: DecoderConfig, live_from: dict[int, int] | None = None) -> np.ndarray`, where `proba_fn(F, names) -> (T, 3)`.
- Produces `possession.io_atd.kickoff_tag_frames() -> dict[int, int]` — from the vendored `load_smart_tagging.load(data_dir=str(data_dir()))`: for half labels `"1st Half" -> 1`, `"2nd Half" -> 2`, the smallest `start_frame` among events whose `code` lower-cased is `set pieces`, whose `tags["Type"]` lower-cased is `kick off (start)` and whose `tags["Half"]` is that label. A half with no such tag is omitted. (The real file has exactly one such tag per half.)

**`rule_proba`.** Start from logits `(0, 0, 0)` per frame. These constants are football priors, fixed here, not tuned:
- If `ball_vis == 1` and `min(d1_home, d1_away) < 3.0`: with `m = clip(d_diff, −5, 5)`, add `0.8·m` to H and `−0.8·m` to A.
- If `ball_vis == 1` and `out_margin > 0.5`: add `4.0` to D.
- If `frac_slow_m50 > 0.7`: add `2.0` to D.
- NaN comparisons count as false. Return the softmax. A frame with no evidence is uniform, so the decoder carries the previous state through a blackout.
- It looks columns up by name and needs only `ball_vis, d1_home, d1_away, d_diff, out_margin, frac_slow_m50`.

The rules cannot see a turnover inside a blackout. They are the reference point and the fallback, not the answer to blackouts; that is the model's job (Tasks 7–8).

**Test scenes.** `scene_handover()` is 400 frames in one period: a home player on the ball and the away player 10 m away for frames 0–199, the reverse for 200–399. `scene_with_long_gap()` is 400 frames with the home player on the ball throughout and the ball missing on frames 100–249. In both, every player has speed 3.0 m/s, so the standing-players rule does not fire.

**`predict_codes`.** `F, names = build_features(ft)`; `proba = proba_fn(F, names)`; return `decode_periods(proba, ft.period, cfg, live_from)`.

**`scripts/evaluate_external.py --proba baseline|rules|model --degrade none|default|atd [--matches ...]`.** For each DFL match: load, optionally degrade (Task 7; until then only `none`), predict, `score` against the DFL labels. Prints one row per match (`spell_f1, macro_f1, n_gt_spells, n_pred_spells`) and the mean; writes `outputs/external_<proba>_<degrade>.json`. `baseline` uses `baseline_codes`; `model` loads `models/possession_lgb.txt` and `models/decoder.json` (Task 8).

**`scripts/make_submission.py --team NAME --proba rules|model`.** `ft = load_competition()`; `tags = kickoff_tag_frames()`; `live_from = {p: max(start, tags[p]) for p, start, stop in ft.period_slices() if p in tags}`; predict; `write_submission(OUTPUTS / f"{NAME}.csv", codes)`; validate with the vendored `validate_submission(table, 199319)`; print only: label counts, number of possessions, mean possession duration, the SHA-256 of the file. Any `IOError` from `games.require` is printed and the script exits with code 2.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rules.py
def test_rule_proba_follows_nearest_player_and_abstains_without_ball():
    names = ["ball_vis", "d1_home", "d1_away", "d_diff", "out_margin", "frac_slow_m50"]
    F = np.array([[1, 0.5, 6.0, 5.5, -10, 0.1],      # home on the ball
                  [1, 6.0, 0.5, -5.5, -10, 0.1],     # away on the ball
                  [0, np.nan, np.nan, np.nan, np.nan, 0.1],   # ball missing
                  [1, 9.0, 9.0, 0.0, 2.0, 0.1],      # ball outside the pitch
                  [0, np.nan, np.nan, np.nan, np.nan, 0.9]],  # everyone standing
                 dtype=np.float32)
    P = rule_proba(F, names)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[0].argmax() == H and P[1].argmax() == A
    assert np.allclose(P[2], 1 / 3)
    assert P[3].argmax() == D and P[4].argmax() == D

def test_baseline_codes_matches_starter_logic():
    codes = baseline_codes(scene_handover(), window=50)
    assert (codes[:150] == H).all() and (codes[250:] == A).all()

# tests/test_pipeline.py
def test_pipeline_recovers_a_clean_handover():
    codes = predict_codes(scene_handover(), rule_proba, DecoderConfig())
    change = int(np.flatnonzero(np.diff(codes))[0]) + 1
    assert (codes[:change] == H).all() and (codes[change:] == A).all()
    assert abs(change - 200) <= 5

def test_pipeline_carries_state_through_a_ball_gap():
    codes = predict_codes(scene_with_long_gap(), rule_proba, DecoderConfig())
    assert (codes == H).all()

def test_pipeline_survives_empty_tracking():
    ft = empty_table(T=500, periods=[(1, 50, 450)])      # all positions NaN
    codes = predict_codes(ft, rule_proba, DecoderConfig())
    assert (codes[:50] == X).all() and (codes[450:] == X).all()
    assert set(codes[50:450].tolist()) <= {H, A, D}

# tests/test_make_submission.py
def test_make_submission_without_data_exits_cleanly(tmp_path):
    env = dict(os.environ, MTAH_DATA_DIR=str(tmp_path))
    r = subprocess.run([sys.executable, "scripts/make_submission.py", "--team", "t", "--proba", "rules"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2
    assert "match data not found" in (r.stdout + r.stderr)
    assert "Traceback" not in r.stderr
```

- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Implement `rules.py`, `pipeline.py`, `kickoff_tag_frames`, and the two scripts**
- [ ] **Step 4: Run to verify they pass** — `uv run pytest -q` — Expected: all tests pass.
- [ ] **Step 5: Record the external reference numbers on clean tracking**

```bash
uv run python scripts/evaluate_external.py --proba baseline --degrade none
uv run python scripts/evaluate_external.py --proba rules --degrade none
```

Expected: two 7-row tables. On clean tracking `rules` should beat `baseline` on mean `spell_f1`; if it does not, re-read the decoder tests before going on. Report both means.

- [ ] **Step 6: First real submission file** — `uv run python scripts/make_submission.py --team dev_rules --proba rules` — Expected: "valid", label counts, a possession count, a hash. No `X` inside the halves: `H + A + D == 158857`.
- [ ] **Step 7: Commit** — `git commit -m "feat: rule-based pipeline, external evaluation, submission script"`

---

### Task 7: Degradation: long blackouts and missing players

**Files:**
- Create: `src/possession/degrade.py`, `scripts/measure_atd_profile.py`
- Modify: `scripts/evaluate_external.py` (enable `--degrade default|atd`)
- Test: `tests/test_degrade.py`

**What is being imitated** (see the spec): on the provided tracking the ball is visible in uninterrupted runs (median 345 frames) separated by blackouts (median 268 frames, 90th percentile 836, maximum 5,537); a team has 11 tracked players at the median, 9 at the 10th percentile, 2 at the 1st.

**Interfaces:**
- Produces:

```python
BOUNDARY_MARGIN = 2.0      # a ball is "at the boundary" if out_margin > -2.0 (within 2 m of a line, or outside)
DEFICIT_BLOCK = 250        # frames

@dataclass
class DegradeProfile:
    blackout_hazard: list[float]            # [2]: P(a blackout starts on the next frame | ball visible now), by context: 0 in field, 1 at boundary
    blackout_quantiles: list[list[float]]   # [2][101]: blackout length in frames at quantiles 0..100, by start context
    deficit_pmf: list[float]                # [12]: P(k of 11 players of a team untracked during a 250-frame block), k = 0..11
    jitter_std_m: float
    ball_jitter_std_m: float
    def to_json(self, path) -> None
    @classmethod
    def from_json(cls, path) -> "DegradeProfile"

DEFAULT_PROFILE: DegradeProfile
def measure_profile(ft: FrameTable) -> DegradeProfile
def degrade(ft: FrameTable, profile: DegradeProfile, seed: int) -> FrameTable
```

**`DEFAULT_PROFILE`** (for tests and for runs before the real profile exists): `blackout_hazard = [1/640, 1/640]`; both `blackout_quantiles` rows are the piecewise-linear interpolation through `(q, frames)` = `(0, 1), (10, 22), (25, 153), (50, 268), (75, 529), (90, 836), (95, 1278), (99, 3466), (100, 5537)`; `deficit_pmf = [0.60, 0.20, 0.10, 0.05, 0.02, 0.03, 0, 0, 0, 0, 0, 0]`; `jitter_std_m = 0.10`; `ball_jitter_std_m = 0.15`.

**`measure_profile`** uses only frames with `period > 0` and never pairs frames across a period boundary.
- A blackout is a maximal run of missing ball that does not touch a period edge. Its context is that of the last visible frame before it: boundary if `out_margin > −BOUNDARY_MARGIN` there, else in field.
- `blackout_hazard[c]` = blackouts starting in context `c` ÷ visible frames in context `c`. A context with fewer than 20 blackouts takes the pooled hazard and the pooled quantiles.
- `blackout_quantiles[c]` = quantiles 0..100 of blackout lengths in context `c`.
- `deficit_pmf`: cut each period into blocks of `DEFICIT_BLOCK` frames; for each block and team take `k = clip(round(11 − mean tracked players of the team), 0, 11)`; the pmf is the histogram of `k` over both teams.
- `jitter_std_m` = `1.4826 · MAD / sqrt(6)` of the second differences of player x and y over three consecutive observed frames of the same slot; `ball_jitter_std_m` likewise for the ball.

**`degrade`** returns a new table and does not modify its input. With `rng = np.random.default_rng(seed)`:
1. Ball: per period, walk forward. At a visible frame `t`, with probability `blackout_hazard[context(t)]` draw a length from `blackout_quantiles[context(t)]` by inverse CDF (linear between quantiles), round, at least 1, set that many following frames of the period to NaN, and continue after them. Contexts come from the input table.
2. Players: per period, team and `DEFICIT_BLOCK`-frame block, draw `k` from `deficit_pmf` and blank position and speed of the `k` players of that team who are farthest along x from the input ball at the block's first frame (from the team centroid if the ball is NaN there), for the whole block. Farthest-first imitates a camera that follows play; Task 11 checks that assumption.
3. Jitter: add Gaussian noise of standard deviation `sqrt(max(profile.jitter_std_m**2 − own**2, 0))` to remaining player positions, where `own` is the input's jitter by the estimator above; likewise for the ball.
4. `proxy` is carried over unchanged.

**`scripts/measure_atd_profile.py`**: `measure_profile(load_competition())` → `models/atd_profile.json`. Prints only the profile's numbers and the implied blackout share. If `models/sources.json` says the broadcast ball is merged (Task 10), it measures the merged table.

- [ ] **Step 1: Write the failing tests** — `smooth_table(T, periods=None)` builds 11 players per team and a ball whose positions are sums of a few low-frequency sinusoids with random frequencies and phases (second differences below 0.01 m), with no missing data inside the periods. Each player wanders over a region at least 20 m wide and the ball's x has a standard deviation above 15 m. `periods` are `(period, start, stop)` half-open, default one period covering everything; frames outside the periods are NaN. Put this and every other helper used by more than one test file in `tests/helpers.py`.

```python
def test_default_profile_gives_long_blackouts_on_about_41_percent():
    out = degrade(smooth_table(400000), DEFAULT_PROFILE, seed=1)
    missing = np.isnan(out.ball[:, 0])
    assert 0.30 <= missing.mean() <= 0.52
    lengths = run_lengths(missing)                       # lengths of the missing runs
    assert 150 <= np.median(lengths) <= 400
    assert (lengths >= 126).sum() / len(lengths) >= 0.6

def test_deficit_drops_whole_blocks_of_far_players():
    out = degrade(smooth_table(100000), DEFAULT_PROFILE, seed=4)
    tracked = (~np.isnan(out.home[:, :, 0])).sum(axis=1)
    assert 0.5 <= (tracked == 11).mean() <= 0.7
    assert tracked.min() >= 6
    block = tracked[:250]
    assert block.min() == block.max()                    # constant inside a block

def test_degrade_is_deterministic_and_pure():
    ft = smooth_table(20000); before = ft.ball.copy()
    a, b, c = degrade(ft, DEFAULT_PROFILE, 7), degrade(ft, DEFAULT_PROFILE, 7), degrade(ft, DEFAULT_PROFILE, 8)
    assert np.array_equal(a.ball, b.ball, equal_nan=True) and np.array_equal(a.home, b.home, equal_nan=True)
    assert not np.array_equal(a.ball, c.ball, equal_nan=True)
    assert np.array_equal(ft.ball, before, equal_nan=True)

def test_measure_recovers_what_degrade_applied():
    p = measure_profile(degrade(smooth_table(600000), DEFAULT_PROFILE, seed=3))
    pooled = np.mean(p.blackout_hazard)
    assert 1 / 900 <= pooled <= 1 / 450                  # 1/640 applied
    assert 180 <= p.blackout_quantiles[0][50] <= 380     # 268 applied
    assert 0.5 <= p.deficit_pmf[0] <= 0.7
    assert 0.05 <= p.jitter_std_m <= 0.20

def test_profile_json_roundtrip(tmp_path):
    DEFAULT_PROFILE.to_json(tmp_path / "p.json")
    assert DegradeProfile.from_json(tmp_path / "p.json") == DEFAULT_PROFILE

def test_blackouts_do_not_cross_periods():
    ft = smooth_table(40000, periods=[(1, 0, 19000), (2, 21000, 40000)])
    out = degrade(ft, DEFAULT_PROFILE, seed=2)
    assert np.isnan(out.ball[19000:21000]).all() and out.period.tolist() == ft.period.tolist()
    assert not np.isnan(out.ball[[0, 21000]]).any()      # a blackout never starts before a visible frame
```

- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Implement `degrade.py` and the script; wire `--degrade` into `evaluate_external.py`** (`default` → `DEFAULT_PROFILE`, `atd` → `models/atd_profile.json`; seed = index of the match in `DFL_MATCH_IDS`).
- [ ] **Step 4: Run to verify they pass**
- [ ] **Step 5: Measure the real profile** — `uv run python scripts/measure_atd_profile.py` — Expected: writes `models/atd_profile.json`; the pooled blackout median is near 268 frames and the implied blackout share near 41%.
- [ ] **Step 6: Reference numbers under degradation**

```bash
uv run python scripts/evaluate_external.py --proba baseline --degrade atd
uv run python scripts/evaluate_external.py --proba rules --degrade atd
```

Report both means. The challenge documents `spell_f1 = 0.2433` and `macro_f1 = 0.6724` for this baseline on the real match. This comparison is the main evidence about how realistic the simulation is: report how far the baseline on degraded DFL is from those two numbers. Do not adjust the profile to close the distance.

- [ ] **Step 7: Commit** — `git commit -m "feat: blackout and player-loss degradation profile"` including `models/atd_profile.json` (aggregates only).

---

### Task 8: Model training, cross-validation, decoder tuning

**Files:**
- Create: `src/possession/model.py`, `scripts/train.py`
- Modify: `scripts/evaluate_external.py`, `scripts/make_submission.py` (`--proba model`)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces `possession.model`:

```python
LGB_PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "seed": 0, "deterministic": True,
              "force_row_wise": True, "num_threads": 8, "verbose": -1}
N_ROUNDS = 400
def train(X: np.ndarray, y: np.ndarray, names: list[str], params: dict = LGB_PARAMS, n_rounds: int = N_ROUNDS) -> lgb.Booster
def predict_proba(booster: lgb.Booster, X: np.ndarray) -> np.ndarray     # (T, 3); rows that are all-NaN -> 1/3
def save(booster, path) -> None
def load(path) -> lgb.Booster
def proba_fn(booster) -> Callable[[np.ndarray, list[str]], np.ndarray]
    # selects the booster's own features from the given columns by name, in the booster's order;
    # extra columns are ignored; a missing one raises ValueError
```

`models/sources.json` is created in Tasks 9 and 10. Wherever it is read, an absent file means both sources are off.

**`scripts/train.py --profile atd|default [--drop FEATURE ...]`**
1. For each DFL match `i`: load; attach a simulated proxy if `models/sources.json` enables it (Task 9); `degrade(ft, profile, seed=i)`; build features and labels for the table and for `ft.swapped()` with `swap_codes(labels)`. Cache under `cache/`. Keep, per match, the boolean mask of frames the degradation blacked out.
2. Leave-one-match-out: for each held-out match train on the other six, both orientations, every second frame, rows with label `X` dropped. Predict the held-out match, original orientation, all frames.
3. Decoder grid on the out-of-fold probabilities: possession `min_len ∈ {6, 12, 25, 38, 50}` (same for H and A), dead-ball `min_len ∈ {25, 50, 75, 125}`, `switch_penalty ∈ {0, 2000, 5000, 10000, 20000}`. Choose the configuration with the highest mean `spell_f1` over the seven matches; break ties by higher mean `macro_f1`. Write `models/decoder.json`.
4. Train the final model on all seven matches, both orientations, every second frame. Write `models/possession_lgb.txt`.
5. Write `models/cv_report.json` and print it: per match and mean `spell_f1`/`macro_f1` for `baseline_codes`, for `rule_proba` with the default `DecoderConfig`, and for the model with the chosen configuration; for the model also the frame accuracy separately on blacked-out frames and on visible frames; the chosen configuration; the 20 most important features by gain.

- [ ] **Step 1: Write the failing tests**

```python
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
    assert train(X, y, names, n_rounds=20).model_to_string() == train(X, y, names, n_rounds=20).model_to_string()

def test_save_load_and_all_nan_rows(tmp_path):
    X, y, names = synthetic()
    booster = train(X, y, names, n_rounds=10); save(booster, tmp_path / "m.txt")
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
```

- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Implement `model.py`, `scripts/train.py`, and `--proba model` in both scripts** (`make_submission.py` and `evaluate_external.py` read `models/possession_lgb.txt` and `models/decoder.json`).
- [ ] **Step 4: Run to verify they pass** — `uv run pytest -q`
- [ ] **Step 5: Train** — `uv run python scripts/train.py --profile atd`. Expect roughly 45–90 minutes.
- [ ] **Step 6: STOP gate — read `models/cv_report.json`**

Pass condition: mean out-of-fold `spell_f1` of the model is higher than that of the rules, and the rules are higher than the baseline. If the model does not beat the rules, **STOP** and report the three means and the per-match table.

Report in any case the model's accuracy on blacked-out frames against visible frames. That difference is how much the blackouts cost, and it decides how much Tasks 9 and 10 are worth.

- [ ] **Step 7: First model submission** — `uv run python scripts/make_submission.py --team dev_model --proba model` — Expected: valid; report label counts, possession count, mean duration.
- [ ] **Step 8: Commit** — `git commit -m "feat: LightGBM possession model, cross-validation, tuned decoder"` including `models/possession_lgb.txt`, `models/decoder.json`, `models/cv_report.json`.

---

### Task 9: Camera view centre as a coarse ball position (measure, then adopt or drop)

**Files:**
- Create: `src/possession/proxy.py`, `scripts/measure_proxy.py`
- Test: `tests/test_proxy.py`

**Idea.** The tactical camera follows play. The homography gives, for every frame, the pitch point shown at the image centre. If that point tracks the ball, it is a ball position that never blacks out.

**Interfaces:**
- Produces:

```python
IMAGE_CENTRE = (960.0, 540.0)
def camera_centre(H: np.ndarray, valid: np.ndarray, pitch: tuple[float, float]) -> np.ndarray
    # H: (T,3,3) mapping normalised pitch (x, y, 1) to pixels. Returns (T,2) metres, NaN where valid == 0.
def measure_proxy(proxy: np.ndarray, ft: FrameTable) -> dict
    # keys: std_x, corr_x, med_abs_dx, med_abs_dy, best_lag, err_std_x, err_std_y, err_corr_frames
def simulate_proxy(ft: FrameTable, stats: dict, seed: int) -> np.ndarray     # (T,2) for a table with a clean ball
```

- `camera_centre`: apply the inverse of `H[t]` to `IMAGE_CENTRE`, divide by the third coordinate, convert normalised to metres as in Task 2.
- `measure_proxy`, on frames with `period > 0`: `std_x` = standard deviation of proxy x; over frames where the ball is visible: `best_lag` = the shift in `[−50, 50]` frames of the proxy that maximises the correlation of proxy x with ball x; at that lag `corr_x`, `med_abs_dx`, `med_abs_dy`, and the error `e = proxy − ball` with `err_std_x`, `err_std_y`; `err_corr_frames` = the smallest lag at which the autocorrelation of `e_x` within visible runs falls below 0.5 (capped at 250).
- `simulate_proxy`: `proxy[t] = ball[t − best_lag] + noise[t]`, with the table's ball linearly interpolated if needed, and noise that is white Gaussian noise smoothed by a moving average of `err_corr_frames` frames and rescaled to `err_std_x` / `err_std_y`. NaN outside periods.
- Lag convention, shared by both functions: a positive `best_lag` means the proxy trails the ball. `measure_proxy` returns the `L` maximising the correlation of `proxy[t + L]` with `ball[t]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_camera_centre_inverts_the_homography():
    # normalised (x, y) -> pixel (1920 x, 1080 y): the image centre shows the pitch centre
    H0 = np.array([[1920.0, 0, 0], [0, 1080.0, 0], [0, 0, 1]])
    # shifted so that the image centre shows normalised (0.75, 0.5)
    H1 = np.array([[1920.0, 0, -480.0], [0, 1080.0, 0], [0, 0, 1]])
    out = camera_centre(np.stack([H0, H1, H0]), np.array([1, 1, 0]), pitch=(105.0, 68.0))
    assert np.allclose(out[0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(out[1], [26.25, 0.0], atol=1e-6)
    assert np.isnan(out[2]).all()

def test_measure_recovers_lag_from_a_quiet_proxy():
    ft = smooth_table(60000)
    quiet = {"best_lag": 10, "err_std_x": 0.5, "err_std_y": 0.5, "err_corr_frames": 5}
    assert abs(measure_proxy(simulate_proxy(ft, quiet, seed=0), ft)["best_lag"] - 10) <= 3

def test_measure_recovers_noise_from_a_noisy_proxy():
    ft = smooth_table(60000)
    noisy = {"best_lag": 0, "err_std_x": 6.0, "err_std_y": 4.0, "err_corr_frames": 50}
    proxy = simulate_proxy(ft, noisy, seed=0)
    m = measure_proxy(proxy, ft)
    assert 4.5 <= m["err_std_x"] <= 8.5 and 3.0 <= m["err_std_y"] <= 6.0
    assert m["corr_x"] > 0.8
    assert np.array_equal(proxy, simulate_proxy(ft, noisy, seed=0), equal_nan=True)
```

- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Implement `proxy.py` and `scripts/measure_proxy.py`** — the script loads the tactical homography with the vendored `load_homography.load(data_dir=str(data_dir()))` (returns `H, reprojection_error, valid`), computes `camera_centre`, runs `measure_proxy` against `load_competition()`, prints the dictionary, and writes it to `outputs/proxy_stats.json`.
- [ ] **Step 4: Run to verify they pass**
- [ ] **Step 5: Measure and decide** — `uv run python scripts/measure_proxy.py`

Adoption rule, fixed in advance: adopt if `corr_x >= 0.8`, `med_abs_dx <= 10.0` and `std_x >= 5.0`. If adopted, write `{"camera_proxy": true, "proxy_stats": {...}}` into `models/sources.json` (create it with `"broadcast_ball": false` if absent) and make `load_competition()` attach `ft.proxy` when the flag is set. If not adopted, write `"camera_proxy": false` and leave everything else alone. Report the numbers either way.

- [ ] **Step 6: Commit** — `git commit -m "feat: camera view centre proxy, measured"`

---

### Task 10: Broadcast ball (align timelines, measure, then adopt or drop)

**Files:**
- Create: `src/possession/sync.py`, `scripts/sync_broadcast.py`
- Test: `tests/test_sync.py`

**Idea.** The broadcast recording has its own tracking with its own ball (visible on 50.5% of its 164,488 frames) on its own timeline, with cuts and replays. Both trackings give player positions in pitch coordinates, so a broadcast frame can be matched to the tactical frame showing the same instant. Where the tactical ball is blacked out and the matched broadcast frame has a ball, copy it.

**Interfaces:**
- Produces:

```python
MATCH_MAX_COST = 2.0      # metres
MIN_PLAYERS = 6
MIN_RUN = 25              # frames

def pair_cost(b_home, b_away, t_home, t_away) -> np.ndarray
    # b_*: (Pb, 2) one broadcast frame; t_*: (N, Pt, 2) tactical frames. Returns (N,) cost in metres:
    # the median, over the tracked broadcast players, of the distance to the nearest tracked tactical
    # player of the same team; inf where either frame has fewer than MIN_PLAYERS tracked players.
def anchor_offsets(bft: FrameTable, tft: FrameTable, n_anchors: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # anchors: evenly spaced broadcast frames with at least MIN_PLAYERS tracked players. For each, the
    # tactical frame of minimum pair_cost over the whole tactical table. Returns (anchor_frames, offsets, costs)
    # with offset = tactical_frame - broadcast_frame.
def fit_mapping(bft: FrameTable, tft: FrameTable) -> np.ndarray
    # (Tb,) int32: the tactical frame for each broadcast frame, or -1.
def merge_ball(tft: FrameTable, bft: FrameTable, mapping: np.ndarray) -> tuple[FrameTable, dict]
```

- `fit_mapping`: take `anchor_offsets`; keep anchors with cost below `MATCH_MAX_COST`; cluster their offsets (two offsets within 3 frames are the same cluster); a cluster needs at least 5 anchors to be a candidate, and its offset is the cluster median. For every broadcast frame and every candidate offset `d`, compute `pair_cost` against tactical frame `i + d`; assign the candidate with the lowest cost if that cost is below `MATCH_MAX_COST`, else `−1`. Finally set to `−1` every run of consecutive frames sharing one offset that is shorter than `MIN_RUN`. Process frames in chunks so memory stays bounded.
- `merge_ball`: a new tactical table whose ball is the tactical ball where visible, else the ball of a broadcast frame mapped to that tactical frame (if several map there, the one with the lowest cost). The dictionary holds: `n_candidates`, `candidate_offsets`, `mapped_share` (of broadcast frames), `agree_median_m` and `agree_p90_m` (distance between the two balls on tactical frames where both exist), `blackout_frames_before`, `blackout_frames_after`, `filled_share` (of blackout frames inside periods).

- [ ] **Step 1: Write the failing tests** — `tactical = smooth_table(6000)`; `broadcast_from(tactical, pieces, keep=8, noise=0.2, seed=0)` concatenates the given `(start, stop)` slices of the tactical table, keeps 8 random players per team per piece, adds 0.2 m Gaussian noise, and returns the table plus the true tactical frame of each broadcast frame.

```python
def test_pair_cost_is_small_only_for_the_same_instant():
    t = smooth_table(3000)
    b, truth = broadcast_from(t, [(1000, 1001)])
    cost = pair_cost(b.home[0], b.away[0], t.home, t.away)
    assert abs(int(np.argmin(cost)) - 1000) <= 3 and cost[1000] < 0.6
    assert np.median(cost[np.isfinite(cost)]) > 2.5

def test_mapping_recovers_live_pieces_across_a_cut_and_a_replay():
    t = smooth_table(6000)
    b, truth = broadcast_from(t, [(1000, 3000), (1500, 1800), (3500, 5000)])   # live, replay, live
    mapping = fit_mapping(b, t)
    live = np.r_[0:2000, 2300:3800]
    ok = np.abs(mapping[live] - truth[live]) <= 3
    assert ok.mean() >= 0.95
    wrong = (mapping >= 0) & (np.abs(mapping - truth) > 3)
    assert wrong.mean() <= 0.01                       # a replay mapped to its true instant is not wrong

def test_merge_fills_only_blackouts_and_reports():
    t = smooth_table(6000)
    truth_ball = t.ball.copy()
    t.ball[2000:2600] = np.nan                        # a tactical blackout
    b, truth = broadcast_from(t, [(1000, 5000)])
    b.ball[:] = truth_ball[1000:5000]                 # the broadcast saw the ball throughout
    merged, stats = merge_ball(t, b, fit_mapping(b, t))
    assert np.isnan(merged.ball[2000:2600, 0]).mean() <= 0.05
    assert np.array_equal(merged.ball[:2000], t.ball[:2000])
    assert stats["filled_share"] >= 0.95 and stats["agree_median_m"] < 0.5
```

- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Implement `sync.py` and `scripts/sync_broadcast.py`** — the script loads `load_competition("tactical")` and `load_competition("broadcast")`, runs `fit_mapping` and `merge_ball`, saves the mapping to `cache/broadcast_mapping.npy`, prints the statistics dictionary and writes it to `outputs/sync_stats.json`.
- [ ] **Step 4: Run to verify they pass** — the synthetic tests must finish in under 60 s.
- [ ] **Step 5: Measure and decide** — `uv run python scripts/sync_broadcast.py`. Expect several minutes.

Adoption rule, fixed in advance: adopt if `agree_median_m <= 3.0`, `agree_p90_m <= 10.0` and `filled_share >= 0.10`. If adopted, set `"broadcast_ball": true` in `models/sources.json` and make `load_competition("tactical")` return the merged table when the flag is set (recomputing the mapping deterministically if the cache is missing). If not adopted, set it to `false`. Report all statistics either way; `n_candidates == 0` means the alignment found nothing and the source is dropped.

- [ ] **Step 6: Commit** — `git commit -m "feat: broadcast timeline alignment and ball merge, measured"`

---

### Task 11: Retrain with the adopted sources, final submission, gates

**Files:**
- Create: `scripts/compare_domains.py`
- Modify: `scripts/train.py`, `scripts/make_submission.py` as needed to honour `models/sources.json`

**Interfaces:**
- Consumes everything above.
- Produces: `outputs/<team>.csv`, `outputs/domain_report.json`, updated `models/`.

- [ ] **Step 1: Retrain if anything was adopted** — if `models/sources.json` enables either source:

```bash
uv run python scripts/measure_atd_profile.py      # blackout profile of the merged ball
uv run python scripts/train.py --profile atd      # attaches simulate_proxy(...) to DFL tables when camera_proxy is true
```

Re-check the Task 8 gate. Compare the new mean out-of-fold `spell_f1` with the one committed in Task 8. A source stays only if that mean does not go down; if it goes down, switch the source off in `models/sources.json`, retrain, and say so in the report. If nothing was adopted, skip this step.

- [ ] **Step 2: Write `scripts/compare_domains.py`** — builds features for the competition table (frames with `period > 0`) and for the seven degraded DFL matches pooled. For each feature prints: mean, standard deviation and NaN share in each domain, and the standardised mean difference `(mean_atd − mean_dfl) / pooled_std`. Sorted by absolute difference; writes `outputs/domain_report.json`. Aggregates only.
- [ ] **Step 3: Run it** — `uv run python scripts/compare_domains.py`

Rule fixed in advance: every feature with absolute standardised mean difference above 1.0, or NaN shares differing by more than 0.25, is dropped. If any feature qualifies, rerun `uv run python scripts/train.py --profile atd --drop <names>`, re-check the Task 8 gate, and commit the new models. List the dropped features in your report. Look in particular at `len_*`, `wid_*` and `n_*`: they show whether "the farthest players are the untracked ones" (Task 7) holds.

- [ ] **Step 4: Generate the submission** — ask the human for the registered team name, then

```bash
uv run python scripts/make_submission.py --team <team_name> --proba model
uv run python third_party/mtah2027/evaluation/validate_submission.py outputs/<team_name>.csv
```

Expected: `OK`, 199,319 rows. Report the label counts, possession count and mean possession duration it prints.

- [ ] **Step 5: Determinism** — run `make_submission.py` a second time and compare the two printed SHA-256 values. Expected: identical.
- [ ] **Step 6: Blackout consistency on the real match** — add `--consistency` to `make_submission.py`: apply `degrade(ft, profile, seed=0)` to the competition table (which adds blackouts where the ball is currently visible), predict, and compare with the normal prediction on the newly blacked-out frames only. Print the frame agreement there and the `spell_f1` of the degraded prediction scored against the normal prediction. Compute the same two numbers on the DFL matches (degraded prediction against clean-ball prediction) in `scripts/evaluate_external.py --consistency`, where the true score is also known. Report all of them; this is an unlabelled estimate of what the blackouts cost on the real match. No pass condition.
- [ ] **Step 7: STOP gate — correctness check against the public window**

```bash
uv run python -c "from possession.labels import read_codes; from possession.paths import PUBLIC_GT, OUTPUTS; from possession.scoring import score; print(score(read_codes(PUBLIC_GT), read_codes(OUTPUTS / '<team_name>.csv')))"
```

Pass condition, fixed in advance: `spell_f1` is at least 0.3256 (what the published floor baseline scores on the public window) and `team_acc` is at least 0.5.

This is a correctness gate, not a leaderboard, and a weak one: the ball is visible on 88% of that window against 59% of the match. If it fails, the likely causes are frame alignment, home/away orientation, or a domain gap; **STOP** and report. Never adjust a parameter and re-run to see whether the number moves.

- [ ] **Step 8: Commit** — `git commit -m "feat: final model with adopted sources, domain comparison"` (no files from `outputs/`).

---

### Task 12: README and technical document

**Files:**
- Create: `README.md`, `docs/technical_document.md`

- [ ] **Step 1: Write `README.md`** — what the repository does in three sentences; requirements (Python 3.12, uv, `libomp` on macOS); where to unpack the data (`data/`, or `MTAH_DATA_DIR`); the single command that regenerates the submission: `uv sync && uv run python scripts/make_submission.py --team <team_name> --proba model`; how to retrain from scratch (`fetch_external.py`, `measure_proxy.py`, `sync_broadcast.py`, `measure_atd_profile.py`, `train.py`); how to run the tests; licences (own code, vendored MIT code, DFL data CC-BY 4.0 with the Bassek et al. 2025 citation and the DFL credit).
- [ ] **Step 2: Write `docs/technical_document.md`** with the sections the challenge's `SUBMISSION.md` asks for, filled from `models/cv_report.json`, `models/sources.json`, `outputs/domain_report.json`, `outputs/proxy_stats.json`, `outputs/sync_stats.json` and the spec:
  - **Approach** — what was built and why this rather than alternatives; the blackout finding and how the method answers it.
  - **Assets used and dropped** — tactical tracking; Smart Tagging only for the kick-off of each half; homography and broadcast tracking as measured in Tasks 9 and 10, adopted or not, with the numbers; videos not used.
  - **Training data** — the DFL dataset: source, licence, who produced the labels (DFL official data), how much (7 matches, frame counts). State that no label from the provided match was used and that the public five minutes served only as a pass/fail check.
  - **Possession rules implemented** — how loose balls, blackouts, restarts and the pre-kick-off stretch are handled.
  - **Limitations** — the measured external scores, accuracy inside against outside blackouts, the blackout-consistency numbers, the domain comparison, known failure cases.
  - **Reproduction** — the commands.
- [ ] **Step 3: Verify the documented commands** — run each command from the README exactly as written. Expected: they work.
- [ ] **Step 4: Commit** — `git commit -m "docs: README and technical document"`

---

## After the plan

Report to the human: the reference means on DFL (baseline, rules, model), accuracy inside and outside blackouts, which extra sources were adopted and their measured numbers, the chosen decoder configuration, any dropped features, the blackout-consistency numbers, the public-window gate result, and the SHA-256 of the submission. The human creates the public repository and uploads `outputs/<team_name>.csv` and the technical document through the organisers' form.
