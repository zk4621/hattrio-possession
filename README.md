# Ball possession, frame by frame — team HatTrio

Submission of team **HatTrio** to *More than a Hack 2027*, Phase 1 (Metrica Sports / FC Barcelona).
For each of the 199,319 frames of the tactical video of Barcelona vs Athletic Club the pipeline
outputs `H` (home in possession), `A` (away in possession), `D` (dead ball) or `X` (outside the
two halves). A LightGBM model trained on seven public Bundesliga matches, whose clean tracking is
corrupted to look like the provided tracking, gives per-frame probabilities; a minimum-duration
Viterbi decoder turns them into possessions.

The approach, the data used and the measured results are in
[docs/technical_document.md](docs/technical_document.md).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- macOS: `brew install libomp` (needed by LightGBM)

All Python dependencies are pinned in `uv.lock`.

## Data

The match data is not in this repository (it is not redistributable). Unpack the organisers'
bundle so that it looks like this:

```
data/
  tactical/BAR-ATH_tactical_atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml
  broadcast/BAR-ATH_broadcast_atd.xml, _atd.txt, _homography.csv
```

or set `MTAH_DATA_DIR` to wherever you unpacked it. The videos are not used.

## Regenerate the submission

```bash
uv sync && uv run python scripts/make_submission.py --team HatTrio --proba model
```

This writes `outputs/HatTrio.csv` and prints its SHA-256. It uses only the committed model
(`models/`) and the match data; it takes a few seconds. Expected SHA-256 of the submitted file:

```
2f4f40bd8dc60b14ea9b5e35b1005cb969f4267323ecfa3174519787f44b0d02
```

Check the format with the organisers' validator:

```bash
uv run python third_party/mtah2027/evaluation/validate_submission.py outputs/HatTrio.csv
```

## Retrain from scratch

```bash
uv run python scripts/fetch_external.py        # DFL dataset, 2.6 GB, into external/dfl/
uv run python scripts/describe_external.py     # possession statistics of the DFL matches
uv run python scripts/measure_proxy.py         # camera view centre: measured, adopted or not
uv run python scripts/sync_broadcast.py        # broadcast ball: measured, adopted or not
uv run python scripts/measure_atd_profile.py   # blackout profile of the provided tracking
uv run python scripts/describe_tagging.py      # what the Smart Tagging contains (aggregates)
```

The Smart Tagging profile needs a prediction made without tags, so the model is trained twice.
First with `"smart_tagging": false` and `"smart_tagging_phases": false` in `models/sources.json`:

```bash
uv run python scripts/train.py --profile atd   # CV, decoder tuning, final model (~10 min)
uv run python scripts/make_submission.py --team HatTrio --proba model
uv run python scripts/measure_tag_profile.py --pred outputs/HatTrio.csv
```

then with both set to `true`:

```bash
uv run python scripts/train.py --profile atd
uv run python scripts/compare_domains.py       # feature distributions, provided vs DFL
```

`models/sources.json` records which extra sources were adopted (Smart Tagging yes; camera
proxy and broadcast ball no); `models/tag_profile.json` holds the measured Smart Tagging
profile; `models/cv_report.json` holds the cross-validation results.

Reference evaluations on the DFL matches:

```bash
uv run python scripts/evaluate_external.py --proba model --degrade atd
uv run python scripts/evaluate_external.py --proba baseline --degrade atd
```

(The committed model was trained on all seven matches, so the first command is in-sample; the
honest estimate is the leave-one-match-out score in `models/cv_report.json`.)

## Tests

```bash
uv run pytest -q
```

The tests use synthetic data and the public files of the challenge repository only.

## Layout

```
src/possession/   loaders (io_atd, io_dfl), features, decoder, rules, degradation, model,
                  camera proxy, broadcast sync, pipeline
scripts/          the commands above
models/           committed model, decoder settings, profiles and reports (aggregates only)
third_party/mtah2027/   the challenge repository, vendored unmodified (MIT)
tests/            pytest suite
```

## Licences and credits

- Own code: team HatTrio (no licence file yet; the team chooses one before publishing).
- `third_party/mtah2027/`: the challenge repository by Metrica Sports, MIT licence, vendored
  unmodified at commit `f8f2629` (see `VENDORED.md`).
- Training data: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of
  spatiotemporal and event data in elite soccer.* Scientific Data.
  https://doi.org/10.1038/s41597-025-04505-y — dataset https://doi.org/10.6084/m9.figshare.28196177.v1,
  CC-BY 4.0, tracking data of the Deutsche Fußball Liga (DFL). Downloaded by
  `scripts/fetch_external.py`; not redistributed here.
