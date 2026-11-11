# Ball possession, frame by frame — team HatTrio

Submission of team **HatTrio** to *More than a Hack 2027*, Phase 1 (Metrica Sports / FC Barcelona).
For each of the 199,319 frames of the tactical video of Barcelona vs Athletic Club the pipeline
outputs `H` (home in possession), `A` (away in possession), `D` (dead ball) or `X` (outside the
two halves). A LightGBM model trained on seven public Bundesliga matches, whose clean tracking is
corrupted to look like the provided tracking, and the three Metrica Sports sample games gives
per-frame probabilities, averaged with those of a small convolutional network over the sequence of
frames; a second LightGBM model scores where possession changes, and a minimum-duration Viterbi
decoder turns both into possessions.

The approach, the data used and the measured results are in
[docs/technical_document.md](docs/technical_document.md).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- macOS: `brew install libomp` (needed by LightGBM)

All Python dependencies are pinned in `uv.lock`. PyTorch is needed only to retrain the sequence
network (`uv run --with torch ...`, below); the submission runs it in NumPy from the committed
weights.

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
fd0544fa184435f9397eb070440b636123a4cb9d5c161eef2f8dc0a7875e37eb
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

then with both set to `true`. The submitted model is trained on the DFL matches and the three
Metrica Sports sample games, weighted twice (clone https://github.com/metrica-sports/sample-data
into `external/metrica`), and its decoder is chosen on the Metrica games:

```bash
uv run python scripts/compare_training_sets.py <dir A>          # DFL / Metrica / both (~30 min)
uv run python scripts/tune_metrica_weight.py <dir A> <dir B>     # Metrica weight 2, 3, 5 (~25 min)
uv run python scripts/check_weight2.py <dir C>                   # DFL check, SkillCorner test,
cp <dir C>/{possession_lgb.txt,change_lgb.txt} models/           # final LightGBM models (~30 min)
uv run python scripts/compare_domains.py       # feature distributions, provided vs DFL
```

Then the sequence network, averaged with LightGBM. `test_ensemble.py` writes the single model's
out-of-fold predictions that the network is compared with (and tests averaging five LightGBM
models, which was not adopted); the network itself needs PyTorch:

```bash
uv run python scripts/test_ensemble.py <dir D>                                  # (~1 h)
uv run --with torch python scripts/test_sequence_model.py train <dir E>         # (~11 min, Apple GPU)
uv run python scripts/test_sequence_model.py evaluate <dir E> --oof <dir D>     # (~11 min)
cp <dir E>/nn_s0.npz models/sequence_net.npz
cp <dir E>/decoder_blend.json models/decoder.json
```

Training the network on a GPU is not bit-for-bit reproducible; the submission uses the committed
weights (`models/sequence_net.npz`).

Three follow-ups, none adopted: five networks averaged, and a network with a change output on
every 5th frame or (second head) on every frame, whose change probabilities are averaged with the
change model's or, on every frame, also used in its place:

```bash
uv run --with torch python scripts/test_sequence_model.py train <dir E> --seeds 1 2 3 4  # (~45 min)
uv run python scripts/test_sequence_model.py evaluate <dir E> --oof <dir D> --experiment avg \
    --seeds 0 1 2 3 4                                                            # (~1 min)
uv run --with torch python scripts/test_sequence_model.py train <dir E> --kind mt  # (~11 min)
uv run python scripts/test_sequence_model.py evaluate <dir E> --oof <dir D> --experiment boundary
uv run --with torch python scripts/test_sequence_model.py train <dir E> --kind fr --parts metrica
# (~12 min and ~13 GB of memory)
uv run python scripts/test_sequence_model.py evaluate <dir E> --oof <dir D> --experiment fullrate
```

`check_weight2.py` also loads the SkillCorner matches (for the test that rejected them), so
`external/skillcorner` must be in place.

`models/sources.json` records which extra sources were adopted (Smart Tagging yes; camera
proxy and broadcast ball no); `models/tag_profile.json` holds the measured Smart Tagging
profile; `models/change_lgb.txt` is the change model; `models/sequence_net.npz` the sequence
network; `models/cv_report.json` holds the comparison of training sets,
`models/sequence_cv_report.json` the network's test, `models/ensemble_cv_report.json` the
ensemble's, `models/sequence_avg_cv_report.json`, `models/sequence_boundary_cv_report.json` and
`models/sequence_fullrate_cv_report.json` the three follow-ups (all aggregates only).

Experiment not in the submission: training also on the SkillCorner Open Data (MIT licence,
credit SkillCorner) and the Metrica Sports sample games (acknowledge Metrica Sports), both cloned
into `external/` (the SkillCorner tracking files are Git LFS objects, 1.8 GB):

```bash
uv run python scripts/train_multi.py --out <scratch dir> --compare
```

It was not adopted; see the technical document.

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
src/possession/   loaders (io_atd, io_dfl, io_metrica, io_skillcorner), features, decoder,
                  rules, degradation, model, sequence network (NumPy inference), camera proxy,
                  broadcast sync, pipeline
scripts/          the commands above
models/           committed model, decoder settings, profiles and reports (aggregates only)
third_party/mtah2027/   the challenge repository, vendored unmodified (MIT)
tests/            pytest suite
```

## Licences and credits

- Own code: team HatTrio (no licence file yet; the team chooses one before publishing).
- `third_party/mtah2027/`: the challenge repository by Metrica Sports, MIT licence, vendored
  unmodified at commit `f8f2629` (see `VENDORED.md`).
- Training data, Metrica Sports sample games (https://github.com/metrica-sports/sample-data),
  used with acknowledgement as the repository asks; not redistributed here.
- Training data: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of
  spatiotemporal and event data in elite soccer.* Scientific Data.
  https://doi.org/10.1038/s41597-025-04505-y — dataset https://doi.org/10.6084/m9.figshare.28196177.v1,
  CC-BY 4.0, tracking data of the Deutsche Fußball Liga (DFL). Downloaded by
  `scripts/fetch_external.py`; not redistributed here.
