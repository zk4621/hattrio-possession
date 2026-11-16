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
network (`uv run --with torch==2.15.0 ...`, below); the submission runs it in NumPy from the
committed weights.

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
fdaf2ed296c5f9ab21eaab832e6f57e17034b481facdf53d2885154a164d1a9e
```

Check the format with the organisers' validator:

```bash
uv run python third_party/mtah2027/evaluation/validate_submission.py outputs/HatTrio.csv
```

## Reproduce the submitted models

The four committed model files can be retrained from this repository and two public datasets; the
match data is needed only for the last command. The longer history under "Retrain from scratch"
records how the method was chosen and is not needed for this.

```bash
uv run python scripts/fetch_external.py                     # DFL dataset, 2.6 GB, MD5-checked
git clone https://github.com/metrica-sports/sample-data external/metrica
git -C external/metrica checkout e706dd506b360d69d9d123d5b8026e7294b13996
uv run python scripts/check_metrica_relabel.py lgb <dir D>                         # (~19 min)
uv run --with torch==2.15.0 python scripts/test_sequence_model.py train <dir E>   # (~10 min)
uv run python scripts/check_metrica_relabel.py compare <dir D> --nn <dir E> \
    --old-oof <dir D> --old-nn <dir E> --out <dir H>                               # (~1 min)
git checkout models/relabel_report.json
cp <dir D>/{possession_lgb.txt,change_lgb.txt} models/
cp <dir E>/nn_s0.npz models/sequence_net.npz
cp <dir H>/decoder.json models/decoder.json
uv run python scripts/make_submission.py --team HatTrio --proba model
```

- `lgb` trains the frame and change models (LightGBM, the Metrica games weighted twice); `train`
  the sequence network; `compare` chooses the decoder settings on the three Metrica games.
- `compare` also scores a previous pipeline (`--old-oof`, `--old-nn`) for its report. The decoder
  it saves depends on the retrained pipeline alone, so the same directories can be passed twice;
  it then overwrites `models/relabel_report.json` with that self-comparison, which the
  `git checkout` restores.
- Training reads the profiles measured earlier and committed in `models/` (`atd_profile.json`,
  `tag_profile.json`, `sources.json`); how they were measured is under "Retrain from scratch".

Checked on 2026-11-16 from a fresh clone of this repository on the machine that trained the
submission (Apple M4 Pro, PyTorch 2.15.0 on the Apple GPU): all four model files came out
byte-identical to the committed ones, and so did `outputs/HatTrio.csv`. LightGBM is deterministic.
The network's GPU training is not guaranteed to repeat exactly on other hardware or PyTorch
versions, which is why its weights are committed.

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
into `external/metrica`, commit `e706dd5`), and its decoder is chosen on the Metrica games:

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
uv run --with torch==2.15.0 python scripts/test_sequence_model.py train <dir E>  # (~11 min, GPU)
uv run python scripts/test_sequence_model.py evaluate <dir E> --oof <dir D>     # (~11 min)
cp <dir E>/nn_s0.npz models/sequence_net.npz
cp <dir E>/decoder_blend.json models/decoder.json
```

Training the network on a GPU is not guaranteed to be bit-for-bit reproducible on other hardware
or PyTorch versions (on the machine that trained it, it was; see "Reproduce the submitted models");
the submission uses the committed weights (`models/sequence_net.npz`).

Three follow-ups, none adopted: five networks averaged, and a network with a change output on
every 5th frame or (second head) on every frame, whose change probabilities are averaged with the
change model's or, on every frame, also used in its place:

```bash
uv run --with torch==2.15.0 python scripts/test_sequence_model.py train <dir E> \
    --seeds 1 2 3 4                                                              # (~45 min)
uv run python scripts/test_sequence_model.py evaluate <dir E> --oof <dir D> --experiment avg \
    --seeds 0 1 2 3 4                                                            # (~1 min)
uv run --with torch==2.15.0 python scripts/test_sequence_model.py train <dir E> \
    --kind mt                                                                    # (~11 min)
uv run python scripts/test_sequence_model.py evaluate <dir E> --oof <dir D> --experiment boundary
uv run --with torch==2.15.0 python scripts/test_sequence_model.py train <dir E> \
    --kind fr --parts metrica
# (~12 min and ~13 GB of memory)
uv run python scripts/test_sequence_model.py evaluate <dir E> --oof <dir D> --experiment fullrate
```

`check_weight2.py` also loads the SkillCorner matches (for the test that rejected them), so
`external/skillcorner` must be in place.

A published ball-free model, PathCRF (not adopted), against our frame model on DFL match J03WR9,
its test match. It runs in its own environment (tested at the repository's commit `1b7c6f8`);
`<dir A>` and `<dir D>` are the directories written above:

```bash
git clone https://github.com/hyunsungkim-ds/pathcrf external/pathcrf
cd external/pathcrf && uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install torch==2.15.0 \
    torch_geometric kloppy "pandas<3" pyarrow scipy scikit-learn matplotlib tqdm rdp seaborn && cd ../..
external/pathcrf/.venv/bin/python scripts/test_pathcrf.py prep <dir F>                # (~1.5 min)
external/pathcrf/.venv/bin/python scripts/test_pathcrf.py infer <dir F> --device mps  # (~1 min)
external/pathcrf/.venv/bin/python scripts/test_pathcrf.py compare <dir F> \
    --oof <dir A>/oof.npz --oof-single <dir D>/oof_dfl_single.npz
```

Self-training on the match itself (the model's own labels as training data; not adopted), judged
on the Metrica games; `<dir D>` and `<dir E>` are the directories written above:

```bash
uv run python scripts/test_self_training.py <dir G> --oof <dir D> --nn <dir E>   # (~11 min)
```

The Metrica labels were then corrected (ball missing for at least 2 s inside a team's stretch =
dead ball, `possession/io_metrica.py`), which the loaders apply automatically, and everything
trained on them was retrained and compared with the previous submission (adopted). This produced
the committed `possession_lgb.txt`, `change_lgb.txt`, `sequence_net.npz` and `decoder.json`:

```bash
uv run python scripts/check_metrica_relabel.py lgb <dir D'>                           # (~20 min)
uv run --with torch==2.15.0 python scripts/test_sequence_model.py train <dir E'>    # (~11 min)
uv run python scripts/check_metrica_relabel.py compare <dir D'> --nn <dir E'> \
    --old-oof <dir D> --old-nn <dir E> --out <dir H>                                 # (~1 min)
cp <dir D'>/{possession_lgb.txt,change_lgb.txt} models/
cp <dir E'>/nn_s0.npz models/sequence_net.npz
cp <dir H>/decoder.json models/decoder.json
```

The earlier steps above were run with the uncorrected labels; rerun now, they use the corrected
ones.

`models/sources.json` records which extra sources were adopted (Smart Tagging yes; camera
proxy and broadcast ball no); `models/tag_profile.json` holds the measured Smart Tagging
profile; `models/change_lgb.txt` is the change model; `models/sequence_net.npz` the sequence
network; `models/cv_report.json` holds the comparison of training sets,
`models/sequence_cv_report.json` the network's test, `models/ensemble_cv_report.json` the
ensemble's, `models/sequence_avg_cv_report.json`, `models/sequence_boundary_cv_report.json` and
`models/sequence_fullrate_cv_report.json` the three follow-ups, `models/pathcrf_report.json` the
PathCRF test, `models/self_training_report.json` the self-training test,
`models/relabel_report.json` the corrected Metrica labels (all aggregates only).

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

- Own code and documentation: team HatTrio, MIT licence (`LICENSE`). The licence does not cover
  the match data, which is not in this repository and must not be added to it.
- `third_party/mtah2027/`: the challenge repository by Metrica Sports, MIT licence, vendored
  unmodified at commit `f8f2629` (see `VENDORED.md`).
- Training data, Metrica Sports sample games (https://github.com/metrica-sports/sample-data),
  used with acknowledgement as the repository asks; not redistributed here.
- Training data: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of
  spatiotemporal and event data in elite soccer.* Scientific Data.
  https://doi.org/10.1038/s41597-025-04505-y — dataset https://doi.org/10.6084/m9.figshare.28196177.v1,
  CC-BY 4.0, tracking data of the Deutsche Fußball Liga (DFL). Downloaded by
  `scripts/fetch_external.py`; not redistributed here.
- Experiment only: PathCRF code and trained model, Kim, H., Lee, K., Seo, S., Ko, S.-K., Yoon, J.,
  Park, C. (2026). *PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from
  Player Trajectories.* KDD 2026. https://github.com/hyunsungkim-ds/pathcrf, MPL-2.0; cloned into
  `external/pathcrf`, not redistributed here.
