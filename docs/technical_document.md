# HatTrio — ball possession from tracking data

*More than a Hack 2027, Phase 1. Team HatTrio. Technical document.*

## 1. Approach

**Task.** One label per frame of the tactical video (199,319 frames): `H`, `A`, `D` or `X`.
Ranking is on `spellF1`: possessions (maximal runs of `H` or `A`) are matched one-to-one by
temporal IoU, so the output has to be a clean segmentation, not a set of independent frame
decisions.

**The finding that shaped the method.** We measured the provided tactical tracking before
designing anything (aggregates only). The ball is not missing in short flickers. It is missing in
**147 long blackouts**: median 268 frames (11 s), 90th percentile 836 frames, longest 5,537
frames (3.7 min). 99.9 % of the missing frames sit in blackouts longer than 5 s. Interpolating
the ball across them is pointless; during a blackout, possession has to be read from the
players.

**Pipeline.**

1. **Common frame table.** Every tracking source (provided ATD, DFL) is loaded into one structure:
   metres, origin at the pitch centre, players as two unordered sets (no identities), ball or NaN.
2. **Features (195 per frame).** Three families, computed per half, never across halves:
   - *ball*: position, distance to the pitch boundary, speed, acceleration, nearest and
     second-nearest player of each team, players within 3 m (ball interpolated only over gaps of
     at most 12 frames);
   - *blackout counters*: is the ball visible, length of the current gap, frames since and until
     the ball was last and next seen, which team was last and next clearly on the ball (closer
     than 2 m and closer than any opponent) and how long ago, where the ball was (inside or
     outside the pitch) when it was last and next seen;
   - *players only*: tracked players per team, team centroids and their velocity, spread,
     length, width, defensive line, mean speeds, share of standing players.

   Seventeen of these get nine window means each (centred ±0.5/2/10 s, past and future 1/5/20 s),
   so a frame deep inside a blackout still sees how both teams moved before and after it.
3. **Model.** LightGBM, three classes (`H`, `A`, `D`), deterministic CPU training, trained on
   seven Bundesliga matches whose tracking was corrupted to look like the provided one (section 3).
   Home/away symmetry is enforced by training on every match in both orientations (teams and
   labels swapped).
4. **Decoder.** A minimum-duration Viterbi over `H`/`A`/`D`, run per half on integer-quantised
   log-probabilities (`round(1000·log p)`, so the result is bit-for-bit reproducible): it finds the
   labelling that maximises the summed scores minus a penalty per switch, with every possession at
   least `min_len` frames long. Tuned on the training matches: possessions ≥ 50 frames (2 s), dead
   ball ≥ 75 frames (3 s), switch penalty 10,000 (= 10 nats).
5. **Kick-off.** Frames of a half before the Smart Tagging "Kick Off (Start)" tag of that half are
   forced to `D` (the second half's graded window opens before its kick-off).

**Why this rather than something else.**

- *Train on other matches rather than hand-tune thresholds*: the rules forbid labels from this
  match, so any threshold would have to be learned elsewhere anyway; a learned model can combine
  the 195 weak signals that matter inside blackouts.
- *LightGBM rather than a neural sequence model*: deterministic, fast on a CPU, tolerant of
  missing values, and the long windows give it the temporal context a sequence model would.
- *A segment decoder rather than frame smoothing*: it outputs possessions directly, and its two
  parameters are tuned on the metric itself (`spellF1`), on the training matches.
- A rule-based probability source (nearest player, ball outside the pitch, players standing)
  feeding the same decoder was built as a reference and fallback. The learned model beat it by a
  wide margin (section 6).

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (tracking) | the only input of the model |
| Smart Tagging | only the "Kick Off (Start)" tag of each half (start of live play); nothing else |
| tactical homography | **measured, dropped.** The pitch point at the image centre as a coarse ball position: correlation with the ball's x 0.84, but median error 11.7 m in x and 16.1 m in y, lagging the ball by 37 frames. Rule fixed in advance: adopt only if correlation ≥ 0.8, median x error ≤ 10 m and x spread ≥ 5 m. Failed on the median error. |
| broadcast ATD | **measured, dropped.** We aligned the broadcast timeline to the tactical one by matching player positions (389 of 400 sampled broadcast frames matched a tactical frame with median player error 0.27 m; two constant offsets, one per half; 58.8 % of broadcast frames mapped). Where the tactical ball is missing, the broadcast ball of the matching frame would fill it. Both balls agree well (median 0.76 m, 90th percentile 5.5 m), but the broadcast ball fills only **9.1 %** of the tactical blackout frames: when the tactical camera loses the ball, the broadcast usually shows a replay or a close-up. Rule fixed in advance: adopt only if it fills ≥ 10 %. Dropped. |
| broadcast homography | not used |
| videos (tactical, broadcast) | not used |

Both extra sources were judged with thresholds written down before measuring, so the decision was
not tuned against anything.

## 3. Training data

**Dataset.** Bassek, Rein, Weber, Memmert (2025), *An integrated dataset of spatiotemporal and
event data in elite soccer*, Scientific Data, https://doi.org/10.1038/s41597-025-04505-y; data on
figshare, https://doi.org/10.6084/m9.figshare.28196177.v1, CC-BY 4.0. Seven Bundesliga matches
(IDs J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), 25 Hz official DFL tracking.

**Labels.** Every ball frame of the DFL position data carries `BallStatus` (alive / dead) and
`BallPossession` (home / away), produced by the DFL's official tracking provider, not by us. A frame
is `D` if the ball is dead, otherwise `H` or `A` from the possession flag — the same kind of
mechanical derivation from tracking that the challenge describes for its ground truth.

One adjustment, from the challenge's definition and decided before any comparison with this
match: the DFL flag changes on *any* touch (25–37 % of its possessions last under 2 s, median
mean possession 7.2 s), while the challenge counts a change only on *controlled* possession. An
opponent spell shorter than 2 s that ends with the ball back with the previous team, or with a
stoppage, is therefore given to the previous team (a touch, a deflection, a won duel without
control). After it the median mean possession is 11.5 s. The 2 s comes from the challenge's own
proxy ("a player keeps the ball for a few seconds"); it was not tuned on any ground truth.

**How much.** 1,002,644 frames in play (137,214–146,810 per match), 2,017 possessions after the
adjustment; dead ball 41–52 % of frames. Training used every second frame of each match in both
orientations.

**Making the clean tracking look like the provided tracking.** From the provided tracking we
measured, as aggregate numbers only: how often a blackout starts per visible frame (1/816 with the
ball in the field, 1/203 within 2 m of a line or outside), the distribution of blackout lengths in
each case (median 234 and 316 frames, up to 5,537), and how many players per team are untracked
in 10-second blocks (0 missing 65 % of the time, 1 missing 22 %, more otherwise). The same
corruption is applied to each DFL match: long ball blackouts drawn from those distributions, and
whole blocks of the players farthest from the ball removed, as a camera that follows play would
lose them. A check afterwards: the published starter baseline scores 0.231 `spellF1` on the
corrupted DFL matches and 0.243 on the real match.

**Nothing from the provided match was used as a label.** No frame of it was annotated by a person
for any purpose, and the five public minutes of ground truth were used once, at the end, as a
pass/fail correctness check with a threshold fixed in advance (section 6).

## 4. Possession rules implemented

- **Dead ball** is whatever the model calls dead: in training, the DFL "ball dead" status. The
  strongest dead-ball cues the model learned are the ball outside the pitch when last seen and
  players standing still.
- **Loose and contested balls** stay with the last team that controlled it: the training labels
  give short uncontrolled opponent touches (< 2 s, ball back with the previous team or out of
  play) to the previous team, and the decoder will not create a possession shorter than 2 s.
- **Blackouts.** No label is ever left open: the model reads possession from team shape and
  movement and from who had the ball before and after the blackout; the decoder keeps a
  possession through a blackout unless the evidence for a change outweighs a fixed penalty.
- **Restarts** split possessions: a dead-ball stretch of at least 3 s separates two possessions,
  even of the same team.
- **Before kick-off** of each half (from the Smart Tagging kick-off tag) every frame is `D`.
- Outside the two halves the output is `X`; inside them it is never `X`.

## 5. Reproduction

```bash
uv sync
uv run python scripts/make_submission.py --team HatTrio --proba model
```

With the data bundle in `data/` (or `MTAH_DATA_DIR`) this regenerates `outputs/HatTrio.csv`
exactly (SHA-256 `4ac3049695a9e48e863420e5e9781d8e4eaa9d1e5a693f925dab77fb6b5a515a`; checked by
running it twice). Retraining from scratch, including the DFL download, is described in the
README; LightGBM runs with `deterministic=True`, every random draw has a fixed seed.

## 6. Results and limitations

**Cross-validation on the DFL matches** (leave one match out, corrupted tracking, official
scorer):

| match | baseline `spellF1` | rules `spellF1` | model `spellF1` | model `macroF1` |
|---|---|---|---|---|
| J03WMX | 0.236 | 0.279 | 0.667 | 0.895 |
| J03WN1 | 0.212 | 0.234 | 0.644 | 0.880 |
| J03WOH | 0.239 | 0.234 | 0.703 | 0.907 |
| J03WOY | 0.212 | 0.214 | 0.669 | 0.894 |
| J03WPY | 0.250 | 0.246 | 0.714 | 0.913 |
| J03WQQ | 0.269 | 0.221 | 0.696 | 0.911 |
| J03WR9 | 0.201 | 0.256 | 0.685 | 0.910 |
| **mean** | **0.231** | **0.241** | **0.683** | **0.901** |

Frame accuracy of the model is 88.2 % inside simulated blackouts and 94.3 % where the ball is
visible.

**On the provided match** (no labels, so no score): 277 possessions, mean 14.5 s; graded frames
37 % `H`, 26 % `A`, 37 % `D`. On the five public minutes (a correctness check only; the ball is
visible on 88 % of them against 59 % of the match, so it is optimistic): `spellF1` 0.621, home/away
correct on 94.6 % of live frames, 12 predicted possessions against 17 true ones.

**Blackout consistency without labels.** We added simulated blackouts where the ball is visible
on the real match and compared the two predictions: on the newly blacked-out frames they agree on
79.5 % of frames, and the possessions match with `spellF1` 0.82. Real blackouts therefore cost
something like a fifth of the frame decisions they cover.

**Domain check.** Feature distributions on the real match and on the corrupted DFL matches were
compared feature by feature; no standardised mean difference exceeds 0.7 and no missing-value
share differs by more than 0.25, so no feature was dropped. Tracked players per team match
(10.2 vs 10.2). The largest differences: the ball moves faster (6.8 vs 4.4 m/s averaged over
20 s) and Barcelona's team length is larger (55 vs 44 m).

**Limitations.**

- *The DFL score is not the score on this match.* The training labels are another provider's
  possession flag, adjusted by a rule, and the corruption is a model of the real tracking's
  defects. We expect the real score to be clearly lower than 0.68; the public window is too short
  to say how much.
- *The model under-segments* on the public window (12 possessions against 17): the 2 s minimum and
  the switch penalty merge short possessions. Short real possessions are the main loss.
- *A large gap between in-sample and out-of-fold scores* (0.98 vs 0.68 on DFL) says the model
  memorises match specifics; stronger regularisation or more matches would help.
- Turnovers deep inside a long blackout are guessed from team movement only.

**With more time** we would: label short contested phases better (event data of the DFL dataset
marks passes and tackles, which would give a sharper "controlled possession" rule); add matches
from other providers; use the broadcast ball despite its small coverage, with a threshold judged on
the training matches; and detect stoppages from the referee's whistle in the audio track.
