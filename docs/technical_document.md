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
2. **Features (223 per frame).** Four families, computed per half, never across halves:
   - *ball*: position, distance to the pitch boundary, speed, acceleration, nearest and
     second-nearest player of each team, players within 3 m (ball interpolated only over gaps of
     at most 12 frames);
   - *blackout counters*: is the ball visible, length of the current gap, frames since and until
     the ball was last and next seen, which team was last and next clearly on the ball (closer
     than 2 m and closer than any opponent) and how long ago, where the ball was (inside or
     outside the pitch) when it was last and next seen;
   - *players only*: tracked players per team, team centroids and their velocity, spread,
     length, width, defensive line, mean speeds, share of standing players;
   - *Smart Tagging*: frames since and until the nearest set-piece tag (any, and per kind:
     throw-in, corner, goal kick, free kick, kick-off) and the team it names; for the eight phase
     codes that name the team in possession (build-up, progression, creating chances, attacking
     transition, long ball, ball in final third, ball in the box, players in the box), the
     team-signed number of active tags of each code, and the team and distance of the nearest
     phase-tag start.

   Seventeen of these get nine window means each (centred ±0.5/2/10 s, past and future 1/5/20 s),
   so a frame deep inside a blackout still sees how both teams moved before and after it.
3. **Model.** LightGBM, three classes (`H`, `A`, `D`), deterministic CPU training, trained on
   seven Bundesliga matches whose tracking was corrupted to look like the provided one (section 3).
   Home/away symmetry is encouraged (not guaranteed) by training on every match in both orientations (teams and
   labels swapped).
4. **Decoder.** A minimum-duration Viterbi over `H`/`A`/`D`, run per half on integer-quantised
   log-probabilities (`round(1000·log p)`, so the result is bit-for-bit reproducible): it finds the
   labelling that maximises the summed scores minus a penalty per switch, with every possession at
   least `min_len` frames long. Tuned on the training matches: possessions and dead-ball
   stretches ≥ 50 frames (2 s), no extra switch penalty (the minimum lengths do the work).
5. **Kick-off.** Frames of a half before the Smart Tagging "Kick Off (Start)" tag of that half are
   forced to `D` (the second half's graded window opens before its kick-off).

**Why this rather than something else.**

- *Train on other matches rather than hand-tune thresholds*: the rules forbid labels from this
  match, so any threshold would have to be learned elsewhere anyway; a learned model can combine
  the 223 weak signals that matter inside blackouts.
- *LightGBM rather than a neural sequence model*: deterministic, fast on a CPU, tolerant of
  missing values, and the long windows give it the temporal context a sequence model would.
- *A segment decoder rather than frame smoothing*: it outputs possessions directly, and its two
  parameters are tuned on the metric itself (`spellF1`), on the training matches.
- *Smart Tagging as model input, learned on other matches*: the DFL matches have no Smart
  Tagging, so we built Smart-Tagging-like tags for them (section 3) and let the model learn how
  far to trust a tag, instead of writing rules on top of the tags whose effect nobody could
  measure on this match.
- A rule-based probability source (nearest player, ball outside the pitch, players standing)
  feeding the same decoder was built as a reference and fallback. The learned model beat it by a
  wide margin (section 6).

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (tracking) | the only input of the model |
| Smart Tagging | **used.** The set-piece tags (59: 26 throw-ins, 12 corners, 10 goal kicks, 5 free kicks, 6 kick-offs) and the eight phase codes that name the team in possession (433 tags), as model input; the "Kick Off (Start)" tag of each half also marks the start of live play. Not used: the four "defending" codes (the same instances seen from the other team) and the 4 "GOALS" tags. Adopted in two steps, each under a rule fixed before the run: the mean cross-validated `spellF1` must rise and at least 5 of the 7 matches must improve (section 6). |
| tactical homography | **measured, dropped.** The pitch point at the image centre as a coarse ball position: correlation with the ball's x 0.84, but median error 11.7 m in x and 16.1 m in y, lagging the ball by 37 frames. Rule fixed in advance: adopt only if correlation ≥ 0.8, median x error ≤ 10 m and x spread ≥ 5 m. Failed on the median error. |
| broadcast ATD | **measured, dropped.** We aligned the broadcast timeline to the tactical one by matching player positions (389 of 400 sampled broadcast frames matched a tactical frame with median player error 0.27 m; two constant offsets, one per half; 58.8 % of broadcast frames mapped). Where the tactical ball is missing, the broadcast ball of the matching frame would fill it. Both balls agree well (median 0.76 m, 90th percentile 5.5 m), but the broadcast ball fills only **9.1 %** of the tactical blackout frames: when the tactical camera loses the ball, the broadcast usually shows a replay or a close-up. Rule fixed in advance: adopt only if it fills ≥ 10 %. Dropped. |
| broadcast homography | not used |
| videos (tactical, broadcast) | not used |

Every extra source was judged with a rule written down before measuring, so no decision was tuned
against anything.

## 3. Training data

**Dataset.** Bassek, Rein, Weber, Memmert (2025), *An integrated dataset of spatiotemporal and
event data in elite soccer*, Scientific Data, https://doi.org/10.1038/s41597-025-04505-y; data on
figshare, https://doi.org/10.6084/m9.figshare.28196177.v1, CC-BY 4.0. Seven Bundesliga matches
(IDs J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), 25 Hz official DFL tracking.

**Labels.** Every ball frame of the DFL position data carries `BallStatus` (alive / dead) and
`BallPossession` (home / away), produced by the DFL's official tracking provider, not by us. A frame
is `D` if the ball is dead, otherwise `H` or `A` from the possession flag — the same kind of
mechanical derivation from tracking that the challenge describes for its ground truth.

The DFL flag is used as it is. A first version gave short opponent spells (under 2 s, ending
with the ball back with the previous team or with a stoppage) to the previous team, reading the
challenge's "controlled possession" as "keeps the ball for a few seconds". We dropped that rule
after checking what the labels mean on the public five minutes, which the rules allow for
exactly this purpose: the public ground truth counts possessions of 0.8 s and 1.3 s, each between
two possessions of the other team, and the rule would have merged five of its possessions into
one. The challenge states that where a reading of the definition disagrees with the public ground
truth, the ground truth is right. We decided before retraining to keep this change whatever it
did to the public-window score (it moved from 0.621 to 0.619).

**How much.** 1,002,644 frames in play (137,214–146,810 per match), 3,061 possessions (mean
duration 5.5–9.2 s per match); dead ball 41–52 % of frames. Training used every second frame of
each match in both orientations.

**Making the clean tracking look like the provided tracking.** From the provided tracking we
measured, as aggregate numbers only: how often a blackout starts per visible frame (1/816 with the
ball in the field, 1/203 within 2 m of a line or outside), the distribution of blackout lengths in
each case (median 234 and 316 frames, up to 5,537), and how many players per team are untracked
in 10-second blocks (0 missing 65 % of the time, 1 missing 22 %, more otherwise). The same
corruption is applied to each DFL match: long ball blackouts drawn from those distributions, and
whole blocks of the players farthest from the ball removed, as a camera that follows play would
lose them. A check afterwards: the published starter baseline scores 0.207 `spellF1` on the
corrupted DFL matches and 0.243 on the real match.

**Simulated Smart Tagging on the DFL matches.** What the real tags look like was measured on the
provided match as aggregates only: the count per code and set-piece kind, the share that names a
team, the tag lengths (set pieces mostly exactly 35 s, "creating chances" exactly 30 s), and
where the restart or the possession sits inside a tag. For the last, a prediction made without
tags served as the reference: play resumes 3–6 s after a set-piece tag starts (3.3 s for
kick-offs, which matches the published second-half kick-off), and inside "attacking transition"
tags the tagged team gets the ball 2–4 s after the start. The zone tags were timed on the
tracking itself: the ball enters the box exactly 10.0 s after a "ball in the box" tag starts
(median and upper quartiles), the first attacker enters the box 10.0 s after a "players in the
box" tag starts, the ball enters the final third 10.0 s after a "ball in final third" tag starts,
and the central zone 11 s after a "creating chances" tag starts. These tags are video clips that
start 10 s before their moment.

- *Set-piece tags* on the DFL matches come from their event data (throw-ins, corners, goal kicks,
  free kicks and kick-offs, with team and time), thinned to the share the real tagging contains
  (all corners and kick-offs, about 59 % of throw-ins and goal kicks, 20 % of free kicks), started
  before the restart by the measured offsets, and naming a team only where the real tags do
  (never for throw-ins and kick-offs).
- *Phase tags* on the DFL matches are moments found in the tracking and the possession labels
  (a regain, every entry of the ball into the opponent's third, box or central zone, the first
  attacker entering the box, a long ball of 30 m in 2 s, ...), merged when closer than 10 s as in
  the real tags, thinned to the real count, started before the moment by the delay measured for
  that code (10 s for the zone codes, 2–4 s for regains), and given the wrong team in the share
  by which the real tags disagree with the reference prediction more than the simulated ones
  disagree with the truth (0–4 % for most codes, 18 % for "long ball").
- The simulated tags now match the real ones in how many are active at once (1.64 against 1.62
  on average) and in how long the tagged team keeps the ball after the tag starts.

These choices were made by comparing the simulated tags with the real ones, never by the
cross-validated score. Two earlier versions of the simulation scored higher (0.607 and 0.595):
the first gave every tag its true team, the second counted a box entry only when the attacking
team kept the ball a second before and after it (41 entries per match instead of 70; the real
tagging has 72) and started every tag 2–4 s before its moment. Both would have taught the model
to read the real tags wrongly; the submitted model uses the corrected simulation.

**Nothing from the provided match was used as a label.** No frame of it was annotated by a person
for any purpose, and the five public minutes of ground truth were used once, at the end, as a
pass/fail correctness check with a threshold fixed in advance (section 6).

## 4. Possession rules implemented

- **Dead ball** is whatever the model calls dead: in training, the DFL "ball dead" status. The
  strongest dead-ball cues the model learned are the ball outside the pitch when last seen and
  players standing still.
- **Loose and contested balls** follow the DFL tracking's possession flag in training. The
  decoder will not create a possession shorter than 2 s, so a touch shorter than that stays with
  the team that had the ball; the minimum was chosen on the DFL matches.
- **Blackouts.** No label is ever left open: the model reads possession from team shape and
  movement and from who had the ball before and after the blackout; the decoder keeps a
  possession through a blackout unless the evidence for a change outweighs that for staying.
- **Restarts** split possessions: a dead-ball stretch of at least 2 s separates two possessions,
  even of the same team.
- **Smart Tagging** enters as evidence, not as a rule: from the simulated tags the model learned
  how often a tag's team really has the ball, and when play resumes around a set-piece tag.
- **Before kick-off** of each half (from the Smart Tagging kick-off tag) every frame is `D`.
- Outside the two halves the output is `X`; inside them it is never `X`.

## 5. Reproduction

```bash
uv sync
uv run python scripts/make_submission.py --team HatTrio --proba model
```

With the data bundle in `data/` (or `MTAH_DATA_DIR`) this regenerates `outputs/HatTrio.csv`
exactly (SHA-256 `19771005d42e5cc6c687d97bcca6cbd22d35a6d77379cc913b17775f97287532`; checked by
running it twice). Retraining from scratch, including the DFL download, is described in the
README; LightGBM runs with `deterministic=True`, every random draw has a fixed seed.

## 6. Results and limitations

**Cross-validation on the DFL matches** (leave one match out, corrupted tracking, official
scorer):

| match | baseline `spellF1` | rules `spellF1` | model `spellF1` | model `macroF1` |
|---|---|---|---|---|
| J03WMX | 0.199 | 0.244 | 0.509 | 0.895 |
| J03WN1 | 0.182 | 0.230 | 0.567 | 0.894 |
| J03WOH | 0.229 | 0.247 | 0.630 | 0.912 |
| J03WOY | 0.192 | 0.227 | 0.561 | 0.889 |
| J03WPY | 0.209 | 0.247 | 0.597 | 0.909 |
| J03WQQ | 0.257 | 0.233 | 0.645 | 0.916 |
| J03WR9 | 0.183 | 0.256 | 0.602 | 0.911 |
| **mean** | **0.207** | **0.241** | **0.587** | **0.904** |

What Smart Tagging added, on the same matches:

| model | mean `spellF1` | matches improved over the line above |
|---|---|---|
| tracking only | 0.572 | – |
| + set-piece tags | 0.577 | 6 of 7 |
| + phase tags (submitted) | 0.587 | 5 of 7 |

Frame accuracy of the model is 88.9 % inside simulated blackouts (87.7 % without tags) and
94.2 % where the ball is visible. The model predicts 274–321 possessions per match against
370–511 true ones: most of the missing ones are very short.

**On the provided match** (no labels, so no score): 355 possessions, mean 11.3 s; graded frames
37 % `H`, 26 % `A`, 37 % `D`. On the five public minutes (a correctness check only; the ball is
visible on 88 % of them against 59 % of the match, so it is optimistic): `spellF1` 0.632, home/away
correct on 95.5 % of live frames, 14 predicted possessions against 17 true ones. Without Smart
Tagging it was 0.619; the difference is smaller than one possession (about 0.06 there), and by
the rules the public window does not choose between versions.

**Blackout consistency without labels.** We added simulated blackouts where the ball is visible
on the real match and compared the two predictions: on the newly blacked-out frames they agree on
79.7 % of frames (77.6 % without tags), and the possessions match with `spellF1` 0.84 (0.79).
Real blackouts therefore cost something like a fifth of the frame decisions they cover.

**Domain check.** Feature distributions on the real match and on the corrupted DFL matches were
compared feature by feature; no standardised mean difference exceeds 0.7 and no missing-value
share differs by more than 0.25, so no feature was dropped. Tracked players per team match
(10.2 vs 10.2). The largest differences: the ball moves faster (6.8 vs 4.4 m/s averaged over
20 s) and Barcelona's team length is larger (55 vs 44 m). The tag features agree closely (for
example 1.62 against 1.64 phase tags active at once).

**Limitations.**

- *The DFL score is not the score on this match.* The training labels are another provider's
  possession flag, the corruption is a model of the real tracking's defects, and the DFL tags are
  our model of Metrica's tagging; "players in the box" fires 97 times on this match and only 57
  times per DFL match in the simulation. The public window is too short to say how far the real
  score is from 0.59.
- *Short possessions are missed.* The public window has two (0.8 s and 1.3 s) and we find
  neither; on the DFL matches we predict a quarter to a third fewer possessions than there are.
- *Stoppages are under-detected on the public window*: dead ball is 19.6 % of it, our output
  4.8 %. Three stoppages of 14–20 s are mostly labelled as possession there. On the whole match
  our output is 37 % dead ball, so this may be specific to those minutes; we cannot tell without
  labels.
- *A large gap between in-sample and out-of-fold scores* (0.82 vs 0.59 on DFL) says the model
  memorises match specifics; stronger regularisation or more matches would help.
- Turnovers deep inside a long blackout are guessed from team movement only.
- *Ceilings measured on the DFL matches.* With the ball never lost, the same method scores 0.69
  (0.59 with blackouts), so the blackouts cost about 0.10. With perfect knowledge of every frame,
  our 2 s minimum possession length alone caps the score at 0.87; the true labels made 0.2 s
  late score 0.93, 0.5 s late 0.78 and 1 s late 0.56. Scores near 0.9 would need almost every
  possession, including sub-second ones, with boundaries right to a few frames.
- *A decoder that allows short possessions only where the ball is seen* (down to 6 frames, with
  2 s kept inside blackouts) moved the cross-validated score from 0.587 to 0.588 and was better on
  only 3 of 7 matches, so it was not adopted (`scripts/train.py --seen-grid`). The short
  possessions are lost before the decoder: the frame model does not single them out.

**With more time** we would: find what else Metrica's "players in the box" counts; add matches
from other providers; use the broadcast ball despite its small coverage, with a threshold judged on
the training matches; and detect stoppages from the referee's whistle in the audio track.
