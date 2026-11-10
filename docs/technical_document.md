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
   seven Bundesliga matches, whose clean tracking was corrupted to look like the provided one, and
   the three Metrica Sports sample games, which come from the provided match's tracking provider
   and lose the ball as often (section 3).
   Home/away symmetry is encouraged (not guaranteed) by training on every match in both orientations (teams and
   labels swapped).
4. **Change model.** A second LightGBM model scores every frame for "the label changes within
   2 frames of here". Besides the frame model's features it reads features that react on the very
   frame something happens: the ball turning or speeding up over 3 frames, the nearest player
   switching team, distances to the ball jumping, the ball crossing a line, the ball appearing or
   vanishing. Its strongest inputs are the ball's acceleration, the players' mean speed jumping and
   the ball turning.
5. **Decoder.** A minimum-duration Viterbi over `H`/`A`/`D`, run per half on integer-quantised
   log-probabilities (`round(1000·log p)`, so the result is bit-for-bit reproducible): it finds the
   labelling that maximises the summed scores minus a cost per switch, with every run at least
   `min_len` frames long. The cost of a switch on frame t is 2,000 + 500 × (−logit of the change
   model on t), at least 0: cheap where the change model expects a change, dear elsewhere. Tuned
   on the Metrica games, the training matches closest to the provided one: possessions ≥ 25 frames
   (1 s), dead-ball stretches ≥ 75 frames (3 s), change-model weight 0.5.
6. **Kick-off.** Frames of a half before the Smart Tagging "Kick Off (Start)" tag of that half are
   forced to `D` (the second half's graded window opens before its kick-off).

**Why this rather than something else.**

- *Train on other matches rather than hand-tune thresholds*: the rules forbid labels from this
  match, so any threshold would have to be learned elsewhere anyway; a learned model can combine
  the 223 weak signals that matter inside blackouts.
- *LightGBM rather than a neural sequence model*: deterministic, fast on a CPU, tolerant of
  missing values, and the long windows give it the temporal context a sequence model would.
- *A segment decoder rather than frame smoothing*: it outputs possessions directly, and its
  parameters are tuned on the metric itself (`spellF1`), on the training matches.
- *A separate model for where a change happens*: the metric rewards boundaries within a few
  frames, and the frame model, reading windows of up to 20 s, smears them; the change model reads
  what happens on the frame itself.
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
| tactical video | **measured, dropped.** A ball detected in the video could fill the tracking's blackouts. We ran a public football ball detector (`football-ball-detection-v2.pt` from Roboflow's `sports` examples, YOLOv8 trained on broadcast football footage; Ultralytics, one pass per frame at native resolution, 0.39 s per frame on an Apple M4 Pro GPU) on 3,000 sampled frames of the two halves and mapped each frame's highest-scoring box to the pitch with the homography. The ball is small in this video (about 6 pixels across, measured from the tracking and the homography). On 1,000 frames where the tracking sees the ball, the detector reports a confident ball (score ≥ 0.25) on 985: in the typical case it is close (median distance to the tracking ball 1.2 m), but only 60 % are within 2 m and the 90th percentile is **31.6 m**. On 22 % of these frames the top box is more than 8 m away; half of those boxes lie outside the touchlines (where the tracked ball is on only 4 % of the frames) and they score almost as high as the right ones (median 0.80 against 0.85), so no score threshold separates them. On 2,000 blackout frames a confident box exists on 93.5 %, but half of them lie outside the lines, so that coverage is mostly false detections. Rule fixed in advance (committed before the run): median ≤ 2 m, 90th percentile ≤ 8 m and confident detections on ≥ 20 % of blackout frames. Failed on the 90th percentile. Dropped; the probe code is not part of the submitted repository. |
| broadcast video | not used |

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

**Other open matches.** We also trained on 23 more public matches:
SkillCorner Open Data (20 A-League 2024/25 matches, broadcast tracking at 10 fps, MIT licence,
https://github.com/SkillCorner/opendata; possession as SkillCorner names it on each frame, dead
ball outside SkillCorner's phases of play) and the three Metrica Sports sample games
(https://github.com/metrica-sports/sample-data; the provider of the match's tracking; labels
derived by us from Metrica's events: a team has the ball from its pass, carry, recovery, shot,
lost ball or set piece on, the ball is dead from a ball-out, a foul or a goal until the next set
piece). Both already lose the ball on about 40 % of frames, so they were used without simulated
blackouts; their possessions change 15-19 times per 5 minutes, against 26 in the DFL labels. The
rule fixed before the run asked for three things; the first two held and the third did not:

| condition | result |
|---|---|
| held-out SkillCorner and Metrica matches better than the DFL-only model | 0.378 -> 0.547 |
| Metrica games better on at least 2 of 3 | 0.656 -> 0.688, 3 of 3 |
| DFL out-of-fold mean at least 0.5885 | 0.594 -> 0.537 |

The extra matches teach the model longer possessions, which the DFL labels do not have
(`scripts/train_multi.py` reruns the experiment; the same script on the DFL matches alone gives
0.5935, so the drop is not a code effect).

**Which labels to judge by.** That rule judged by the DFL matches, but the DFL labels are not the
challenge's: DFL's possession flag changes on every touch (26 possessions per 5 minutes), the
challenge's public five minutes have 17, the Metrica games 18. The Metrica games are also the only
training matches from the provided match's tracking provider, with its blackouts (32-41 % of
frames without the ball, against 41 % here). We therefore compared three training sets on the
Metrica games, each game predicted by a model that never saw it and the decoder chosen on the
other two games, with a rule fixed before the run: the best mean on the Metrica games wins, and
replaces the DFL-only model only if it beats it on at least 2 of the 3 games.

| training set | Metrica 1 | Metrica 2 | Metrica 3 | Metrica mean | DFL mean |
|---|---|---|---|---|---|
| DFL only | 0.646 | 0.633 | 0.674 | 0.651 | 0.592 |
| Metrica only | 0.693 | 0.703 | 0.705 | 0.700 | 0.362 |
| DFL + Metrica (submitted) | 0.712 | 0.710 | 0.727 | **0.716** | 0.588 |

DFL + Metrica wins on all three games and keeps the DFL matches where they were (0.588 against
0.592). SkillCorner is not used: its broadcast tracking sees about 5 players per team, against 10
here. `scripts/compare_training_sets.py` runs the comparison and `scripts/train_final.py` trains
the submitted model from it.

**Nothing from the provided match was used as a label.** No frame of it was annotated by a person
for any purpose, and the five public minutes of ground truth were used once, at the end, as a
pass/fail correctness check with a threshold fixed in advance (section 6).

## 4. Possession rules implemented

- **Dead ball** is whatever the model calls dead: in training, the DFL "ball dead" status and, on
  the Metrica games, the stoppages in Metrica's events (from a ball-out, foul or goal to the next
  set piece). The
  strongest dead-ball cues the model learned are the ball outside the pitch when last seen and
  players standing still.
- **Loose and contested balls** follow the DFL possession flag and, on the Metrica games,
  Metrica's events: a recovery changes possession, a challenge does not. The decoder will not
  create a possession shorter than 1 s, and creates a short one only where the change model sees
  a change; the minimum was chosen on the Metrica games.
- **Blackouts.** No label is ever left open: the model reads possession from team shape and
  movement and from who had the ball before and after the blackout; the decoder keeps a
  possession through a blackout unless the evidence for a change outweighs that for staying.
- **Restarts** split possessions: a dead-ball stretch of at least 3 s separates two possessions,
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
exactly (SHA-256 `2a3f396b219da3be134bcc6819f68996d351219b577315b8a813ffdd96914f87`; checked by
running it twice). Retraining from scratch, including the DFL download, is described in the
README; LightGBM runs with `deterministic=True`, every random draw has a fixed seed.

## 6. Results and limitations

**The submitted model** (DFL + Metrica) scores 0.712, 0.710 and 0.727 on the three Metrica games
it never saw (section 3). **How the model was built up**, on the DFL matches with DFL-only
training (leave one match out, corrupted tracking, official scorer):

| match | baseline `spellF1` | rules `spellF1` | model `spellF1` | model `macroF1` |
|---|---|---|---|---|
| J03WMX | 0.199 | 0.244 | 0.520 | 0.896 |
| J03WN1 | 0.182 | 0.230 | 0.572 | 0.896 |
| J03WOH | 0.229 | 0.247 | 0.619 | 0.914 |
| J03WOY | 0.192 | 0.227 | 0.578 | 0.891 |
| J03WPY | 0.209 | 0.247 | 0.612 | 0.913 |
| J03WQQ | 0.257 | 0.233 | 0.637 | 0.916 |
| J03WR9 | 0.183 | 0.256 | 0.618 | 0.914 |
| **mean** | **0.207** | **0.241** | **0.594** | **0.906** |

What each step added, on the same matches (each adopted only if the mean rose and at least 5 of
the 7 matches improved):

| model | mean `spellF1` | matches improved over the line above |
|---|---|---|
| tracking only | 0.572 | – |
| + set-piece tags | 0.577 | 6 of 7 |
| + phase tags | 0.587 | 5 of 7 |
| + change model | 0.594 | 5 of 7 |
| + Metrica games in training (submitted) | 0.588 | – (judged on the Metrica games) |

Frame accuracy of the DFL-only model is 88.9 % inside simulated blackouts (87.7 % without tags)
and 94.4 % where the ball is visible. The model predicts 279–339 possessions per match against
370–511 true ones: most of the missing ones are very short.

**On the provided match** (no labels, so no score): 356 possessions, mean 11.2 s; graded frames
37 % `H`, 26 % `A`, 37 % `D`. On the five public minutes (a correctness check only; the ball is
visible on 88 % of them against 59 % of the match, so it is optimistic): `spellF1` 0.571, home/away
correct on 93.1 % of live frames, 18 predicted possessions against 17 true ones. The DFL-only model
scored 0.645 there and tracking only 0.619; one possession moves the score by about 0.06 in those
five minutes, against about 1,000 possessions in the three Metrica games, and by the rules the
public window does not choose between versions.

**Blackout consistency without labels.** We added simulated blackouts where the ball is visible
on the real match and compared the two predictions: on the newly blacked-out frames they agree on
78.5 % of frames (77.6 % with tracking only), and the possessions match with `spellF1` 0.81
(0.79).
Real blackouts therefore cost something like a fifth of the frame decisions they cover.

**Domain check.** Feature distributions on the real match and on the corrupted DFL matches were
compared feature by feature; no standardised mean difference exceeds 0.7 and no missing-value
share differs by more than 0.25, so no feature was dropped. Tracked players per team match
(10.2 vs 10.2). The largest differences: the ball moves faster (6.8 vs 4.4 m/s averaged over
20 s) and Barcelona's team length is larger (55 vs 44 m). The tag features agree closely (for
example 1.62 against 1.64 phase tags active at once).

**Limitations.**

- *The closest estimate of the score on this match* is the submitted model's 0.716 on the three
  Metrica sample games, each predicted by a model that never saw it: same tracking provider, the
  ball missing on 32-41 % of frames, possessions changing about as often as in the public five
  minutes. Their labels were derived by us from Metrica's events, not from corrected tracking, so
  this is an estimate.
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
- *A large gap between in-sample and out-of-fold scores* (0.89 vs 0.59 on DFL, DFL-only model) says the model
  memorises match specifics; stronger regularisation or more matches would help.
- Turnovers deep inside a long blackout are guessed from team movement only.
- *Ceilings measured on the DFL matches* (before the change model). With the ball never lost,
  the method scored 0.69 (0.59 with blackouts), so the blackouts cost about 0.10. With perfect
  knowledge of every frame, a 2 s minimum possession length alone caps the score at 0.87 and a
  0.5 s minimum at 0.97; the true labels made 0.2 s late score 0.93, 0.5 s late 0.78 and 1 s late
  0.56. Scores near 0.9 would need almost every
  possession, including sub-second ones, with boundaries right to a few frames.
- *A decoder that allows short possessions only where the ball is seen* (down to 6 frames, with
  2 s kept inside blackouts) moved the cross-validated score from 0.587 to 0.588 and was better on
  only 3 of 7 matches, so it was not adopted (`scripts/train.py --seen-grid`). The frame model does
  not single short possessions out; the change model, which does look at single frames, did help.
- *Larger change-model weights* (up to 3, with possessions down to 6 frames) moved the
  cross-validated score from 0.5935 to 0.5949 and were better on only 3 of 7 matches, so they were
  not adopted (`scripts/train.py --change --wide-change-grid`). The best settings all lie within
  0.0004 of each other: the decoder is on a plateau.

**With more time** we would: find what else Metrica's "players in the box" counts; add matches
from other providers; use the broadcast ball despite its small coverage, with a threshold judged on
the training matches; and detect stoppages from the referee's whistle in the audio track. The
video ball detector puts the true ball first on 59 % of the frames where the tracking sees it and
among its top five boxes on 66 %; a tracker that picks, among the boxes, the one consistent with
where the ball was last seen might keep the right one. That would need a new rule fixed in
advance and a detection run over all 64,643 blackout frames (about 7 hours on an Apple GPU).
