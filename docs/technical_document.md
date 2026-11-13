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
   the three Metrica Sports sample games, weighted twice, which come from the provided match's
   tracking provider and lose the ball as often (section 3). Their labels come from Metrica's
   events, corrected where the events miss a stoppage (section 3).
   Home/away symmetry is encouraged (not guaranteed) by training on every match in both orientations (teams and
   labels swapped).
4. **Sequence network.** A small convolutional network over time reads the same 223 features on
   every 5th frame and sees ±25 s of play around each step (seven dilated convolutions, 64
   channels, 141,507 weights). Its `H`/`A`/`D` probabilities, averaged over the match as it is and
   with home and away exchanged, are interpolated back to 25 frames per second and averaged with
   LightGBM's, half and half. Trained with PyTorch on the same matches; inference is plain NumPy,
   so the submission does not need PyTorch (section 3).
5. **Change model.** A second LightGBM model scores every frame for "the label changes within
   2 frames of here". Besides the frame model's features it reads features that react on the very
   frame something happens: the ball turning or speeding up over 3 frames, the nearest player
   switching team, distances to the ball jumping, the ball crossing a line, the ball appearing or
   vanishing. Its strongest inputs are the ball's acceleration, the players' mean speed jumping and
   the ball turning.
6. **Decoder.** A minimum-duration Viterbi over `H`/`A`/`D`, run per half on integer-quantised
   log-probabilities (`round(1000·log p)`, so the result is bit-for-bit reproducible): it finds the
   labelling that maximises the summed scores minus a cost per switch, with every run at least
   `min_len` frames long. The cost of a switch on frame t is 1,000 × (−logit of the change model
   on t), at least 0: free where the change model expects a change more likely than not, dear
   elsewhere. Tuned on the Metrica games, the training matches closest to the provided one:
   possessions ≥ 12 frames (0.5 s), dead-ball stretches ≥ 25 frames (1 s), change-model weight 1,
   a fixed cost of 2,000 per switch (chosen again after the Metrica labels were corrected).
7. **Kick-off.** Frames of a half before the Smart Tagging "Kick Off (Start)" tag of that half are
   forced to `D` (the second half's graded window opens before its kick-off).

**Why this rather than something else.**

- *Train on other matches rather than hand-tune thresholds*: the rules forbid labels from this
  match, so any threshold would have to be learned elsewhere anyway; a learned model can combine
  the 223 weak signals that matter inside blackouts.
- *LightGBM first, a sequence network next to it*: LightGBM is deterministic, fast on a CPU and
  tolerant of missing values, and its long windows give it temporal context. A network over the
  sequence of frames was weaker on its own (0.712 against 0.724 on the Metrica games) but its
  average with LightGBM was better on all three games (0.730), so it complements LightGBM rather
  than replacing it (section 3).
- *A segment decoder rather than frame smoothing*: it outputs possessions directly, and its
  parameters are tuned on the metric itself (`spellF1`), on the training matches.
- *A separate model for where a change happens*: the metric rewards boundaries within a few
  frames, and the frame model, reading windows of up to 20 s, smears them; the change model reads
  what happens on the frame itself. A change output on the sequence network did not help: on
  every 5th frame it marked the change frames far less sharply; on every frame it marked them as
  sharply as the change model, yet did not raise the score (section 3).
- *Our own frame model rather than a published ball-free one*: PathCRF (Kim et al. 2026) reads
  possession from the players alone, decodes it with legal hand-overs only, and reports the right
  team on 92 % of in-play frames of one of our DFL matches. Measured on that match against the
  DFL labels, it names the right team less often than our frame model (88.7 % against 91.0 %;
  88.1 % against 89.3 % where our simulated tracking loses the ball), and it needs every player
  tracked with an identity and has no dead-ball state (section 3).
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
piece; corrected later, see *Corrected Metrica labels* below). Both already lose the ball on about 40 % of frames, so they were used without simulated
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
| DFL + Metrica (adopted) | 0.712 | 0.710 | 0.727 | **0.716** | 0.588 |

DFL + Metrica wins on all three games and keeps the DFL matches where they were (0.588 against
0.592).

**How much weight for the Metrica games.** They are 3 of the 10 training matches. We then gave
their rows more weight, and tried stronger regularisation (31 leaves, 400 rows per leaf, L2 5),
judged the same way, with a stricter rule because several variants were tried: the best mean
wins only if better on all three games and higher by at least 0.005, and the DFL mean may not
fall below 0.572. "Strict" is the score at 70 % overlap alone, the part that depends most on
accurate possession boundaries.

| variant | Metrica 1 | Metrica 2 | Metrica 3 | mean | strict |
|---|---|---|---|---|---|
| weight 1 | 0.712 | 0.710 | 0.727 | 0.716 | 0.628 |
| weight 2 (adopted; the submission averages it with the network below) | 0.723 | 0.715 | 0.733 | **0.724** | **0.642** |
| weight 3 | 0.720 | 0.711 | 0.735 | 0.722 | 0.640 |
| weight 5 | 0.728 | 0.705 | 0.732 | 0.722 | 0.638 |
| regularised, weight 1 | 0.717 | 0.709 | 0.727 | 0.718 | 0.631 |
| regularised, weight 3 | 0.716 | 0.705 | 0.739 | 0.720 | 0.636 |
| weight 2 + 20 SkillCorner matches | 0.705 | 0.682 | 0.693 | 0.693 | 0.598 |

Weight 2 passes (DFL mean 0.590). Adding SkillCorner, judged the same way, is worse on all three
games: its broadcast tracking sees about 5 players per team, against 10 here. The StatsBomb open
data (github.com/hudl/open-data) has events and 360 snapshots but no continuous tracking, so the
model cannot use it. `scripts/compare_training_sets.py`, `scripts/tune_metrica_weight.py` and
`scripts/check_weight2.py` run these comparisons; the last one also trains the submitted LightGBM
models.

**Several models averaged (an ensemble; not adopted).** The model scores 0.89 on DFL matches it
has seen and 0.59 on matches it has not, so we averaged five models whose training differs, in
three ways, each judged like the weights above (better on all three Metrica games, mean higher by
at least 0.005, DFL mean at least 0.572), with the rule fixed before the run:

| five models that differ in | Metrica 1 | Metrica 2 | Metrica 3 | mean | strict |
|---|---|---|---|---|---|
| nothing (one model, weight 2) | 0.723 | 0.715 | 0.733 | 0.724 | 0.642 |
| LightGBM's random seed | 0.727 | 0.717 | 0.732 | 0.725 | 0.647 |
| the seed and freshly simulated blackouts and tags | 0.728 | 0.701 | 0.735 | 0.722 | 0.635 |
| the seed, each without one DFL match | 0.722 | 0.709 | 0.733 | 0.722 | 0.636 |

The best (seeds) gains 0.002 and is better on 2 of 3 games: not adopted
(`scripts/test_ensemble.py`).

**A sequence network (adopted).** LightGBM sees the past and future of a frame only through
window means of up to 20 s. A network over the sequence (section 1, step 4) learns its own
context. Every match was predicted by a network that never saw it: for a Metrica game, trained on
the other two (weighted 2) and six DFL matches, with the seventh DFL match used only to stop
training where its loss was lowest (checked every 250 steps; the stops fell between 500 and 2,500
of at most 3,000 steps of 16 windows of 1,024 steps; AdamW, learning rate 0.001, dropout 0.2).
All settings were fixed before the run, nothing was tuned on the Metrica games, and the rule was
the same as for the ensemble:

| candidate | Metrica 1 | Metrica 2 | Metrica 3 | mean | strict | DFL mean |
|---|---|---|---|---|---|---|
| LightGBM alone (weight 2) | 0.723 | 0.715 | 0.733 | 0.724 | 0.642 | 0.590 |
| network alone | 0.712 | 0.711 | 0.714 | 0.712 | 0.623 | – |
| average of both (then submitted) | 0.736 | 0.717 | 0.737 | **0.730** | **0.648** | 0.592 |

The average is better on all three games (+0.006) and keeps the DFL matches (0.592), so it was
adopted; the decoder was then chosen again on the Metrica games (no fixed switch cost). The final
network is trained on nine matches (DFL match J03WR9 decides when to stop). The NumPy inference
agrees with PyTorch to within 6·10⁻⁷ in probability. `scripts/test_sequence_model.py` trains
(with PyTorch, about 11 minutes on an Apple M4 Pro GPU), evaluates and exports it.

**Five networks, and a change output on the network (neither adopted).** Two follow-ups, each
under the same rule, fixed before its run. First, five networks with different random seeds
(seed 0 is the submitted one; the others stopped between 500 and 3,000 steps), averaged with each
other and then with LightGBM. Second, a network with a fourth output, "the label changes within
2 frames of this step" (the change detector's target, on every 5th frame), trained on the sum of
both losses; its change probabilities were averaged half and half with the change detector's,
alone or together with its frame probabilities in place of the submitted network's. Because the
five networks were not adopted, this second test used one network, as fixed before the run.

| candidate | Metrica 1 | Metrica 2 | Metrica 3 | mean | strict |
|---|---|---|---|---|---|
| then submitted: LightGBM averaged with one network | 0.736 | 0.717 | 0.737 | **0.730** | **0.648** |
| five networks instead of one | 0.736 | 0.714 | 0.748 | 0.732 | 0.651 |
| change probabilities averaged with the new network's | 0.727 | 0.713 | 0.730 | 0.724 | 0.636 |
| the same, and frame probabilities from the new network | 0.716 | 0.714 | 0.734 | 0.722 | 0.637 |

Five networks gain 0.0025 and are better on one game only (game 3; game 1 is 0.00002 lower,
game 2 0.003 lower).
The change output is worse on all three games. It marks the change frames much less sharply than
the detector: their average precision (1 = every change frame ranked above every other frame) is
0.431–0.442 against 0.665–0.735, and the mean probability on change frames 0.23–0.28 against
0.49–0.53. The network sees every 5th frame, so it knows that a change is near but not on which
frame; averaged in, it blurs the detector's boundaries (average precision 0.635–0.698). Run with
`scripts/test_sequence_model.py --experiment avg` and `--experiment boundary`; the reports are
`models/sequence_avg_cv_report.json` and `models/sequence_boundary_cv_report.json`.

**A change output on every frame (not adopted).** A third follow-up, under the same rule fixed
before its run, tested whether the second failed only because it was too coarse. The network got
a second head that works on every frame: the network's state, interpolated between steps, plus
the change detector's 87 inputs that are not window averages (what the frame shows on its own,
and the 17 change features), through five dilated convolutions that see ±31 frames (1.2 s). Its
target is the detector's own; frames labelled `X` are left out, as for the detector. One network
(seed 0, stopped after 750 to 1,500 steps); its change probabilities were averaged half and half
with the detector's, or used in its place.

| candidate | Metrica 1 | Metrica 2 | Metrica 3 | mean | strict |
|---|---|---|---|---|---|
| then submitted | 0.736 | 0.717 | 0.737 | **0.730** | 0.648 |
| change probabilities averaged with the new network's | 0.730 | 0.712 | 0.735 | 0.726 | 0.646 |
| the new network's change probabilities alone | 0.733 | 0.714 | 0.744 | 0.731 | 0.652 |

On every frame the network marks the change frames as sharply as the detector (average precision
0.661–0.740 against 0.665–0.735; mean probability on change frames 0.52–0.65 against 0.49–0.53),
and the average of the two more sharply than either (0.687–0.762). The score does not follow: the
average is worse on all three games, and the network alone gains 0.0005 and is better on game 3
only. Not adopted, so the DFL networks were not trained (`--experiment fullrate`;
`models/sequence_fullrate_cv_report.json`). Together with the larger change-model weights that did
not help (section 6), this suggests that how sharply the change frames are marked no longer limits
the score.

**A published ball-free model, PathCRF (not adopted).** Kim, Lee, Seo, Ko, Yoon and Park (2026),
*PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player
Trajectories*, KDD 2026 (https://arxiv.org/abs/2602.12080; code and trained models
https://github.com/hyunsungkim-ds/pathcrf, MPL-2.0), read from the players' tracks alone, without
the ball, which player has the ball or which pass is under way, every 5th frame, and decode the
sequence with a conditional random field that allows only legal hand-overs (a pass ends with the
player it was played to). The model is trained on the same DFL dataset (five matches, J03WQQ for
validation) and tested on J03WR9, where the paper reports the right team on 92.4 % of in-play
frames. That would beat our frame model where it is weakest, inside blackouts, so we ran the
published model on J03WR9 with a rule fixed before the run: continue only if it names the right
team on at least 92 of 100 in-play frames where our simulated tracking of that match has no ball
(42 % of them), judged against the DFL flag. Our frame model, trained on the other six DFL matches
and never on J03WR9, does 89.3 there. Our numbers are the frame model's more likely team (`H` or
`A`), before the decoder.

Run as its repository's tutorial runs it, the published model reproduces the paper: the right
possession state (one of 676: a player on the ball, a pass between two players, or the ball going
out over one of the four lines) on 69.5 % of steps (paper 69.64 %), events found with F1 75.6 %
(75.69 %), the right team on 91.6 % of frames (92.40 %; our run of its preprocessing finds 1,783
events where the paper counts 1,831). Which team, on the 85,316 in-play frames of J03WR9, in
percent of frames:

| model | DFL flag: all | DFL flag: no ball | PathCRF's labels: all | PathCRF's labels: no ball |
|---|---|---|---|---|
| PathCRF (published) | 88.7 | **88.1** | 92.7 | 91.7 |
| our frame model, other 6 DFL matches | 91.0 | 89.3 | 89.5 | 87.8 |
| submitted LightGBM (+ Metrica, weight 2) | 90.7 | 88.7 | 89.4 | 86.8 |

Not passed: 88.1, and 88.9 if the 1.2 % of frames on which PathCRF names no team (it reckons the
ball is going out) are left out. Each model is best on the labels it was trained on. PathCRF's
labels (the team of the ball touches, split half way between two touches by different teams)
agree with the DFL flag on 90 % of in-play frames, so its 92 % measures how well it reproduces its
own definition of possession rather than a better reading of the players. It also had two
advantages ours did not: all 22 players tracked without gaps and with identities (the provided
tracking has 46 anonymous track slots and 20.5 players on average, and the published code skips
any stretch in which a player is missing), and the true in-play periods given, since it has no
dead-ball state. Not adopted. `scripts/test_pathcrf.py` reruns it; the report is
`models/pathcrf_report.json`.

**Self-training on the match itself (not adopted).** The frame model scores 0.89 on DFL matches
it was trained on and 0.59 on the others, so knowing the match is worth a lot, and the rules allow
labels produced by an algorithm on the provided match, provided no person decides any of them. We
tested whether the model's own labels can teach it the match. The test was run on the Metrica
games, where the true labels are known. For each game, the submitted pipeline's out-of-fold labels
(decoder chosen on the other two games) became training targets on the frames where the ball is
visible. On the in-play frames among them those labels are right 93.6–94.4 % of the time, against
67–80 % where the ball is missing. The rows themselves came from a copy of the game with extra
ball blackouts drawn from the measured profile (28–36 % of those frames hidden), so they show this
match with the ball lost. We tried two ways to use them, both on LightGBM only (the network, the
change model and the decoder grid stayed as they were):

- *pool*: LightGBM retrained from scratch with these rows added at weight 2;
- *tune*: the out-of-fold model with 100 more rounds on these rows alone.

The rule was fixed before the run (2026-11-13), as for the network: the candidate must beat the
submission on all three games and raise the mean by at least 0.005, then pass the DFL check. The
retrained out-of-fold models reproduce the saved predictions exactly. `spellF1` on each game,
with the decoder chosen on the other two:

| model | game 1 | game 2 | game 3 | mean |
|---|---|---|---|---|
| then submitted | 0.736 | 0.717 | 0.737 | 0.730 |
| pool | 0.737 | 0.714 | 0.728 | 0.726 |
| tune | 0.693 | 0.684 | 0.701 | 0.693 |

The rule was not met: *pool* is better on one game, by 0.001, and *tune* is worse on all three.
Our reading, not measured: the labels the model learns from are the decoder's, already merged
into possessions of at least 1 s. The frame model therefore learns its own mistakes and blurrier
hand-overs, and `spellF1` depends heavily on hand-overs being on time (section 6). *Tune*, which
learns from these labels alone, loses the most. Not adopted, so the DFL check was not run.
`scripts/test_self_training.py` reruns the test (about 11 minutes); the report is
`models/self_training_report.json`.

**Corrected Metrica labels (adopted).** Checking where the ball goes missing showed that the
Metrica events miss many stoppages. On every game, 8–14 minutes were labelled as one team's
possession while the tracking had no ball and the players moved as slowly as in labelled
stoppages (1.1–1.2 m/s on average, against 1.9–2.1 m/s in labelled play and 1.1–1.2 m/s in
labelled stoppages). The events just before those stretches are mostly a shot wide (31 % of those
frames), a foul recorded as a challenge (about 18 %), a ball lost under pressure (12 %) or an
offside (4 %): restarts the event-to-label rule did not know. The labels now mark the ball dead
wherever the tracking has no ball for at least 2 s inside a stretch labelled with a team
(`possession/io_metrica.py`, `dead_in_gaps`). This changes 8–15 % of each game's frames; dead ball
rises from 25–34 % to 33–44 % of in-play frames, close to the DFL matches (45 %).

Everything trained on the Metrica games was retrained on the corrected labels, with the same
settings: the LightGBM frame and change models (Metrica weight 2), the sequence network, and the
decoder, chosen again on the Metrica games. Rule fixed before the run (2026-11-13): adopt if the
mean on the 7 DFL matches (labels unchanged) is at least the submission's minus 0.005 and the
mean on the 3 Metrica games, scored against the corrected labels, is at least the submission's.
Both pipelines are scored the same way, each match predicted by models that never saw it, decoder
chosen on the other matches of its test set:

| pipeline | Metrica 1 | Metrica 2 | Metrica 3 | Metrica mean | DFL mean |
|---|---|---|---|---|---|
| previous submission | 0.731 | 0.742 | 0.726 | 0.733 | 0.592 |
| retrained on corrected labels (submitted) | 0.770 | 0.793 | 0.768 | **0.777** | **0.597** |

Both conditions held: the Metrica games rose on all three, and the DFL matches, whose labels did
not change, rose too (5 of 7 better). Adopted. The comparisons earlier in this section were made
with the old Metrica labels and are kept as they were run. `scripts/check_metrica_relabel.py`
reruns it; the report is `models/relabel_report.json`.

**Nothing from the provided match was used as a label.** No frame of it was annotated by a person
for any purpose, and the five public minutes of ground truth were used once, at the end, as a
pass/fail correctness check with a threshold fixed in advance (section 6).

## 4. Possession rules implemented

- **Dead ball** is whatever the model calls dead: in training, the DFL "ball dead" status and, on
  the Metrica games, the stoppages in Metrica's events (from a ball-out, foul or goal to the next
  set piece) plus every stretch of at least 2 s without a ball inside a team's possession. The
  strongest dead-ball cues the model learned are the ball outside the pitch when last seen and
  players standing still.
- **Loose and contested balls** follow the DFL possession flag and, on the Metrica games,
  Metrica's events: a recovery changes possession, a challenge does not. The decoder will not
  create a possession shorter than 0.5 s, and creates a short one only where the change model sees
  a change; the minimum was chosen on the Metrica games.
- **Blackouts.** No label is ever left open: the model reads possession from team shape and
  movement and from who had the ball before and after the blackout; the decoder keeps a
  possession through a blackout unless the evidence for a change outweighs that for staying.
- **Restarts** split possessions: a dead-ball stretch of at least 1 s separates two possessions,
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
exactly (SHA-256 `fdaf2ed296c5f9ab21eaab832e6f57e17034b481facdf53d2885154a164d1a9e`; checked by
running it twice). Retraining from scratch, including the DFL download, is described in the
README; LightGBM runs with `deterministic=True`, every random draw has a fixed seed. The sequence
network's weights are committed (`models/sequence_net.npz`, 0.6 MB): retraining it on a GPU is
not bit-for-bit reproducible, and the submission uses the committed weights.

## 6. Results and limitations

**The submitted model** (LightGBM on DFL + Metrica, weight 2, averaged with the sequence network,
all trained on the corrected Metrica labels) scores 0.770, 0.793 and 0.768 (mean 0.777) on the
three Metrica games it never saw, and 0.597 on the DFL matches (section 3). **How the model was built up**, on the DFL matches with DFL-only
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
| + Metrica games in training, weight 2 | 0.590 | – (judged on the Metrica games) |
| + sequence network, averaged | 0.592 | – (judged on the Metrica games) |
| + corrected Metrica labels (submitted) | 0.597 | 5 of 7 |

Frame accuracy of the DFL-only model is 88.9 % inside simulated blackouts (87.7 % without tags)
and 94.4 % where the ball is visible. The model predicts 279–339 possessions per match against
370–511 true ones: most of the missing ones are very short.

**On the provided match** (no labels, so no score): 285 possessions, mean 12.7 s; graded frames
34 % `H`, 23 % `A`, 43 % `D`. On the five public minutes (a correctness check only, passed: it
required `spellF1` ≥ 0.3256 and home/away right on at least half of the live frames; the ball is
visible on 88 % of them against 59 % of the match, so it is optimistic): `spellF1` 0.568, home/away
correct on 91.9 % of live frames, 14 predicted possessions against 17 true ones. Earlier models
scored 0.594 (before the Metrica labels were corrected), 0.613 (LightGBM alone, weight 2), 0.571 (weight 1), 0.645 (DFL only) and 0.619 (tracking
only) there; one possession moves the score by about 0.06 in those
five minutes, against about 1,000 possessions in the three Metrica games, and by the rules the
public window does not choose between versions.

**Blackout consistency without labels.** We added simulated blackouts where the ball is visible
on the real match and compared the two predictions: on the newly blacked-out frames they agree on
83.2 % of frames (84.5 % before the Metrica labels were corrected, 78.2 % with LightGBM alone,
77.6 % with tracking only), and the possessions match with `spellF1` 0.84 (0.85, 0.82, 0.79). The sequence network makes the prediction steadier through
blackouts; they still cost something like a sixth of the frame decisions they cover.

**Domain check.** Feature distributions on the real match and on the corrupted DFL matches were
compared feature by feature; no standardised mean difference exceeds 0.7 and no missing-value
share differs by more than 0.25, so no feature was dropped. Tracked players per team match
(10.2 vs 10.2). The largest differences: the ball moves faster (6.8 vs 4.4 m/s averaged over
20 s) and Barcelona's team length is larger (55 vs 44 m). The tag features agree closely (for
example 1.62 against 1.64 phase tags active at once).

**Limitations.**

- *The closest estimate of the score on this match* is the submitted model's 0.777 on the three
  Metrica sample games, each predicted by a model that never saw it: same tracking provider, the
  ball missing on 32-41 % of frames, possessions changing about as often as in the public five
  minutes. Their labels were derived by us from Metrica's events and corrected where the ball is
  missing for at least 2 s, not taken from corrected tracking, so this is an estimate; the
  correction itself raises the score these games give (the previous submission scores 0.733
  against the corrected labels, 0.730 against the old ones).
- *The DFL score is not the score on this match.* The training labels are another provider's
  possession flag, the corruption is a model of the real tracking's defects, and the DFL tags are
  our model of Metrica's tagging; "players in the box" fires 97 times on this match and only 57
  times per DFL match in the simulation. The public window is too short to say how far the real
  score is from 0.59.
- *Short possessions are missed.* The public window has two (0.8 s and 1.3 s) and we find
  neither; on the DFL matches we predict a quarter to a third fewer possessions than there are.
- *Stoppages are under-detected on the public window*: dead ball is 19.6 % of it, our output
  6.5 %. Three stoppages of 14–20 s are mostly labelled as possession there. On the whole match
  our output is 43 % dead ball, so this may be specific to those minutes; we cannot tell without
  labels.
- *A large gap between in-sample and out-of-fold scores* (0.89 vs 0.59 on DFL, DFL-only model) says the model
  memorises match specifics. Stronger regularisation, averaging several LightGBM models or several
  networks, and training on the match's own predicted labels did not help on the Metrica games
  (section 3); more matches with real labels would.
- Turnovers deep inside a long blackout are guessed from team movement only. A published model
  that reads possession from the players alone did no better on the DFL labels (section 3).
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
from other providers; use the broadcast ball despite its small coverage, with a threshold judged
on the training matches; and detect stoppages from the referee's whistle in the audio track. The
video ball detector puts the true ball first on 59 % of the frames where the tracking sees it and
among its top five boxes on 66 %; a tracker that picks, among the boxes, the one consistent with
where the ball was last seen might keep the right one. That would need a new rule fixed in
advance and a detection run over all 64,643 blackout frames (about 7 hours on an Apple GPU).
