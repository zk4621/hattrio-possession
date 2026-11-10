# Ball-possession pipeline: design

More than a Hack 2027, Phase 1 (Metrica Sports / FC Barcelona). Written 2026-11-09.
Status: hybrid approach approved by the team in conversation; they asked to go straight to the
implementation plan, so this document is the short record of what was decided and why.

Challenge repository: https://github.com/metrica-sports/more-than-a-hack-2027 (commit `f8f2629`).

## The task

For each of the 199,319 frames of the tactical video of Barcelona vs Athletic Club, output one
label: `H` (home in possession, ball alive), `A` (away), `D` (dead ball) or `X` (no answer).
The submission is a `frame,label` CSV. Only the two halves are graded: video frames
11,185–90,651 and 110,808–190,197 (158,857 frames).

Deliverables: the CSV, a public repository that regenerates exactly that CSV from the provided
data, and a short technical document. Deadline 17 November 2026, 23:59 CET. Top 20 teams advance.

## What the ranking metric rewards

Ranking is on `spellF1`. A possession is a maximal run of `H` or of `A`. Predicted possessions are
matched one-to-one to true ones of the same team by temporal IoU; F1 is averaged over IoU
thresholds 0.3–0.7. Consequences:

- Each possession counts once, whatever its length. Flicker creates spurious possessions and
  destroys precision. The published baseline moves from 0.05 to 0.24 by smoothing alone.
- A dead ball splits a possession (`H H D D H H` is two possessions), so finding stoppages matters
  as much as finding turnovers.
- Boundaries must be accurate: a perfect answer shifted by one second scores 0.65.
- Inside a graded half the scorer forward-fills `X`, so `X` there is never better than carrying
  the previous label ourselves. We write `X` only outside the halves.

The ground truth is "sticky": while the ball is loose or contested, possession stays with the last
team; it changes only when the opponent gains controlled possession.

## Rules that shape the design

1. No hand labels on the provided match, in any amount, for any purpose. No rules or constants
   tied to specific frames that a person read off the video.
2. The 5 public minutes of ground truth are for checking that the pipeline is correct. They are
   not training data and not a tuning set.
3. Labelled data from other matches is allowed without limit, if the organisers can access it and
   the technical document describes it.
4. The match data, or any extract of it, must not be given to a cloud-hosted LLM. LLMs may
   otherwise be used freely.
5. The repository must regenerate the submitted file exactly.

## What the data showed (measured 2026-11-10, aggregates only)

Produced by `scripts/describe_data.py`; nothing below is frame content.

- The bundle matches the documentation: 199,319 tactical records, 46 track slots (24 home, 22
  away), ball visible on 56.94% of all frames and 59.31% of graded frames, 20.49 tracked players
  on average, 638 Smart Tagging instances, one kick-off tag per half.
- Each record is `frame_id:46 groups:ball`, every group `x,y,speed`, missing values `NaN`.
  The ball's speed field is finite on only 9% of ball frames.
- **The ball is not lost in short flickers. It is lost in 147 long blackouts.** Median blackout
  268 frames (11 s), 90th percentile 836 frames, longest 5,537 frames. 99.9% of missing frames sit
  in blackouts longer than 5 s; 44% in blackouts longer than 30 s. Between blackouts the ball is
  visible without interruption, median 345 frames.
- Ball displacement across a blackout: median 10 m, 90th percentile 35 m.
- The ball is observed outside the pitch on 4.4% of ball frames.
- Tracked players per team: median 11, 10th percentile 9, 1st percentile 2. A track slot lives a
  median of 888 frames.
- Broadcast tracking: 164,488 records, 51 slots, ball visible on 50.5%, homography valid on 65.8%.
- The tactical video has 199,319 frames at 25 fps and carries an audio track.

From the public five minutes, used only to understand what the labels mean:

- A blackout is not a stoppage: about half of the missing frames there are dead ball, half are
  live play. Two of the four stoppages have the ball visible throughout.
- That window is unrepresentative: the ball is visible on 88% of it, against 59% overall.
- Possessions can be very short; the shortest there lasts 0.8 s.

Consequences for the design:

- Interpolating the ball across gaps is pointless. During a blackout, possession has to be read
  from the players alone, or the ball has to come from another source.
- The corruption applied to the training matches must be long blackouts with the measured length
  distribution, not short random dropouts.
- Two extra sources become worth measuring, because each can be checked against the tactical ball
  on frames where that is visible: the broadcast tracking's ball, once its timeline is aligned,
  and the tactical camera's view centre as a coarse ball position.

## Approach

A per-frame probability model trained on other matches, followed by a segment decoder.

1. **Common frame table.** Tracking from any source becomes one structure: metres, origin at the
   pitch centre, players as two unordered sets (no identities), ball position or NaN.
2. **Features** computed per frame from that table only: ball-to-nearest-player distances per
   team, ball kinematics, ball position relative to the pitch boundary, team shape and motion,
   ball-visibility gap structure, and past/future window aggregates of those.
3. **Training data: the public DFL dataset** (Bassek et al., Scientific Data 2025; 7 Bundesliga
   matches, 25 Hz, CC-BY 4.0). Each ball frame carries `BallPossession` (1 home / 2 away) and
   `BallStatus` (1 alive / 0 dead), which maps directly onto `H`/`A`/`D`.
4. **Degradation.** DFL tracking is clean; the provided tracking loses the ball in long blackouts
   and tracks fewer than 11 players per team part of the time. We measure aggregate statistics of
   that from the provided tracking (how often a blackout starts, by whether the ball is at the
   pitch boundary; blackout lengths; how many players are missing; jitter) and apply the same
   corruption to the DFL matches before computing features. The model therefore learns to label
   long stretches from player positions alone.
5. **Model.** LightGBM, three classes, deterministic CPU training. Weights are committed, so
   regenerating the submission is inference only.
6. **Decoder.** Minimum-duration Viterbi over `H`/`A`/`D` with a switch penalty, run per half, on
   integer-quantised log-probabilities. This is where `spellF1` is won: it outputs segments, not
   frames.
7. **Validation.** Leave-one-match-out on the 7 degraded DFL matches, scored with the official
   scorer. Every hyperparameter is chosen there. The public 5 minutes are used once, as a
   pass/fail correctness gate with a threshold fixed in advance.

A rule-based probability source (no training) feeds the same decoder. It gives an end-to-end
submission early, a comparison point, and a fallback if the learned model does not beat it on the
external matches.

## Decisions and why

- **Train on other matches rather than hand-tune thresholds.** Rule 2 leaves no labelled data on
  this match to tune against, so thresholds would need other matches anyway.
- **LightGBM rather than a neural sequence model.** Deterministic, fast on CPU, tolerant of NaN
  features, and easy to reproduce bit-for-bit.
- **Integer scores in the decoder.** Removes platform-dependent floating-point ties, which
  protects rule 5.
- **Set features, no player identities.** The provided tracking has 46 anonymous slots that swap;
  only team membership is reliable.
- **Vendor the challenge repository** (MIT) rather than depend on it, so one clone reproduces.
- **No kloppy.** Both formats are simple enough to parse directly, and kloppy's object model is
  slow and memory-hungry on 400 MB position files.

## Extra ball sources, measured before they are built in

Both produce a ball position, so both can be checked without labels against the tactical ball on
frames where it is visible, and both can be imitated on the DFL matches.

- **Broadcast ball.** Align the broadcast timeline to the tactical one by matching player
  positions, then copy the broadcast ball into tactical blackouts. Adopted only if it agrees with
  the tactical ball and fills a meaningful share of blackout frames. After merging, the blackout
  statistics are measured again on the merged ball, so the DFL corruption follows automatically.
- **Camera view centre.** The pitch point at the centre of the tactical image, from the
  homography, as a coarse ball position on every frame. Adopted only if it tracks the ball. On
  DFL it is imitated as the true ball plus noise of the measured size.

## Out of scope

- Smart Tagging as extra evidence (used only to find the kick-off of each half); it has no
  counterpart on external matches.
- Referee-whistle detection from the audio track.
- Our own ball detection from video.
- Training on pseudo-labels from this match.

## Open risks

- **Label definition mismatch.** DFL's possession flag may not follow the same "controlled
  possession" rule. The plan measures possession-length statistics on DFL first and stops for a
  human decision if they are implausible.
- **Domain gap.** A model trained on degraded clean tracking may not transfer. Mitigations: noise
  profile measured from the real tracking, a feature-distribution comparison between domains, the
  rule-based fallback.
- **No labelled validation on the target match.** By rule. The external score is an estimate, not
  a guarantee.
