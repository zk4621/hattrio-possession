"""Smart Tagging as model input, and Smart-Tagging-like tags simulated on the DFL matches.

The provided match carries Metrica's automatic Smart Tagging; the DFL matches do not. To train
and validate a model that reads tags, the DFL matches get simulated tags built from their own
event data (set pieces with team and time), thinned and shifted to match what the real tags look
like. What the real tags look like is measured on the provided match as aggregates only
(`scripts/measure_tag_profile.py` -> `models/tag_profile.json`).
"""
from __future__ import annotations

import mmap
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .frames import FPS

CLIP = 3000
SP_CODE = "SET PIECES"
SP_KINDS = ("throw", "corner", "goalkick", "freekick", "kickoff")
REAL_TYPE_TO_KIND = {"throw-in": "throw", "corner kick": "corner", "goal kick": "goalkick",
                     "free kick": "freekick", "kick off (after goal)": "kickoff",
                     "kick off (start)": "kickoff"}
DFL_EVENT_TO_KIND = {"ThrowIn": "throw", "CornerKick": "corner", "GoalKick": "goalkick",
                     "FreeKick": "freekick", "KickOff": "kickoff"}
SP_NAMES = (["sp_since", "sp_until", "sp_team_last", "sp_team_next"]
            + [f"sp_{k}_{w}" for k in SP_KINDS for w in ("since", "until")])
# Phase codes that name the team in possession. Their "DEFENDING ..." / "DEFENSIVE TRANSITION"
# twins are the same instances seen from the other team, so they add nothing and are not used.
PH_KEYS = {"BUILD UP": "buildup", "PROGRESSION": "progression", "CREATING CHANCES": "chances",
           "ATTACKING TRANSITION": "transition", "LONG BALL": "longball",
           "BALL IN FINAL THIRD": "final3", "BALL IN THE BOX": "ballbox",
           "PLAYERS IN THE BOX": "playersbox"}
PH_NAMES = (["ph_active", "ph_count", "ph_team_last", "ph_since", "ph_team_next", "ph_until"]
            + [f"ph_{k}" for k in PH_KEYS.values()])
TAG_NAMES = SP_NAMES + PH_NAMES


@dataclass(frozen=True)
class Tag:
    code: str     # Smart Tagging code, upper case, e.g. "SET PIECES"
    kind: str     # set-piece kind (SP_KINDS) or "" for phases
    start: int    # first frame
    end: int      # first frame after the tag
    team: int     # +1 home, -1 away, 0 not given


# -- the provided tags ---------------------------------------------------------------------------
def competition_tags() -> list[Tag]:
    """The provided Smart Tagging, teams mapped by the names in the tracking metadata."""
    from .paths import data_dir, vendor_import
    games = vendor_import("games")
    st = vendor_import("load_smart_tagging")
    meta, = games.require(["atd_meta"], "BAR-ATH", data_dir=str(data_dir()))
    names = [(t.findtext("Name") or "").strip() for t in ET.parse(meta).getroot().iter("Team")]
    sign = {names[0]: 1, names[1]: -1}
    out = []
    for e in st.load(data_dir=str(data_dir())):
        tags = {str(k).strip(): str(v or "").strip() for k, v in (e.get("tags") or {}).items()}
        code = str(e.get("code") or "").strip().upper()
        kind = REAL_TYPE_TO_KIND.get(tags.get("Type", "").lower(), "") if code == SP_CODE else ""
        out.append(Tag(code, kind, int(e["start_frame"]), int(e["end_frame"]),
                       sign.get(tags.get("Team", ""), 0)))
    return out


# -- features ------------------------------------------------------------------------------------
def _since_until(starts: np.ndarray, teams: np.ndarray, t: np.ndarray):
    if len(starts) == 0:
        clip = np.full(len(t), float(CLIP))
        zero = np.zeros(len(t))
        return clip, clip.copy(), zero, zero.copy()
    i = np.searchsorted(starts, t, side="right")          # tags starting at or before t
    last, nxt = i - 1, i
    has_last, has_next = last >= 0, nxt < len(starts)
    li, ni = np.maximum(last, 0), np.minimum(nxt, len(starts) - 1)
    since = np.where(has_last, np.minimum(t - starts[li], CLIP), CLIP).astype(np.float64)
    until = np.where(has_next, np.minimum(starts[ni] - t, CLIP), CLIP).astype(np.float64)
    team_last = np.where(has_last, teams[li], 0).astype(np.float64)
    team_next = np.where(has_next, teams[ni], 0).astype(np.float64)
    return since, until, team_last, team_next


def set_piece_features(tags: list[Tag], a: int, z: int) -> dict[str, np.ndarray]:
    """Frames since and until the set-piece tags around each frame of [a, z), and their team."""
    t = np.arange(a, z)
    sp = sorted((g for g in tags if g.code == SP_CODE), key=lambda g: (g.start, g.kind))
    starts = np.array([g.start for g in sp], dtype=np.int64)
    teams = np.array([g.team for g in sp], dtype=np.int64)
    f = {}
    f["sp_since"], f["sp_until"], f["sp_team_last"], f["sp_team_next"] = \
        _since_until(starts, teams, t)
    for k in SP_KINDS:
        m = np.array([g.kind == k for g in sp], dtype=bool)
        s, u, _, _ = _since_until(starts[m] if len(sp) else starts,
                                  teams[m] if len(sp) else teams, t)
        f[f"sp_{k}_since"], f[f"sp_{k}_until"] = s, u
    return f


def phase_features(tags: list[Tag], a: int, z: int) -> dict[str, np.ndarray]:
    """Team-signed count of the phase tags active at each frame of [a, z), and the nearest starts."""
    t = np.arange(a, z)
    ph = sorted((g for g in tags if g.code in PH_KEYS), key=lambda g: (g.start, g.code))
    f = {}
    total = np.zeros(z - a + 1)
    count = np.zeros(z - a + 1)
    for code, key in PH_KEYS.items():
        d = np.zeros(z - a + 1)
        for g in ph:
            lo, hi = max(g.start, a), min(g.end, z)
            if g.code != code or hi <= lo:
                continue
            d[lo - a] += g.team
            d[hi - a] -= g.team
            count[lo - a] += 1
            count[hi - a] -= 1
        f[f"ph_{key}"] = np.cumsum(d)[:-1]
        total += d
    f["ph_active"] = np.cumsum(total)[:-1]
    f["ph_count"] = np.cumsum(count)[:-1]
    starts = np.array([g.start for g in ph], dtype=np.int64)
    teams = np.array([g.team for g in ph], dtype=np.int64)
    f["ph_since"], f["ph_until"], f["ph_team_last"], f["ph_team_next"] = \
        _since_until(starts, teams, t)
    return f


def tag_features(tags: list[Tag], a: int, z: int) -> dict[str, np.ndarray]:
    f = set_piece_features(tags, a, z)
    f.update(phase_features(tags, a, z))
    return f


# -- restarts seen in tracking -------------------------------------------------------------------
def ball_speed(ball: np.ndarray) -> np.ndarray:
    """Centred ball speed in m/s, NaN where a neighbour frame is missing."""
    b = np.asarray(ball, dtype=np.float64)
    s = np.full(len(b), np.nan)
    if len(b) > 2:
        s[1:-1] = np.linalg.norm(b[2:] - b[:-2], axis=1) * FPS / 2
    return s


def detect_restart(ball: np.ndarray, center: int, lo: int, hi: int, still: int = 12,
                   still_speed: float = 1.5, kick_speed: float = 4.0, ramp: int = 10,
                   speed: np.ndarray | None = None) -> int | None:
    """First frame in [center+lo, center+hi] where a ball seen still for `still` frames moves off.

    A restart is a still ball (placed, or held for a throw-in) that is suddenly played: the frame
    returned is the first one at or above `still_speed` after `still` frames below it, provided
    the ball passes `kick_speed` within `ramp` frames (smoothed tracking speeds up gradually).
    Returns None if the ball is not seen doing that inside the window.
    """
    s = ball_speed(ball) if speed is None else speed
    T = len(s)
    with np.errstate(invalid="ignore"):
        is_still = s < still_speed                 # NaN compares False: an unseen ball is not still
    r0, r1 = max(center + lo, still), min(center + hi, T - 1)
    for r in range(r0, r1 + 1):
        if not s[r] >= still_speed or not is_still[r - still:r].all():
            continue
        if np.nanmax(s[r:r + ramp]) > kick_speed:
            return r
    return None


# -- DFL events ----------------------------------------------------------------------------------
_FRAMESET = re.compile(rb"<FrameSet ([^>]*)>")
_FIRST_FRAME = re.compile(rb'<Frame N="(\d+)" T="([^"]+)"')


def ball_anchors(positions_path) -> dict[int, datetime]:
    """Wall-clock time of the first ball frame of each half of a DFL position file."""
    out = {}
    with open(positions_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for m in _FRAMESET.finditer(mm):
            attrs = m.group(1)
            if b'TeamId="BALL"' not in attrs:
                continue
            half = 1 if b'GameSection="firstHalf"' in attrs else \
                2 if b'GameSection="secondHalf"' in attrs else None
            if half is None:
                continue
            fm = _FIRST_FRAME.search(mm, m.end(), m.end() + 4096)
            if fm:
                out[half] = datetime.fromisoformat(fm.group(2).decode())
    return out


def dfl_set_pieces_from_files(meta_path, positions_path, events_path,
                              period: np.ndarray) -> list[tuple[str, int, int]]:
    """[(kind, row, team)] for the DFL set pieces, row = the restart's row of the frame table."""
    root = ET.parse(meta_path).getroot()
    role = {t.get("TeamId"): t.get("Role") for t in root.iter("Team")}
    anchors = ball_anchors(positions_path)
    period = np.asarray(period)
    rows = {p: np.flatnonzero(period == p) for p in anchors}
    out = []
    for ev in ET.parse(events_path).getroot().iter("Event"):
        kids = list(ev)
        if not kids or kids[0].tag not in DFL_EVENT_TO_KIND or not ev.get("EventTime"):
            continue
        t = datetime.fromisoformat(ev.get("EventTime"))
        half = 2 if 2 in anchors and t >= anchors[2] - _seconds(5) else 1
        if half not in anchors or not len(rows[half]):
            continue
        row = int(rows[half][0] + round((t - anchors[half]).total_seconds() * FPS))
        if not rows[half][0] <= row <= rows[half][-1]:
            continue
        team_id = kids[0].get("Team")
        if team_id is None:
            play = kids[0].find("Play")
            team_id = play.get("Team") if play is not None else None
        team = {"home": 1, "guest": -1}.get(role.get(team_id), 0)
        out.append((DFL_EVENT_TO_KIND[kids[0].tag], row, team))
    out.sort(key=lambda e: e[1])
    return out


def _seconds(s: float):
    from datetime import timedelta
    return timedelta(seconds=s)


def dfl_set_pieces(match_id: str, period: np.ndarray) -> list[tuple[str, int, int]]:
    """Set pieces of one DFL match, cached under cache/."""
    import json

    from .io_dfl import _find
    from .paths import CACHE
    cache = CACHE / f"dfl_{match_id}_setpieces.json"
    if cache.exists():
        return [tuple(e) for e in json.loads(cache.read_text())]
    meta = _find(f"DFL_02_01_matchinformation_*_DFL-MAT-{match_id}.xml")
    positions = _find(f"DFL_04_03_positions_raw_observed_*_DFL-MAT-{match_id}.xml")
    events = _find(f"DFL_03_02_events_raw_*_DFL-MAT-{match_id}.xml")
    out = dfl_set_pieces_from_files(meta, positions, events, period)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


# -- phase candidates on DFL ---------------------------------------------------------------------
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
CHANCE_DEPTH, CHANCE_HALF_WIDTH = 25.0, 20.0
LONG_BALL_M, LONG_BALL_FRAMES = 30.0, 50
# (frames of possession before the moment, frames after it) each code requires, and whether
# every moment of a possession counts or only its first. Build-ups and regains in the real tags
# are sustained (their team keeps the ball for 10-20 s), so the simulation requires that. The
# real zone tags fire on nearly every entry, also when the ball is lost right after it (a cross
# that is cleared), so the zone codes require nothing.
NEEDS = {"ATTACKING TRANSITION": (0, 150, False), "BUILD UP": (100, 250, False),
         "PROGRESSION": (50, 75, True), "CREATING CHANCES": (0, 0, True),
         "LONG BALL": (100, 75, False), "BALL IN FINAL THIRD": (0, 0, True),
         "BALL IN THE BOX": (0, 0, True), "PLAYERS IN THE BOX": (0, 0, True)}
MIN_GAP = 250         # two moments of one code closer than 10 s are one moment, as in the real tags
# Codes whose moment can be found in tracking alone, on the provided match as on DFL.
ENTRY_CODES = ("PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX", "CREATING CHANCES",
               "PLAYERS IN THE BOX", "LONG BALL")


def _segments(y: np.ndarray, a: int, z: int):
    w = y[a:z]
    if not len(w):
        return []
    b = np.flatnonzero(w[1:] != w[:-1]) + 1
    bounds = np.concatenate(([0], b, [len(w)]))
    return [(a + int(s0), a + int(s1), int(w[s0])) for s0, s1 in zip(bounds[:-1], bounds[1:])]


def _crossings(v: np.ndarray, thresh: float) -> np.ndarray:
    """Indices where v, over its non-NaN values, goes from below thresh to at or above it."""
    idx = np.flatnonzero(~np.isnan(v))
    if len(idx) < 2:
        return np.array([], dtype=np.int64)
    w = v[idx]
    k = np.flatnonzero((w[:-1] < thresh) & (w[1:] >= thresh)) + 1
    return idx[k]


def _masked(bx: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """bx where `inside`, -inf where the ball is seen elsewhere, NaN where it is not seen."""
    return np.where(np.isfinite(bx), np.where(inside, bx, -np.inf), np.nan)


def entry_moments(code: str, ft, a: int, z: int, team: int, dirx: float) -> np.ndarray:
    """Offsets in [0, z-a) where the moment of `code` happens for `team` attacking along dirx."""
    L = ft.pitch[0]
    b = ft.ball[a:z].astype(np.float64)
    bx, by = dirx * b[:, 0], np.abs(b[:, 1])
    with np.errstate(invalid="ignore"):
        if code == "PROGRESSION":
            return _crossings(bx, -L / 6)
        if code == "BALL IN FINAL THIRD":
            return _crossings(bx, L / 6)
        if code == "BALL IN THE BOX":
            return _crossings(_masked(bx, by < BOX_HALF_WIDTH), L / 2 - BOX_DEPTH)
        if code == "CREATING CHANCES":
            return _crossings(_masked(bx, by < CHANCE_HALF_WIDTH), L / 2 - CHANCE_DEPTH)
        if code == "PLAYERS IN THE BOX":
            players = (ft.home if team == 1 else ft.away)[a:z].astype(np.float64)
            inside = ((dirx * players[..., 0] > L / 2 - BOX_DEPTH)
                      & (np.abs(players[..., 1]) < BOX_HALF_WIDTH)).sum(axis=1)
            return _crossings(inside.astype(np.float64), 1.0)
        if code == "LONG BALL":
            travel = np.full(len(b), np.nan)
            if len(b) > LONG_BALL_FRAMES:
                travel[:-LONG_BALL_FRAMES] = np.linalg.norm(
                    b[LONG_BALL_FRAMES:] - b[:-LONG_BALL_FRAMES], axis=1)
            return _crossings(travel, LONG_BALL_M)
    raise KeyError(code)


def attack_sign(ft) -> np.ndarray:
    """Per frame: +1 where the home team attacks towards +x, -1 where towards -x, 0 outside play."""
    out = np.zeros(ft.n_frames)
    for _, a, z in ft.period_slices():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # halves with no tracked team
            gap = np.nanmean(np.nanmean(ft.away[a:z, :, 0], 1) - np.nanmean(ft.home[a:z, :, 0], 1))
        out[a:z] = -1.0 if np.isfinite(gap) and gap < 0 else 1.0
    return out


def dfl_phase_candidates(ft, y: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    """Moments on a clean DFL match that the real phase tags appear to mark: [(row, team)].

    team: +1 home, -1 away. Possession comes from the labels y, geometry from the tracking.
    """
    from .labels import A, H
    L, W = ft.pitch
    out = {code: [] for code in PH_KEYS}
    sign = attack_sign(ft)

    def add(code, s0, n, idx):
        before, after, every = NEEDS[code]
        ok = [int(i) for i in np.atleast_1d(idx) if i >= before and n - i >= after]
        for i in ok if every else ok[:1]:
            if out[code] and s0 + i - out[code][-1][0] < MIN_GAP:
                continue
            out[code].append((s0 + i, team))

    for _, a, z in ft.period_slices():
        segs = _segments(y, a, z)
        for i, (s0, s1, k) in enumerate(segs):
            if k not in (H, A):
                continue
            team = 1 if k == H else -1
            dirx = sign[s0] * team
            n = s1 - s0
            bx = dirx * ft.ball[s0:s1, 0].astype(np.float64)
            prev = segs[i - 1][2] if i else None
            if prev in (H, A) and prev != k:
                add("ATTACKING TRANSITION", s0, n, 0)
            add("BUILD UP", s0, n, np.flatnonzero(bx < -L / 6))
            for code in ENTRY_CODES:
                add(code, s0, n, entry_moments(code, ft, s0, s1, team, dirx))
    return out


def simulate_phase_tags(candidates: dict, profile: dict, seed: int) -> list[Tag]:
    """Phase tags from candidate moments, thinned to the real count and started a little before.

    profile["codes"][code] holds `keep`, `team_known`, `team_flip` (share of tags naming the
    wrong team, as the real tags sometimes do), `duration_q` and, where it was measured,
    `pre_q`: how long before its moment a tag starts, quantiles 0..100. Codes without `pre_q`
    use the [low, high] range profile["pre_frames"].
    """
    rng = np.random.default_rng(seed)
    lo, hi = profile["pre_frames"]
    out = []
    for code in sorted(candidates):
        k = profile["codes"].get(code)
        for row, team in candidates[code]:
            u_keep, u_pre, u_dur, u_team, u_flip = rng.random(5)
            if k is None or u_keep >= k["keep"]:
                continue
            if "pre_q" in k:
                pre = int(round(float(np.interp(u_pre * 100.0, np.arange(101), k["pre_q"]))))
            else:
                pre = int(round(lo + u_pre * (hi - lo)))
            dur = max(1, int(round(float(np.interp(u_dur * 100.0, np.arange(101),
                                                    k["duration_q"])))))
            start = int(row) - pre
            team = -int(team) if u_flip < k.get("team_flip", 0.0) else int(team)
            out.append(Tag(code, "", start, start + dur, team if u_team < k["team_known"] else 0))
    return out


# -- simulation ----------------------------------------------------------------------------------
def simulate_set_piece_tags(events, profile: dict, seed: int) -> list[Tag]:
    """Smart-Tagging-like set-piece tags from true set pieces, thinned and shifted as measured.

    profile["kinds"][kind] holds `keep` (share of that kind the real tagging contains),
    `team_known` (share with a team) and `offset_q` (restart frame minus tag start, quantiles
    0..100); profile["duration_q"] holds the tag length quantiles.
    """
    rng = np.random.default_rng(seed)
    out = []
    for kind, row, team in events:
        k = profile["kinds"].get(kind)
        u_keep, u_off, u_dur, u_team = rng.random(4)
        if k is None or u_keep >= k["keep"]:
            continue
        off = int(round(float(np.interp(u_off * 100.0, np.arange(101), k["offset_q"]))))
        dur = max(1, int(round(float(np.interp(u_dur * 100.0, np.arange(101),
                                                profile["duration_q"])))))
        start = int(row) - off
        out.append(Tag(SP_CODE, kind, start, start + dur, int(team) if u_team < k["team_known"]
                       else 0))
    return out
