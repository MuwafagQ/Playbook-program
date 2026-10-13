"""Match report from tagged passes and pitch positions: possessions, automatic phases (by thirds),
regains, counter-attacks and tempo.

  python -m tools.match_report --passes passes_final.csv --reviews <dir of review jsons> \
      --pass-pieces pass_pieces.json --tracks <dir of <segment>.csv.gz> --index review_index.json \
      [--moments <dir of momentum marks>] --out <dir>

Inputs
  passes_final.csv   tools/pass_file/final.py output (shirt numbers, frames, outcome, notes)
  reviews            the app's pass documents (pixel position of passer / receiver, track pieces)
  tracks             notebooks/review_export.ipynb tracks_small.csv.gz per segment (x_m, y_m: pitch cm,
                     120 x 70 m pitch model)
  moments            the app's Momentum marks (optional): used only to measure how well the automatic
                     phases agree with a person's marks

Positions: raw pitch positions jitter (camera calibration per frame), so tracks are smoothed (rolling
median then mean over ~0.5 s) and positions are read at pass events (passer at the kick, receiver at the
touch). Thirds are 40 m bands of the 120 m model, in each team's attacking direction (from where its
goalkeeper plays, per half).

Phases (the app's Momentum labels): the team with the ball is in Buildup (own third), Transition
(middle third) or Finishing (final third); the other team is in Retreat, or Transition when the ball is
in the middle third. Dead ball: after the ball goes out until the restart.

Regain: the ball changes team in open play: an interception, a loose ball (incomplete pass) or a tackle
/ duel (the next pass is by the other team without an interception). Restarts after the ball went out are
not regains.

Counter-attack (scored per regain; the person's review in the app has the final word):
  quick     first forward action within FIRST_S seconds: a pass gaining >= FWD_M metres toward goal
            (within 45 degrees of straight forward), or the ball carried >= CARRY_M metres
  direct    progression toward goal >= DIRECT_MPS m/s over the first HORIZON_S seconds
  exposed   at the regain, at most EXPOSED_N visible opponent outfield players are goal-side of the ball
  label     counter: quick and (direct or exposed); possible: quick or direct; else none
  outcome   the furthest third the ball reaches before the team loses it (or HORIZON_S runs out),
            and goal if a goal note follows
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

FPS = 30.0
PITCH_L, PITCH_W = 120.0, 70.0           # model pitch (m)
FIRST_S, HORIZON_S = 5.0, 12.0           # first forward action / how long a counter is followed
FWD_M, CARRY_M = 10.0, 10.0
DIRECT_MPS, EXPOSED_N = 2.5, 5
SET_PIECES = ("throw", "free kick", "corner", "goal kick", "kick off", "kick-off", "penalty")
PHASES = ("Buildup", "Transition", "Finishing")


# ---------- tracks
def load_tracks(tracks_dir: str) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(os.path.join(tracks_dir, "*.csv.gz"))):
        t = pd.read_csv(f, usecols=["frame", "raw_tracker_id", "class_id", "team_id", "x1", "y1", "x2", "y2", "x_m", "y_m"])
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t = t[t.class_id.isin([1, 2])].copy()
    t["x"], t["y"] = t.x_m / 100.0, t.y_m / 100.0
    t = t.sort_values(["raw_tracker_id", "frame"])
    g = t.groupby("raw_tracker_id")
    for c in ("x", "y"):
        med = g[c].transform(lambda s: s.rolling(9, center=True, min_periods=3).median())
        t["s" + c] = med.groupby(t.raw_tracker_id).transform(lambda s: s.rolling(9, center=True, min_periods=3).mean())
    t["cx"], t["foot"] = (t.x1 + t.x2) / 2, t.y2
    return t.reset_index(drop=True)


def ball_owner(tracks_dir: str, n_frames: int) -> np.ndarray:
    """Per frame: the team (0 / 1) of the player whose feet are nearest the ball on screen, if close
    (within 0.6 of that player's box height), else -1."""
    own = np.full(n_frames, -1, dtype=np.int8)
    for f in sorted(glob.glob(os.path.join(tracks_dir, "*.csv.gz"))):
        t = pd.read_csv(f, usecols=["frame", "class_id", "team_id", "x1", "y1", "x2", "y2"])
        b = t[t.class_id == 0].drop_duplicates("frame").set_index("frame")
        b = pd.DataFrame({"bx": (b.x1 + b.x2) / 2, "by": (b.y1 + b.y2) / 2})
        p = t[t.class_id.isin([1, 2]) & t.team_id.isin([0, 1])].join(b, on="frame", how="inner")
        p = p.assign(d=np.hypot((p.x1 + p.x2) / 2 - p.bx, p.y2 - p.by), h=p.y2 - p.y1)
        p = p[p.d <= 0.6 * p.h]
        best = p.loc[p.groupby("frame").d.idxmin()]
        fr = best.frame.values.astype(int)
        own[fr[fr < n_frames]] = best.team_id.values[fr < n_frames]
    return own


class Positions:
    """Smoothed pitch position of a track (piece) or of the box at a pixel, at a frame."""
    def __init__(self, t: pd.DataFrame):
        self.t = t
        self.by_frame = {f: g for f, g in t.groupby("frame")}
        self.by_piece = {p: g.drop_duplicates("frame").set_index("frame") for p, g in t.groupby("raw_tracker_id")}

    def at(self, frame: int, piece=None, px=None, py=None):
        if piece is not None:
            try:
                base = int(str(piece).split("_")[0])
            except ValueError:
                base = None
            g = self.by_piece.get(base)
            if g is not None:
                for d in (0, -1, 1, -2, 2, -3, 3, -5, 5):
                    if frame + d in g.index:
                        r = g.loc[frame + d]
                        if not np.isnan(r["sx"]):
                            return float(r["sx"]), float(r["sy"])
        if px is not None:
            for d in (0, -1, 1, -2, 2):
                g = self.by_frame.get(frame + d)
                if g is None:
                    continue
                dist = np.hypot(g.cx - px, g.foot - py)
                i = dist.idxmin() if len(dist) else None
                if i is not None and dist[i] < 40:
                    return float(g.at[i, "sx"]), float(g.at[i, "sy"])
        return None

    def players(self, frame: int, span: int = 3) -> pd.DataFrame:
        """Visible people around a frame, one row per track, smoothed position."""
        gs = [self.by_frame[f] for f in range(frame - span, frame + span + 1) if f in self.by_frame]
        if not gs:
            return pd.DataFrame(columns=["raw_tracker_id", "team_id", "class_id", "sx", "sy"])
        g = pd.concat(gs)
        return g.groupby("raw_tracker_id").agg(team_id=("team_id", "first"), class_id=("class_id", "first"),
                                               sx=("sx", "median"), sy=("sy", "median")).reset_index()


# ---------- events
def load_events(passes_csv: str, reviews_dir: str, pass_pieces: str, pos: Positions) -> pd.DataFrame:
    p = pd.read_csv(passes_csv, keep_default_na=False)
    rv = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(reviews_dir, "*.json"))}
    pp = json.load(open(pass_pieces)) if pass_pieces and os.path.exists(pass_pieces) else {}
    rows = []
    for r in p.itertuples():
        d = rv.get(r.pass_id, {})
        links = pp.get(r.pass_id, [None, None])
        out = {"pass_id": r.pass_id, "kick": int(r.kick_frame), "touch": int(r.touch_frame) if str(r.touch_frame) not in ("", "nan") else None,
               "half": int(r.half), "segment": r.segment, "clock": r.kick_clock, "outcome": r.outcome,
               "notes": str(r.notes).strip().lower(), "team": r.passer_team, "passer": r.passer_label,
               "receiver": r.receiver_label, "receiver_team": r.receiver_team}
        for i, (side, f) in enumerate((("passer", out["kick"]), ("receiver", out["touch"]))):
            w = d.get(side) or {}
            piece = w.get("piece", links[i] if links else None)
            xy = pos.at(f, piece, w.get("x"), w.get("y")) if f is not None and w else None
            out[side + "_x"], out[side + "_y"] = (xy if xy else (np.nan, np.nan))
        rows.append(out)
    e = pd.DataFrame(rows).sort_values("kick").reset_index(drop=True)
    return e


def attack_direction(e: pd.DataFrame) -> dict:
    """{(half, team): +1 if the team attacks toward x = 120 m}: from where each goalkeeper plays (A97 / BGK),
    else from the team's goal kicks, else from the deepest third of its passes."""
    out = {}
    for h in (1, 2):
        for team, gk in (("A", r"^A97$"), ("B", r"^BGK$")):
            s = e[(e.half == h) & (e.team == team) & e.passer.str.match(gk)].passer_x.dropna()
            if len(s) < 2:
                s = e[(e.half == h) & (e.team == team) & (e.notes == "goal kick")].passer_x.dropna()
            if len(s):
                out[(h, team)] = 1 if s.median() < PITCH_L / 2 else -1
        for team, other in (("A", "B"), ("B", "A")):
            if (h, team) not in out and (h, other) in out:
                out[(h, team)] = -out[(h, other)]
    return out


def fwd(x, h, team, D):
    """Distance from the team's own goal line (m)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return x if D.get((h, team), 1) > 0 else PITCH_L - x


def third(u):
    return np.nan if np.isnan(u) else (0 if u < PITCH_L / 3 else 1 if u < 2 * PITCH_L / 3 else 2)


# ---------- possessions and regains
def is_set_piece(notes: str) -> bool:
    return any(notes.startswith(x) for x in SET_PIECES)


def possessions(e: pd.DataFrame, gained_frames: dict) -> pd.DataFrame:
    """Possessions: consecutive passes of one team. Each has how it started (regain kind, set piece,
    restart) and the frame the team got the ball (the interception touch, else its first kick)."""
    poss, cur, how_next, gained_at = [], None, "kick off", None
    for i, r in e.iterrows():
        new = cur is None or cur["team"] != r.team or r.half != cur["half"] or is_set_piece(r.notes)
        if new:
            if cur is not None:
                cur["end"] = int(r.kick); cur["ended"] = cur.get("ended") or ("lost" if cur["team"] != r.team else "stoppage")
                poss.append(cur)
            if is_set_piece(r.notes):
                kind = "set piece: " + r.notes
            elif cur is not None and cur["team"] != r.team and how_next is None:
                kind = "tackle / duel"
            else:
                kind = how_next or "restart"
            if cur is not None and r.half != cur["half"]:
                kind = "kick off"
            cur = {"team": r.team, "half": int(r.half), "segment": r.segment, "clock": r.clock, "kind": kind,
                   "gained": int(gained_at if (gained_at is not None and kind == "interception")
                                 else gained_frames.get(i, r.kick) if kind in ("tackle / duel", "loose ball") else r.kick),
                   "events": [], "goal": False}
        cur["events"].append(i)
        how_next, gained_at = None, None
        if "goal" in r.notes and "goal kick" not in r.notes:
            cur["goal"] = True
        if r.outcome == "out":
            cur["ended"] = "out"; how_next = "restart"
        elif r.outcome == "intercepted":
            cur["ended"] = "intercepted"; how_next, gained_at = "interception", r.touch
        elif r.outcome == "incomplete":
            cur["ended"] = "loose ball"; how_next = "loose ball"
    if cur is not None:
        cur["end"] = int(e.kick.iloc[-1] + FPS); poss.append(cur)
    return pd.DataFrame(poss)


def ball_path(e: pd.DataFrame, D: dict, n_frames: int) -> pd.DataFrame:
    """Where the ball is (m from each team's own goal) between pass events: the passer at the kick, the
    receiver at the touch, held with the receiver until the next kick; linear in between."""
    pts = []
    for r in e.itertuples():
        if not np.isnan(r.passer_x): pts.append((r.kick, r.passer_x, r.half))
        if r.touch is not None and not np.isnan(r.receiver_x): pts.append((int(r.touch), r.receiver_x, r.half))
    pts = sorted(pts)
    f = np.arange(n_frames)
    x = np.interp(f, [p[0] for p in pts], [p[1] for p in pts])
    return pd.DataFrame({"frame": f, "x": x})


def switch_time(own: np.ndarray, start: int, stop: int, team: str, hold: int = 10) -> int:
    """First frame in [start, stop) from which the new team is nearest the ball in >= hold of the next 15
    frames; else one second after start (never after stop)."""
    want = 0 if team == "A" else 1
    w = own[start:stop] == want
    if len(w) >= 15:
        c = np.convolve(w.astype(int), np.ones(15, int), "valid")
        k = np.nonzero(c >= hold)[0]
        if len(k):
            return int(start + k[0])
    return int(min(stop, start + FPS))


def timeline(e: pd.DataFrame, own: np.ndarray, n_frames: int):
    """Per frame: team with the ball ('A' / 'B' / '' unknown) and dead-ball frames; plus, per pass index,
    the frame its team got the ball when that came by a tackle, duel or loose ball."""
    team = np.full(n_frames, "", dtype=object); dead = np.zeros(n_frames, bool); gained = {}
    for i in range(len(e)):
        r = e.iloc[i]; nxt = e.iloc[i + 1] if i + 1 < len(e) else None
        k = int(r.kick); end = int(nxt.kick) if nxt is not None else min(n_frames, k + int(5 * FPS))
        touch = int(r.touch) if r.touch is not None and not pd.isna(r.touch) else k + int(FPS)
        other = "B" if r.team == "A" else "A"
        team[k:min(touch, end)] = r.team
        if r.outcome == "out":
            dead[touch:end] = True
            continue
        holder = other if r.outcome == "intercepted" else (r.receiver_team if r.receiver_team in ("A", "B") else r.team)
        if r.outcome == "incomplete":
            holder = None
        if nxt is not None and nxt.team != (holder or "") and nxt.half == r.half and not is_set_piece(nxt.notes):
            # the next pass is by the other team: when did it win the ball?
            sw = switch_time(own, touch, end, nxt.team)
            if holder: team[touch:sw] = holder
            team[sw:end] = nxt.team; gained[i + 1] = sw
        elif nxt is not None and is_set_piece(nxt.notes):
            team[touch:min(end, touch + int(2 * FPS))] = holder or ""
            dead[min(end, touch + int(2 * FPS)):end] = True
        else:
            team[touch:end] = holder or ""
    return team, dead, gained


def phases(e, own, D, n_frames, half2):
    """Per frame: the team with the ball, its phase by third, and the other team's (Momentum labels)."""
    bx = ball_path(e, D, n_frames).x.values
    team, dead, _ = timeline(e, own, n_frames)
    A, B = np.full(n_frames, "", dtype=object), np.full(n_frames, "", dtype=object)
    for f in range(n_frames):
        if dead[f] or not team[f]:
            A[f] = B[f] = "Dead ball" if dead[f] else ""
            continue
        h = 2 if f >= half2 else 1
        u = fwd(bx[f], h, team[f], D); th = third(u)
        ph = PHASES[th] if not np.isnan(th) else ""
        other = "Transition" if ph == "Transition" else ("Retreat" if ph else "")
        A[f], B[f] = (ph, other) if team[f] == "A" else (other, ph)
    return pd.DataFrame({"frame": np.arange(n_frames), "possession": team, "A": A, "B": B, "ball_x": bx})


def counters(e, P, D, pos: Positions) -> pd.DataFrame:
    """Score every open-play regain as a counter-attack."""
    rows = []
    for p in P.itertuples():
        if p.kind not in ("interception", "tackle / duel", "loose ball"):
            continue
        h, T = p.half, p.team
        ev = e.loc[p.events]
        g0 = p.gained
        first = ev.iloc[0]
        # where the ball was won: the interception touch (receiver of the lost pass) or the first kick
        u0 = fwd(first.passer_x, h, T, D)
        if p.kind == "interception":
            prev = e[(e.touch == g0)]
            if len(prev) and not np.isnan(prev.iloc[0].receiver_x):
                u0 = fwd(prev.iloc[0].receiver_x, h, T, D)
        t_first = (first.kick - g0) / FPS
        u_kick = fwd(first.passer_x, h, T, D); u_recv = fwd(first.receiver_x, h, T, D)
        gain = u_recv - u_kick if first.outcome == "complete" or first.receiver_team == T else u_recv - u_kick
        dy = abs((first.receiver_y if not np.isnan(first.receiver_y) else np.nan) - first.passer_y)
        angle = float(np.degrees(np.arctan2(dy, gain))) if not np.isnan(gain) and not np.isnan(dy) else np.nan
        carry = u_kick - u0 if not np.isnan(u0) else np.nan
        fwd_pass = t_first <= FIRST_S and not np.isnan(gain) and gain >= FWD_M and (np.isnan(angle) or angle <= 45)
        fwd_carry = t_first <= FIRST_S + 2 and not np.isnan(carry) and carry >= CARRY_M
        quick = bool(fwd_pass or fwd_carry)
        # progression over the horizon: furthest point reached by the team's events in HORIZON_S
        lim = g0 + HORIZON_S * FPS
        us = [u0]
        for r in ev.itertuples():
            if r.kick > lim: break
            us.append(fwd(r.passer_x, h, T, D))
            if r.touch and r.touch <= lim and r.receiver_team == T: us.append(fwd(r.receiver_x, h, T, D))
        us = [u for u in us if not np.isnan(u)]
        best = max(us) if us else np.nan
        span = min(lim, p.end) - g0
        speed = (best - u0) / (span / FPS) if us and span > 0 and not np.isnan(u0) else np.nan
        direct = bool(not np.isnan(speed) and speed >= DIRECT_MPS)
        # shape at the regain: visible opponents goal-side of the ball, and teammates ahead of it
        pl = pos.players(int(g0))
        opp_id, own_id = (1, 0) if T == "A" else (0, 1)
        bu = u0
        o = pl[(pl.team_id == opp_id) & (pl.class_id == 2)]
        m = pl[(pl.team_id == own_id) & (pl.class_id == 2)]
        ou = np.array([fwd(x, h, T, D) for x in o.sx]); mu = np.array([fwd(x, h, T, D) for x in m.sx])
        goal_side = int((ou > bu).sum()) if not np.isnan(bu) else None
        ahead = int((mu > bu + 2).sum()) if not np.isnan(bu) else None
        exposed = bool(goal_side is not None and len(o) >= 6 and goal_side <= EXPOSED_N)
        label = "counter" if quick and (direct or exposed) else ("possible" if quick or direct else "none")
        # outcome: furthest third over the whole possession (capped at the horizon for the counter)
        reach = third(best) if not np.isnan(best) else np.nan
        end_third = ["own third", "middle third", "final third"][int(reach)] if not np.isnan(reach) else "unknown"
        rows.append({"team": T, "half": h, "segment": p.segment, "clock": first.clock, "regain_frame": g0, "kind": p.kind,
                     "won_at_m": round(u0, 1) if not np.isnan(u0) else None,
                     "won_in": ["own third", "middle third", "final third"][third(u0)] if not np.isnan(u0) else "unknown",
                     "first_pass_s": round(t_first, 1), "first_pass_gain_m": None if np.isnan(gain) else round(gain, 1),
                     "first_pass_angle": None if np.isnan(angle) else round(angle), "carry_m": None if np.isnan(carry) else round(carry, 1),
                     "progress_m": None if np.isnan(best) or np.isnan(u0) else round(best - u0, 1),
                     "speed_mps": None if np.isnan(speed) else round(speed, 2),
                     "opp_visible": len(o), "opp_goal_side": goal_side, "mates_ahead": ahead,
                     "quick": quick, "direct": direct, "exposed": exposed, "label": label,
                     "reached": end_third, "passes": len(ev), "ended": p.ended, "goal": bool(p.goal),
                     "first_kick_frame": int(first.kick)})
    return pd.DataFrame(rows)


def tempo(e, P, D):
    """Per possession: passes, seconds, seconds per pass, forward progress and its speed."""
    rows = []
    for p in P.itertuples():
        ev = e.loc[p.events]; h, T = p.half, p.team
        us = [fwd(x, h, T, D) for x in list(ev.passer_x) + list(ev[ev.receiver_team == T].receiver_x)]
        us = [u for u in us if not np.isnan(u)]
        dur = (ev.kick.iloc[-1] - ev.kick.iloc[0]) / FPS
        u_start = fwd(ev.passer_x.iloc[0], h, T, D)
        rows.append({"team": T, "half": h, "segment": p.segment, "clock": p.clock, "start_kind": p.kind,
                     "passes": len(ev), "seconds": round(dur, 1),
                     "sec_per_pass": round(dur / (len(ev) - 1), 2) if len(ev) > 1 else None,
                     "start_m": None if np.isnan(u_start) else round(u_start, 1),
                     "furthest_m": round(max(us), 1) if us else None,
                     "progress_m": round(max(us) - u_start, 1) if us and not np.isnan(u_start) else None,
                     "progress_mps": round((max(us) - u_start) / dur, 2) if us and dur > 1 and not np.isnan(u_start) else None,
                     "ended": p.ended, "goal": bool(p.goal), "start_frame": int(p.gained), "end_frame": int(p.end)})
    return pd.DataFrame(rows)


def team_speed(t: pd.DataFrame, n_frames: int, step: int = 30) -> pd.DataFrame:
    """Mean running speed (m/s) of each team's visible outfield players per second of play, from smoothed
    tracks; jumps (calibration glitches, > 9 m/s) are left out."""
    t = t[t.class_id == 2].sort_values(["raw_tracker_id", "frame"])
    g = t.groupby("raw_tracker_id")
    d = np.hypot(g.sx.diff(step), g.sy.diff(step)); df = g.frame.diff(step)
    v = d / (df / FPS)
    ok = (df == step) & (v < 9)
    s = pd.DataFrame({"sec": (t.frame // FPS).astype(int), "team": t.team_id, "v": v})[ok]
    return s[s.team.isin([0, 1])].groupby(["sec", "team"]).v.mean().unstack()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", required=True); ap.add_argument("--reviews", required=True)
    ap.add_argument("--pass-pieces"); ap.add_argument("--tracks", required=True); ap.add_argument("--index", required=True)
    ap.add_argument("--moments"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = json.load(open(a.index))
    segs = idx["segments"]; n_frames = segs[-1]["seg_start"] + segs[-1]["frames"]
    half2 = min(g["seg_start"] for g in segs if "_2H_" in g["name"])
    t = load_tracks(a.tracks)
    pos = Positions(t)
    own = ball_owner(a.tracks, n_frames)
    e = load_events(a.passes, a.reviews, a.pass_pieces, pos)
    D = attack_direction(e)
    _, _, gained = timeline(e, own, n_frames)
    P = possessions(e, gained)
    ph = phases(e, own, D, n_frames, half2)
    C = counters(e, P, D, pos)
    TP = tempo(e, P, D)
    SP = team_speed(t, n_frames)
    os.makedirs(a.out, exist_ok=True)
    e.to_csv(os.path.join(a.out, "events.csv"), index=False)
    C.to_csv(os.path.join(a.out, "counters.csv"), index=False)
    TP.to_csv(os.path.join(a.out, "possessions.csv"), index=False)
    ph.to_csv(os.path.join(a.out, "phases_per_frame.csv.gz"), index=False, compression="gzip")
    SP.to_csv(os.path.join(a.out, "team_speed_per_second.csv"))
    json.dump({str(k): v for k, v in D.items()}, open(os.path.join(a.out, "directions.json"), "w"))
    # agreement with a person's momentum marks, where there are any
    if a.moments and os.path.isdir(a.moments):
        M = sorted((json.load(open(f)) for f in glob.glob(os.path.join(a.moments, "*.json"))), key=lambda m: m["f"])
        OPP = {"Buildup": "Retreat", "Transition": "Transition", "Finishing": "Retreat", "Retreat": "Buildup"}
        if M:
            lab = np.full(n_frames, "", dtype=object)
            for i, m in enumerate(M):
                end = M[i + 1]["f"] if i + 1 < len(M) else next(g["seg_start"] + g["frames"] for g in segs if g["seg_start"] <= m["f"] < g["seg_start"] + g["frames"])
                lab[m["f"]:end] = m["phase"] if m["team"] == 0 else OPP[m["phase"]]
            k = (lab != "") & (ph.A.values != "") & (ph.A.values != "Dead ball")
            agree = float((lab[k] == ph.A.values[k]).mean()) if k.any() else None
            json.dump({"frames_compared": int(k.sum()), "agreement_team_A": agree,
                       "confusion": pd.crosstab(pd.Series(lab[k], name="marked"), pd.Series(ph.A.values[k], name="auto")).to_dict()},
                      open(os.path.join(a.out, "phase_agreement.json"), "w"), indent=1)
    print("directions", D, "| positions: passer", round(e.passer_x.notna().mean(), 3), "receiver", round(e.receiver_x.notna().mean(), 3))
    print("possessions", len(P), "| regains", len(C), C.label.value_counts().to_dict() if len(C) else {})


if __name__ == "__main__":
    main()
