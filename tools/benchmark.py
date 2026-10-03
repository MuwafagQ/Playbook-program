"""Benchmark a pipeline run from its per_frame_tracks.csv.

Subcommands:
  run              proxy metrics, no ground truth -> benchmark.json + benchmark.md
                   (+ suspects.csv, spotcheck.csv)
  score-gt         real tracking accuracy (IDF1, ID switches, MOTA) against a
                   hand-verified CSV such as per_frame_tracks_half*_unified.csv
  score-spotcheck  turn a human-filled spotcheck.csv into a measured ID-switch rate
  compare          side-by-side of two benchmark/score json files (baseline vs candidate)

Works on raw pipeline output and on the hand-cleaned *_unified.csv files, so the
gap between "unattended" and "after manual cleanup" can be measured the same way.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BALL, GK, PLAYER, REF = 0, 1, 2, 3
PEOPLE = (GK, PLAYER, REF)
ROLE = {GK: "goalkeeper", PLAYER: "player", REF: "referee"}


def _pct(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else float("nan")


def _r(x, nd: int = 4):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return _r(obj)


def load_tracks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for c in ("frame", "class_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["frame", "class_id"])
    df["frame"] = df["frame"].astype(int)
    df["class_id"] = df["class_id"].astype(int)
    for c in ("x1", "y1", "x2", "y2", "x_m", "y_m", "conf"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------

def detection_metrics(df: pd.DataFrame, all_frames: np.ndarray) -> dict:
    out = {}
    for cid in PEOPLE:
        per_frame = df[df.class_id == cid].groupby("frame").size().reindex(all_frames, fill_value=0)
        out[f"{ROLE[cid]}s_per_frame"] = {
            "mean": per_frame.mean(),
            "median": per_frame.median(),
            "p10": _pct(per_frame, 0.10),
            "p90": _pct(per_frame, 0.90),
        }
        if cid == PLAYER:
            out["frames_with_zero_players"] = int((per_frame == 0).sum())
    return out


def ball_metrics(df: pd.DataFrame, all_frames: np.ndarray) -> dict:
    ball = df[df.class_id == BALL]
    n = len(all_frames)
    per_frame = ball.groupby("frame").size()
    out = {
        "frames_with_ball": len(per_frame) / max(n, 1),
        "frames_with_multiple_ball_rows": int((per_frame > 1).sum()),
    }
    if "ball_interpolated" in ball.columns:
        interp = pd.to_numeric(ball["ball_interpolated"], errors="coerce").fillna(0).astype(int)
        real_frames = ball[interp == 0]["frame"].nunique()
        out["frames_with_real_ball_detection"] = real_frames / max(n, 1)
        out["frames_with_interpolated_ball"] = ball[interp == 1]["frame"].nunique() / max(n, 1)
    if "x_m" in ball.columns:
        pitch = ball.dropna(subset=["x_m", "y_m"])
        off = pitch[(pitch.x_m < 0) | (pitch.x_m > 12000) | (pitch.y_m < 0) | (pitch.y_m > 7000)]
        out["ball_rows_off_pitch"] = len(off) / max(len(pitch), 1)
    return out


def homography_metrics(df: pd.DataFrame, all_frames: np.ndarray) -> dict:
    out = {}
    first = df.sort_values("frame").groupby("frame").first()
    n = len(all_frames)
    if "homography_ok" in first.columns:
        ok = first["homography_ok"].astype(str).str.lower().isin(["true", "1", "1.0"])
        out["homography_ok_rate"] = ok.sum() / max(n, 1)
    if "homography_state" in first.columns:
        vc = first["homography_state"].astype(str).value_counts()
        out["homography_state_share"] = {k: v / max(n, 1) for k, v in vc.items()}
    if "kp_used" in first.columns:
        kp = pd.to_numeric(first["kp_used"], errors="coerce").dropna()
        out["kp_used_median"] = kp.median() if len(kp) else None
        out["frames_kp_used_lt4"] = int((kp < 4).sum())
    people = df[df.class_id.isin(PEOPLE)]
    if "x_m" in people.columns and len(people):
        out["people_rows_with_pitch_xy"] = people[["x_m", "y_m"]].notna().all(axis=1).mean()
    return out


def _segments(frames: np.ndarray, gap_tol: int) -> list[tuple[int, int]]:
    frames = np.sort(np.unique(frames))
    if len(frames) == 0:
        return []
    breaks = np.where(np.diff(frames) > gap_tol + 1)[0]
    starts = np.r_[frames[0], frames[breaks + 1]]
    ends = np.r_[frames[breaks], frames[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def identity_metrics(
    df: pd.DataFrame, id_col: str, fps: float, duration_min: float,
    gap_tol: int, expected: dict[int, int], long_s: float,
) -> dict:
    out = {}
    for cid in PEOPLE:
        sub = df[(df.class_id == cid) & (pd.to_numeric(df[id_col], errors="coerce") >= 0)]
        if sub.empty:
            continue
        ids = pd.to_numeric(sub[id_col], errors="coerce").astype(int)
        seg_len_s = []
        rows_in_long = 0
        for tid, g in sub.groupby(ids):
            segs = _segments(g["frame"].to_numpy(), gap_tol)
            for a, b in segs:
                seg_len_s.append((b - a + 1) / fps)
            life_s = sum((b - a + 1) for a, b in segs) / fps
            if life_s >= long_s:
                rows_in_long += len(g)
        seg = pd.Series(seg_len_s)
        n_ids = int(ids.nunique())
        exp = expected.get(cid, 0)
        out[ROLE[cid]] = {
            "unique_ids": n_ids,
            "expected_ids": exp,
            "ids_over_expected": (n_ids / exp) if exp else None,
            "new_ids_per_minute": n_ids / max(duration_min, 1e-9),
            "segments": int(len(seg)),
            "segment_seconds_mean": seg.mean(),
            "segment_seconds_median": seg.median(),
            "segment_seconds_p90": _pct(seg, 0.90),
            f"row_share_in_ids_lasting_{int(long_s)}s": rows_in_long / max(len(sub), 1),
        }
    return out


def duplicate_metrics(df: pd.DataFrame, id_col: str) -> dict:
    sub = df[df.class_id.isin(PEOPLE) & (pd.to_numeric(df[id_col], errors="coerce") >= 0)]
    dup = sub.groupby(["frame", "class_id", id_col]).size()
    return {"duplicate_id_rows_in_same_frame": int((dup > 1).sum())}


def jump_events(
    df: pd.DataFrame, id_col: str, fps: float,
    max_speed_mps: float, max_body_heights_per_frame: float, max_gap: int,
) -> pd.DataFrame:
    """Frame-to-frame image jumps of the same id that no human can make.

    The bottom-centre moves > N body-heights/frame in the image. Independent of the
    homography, so it points at an ID swap (or a bad box). Pitch-space motion is not
    used here: per-frame homography jitter swamps it (see pitch_jitter_metrics).
    """
    sub = df[df.class_id.isin((PLAYER, GK)) & (pd.to_numeric(df[id_col], errors="coerce") >= 0)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_id"] = pd.to_numeric(sub[id_col], errors="coerce").astype(int)
    sub = sub.sort_values(["class_id", "_id", "frame"])
    sub["cx"] = 0.5 * (sub.x1 + sub.x2)
    sub["by"] = sub.y2
    sub["bh"] = (sub.y2 - sub.y1).clip(lower=1.0)
    g = sub.groupby(["class_id", "_id"])
    sub["dframe"] = g["frame"].diff()
    sub["dpx"] = np.hypot(g["cx"].diff(), g["by"].diff())
    sub["prev_bh"] = g["bh"].shift()
    sub = sub[(sub.dframe >= 1) & (sub.dframe <= max_gap)].copy()
    sub["body_heights_per_frame"] = sub.dpx / (0.5 * (sub.bh + sub.prev_bh)) / sub.dframe
    if "x_m" in sub.columns:
        dm = np.hypot(g["x_m"].diff(), g["y_m"].diff()).reindex(sub.index) / 100.0
        sub["speed_mps"] = dm / (sub.dframe / fps)
    else:
        sub["speed_mps"] = np.nan
    ev = sub[sub.body_heights_per_frame > max_body_heights_per_frame].assign(kind="image_jump")
    return ev[["frame", "class_id", "_id", "kind", "dframe", "body_heights_per_frame", "speed_mps"]].rename(
        columns={"_id": id_col}
    ).sort_values(["kind", "frame"])


def pitch_jitter_metrics(df: pd.DataFrame, id_col: str, fps: float, max_speed_mps: float) -> dict:
    """How noisy the pitch coordinates are.

    Real players barely change speed within a second, so for a clean projection the
    median frame-to-frame speed matches the median speed over one second. Jitter
    inflates the first but not the second; jitter_ratio = frame-to-frame / 1-second.
    """
    if "x_m" not in df.columns:
        return {}
    sub = df[(df.class_id == PLAYER) & (pd.to_numeric(df[id_col], errors="coerce") >= 0)]
    sub = sub.dropna(subset=["x_m", "y_m"])
    win = max(1, int(round(fps)))
    ff, net = [], []
    for _, g in sub.groupby(pd.to_numeric(sub[id_col], errors="coerce").astype(int)):
        g = g.drop_duplicates("frame").set_index("frame").reindex(range(g.frame.min(), g.frame.max() + 1))
        xy = g[["x_m", "y_m"]].to_numpy(float) / 100.0
        d1 = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
        ff.append(d1[np.isfinite(d1)])
        if len(xy) > win:
            dn = np.linalg.norm(xy[win:] - xy[:-win], axis=1) * fps / win
            net.append(dn[np.isfinite(dn)])
    ff = np.concatenate(ff) if ff else np.array([])
    net = np.concatenate(net) if net else np.array([])
    pitch = df[df.class_id.isin(PEOPLE)].dropna(subset=["x_m", "y_m"])
    off = (pitch.x_m < -500) | (pitch.x_m > 12500) | (pitch.y_m < -500) | (pitch.y_m > 7500)
    med_ff = float(np.median(ff)) if len(ff) else None
    med_net = float(np.median(net)) if len(net) else None
    return {
        "speed_frame_to_frame_median_mps": med_ff,
        "speed_1s_median_mps": med_net,
        "jitter_ratio": (med_ff / med_net) if med_ff and med_net else None,
        f"frame_pairs_over_{int(max_speed_mps)}mps_share": float((ff > max_speed_mps).mean()) if len(ff) else None,
        "people_rows_over_5m_off_pitch_share": float(off.mean()) if len(pitch) else None,
    }


def team_flip_metrics(df: pd.DataFrame, id_col: str) -> dict:
    if "team_id" not in df.columns:
        return {}
    sub = df[(df.class_id == PLAYER) & (pd.to_numeric(df[id_col], errors="coerce") >= 0)].copy()
    sub["team_id"] = pd.to_numeric(sub["team_id"], errors="coerce")
    sub = sub[sub.team_id >= 0].sort_values("frame")
    if sub.empty:
        return {"players_with_team_label": 0}
    flips = sub.groupby(id_col)["team_id"].apply(lambda s: int((s.diff().fillna(0) != 0).sum()))
    return {
        "players_with_team_label": int(len(flips)),
        "ids_with_team_flip": int((flips > 0).sum()),
        "team_flips_total": int(flips.sum()),
        "unknown_team_row_share": float(
            (pd.to_numeric(df.loc[df.class_id == PLAYER, "team_id"], errors="coerce") < 0).mean()
        ),
    }


def spotcheck_windows(first: int, last: int, fps: float, n: int, window_s: float, seed: int) -> pd.DataFrame:
    win = int(round(window_s * fps))
    span = last - first + 1
    if n <= 0 or span < win:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    # Stratified: one random window per equal slice, so the sample covers the whole run.
    edges = np.linspace(first, last - win + 1, n + 1).astype(int)
    rows = []
    for i in range(n):
        lo, hi = edges[i], max(edges[i], edges[i + 1] - 1)
        s = int(rng.integers(lo, hi + 1))
        rows.append({
            "window_id": i + 1,
            "start_frame": s,
            "end_frame": s + win - 1,
            "start_time": f"{int(s / fps // 60):02d}:{s / fps % 60:05.2f}",
            "players_checked": "",
            "id_switches_found": "",
            "notes": "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(args) -> dict:
    df = load_tracks(args.csv)
    kpi = {}
    if args.kpi and Path(args.kpi).exists():
        kpi = json.loads(Path(args.kpi).read_text())
    fps = float(args.fps or kpi.get("source_fps") or 25.0)
    first, last = int(df.frame.min()), int(df.frame.max())
    all_frames = np.arange(first, last + 1)
    duration_min = len(all_frames) / fps / 60.0
    expected = {PLAYER: args.expected_players, GK: 2, REF: 1}

    report = {
        "input": {"csv": args.csv, "id_col": args.id_col, "fps": fps,
                  "frame_first": first, "frame_last": last,
                  "duration_min": duration_min, "rows": len(df)},
        "detection": detection_metrics(df, all_frames),
        "ball": ball_metrics(df, all_frames),
        "homography": homography_metrics(df, all_frames),
        "identity": identity_metrics(df, args.id_col, fps, duration_min, args.gap_tol,
                                     expected, args.long_seconds),
        "duplicates": duplicate_metrics(df, args.id_col),
        "team": team_flip_metrics(df, args.id_col),
    }
    if "raw_tracker_id" in df.columns:
        report["identity_raw_tracker"] = identity_metrics(
            df, "raw_tracker_id", fps, duration_min, args.gap_tol, expected, args.long_seconds)

    ev = jump_events(df, args.id_col, fps, args.max_speed_mps,
                     args.max_body_heights_per_frame, args.max_gap)
    player_minutes = (
        df[df.class_id == PLAYER].groupby("frame").size().sum() / fps / 60.0
    )
    report["jumps"] = {
        "player_minutes_observed": player_minutes,
        "image_jumps": int(len(ev)),
        "image_jumps_per_player_minute": len(ev) / max(player_minutes, 1e-9),
        "thresholds": {"max_body_heights_per_frame": args.max_body_heights_per_frame},
    }
    report["pitch_jitter"] = pitch_jitter_metrics(df, args.id_col, fps, args.max_speed_mps)
    if kpi:
        report["pipeline_kpi"] = kpi

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(ev):
        ev.assign(time_s=ev.frame / fps).to_csv(out / "suspects.csv", index=False)
    sc = spotcheck_windows(first, last, fps, args.spotcheck, args.window_seconds, args.seed)
    if len(sc):
        sc.to_csv(out / "spotcheck.csv", index=False)
    report = _clean(report)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2))
    (out / "benchmark.md").write_text(render_md(report))
    return report


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_md(r: dict) -> str:
    lines = ["# Pipeline benchmark", ""]
    i = r["input"]
    lines += [f"Source: `{i['csv']}` · id column `{i['id_col']}` · frames {i['frame_first']}–{i['frame_last']}"
              f" · {i['duration_min']:.2f} min @ {i['fps']:.2f} fps", ""]

    def table(title: str, d: dict):
        lines.extend([f"## {title}", "", "| metric | value |", "|---|---|"])
        for k, v in d.items():
            if isinstance(v, dict):
                v = ", ".join(f"{kk}={_fmt(vv)}" for kk, vv in v.items())
            lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")

    table("Detection", r["detection"])
    table("Ball", r["ball"])
    table("Homography", r["homography"])
    for role, d in r["identity"].items():
        table(f"Identity — {role} (`{i['id_col']}`)", d)
    for role, d in r.get("identity_raw_tracker", {}).items():
        table(f"Identity — {role} (raw BoT-SORT id)", d)
    table("Impossible jumps", r["jumps"])
    if r.get("pitch_jitter"):
        table("Pitch-coordinate jitter", r["pitch_jitter"])
    table("Duplicates", r["duplicates"])
    if r.get("team"):
        table("Team labels", r["team"])
    if r.get("pipeline_kpi"):
        table("Pipeline KPI (from run)", r["pipeline_kpi"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spot-check scoring and comparison
# ---------------------------------------------------------------------------

def _poisson_ci(k: int, exposure: float) -> tuple[float, float]:
    try:
        from scipy.stats import chi2
        lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
        hi = chi2.ppf(0.975, 2 * k + 2) / 2
    except Exception:
        lo = max(0.0, k - 1.96 * math.sqrt(k))
        hi = k + 1.96 * math.sqrt(max(k, 1))
    return lo / exposure, hi / exposure


def score_spotcheck(path: str, fps: float) -> dict:
    sc = pd.read_csv(path)
    for c in ("players_checked", "id_switches_found"):
        sc[c] = pd.to_numeric(sc[c], errors="coerce")
    done = sc.dropna(subset=["players_checked", "id_switches_found"])
    if done.empty:
        raise SystemExit("No reviewed windows: fill players_checked and id_switches_found.")
    minutes = (done.end_frame - done.start_frame + 1) / fps / 60.0
    exposure = float((done.players_checked * minutes).sum())
    k = int(done.id_switches_found.sum())
    lo, hi = _poisson_ci(k, exposure)
    return _clean({
        "windows_reviewed": len(done),
        "player_minutes_reviewed": exposure,
        "id_switches_found": k,
        "id_switches_per_player_minute": k / max(exposure, 1e-9),
        "ci95_low": lo,
        "ci95_high": hi,
        "per_90_min_per_player": 90.0 * k / max(exposure, 1e-9),
    })


# ---------------------------------------------------------------------------
# Ground-truth scoring (CLEAR-MOT + IDF1)
# ---------------------------------------------------------------------------

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _assign_max(weights: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one assignment maximising total weight (Hungarian, greedy fallback)."""
    if weights.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-weights)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        order = np.dstack(np.unravel_index(np.argsort(-weights, axis=None), weights.shape))[0]
        used_r, used_c, out = set(), set(), []
        for r, c in order:
            if r not in used_r and c not in used_c:
                used_r.add(r)
                used_c.add(c)
                out.append((int(r), int(c)))
        return out


def _people(df: pd.DataFrame, id_col: str, classes: tuple[int, ...]) -> pd.DataFrame:
    ids = pd.to_numeric(df[id_col], errors="coerce")
    sub = df[df.class_id.isin(classes) & (ids >= 0)].copy()
    # Namespace ids by class: display ids restart per class (player 1 vs GK 1).
    sub["_key"] = sub.class_id.astype(str) + ":" + ids[sub.index].astype(int).astype(str)
    return sub.dropna(subset=["x1", "y1", "x2", "y2"])


def _frame_boxes(sub: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        int(f): (g[["x1", "y1", "x2", "y2"]].to_numpy(float), g["_key"].to_numpy())
        for f, g in sub.groupby("frame")
    }


def _switch_attribution(matches: list[tuple[int, str, str, int]]) -> dict:
    """Split identity changes between the base tracker and the ID stabiliser.

    matches: (frame, gt_key, pred_key, raw_tracker_id) for every matched box.
    """
    if not matches:
        return {}
    m = pd.DataFrame(matches, columns=["frame", "gt", "pid", "raw"]).sort_values(["gt", "frame"])
    g = m.groupby("gt")
    m["prev_raw"], m["prev_pid"], m["gap"] = g.raw.shift(), g.pid.shift(), g.frame.diff()
    ev = m.dropna(subset=["prev_pid"])
    raw_ch = ev.raw != ev.prev_raw
    pid_ch = ev.pid != ev.prev_pid
    failed = ev[raw_ch & pid_ch]
    bins = pd.cut(failed.gap, [0, 5, 30, 90, np.inf], labels=["<=5", "6-30", "31-90", ">90"])
    return {
        "tracker_id_changes": int(raw_ch.sum()),
        "stabilizer_bridged": int((raw_ch & ~pid_ch).sum()),
        "stabilizer_failed_to_bridge": int(len(failed)),
        "stabilizer_caused": int((~raw_ch & pid_ch).sum()),
        "failed_bridges_by_gap_frames": {str(k): int(v) for k, v in bins.value_counts().sort_index().items()},
    }


def _best_offset(pred: dict, gt: dict, search: int, iou_thr: float) -> dict[int, int]:
    frames = sorted(gt)[:: max(1, len(gt) // 300)]
    scores = {}
    for off in range(-search, search + 1):
        n = 0
        for f in frames:
            p = pred.get(f - off)
            if p is None:
                continue
            n += int((_iou_matrix(gt[f][0], p[0]) >= iou_thr).any(axis=1).sum())
        scores[off] = n
    return scores


def score_against_gt(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, pred_id_col: str = "display_track_id",
    gt_id_col: str = "display_track_id", classes: tuple[int, ...] = (PLAYER,),
    iou_thr: float = 0.5, frame_offset: int = 0, offset_search: int = 3,
    drop_gt_interp: bool = True, min_merge_frames: int = 10, fps: float = 25.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Score predicted tracks against hand-verified tracks.

    frame_offset: pred frame + offset = gt frame (0 when both use absolute frames).
    Detection-level numbers (FP/FN, MOTA) partly reflect the manual cleanup itself
    (junk rows removed, gaps filled); the identity numbers (IDF1, ID switches,
    merged ids) are the ones that measure ID assignment.
    """
    gt_df = gt_df.copy()
    n_interp = 0
    if drop_gt_interp and "notes" in gt_df.columns:
        fabricated = gt_df["notes"].astype(str).str.contains("idfix_interp", na=False)
        n_interp = int(fabricated.sum())
        gt_df = gt_df[~fabricated]
    gt = _people(gt_df, gt_id_col, classes)
    pred = _people(pred_df, pred_id_col, classes)
    has_raw = "raw_tracker_id" in pred.columns and pred_id_col != "raw_tracker_id"
    raw_by_frame = (
        {int(f): pd.to_numeric(g["raw_tracker_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
         for f, g in pred.groupby("frame")} if has_raw else {}
    )
    matches = []
    pred_fb = _frame_boxes(pred)
    gt_fb_all = _frame_boxes(gt)

    lo = max(min(gt_fb_all, default=0), min(pred_fb, default=0) + frame_offset)
    hi = min(max(gt_fb_all, default=-1), max(pred_fb, default=-1) + frame_offset)
    gt_fb = {f: v for f, v in gt_fb_all.items() if lo <= f <= hi}
    offset_scores = _best_offset(pred_fb, gt_fb, offset_search, iou_thr) if offset_search else {}
    best_off = max(offset_scores, key=offset_scores.get) if offset_scores else frame_offset

    tp = fp = fn = idsw = 0
    gt_total = pred_total = 0
    last_match: dict[str, str] = {}
    co = {}  # (gt_key, pred_key) -> frames with IoU >= thr
    gt_count: dict[str, int] = {}
    pred_count: dict[str, int] = {}
    matched_gt: dict[str, int] = {}
    switches = []
    h_ratio = []
    for f in range(lo, hi + 1):
        g_boxes, g_keys = gt_fb.get(f, (np.zeros((0, 4)), np.array([])))
        p_boxes, p_keys = pred_fb.get(f - frame_offset, (np.zeros((0, 4)), np.array([])))
        gt_total += len(g_keys)
        pred_total += len(p_keys)
        for k in g_keys:
            gt_count[k] = gt_count.get(k, 0) + 1
        for k in p_keys:
            pred_count[k] = pred_count.get(k, 0) + 1
        iou = _iou_matrix(g_boxes, p_boxes)
        for gi, pi in zip(*np.where(iou >= iou_thr)):
            key = (g_keys[gi], p_keys[pi])
            co[key] = co.get(key, 0) + 1
        pairs = [(r, c) for r, c in _assign_max(iou) if iou[r, c] >= iou_thr]
        tp += len(pairs)
        fn += len(g_keys) - len(pairs)
        fp += len(p_keys) - len(pairs)
        for r, c in pairs:
            gk, pk = g_keys[r], p_keys[c]
            matched_gt[gk] = matched_gt.get(gk, 0) + 1
            h_ratio.append((p_boxes[c, 3] - p_boxes[c, 1]) / max(g_boxes[r, 3] - g_boxes[r, 1], 1e-9))
            prev = last_match.get(gk)
            if prev is not None and prev != pk:
                idsw += 1
                switches.append({"frame": f, "gt_id": gk, "from_pred_id": prev, "to_pred_id": pk})
            last_match[gk] = pk
            if has_raw:
                matches.append((f, gk, pk, int(raw_by_frame[f - frame_offset][c])))

    # IDF1: best global one-to-one gt<->pred identity mapping.
    g_list = sorted(gt_count)
    p_list = sorted(pred_count)
    gi_ = {k: i for i, k in enumerate(g_list)}
    pi_ = {k: i for i, k in enumerate(p_list)}
    w = np.zeros((len(g_list), len(p_list)))
    for (gk, pk), n in co.items():
        w[gi_[gk], pi_[pk]] = n
    mapping = {g_list[r]: p_list[c] for r, c in _assign_max(w) if w[r, c] > 0}
    idtp = int(sum(w[gi_[g], pi_[p]] for g, p in mapping.items()))

    per_player = []
    for gk in g_list:
        row_w = w[gi_[gk]]
        mp = mapping.get(gk)
        per_player.append({
            "gt_id": gk,
            "gt_frames": gt_count[gk],
            "matched_frames": matched_gt.get(gk, 0),
            "frames_with_correct_id": int(row_w[pi_[mp]]) if mp else 0,
            "id_accuracy": (row_w[pi_[mp]] / gt_count[gk]) if mp else 0.0,
            "mapped_pred_id": mp,
            "distinct_pred_ids": int((row_w > 0).sum()),
            "id_switches": sum(1 for s in switches if s["gt_id"] == gk),
        })
    per_player_df = pd.DataFrame(per_player).sort_values("id_accuracy") if per_player else pd.DataFrame()

    # Pred ids that cover 2+ real players for a meaningful stretch = mid-track swaps.
    merged = []
    for pk in p_list:
        col = w[:, pi_[pk]]
        players = [(g_list[i], int(col[i])) for i in np.where(col >= min_merge_frames)[0]]
        if len(players) >= 2:
            merged.append({"pred_id": pk, "gt_players": players})

    minutes = (hi - lo + 1) / fps / 60.0
    res = {
        "frames_scored": hi - lo + 1,
        "frame_range": [lo, hi],
        "classes": list(classes),
        "iou_threshold": iou_thr,
        "gt_rows_dropped_as_fabricated": n_interp,
        "frame_offset_used": frame_offset,
        "frame_offset_best_fit": best_off,
        "pred_to_gt_box_height_ratio_median": float(np.median(h_ratio)) if h_ratio else None,
        "gt_players": len(g_list),
        "pred_ids": len(p_list),
        "IDF1": 2 * idtp / max(gt_total + pred_total, 1),
        "IDP": idtp / max(pred_total, 1),
        "IDR": idtp / max(gt_total, 1),
        "id_switches": idsw,
        "id_switches_per_player_minute": idsw / max(len(g_list) * minutes, 1e-9),
        "pred_ids_covering_2plus_players": len(merged),
        "MOTA": 1 - (fn + fp + idsw) / max(gt_total, 1),
        "detection_recall": tp / max(gt_total, 1),
        "detection_precision": tp / max(tp + fp, 1),
        "gt_boxes": gt_total, "pred_boxes": pred_total, "TP": tp, "FP": fp, "FN": fn,
        "merged_pred_ids": merged,
    }
    if has_raw:
        res["switch_attribution"] = _switch_attribution(matches)
    warnings = []
    if best_off != frame_offset and offset_scores.get(best_off, 0) > 1.2 * offset_scores.get(frame_offset, 0):
        warnings.append(f"frames look misaligned: best-fitting offset is {best_off}, not {frame_offset}")
    r_h = res["pred_to_gt_box_height_ratio_median"]
    if r_h is not None and not 0.9 <= r_h <= 1.1:
        warnings.append(f"box sizes differ (pred/gt height {r_h:.2f}): different video resolution?")
    if res["detection_recall"] < 0.3:
        warnings.append("very few boxes match: check the video, frame window and resolution")
    res["warnings"] = warnings
    return _clean(res), per_player_df, pd.DataFrame(switches)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


def compare(a_path: str, b_path: str) -> str:
    a = _flatten(json.loads(Path(a_path).read_text()))
    b = _flatten(json.loads(Path(b_path).read_text()))
    lines = [f"| metric | {Path(a_path).parent.name or 'A'} | {Path(b_path).parent.name or 'B'} | delta |",
             "|---|---|---|---|"]
    for k in sorted(set(a) | set(b)):
        if k.startswith(("input.", "pipeline_kpi.")) or ".thresholds." in k:
            continue
        va, vb = a.get(k), b.get(k)
        d = (vb - va) if (va is not None and vb is not None) else None
        lines.append(f"| {k} | {_fmt(va)} | {_fmt(vb)} | {_fmt(d)} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run", help="compute metrics for one run")
    r.add_argument("--csv", required=True)
    r.add_argument("--kpi", default=None, help="kpi_summary.json from the same run (optional)")
    r.add_argument("--out-dir", default="benchmark")
    r.add_argument("--fps", type=float, default=None, help="default: kpi source_fps, else 25")
    r.add_argument("--id-col", default="display_track_id")
    r.add_argument("--expected-players", type=int, default=20,
                   help="outfield players expected over the run (20 + substitutes)")
    r.add_argument("--gap-tol", type=int, default=5, help="frames an id may vanish without splitting a segment")
    r.add_argument("--long-seconds", type=float, default=60.0)
    r.add_argument("--max-speed-mps", type=float, default=12.0)
    r.add_argument("--max-body-heights-per-frame", type=float, default=1.0)
    r.add_argument("--max-gap", type=int, default=5, help="max frame gap to test for jumps")
    r.add_argument("--spotcheck", type=int, default=12, help="random windows for human review (0 = none)")
    r.add_argument("--window-seconds", type=float, default=20.0)
    r.add_argument("--seed", type=int, default=0)

    s = sp.add_parser("score-spotcheck", help="score a human-filled spotcheck.csv")
    s.add_argument("--spotcheck", required=True)
    s.add_argument("--fps", type=float, default=25.0)

    g = sp.add_parser("score-gt", help="IDF1 / ID switches / MOTA against a hand-verified CSV")
    g.add_argument("--pred", required=True, help="per_frame_tracks.csv from an unattended run")
    g.add_argument("--gt", required=True, help="hand-verified CSV, e.g. per_frame_tracks_half1_unified.csv")
    g.add_argument("--out-dir", default="benchmark_gt")
    g.add_argument("--pred-id-col", default="display_track_id")
    g.add_argument("--gt-id-col", default="display_track_id")
    g.add_argument("--classes", default="2", help="comma list: 2=players, 1=GK, 3=referee")
    g.add_argument("--iou", type=float, default=0.5)
    g.add_argument("--frame-offset", type=int, default=0, help="pred frame + offset = gt frame")
    g.add_argument("--offset-search", type=int, default=3)
    g.add_argument("--fps", type=float, default=25.0)
    g.add_argument("--keep-gt-interp", action="store_true",
                   help="keep GT rows fabricated by gap interpolation (notes=idfix_interp)")

    c = sp.add_parser("compare", help="compare two benchmark.json files")
    c.add_argument("baseline")
    c.add_argument("candidate")

    args = p.parse_args(argv)
    if args.cmd == "run":
        rep = build_report(args)
        print(render_md(rep))
        print(f"\nWrote {Path(args.out_dir) / 'benchmark.json'} and benchmark.md")
    elif args.cmd == "score-gt":
        res, per_player, switches = score_against_gt(
            load_tracks(args.pred), load_tracks(args.gt),
            pred_id_col=args.pred_id_col, gt_id_col=args.gt_id_col,
            classes=tuple(int(c) for c in args.classes.split(",")),
            iou_thr=args.iou, frame_offset=args.frame_offset,
            offset_search=args.offset_search, drop_gt_interp=not args.keep_gt_interp,
            fps=args.fps,
        )
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "score_gt.json").write_text(json.dumps(res, indent=2))
        per_player.to_csv(out / "gt_per_player.csv", index=False)
        switches.to_csv(out / "gt_switches.csv", index=False)
        summary = {k: v for k, v in res.items() if k != "merged_pred_ids"}
        print(json.dumps(summary, indent=2))
        if len(per_player):
            print("\nWorst-tracked players:")
            print(per_player.head(8).to_string(index=False))
        print(f"\nWrote {out}/score_gt.json, gt_per_player.csv, gt_switches.csv")
    elif args.cmd == "score-spotcheck":
        print(json.dumps(score_spotcheck(args.spotcheck, args.fps), indent=2))
    elif args.cmd == "compare":
        print(compare(args.baseline, args.candidate))


if __name__ == "__main__":
    main()
