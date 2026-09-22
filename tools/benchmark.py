"""Benchmark a pipeline run from its per_frame_tracks.csv (no ground truth needed).

Subcommands:
  run              compute metrics -> benchmark.json + benchmark.md (+ suspects.csv, spotcheck.csv)
  score-spotcheck  turn a human-filled spotcheck.csv into a measured ID-switch rate
  compare          side-by-side of two benchmark.json files (baseline vs candidate)

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
    """Frame-to-frame jumps of the same id that no human can make.

    image_jump: bottom-centre moves > N body-heights/frame in the image. Independent
                of the homography, so it points at an ID swap (or a bad box).
    pitch_jump: implied ground speed > max_speed_mps while the image motion is
                plausible, so it points at homography jitter rather than identity.
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
    img = sub.body_heights_per_frame > max_body_heights_per_frame
    pitch = (sub.speed_mps > max_speed_mps) & ~img
    sub["kind"] = np.where(img, "image_jump", np.where(pitch, "pitch_jump", ""))
    ev = sub[sub.kind != ""]
    return ev[["frame", "class_id", "_id", "kind", "dframe", "body_heights_per_frame", "speed_mps"]].rename(
        columns={"_id": id_col}
    ).sort_values(["kind", "frame"])


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
    counts = ev["kind"].value_counts().to_dict() if len(ev) else {}
    report["jumps"] = {
        "player_minutes_observed": player_minutes,
        "image_jumps": int(counts.get("image_jump", 0)),
        "pitch_jumps": int(counts.get("pitch_jump", 0)),
        "image_jumps_per_player_minute": counts.get("image_jump", 0) / max(player_minutes, 1e-9),
        "pitch_jumps_per_player_minute": counts.get("pitch_jump", 0) / max(player_minutes, 1e-9),
        "thresholds": {"max_speed_mps": args.max_speed_mps,
                       "max_body_heights_per_frame": args.max_body_heights_per_frame},
    }
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

    c = sp.add_parser("compare", help="compare two benchmark.json files")
    c.add_argument("baseline")
    c.add_argument("candidate")

    args = p.parse_args(argv)
    if args.cmd == "run":
        rep = build_report(args)
        print(render_md(rep))
        print(f"\nWrote {Path(args.out_dir) / 'benchmark.json'} and benchmark.md")
    elif args.cmd == "score-spotcheck":
        print(json.dumps(score_spotcheck(args.spotcheck, args.fps), indent=2))
    elif args.cmd == "compare":
        print(compare(args.baseline, args.candidate))


if __name__ == "__main__":
    main()
