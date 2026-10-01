"""Render pipeline output (per_frame_tracks.csv) over the video, optionally side by side with
another run and with the ground-truth ball, as a browser-friendly H.264 mp4.

  python -m tools.render_run --video clip.mp4 --run new=runs/new/per_frame_tracks.csv \
      --run old=runs/old/per_frame_tracks.csv --gt data/cvat/hilal_hazm_B/tracks.csv --out cmp.mp4

Boxes: players white, goalkeepers yellow, referees magenta, each labelled with its ID.
Ball: orange dot = the run's ball; green ring = the true ball (with --gt). The banner says
per frame whether the run has the ball right ("ball OK"), wrong or missing.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

COLORS = {1: (0, 220, 255), 2: (255, 255, 255), 3: (255, 0, 255)}  # BGR: GK yellow, player white, referee magenta
TEAM_COLORS = {0: (180, 105, 255), 1: (255, 200, 0)}  # BGR: team 0 pink, team 1 cyan (as in the tagging tool)
BALL = (0, 140, 255)
TRUE_BALL = (0, 220, 0)


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _centre(r):
    return ((r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2)


def draw(frame, rows: pd.DataFrame, gt_ball, label: str, scale: float, tol_k: float = 1.5, min_px: float = 8.0,
         id_col: str = "display_track_id", team_colors: bool = False):
    size = (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale / 2) * 2))
    img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    status, color = "no ball", (0, 0, 255)
    for r in rows.itertuples():
        c = int(r.class_id)
        if c in COLORS:
            col = COLORS[c]
            if team_colors and c == 2 and int(getattr(r, "team_id", -1)) in TEAM_COLORS:
                col = TEAM_COLORS[int(r.team_id)]
            p1, p2 = (int(r.x1 * scale), int(r.y1 * scale)), (int(r.x2 * scale), int(r.y2 * scale))
            cv2.rectangle(img, p1, p2, col, 1 if scale < 0.6 else 2)
            v = getattr(r, id_col, -1)
            tid = int(v) if pd.notna(v) else -1
            if tid >= 0:
                fs = 0.4 if scale < 0.6 else 0.55
                cv2.putText(img, str(tid), (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
    balls = rows[rows.class_id == 0]
    if gt_ball is not None:
        gx, gy = _centre(gt_ball)
        cv2.circle(img, (int(gx * scale), int(gy * scale)), 11, TRUE_BALL, 2, cv2.LINE_AA)
    if len(balls):
        b = balls.iloc[0]
        bx, by = _centre(b)
        cv2.circle(img, (int(bx * scale), int(by * scale)), 5, BALL, -1, cv2.LINE_AA)
        if gt_ball is not None:
            tol = max(min_px, tol_k * float(gt_ball.x2 - gt_ball.x1))
            ok = np.hypot(bx - gx, by - gy) <= tol
            status, color = ("ball OK", (0, 200, 0)) if ok else ("ball WRONG", (0, 0, 255))
        else:
            status, color = "ball", (255, 255, 255)
    elif gt_ball is None:
        status, color = "", (255, 255, 255)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if status:
        (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(img, status, (img.shape[1] - tw - 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


def render(video: str, runs: dict, out: str, gt: pd.DataFrame | None = None, width: int = 960,
           fps: float | None = None, frames=None, crf: int = 26, coord_size: tuple[int, int] | None = None,
           id_col: str = "display_track_id", team_colors: bool = False):
    """coord_size: (width, height) the csv coordinates refer to, when the video is a smaller proxy.
    frames: only these frame numbers (reading stops after the last one). id_col: the id drawn on boxes
    (e.g. joined_id); team_colors: players' boxes in their team colour."""
    cap = cv2.VideoCapture(video)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if coord_size:
        w0, h0 = coord_size
    scale = width / w0
    h = int(round(h0 * scale / 2) * 2)
    W = width * len(runs)
    by_run = {n: {f: g for f, g in pd.read_csv(p, low_memory=False).groupby("frame")} for n, p in runs.items()}
    gtb = gt[gt.class_id == 0].drop_duplicates("frame").set_index("frame") if gt is not None else None
    proc = subprocess.Popen([_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                            stdin=subprocess.PIPE)
    last = max(frames) if frames is not None else None
    frames = set(frames) if frames is not None else None
    f = 0
    empty = pd.DataFrame(columns=["class_id", "x1", "y1", "x2", "y2", "display_track_id"])
    while True:
        ok, frame = cap.read()
        if not ok or (last is not None and f > last):
            break
        if frames is None or f in frames:
            if coord_size and frame.shape[1] != w0:
                frame = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_LINEAR)
            g = gtb.loc[f] if gtb is not None and f in gtb.index else None
            panels = [draw(frame, by_run[n].get(f, empty), g, f"{n}   frame {f}", scale, id_col=id_col,
                           team_colors=team_colors) for n in runs]
            img = np.hstack(panels)[:h]
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        f += 1
    cap.release()
    proc.stdin.close()
    proc.wait()
    return out


def stills(video: str, runs: dict, frames: list[int], out: str, gt: pd.DataFrame | None = None, width: int = 960):
    """Selected frames, runs side by side, stacked vertically, as one JPEG."""
    cap = cv2.VideoCapture(video)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = width / w0
    by_run = {n: {f: g for f, g in pd.read_csv(p, low_memory=False).groupby("frame")} for n, p in runs.items()}
    gtb = gt[gt.class_id == 0].drop_duplicates("frame").set_index("frame") if gt is not None else None
    empty = pd.DataFrame(columns=["class_id", "x1", "y1", "x2", "y2", "display_track_id"])
    rows, f, want = [], 0, set(frames)
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if f in want:
            g = gtb.loc[f] if gtb is not None and f in gtb.index else None
            rows.append(np.hstack([draw(frame, by_run[n].get(f, empty), g, f"{n}   frame {f}", scale) for n in runs]))
            want.discard(f)
        f += 1
    cap.release()
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--run", action="append", required=True, help="label=per_frame_tracks.csv (repeatable, side by side)")
    p.add_argument("--gt", default=None, help="ground truth tracks.csv (shows the true ball)")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=960, help="width of each panel")
    p.add_argument("--coord-size", default=None, help="WxH of the csv coordinates if the video is a proxy, e.g. 1920x1080")
    a = p.parse_args(argv)
    runs = dict(s.split("=", 1) for s in a.run)
    gt = pd.read_csv(a.gt) if a.gt else None
    cs = tuple(int(v) for v in a.coord_size.split("x")) if a.coord_size else None
    print(render(a.video, runs, a.out, gt, a.width, coord_size=cs))


if __name__ == "__main__":
    main()
