"""Temporal smoothing of pitch positions, as a post-pass over a recorded run.

  python -m tools.pitch_smooth --run runs/SNGS-021              # rewrites per_frame_tracks.csv
  python -m tools.pitch_smooth --run runs/SNGS-021 --out smoothed.csv

Per-frame homographies jitter (keypoint noise), which made frame-to-frame player speeds
3-13x their real value (SoccerNet ground truth: 1.0-1.2x). This pass:
  1. recomputes each frame's homography from the run's recorded keypoints with the
     pipeline's own HomographyEstimator (so it matches what the run used),
  2. smooths the camera: pushes a fixed grid of image points through each homography,
     smooths their pitch positions over time (short median against spikes, then a local
     straight-line fit over `window` frames) and refits the homography from the smoothed
     grid. Windows never span a camera cut
     (a lasting jump of the view, detected from the median grid position over the
     frames before vs after) or a stretch without homography,
  3. re-projects every box's bottom-centre with the smoothed homography,
  4. lightly smooths each tracked person's pitch path (track_window frames) to remove
     the remaining foot-point wobble.
The raw positions are kept as x_m_raw / y_m_raw. Needs the run's model-output cache
(main.py --record-cache). On 5 SoccerNet clips: jitter ratio 3.2-12.8 -> 1.0-1.5, position
error unchanged or slightly better.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PERSON_CLASSES = (1, 2, 3)


def per_frame_homographies(cache_dir: str, frames: int, settings=None) -> dict[int, np.ndarray | None]:
    """The homography main.py used on each frame, recomputed from the recorded keypoints."""
    from sports.configs.soccer import SoccerPitchConfiguration

    from geometry.homography import HomographyEstimator
    from vision.det_cache import DetCacheReader

    if settings is None:
        from core.config import load_settings
        settings = load_settings()
    cache = DetCacheReader(cache_dir)
    est = HomographyEstimator(
        config=SoccerPitchConfiguration(), kp_conf=settings.KP_CONF,
        min_kp_spread_px=settings.H_MIN_KP_SPREAD_PX, max_hold_frames=settings.H_MAX_HOLD_FRAMES,
    )
    start = int(cache.meta.get("start_frame", 0))
    out = {}
    for f in range(start, start + frames):
        out[f] = est.estimate(cache.kp(f)).H if f in cache.kp_frames else None
    return out


def _grid(width: int, height: int) -> np.ndarray:
    # the lower part of a broadcast frame is pitch; points near the horizon are avoided
    xs = np.linspace(0.08 * width, 0.92 * width, 6)
    ys = np.linspace(0.35 * height, 0.94 * height, 4)
    return np.array([[x, y] for x in xs for y in ys], np.float64)


def find_cuts(frames: list[int], P: np.ndarray, k: int = 5, cut_cm: float = 1000.0) -> list[int]:
    """Indices where a new shot starts: a gap in frames, or a lasting jump of the view."""
    cuts = [i for i in range(1, len(frames)) if frames[i] - frames[i - 1] > 1]
    step = np.r_[0.0, np.median(np.linalg.norm(np.diff(P, axis=0), axis=2), axis=1)]  # per-frame view change
    last = -10
    for i in range(k, len(frames) - k + 1):
        if i - last <= 2 * k:
            continue
        before = np.median(P[i - k:i], axis=0)
        after = np.median(P[i:i + k], axis=0)
        if np.median(np.linalg.norm(after - before, axis=1)) > cut_cm:
            # the windows flag a lasting change a few frames early; the cut itself is
            # the largest single-frame change around here
            lo, hi = max(1, i - k), min(len(frames), i + k)
            j = lo + int(np.argmax(step[lo:hi]))
            cuts.append(j)
            last = j
    return sorted(set(cuts))


def _smooth_series(X: np.ndarray, window: int) -> np.ndarray:
    """Short rolling median (removes one-off spikes), then a local straight-line fit
    (Savitzky-Golay, order 1): smooths like a moving average but follows a steadily
    panning camera without lagging, including at the edges of a shot."""
    from scipy.signal import savgol_filter

    X = pd.DataFrame(X).rolling(5, center=True, min_periods=1).median().to_numpy()
    n = len(X)
    w = min(window, n if n % 2 else n - 1)
    if w < 5:
        return X
    return savgol_filter(X, w, polyorder=1, axis=0, mode="interp")


def smooth_homographies(Hs: dict[int, np.ndarray | None], width: int, height: int,
                        window: int = 25, cut_cm: float = 1000.0) -> tuple[dict, int]:
    grid = _grid(width, height)
    frames = sorted(f for f, H in Hs.items() if H is not None)
    if len(frames) < 2:
        return dict(Hs), 0
    P = np.stack([cv2.perspectiveTransform(grid.reshape(-1, 1, 2), Hs[f]).reshape(-1, 2) for f in frames])
    cuts = find_cuts(frames, P, cut_cm=cut_cm)
    out = dict(Hs)
    bounds = [0] + cuts + [len(frames)]
    for a, b in zip(bounds, bounds[1:]):
        seg = frames[a:b]
        if len(seg) < 2:
            continue
        sm = _smooth_series(P[a:b].reshape(len(seg), -1), window)
        for i, f in enumerate(seg):
            H, _ = cv2.findHomography(grid.astype(np.float32), sm[i].reshape(-1, 2).astype(np.float32))
            if H is not None and np.isfinite(H).all():
                out[f] = H / H[2, 2]
    return out, len(cuts)


def reproject(tracks: pd.DataFrame, Hs: dict[int, np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    feet = np.c_[((tracks.x1 + tracks.x2) / 2).to_numpy(np.float64), tracks.y2.to_numpy(np.float64)]
    xm = np.full(len(tracks), np.nan)
    ym = np.full(len(tracks), np.nan)
    for f, idx in tracks.groupby("frame").indices.items():
        H = Hs.get(int(f))
        if H is None:
            continue
        q = cv2.perspectiveTransform(feet[idx].reshape(-1, 1, 2), H).reshape(-1, 2)
        xm[idx], ym[idx] = q[:, 0], q[:, 1]
    return xm, ym


def smooth_tracks(tracks: pd.DataFrame, window: int = 5, id_col: str = "track_id") -> pd.DataFrame:
    """Smooth each person's path over consecutive frames: a 3-frame median against spikes,
    then a local straight-line fit, so a running player is not dragged back at the start
    or end of his track (a plain moving average lags there)."""
    from scipy.signal import savgol_filter

    out = tracks.copy()
    if window <= 1:
        return out
    people = out[out.class_id.isin(PERSON_CLASSES) & (out[id_col] >= 0)]
    for (cls, tid), g in people.groupby(["class_id", id_col]):
        g = g.drop_duplicates("frame")
        keep = g[["x_m", "y_m"]].notna().all(axis=1).to_numpy()
        if keep.sum() < 3:
            continue
        s = g.set_index("frame")[["x_m", "y_m"]].reindex(range(int(g.frame.min()), int(g.frame.max()) + 1))
        s = s.interpolate(limit_area="inside").rolling(3, center=True, min_periods=1).median()
        vals = s.to_numpy(copy=True)
        ok = np.isfinite(vals).all(axis=1)
        n = int(ok.sum())
        w = min(window, n if n % 2 else n - 1)
        if w >= 3:
            vals[ok] = savgol_filter(vals[ok], w, polyorder=1, axis=0, mode="interp")
        sm = pd.DataFrame(vals, index=s.index, columns=["x_m", "y_m"]).loc[g.frame.to_numpy()]
        out.loc[g.index[keep], "x_m"] = sm["x_m"].to_numpy()[keep]
        out.loc[g.index[keep], "y_m"] = sm["y_m"].to_numpy()[keep]
    return out


def smooth_run(tracks: pd.DataFrame, cache_dir: str, window: int = 25, track_window: int = 5,
               cut_cm: float = 1000.0, settings=None) -> tuple[pd.DataFrame, dict]:
    import json

    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    Hs = per_frame_homographies(cache_dir, int(meta["frames"]), settings)
    Hs_s, n_cuts = smooth_homographies(Hs, int(meta["width"]), int(meta["height"]), window, cut_cm)
    out = tracks.copy()
    if "x_m_raw" not in out.columns:
        out["x_m_raw"], out["y_m_raw"] = out["x_m"], out["y_m"]
    out["x_m"], out["y_m"] = reproject(out, Hs_s)
    out = smooth_tracks(out, track_window)
    stats = {"frames_with_homography": sum(H is not None for H in Hs.values()), "camera_cuts": n_cuts,
             "window": window, "track_window": track_window}
    return out, stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder with per_frame_tracks.csv and cache/")
    p.add_argument("--out", default=None, help="default: overwrite the run's per_frame_tracks.csv")
    p.add_argument("--window", type=int, default=25, help="camera smoothing window, frames")
    p.add_argument("--track-window", type=int, default=5, help="per-person smoothing window, frames")
    a = p.parse_args(argv)
    run = Path(a.run)
    tracks = pd.read_csv(run / "per_frame_tracks.csv", low_memory=False)
    out, stats = smooth_run(tracks, str(run / "cache"), a.window, a.track_window)
    dst = Path(a.out) if a.out else run / "per_frame_tracks.csv"
    out.to_csv(dst, index=False)
    print(f"{dst}: {stats}")


if __name__ == "__main__":
    main()
