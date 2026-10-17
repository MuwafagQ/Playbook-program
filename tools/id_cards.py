"""Photo cards for checking who is who, and full-frame ball candidates for re-solving the ball path.

Photo cards: every track piece (one unbroken tracker id, nearly always one person) that lasts at least
`min_frames` gets a strip of `per_piece` crops taken across its life. Strips are packed into one JPEG
sheet per segment (cards_<segment>.jpg) with an index (cards_<segment>.json) saying where each piece's
strip is, so the ID Review page can show them without the video.

Ball candidates: the ball model run on tiles covering the whole frame (not only around the tracker's
guess), every `every`-th frame, keeping the `top` most confident boxes. The pipeline's online tracker
can lock onto a wrong blob and drift; with candidates everywhere the whole path can be solved at once.

  plan = card_plan(tracks)                         # which frames to crop for which piece
  index = build_sheet(video, plan, 'cards_01.jpg')  # crops -> one JPEG, returns the index
  cands = ball_candidates(video, predict_fn)       # frame, x, y, w, h, conf
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CARD_W, CARD_H = 64, 128     # one crop on the sheet
STRIPS_PER_ROW = 4           # piece strips side by side on a sheet row


def card_plan(tracks: pd.DataFrame, min_frames: int = 30, per_piece: int = 4,
              people=(1, 2, 3), frame_col: str = "segment_frame") -> pd.DataFrame:
    """One row per crop: piece, frame (video frame of the segment), box, slot (0..per_piece-1).
    Crops are spread across the piece's life (10% .. 90%) so a card shows it more than once."""
    t = tracks[tracks.class_id.isin(people) & (tracks.raw_tracker_id >= 0)]
    out = []
    for piece, g in t.sort_values(frame_col).groupby("raw_tracker_id"):
        if len(g) < min_frames:
            continue
        pick = np.linspace(0.1, 0.9, per_piece) * (len(g) - 1)
        rows = g.iloc[np.round(pick).astype(int)]
        out.append(pd.DataFrame({"piece": int(piece), "frame": rows[frame_col].to_numpy(),
                                 "x1": rows.x1.to_numpy(), "y1": rows.y1.to_numpy(),
                                 "x2": rows.x2.to_numpy(), "y2": rows.y2.to_numpy(),
                                 "slot": np.arange(per_piece)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["piece", "frame", "x1", "y1", "x2", "y2", "slot"])


def sheet_layout(pieces: list[int], per_piece: int = 4) -> dict[int, tuple[int, int]]:
    """piece -> (x, y) of its strip's top-left corner on the sheet."""
    sw = per_piece * CARD_W
    return {p: ((i % STRIPS_PER_ROW) * sw, (i // STRIPS_PER_ROW) * CARD_H) for i, p in enumerate(pieces)}


def build_sheet(video: str, plan: pd.DataFrame, out_jpg: str, per_piece: int = 4, quality: int = 82) -> dict:
    """Crop the plan's boxes from the video and write one JPEG sheet. Returns
    {'sheet': file name, 'w': CARD_W, 'h': CARD_H, 'per_piece': n, 'pieces': {piece: [x, y, [frames]]}}."""
    import os

    import cv2

    from tools.reid_appearance import crops_from_video

    pieces = sorted(int(p) for p in plan.piece.unique())
    layout = sheet_layout(pieces, per_piece)
    rows_n = (len(pieces) + STRIPS_PER_ROW - 1) // STRIPS_PER_ROW
    sheet = np.full((max(rows_n, 1) * CARD_H, STRIPS_PER_ROW * per_piece * CARD_W, 3), 24, np.uint8)
    crops, rows = crops_from_video(video, plan)
    frames: dict[int, list[int]] = {p: [] for p in pieces}
    for c, r in zip(crops, rows.itertuples()):
        x, y = layout[int(r.piece)]
        x += int(r.slot) * CARD_W
        sheet[y:y + CARD_H, x:x + CARD_W] = cv2.resize(c, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        frames[int(r.piece)].append(int(r.frame))
    cv2.imwrite(out_jpg, sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return {"sheet": os.path.basename(out_jpg), "w": CARD_W, "h": CARD_H, "per_piece": per_piece,
            "pieces": {str(p): [layout[p][0], layout[p][1], sorted(frames[p])] for p in pieces}}


def ball_candidates(video: str, predict_fn, every: int = 2, conf: float = 0.05, top: int = 8,
                    tile: int = 320, preprocess: bool = True, progress=None) -> pd.DataFrame:
    """Ball-model boxes on tiles covering the whole frame, every `every`-th frame (video frames)."""
    import cv2

    from vision.ball_model import predict_tiles
    from vision.preprocess import enhance_frame

    cap = cv2.VideoCapture(video)
    out, fi = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if fi % every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            det = predict_tiles(predict_fn, enhance_frame(frame, enabled=preprocess), tile=tile, conf=conf)
            if len(det):
                k = np.argsort(-det.confidence)[:top]
                xy = det.xyxy[k]
                out.append(np.c_[np.full(len(k), fi), (xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2,
                                 xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1], det.confidence[k]])
            if progress and fi % 900 == 0:
                progress(fi)
        fi += 1
    cap.release()
    a = np.concatenate(out) if out else np.zeros((0, 6))
    df = pd.DataFrame(a, columns=["frame", "x", "y", "w", "h", "conf"])
    df["frame"] = df.frame.astype(int)
    return df.round({"x": 1, "y": 1, "w": 1, "h": 1, "conf": 3})


def dark_share(sheet: np.ndarray, entry, per_piece: int = 4, value_max: int = 70) -> float:
    """Share of dark pixels in the chest/shorts centre of a piece's crops (median over its crops):
    high for black kits (match_video_2: team A keeper and the referees), low otherwise."""
    import cv2

    x0, y0 = entry[0], entry[1]
    vals = []
    for k in range(per_piece):
        c = sheet[y0 + 25:y0 + 75, x0 + k * CARD_W + 18:x0 + k * CARD_W + 46]
        vals.append(float((cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2] < value_max).mean()))
    return float(np.median(vals))


def red_shares(sheet: np.ndarray, entry, per_piece: int = 4) -> list[float]:
    """Per crop of a piece: share of strongly red pixels in the chest/shorts centre (match_video_2:
    team A plays in red). Per crop, so a card that changes team between photos can be spotted."""
    import cv2

    x0, y0 = entry[0], entry[1]
    out = []
    for k in range(per_piece):
        c = cv2.cvtColor(sheet[y0 + 25:y0 + 75, x0 + k * CARD_W + 18:x0 + k * CARD_W + 46], cv2.COLOR_BGR2HSV)
        h, s, v = c[..., 0], c[..., 1], c[..., 2]
        out.append(float((((h <= 8) | (h >= 170)) & (s > 90) & (v > 60)).mean()))
    return out
