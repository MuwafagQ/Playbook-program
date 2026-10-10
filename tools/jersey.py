"""Make shirt numbers readable and read them: best views, enhancement, multi-frame fusion, OCR voting.

Per track piece:
  1. best_views()   pick the frames where the player is biggest and sharpest (the cards so far were evenly
                    spaced and shrunk to 64x128 px, which throws the close-ups away)
  2. torso_crop()   the shirt area at native video resolution (back numbers sit between shoulders and waist)
  3. enhance()      contrast stretch (CLAHE on lightness) + gentle unsharp mask; no generative upscaling,
                    which can invent digits
  4. fuse()         align a few consecutive frames of the same player (ECC) and take their median: less
                    night noise and compression blur than any single frame
  5. vote()         per piece, the number read most confidently across its views (one bad frame does not
                    decide); team B pieces with known numbers measure the reader before it is trusted

  python: rows = best_views(piece_rows, k=6); crop = enhance(torso_crop(frame, box))
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def best_views(rows: pd.DataFrame, k: int = 6, min_gap: int = 15, min_height: int = 60) -> pd.DataFrame:
    """rows of one piece (segment_frame, x1..y2). The k tallest boxes, at least min_gap frames apart
    (different poses, so the back is likely shown in some), skipping boxes shorter than min_height."""
    r = rows.assign(h=rows.y2 - rows.y1)
    r = r[r.h >= min_height].sort_values("h", ascending=False)
    picked: list[int] = []
    out = []
    for row in r.itertuples():
        if all(abs(row.segment_frame - f) >= min_gap for f in picked):
            picked.append(row.segment_frame)
            out.append(row.Index)
            if len(out) == k:
                break
    return rows.loc[out]


def torso_crop(frame: np.ndarray, box, pad: float = 0.10) -> np.ndarray:
    """Shirt area: 12%..58% of the box height, 10%..90% of its width (plus padding), native pixels."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    X1 = int(max(0, x1 + (0.10 - pad) * w))
    X2 = int(min(frame.shape[1], x2 - (0.10 - pad) * w))
    Y1 = int(max(0, y1 + 0.12 * h))
    Y2 = int(min(frame.shape[0], y1 + 0.58 * h))
    if X2 <= X1 or Y2 <= Y1:
        return np.zeros((8, 8, 3), np.uint8)
    return frame[Y1:Y2, X1:X2].copy()


def sharpness(img: np.ndarray) -> float:
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def enhance(img: np.ndarray, scale: int = 3) -> np.ndarray:
    """Upsample (bicubic, no invented detail), stretch local contrast on lightness, unsharp mask."""
    import cv2

    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


def fuse(crops: list[np.ndarray]) -> np.ndarray:
    """Align crops of the same player in consecutive frames to the middle one (ECC, affine) and take the
    per-pixel median. Crops are resized to the middle crop's size first."""
    import cv2

    if not crops:
        return np.zeros((8, 8, 3), np.uint8)
    ref = crops[len(crops) // 2]
    h, w = ref.shape[:2]
    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    stack = []
    for c in crops:
        c = cv2.resize(c, (w, h), interpolation=cv2.INTER_CUBIC)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(ref_g, cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), warp,
                                           cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4),
                                           None, 3)
            c = cv2.warpAffine(c, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        except cv2.error:
            pass    # could not align (big pose change): use as is
        stack.append(c)
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def clean_number(text: str) -> str | None:
    """OCR text -> a shirt number (1-99) or None."""
    d = re.sub(r"\D", "", text or "")
    if not d or len(d) > 2 or d.startswith("0"):
        return None
    return d


def vote(reads: pd.DataFrame, min_conf: float = 0.5) -> pd.DataFrame:
    """reads: piece, number (str or None), conf. Per piece: the number with the largest summed confidence,
    its share of that piece's confident reads, and how many views read it."""
    r = reads[reads.number.notna() & (reads.conf >= min_conf)]
    if not len(r):
        return pd.DataFrame(columns=["piece", "number", "score", "share", "views"])
    g = r.groupby(["piece", "number"]).conf.agg(["sum", "size"]).reset_index()
    tot = g.groupby("piece")["sum"].transform("sum")
    g["share"] = g["sum"] / tot
    best = g.sort_values("sum", ascending=False).drop_duplicates("piece")
    return best.rename(columns={"sum": "score", "size": "views"})[["piece", "number", "score", "share", "views"]]
