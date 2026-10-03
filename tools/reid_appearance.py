"""Football player appearance model: tells players apart from their image crops, so track pieces can
be joined across camera pans (notebooks/reid_appearance.ipynb trains and tests it).

Training data are the hand-unified HIL-HAZ windows (same id = same player); the backbone is
DINOv2-small (Apache-2.0), fine-tuned with identity classification + batch-hard triplet loss. A
piece's appearance is the mean of its crop embeddings (L2-normalised); two pieces look alike when
the cosine similarity of their means is high.

  crops_from_video(video, boxes, ...)   crops (N, H, W, 3) uint8 RGB + the rows they came from
  sample_rows(...)                      which rows to crop: every k-th frame / n per piece
  Embedder                              backbone + BN neck; .embed(crops) -> (N, D) normalised
  train(...)                            fine-tunes an Embedder on labelled crops
  piece_means(emb, piece_ids)           mean embedding per piece
  retrieval(emb, ids, times, ...)       rank-1 / mAP of tracklet chunks against other chunks
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

H, W = 196, 98  # crop size: DINOv2 patches are 14 px -> 14 x 7 patches
PAD = 0.08      # box padding (fraction of box size) before resizing


# ---------------------------------------------------------------- crops
def sample_rows(rows: pd.DataFrame, id_col: str, every: int = 1, per_id: int | None = None,
                seed: int = 0) -> pd.DataFrame:
    """Rows to crop: frames that are a multiple of `every`, then at most `per_id` evenly spaced rows
    per id (None: all)."""
    r = rows[rows.frame % every == 0]
    if per_id is None:
        return r
    out = []
    for _, g in r.sort_values("frame").groupby(id_col):
        if len(g) > per_id:
            g = g.iloc[np.linspace(0, len(g) - 1, per_id).round().astype(int)]
        out.append(g)
    return pd.concat(out) if out else r.iloc[:0]


def _crop(frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, int(x1 - PAD * bw)), min(w, int(math.ceil(x2 + PAD * bw)))
    y1, y2 = max(0, int(y1 - PAD * bh)), min(h, int(math.ceil(y2 + PAD * bh)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((H, W, 3), np.uint8)
    return cv2.resize(frame[y1:y2, x1:x2], (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]  # BGR -> RGB


def crops_from_video(video: str, rows: pd.DataFrame, frame_offset: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """Crop every row's box (x1..y2, `frame` + frame_offset = video frame). Reads the video once,
    in order. Returns crops and the rows (same order)."""
    import cv2

    rows = rows.sort_values("frame").reset_index(drop=True)
    out = np.zeros((len(rows), H, W, 3), np.uint8)
    by_frame = {f: g.index.to_numpy() for f, g in rows.groupby("frame")}
    wanted = sorted(by_frame)
    cap = cv2.VideoCapture(video)
    pos, k = 0, 0
    while k < len(wanted):
        target = wanted[k] + frame_offset
        while pos < target:
            if not cap.grab():
                break
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok:
            break
        for i in by_frame[wanted[k]]:
            r = rows.loc[i]
            out[i] = _crop(frame, r.x1, r.y1, r.x2, r.y2)
        k += 1
    cap.release()
    return out, rows


# ---------------------------------------------------------------- model
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(crops: np.ndarray):
    import torch

    x = torch.from_numpy(crops).float().div_(255).permute(0, 3, 1, 2)
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


class Embedder:
    """backbone (anything returning a (B, D) feature from a (B, 3, H, W) batch) + BatchNorm neck."""

    def __init__(self, backbone=None, dim: int = 384, device: str | None = None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = (backbone or dinov2_small()).to(self.device)
        self.neck = torch.nn.BatchNorm1d(dim).to(self.device)
        self.neck.bias.requires_grad_(False)
        self.dim = dim

    def features(self, x):
        return self.neck(self.backbone(x))

    def embed(self, crops: np.ndarray, batch: int = 128) -> np.ndarray:
        import torch

        self.backbone.eval(); self.neck.eval()
        out = []
        with torch.no_grad(), torch.autocast(self.device.split(":")[0], enabled=self.device.startswith("cuda")):
            for i in range(0, len(crops), batch):
                x = to_tensor(crops[i:i + batch]).to(self.device)
                f = self.features(x) + self.features(torch.flip(x, dims=[3]))  # + mirrored crop
                out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def state_dict(self):
        return {"backbone": self.backbone.state_dict(), "neck": self.neck.state_dict()}

    def load_state_dict(self, sd):
        self.backbone.load_state_dict(sd["backbone"]); self.neck.load_state_dict(sd["neck"])


def dinov2_small():
    """DINOv2-small (facebook/dinov2-small, Apache-2.0): CLS token + mean of patch tokens, 384-d."""
    import torch
    from transformers import AutoModel

    class _Dino(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = AutoModel.from_pretrained("facebook/dinov2-small")

        def forward(self, x):
            t = self.m(pixel_values=x).last_hidden_state
            return 0.5 * (t[:, 0] + t[:, 1:].mean(1))

    return _Dino()


def _batch_hard_triplet(f, y, margin: float = 0.3):
    import torch

    d = torch.cdist(f, f)
    same = y[:, None] == y[None, :]
    pos = (d * same.float()).max(1).values
    neg = (d + same.float() * 1e6).min(1).values
    return torch.relu(pos - neg + margin).mean()


def pk_batches(labels: np.ndarray, p: int, k: int, steps: int, seed: int = 0):
    """Index batches with p identities x k crops each (identities with fewer crops are sampled with
    replacement)."""
    rng = np.random.default_rng(seed)
    by_id = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    ids = np.array(list(by_id))
    for _ in range(steps):
        chosen = rng.choice(ids, size=min(p, len(ids)), replace=False)
        yield np.concatenate([rng.choice(by_id[c], size=k, replace=len(by_id[c]) < k) for c in chosen])


def _augment(x):
    """Light augmentation on a normalised batch: random flip, brightness/contrast, small shift."""
    import torch

    b = x.shape[0]
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    gain = 1 + 0.25 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
    bias = 0.25 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
    x = x * gain + bias
    dy, dx = np.random.randint(-8, 9), np.random.randint(-4, 5)
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def train(model: Embedder, crops: np.ndarray, labels: np.ndarray, steps: int = 1500, p: int = 16, k: int = 4,
          lr: float = 3e-5, log_every: int = 100, seed: int = 0) -> list[dict]:
    """Fine-tune on labelled crops (identity classification with label smoothing + batch-hard triplet)."""
    import torch

    torch.manual_seed(seed)
    classes = {c: i for i, c in enumerate(np.unique(labels))}
    y_all = np.array([classes[c] for c in labels])
    head = torch.nn.Linear(model.dim, len(classes), bias=False).to(model.device)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": lr},
                             {"params": list(model.neck.parameters()) + list(head.parameters()), "lr": lr * 30}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 30], total_steps=steps, pct_start=0.1)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = model.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    model.backbone.train(); model.neck.train()
    log, run = [], []
    for step, idx in enumerate(pk_batches(y_all, p, k, steps, seed), start=1):
        x = _augment(to_tensor(crops[idx]).to(model.device))
        y = torch.from_numpy(y_all[idx]).to(model.device)
        with torch.autocast(model.device.split(":")[0], enabled=use_amp):
            raw = model.backbone(x)
            f = model.neck(raw)
            loss = ce(head(f), y) + _batch_hard_triplet(raw.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        run.append(float(loss.detach()))
        if step % log_every == 0 or step == steps:
            log.append({"step": step, "loss": round(float(np.mean(run)), 4)}); run = []
    return log


# ---------------------------------------------------------------- evaluation helpers
def piece_means(emb: np.ndarray, piece_ids) -> tuple[np.ndarray, np.ndarray]:
    """(pieces, (n_pieces, D)) mean embedding per piece, L2-normalised."""
    piece_ids = np.asarray(piece_ids)
    pieces = np.unique(piece_ids)
    m = np.stack([emb[piece_ids == p].mean(0) for p in pieces]) if len(pieces) else np.zeros((0, emb.shape[1]))
    return pieces, m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def retrieval(emb: np.ndarray, ids, frames, chunk: int = 60, min_gap: int = 60, groups=None) -> dict:
    """Split each id's crops into chunks of `chunk` frames; for every chunk, rank the other chunks
    that are at least `min_gap` frames away in time (and, with `groups`, of the same group, e.g. the
    same team) by cosine similarity. rank-1 = the most similar one is the same player."""
    ids, frames = np.asarray(ids), np.asarray(frames)
    groups = np.asarray(groups) if groups is not None else np.zeros(len(ids), int)
    keys = [(i, f // chunk) for i, f in zip(ids, frames)]
    ck = sorted(set(keys))
    kidx = {k: j for j, k in enumerate(ck)}
    lab = np.array([kidx[k] for k in keys])
    _, m = piece_means(emb, lab)
    cid = np.array([k[0] for k in ck]); cmid = np.array([frames[lab == j].mean() for j in range(len(ck))])
    cgrp = np.array([pd.Series(groups[lab == j]).mode().iloc[0] for j in range(len(ck))])
    sims = m @ m.T
    r1, aps, n = 0, [], 0
    for j in range(len(ck)):
        ok = (np.abs(cmid - cmid[j]) >= min_gap) & (cgrp == cgrp[j])
        ok[j] = False
        if not ok.any() or not (ok & (cid == cid[j])).any():
            continue
        order = np.argsort(-sims[j, ok])
        rel = (cid[ok] == cid[j])[order]
        n += 1
        r1 += bool(rel[0])
        hits = np.cumsum(rel)
        aps.append(float((hits[rel] / (np.where(rel)[0] + 1)).mean()))
    return {"queries": n, "rank1": round(r1 / max(n, 1), 4), "mAP": round(float(np.mean(aps)) if aps else 0.0, 4)}
