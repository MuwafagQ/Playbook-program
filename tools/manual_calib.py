"""Manual homography keyframe-anchor calibration tool (for Google Colab).

Workflow (run in a Colab notebook, GPU not required for clicking):

    from tools import manual_calib as mc

    VIDEO = "/content/playbook/HILAL-HAZM_match_B_up7_Trim.mp4"

    # 1) Pick which frames to anchor. Aim for ~1 anchor per 0.5-1.0s of moving
    #    camera (denser where the camera pans/zooms fast). You can start sparse
    #    and add more later where you see drift in the output.
    #    e.g. every 20 frames from 170 to 396:
    keyframes = list(range(170, 396, 20))

    # 2) Annotate each keyframe. For each one:
    #    Cell A:  clk = mc.annotate(VIDEO, f); clk.show()
    #             -> click >=4 points on the FRAME, then the matching landmark
    #               on the PITCH diagram (clicks snap to the nearest numbered
    #               vertex). Alternate frame/pitch; do 4-8 pairs. Then move on.
    #    Cell B:  anchors[f] = clk.build()          # solves H for this frame
    #
    #    (annotate() returns a picker; you read .build() in the NEXT cell because
    #     Colab click callbacks fire after the showing cell returns.)

    anchors = {}     # {frame_idx: {"image_pts","pitch_pts","H"}}
    # ... repeat annotate/build per keyframe ...

    # 3) Densify to every frame via optical flow and save the sidecar:
    mc.build_sidecar(VIDEO, anchors,
                     out_path="/content/outputs/manual_h.json",
                     anchors_path="/content/outputs/manual_h_anchors.json")

    # 4) Point the pipeline at it:
    #    export H_MANUAL_SIDECAR=/content/outputs/manual_h.json   (or set in .env)

If the in-browser clicker misbehaves in your Colab runtime, you can build an
anchor programmatically instead:

    anchors[f] = mc.anchor_from_points(VIDEO, f,
                    image_pts=[[x1,y1],...], vertex_ids=[v1,...])

where vertex_ids are 1-based pitch landmark numbers (the labels drawn on the
pitch diagram by mc.show_pitch_reference()).
"""
from __future__ import annotations

import base64
import json
from typing import Sequence

import cv2
import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration

from geometry.manual_h import (
    compute_homography,
    densify,
    save_anchors,
    save_dense,
)
from geometry.motion import CameraMotionEstimator


# --------------------------------------------------------------------------- #
# Pitch reference: vertices, numbering, and a drawable diagram.
# --------------------------------------------------------------------------- #
def pitch_vertices(config: SoccerPitchConfiguration | None = None) -> np.ndarray:
    """Nx2 array of pitch landmark coordinates (cm), index 0 == label "1"."""
    cfg = config or SoccerPitchConfiguration()
    return np.asarray(cfg.vertices, dtype=np.float64).reshape(-1, 2)


def draw_pitch_reference(
    config: SoccerPitchConfiguration | None = None,
    scale: float = 0.08,
    margin: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a top-down pitch with numbered vertices for picking.

    Returns (bgr_image, vertex_px) where vertex_px is the Nx2 pixel location of
    each numbered landmark on the rendered image (same index order as
    pitch_vertices()).
    """
    cfg = config or SoccerPitchConfiguration()
    verts = pitch_vertices(cfg)
    length = float(getattr(cfg, "length", verts[:, 0].max()))
    width = float(getattr(cfg, "width", verts[:, 1].max()))

    W = int(round(length * scale)) + 2 * margin
    H = int(round(width * scale)) + 2 * margin
    img = np.full((H, W, 3), 40, dtype=np.uint8)
    # field
    cv2.rectangle(img, (margin, margin), (W - margin, H - margin), (40, 110, 40), -1)

    vertex_px = np.zeros_like(verts)
    vertex_px[:, 0] = verts[:, 0] * scale + margin
    vertex_px[:, 1] = verts[:, 1] * scale + margin

    # edges (if the config exposes them) for visual orientation
    edges = getattr(cfg, "edges", None)
    if edges:
        for a, b in edges:
            ia, ib = int(a) - 1, int(b) - 1
            if 0 <= ia < len(verts) and 0 <= ib < len(verts):
                pa = tuple(np.round(vertex_px[ia]).astype(int))
                pb = tuple(np.round(vertex_px[ib]).astype(int))
                cv2.line(img, pa, pb, (200, 200, 200), 1, cv2.LINE_AA)

    for i, (px, py) in enumerate(vertex_px):
        c = (int(round(px)), int(round(py)))
        cv2.circle(img, c, 4, (0, 215, 255), -1)
        cv2.putText(img, str(i + 1), (c[0] + 5, c[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (c[0] + 5, c[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img, vertex_px


def show_pitch_reference(config: SoccerPitchConfiguration | None = None) -> None:
    """Display the numbered pitch diagram inline (for reference while clicking)."""
    img, _ = draw_pitch_reference(config)
    _imshow_inline(img, "pitch reference (vertex numbers are 1-based)")


# --------------------------------------------------------------------------- #
# Frame access.
# --------------------------------------------------------------------------- #
def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read frame {frame_idx} from {video_path}")
        return frame
    finally:
        cap.release()


def _snap_to_vertex(click_xy, vertex_px: np.ndarray) -> int:
    d = np.linalg.norm(vertex_px - np.asarray(click_xy, dtype=np.float64), axis=1)
    return int(np.argmin(d))


# --------------------------------------------------------------------------- #
# Programmatic anchor (clicker-free fallback / scripting).
# --------------------------------------------------------------------------- #
def anchor_from_points(
    video_path: str,
    frame_idx: int,
    image_pts: Sequence[Sequence[float]],
    vertex_ids: Sequence[int],
    config: SoccerPitchConfiguration | None = None,
) -> dict:
    """Build one anchor from explicit image points + 1-based pitch vertex ids."""
    verts = pitch_vertices(config)
    image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    idx = np.asarray(vertex_ids, dtype=int) - 1
    if image_pts.shape[0] != idx.shape[0]:
        raise ValueError("image_pts and vertex_ids must have equal length")
    pitch_pts = verts[idx]
    H = compute_homography(image_pts, pitch_pts)
    return {"image_pts": image_pts, "pitch_pts": pitch_pts, "H": H}


# --------------------------------------------------------------------------- #
# Interactive Colab clicker (HTML canvas + kernel callback).
# --------------------------------------------------------------------------- #
def annotate(
    video_path: str,
    frame_idx: int,
    config: SoccerPitchConfiguration | None = None,
    max_width: int = 1100,
):
    """Return a KeyframeClicker for `frame_idx`. Call .show(), click, then read
    .build() in a LATER cell (Colab callbacks fire after the showing cell ends).
    """
    frame = read_frame(video_path, frame_idx)
    return KeyframeClicker(frame, frame_idx, config=config, max_width=max_width)


class KeyframeClicker:
    """Two-canvas picker: click points on the frame, snap matches on the pitch.

    Frame clicks and pitch clicks are paired by order, so alternate them and do
    an equal number of each (>=4). build() solves the homography.
    """

    _counter = 0

    def __init__(self, frame_bgr, frame_idx, config=None, max_width=1100):
        self.frame = frame_bgr
        self.frame_idx = int(frame_idx)
        self.cfg = config or SoccerPitchConfiguration()
        self.max_width = int(max_width)
        self.pitch_img, self.vertex_px = draw_pitch_reference(self.cfg)
        self.verts = pitch_vertices(self.cfg)
        self.image_clicks: list[list[float]] = []
        self.vertex_ids: list[int] = []
        KeyframeClicker._counter += 1
        self._uid = f"mc_{KeyframeClicker._counter}"

    # -- callbacks invoked from JS --
    def _on_frame_click(self, x, y):
        self.image_clicks.append([float(x), float(y)])

    def _on_pitch_click(self, x, y):
        vi = _snap_to_vertex((x, y), self.vertex_px)
        self.vertex_ids.append(vi)

    def _on_undo(self):
        # drop the most recent of whichever list is longer (keeps them aligned)
        if len(self.image_clicks) >= len(self.vertex_ids) and self.image_clicks:
            self.image_clicks.pop()
        elif self.vertex_ids:
            self.vertex_ids.pop()

    def show(self):
        from IPython.display import HTML, display
        from google.colab import output as colab_output

        colab_output.register_callback(f"{self._uid}.frame", self._on_frame_click)
        colab_output.register_callback(f"{self._uid}.pitch", self._on_pitch_click)
        colab_output.register_callback(f"{self._uid}.undo", self._on_undo)

        frame_b64 = _png_b64(self.frame)
        pitch_b64 = _png_b64(self.pitch_img)
        fh, fw = self.frame.shape[:2]
        ph, pw = self.pitch_img.shape[:2]
        display(HTML(_clicker_html(self._uid, frame_b64, pitch_b64, fw, fh, pw, ph, self.max_width)))
        print(f"Frame {self.frame_idx}: click >=4 points on the FRAME and the matching "
              f"vertex on the PITCH (alternating). Then run .build() in the next cell.")

    def build(self, ransac_thresh: float = 15.0) -> dict:
        n = min(len(self.image_clicks), len(self.vertex_ids))
        if n < 4:
            raise RuntimeError(
                f"need >=4 paired clicks, have frame={len(self.image_clicks)} "
                f"pitch={len(self.vertex_ids)}"
            )
        image_pts = np.asarray(self.image_clicks[:n], dtype=np.float64)
        pitch_pts = self.verts[np.asarray(self.vertex_ids[:n], dtype=int)]
        H = compute_homography(image_pts, pitch_pts, ransac_thresh=ransac_thresh)
        if H is None:
            raise RuntimeError("homography solve failed (degenerate correspondences)")
        return {"image_pts": image_pts, "pitch_pts": pitch_pts, "H": H}


# --------------------------------------------------------------------------- #
# Densify + save.
# --------------------------------------------------------------------------- #
def build_sidecar(
    video_path: str,
    anchors: dict[int, dict],
    out_path: str,
    anchors_path: str | None = None,
    motion_estimator: CameraMotionEstimator | None = None,
    overlay_top_frac: float | None = None,
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    """Optical-flow densify the clicked anchors and write the pipeline sidecar.

    Runs optical flow across [min_anchor, max_anchor], chains/blends each
    anchor's H to every intermediate frame, and saves a dense sidecar to
    out_path (and optionally the raw anchors to anchors_path for re-editing).
    Returns the dense {frame_idx: H} map.
    """
    anchors = {int(k): v for k, v in anchors.items()}
    anchor_H = {k: np.asarray(v["H"], dtype=np.float64)
                for k, v in anchors.items() if v.get("H") is not None}
    if len(anchor_H) < 2:
        raise ValueError("need at least 2 anchors with a solved H to densify")

    motion = motion_estimator or CameraMotionEstimator()
    if overlay_top_frac is not None:
        motion.overlay_top_frac = float(overlay_top_frac)

    lo, hi = min(anchor_H), max(anchor_H)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step_M: dict[int, np.ndarray | None] = {}
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        ok, prev = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {lo}")
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        for t in range(lo + 1, hi + 1):
            ok, cur = cap.read()
            if not ok or cur is None:
                step_M[t] = None
                prev_gray = None
                continue
            cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            M = motion.estimate(prev_gray, cur_gray) if prev_gray is not None else None
            step_M[t] = M
            prev_gray = cur_gray
            if verbose and (t - lo) % 25 == 0:
                nbroken = sum(1 for v in step_M.values() if v is None)
                print(f"  flow {t - lo}/{hi - lo} (broken links so far: {nbroken})")
    finally:
        cap.release()

    dense = densify(anchor_H, step_M)
    meta = {
        "video": video_path,
        "fps": float(fps) if fps and fps == fps else None,
        "anchor_frames": sorted(anchor_H.keys()),
        "n_anchors": len(anchor_H),
        "covered_frames": len(dense),
        "range": [lo, hi],
    }
    save_dense(out_path, dense, meta=meta)
    if anchors_path:
        save_anchors(anchors_path, anchors, meta=meta)
    if verbose:
        nbroken = sum(1 for v in step_M.values() if v is None)
        print(f"Saved sidecar: {len(dense)} frames covering [{lo},{hi}] "
              f"from {len(anchor_H)} anchors ({nbroken} broken flow links) -> {out_path}")
    return dense


# --------------------------------------------------------------------------- #
# QA: overlay a dense H on a frame to eyeball projection quality.
# --------------------------------------------------------------------------- #
def preview_projection(
    video_path: str,
    dense: dict[int, np.ndarray],
    frame_idx: int,
    config: SoccerPitchConfiguration | None = None,
):
    """Draw the pitch template back-projected onto a frame using its dense H,
    so you can visually confirm a keyframe / interpolated frame lines up."""
    H = dense.get(int(frame_idx))
    if H is None:
        print(f"frame {frame_idx} not covered by the sidecar")
        return
    cfg = config or SoccerPitchConfiguration()
    verts = pitch_vertices(cfg)
    H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))  # pitch -> image
    proj = cv2.perspectiveTransform(verts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    frame = read_frame(video_path, frame_idx).copy()
    edges = getattr(cfg, "edges", None) or []
    for a, b in edges:
        ia, ib = int(a) - 1, int(b) - 1
        pa, pb = proj[ia], proj[ib]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            cv2.line(frame, tuple(np.round(pa).astype(int)),
                     tuple(np.round(pb).astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
    _imshow_inline(frame, f"projection @ frame {frame_idx}")


# --------------------------------------------------------------------------- #
# Small display helpers.
# --------------------------------------------------------------------------- #
def _png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _imshow_inline(bgr: np.ndarray, title: str = "") -> None:
    try:
        from IPython.display import HTML, display
        b64 = _png_b64(bgr)
        cap = f"<div style='font:13px sans-serif;margin:4px 0'>{title}</div>" if title else ""
        display(HTML(f"{cap}<img src='data:image/png;base64,{b64}'/>"))
    except Exception:
        # Non-notebook fallback.
        cv2.imwrite("/tmp/manual_calib_preview.png", bgr)
        print(f"[manual_calib] wrote /tmp/manual_calib_preview.png ({title})")


def _clicker_html(uid, frame_b64, pitch_b64, fw, fh, pw, ph, max_width) -> str:
    disp_fw = min(fw, max_width)
    scale_f = disp_fw / fw
    disp_fh = int(round(fh * scale_f))
    disp_pw = min(pw, max_width)
    scale_p = disp_pw / pw
    disp_ph = int(round(ph * scale_p))
    return f"""
<div style="font:13px sans-serif">
  <div style="margin-bottom:6px">
    <b>Frame</b> — click the same real-world points you'll match on the pitch.
    <button onclick="{uid}_undo()">Undo last</button>
    <span id="{uid}_count"></span>
  </div>
  <canvas id="{uid}_fc" width="{disp_fw}" height="{disp_fh}"
          style="border:1px solid #888;cursor:crosshair"></canvas>
  <div style="margin:6px 0"><b>Pitch</b> — click the matching numbered vertex
    (snaps to nearest).</div>
  <canvas id="{uid}_pc" width="{disp_pw}" height="{disp_ph}"
          style="border:1px solid #888;cursor:crosshair"></canvas>
<script>
(function(){{
  const fimg = new Image(), pimg = new Image();
  const fc = document.getElementById("{uid}_fc"), pc = document.getElementById("{uid}_pc");
  const fctx = fc.getContext("2d"), pctx = pc.getContext("2d");
  const sf = {scale_f}, sp = {scale_p};
  let fpts = [], ppts = [];
  fimg.onload = ()=>fctx.drawImage(fimg,0,0,{disp_fw},{disp_fh});
  pimg.onload = ()=>pctx.drawImage(pimg,0,0,{disp_pw},{disp_ph});
  fimg.src = "data:image/png;base64,{frame_b64}";
  pimg.src = "data:image/png;base64,{pitch_b64}";
  function redraw(ctx,img,W,Hh,pts){{
    ctx.drawImage(img,0,0,W,Hh);
    ctx.fillStyle="red"; ctx.strokeStyle="yellow"; ctx.font="14px sans-serif";
    pts.forEach((p,i)=>{{ctx.beginPath();ctx.arc(p[0],p[1],4,0,7);ctx.fill();
      ctx.fillStyle="yellow";ctx.fillText(i+1,p[0]+6,p[1]-6);ctx.fillStyle="red";}});
  }}
  function count(){{document.getElementById("{uid}_count").innerText=
     " frame:"+fpts.length+" pitch:"+ppts.length;}}
  fc.addEventListener("click",e=>{{
    const r=fc.getBoundingClientRect();
    const dx=e.clientX-r.left, dy=e.clientY-r.top;
    fpts.push([dx,dy]); redraw(fctx,fimg,{disp_fw},{disp_fh},fpts); count();
    google.colab.kernel.invokeFunction("{uid}.frame",[dx/sf,dy/sf],{{}});
  }});
  pc.addEventListener("click",e=>{{
    const r=pc.getBoundingClientRect();
    const dx=e.clientX-r.left, dy=e.clientY-r.top;
    ppts.push([dx,dy]); redraw(pctx,pimg,{disp_pw},{disp_ph},ppts); count();
    google.colab.kernel.invokeFunction("{uid}.pitch",[dx/sp,dy/sp],{{}});
  }});
  window["{uid}_undo"]=function(){{
    if(fpts.length>=ppts.length&&fpts.length)fpts.pop();
    else if(ppts.length)ppts.pop();
    redraw(fctx,fimg,{disp_fw},{disp_fh},fpts);
    redraw(pctx,pimg,{disp_pw},{disp_ph},ppts);count();
    google.colab.kernel.invokeFunction("{uid}.undo",[],{{}});
  }};
}})();
</script>
</div>
"""
