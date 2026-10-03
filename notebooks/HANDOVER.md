# Match Tracking Data — Handover

## 1. What the data is

A full football match (both halves merged) tracked frame-by-frame from broadcast video:
players, goalkeepers, referees, and the ball, each with a pixel-space bounding box and a
pitch-space (homography-projected) position.

Deliverables:
- `per_frame_tracks_half1_unified.csv` — half 1, frames **1066–3333** (reviewed/cleaned window)
- `per_frame_tracks_half2_unified.csv` — half 2, frames **4270–5720** (reviewed/cleaned window)
- `notebooks/merge_halves.ipynb` — merges the two into one continuous-timeline
  `per_frame_tracks_full_match.csv` + one concatenated match video

Each row = one detected object in one frame. Multiple rows share a `frame` value (one per
object visible that frame).

## 2. Column reference

| Column | Meaning |
|---|---|
| `frame` | Frame index (post-merge: 0-based, continuous across both halves; `src_frame`/`half` preserve the original per-half frame number) |
| `track_id` | Raw tracker id (BoT-SORT). **Not stable** — gets reused/recycled across different physical people over time; don't use directly for identity. |
| `display_track_id` | Stabilized/curated id — this is the **canonical player identity** to use for analysis. 20 canonical ids across the match: HIL `{4,5,6,7,8,9,10,11,17,19}`, HAZ `{2,3,13,14,15,16,18,20,21,22}`. `-1` for the ball. |
| `class_id` | `0`=ball, `1`=goalkeeper, `2`=player, `3`=referee |
| `conf` | Detector confidence for this box |
| `x1,y1,x2,y2` | Bounding box in **image pixel space** |
| `x_m`, `y_m` | Position projected to **pitch space via the homography**. Despite the `_m` suffix, these are **NOT meters** — they're in the pitch config's native units (centimeters), on a 12000×7000 grid. To get meters, divide by 100. |
| `team_id` | `0`=HIL, `1`=HAZ, `-1`=unknown/ball/referee |
| `homography_ok` | Whether a valid per-frame homography existed (see §3) — if `False`, `x_m`/`y_m` are unreliable/stale |
| `kp_used` | Number of pitch keypoints used to fit that frame's homography |
| `inlier_ratio`, `reproj_err` | Homography fit quality diagnostics |
| `detector_ran` | Whether the detector ran fresh this frame, vs. tracker-only propagation |
| `homography_state` | Internal state tag (e.g. fresh fit vs. held-over from a previous good frame) |
| `ball_interpolated` | `True` if this ball row is a gap-fill, not a real detection |
| `ball_source` | How the ball position was derived (real detection / pass-aware reconstruction / generic interpolation) |
| `pass_id`, `pass_role`, `pass_phase`, `departure_time`, `arrival_time`, `duration_s`, `outcome` | Manually tagged pass annotations (passer/receiver, in-flight vs. ground, timing, success/fail) |
| `notes` | Free-text audit trail of manual ID corrections applied during cleanup (e.g. `idfix_interp`, `idfix_recovered_from_2104`) |

## 3. The homography (pixel → pitch projection)

Implemented in `geometry/homography.py` (`HomographyEstimator`), run **independently every frame**:

1. A field-keypoint model detects up to 32 named pitch landmarks (corners, penalty/six-yard
   box corners, center circle, penalty spots, halfway line — see `sports.configs.soccer.SoccerPitchConfiguration`).
2. Keypoints below `kp_conf` confidence are dropped; need **≥4** to fit.
3. Each surviving keypoint is matched to its known real-world pitch coordinate (the vertex
   `LABEL`, not positional order — the model's internal class ordering doesn't match vertex
   numbering, so labels are explicitly remapped).
4. `cv2.findHomography(image_points, pitch_points)` → 3×3 matrix `H`.
5. Two degeneracy guards reject a bad fit and instead **hold the last good H** for up to
   15 frames:
   - **Collinearity guard**: if the surviving keypoints are nearly collinear (e.g. only the
     halfway line visible during a mid-pitch camera pan), the DLT solve is rank-deficient → reject.
   - **Condition-number guard**: if `cond(H) > 1e8` (numerically unstable solve) → reject.
6. No EMA smoothing, no RANSAC — it's a plain per-frame DLT fit with hold-over as the only
   stabilization. `x_m,y_m` come from `cv2.perspectiveTransform` of each box's anchor point
   (bottom-center for people, center for the ball) through `H`.

**Pitch coordinate system** (`sports.configs.soccer.SoccerPitchConfiguration`, units = cm):
- `x ∈ [0, 12000]`, `y ∈ [0, 7000]` — origin at one corner flag, x along the touchline, y along the goal line.
- Key landmarks: goal lines at `x=0` and `x=12000`; halfway line at `x=6000`; center spot at `(6000,3500)`; penalty spots at `(1100,3500)` and `(10900,3500)`; six-yard/penalty box corners at the listed y-bands (`1450/2584/4416/5550`) on each side.
- When `homography_ok=False`, treat `x_m,y_m` for that row as stale/unreliable (it's the held-over projection, not a fresh fit).

## 4. Overall pipeline (source video → final CSV)

```
source video
   │
   ├─► player/ball detector (Roboflow model) ──► boxes + class_id + conf
   ├─► field-keypoint detector (Roboflow model) ──► 32 pitch landmarks/frame
   │
   ├─► BoT-SORT tracker (separate tracker instances for players vs. officials)
   │        → track_id (raw, frame-to-frame association, ReID-assisted)
   │
   ├─► HomographyEstimator (per frame) ──► H, on-pitch filtering
   │        (boxes outside pitch bounds + margin are dropped as detector noise)
   │
   ├─► Team classifier (jersey-color/embedding based) ──► team_id (HIL/HAZ)
   │
   ├─► IDStabilizer (separate instances for players / goalkeepers / referees)
   │        → display_track_id: re-links raw track_id fragments using position +
   │          appearance-embedding gallery + frame-continuity IoU, caps max ids
   │          per role, does short-gap "spatial rescue" relinking
   │
   ├─► BallSmoother
   │        → gap interpolation (max N missing frames), jump gating, low-conf rejection
   │
   └─► CSVWriter → per_frame_tracks_*.csv  (one row per object per frame)
```

Then, **manual post-processing** (this session) on top of the raw pipeline output:
1. **20-canonical-ID unification** — mapped every `display_track_id` fragment across the
   match to one of 20 fixed player identities (10 HIL + 10 HAZ), since the raw stabilizer
   still produces fragmented/reused ids over a full match.
2. **Leftover fragment cleanup** — dropped rows belonging to ids that never matched any
   canonical player (noise/false tracks).
3. **Ball trajectory cleaning** — duplicate-box collapse, anti-overshoot (PCHIP) re-interpolation of gaps, iterative rejection of low-confidence outlier anchors, and **pass-aware reconstruction**: gaps inside a manually tagged `pass_id` span are filled using only detections within that pass (or synthetic passer/receiver-position endpoints), instead of globally-nearest anchors.
4. **Manual ID-swap correction** — visual-review-driven fixes for cases where a single raw
   track silently switched which physical player it represents mid-stream (the stabilizer's
   embedding/position matching can miss this); each fix traced via exact-coordinate matching
   back to the pre-cleanup source CSV, verified against a zero-duplicate
   `(frame, display_track_id)` invariant per scoped window.
5. **Half merge** (`merge_halves.ipynb`) — trims both halves to their reviewed windows,
   renumbers frames onto one continuous match timeline, offsets `pass_id` so the two halves'
   pass tags don't collide, concatenates the videos to match.

## 5. Known caveats for analysis

- `x_m`/`y_m` are **centimeters on a 12000×7000 grid**, not meters — convert before computing
  real-world distances/speeds (divide by 100).
- Frames where `homography_ok=False` carry a held-over (not freshly fit) pitch position —
  filter on this column if you need strictly fresh projections.
- One known unresolved gap: HAZ `#16` loses identity from frame ~5479 to the end of the half2
  window (genuine tracker dropout, no recoverable raw track found) — currently a true gap in
  that player's trajectory, not a labeling bug.
- `ball_interpolated=True` rows are reconstructed, not real detections — weight/flag
  accordingly in any ball-speed or possession analysis.
