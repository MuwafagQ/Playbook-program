# Handover to the code-pipeline session

*Written 23 Sep 2026. Everything the technical session needs from the strategy session.
Read this first, then `DETECTION-PLAN.md`, `REID-REVIEW-TOOL.md`, `SN-GAMESTATE.md`.*

---

## The headline finding

**The Re-ID problem is substantially a detection problem.**

Measured on `per_frame_tracks_half2_reid.csv` (half 2, 5,958 frames, ~4 minutes):

| | |
|---|---|
| Player-frames | 96,188 |
| Distinct track IDs (class 2) | 33 |
| Tracklet segments (gap > 5 frames) | **952** |
| Segments ≥ 0.5s, worth labelling | 328 |
| Top 140 segments | cover **90.5%** of all player-frames |
| Detection gaps inside tracks | **1,802** |
| — of which **1–5 frames long** | **883 (49%)** |
| Median players detected per frame | 16 (min 6, max 23) |

A 1–5 frame gap is not an occlusion and not a player leaving frame. **It is the detector
blinking**, 883 times in four minutes. That is the origin of the 952 fragments, and of the
manual remapping that has been costing months.

The detector responsible: `players-detection-my09y/2` — **yolov8n**, trained on **78 source
images**, **recall 72.5%**, preprocessing **"Stretch to 640×640"** on 16:9 footage.

**Priority #1 is fixing detection, not Re-ID.**

---

## Roboflow workspace, as found

| Project | Images | Model | Architecture | mAP@50 | Recall | Licence |
|---|---|---|---|---|---|---|
| `players-detection-my09y` | 78 | `/2` | **yolov8n** | 78.6% | **72.5%** | **AGPL-3.0** |
| `footballs-player-detection-zkams-zia6c` | 750 | `/2` | rfdetr-small | 94.8% | 89.3% | Apache-2.0 |
| `football-field-detection-f07vi-it2xv` | 477 | keypoints | RF-DETR keypoint XL | 97.8% | 96.9% | Apache-2.0 (verify) |

Class counts in `players-detection-my09y`: player 1,465 · referee 160 · ball 81 ·
**goalkeeper 48**.

**All three projects are public on Roboflow Universe.** Decide deliberately whether that is
intended.

`footballs-player-detection` is a forked public dataset in **tactical view** — the wrong domain.
Tested on our broadcast footage and gave bad output. **Use it as a training checkpoint only,
never as the deployed model.**

---

## ⚠ CVAT: the annotation work is valuable, but for a different purpose than assumed

The CVAT data is continuous sequences from three ~3-minute clips, annotated with
**keyframe interpolation** — annotate frame 1920, skip to 1930, and CVAT fills the ten frames
between.

**Those interpolated boxes are derived, not observed.** They are linear guesses between two
keyframes. For a player running a curve they are slightly wrong, and they carry almost no new
information for a detector. Exporting all of them produces a dataset that looks ten times
bigger and teaches roughly the same amount — while introducing correlated, partly inaccurate
labels and, if split randomly, **validation leakage that inflates the reported score**.

### But the same data is exactly what we are missing elsewhere

CVAT assigns **consistent object IDs across frames**. Interpolated tracks with stable identities
are **tracking ground truth** — and that is the one thing we do not have and urgently need:

- It is what scoring our pipeline on the SoccerNet GSR metric requires.
- It is what turns the KAUST ask from *"we have gaps"* into *"here is our benchmark score and
  where we lose points."*
- It is what tells us whether a change to the pipeline actually helped.

**So split the export by purpose:**

| Purpose | Export | What to include |
|---|---|---|
| **Detector training** → Roboflow | **COCO 1.0** or **CVAT for images 1.1** | **Keyframes only.** Drop the interpolated frames |
| **Tracking / Re-ID ground truth** → keep in the repo | **MOT 1.1**, **CVAT for video 1.1**, or **Datumaro** | Everything, **with track IDs preserved** |

Roboflow object-detection projects are per-image and **do not preserve track IDs across frames**,
so the tracking ground truth must live outside Roboflow. Keep it in version control as the
evaluation set.

**Keypoints and players are annotated together in CVAT but are separate Roboflow projects**
(object-detection vs keypoint-detection are different task types). Export twice, filtering
labels each time, into the two projects.

---

## Detection: what to do, in order

Full detail in `DETECTION-PLAN.md`. Summary:

1. **Preprocessing → "Fit within"**, not "Stretch to". 1920×1080 squeezed into a square
   compresses players horizontally by ~44%. Minutes to fix.
2. **Import CVAT keyframes** into `players-detection-my09y`, tagged with their source clip.
3. **Mine hard negatives.** Pull the exact frames behind the **883 short gaps** from the
   tracking CSV and annotate them. Each is a confirmed miss. Highest value per image available.
4. **Widen coverage.** Twenty matches at ten frames each beats three matches at two hundred.
   Vary stadium, time of day, zoom, pitch end, occlusion, distance. Target 300–500 varied
   images, 800–1,000 for strong.
5. **Goalkeeper has 48 instances and will not learn.** Either over-sample keeper frames
   deliberately, or fold it into `player` and recover the role afterwards from kit colour and
   penalty-area position.
6. **Generate the version: split by clip, never randomly.** ≥15 validation images (NAS
   requirement; current version has 22).
7. **Train `rfdetr-nas-parent`**, using a football model as the checkpoint rather than COCO.
   Fall back to `rfdetr-medium` if NAS is not entitled on the plan. Do **not** fall back to
   hand-tuning hyperparameters.
8. **Verify the licence badge** on the trained model before deploying.
9. **Re-run half 2 and count the fragments.** 952 today. **That number, not mAP, is the test.**

### Also worth trying, cheap

- **Lower the confidence threshold.** Precision 90.2% against recall 72.5% means the model is too
  cautious. A tracker can reject a false positive by temporal consistency; it can never invent a
  player it never saw. For tracking, recall is worth more than precision.
- **Slice the frame for detection.** If whole 1080p frames are being resized to 640 before
  detection, small distant players shrink further. Detecting on overlapping tiles at native
  resolution and merging results is a known gain on small objects.

---

## Licences — current position

| | Licence | Note |
|---|---|---|
| RF-DETR Nano → Large | **Apache-2.0** | Clean for commercial use |
| RF-DETR **XL / 2XL** | **PML 1.0** | Different terms |
| **RF-DETR NAS children** | **Undocumented** | ⚠ Verify the badge on the trained model before deploying |
| All Ultralytics YOLO — v8, v11, **YOLO26** | **AGPL-3.0** by default | Enterprise licence required for proprietary commercial use |
| TrackLab | MIT | Permissive |
| sn-gamestate | GPL-3.0 | Triggers on distribution, not network use — take legal advice |
| SoccerNet videos | NDA, non-commercial | Do not train a shipped model on them |

**A paid Roboflow plan is a compute tier and does not change any model's licence.** YOLO is not a
route out of the AGPL question at any price.

**The pitch-keypoint model** shows Apache-2.0 in the console but the RF-DETR repository lists XL
under PML 1.0. These disagree, and the badge may reflect the *dataset* licence. **Ask Roboflow
support in writing.** If it turns out restricted, do not switch to YOLO-pose — that is Ultralytics
and therefore AGPL, which is worse. The clean alternative is a classical line detector (Hough plus
fitting the known pitch geometry).

---

## Re-ID review tool

Full spec in `REID-REVIEW-TOOL.md`. The short version: the current method is slow because it
reviews **all** segments, **as video**, with **no pre-labels** — so every decision is an act of
recall rather than verification.

The replacement: triage to the ~330 segments worth labelling · export crop montages instead of
clips · auto-label jersey numbers with majority voting over each tracklet (`SoccerNet/sn-jersey`,
~90% tracklet accuracy on their data — **measure it on ours**) · cluster tracklets by appearance
using embeddings · review grouped, keyboard-driven, uncertainty-first.

**Tested and honest:** geometric gap-bridging (a player cannot move faster than ~9 m/s) gave only
**23 of 328** segments an unambiguous continuation within a one-second window. Implement it
because it is nearly free, but it is not a lever.

**Every confirmed tracklet becomes training data we own** — so match #4 costs less than match #1.

---

## SoccerNet

`sn-gamestate` is our pipeline as a public benchmark: **single moving broadcast camera** → player
positions and identities on a minimap. Stages: YOLOv11 detection (swappable), tracking, PRTReid /
BPBreID re-identification, **TVCalib** calibration, **MMOCR** jersey numbers, team affiliation.
Built on **TrackLab** (MIT), configured with Hydra YAML, so components are swappable by config.

Documented gap: no path for running on your own video. Writing that **dataset adapter** is the
first piece of work and a good scoped first task for a collaborator.

Highest-value cheap step: **score our own pipeline with their evaluation metric**, using the CVAT
tracking ground truth described above.

Repos: [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) ·
[sn-jersey](https://github.com/SoccerNet/sn-jersey) ·
[sn-reid](https://github.com/SoccerNet/sn-reid) ·
[sn-calibration](https://github.com/SoccerNet/sn-calibration) ·
[TrackLab](https://github.com/TrackingLaboratory/tracklab)

---

## Product context the pipeline session should know

- **The product is the metric layer, not the data layer.** Layer 1 (manual event tagging,
  ~12 person-hours/match) delivers the real product today because the defining metrics are
  mostly event-based. See `MVP-STRATEGY.md`.
- **xP needs player identity**, because the value proposition at youth level is *development*,
  which requires per-player aggregation across matches.
- **Broadcast is bad for whole-team geometry** (never all 22 in frame) **but fine for ball-local
  events** — the camera follows the ball. Our metric set is event-based, so broadcast is the
  right input, not a compromise.
- **The pilot runs on Layer 1 regardless.** Pipeline improvements reduce its cost; they are not
  a precondition for it. Do not let the pilot wait on the pipeline.
- **The test of any pipeline change is the fragment count and human hours per match**, not mAP.

## Known gaps, unresolved

- The dashboard **front-end source is in no version control anywhere**.
- The repository is **public** and contains the risk register and named interviewees.
- Metrics shown in the prototype (PR, BP, xT) are demo placeholders, not computed.
