# Detection: the annotation and training plan

*23 Sep 2026. Written after finding that the player detector — trained on 78 source images —
misses more than one player in four, and that this is the root cause of the tracklet
fragmentation costing months of manual Re-ID work.*

## The evidence

| Measurement | Value | Source |
|---|---|---|
| Player detector recall | **72.5%** | `players-detection-my09y/2` |
| Source images it was trained on | **78** (128 after augmentation) | Roboflow project |
| Architecture | **yolov8n** → **AGPL-3.0** | Roboflow model record |
| Preprocessing | **Stretch to 640×640** on 16:9 footage | Version 2 config |
| Median players detected per frame | **16** (min 6) | `per_frame_tracks_half2_reid.csv` |
| Detection gaps inside tracks | **1,802**, of which **883 (49%) are 1–5 frames** | same |
| Tracklet fragments | **952**, from 33 track IDs | same |

A gap of one to five frames is not an occlusion and not a player leaving frame. **It is the
detector blinking** — 883 times in four minutes. That is where the fragments come from.

## Three defects, in order of cost to fix

### 1. Aspect-ratio distortion — free to fix

Version 2 preprocesses with **"Stretch to 640×640"**. The source is 1920×1080.

Squeezing 16:9 into a square compresses horizontally by ~44%. Every training image and every
inference frame contains **players squashed narrow**. Small distant players — the ones being
missed — are worst affected.

**Fix: "Fit within" (letterbox).** One setting. Do this before anything else.

### 2. Near-duplicate frames and a leaking validation split ⚠

The CVAT annotations are **continuous sequences from three clips of ~3 minutes**. Two problems:

**Near-duplicates.** Frame 100 and frame 101 are almost the same picture. A hundred consecutive
frames carry roughly the information of one. Volume in this dataset is not the same as data.

**Validation leakage.** If a random split puts frame 100 in train and frame 101 in validation,
the model is validated on a picture it has effectively memorised. **Reported accuracy is then
optimistic — the real-world 72.5% recall may be flattering the model, not maligning it.**

**Fixes, both mandatory:**
- **Sample sparsely.** One frame every 3–5 seconds (75–125 frames at 25fps), not every frame.
- **Split by clip, never randomly.** All frames from one clip go to one split. A clip the model
  trained on must never appear in validation.

### 3. Too little diversity, and one class that cannot learn

Three clips is three camera setups, three lighting conditions, two or three kit designs.

**Twenty matches at ten frames each beats three matches at two hundred frames each.**

And the class counts in the current project:

| Class | Instances | Verdict |
|---|---|---|
| player | 1,465 | workable, wants more |
| referee | 160 | thin |
| ball | 81 | thin, and the ball is small and fast |
| **goalkeeper** | **48** | **will not learn** |

Forty-eight goalkeeper instances is not a class, it is noise. Either deliberately over-sample
goalkeeper frames, or **fold goalkeeper into player** and recover the role afterwards from kit
colour and pitch position — which is cheap and reliable, since the keeper is the one player in a
different kit standing in the penalty area.

## The annotation plan

### Step 1 — Migrate CVAT

Export from CVAT as **COCO 1.0** or **CVAT for images 1.1**; Roboflow reads both. Import into
**`players-detection-my09y`** — our own broadcast domain — not into a new project.

### Step 2 — Subsample before importing

Do not import every frame. From 3 clips × 3 minutes ≈ 13,500 frames, keep roughly **150–200**,
evenly spread. Record which clip each frame came from; that metadata drives the split.

### Step 3 — Widen coverage, deliberately

The existing clips cannot supply diversity on their own. Add frames chosen to vary:

- **different matches, stadiums and dates** — the single biggest factor
- **day and night**, sun and floodlight, shadow across the pitch
- **camera zoom levels** — wide, medium, tight
- **both ends** of the pitch, and both halves (camera side changes)
- **crowded penalty-box scenes** and sparse midfield
- **players at frame edges**, partly cut off
- **occlusions** — players overlapping
- **small, distant players** — the current failure mode
- **goalkeepers**, if keeping the class
- **referees and assistants**, including on the touchline

Target **300–500** well-spread images for a solid model, **800–1,000** for a strong one. Reaching
1,000 near-duplicates from three clips would be worth less than 300 varied ones.

### Step 4 — Auto-label, then correct

Run the existing model over new frames to pre-annotate, then fix its output. Correcting is far
faster than drawing from scratch, and it focuses attention exactly where the model is wrong.

### Step 5 — Mine hard negatives ⭐

Use the frames where the current model **fails**. We can find them precisely: take the 883
one-to-five-frame gaps in the tracking CSV, pull those exact frames, and annotate them. Each one
is a known, confirmed miss.

This is worth more per image than any random sampling, because it targets the defect directly.

### Step 6 — Generate the version

- Preprocessing: **Fit within**, not stretch
- Split: **by clip**, with ≥15 validation images (required for NAS)
- Augmentation: modest and physical — small rotation, brightness, blur. Avoid vertical flip
  (football video is never upside down) and avoid heavy colour shift (kit colour carries the
  team signal)

### Step 7 — Train

See the architecture section below.

### Step 8 — Measure the thing that matters

The mAP number is not the test. **Re-run half 2 and count the tracklet fragments.**
Today: 952 segments, 883 short gaps. If those fall substantially, the manual Re-ID cost falls
with them. That is the result worth reporting.

## Architecture and licences

### Which project to train on

**`players-detection-my09y`** — our own broadcast footage, the correct domain.

Use **`footballs-player-detection-zkams-zia6c/2`** (750 images, 13,668 players, RF-DETR-small,
94.8% mAP) **only as a training checkpoint**, never as the deployed model. Its tactical-view
footage is the wrong domain for inference, which testing confirmed — but a model that has already
seen 13,668 football players is a far better starting point than COCO. Early layers learn
"person in kit on grass"; our data then teaches the angle, lighting and kits.

### The architecture

**`rfdetr-nas-parent`** — RF-DETR Neural Architecture Search — is Roboflow's current recommended
default for custom object detection. It trains one parent model, then searches the RF-DETR
architecture space against our data and returns a frontier of candidates with measured accuracy
and latency.

Prerequisites: **≥15 validation images** (version 2 has 22, so this is met) and plan entitlement
(Core or Growth). Entitlement cannot be read through the API — a rejection names it. If rejected,
fall back to **`rfdetr-medium`**. Do not fall back to hand-tuning hyperparameters.

### Licences — the current position

| Model | Licence | Status |
|---|---|---|
| RF-DETR, Nano → Large (the `rfdetr` package and Apache-designated weights) | **Apache-2.0** | Clean for commercial use |
| RF-DETR **XL / 2XL** | **PML 1.0** | Different terms — read them |
| **RF-DETR NAS children** | **Unknown** | ⚠ see below |
| All Ultralytics YOLO — v8, v11, **and YOLO26** | **AGPL-3.0** by default | Enterprise licence required for proprietary commercial use |

**⚠ On NAS and licensing:** NAS mines architectures out of an RF-DETR parent. Whether a mined
child inherits the Apache-2.0 terms of the standard sizes, or the PML 1.0 terms of XL/2XL, is not
stated in the documentation available. **Check the licence badge on the trained model in the
Roboflow console before deploying it** — exactly as we did for the two existing models, where the
badges told us one was AGPL-3.0 and the other Apache-2.0. Treat this as verify-then-deploy.

**On YOLO26 and paid plans:** a paid Roboflow plan is a **compute and feature tier**, and has
nothing to do with the model's licence. Ultralytics state that all their YOLO trained models,
YOLO26 included, are **AGPL-3.0 by default**, and that embedding them in a proprietary commercial
product requires their Enterprise Licence. Paying Roboflow does not buy that. **YOLO is not a way
out of the licence problem at any price tier.**

### The pitch-keypoint model

`football-field-detection` scores 97.8% mAP on 477 images and the console shows **Apache-2.0**.
That model is in good shape — leave it alone.

Two cautions:
- The Roboflow console reports Apache-2.0, but the RF-DETR repository lists **XL** under PML 1.0.
  These disagree. It is also possible the badge reflects the **dataset** licence rather than the
  model's. **Ask Roboflow support which it is**, and get it in writing.
- If it turns out restricted, the alternatives are awkward: YOLO-pose (v8/v11/v26) is Ultralytics
  and therefore AGPL, which is worse. The genuinely clean routes would be a **classical line
  detector** (Hough transform plus fitting the known pitch model — licence-free, and football
  pitches are a well-suited case), or negotiating terms. Do not switch to YOLO-pose thinking it
  solves anything.

## Order of work

| # | Action | Cost |
|---|---|---|
| 1 | Change preprocessing to **Fit within** | minutes |
| 2 | Migrate CVAT annotations, subsampled, with clip labels | hours |
| 3 | Annotate hard negatives from the 883 known gap frames | a day |
| 4 | Widen coverage with frames from more matches | days |
| 5 | Generate version: fit-within, **split by clip** | minutes |
| 6 | Train `rfdetr-nas-parent` from the football checkpoint | hours of compute |
| 7 | **Verify the licence badge** on the trained model | minutes |
| 8 | Re-run half 2, **count fragments**, compare to 952 | hours |

## Sources
- https://www.ultralytics.com/license — AGPL-3.0 default, Enterprise for commercial
- https://github.com/roboflow/rf-detr — Apache-2.0 for standard sizes, PML 1.0 for XL/2XL
- Roboflow training-and-evaluation guidance (MCP skill resource)
- Measurements from `per_frame_tracks_half2_reid.csv` and the Roboflow workspace
