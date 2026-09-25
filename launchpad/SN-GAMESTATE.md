# sn-gamestate — what it is, what we can take, and the licence position

*Researched 23 Sep 2026 from the repository itself.*

## It is single-camera broadcast — exactly our case

Two misconceptions to clear up:

- **sn-gamestate is built for "a single moving camera."** That is its stated premise: extract
  high-level information about a game from raw broadcast video and place players on a minimap.
  It is not a multi-camera system.
- **sn-jersey's tracklets come from broadcast footage too**, and the task description calls them
  low-resolution and motion-blurred. The multi-camera dataset is **sn-reid**, which is a
  different task.

So the objection "they use multi-camera and high-quality pro footage, we have one camera" does
not apply to the two repositories that matter most. Our input is the input they designed for.

## What the pipeline actually contains

Built on **TrackLab**, a modular multi-object-tracking framework configured with Hydra YAML.
Stages, with the models the baseline uses:

| Stage | Model in the baseline |
|---|---|
| Detection | **YOLOv11** |
| Tracking | TrackLab trackers (DeepSORT / StrongSORT / OC-SORT family) |
| Re-identification | **PRTReid**, **BPBreID** |
| Camera calibration / pitch localisation | **TVCalib** (alternatives: PnLCalib, NBJW) |
| Jersey number recognition | **MMOCR** |
| Team affiliation | included |

Requirements: Python 3.9, PyTorch 1.13.1 + CUDA 11.7, MMCV 2.0.1. Runs as
`uv run tracklab -cn soccernet`.

**One real gap:** the repository documents running on *their* dataset. There is no documented
path for running on your own video. Writing that adapter is the first piece of work — and it is
a well-defined, contained task, which makes it a good first ask of a collaborator.

## Licences — read this before building on it

| Component | Licence | What it means for us |
|---|---|---|
| **TrackLab** (the framework) | **MIT** | Permissive. Use freely, including commercially |
| **sn-gamestate** (the plugins) | **GPL-3.0** | Copyleft, but see below |
| SoccerNet **videos** | NDA, non-commercial | Do not train a shipped model on them |
| SoccerNet **annotations** | MIT, with a restriction against exploiting the source videos | Usable, carefully |

**On GPL-3.0 and SaaS.** GPL obligations are triggered by *distributing* the software. Running
it on our own servers and delivering only reports to a customer is not distribution — that is
the well-known difference between GPL and **AGPL**, which does trigger on network use. So a
server-side deployment is the natural position. **Get this confirmed by a lawyer before it
matters commercially** — do not act on this note alone.

**The cleaner option:** TrackLab is MIT. We can build our own plugins on TrackLab and take only
the *methods* from sn-gamestate rather than its code, which avoids the question entirely.

### ⚠ A licence issue in our own stack, unrelated to SoccerNet

**Ultralytics YOLO (v8, v11) is AGPL-3.0**, with a paid enterprise licence sold for commercial
use. AGPL **does** trigger on network use — unlike GPL. If Playbook-IQ serves customers from a
pipeline containing Ultralytics YOLO, that is a real commercial exposure, and it exists today,
before any SoccerNet involvement.

Options: buy the Ultralytics enterprise licence, or move to a detector with a permissive licence.
**Verify the current terms and take legal advice.** Flagging it because it is the kind of thing
that surfaces during investment due diligence, and it is much cheaper to resolve now.

## How to get the maximum out of it

**Do not rewrite our pipeline.** Ours works and has quality instrumentation theirs does not.
Take the pieces that fill our gaps.

1. **Run the baseline on their data first.** Confirm the environment works before changing
   anything. A day.
2. **Take their evaluation metric and score our own pipeline with it.** ⭐ This is the highest-value
   step and the cheapest. It gives us the failure numbers we need — and it converts the KAUST ask
   from "we have gaps" into "here is our score on the public benchmark, and here is where we lose
   points."
3. **Cherry-pick modules against our weaknesses:**
   - **TVCalib** for pitch calibration — directly relevant to our homography, which is our most
     fragile stage.
   - **MMOCR jersey recognition** — feeds the review tool in `REID-REVIEW-TOOL.md`.
   - **PRTReid / BPBreID embeddings** — purpose-built for player appearance, likely stronger for
     clustering than our SigLIP embeddings, which we chose for team classification.
4. **Write the custom-video adapter.** The documented gap. Needed for any of this to touch our
   own footage.
5. **Train on our own footage.** Keeps the commercial position clean and makes our annotation
   work an owned asset.

## Why this matters for the KAUST conversation

We are no longer asking a lab to help with our engineering. We are saying: *we run a Game State
Reconstruction pipeline on Saudi broadcast footage with real club deployment, we have scored
ourselves on the public benchmark, and here is precisely where we lose points.* That is a
research problem with a number attached, which is something an academic can act on.

## Sources
- https://github.com/SoccerNet/sn-gamestate (GPL-3.0)
- https://github.com/TrackingLaboratory/tracklab (MIT)
- https://arxiv.org/pdf/2404.11335 — the GSR paper
