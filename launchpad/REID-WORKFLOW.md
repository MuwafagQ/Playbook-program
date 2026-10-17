# Cutting the manual Re-ID cost
*Research note, 23 Sep 2026. Answers the founder's question: is there a faster way than
eyeballing every player in every frame?*

## The current method, and why it is the bottleneck

Today: watch the output clip, visually follow each player, write down the track ID and the
frame number, repeat for every player across the whole clip, then remap the CSV.

**This labels at the frame level. The correct unit of work is the tracklet.**

A tracker does not emit random IDs per frame — it emits **tracklets**: continuous segments
where identity is already internally consistent. The manual job is not "who is this player in
this frame" but **"which tracklets belong to the same player"**.

For a 3-minute clip that is on the order of a few hundred tracklets, not tens of thousands of
frame-level decisions. The work is not reduced by a little. It changes shape.

## The workflow that replaces it

**1. Work from tracklets, not video.** Export each tracklet as a montage of a few crops (pick
the sharpest, largest, least-occluded frames). Review a contact sheet of images, not a video
timeline. This alone is a large multiplier.

**2. Automatic jersey number recognition, with majority voting across the tracklet.** ⭐

This is an established task with a public benchmark — the **SoccerNet Jersey Number Recognition
challenge** — built on exactly the same input we have: short, low-resolution, motion-blurred
player tracklets from broadcast footage, where the number is visible in only a small subset of
frames. Winning methods classify each crop, keep only predictions above a confidence threshold,
and take a **majority vote over the tracklet**. Reported accuracy: **~90% at tracklet level**
on both the test and challenge sets.

Open starter code: `github.com/SoccerNet/sn-jersey`.

If ~90% of tracklets are auto-labelled with a jersey number, the human only adjudicates the
remainder. **That is the single biggest available win.**

Caveat to verify, not assume: that figure is on SoccerNet's data. Saudi U18 broadcast footage
may be lower quality. Measure it on our own clips before planning around it.

**3. Cluster tracklets by appearance before reviewing them.** We already run SigLIP embeddings
for team classification. The same embeddings cluster a player's tracklets together. The human
then confirms or splits clusters rather than labelling tracklets one at a time.

**4. Review only what is uncertain.** Rank tracklets by confidence and work down the list.
Stop when the remaining ones do not move any metric.

## Can an LLM do this? — an honest split

**No for tracking.** Vision-language models do not perform frame-level multi-object tracking
with persistent identity. They sample frames and reason coarsely; they do not maintain identity
across thousands of frames. **GPT-6 Astra** (OpenAI, September 2026) is currently the strongest
vision model available and leads object detection benchmarks — but that is *per-image detection*,
which is a different task from *identity over time*. It will not replace the tracker, and
Claude Code could not read the output video for the same underlying reason.

**Yes for jersey numbers on crops.** Reading a number from a still image is narrow, visual and
well suited to a VLM. Batch it: assemble a grid of 20–50 tracklet montages with letter labels,
ask for the number in each, allow "unknown". One call covers many tracklets.

**But prefer the specialist first.** A purpose-built jersey-number model from the SoccerNet
challenge will likely beat a general VLM on this exact input and costs nothing per call. The
sensible design is: **specialist model first, VLM only on what it flags as uncertain, human only
on what the VLM also cannot read.** That is the LLM-in-the-loop, placed where it actually helps.

## The larger finding — SoccerNet Game State Reconstruction ⭐

**Our pipeline is an established open research task.** SoccerNet-GSR is defined as recovering
the positions *and identities* of all players from broadcast football video and placing them on
a 2D pitch minimap — which is precisely what Playbook-IQ does.

- **SoccerNet-GSR dataset:** 200 broadcast sequences, **9.37 million line points** for pitch
  localisation and camera calibration, and **over 2.36 million athlete positions** annotated
  with role, team and **jersey number**.
- **An open baseline exists**, built on **TrackLab** (a PyTorch multi-object-tracking framework),
  combining detection (YOLOv8), tracking and re-identification.

Two consequences:

1. **We should be building on this rather than from scratch.** The calibration annotations alone
   address the homography work, and the identity annotations are training data for exactly our
   blocked stage.
2. **This is the KAUST conversation.** KAUST has a SoccerNet partnership. The ask is no longer
   "we have gaps" — it is *"we run a GSR pipeline on Saudi broadcast footage with real club
   deployment; we want to work on identity under these conditions."* That is a research
   collaboration with a named benchmark, which is a proposal an academic lab can act on.

## What this means for the five sprints

The answer to *"should we run detection and then remap IDs manually?"* is **yes — but remap at
tracklet level with jersey-number assistance, never frame by frame.**

Suggested order:
1. Export tracklets as crop montages from an existing clip. Count them. This sizes the real job.
2. Run `sn-jersey` over the tracklets and measure accuracy on our own footage.
3. Build the review tool: contact sheet, keyboard-driven, one keystroke per tracklet.
4. Only then time a full match, and log **human hours per match**.

Steps 1–3 are days of work, not months, and they are the difference between a pilot that fits
in five weeks and one that does not.

## Sources
- SoccerNet Game State Reconstruction (CVPR 2024 workshop): https://arxiv.org/pdf/2404.11335
- SoccerNet Jersey Number Recognition: https://www.soccer-net.org/tasks/jersey-number-recognition
- Starter code: https://github.com/SoccerNet/sn-jersey
- SoccerNet 2023 challenge results (majority-voting accuracy): https://arxiv.org/pdf/2309.06006
- GPT-6 Astra vision evaluation: https://blog.roboflow.com/gpt-6-astra-vision/
