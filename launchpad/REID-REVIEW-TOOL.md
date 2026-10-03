# The tracklet review tool — build spec

*23 Sep 2026. The founder already works at tracklet level and it is still slow. This is why,
and what to build. All numbers measured on `per_frame_tracks_half2_reid.csv` (half 2,
5,958 frames, ~4 minutes).*

## Why tracklet review is still slow

Working at tracklet level removed the frame-by-frame cost. Three costs remain, and they are
the whole problem:

| Cost | What happens now | What it should be |
|---|---|---|
| **No triage** | All ~950 segments get looked at | 706 are under 1 second. Drop them. ~330 remain |
| **Video interface** | Scrub a clip to see who it is | Look at one montage image |
| **Recall, not verification** | "Who is this? Let me follow him" | "Is this #7? y/n" |

The third is the largest. **Recall is slow; verification is fast.** Everything below exists to
turn every human decision into a yes/no on a proposed answer.

## The real size of the task

Measured on half 2:

| | |
|---|---|
| Player-frames | 96,188 |
| Raw tracklet segments | 952 |
| Segments ≥ 0.5s (worth labelling) | **328** |
| Top 140 segments | cover **90.5%** of all player-frames |
| Top 200 segments | cover 95.3% |

So the job is: **assign ~330 tracklets to ~30 identities**, with a team filter halving the
choice set. That is 330 verification decisions, not 96,188 recalls.

## Pipeline

### Stage 0 — Triage
Segment tracks on gaps > 5 frames. Drop segments under ~0.5s; they are mostly detection noise
and contribute little. Sort the rest by length descending and work down — the list has a long
tail that stops mattering.

### Stage 1 — Crop montages
For each tracklet pick 6–8 frames maximising `conf` and bounding-box area, spread across its
duration, avoiding frame edges. Crop with padding, normalise height, tile into one strip.

Reviewing a strip takes seconds. Scrubbing a clip takes a minute. Same decision.

### Stage 2 — Automatic jersey number ⭐ the biggest lever
Run a jersey-number classifier over **every** crop in the tracklet — not just the montage
frames; inference is cheap. Discard predictions below ~70% confidence. **Majority-vote the
survivors.** Store the winning number and its margin.

This is why it works when the human eye fails: the number may be legible in 5% of frames, but
a 500-frame tracklet still yields ~25 clean readings. The model is not sharper than the human —
it looks 500 times where the human looks once.

Start with `SoccerNet/sn-jersey` unmodified and **measure it on our own footage** before
planning around it. The benchmark figure (~90% tracklet accuracy) is on SoccerNet data.

### Stage 3 — Cluster by appearance
Mean the SigLIP embedding (already computed for team classification) over each tracklet.
Cluster within each team. A player's tracklets land together, so the human confirms a group
rather than its members.

### Stage 4 — Constraints that need no human
- **Temporal exclusion:** two tracklets that overlap in time cannot be the same player. Free,
  and it prunes the candidate set hard.
- **Physical plausibility:** a player cannot move faster than ~9 m/s, so a tracklet ending at
  one pitch position can only continue in a tracklet starting nearby soon after. We have
  `x_m, y_m`, so this is computable today.

  ⚠ **Measured, and it is a modest win, not a large one.** On half 2, with a 1-second gap
  window, only **23 of 328** segments had exactly one physically-possible continuation (3 had
  several). Most gaps are longer than a second, so geometry alone bridges few of them. Worth
  implementing because it is nearly free — but **do not plan around it**. Jersey OCR and
  clustering are the real levers.

### Stage 5 — Review UI
- Grid of montages, grouped by proposed (team, number), longest first.
- One keystroke accepts a whole group. One keystroke pulls an outlier out.
- Sort ungrouped tracklets by uncertainty; work down and stop when the rest stop mattering.
- Never show video unless the human explicitly asks for it on one tracklet.

### Stage 6 — Write back, and compound
Rewrite `display_track_id`. **Every confirmed tracklet's crops become training data we own.**
Fine-tune the jersey model on our own footage after the first few matches, so match #4 costs
less than match #1. That compounding does not exist in the current workflow.

## Stopping rule

Do not aim for perfect. Label until the remaining unlabelled tracklets stop changing the
metrics, and record how far that was. Since identity is needed for **aggregation** (per-player
development over matches), the test is whether a player's aggregate moves — not whether every
frame is attributed.

## Build order

1. Triage + montage export. Count the tracklets. Sizes the real job. *(hours)*
2. Run `sn-jersey` as-is. Measure accuracy on our footage. *(a day)*
3. Review UI. Keyboard-driven, grouped, no video. *(days)*
4. Only then time a full match and log **human hours per match**.

## SoccerNet repositories

| Repo | What |
|---|---|
| https://github.com/SoccerNet | The organisation — everything below lives here |
| https://github.com/SoccerNet/sn-gamestate | **Game State Reconstruction** — our pipeline as a benchmark. Includes jersey number recognition and team affiliation. CVPR'24 CVSports |
| https://github.com/SoccerNet/sn-jersey | Jersey number recognition — start here |
| https://github.com/SoccerNet/sn-reid | Re-identification. 340,993 player thumbnails from 400 games |
| https://github.com/SoccerNet/sn-tracking | Multi-object tracking |
| https://github.com/SoccerNet/sn-calibration | Camera calibration — relevant to our homography |
| https://www.soccer-net.org | Task descriptions and challenge results |

**Licensing:** the **videos** are NDA-gated and non-commercial. **Annotations** are MIT but
carry a restriction against reconstructing or commercially exploiting the original videos.
Code repositories are licensed individually — check each. The clean commercial path is to use
their **methods and code**, and train on **our own footage**, which we own outright.
