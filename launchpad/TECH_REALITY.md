# Is the pipeline actually applicable? — honest technical assessment

*Written 8 Sep 2026 after reading `main.py`, `vision/`, `geometry/`, and
`notebooks/HANDOVER.md`. This is the answer to the question Muwafag is most
worried about. It is deliberately blunt — it is an internal document, not the pitch.*

## Short answer

**Yes, but not as the product currently described.** The vision is sound and the
foundations are real. The gap is not "does computer vision work on football" — it does.
The gap is that **the deck promises unattended automation, and the pipeline currently
requires a human to produce a clean result.**

That gap is survivable, and the fix is mostly positioning, not code. Details below.

## What works well today

| Layer | State | Verdict |
|---|---|---|
| Player / ball / referee / GK detection | Roboflow model, runs every frame, confidence-scored | **Solid.** Commodity, and it works. |
| Team classification | Jersey-colour + embedding, `vision/teams.py`, `team_memory.py` | **Works.** Heavy (SigLIP) but reliable. |
| Homography → pitch coordinates | 32 pitch keypoints → `cv2.findHomography`, with collinearity + condition-number guards, 15-frame hold-over | **Works, and is genuinely good engineering.** The degeneracy guards are more careful than most open-source attempts. |
| Frame-to-frame tracking | BoT-SORT | **Works short-range.** This is what it's for. |
| Output discipline | Per-frame CSV with `homography_ok`, `kp_used`, `inlier_ratio`, `reproj_err`, `ball_interpolated`, `ball_source`, `detector_ran` | **Better than most commercial products.** Every value carries its own quality flag. This is a real asset — see "The unexpected advantage" below. |

## Where it actually breaks

### 1. Re-ID over a full match — the real bottleneck
`track_id` from BoT-SORT is explicitly "not stable — gets reused/recycled across
different physical people." `IDStabilizer` re-links fragments using position +
appearance embeddings + IoU, and it still fragments over a full match.

The handover states plainly that the 20 canonical player identities were produced by
**manual unification, manual fragment cleanup, and manual ID-swap correction**, verified
by hand against a zero-duplicate invariant.

**Scale check:** the two reviewed windows are frames 1066–3333 and 4270–5720 —
about **3,700 frames total, roughly 2–2.5 minutes of play** at broadcast frame rate.
A 90-minute match is on the order of 135,000 frames. *(Muwafag: confirm whether the
source was a full half or already a clip — this number drives everything below.)*

Manual cleaning does not scale linearly, it scales worse: errors compound and a human
must watch the video to catch a silent identity swap.

**This is the honest reason the Ask asks for a CV lab. That Ask is correct.**

### 2. Event data (passes) is hand-tagged
`pass_id`, `pass_role`, `pass_phase`, `departure_time`, `arrival_time`, `outcome` come
from `tag_passes.ipynb` — **manual annotation**. The ball-gap reconstruction is
*pass-aware*, meaning it depends on those manual tags to work well.

This matters because **xT and pass networks are event metrics.** Slide 4 lists them as
"instant, out-of-the-box." Today they rest on a human tagging every pass.

### 3. Homography stability
Plain per-frame DLT — **no RANSAC, no temporal smoothing.** Hold-over is the only
stabilization. Consequence: pitch coordinates jitter frame to frame, which corrupts any
derived velocity, acceleration, or distance metric. Pressure Resistance in particular
needs stable velocities.

**This is the cheapest big win available.** RANSAC + a temporal filter on the homography
(or on the keypoints) is standard, well-documented work — days, not months.

### 4. Known unresolved gap
HAZ #16 loses identity from frame ~5479 to the end of the half-2 window. No recoverable
raw track. That is a genuine hole in a player's trajectory in the one match that exists.

## The strategic read — and why this is better news than it looks

Muwafag's own words: **"we need to function as a science lab, not a solution provider."**
The PDD calls pillar two **"Fractional Sports Science Labs."**

That positioning is *already* the one that survives an honest technical state.

- A **pure automated SaaS** promise dies on contact with the Re-ID problem. A club uploads
  a match, gets fragmented identities, and churns.
- A **fractional lab** promise does not. A lab is *expected* to have analysts in the loop.
  Human-in-the-loop is not a confession — it is the operating model. Every serious sports
  science provider has humans in the loop; the honest ones say so.

So the recommendation is: **stop describing Playbook-IQ as automation, describe it as a
managed analytical service with a proprietary automation stack that keeps cost low.**
The technology is the margin engine, not the product promise. That framing is true today,
survives due diligence in Phase 4, and is *the same business* — only the automation ratio
changes over time, and improving it is the roadmap.

## The unexpected advantage

Every output row carries `homography_ok`, `inlier_ratio`, `reproj_err`, `conf`,
`ball_interpolated`, `detector_ran`. **The pipeline knows when it is unsure.**

Nobody in this market ships confidence-flagged tracking data. For a product whose core
claim is *"mathematical proof of player progression"* (the Verified Asset Ledger), being
able to state the uncertainty on every number is not a footnote — it is the thing that
makes "proof" defensible to a buying club's analysts. Lead with it.

## ⚠ CORRECTION (22 Sep 2026) — the camera constraint

An earlier version of this file claimed a "Tier 1" of team-level metrics that needed no
Re-ID and ran automatically today. **That claim was wrong**, and the founder corrected it
directly: those metrics require a **tactical (wide / panoramic) camera view**, and
Playbook-IQ currently works from **broadcast footage only**.

Broadcast footage is a fundamentally harder input:

- The camera **pans, zooms and cuts**, so homography must be re-estimated every frame.
- **Never all 22 players are in frame.** Anything computed over the whole team —
  compactness, block height, inter-line distance, pitch control, formation shape — is
  undefined or misleading when a third of the players are off-screen.
- Players **leave and re-enter frame constantly**, which is precisely what makes Re-ID
  from broadcast the hardest version of the identity problem.

So there is **no free automated tier**. The team-level metric set is not "nearly free
given what already runs" — it is blocked on an input Playbook-IQ does not have.

### What this changes

**Re-ID is must-have, not a premium upsell.** Walid was right. And the interviews agree:
clubs have seen the competitor fail on unstable IDs and are uncomfortable buying because
of it. Stable identity is the purchase condition, not a nice-to-have.

**Two distinct routes out, and they are not the same kind of problem:**

| | Route A — solve Re-ID from broadcast | Route B — change the input |
|---|---|---|
| Nature | Research problem | Access / business problem |
| Horizon | Months to years | Weeks, if a club shares footage |
| Mechanism | CV lab partnership (KAUST / SoccerNet) | Get tactical-cam footage from one club |
| Risk | May not resolve in time to sell anything | Depends on someone saying yes |

These are not alternatives — Route B buys the time that Route A needs. A club that already
films its own matches with a fixed wide camera (Veo, Pixellot, or a club-owned tactical
cam) can unlock the team-level metric set in weeks, while the lab partnership works the
identity problem on the broadcast path in parallel.

**KAUST is therefore central, not opportunistic.** Their CV lab, SoccerNet partnership and
FIFA connections sit exactly on the blocking problem. This is the mechanism that makes
"a wide network of researchers is the core of the business" true rather than aspirational.

### What is still genuinely true

The quality instrumentation stands and remains the real differentiator: every output row
carries `homography_ok`, `inlier_ratio`, `reproj_err`, `conf` and `ball_interpolated`.
**The pipeline knows when it is unsure.** Against a competitor whose IDs silently drift,
being able to show a club exactly which frames are trustworthy is a defensible claim —
and it is one no one else in this market is making.

## Recommended technical priorities

1. **Homography RANSAC + temporal smoothing.** Cheapest quality win. Days.
2. **Secure tactical-camera footage from one club** (Route B above). This, not code, is
   what unlocks the team-level metric set. Treat it as a sprint goal with a named target.
3. **Quantify the automation ratio honestly.** Run the pipeline unattended on one full
   half and measure: mean identity duration before fragmentation, % frames with
   `homography_ok`, ID switches per player per minute. You need these numbers before any
   technical due diligence — and having them makes you the rare founder who does.
4. **Automatic pass detection** (ball possession-change from trajectory + nearest player)
   to replace manual tagging. Harder, but it is what unlocks xT at scale.
5. **Re-ID is the CV-lab ask — and it is must-have, not optional.** Do not try to solve
   it solo. It is a research problem (jersey-number OCR, long-horizon appearance
   galleries, tactical-role priors) and clubs have already rejected a competitor over it.
   The sprint goal is **securing the lab partnership** (KAUST), not solving the problem.
