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

## What can be fully automated TODAY — the honest MVP scope

The insight that unlocks Phase 2: **not every valuable metric needs persistent individual
identity.** Anything computed from detection + team classification + homography alone is
already automatic and already reliable:

- Team territory / occupation heatmaps
- Team centroid, block height, width, compactness over time
- Defensive line height and inter-line distances
- Pressing intensity by zone (players in zone × phase)
- Formation shape and its evolution by match phase
- Space control / pitch-control surfaces
- Phase segmentation (build-up / transition / finishing) at team level

**None of these need Re-ID.** All of them are things a youth-academy coach genuinely
cares about, and none of them exist affordably in this market today.

Individual-player metrics (xT contribution, PR, individual pass networks, the Verified
Asset Ledger) then become the **premium tier delivered with an analyst in the loop** —
which is exactly the "Custom Metric Library" upsell already in the business model.

**Proposed reframe for the MVP Scope Canvas session (Thu 17 Sep):**
> Tier 1 (automated, self-serve): team-level tactical intelligence.
> Tier 2 (lab, analyst-assisted): individual player ledger and custom KPIs.

This is a defensible, honest, deliverable Phase 2 MVP — and it does not shrink the vision,
it sequences it.

## Recommended technical priorities

1. **Homography RANSAC + temporal smoothing.** Cheapest quality win. Days.
2. **Build the Tier-1 team-level metric set.** It is nearly free given what already runs,
   and it turns "prototype" into "product" for Phase 2.
3. **Quantify the automation ratio honestly.** Run the pipeline unattended on one full
   half and measure: mean identity duration before fragmentation, % frames with
   `homography_ok`, ID switches per player per minute. You need these numbers before any
   technical due diligence — and having them makes you the rare founder who does.
4. **Automatic pass detection** (ball possession-change from trajectory + nearest player)
   to replace manual tagging. Harder, but it is what unlocks xT at scale.
5. **Re-ID is the CV-lab ask.** Do not try to solve it solo. It is a research problem
   (jersey-number OCR, long-horizon appearance galleries, tactical-role priors). This is
   correctly an Ask, not a sprint.
