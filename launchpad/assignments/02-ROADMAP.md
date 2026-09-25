# Playbook-IQ — MVP Roadmap
### Misk Launchpad Cohort 10 · Assignment #2 · Sprints to 30 October 2026

> **Working draft — rev. 2, 23 Sep 2026.** Being revised up to the mentor sync on Sunday
> 27 Sep. The version sent to Walid on Saturday is the one that goes into the form.

**Product stage:** Prototype
**Sprint cadence:** 1 week (per mentor guidance, Walid)
**Number of sprints:** 5
**Team:** Founder (full-time) + Data Operations Specialist (part-time)

---

## The goal of these five sprints

Run **one real pilot**: four consecutive matches of a single youth academy team, analysed and
delivered to the coaching staff before each next training session, to answer the question seven
customer interviews did not — **will anyone pay for this?**

**Why four matches, not one.** The product measures **development**, not single-match
performance. Development only appears in **aggregation**, so four consecutive games is the
minimum that produces a trend a coach can act on. A one-match pilot would under-sell what we do
and would not test the actual claim.

---

## How the work is produced — Layer 1, the hybrid

| | |
|---|---|
| **Automatic (the pipeline)** | Detection, tracking and homography → every visible player's position on a 2D pitch, every frame, with a confidence score per row. No player identity required for this stage. |
| **Manual (founder + trainee)** | Assign real identities to tracklets, and mark the events — which frame, which two players. |

The human never measures a position. xP and the other metrics compute from the positions already
on the pitch at the event frame.

**Consequence: pipeline quality sets the pilot's labour budget.** Fixing detection is not a
parallel nice-to-have — it is what makes four matches affordable in five weeks. See
`../DETECTION-PLAN.md`.

---

## Three tracks

| Track | Buys us |
|---|---|
| **A — Pilot** | Evidence. The willingness-to-pay answer we do not have |
| **B — Models** | Cost. Every point of detection recall reduces manual identity work |
| **C — Benchmarking** | Proof. A number comparable to published research, and the basis for a research partnership |

---

## Sprint plan

| # | Dates | Track A — Pilot | Track B — Models | Track C — Benchmark |
|---|---|---|---|---|
| **1** | 28 Sep – 4 Oct | Pilot academy approached, footage secured | Processing capacity secured · CVAT annotations migrated · detector retraining started | GSR benchmark set up, reference baseline reproduced |
| **2** | 5 – 11 Oct | **Match 1** delivered before next training session · hours logged | Retrained detector in place · recall and identity-break rate measured against baseline | Our footage converted to ground truth with verified identities |
| **3** | 12 – 18 Oct | **Match 2** delivered · coach asking his own questions through the Arabic assistant | Pitch model validated on pilot footage | **Pipeline scored against the public benchmark** — first comparable number |
| **4** | 19 – 25 Oct | **Match 3** delivered · first cross-match development trend | Review tooling in use · identity work per match measurably reduced | Failure modes ranked: where we lose accuracy and why |
| **5** | 26 – 30 Oct | **Match 4** + full pilot development report · **a price named to the decision-maker and the reaction recorded** | **Human-hours per match** reported as a measured number | Results written up and taken to a research group as the basis for a first joint problem |

**Every sprint ends in a measured result or something delivered to a real user.**

Running through all five: buyer interviews (academy director, scouting agency owner) and a PDPL
consent note for under-18 player data.

---

## Main dependency — compute for match processing

Training runs on Roboflow (plan upgrade in Sprint 1). The pilot itself needs inference over
**4 matches × 90 minutes × 25 fps ≈ 540,000 frames**, on the order of **10–25 GPU-hours** across
the five weeks. This gates Sprints 2–5.

Routes being pursued: Roboflow hosted batch processing (may cover the detection pass entirely),
cloud startup credits, the programme, and a research partner.

---

## Key planned features

| Feature | Sprint | Track |
|---|---|---|
| Player profiles in the prototype | shipped | A |
| Retrained player detector (RF-DETR) | 1–2 | B |
| Event-tagging schema and match report template | 1 | A |
| Match, team and player analysis on pilot footage | 2–3 | A |
| Arabic assistant answering the coach's own questions | 3 | A |
| Tracklet review tooling (jersey-number assisted) | 3–4 | B |
| Cross-match development trend | 4 | A |
| GSR benchmark score on our own footage | 3–4 | C |
| PDPL consent note for minors | 4 | — |

---

## Beyond 30 October

| | Horizon | What |
|---|---|---|
| **Automatic identity** | Research track | Stable player identity across a match from a single moving broadcast camera. Pursued with a research partner, not solo. The **outcome** is must-have; **automating** it is should-have. |
| **Full automation** | Triggered by demand | Layer 1 capacity is roughly three teams. Automation becomes urgent at **customer #4**, not before. |
| **The research network** | Ongoing | Specialist partners by domain — computer vision, tactical modelling, sports science — with Playbook-IQ holding the club relationships, deployment and product. *"Football by the book."* |

---

## What these sprints deliberately do **not** attempt

Solving player re-identification. It is an open research problem with a public benchmark that
well-funded groups have not closed, and five weeks of founder time would produce neither a
solution nor a customer. It is pursued in parallel through benchmarking and a research
partnership while the pilot tests demand.

---

## Changes in rev. 2

- **Layer 1 corrected to the hybrid.** Rev. 1 described it as manual event tagging, which cannot
  produce xP — positions come from the pipeline.
- **Detection moved to Sprint 1.** Measurement showed the detector misses 27% of players and is
  the root cause of the manual identity work.
- **Benchmarking added as its own track**, replacing "open KAUST" — the benchmark is work we
  control and it makes the lab conversation concrete.
- **Sprint 5 no longer promises a partnership.** It promises a scoped joint problem, which is
  ours to drive.
- **Compute named as the main dependency**, with the frame count and hour estimate.
- **The four-match rationale stated:** development is only visible in aggregation.
