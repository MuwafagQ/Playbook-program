# Playbook-IQ — MVP Roadmap
### Misk Launchpad Cohort 10 · Assignment #2 · 28 September – 30 October 2026

> **Rev. 3, 26 Sep 2026.** Sent to Walid ahead of the Sunday sync. This is the version
> intended for submission unless that conversation changes it.

**Product stage:** Prototype
**Sprint cadence:** 1 week · **Number of sprints:** 5
**Team:** Founder (full-time) + Data Operations Specialist (part-time)

---

## The pilot

**Al-Ittifaq U21** — currently top of the table, three points clear of Al-Nassr and Al-Hilal.
Access through their former U18/U21 analyst, now with the first team, who is introducing us to
the U21 staff.

**Four matches:**

| | Match | Status | Report due |
|---|---|---|---|
| 1 | **23 Aug** | already played — analysed retrospectively | Sprint 1 |
| 2 | **31 Aug** | already played — analysed retrospectively | Sprint 2 |
| 3 | **12 Oct** | live | within 48h — Sprint 3 |
| 4 | **18 Oct** | live | within 48h — Sprint 4 |

**Why four.** We measure **development**, not single-match performance. Development only appears
in aggregation, so four games is the minimum that produces a trend a coach can act on.

**Why two are retrospective.** The August matches are a rehearsal with no deadline pressure: we
find out what a report costs us and whether it is any good *before* the live matches, where the
48-hour promise is real. They also give the coach a two-match baseline before he has ever seen
our work.

**What we are not doing:** opposition scouting. This pilot is development analysis — supporting
the coaching staff's decisions about their own players.

**Follow-through:** Al-Ittifaq's fifth match is **2 November**, after this window. It is the
Phase 2 story, not a deliverable here.

---

## How success is measured

After each report, the same three questions to the analyst — so answers compare across matches:

1. Which findings did you already know?
2. Which were new to you?
3. Which changed something you did — a drill, a selection, an individual conversation?

| Metric | Target by 30 Oct |
|---|---|
| **Actions taken** — findings that changed a training decision | **≥ 5 across four reports**, ≥ 1 in every report |
| **Novelty rate** — findings rated new ÷ findings delivered | **≥ 40%** |
| **Turnaround** — match end to report delivered | **< 48 h, four times out of four** |
| **Human hours per match** — our production cost | measured every match, **falling from match 1 to match 4** |

Novelty matters as much as action. If a professional analyst says *"I knew all of this,"* we have
built a nicer version of his own eyes.

### Kill criteria

> **Fewer than 5 actions across the four reports, or no price accepted by any club by
> 30 October → we stop targeting club development squads and reconsider the segment.**

---

## Three tracks

| Track | Buys us |
|---|---|
| **A — Pilot** | Evidence. The willingness-to-pay answer we do not have |
| **B — Models** | Cost. Every point of detection recall reduces manual identity work |
| **C — Benchmark** | Proof. A number comparable to published research |

## Sprint plan

| # | Dates | A — Pilot | B — Models | C — Benchmark |
|---|---|---|---|---|
| **1** | 28 Sep – 4 Oct | U21 staff engaged, review process agreed · **Match 1 (23 Aug) report delivered** | Compute secured · CVAT annotations migrated · detector retraining started | SoccerNet-GSR obtained, reference baseline reproduced |
| **2** | 5 – 11 Oct | **Match 2 (31 Aug) report delivered** · hours logged for both | Retrained detector live · recall and identity-break rate measured against the 952-fragment baseline | Our footage converted to ground truth with verified identities |
| **3** | 12 – 18 Oct | **Match 3 (12 Oct) delivered within 48h** · coach using the Arabic assistant directly | Pitch model validated on pilot footage | **Pipeline scored on the GSR benchmark** — first comparable number |
| **4** | 19 – 25 Oct | **Match 4 (18 Oct) delivered within 48h** · **four-match development trend in the coach's hands** | Review tooling in use · identity work per match measurably reduced | Failure modes ranked: where we lose accuracy and why |
| **5** | 26 – 30 Oct | Full pilot report · **a price named to the decision-maker and the reaction recorded** | **Human-hours per match** reported as a measured number | Results written up and taken to a research group as the basis for a first joint problem |

**Every sprint ends in a measured result or something delivered to a real user.**

Running through all five: reaching the budget holder at the club through the staff, and a PDPL
consent note for under-18 player data.

---

## Main dependency — compute

Training runs on Roboflow (plan upgrade, Sprint 1). The pilot needs inference over
**4 matches × 90 min × 25 fps ≈ 540,000 frames**, on the order of **10–25 GPU-hours** across the
five weeks. This gates Sprints 1–4.

Routes: Roboflow hosted batch processing, cloud startup credits, the programme, a research partner.

---

## MVP Scope Canvas

| | |
|---|---|
| **01 One user** | The U21 coaching staff at Al-Ittifaq — via their analyst, who reads the reports |
| **02 One job** | *"I want to know before the next session which of my players is actually developing, and which isn't."* |
| **03 Riskiest assumption** | That a club will pay for this rather than keep relying on the analyst's eye. Nobody has paid us anything |
| **04 Smallest killing test** | Four consecutive matches for one squad, produced by hand, delivered before the next session. No automation |
| **05 Kill number and date** | Fewer than 5 actions across four reports, or no price accepted, by **30 October** → stop |
| **06 The not-list** | No tactical camera or hardware · no opposition scouting · no first-team product · no mobile app · no automated re-identification. Not this quarter |

---

## Key planned features

| Feature | Sprint |
|---|---|
| Player profiles in the prototype | shipped |
| Retrained player detector (RF-DETR) | 1–2 |
| Development report template and review process | 1 |
| Match, team and player analysis on pilot footage | 1–4 |
| Arabic assistant answering the coach's own questions | 3 |
| Tracklet review tooling (jersey-number assisted) | 3–4 |
| Four-match development trend | 4 |
| GSR benchmark score on our own footage | 3–4 |
| PDPL consent note for minors | 4 |

---

## Beyond 30 October

| | What |
|---|---|
| **Automatic identity** | Open research problem with a public benchmark. Pursued with a research partner. The **outcome** is must-have; **automating** it is should-have |
| **Full automation** | Manual capacity is roughly three squads. Automation becomes urgent at **customer #4** |
| **The research network** | Specialist partners by domain — computer vision, tactical modelling, sports science — with Playbook-IQ holding the club relationships, deployment and product. *"Football by the book."* |

---

## Changes in rev. 3

- **The pilot is named:** Al-Ittifaq U21, with real fixture dates.
- **Two matches are retrospective**, which de-risks the live ones and gives the coach a baseline.
- **Success is now measurable:** a fixed three-question review, four metrics, and a kill number.
- **The MVP Scope Canvas is included** — cells 3, 5 and 6 were missing from rev. 2, and the
  session deck is explicit that a roadmap without kill criteria is a wish list.
- **Opposition scouting moved to the not-list.** Considered and rejected for this pilot: it is a
  different job from development and would double the per-match cost.
- **ICP shifted** from grassroots academies to **development squads at professional clubs**.
