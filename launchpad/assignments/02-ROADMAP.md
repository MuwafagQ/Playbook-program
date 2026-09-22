# Playbook-IQ — MVP Roadmap
### Misk Launchpad Cohort 10 · Assignment #2 · Sprints to 30 October 2026

**Product stage:** Prototype
**Sprint cadence:** 1 week (per mentor guidance, Walid)
**Number of sprints:** 5
**Team:** Founder (full-time) + Data Operations Specialist (part-time)

---

## The goal of these five sprints

Run **one real pilot**: 3–4 consecutive matches of a single youth team, analysed and
delivered to the coaching staff before each next training session, to answer the one
question seven customer interviews did not — **will anyone pay for this?**

Analysis is produced at **Layer 1 (manual event tagging)**. The metrics that define the
product are mostly event-based, so Layer 1 delivers the real product, not a mock-up.
See `../MVP-STRATEGY.md`.

**Why 3–4 matches and not one:** the value proposition at youth level is *development*,
and development is invisible in a single match. Only a trend across fixtures tests the
real claim.

---

## Sprint plan

| # | Dates | Goal | Done when |
|---|---|---|---|
| **1** | 28 Sep – 4 Oct | Lock the pilot. Select the academy and the coach/analyst. Freeze the event-tagging schema and the report template. Open the KAUST conversation. | Pilot team confirmed, footage access confirmed, tagging schema written, KAUST contact reached |
| **2** | 5 – 11 Oct | **Match 1** tagged and delivered before the next training session. Measure human-hours per match. | Report delivered on time; hours logged; first coach feedback captured |
| **3** | 12 – 18 Oct | **Match 2** delivered. Coach uses KEMO to ask questions in Arabic. Measure the automation baseline on one unattended half. | Report delivered; KEMO session held; ID-switch rate, `homography_ok` %, mean identity duration recorded |
| **4** | 19 – 25 Oct | **Match 3** delivered. First cross-match view: the same players and team across three fixtures. | Trend view delivered; coach asked what he would change in training |
| **5** | 26 – 30 Oct | **Match 4** + development report across the pilot. Willingness-to-pay conversation with the academy's decision-maker. KAUST meeting held. | Pilot report delivered; a price named and a reaction recorded; KAUST meeting complete |

**Running through all five sprints:** buyer interviews (academy director, scouting agency
owner), and a PDPL consent note for under-18 player data.

---

## Key planned features

| Feature | Sprint | Layer |
|---|---|---|
| Event-tagging schema (ball events + positional notes) | 1 | 1 |
| Match report template | 1 | 1 |
| Match analysis | 2 | 1 |
| Team analysis (collective player data) | 2–3 | 1 |
| Player analysis | 3 | 1 |
| KEMO question-answering over the delivered analysis, in Arabic | 3 | 1 |
| Cross-match development trend | 4 | 1 |
| Automation baseline measurement (ID switches, homography confidence) | 3 | 2 (research) |
| PDPL consent note for minors | 4 | — |

---

## Beyond 30 October

| | Horizon | What |
|---|---|---|
| **Layer 2 — Tracking** | Research track, opened now | Stable player identities from broadcast footage. Blocked on Re-ID, which clubs have already rejected a competitor over. Pursued as a **CV lab collaboration** (KAUST — computer vision lab, SoccerNet partnership, FIFA connections), not solo. |
| **Layer 3 — Automation** | Triggered by demand | Layer 1 capacity is ~3 teams. Automation becomes urgent at **customer #4**, not before. |

---

## What these sprints deliberately do **not** attempt

Solving Re-ID. It is a research problem measured in months to years, and five weeks
of founder time spent on it would produce neither a solution nor a customer. It is
pursued in parallel through a research partnership while the pilot tests demand.
