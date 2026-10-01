# Running task log

Status: `todo` / `doing` / `done` / `blocked`
Last reviewed: **1 Oct 2026** — Sprint 1 of 5, day 4 of 7.

---

## Where we are

| | |
|---|---|
| **Programme** | Misk Phase 1 complete. **Filtration 27 Sep – 17 Oct** (250 → 125). Phase 2 starts 18 Oct |
| **Assignments** | #1 and #2 both submitted |
| **Pilot** | Sprint 1 of 5. Three matches, five reports, four clubs. Kill date **30 Oct** |
| **Clubs signed** | **0.** Al-Ittifaq U21 access via their analyst; three cold doors still unopened |
| **Pricing evidence** | **None**, after seven interviews. Still the riskiest gap |
| **SPIN accelerator** | **Declined** — 5 Oct start collides with the pilot, and their connections are reachable via the MoS member anyway. Watch for the next cohort |

---

## Sprint 1 — 28 Sep to 4 Oct

| # | Task | Track | Owner | Status |
|---|---|---|---|---|
| 1 | **Full-match end-to-end run with manual tagging** | B | Muwafag | **doing** — running now |
| 2 | Log human hours for the run, split: machine · tagging · identity fixing | A | Muwafag | todo — capture while it runs |
| 3 | Record where the pipeline broke and how often → retraining priority order | B | Muwafag | todo |
| 4 | **Report template agreed with the Al-Ittifaq analyst** | A | Muwafag | **doing** — conversation open, to be continued |
| 5 | Identify the three cold clubs and the route into each | A | Muwafag | todo |
| 6 | **GSR "before" score** — official GS-HOTA on the validation set | C | Muwafag | **doing** — 5 sequences run, internal KPIs only; official metric not yet computed |
| 7 | CVAT annotations migrated to Roboflow | B | Muwafag | todo |
| 8 | Roboflow plan upgrade (needed for RF-DETR NAS training) | B | Muwafag | todo |
| 9 | Secure GPU for match processing — ~10–25 GPU-hours across the pilot | B | Muwafag | **todo — gates sprints 1–4** |

---

## Sprints 2–5

| Sprint | Dates | Main deliverable |
|---|---|---|
| **2** | 5 – 11 Oct | Rehearsal on the 31 Aug match, **both team reports**, hours logged. Detector retrained and live |
| **3** | 12 – 18 Oct | **12 Oct match, both reports within 48h.** First cold door knocked. GSR re-scored — the "after" |
| **4** | 19 – 25 Oct | **18 Oct match, both reports within 48h.** Second cold door knocked. Reactions collected |
| **5** | 26 – 30 Oct | Four clubs' reactions compiled. **A price named to a decision-maker** and the reaction recorded |

Full detail in `assignments/02-ROADMAP.md`.

---

## Outreach — open threads

| # | Task | Owner | Status |
|---|---|---|---|
| 10 | Thank Lizzie Fluke for the IP advice | Muwafag | todo |
| 11 | Ask Lizzie for the KAUST introduction (`outreach/kaust-ask-lizzie.md`) | Muwafag | todo |
| 12 | Ahmad Sait, KAUST-FIFA Research Institute — contact, or wait for Lizzie's route | Muwafag | todo — decide which |
| 13 | Filippo Baldasso (Catapult) video call (`outreach/call-filippo-baldasso.md`) | Muwafag | todo |
| 14 | Al-Ittifaq fixtures — confirm the 12 and 18 Oct opponents | Muwafag | todo — blocks cold-club identification |
| 15 | MoS member — the route to the three cold clubs | Muwafag | todo — ask early, a delay here costs the window |
| 16 | **Pricing evidence** — still zero after seven interviews | Muwafag | todo — Sprint 5 names a price |

---

## Product and method decisions to land

| # | Task | Owner | Status |
|---|---|---|---|
| 17 | Add the **Sean Ellis question** to the post-report interview — *"how would you feel if you could no longer use this?"*, share answering "very disappointed" | Muwafag | todo — our only product-market-fit measure |
| 18 | Find a **behavioural** signal to sit beside the three self-reported questions | Muwafag | todo — the MLP deck's warning |
| 19 | Report template: development vs recruitment variants, no within-squad leaderboard | Muwafag | in #4 |
| 20 | Check Al-Ittifaq U21 squad for under-18s | Muwafag | todo — decides whether PDPL bites in the pilot |
| 21 | PDPL consent note for minors | Muwafag | Sprint 4 |

---

## Carried over — unresolved

| # | Task | Owner | Status |
|---|---|---|---|
| 22 | ⚠ **Repo is public** with commercial and strategic content. Make it private or split `launchpad/` out | Muwafag | **urgent, raised 5×** |
| 23 | ⚠ Three Roboflow projects are public | Muwafag | **urgent** |
| 24 | ⚠ Dashboard front-end source is in no version control | Muwafag | **urgent** |
| 25 | Agree the trainee's scope, hours and compensation in writing | Muwafag | todo |
| 26 | Trainee onboarding: read `HANDOVER.md`, annotate one clip | Trainee | todo |
| 27 | Ultralytics AGPL-3.0 exposure in the shipped stack — decide: replace, or buy the Enterprise licence | Muwafag | todo — before any paid deployment |
| 28 | SAIP trademark registration, ~6,500 SAR | Muwafag | todo — production phase, not now |

---

**Critical path now: the full-match hours number, the report template with the analyst, and
the GSR "before" score. Everything in Sprints 2–5 is measured against those three.**

## Session log
- **8–17 Sep 2026** — `launchpad/` set up; programme, PDD and Phase 1 curriculum ingested.
  **Assignment #1 completed** — seven interviews, JTBD canvas, value proposition rewritten
  around the research/metric engine. `REGULATORY.md`, external one-pager, buyer interview
  guide. Competitor identified: StepOut.ai.
- **22–27 Sep 2026** — Re-ID investigated down to tracklets and found to be substantially a
  **detection** problem (883 of 1,802 gaps are 1–5 frames; detector recall 72.5% on 78 source
  images). SoccerNet **Game State Reconstruction** identified as the field's name for our whole
  pipeline, with a public benchmark and KAUST as a partner. Pilot redesigned from 4 matches ×
  1 club to **3 matches × 4 clubs** on the mentor's advice. Report-template principles agreed
  (profiles not rankings; no within-squad leaderboard; comparison belongs in the recruitment
  tier). `PRODUCT-DECISIONS.md` opened as the standing record. **Assignment #2 submitted.**
- **30 Sep – 1 Oct 2026** — **SPIN accelerator declined** (5 Oct start collides with the pilot;
  its connections are reachable another way). Full-match end-to-end run started. Week 1 session
  decks and the PDD **recovered** from the container's upload directory — they had never been
  committed. `W2-MVP-to-MLP.pdf` digested: adopt the Sean Ellis question, and the Microsoft /
  86-DOS pre-sell precedent supports our disclosure position.
