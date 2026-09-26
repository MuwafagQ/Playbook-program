# Playbook-IQ — MVP Roadmap
### Misk Launchpad Cohort 10 · Assignment #2 · 28 September – 30 October 2026

> **Rev. 4, 26 Sep 2026.** Sent to Walid ahead of the Sunday sync. This is the version
> intended for submission unless that conversation changes it.

**Product stage:** Prototype
**Sprint cadence:** 1 week · **Number of sprints:** 5
**Team:** Founder (full-time) + Data Operations Specialist (part-time)

---

## The pilot — four clubs, three matches

**One match analysed produces two team reports.** That is the whole design: three matches,
five reports, **four different clubs**, for the work of three.

| Match | Date | Reports | Purpose |
|---|---|---|---|
| Rehearsal | **31 Aug** (already played) | Al-Ittifaq + opponent | No deadline. Learn what a report costs and whether it is any good |
| Live 1 | **12 Oct** | Al-Ittifaq + opponent | Delivered within 48h |
| Live 2 | **18 Oct** | Al-Ittifaq + opponent | Delivered within 48h |

**Al-Ittifaq U21 is the anchor** — top of the table, three points clear, access through their
former U18/U21 analyst who is now with the first team. They receive three reports, which also
demonstrates the development tier without us having to sell it.

**The three opponent clubs are cold doors, opened with a finished report.** We produce the
analysis first and approach with it in hand. A real report gets past a gatekeeper that a message
cannot. Routes in: Misk, and Salman at the Ministry of Sport. **If a club will not take the
meeting, that is a finding, not a failure** — it tells us what reaching this buyer actually costs.

**Why one match per club and not four.** Per mentor guidance: one match is enough to test whether
the analysis is worth anything. Four matches tests whether *development tracking over time* works
— which is a **later service tier**, not the thing to validate first. Four independent clubs
reacting beats one friendly club reacting four times.

**What wins is the data and the metrics**, not the volume. Fewer matches, better analysis in each.

**Follow-through:** Al-Ittifaq's next fixture is **2 November**, after this window. Phase 2 story.

---

## How success is measured

After each report, the same three questions to whoever reads it:

1. Which findings did you already know?
2. Which were new to you?
3. Which changed something you did?

| Metric | Target by 30 Oct |
|---|---|
| **Clubs reached** — report delivered and read | **4 of 4** |
| **Actions taken** — findings that changed a decision | **≥ 1 per club that engages** |
| **Novelty rate** — findings rated new ÷ findings delivered | **≥ 40%** |
| **Turnaround** — match end to report delivered | **< 48 h on both live matches** |
| **Human hours per match** | measured every match, **falling** |

### Kill criteria

> **Fewer than 2 of the 4 clubs ask for a second report, and none accepts any price, by
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
| **1** | 28 Sep – 4 Oct | Report template agreed with the Al-Ittifaq analyst · target clubs and routes in identified | Compute secured · CVAT annotations migrated · detector retraining started | SoccerNet-GSR obtained, reference baseline reproduced |
| **2** | 5 – 11 Oct | **Rehearsal: 31 Aug match, both team reports.** Hours logged, analyst critique captured | Retrained detector live · recall and identity-break rate measured against the 952-fragment baseline | Our footage converted to ground truth with verified identities |
| **3** | 12 – 18 Oct | **Match 12 Oct: both reports within 48h.** Al-Ittifaq delivered; first cold door knocked | Pitch model validated on pilot footage | **Pipeline scored on the GSR benchmark** — first comparable number |
| **4** | 19 – 25 Oct | **Match 18 Oct: both reports within 48h.** Second cold door knocked. Reactions collected | Review tooling in use · identity work per match measurably reduced | Failure modes ranked |
| **5** | 26 – 30 Oct | Four clubs' reactions compiled · **a price named and the reaction recorded** | **Human-hours per match** reported as a measured number | Results written up and taken to a research group |

**Every sprint ends in a measured result or something delivered to a real user.**

Running through all five: reaching the budget holder at the club through the staff, and a PDPL
consent note for under-18 player data.

---

## The report template — design principles

Agreed in Sprint 1, before the first report is produced. These are framing choices, not
compliance work, and they are expensive to unwind once a club has seen version one.

**No within-squad leaderboard in the development report.** Ranking a club's own players 1-to-N
invites a cut list; a development profile invites a coaching conversation. Same data, different
consequence. Show each player against their own previous form and against positional norms —
not against their team-mates in a table.

**Comparison is the point in the recruitment report.** A separate report type, for management.
Evaluating a player you might sign requires comparing him to others, and that is the tier
management actually pays for. The subject there is someone you might buy, not someone you might
release — so the leaderboard logic that is wrong in one product is the whole product in the other.

**Every report states its basis.** What it measures, what it does not, and the sample it rests on
(one match, in this pilot). Our pipeline already carries a confidence value on every output row —
surfacing that is a differentiator, not a weakness.

**Supporting buy and sell decisions is deliberate.** It is a primary reason the product exists and
the highest-value thing management buys. It also raises the stakes of being wrong, so it needs
written agreements with clubs covering scope, use and liability before any report is used that
way commercially. **Worth asking whether Misk or the Ministry of Sport can help with a standard
framework** — building that institutional layer is part of why they back startups in this sector.

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
| **01 One user** | The U21 analyst or coach at a Saudi professional club — starting with Al-Ittifaq, whose analyst reads our reports |
| **02 One job** | *"I want to know before the next session which of my players is actually developing, and which isn't."* |
| **03 Riskiest assumption** | That a club will pay for this rather than keep relying on the analyst's eye. Nobody has paid us anything |
| **04 Smallest killing test** | Three matches, five reports, four clubs — produced by hand and delivered unsolicited. No automation, no product |
| **05 Kill number and date** | Fewer than **2 of 4 clubs** ask for a second report, and none accepts any price, by **30 October** → stop |
| **06 The not-list** | No tactical camera or hardware · no opposition scouting · no first-team product · no mobile app · no automated re-identification · **no multi-match development tracking — that is the next tier, not the MVP**. Not this quarter |

---

## Key planned features

| Feature | Sprint |
|---|---|
| Player profiles in the prototype | shipped |
| Retrained player detector (RF-DETR) | 1–2 |
| Report template — development and recruitment variants | 1 |
| Match, team and player analysis on pilot footage | 1–4 |
| Arabic assistant answering the coach's own questions | 3 |
| Tracklet review tooling (jersey-number assisted) | 3–4 |
| Cross-match comparison for the anchor club | 4 |
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

## Changes in rev. 4 — after the mentor sync

- **One match per club, not four matches at one club.** Walid's call: one match is enough to test
  whether the analysis is worth anything. Multi-match development tracking is a **later service
  tier**, now named on the not-list so it cannot creep back in.
- **Four clubs, three matches.** One match analysed yields two team reports, so the freed capacity
  buys breadth instead of depth. Four independent reactions beat one friendly club's four.
- **Cold doors opened with a finished report** rather than a request for a meeting. Routes in:
  Misk and the Ministry of Sport. A club that will not take the meeting is itself a finding.
- **The kill number is now a conversion number:** 2 of 4 clubs asking for a second report.
- **The data and the metrics are the winning points**, so the technical sprints serve depth of
  analysis rather than volume of matches.

## Earlier changes (rev. 3)


- **The pilot is named:** Al-Ittifaq U21, with real fixture dates.
- **Two matches are retrospective**, which de-risks the live ones and gives the coach a baseline.
- **Success is now measurable:** a fixed three-question review, four metrics, and a kill number.
- **The MVP Scope Canvas is included** — cells 3, 5 and 6 were missing from rev. 2, and the
  session deck is explicit that a roadmap without kill criteria is a wish list.
- **Opposition scouting moved to the not-list.** Considered and rejected for this pilot: it is a
  different job from development and would double the per-match cost.
- **ICP shifted** from grassroots academies to **development squads at professional clubs**.
