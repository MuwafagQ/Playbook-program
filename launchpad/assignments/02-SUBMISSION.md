# Assignment #2 — Submission text

Misk Launchpad Cohort 10 · due Sunday 27 September 2026, 8 PM KSA
Submitted alongside `Playbook-IQ-MVP-Roadmap.pdf` (attached in the roadmap field).

Detail behind every claim here is in `02-ROADMAP.md`, `../PRODUCT-DECISIONS.md` and
`../TECH_REALITY.md`. This file is the exact text pasted into the form.

---

## Product stage

Prototype

---

## Cell 1 — Your current product / MVP status

Prototype, running on real match footage.

The computer-vision pipeline works stage by stage — player detection, pitch keypoints,
tracking, event tagging — and we are completing the first end-to-end run this week. Two
detection models are trained on our own annotated data.

The dashboard is live: match, team and player analysis views, momentum notifications during a
match, player profiles going in now, and **KEMO**, an assistant that answers a coach's questions
about his own match in Arabic or English. The whole interface is bilingual.

Not ready yet: the backend API and authentication, so the dashboard is not yet multi-user. The
metrics currently shown are demonstration values — in the pilot we will compute a chosen set for
real (**xP, pass completion, space gained, bypass**) and keep the rest visible as upcoming
features, so we collect feedback on the full picture.

Every output row carries a confidence value. We are benchmarking the pipeline against published
research before retraining.

---

## Cell 2 — Product progress during the Misk Launchpad programme

**Yes.**

**New features:** player profiles being added to the dashboard; momentum notifications; the
Arabic assistant answering coaches' own questions on their own match.

**Bugs and quality:** we measured our player detector honestly for the first time and found its
recall is the real bottleneck — most of our identity errors come from the detector missing
players, not from the tracker. Retraining is underway on a larger annotated dataset.

**Technical decisions:** we moved automatic re-identification from must-have to should-have after
establishing that it is an open research problem with a public benchmark, not an implementation
gap. We will score our pipeline on that benchmark before and after retraining, so improvement is
measured rather than assumed.

**Commercial progress:** the biggest change is not code. We defined the ICP — club development
squads, not first teams — designed a pilot around four clubs, secured access to our anchor club
through their analyst, and opened a route toward a research partnership with KAUST's FIFA
Research Institute. Per our mentor's advice we are anchoring on results, not tasks: the
measurable outcome this month is clubs reading our reports, not features shipped.

---

## Cell 3 — Next MVP sprint goals

**Five one-week sprints, 28 September to 30 October.**

One pilot: three matches analysed, five reports, four clubs. Al-Ittifaq U21 is the anchor; the
others are approached with a finished report in hand. Live matches on 12 and 18 October, each
delivered within 48 hours. Alongside it we retrain the player detector and score our pipeline on
the public Game State Reconstruction benchmark — before and after — so the improvement is
measured, not assumed. Sprint 5 ends with a price named to a decision-maker and the reaction
recorded.

**Kill criteria:** fewer than 2 of the 4 clubs ask for a second report, and none asks what it
would cost, by 30 October → we switch to testing the recruitment report with club management.

---

## Cell 4 — Product vision, 12 months

In 12 months, Playbook-IQ delivers fast, reliable player-development analysis to paying Saudi
clubs and academies within 48 hours of a match — built on a network of research partners rather
than an in-house team. *"Football by the book."*

---

## Cell 5 — Why we can execute the roadmap and vision

**Access is the scarce thing, and we have it.** Our anchor club came through its former U21
analyst, now with the first team. We reach decision-makers through the staff who work with the
data, which is how this market actually opens. Cold outreach does not get past the gate; a
finished report does.

**We speak both languages.** Founder background in geospatial engineering — coordinate systems,
projection, spatial measurement — which is exactly the maths underneath turning a camera view
into positions on a pitch, plus a Barça Innovation Hub diploma in football analysis. The models,
the pipeline and the dashboard were built in-house, so we can promise a turnaround and know what
it costs us.

**We change our mind on evidence.** In the last three weeks we moved re-identification out of the
MVP, cut the pilot from four matches at one club to three matches across four clubs, and found
that our identity errors were really a detection problem. Each was a measurement changing a plan,
not an opinion.

**We do not intend to build everything.** Playbook-IQ holds the club relationships, the
deployment and the product; specialist research partners handle the hard science. That keeps a
two-person team credible against companies with engineering departments.

---

## Cell 6 — Note (optional)

Our largest open dependency is GPU access for match processing. We train on Roboflow, but
inference over the pilot footage — roughly 540,000 frames — needs compute we have not secured
yet. Any route the programme can open (cloud credits, a research partner) directly accelerates
the pilot.

---

## Roadmap link field

"Attached below" — the one-page `Playbook-IQ-MVP-Roadmap.pdf` is uploaded, no external link.
