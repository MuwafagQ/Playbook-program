# Mentor sync #3 — Walid
*Sun 27 Sep 2026 · same day as the Assignment #2 deadline, 8pm KSA*

## What this sync is for

Three things have changed since sync #2, and two of them revise advice Walid gave. He should hear
the reasoning and push back **before** Assignment #2 is submitted. This is not a status update —
it is **four decisions and two asks.**

Shape: **5 min update · 30 min decisions · 12 min asks · 3 min close.**

Six decisions will not fit in 30 minutes. **Ranked — if time runs short, drop from the bottom:**
1 (Re-ID) · 5 (tactical camera) · 3 (network model) · 2 (sprints) · 6 (benchmark) · 4 (youth).
**Ask 1 is non-negotiable** — protect the last ten minutes for it.

**Send him the roadmap on Saturday**, so his input can land in Assignment #2 before it is
submitted on Sunday evening.

---

## Update — 5 minutes, no discussion

- **Assignment #1 submitted.** 7 customer interviews: 4 yes, 3 maybe, 0 no.
- **Assignment #2 due today.** Five one-week sprints to 30 Oct.
- **New contacts.** Lizzie Fluke (Director of Performance) introduced Filippo Baldasso — three
  years as Academy Head of Performance U14–U19, and previously sold performance technology into
  Al-Ittihad at Kitman Labs. She is also introducing us at KAUST.
- **Still zero signed clubs.** That is the hole the pilot exists to fill.

---

## Decision 1 — Re-ID: he said prioritise the must-have. The answer changed.

**His advice:** if the gap is a must-have, prioritise it.

**What we found since:** the player detector was trained on **78 images** and has **72.5%
recall** — it misses more than one player in four. Measured on our own tracking output, **883 of
1,802 gaps inside player tracks are only 1–5 frames long.** That is not occlusion. That is the
detector blinking, 883 times in four minutes, and it is where the 952 tracklet fragments — and
the months of manual identity work — come from.

**The reframe to test with him:**

> The **outcome** — stable player identity in the delivered report — is **must-have**.
> Aggregation needs it, development is the value proposition, and clubs have already rejected a
> competitor over unstable IDs.
>
> The **automation** of it is **should-have**. That is cost and scale, not capability.

So: fix detection now, assign identity by hand during the pilot, and put automatic Re-ID with a
research partner. Detection is days of work and fully in our control. Re-ID is months and is not.

**Ask him:** does that distinction hold, or is it a convenient way of dodging the hard problem?

---

## Decision 2 — the five sprints buy evidence, not automation

We cannot ship automation by 30 Oct by any route. So the sprints are a **four-match pilot with
one youth team**, delivered before each next training session, to answer what seven interviews
did not: **will anyone pay.**

Four matches and not one, because at youth level the value proposition is *development*, and
development is invisible in a single match.

Sprint 5 includes **naming a price to a decision-maker and recording the reaction.** Not the
right price — a real reaction.

**Ask him:** he teaches sprints. Is this the right sprint content, and is one week the right
cadence for work that is half fieldwork? Where would he cut?

---

## Decision 3 — the research-network model

Playbook-IQ as **a network of research arms, each owning one domain** — computer vision at KAUST,
tactical modelling at Barça Innovation Hub — with Playbook-IQ holding the club relationships, the
deployment and the product. Tagline: *"Football by the book."*

**Ask him bluntly:** does that read to an investor at Phase 4 as a credible operating model, or
as a solo founder outsourcing the hard parts? He has seen the diligence.

---

## Decision 4 — narrowing to youth

> **SPL coaches use data to win. Youth coaches use it to develop.**

**Ask him:** is narrowing to youth and grassroots right for Phase 2, or does it shrink the story
in front of the Ministry and the judges?

---

## Decision 5 — the tactical camera question

**The constraint:** whole-team metrics — compactness, block height, defensive line, pitch
control — need a **tactical (wide) camera view**. Broadcast never shows all 22 players, so those
metrics are not computable from our current input at all. Broadcast is fine for ball-local events,
which is most of our metric set, but the team-shape layer is locked behind a different camera.

**What makes it hard:** even if an academy already films with a fixed wide camera, their footage
needs **new detection and pitch-keypoint models** — the camera angle, player scale and pitch
geometry are all different. It is weeks of machine-learning work, not a plug-in. And the cameras
that academies own pan, so the cheap shortcut (calibrate the pitch once, reuse it all match) does
not apply.

**The business question for Walid — this is the real one:**

> Do we stay **video-only**, taking whatever footage a club already has, or do we get into
> **hardware** — telling academies to install a camera, or reselling one?

| | Video-only | Camera involved |
|---|---|---|
| Friction to first customer | None — they already have footage | Capex, procurement, installation, permission |
| Data quality | Limited to ball-local events | Unlocks the whole team-shape layer |
| Sales motion | Software sale | Becomes partly a hardware sale |
| Moat | Weak — anyone can take broadcast | Stronger — we are in their stadium |
| Competitors | Veo, Pixellot already sell cameras into academies | We would be reselling or competing with them |

**Ask him:** at our stage, with zero signed clubs, is adding a hardware dependency a distraction
or the thing that makes the product defensible? And does a Saudi academy have the budget and the
authority to install a camera at all?

---

## Decision 6 — KAUST, SoccerNet, and "Game State Reconstruction"

**Explain it to him in one line:** the exact thing we do — turning broadcast football video into
player positions and identities on a 2D pitch — is an established academic research task with a
public name (**Game State Reconstruction**), a public dataset, an open reference implementation,
and an annual competition. **SoccerNet** is the body that runs it. **KAUST is a partner in it.**

That has three consequences worth his view:

**1. Our hardest problem is a known hard problem.** Identity across a full match from one moving
camera is an open research question that well-funded groups have not solved. That reframes it from
"the founder cannot finish his pipeline" to "we are working at the edge of a public benchmark."

**2. We can score ourselves publicly.** Their evaluation metric would give us a number, comparable
to published results.

> **Ask him:** is publishing a benchmark score good or dangerous for us? It is real credibility
> with technical judges and diligence — but if the number is well below the state of the art, we
> have handed a weakness to anyone who asks. Which way does that cut at Phase 4?

**3. It is the natural opening for KAUST.** They already work in this space, so the conversation
starts with shared vocabulary rather than an explanation. Lizzie is introducing us. The ask is not
"help us with Re-ID" — it is "be the computer vision arm of our research network."

> **Ask him:** does he know anyone at KAUST, and would Misk or the Ministry of Sport open that
> door faster than a personal introduction? A programme-backed approach may carry more weight
> than a founder's email — or it may make us look like a programme participant rather than a
> company.

---

## Ask 1 — one academy for the pilot ⭐ the most valuable thing he can give

Sprint 1's only real blocker is finding the team.

Be specific about what is needed: **one youth team, four consecutive fixtures, footage we can
access, and one coach willing to read the report and say whether it changed anything.** Free.

**Does he know anyone? Will he ask?**

---

## Ask 2 — how do we get in front of directors?

Cold outreach is failing: LinkedIn messages to directors go unanswered. The analysts and the
coach we reached came through personal access, not cold contact.

**What worked for founders he has mentored before?** Not "network more" — the specific move.

---

## Housekeeping

- Confirm the **Week 3 feedback form** is submitted (mandatory).
- Agree what he wants to see before sync #4.

---

## Do not raise unless he does

- The detector licence question (AGPL) — real, but a legal task, not a mentoring question.
- Footage provenance — permission is verbal; being handled.
- The public repository.

These belong on the risk register, not in 50 minutes with a mentor whose value is sprints,
credibility and his network.
