# Mentor sync #3 — Walid
*Sun 27 Sep 2026 · same day as the Assignment #2 deadline, 8pm KSA*

## What this sync is for

Three things have changed since sync #2, and two of them revise advice Walid gave. He should hear
the reasoning and push back **before** Assignment #2 is submitted. This is not a status update —
it is **four decisions and two asks.**

Shape: **5 min update · 25 min decisions · 15 min asks · 5 min close.**

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
