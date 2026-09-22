# MVP Strategy — the three-layer model

*Agreed with the founder, 22 Sep 2026. This supersedes the "Tier 1 needs no Re-ID"
reasoning previously in `TECH_REALITY.md` (now corrected there).*

## The reframe

**The product is the metric layer, not the data layer.**

How the underlying data is produced is an implementation detail that can change without
the customer noticing. That gives Playbook-IQ a product definition that survives Re-ID
not being solved.

| Layer | How data is produced | Status |
|---|---|---|
| **1 — Manual events** | Founder + Data Operations Specialist tag ball events and positional notes by hand | **Now.** ~12 person-hours per match |
| **2 — Tracking** | Automated player tracking with stable identities | Blocked on Re-ID; research track |
| **3 — Full automation** | End-to-end, unattended | After layer 2 |

Two facts make layer 1 viable rather than a stunt:

1. **The metrics that define the product are mostly event-based.** Layer 1 is not a hollow
   proxy for the real product — it *is* the real product, produced slowly.
2. **12 person-hours fits the promise.** The turnaround a youth coach cares about is
   *before the next training session* (~48h). Twelve hours fits inside that comfortably.

## Capacity, and when automation becomes urgent

Two people part-time ≈ 40 hrs/week ≈ **three matches per week**. A grassroots team plays
about one match a week, so layer 1 serves roughly **three teams**.

**Automation is scheduled by demand, not ambition: it becomes urgent at customer #4.**

## Positioning

> **SPL coaches use data to win. Youth coaches use it to develop.**

The current focus is the youth / grassroots side. This has a product consequence:

**Development is invisible in a single match.** A one-match report can show performance
but not development. Any pilot that tests the real value proposition must run **3–4
matches of the same team** so the coach sees a trend, not a snapshot.

## Product shape

Four components, with KEMO as the interface rather than a fifth feature:

- **KEMO** — Arabic/English conversational analyst. The coach asks; KEMO answers from the
  layers below and can deliver the reports itself. For a youth coach with no time, chat
  beats a dashboard login.
- **Match analysis**
- **Team analysis** (including collective player data)
- **Player analysis**

## On disclosure

How the output is produced is Playbook-IQ's business, not the customer's. What is promised
is **the output and the processing time** — and only what can actually be delivered.
Do not put capability in a contract that does not exist yet. Automation and the hybrid
approach are described as **roadmap**, not as current capability.

## Open risks

- **Willingness to pay: still zero evidence** after seven interviews. This is the riskiest
  assumption and what the pilot exists to test.
- **Unit economics unknown.** Track **human hours per match** — it is the number that
  determines both what turnaround can be promised and what the price can be.
- **Grassroots means minors.** Individual player analysis on under-18s, shared with coaches
  and possibly parents, falls under PDPL. See `REGULATORY.md`. Academies will ask; having
  a consent answer ready is a credibility asset.
- **Footage is not a moat.** The Ministry platform is open to all companies. Playbook-IQ's
  asset with a research lab is club access and a deployment site, not the data.
