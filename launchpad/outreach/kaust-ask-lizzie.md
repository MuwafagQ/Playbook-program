# The KAUST ask — note to Lizzie Fluke + forwardable brief

*Drafted 23 Sep 2026. Lizzie offered to pass a request to her KAUST connections.*

## Why this matters beyond Re-ID

Playbook-IQ's positioning is **a bridge between research labs and football teams** — the
researcher network is the core of the business, not a workaround. So the KAUST relationship is
**the first instance of the business model working**, not a favour to fix one technical problem.

Three consequences for how we approach it:

1. **Pursue the relationship, not the fix.** Do not open with our bottleneck.
2. **Do not gate it on having a benchmark score.** Get the meeting now; the number makes the
   *second* conversation concrete.
3. **It is an asset the day it exists.** "A research partnership with KAUST" is a credential a
   club, the Ministry and an investor all understand — before a single line of code improves.

---

## 1 — The note to Lizzie

> Hi Lizzie,
>
> Thank you again — the Catapult introduction is already useful.
>
> You mentioned you have strong connections at KAUST. I've written the ask below as something
> you can forward directly, so it costs you one click rather than an explanation. Please change
> anything that doesn't sound right to you.
>
> The short version: their computer vision group works on the exact problem we're stuck on, and
> they have the SoccerNet partnership. What I can offer them is something academic labs rarely
> get — a working pipeline deployed with real Saudi clubs, and users who will say whether the
> output is actually useful.
>
> No rush, and no obligation if it isn't the right moment.
>
> Muwafag

---

## 2 — The forwardable brief

**Subject: Research collaboration — athlete tracking and identification in Saudi football broadcast**

> I lead Playbook-IQ, a Saudi sports-analytics venture currently in the Misk Launchpad
> accelerator, on the SportTech track run with the Ministry of Sport.
>
> We run a computer vision pipeline over broadcast football footage — player detection, tracking,
> pitch calibration and homography — producing player positions and tactical metrics on a 2D
> pitch. It is deployed with Saudi clubs and youth academies today.
>
> **Where we are stuck is the problem your field knows as Game State Reconstruction: maintaining
> stable player identity across a match from a single moving broadcast camera.** Our positions
> and calibration are sound; identity fragments, and it is currently repaired by hand.
>
> I would like to explore a research collaboration. What I think we can offer is the part that is
> usually hardest for a lab to obtain:
>
> - **A deployment site.** A working production pipeline, not a benchmark script.
> - **Real evaluation.** Clubs and coaches who will say whether an output is useful, not just
>   whether a metric improved.
> - **Saudi broadcast conditions** — a domain the public datasets do not cover.
> - **Annotated ground truth from our own footage**, which we own outright and can share freely.
> - **Openness.** I am happy to share the pipeline, the models and the data with collaborators,
>   and to co-author and publish.
>
> A well-scoped first step exists: the public SoccerNet Game State Reconstruction baseline has no
> documented path for running on footage outside its own dataset. Building that adapter and
> scoring our pipeline against the benchmark is a contained piece of work that would immediately
> tell both of us where the real gap is.
>
> Would a short conversation be of interest? I can come to you.
>
> Muwafag Hussain — Founder, Playbook-IQ
> [email] · [LinkedIn]

---

## Before sending — settle these

- **Intellectual property.** Agree it in writing *before* any work starts. Universities have
  tech transfer offices and default positions. With no budget, the realistic shape is a
  collaborative agreement with a co-authored paper plus an explicit licence to use the outputs —
  and the commercial moat stays in our own data and product layer, not in the research output.
- **Publication is the commercialisation route, not a threat to it.** A published method is one
  we can implement freely. Say yes to publishing.
- **Do not promise SoccerNet data.** Their videos are NDA-gated and non-commercial. What we offer
  is *our* footage and *our* annotations.
- **Ask for a named person, not "KAUST".** A lab with a PI and a research direction.
