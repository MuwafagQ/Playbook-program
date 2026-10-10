# Session handover — MISK Launchpad C10
Last updated 17 Sep 2026, after Assignment #1 was completed.

## Who you are talking to
The team is **two people**, sharing one account:
- **Muwafag Hussain** — founder. Strategy, customers, metrics, pricing, the programme.
- **A Data Operations Specialist** — first team member, telecoms-engineering background, no
  longer in that field, **does not program**. Owns annotation, data quality measurement, and
  research/outreach support. See `ROLE-data-operations-engineer.md` and `ONBOARDING-TEAM.md`.

**Establish which of them you are speaking with at the start of a session** — their needs differ.
The working agreement below is Muwafag's.

## Working agreement
- Muwafag wants a co-pilot through the whole programme — sessions, 1:1s, assigned tasks — not
  one-off deliverables.
- He asks to be **questioned a lot**, to surface his own perspective. Ask before assuming.
- Anything worth remembering must be committed here — containers are ephemeral.
- Branch: `claude/setup-gpu-video-testing-JhgUH`.
- He prefers **short answers**.

## Read these, in order
1. `CONTEXT.md` — the venture
2. `BUSINESS_MODEL.md` — pricing, buying committee, competitors, KEMO as the differentiator
3. `REGULATORY.md` — Saudi Sports Law M/121, PDPL, the FCMM maturity model, the LDI
4. `interviews/CONSOLIDATED.md` — all seven interviews and the cross-cutting patterns
5. `assignments/01-SUBMISSION.md` — what was submitted for Assignment #1
6. `PROGRAM.md` and `OBLIGATIONS.md` — timeline and mandatory deadlines
7. `TASKS.md` — what is live now
8. `TECH_REALITY.md` — honest assessment of the pipeline
9. `../notebooks/HANDOVER.md` — the technical data documentation

## Where things stand (17 Sep)
**Phase 1, week 2 of 3.** Assignment #1 done. Assignment #2 due **27 Sep, 8 PM KSA** — MVP
Roadmap & Product Vision, drafted in `assignments/02-mvp-roadmap.md`.

**Seven interviews completed.** Five technical users, one buyer, one expert informant.
Counts: 4 yes / 3 maybe / 0 no.

**What the interviews established**
1. Every club has game analysts; none has a data analyst.
2. No club can get a new hire approved — at any budget level. **This makes KEMO load-bearing.**
3. The gap is infrastructural, not analytical. It appears at every level of the market, including
   the best-funded clubs. **It is not a budget problem.**
4. Sporting directors hold the power and often lack performance expertise — so the product must
   carry the expertise.
5. Users respond to plain-language notifications, not metric tables.
6. Clubs already keep longitudinal player records; only the performance layer is missing.
7. Named competitor: **StepOut.ai** — two years inside a Saudi academy, weak on computer-vision
   accuracy and on support.
8. **Still zero pricing evidence.** The single biggest gap.

## Immediate priorities
1. **Second session with the performance director** — she asked to test the product and offered
   introductions to Saudi clubs. Most valuable open item from all seven interviews.
2. **Two buyer interviews scheduled** — academy manager and scouting agency owner. **Get pricing
   evidence from these.**
3. **Assignment #2** by 27 Sep.
4. Mentor syncs and weekly feedback forms — Sun 20 and Sun 27 Sep. Mandatory.

## ⚠ Two unresolved risks
1. **This repository is public.** `RISKS.md` names the footage-provenance problem and the
   interview notes carry identifying detail. Either make the repo private or move `launchpad/`
   to a private one.
2. **The product front-end source is not in version control.** `app.playbookiq.ai` — dashboard,
   metrics, KEMO — works, but the code is unlocated. It is the company's core asset.

## Open questions
- Official term for the Al-Ula / NEOM / Al-Qadsiah club category.
- Whether the performance director is a partner, an advisor, or wants equity — she raised
  investment. Agree scope before equity is discussed.
- Whether KEMO is Phase 2 or Phase 3 in the roadmap (the interviews argue Phase 2).
