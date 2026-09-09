# Session handover — MISK Launchpad C10

Written 8 Sep 2026 at the end of the first working session, to be read by the next session.

## Read these first, in order
1. `launchpad/CONTEXT.md` — who Muwafag is, what Playbook-IQ is, the PDD distilled
2. `launchpad/PROGRAM.md` — phase timeline, the 250 → 125 → 34 funnel, full Phase 1 curriculum
3. `launchpad/TECH_REALITY.md` — the candid technical assessment; **the most important file**
4. `launchpad/RISKS.md` — footage provenance, deck-vs-reality gap, solo-founder risk
5. `launchpad/TASKS.md` — running task log
6. `launchpad/assignments/01-customer-discovery.md` — Assignment #1 working notes

## Who you are talking to
The team is now **two people**:
- **Muwafag Hussain** — founder. Strategy, customers, metrics, pricing, the program.
- **A Data Operations Specialist** — first team member, telecom-engineering background, no
  longer in that field, **does not program**. Owns annotation, data quality measurement, and
  research/outreach support. See `ROLE-data-operations-engineer.md` and
  `ONBOARDING-TEAM.md`.

Both use the same account, so **establish which of them you are speaking with at the start of
a session** — their needs differ. The working agreement below is Muwafag's.

## Working agreement
- Muwafag wants a co-pilot through the whole program — sessions, 1:1s, and assigned tasks —
  not one-off deliverables.
- He explicitly asks to be **questioned a lot**, to surface his own perspective. Ask before
  assuming. Do not produce polished artefacts on guessed requirements.
- Anything worth remembering must be committed to this repo — containers are ephemeral.
- Commit to branch `claude/setup-gpu-video-testing-JhgUH`.

## What was established this session

**The central strategic finding.** The PDD sells automated SaaS. The pipeline needs a human
in the loop (Re-ID and pass tagging are manual today). The resolution is *not* to fix the
code first — it is to reframe as a **managed analytical service / fractional science lab
with a proprietary automation stack**, which is the positioning Muwafag already prefers
("a science lab, not a solution provider") and which survives Phase 4 technical diligence.
Same business, different promise. Full argument in `TECH_REALITY.md`.

**The Phase 2 MVP unlock.** Team-level metrics (territory, block height, compactness,
defensive line height, pressing by zone, formation shape, pitch control) need only
detection + team classification + homography — **no Re-ID** — so they are automatic and
reliable *today*. That is Tier 1, self-serve. Individual-player metrics and the Verified
Asset Ledger become Tier 2, analyst-assisted — which is already the "Custom Metric Library"
upsell in the business model. This is the answer for the MVP Scope Canvas session (17 Sep).

**The differentiator nobody else has.** Every output row carries `homography_ok`,
`inlier_ratio`, `reproj_err`, `conf`, `ball_interpolated`. The pipeline knows when it is
unsure. For a product claiming "mathematical proof of player progression," quantified
uncertainty is what makes "proof" defensible. Should be led with, not buried.

## Immediate, time-sensitive
- **Wed 9 Sep 6:40pm — Ministry of Sport SportTech Challenge Deep-Dive.** Footage was taken
  from the Ministry platform informally, without legal access, and the Ministry sponsors this
  track. Ask it as a partnership question: *"our pipeline runs on broadcast footage — what is
  the sanctioned path to licensed league footage for an accredited Launchpad startup?"*
  See `RISKS.md` #1.
- **Wed 9 Sep 4:30pm — Customer Discovery workshop (Katya de Freedericksz).** The how-to for
  Assignment #1.
- **Thu 10 Sep 5:00pm — Track meetups.** Introduction channel; his binding constraint is
  access to interviewees, not interview skill.
- **Thu 17 Sep 11:59pm — Assignment #1 due.**

## First jobs for the new session (Notion is now connected)
1. Pull **Assignment #1's real brief** from the Notion dashboard. Current notes in
   `assignments/01-customer-discovery.md` are built on assumptions and must be corrected
   against the actual requirements.
2. Pull **Assignment #2's brief** (due 27 Sep) and the **Ministry of Sport SportTech
   Challenge track requirements**.
3. Update `PROGRAM.md` and `TASKS.md` with whatever the dashboard reveals, and commit.

## Open questions put to Muwafag, not yet answered
1. Was `HILAL-AHLI_match_B_up1.mp4` a full half or already a clip? The reviewed windows total
   ~3,700 frames (~2–2.5 min). This number drives the whole automation-ratio argument.
2. Does the "science lab, not automated SaaS" reframe sit right, or does it feel like retreat?
3. Have the two warm youth-league experts been asked for referrals?
4. Is he connected to the Barça Innovation Hub alumni network? It is a global network of
   exactly his target persona and is currently unused in the plan.
5. Which funnel number governs — 34 (official timeline) or 50 (his recollection)?
6. Is KEMO in the Phase 2 MVP, or a Phase 3 story? Big build; not what wins a first pilot.
