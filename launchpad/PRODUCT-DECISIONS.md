# Product decisions — the standing record

*Opened 26 Sep 2026. Decisions taken in conversation that are not obvious from the code or the
assignments. Add to this rather than letting a decision live only in a chat log.*

---

## 1 · The product is the metric layer, not the pipeline

How the data underneath is produced — by hand, by tracking, fully automated — is an
implementation detail the customer never sees. This is what lets the business survive Re-ID not
being solved. See `MVP-STRATEGY.md`.

**Layer 1 is a hybrid, not manual tagging.** The pipeline produces player *positions*
automatically; humans assign *identity* and mark *events*. xP needs positions, and no amount of
manual tagging produces those. Consequence: **pipeline quality sets the labour budget.**

## 2 · Re-ID: the outcome is must-have, the automation is should-have

Stable player identity in the delivered report is required — aggregation needs it, development is
the value proposition, and clubs have already rejected a competitor over unstable IDs.

**Automating** it is cost and scale, not capability. It goes to a research partner, not to five
weeks of founder time.

**And the bottleneck was misdiagnosed.** The detector was trained on 78 images and misses 27% of
players; 883 of 1,802 identity breaks last only 1–5 frames. That is the detector blinking, not
occlusion. Detection is priority #1. See `DETECTION-PLAN.md`.

## 3 · Broadcast is the right input, not a compromise

Broadcast is bad for whole-team geometry — never all 22 in frame — but fine for **ball-local
events**, because the camera follows the ball. The defining metrics are event-based. So broadcast
suits the product.

Team-shape metrics (compactness, block height, pitch control) need a tactical wide view and are
therefore out of scope until the camera question is settled.

## 4 · The pilot: four clubs, three matches

One match analysed yields **two team reports**. Per mentor guidance, one match is enough to test
whether the analysis is worth anything; multi-match development tracking tests a *later service
tier*. Four independent clubs reacting beats one friendly club reacting four times.

Cold doors are opened **with a finished report in hand**, not a request for a meeting. A club that
declines is a finding, not a failure.

## 5 · Report design

**No within-squad leaderboard in the development report.** Ranking a club's own players invites a
cut list; a development profile invites a coaching conversation.

**Comparison is the whole point of the recruitment report** — the subject there is a player you
might sign, not one you might release.

**Profiles, not rankings.** A player is not objectively better or worse: he is a different
profile, suited to a different role, style and budget. Comparison as *fit*, not *quality*.

**Every report states its basis** — what it measures, what it does not, and the sample. Our
per-row confidence values make this a differentiator.

**Supporting buy and sell decisions is deliberate.** It is a primary reason the product exists and
the highest-value thing management buys. It needs written agreements on scope, use and liability —
and Misk or the Ministry may be able to provide a standard framework.

## 5b · The pilot metric set — and the placeholder method

**Computed for the pilot: xP, pass completion, space gained (SP), bypass (BP).**

All four are computable from **player positions plus a marked pass event** — none needs player
identity. Identity enters only when aggregating them per player. That is the Layer 1 hybrid
working as designed, and it is why these four come first while PR (pressure resistance) and xT
wait.

**Remaining metrics are shown as clearly-labelled upcoming features**, not hidden. This is
deliberate: showing a planned metric reveals whether anyone asks for it, before it is built.

⚠ **Anything not computed must be visibly marked as upcoming**, in the UI and in the report. An
unlabelled placeholder that a professional analyst checks is the one version of this that costs
credibility.

**Add to the three-question review:** *"Which of the upcoming metrics would you actually use?"*
Free demand signal on unbuilt features.

## 6 · Role-based access — the eventual platform ⭐

Three audiences, one dataset, different views:

| Role | Sees | Pays for |
|---|---|---|
| **Management** | Cross-market comparison; profiles against budget and role | Recruitment decisions — the highest-value tier |
| **Coaching staff** | Squad development, session-level findings. No internal leaderboard | Weekly work |
| **Players** | Own trend; standing against **anonymised positional benchmarks** | **A second revenue line — players and agents pay directly** |

**The player view's limit: own trend always, anonymised benchmarks yes, named team-mates no.**
Named comparison only where that data is already public. Named internal comparison is where
dressing rooms break, and where the minors question bites hardest.

The player view exists to let a player understand his market value and compete — motivation, not
judgement.

**On the not-list for this quarter.** Three role-based views is a quarter of work. Recorded as a
decision, not an omission.

## 7 · The operating model: a research network

Playbook-IQ as a bridge between labs and teams, with specialist partners by domain — computer
vision (KAUST), tactical modelling (Barça Innovation Hub), sports science. Playbook-IQ holds the
club relationships, the deployment and the product. Tagline: **"Football by the book."**

**The reasoning:** before LLMs this pipeline took months to build; now models annotate it
precisely and will soon track it. **The technology was never the moat.** What stays scarce is
knowing which metric matters and what a coach should do about it. That is a research consultancy,
not a technology vendor.

**Consequence:** the moat is the club relationships, the owned data and the product layer — not
the algorithm. Which is precisely what makes it safe to publish openly with research partners.

## 8 · ICP shift

From **grassroots academies with no budget** to **development squads at professional clubs**.
Al-Ittifaq U21 is the anchor. Better budget, real urgency, and a reference that opens other doors.

⚠ This contradicts Assignment #1 cell 3. Say so rather than letting it be noticed.

## 9 · Open vs closed

| | Labs | Commercial |
|---|---|---|
| Posture | Open — methods, data, co-authorship | Closed — outputs only |
| Why | Publication is the commercialisation route | They can build what they see |

Never show the live prototype to a competitor. The external one-pager and deck exist for that.

## 10 · Intellectual property

**Trademark "Playbook-IQ" with SAIP** — highest value per riyal, ~6,500 SAR, 10 years renewable.
Copyright is automatic. **Patents conflict with publishing**, and the moat is not the algorithm —
so unless something in the metric design is genuinely novel, don't. Legal work otherwise waits
until after validation.

## 11 · Licensing — live exposure

**Ultralytics YOLO, including YOLO26, is AGPL-3.0 by default** and triggers on network use. The
production player detector is `yolov8n`. A paid Roboflow plan does not change this. **RF-DETR
standard sizes are Apache-2.0** — retraining on RF-DETR fixes accuracy and licence in one job.
Verify the licence badge on every trained model before deploying it.

---

## Still open

- Squad ages in the pilot clubs — are there under-18s?
- Al-Ittifaq's opponents on 12 and 18 October (decides the three cold doors)
- Whether the Roboflow plan allows private projects
- ⚠ The repository is public, and the three Roboflow projects are public
- Backend API and authentication are not built, and are not on the sprint plan — the pilot
  hand-delivers reports rather than giving clubs logins. A deliberate gap, not an oversight, but
  it becomes urgent if a club asks for access in Sprint 4
- ⚠ The dashboard front-end source is in no version control
