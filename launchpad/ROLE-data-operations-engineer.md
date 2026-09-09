# Role — Data Operations Engineer
First hire. Engineering background. Reports to the founder.

## Title
**Data Operations Engineer**

Accurate for an engineer, respectable on a CV and on LinkedIn, and it covers both the manual
work needed now and the pipeline work it grows into. Avoid "intern" or "annotator" — he will
disengage, and the role is genuinely more than that.

Alternatives if he prefers: *Data Pipeline Engineer*, *ML Data Engineer*.

## The one-line mission
> **Turn raw match footage into trustworthy structured data — and progressively automate
> yourself out of every manual step in that process.**

That framing matters. The manual work is not busywork he's stuck with; it is how he learns
the domain and generates the training data that eventually replaces it. Say this to him
explicitly on day one.

## Why this role exists
Cost to serve is ~10 SAR of compute per match and **months of manual labour**. Gross margin
is a pure function of automation. This role owns that ratio.

---

## Responsibilities

### 1. Data operations (immediate — this is the bottleneck)
- Pass tagging and tactical event annotation on processed matches
- Player identity correction and verification against the zero-duplicate
  `(frame, display_track_id)` invariant
- Quality review of pipeline output; log every case where the system got it wrong
- Maintain the annotation standard so labels stay consistent between people

### 2. Evaluation & measurement (high value, nobody is doing it)
- Build the **ground-truth evaluation set** from annotated matches
- Measure and report the automation ratio: mean identity duration before fragmentation,
  % of frames with `homography_ok`, ID switches per player per minute, events correctly
  auto-detected
- Produce a per-match quality report

*This is needed before Phase 4 technical due diligence and it does not exist yet. It is real
engineering, it is measurable, and it is his to own.*

### 3. Pipeline engineering (grows over time)
- Homography stabilization: add RANSAC and temporal smoothing — the cheapest quality win
  available, days of work
- Tooling to make annotation faster: batch review UI, keyboard-driven correction, semi-
  automatic pre-labelling
- **Automate the tactical notification layer** (phase and event detection). Currently
  hand-labelled and "easy to automate" — and it needs no player Re-ID, so it is genuinely
  achievable. Likely the highest-leverage build available.
- Data infrastructure: storage, versioning, reproducible processing runs

### 4. Not his — keep these with the founder
- Customer conversations and interviews
- Metric definitions and the football logic behind them
- Pricing, positioning, mentor syncs, the deck
- Player Re-ID research (research-grade — this is the CV-lab Ask, not an internal task)

---

## Progression — say this out loud when you offer the role
| Stage | Focus |
|---|---|
| Weeks 1–4 | Annotation and quality review — learn the domain and the data |
| Weeks 4–8 | Evaluation harness and metrics; annotation tooling |
| Month 3+ | Homography stabilization, notification automation, pipeline ownership |
| Later | Candidate for founding engineer as the company grows |

## First week — concrete
1. Read `notebooks/HANDOVER.md` end to end. It is the best documentation of the data.
2. Annotate one 10-minute clip to the existing standard. Compare against the founder's
   labels; discuss every disagreement. That conversation *is* the annotation standard.
3. Write down every ambiguity found — that becomes the annotation guide.
4. Report: how long did it take, and which step was slowest?

That last number is the baseline everything else is measured against.

## Practical
- **Agree scope, hours and compensation in writing before starting.** Working with a friend
  without written terms is how friendships and companies both break. Even an unpaid or
  equity-light arrangement needs the expectations written down.
- If equity is on the table, use **vesting with a cliff** — standard, protects both of you,
  and is what any investor will expect to see at Phase 4.
- Give him a real title and real ownership of the evaluation work. Engineers stay for
  ownership, not for tasks.
