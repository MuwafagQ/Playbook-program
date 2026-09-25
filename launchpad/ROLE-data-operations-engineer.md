# Role — Data Operations Specialist
First team member. Telecommunications engineering background, no longer working in the field
and not attached to engineering as an identity. **No programming required in this role.**

## Title
**Data Operations Specialist**

Respectable, accurate, and honest — he isn't doing engineering, so calling it "Engineer"
would set an expectation the work doesn't meet. If a more senior framing helps, use
**Data Operations Lead**; he would be the only person in it.

What he actually brings, and what the role needs: **systematic thinking, comfort with
structured data, and patience with detail.** An engineering degree trains all three. None of
the work below requires code.

## The one-line mission
> **Own the quality of our match data, and measure how well the system is doing without us.**

## Why this role exists
Cost to serve is ~10 SAR of compute per match and **months of manual work**. Margin is a pure
function of how much of that manual work goes away. This role owns the manual work today and
the evidence that shows it shrinking.

---

## Responsibilities

### 1. Match data operations — the immediate bottleneck
- Pass tagging and tactical event annotation on processed matches
- Player identity correction and verification (checking no frame has a duplicated player ID —
  a systematic checking task, done in the existing notebooks and spreadsheets)
- Reviewing pipeline output against the video and logging every case where it got it wrong
- Keeping the annotation standard consistent, and writing it down as it develops

### 2. Quality measurement — high value, currently nobody's job
Spreadsheet work, not programming. Per match, track and report:
- How long a player's identity holds before the system loses it
- What percentage of frames had a valid pitch projection
- How many identity switches happened per player per match
- How many tactical events the system caught versus how many he had to add by hand
- Total human hours spent per match

**That last number is the most important metric in the company.** It is the automation
ratio, it goes in Assignment #2, and it is what technical due diligence at Phase 4 will ask
for. It does not exist yet. Give it to him and it becomes genuinely his.

### 3. Research & operations support — useful this week
- Building the academy and agency contact list (names, roles, emails, LinkedIn)
- Competitor and pricing research — what Hudl, Metrica and regional tools actually charge
- Interview scheduling, follow-ups, note-taking, transcription
- Maintaining the outreach tracker

### 4. Not his
- Customer interviews and mentor syncs — founder only
- Metric definitions and football logic — founder only
- Pricing, positioning, the deck
- Anything requiring code or model work

---

## Progression
| Stage | Focus |
|---|---|
| Weeks 1–2 | Learn the data; annotate under supervision; research support |
| Weeks 3–6 | Own annotation quality; produce the per-match quality report |
| Month 2+ | Own the automation-ratio measurement; define the annotation standard; potentially manage additional annotators as volume grows |

If he later wants to learn the technical side, the path exists — but do not build the role
on that assumption.

## First week — concrete
1. Read `notebooks/HANDOVER.md`. It is the clearest description of what the data is.
2. Watch one 10-minute clip alongside the system's output and write down everything wrong.
3. Annotate that same clip. Compare with the founder's version, discuss every disagreement —
   **that conversation is the annotation standard.**
4. Report: how long did it take, and which step was slowest?

Step 4 is the baseline. Everything after is measured against it.

## Practical
- **Agree scope, hours and compensation in writing before he starts.** Working with a friend
  on a handshake is how both the friendship and the company get damaged.
- If equity is discussed, use **vesting with a cliff** — protects both of you, and it is what
  any investor will expect to see at Phase 4.
- Give him genuine ownership of the quality report. People stay for ownership, not for tasks —
  and this one is real, visible, and currently missing.
