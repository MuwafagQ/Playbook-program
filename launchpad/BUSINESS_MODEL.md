# Business model, pricing & buyer — as of 8 Sep 2026

Captured from Q&A with Muwafag. Supersedes the PDD where they conflict.

## Positioning — Muwafag's own words (better than the deck)
> "A science lab first, in a SaaS solution form. We have indoor metrics built through our
> science-lab partners; our basic plan covers good things chosen carefully. **Our vision is
> separating the needed metrics from the noise.** Then we have the metrics-on-demand service.
> We define ourselves as a **science lab & data department for mid-league teams.**"

The base tier is not a technical compromise — it is an **editorial conviction**: a small,
curated, science-backed metric set, because most of what the big platforms sell is noise.
Curation is the product. Frame it that way everywhere.

## Pricing (current thinking — unvalidated)
| Tier | Price | Basis |
|---|---|---|
| Basic SaaS | **~30k SAR/year, max** | **per TEAM** (U18, U16, U14 each pay separately) |
| Premium | ~2× basic (~60k SAR/yr) | metrics-on-demand, lab-brokered custom KPIs |

- **Usage cap:** ~4 matches/month/team (roughly what they play). Overage billed extra.
- **AI assistant (KEMO):** token-based metered usage, extra fees beyond an allowance.
- **Per-academy blended ≈ 90k SAR** (3 age groups × 30k) — which *validates* the deck's
  100k ACV. **The deck must show this derivation**, not just assert 100k, or a reviewer
  will assume it's per-club and challenge it.

**Where the numbers came from: "what feels payable." No competitor data. No academy has
ever been quoted a price.** This is the single biggest unvalidated assumption in the plan.

## Cost to serve
- Compute: **~10 SAR per 90-minute match.** Negligible.
- Labour: **months of manual work per match** at present. This is the entire cost base, and
  it is why the 4-matches/month cap is currently unachievable. Models trained so far are
  deliberately weak test models. Gross margin is a function of automation, nothing else.

## THE BUYING COMMITTEE — several roles, one of several triggers

**Correction (8 Sep):** governance is *not* the buyer. It is **one segment of the persona and
one buying trigger among several**, and **it does not apply to all ICPs.**

### The roles, all of them engaged
| Role | What they care about |
|---|---|
| **Owner** | Return on the academy; reputation; player sale value |
| **Sporting director / manager** | Squad quality, development pathway, competitive results |
| **Analyst / game analyst** | Daily workflow; producing insight without a data team |
| **Coach** | What to change on Monday |
| **Governance / internal audit** | Compliance where a mandate applies; unlocking the reward |

Which of these leads varies by account. **Do not assume one universal buyer.** Darrena's
ruling still holds — weight toward whoever pays — but *who pays differs between accounts*,
and part of discovery is finding out which it is.

### The governance trigger — real, but partial
In Saudi, the **Ministry of Sport gives some academies substantial grants and requires them
to implement data-analytics / advanced-analysis tools**; it is a governance matter and
compliance is financially rewarded. Where this applies:
- Budget is externally funded and earmarked — price against the grant, not their own pocket.
- The purchase has urgency attached to someone else's deadline.
- Internal audit or the governance team triggers it, then typically hands evaluation to a
  sporting manager.
- A **compliance artefact** — something an auditor can file as evidence the tool is in use —
  may be worth building.

**But it does not cover every target account.** For the rest, the buying reason is the
football and commercial one: better development, better decisions, provable player value.
So the product needs **both** arguments available, and the discovery interviews must
establish *which trigger is live in each account*, not assume the mandate.

## Competitive framing problem
Academies compare Playbook-IQ to **Hudl, Metrica, and telestration tools** — because those
are the only things they have seen, and because they employ **game analysts, not data
people.**

Those are *video tools for analysts*: watch clips, draw arrows. Playbook-IQ is a *data
department*. Same footage, different category. If the buyer thinks "Hudl with extra steps,"
the price argument is lost before it starts. **Differentiation must be legible to a game
analyst, not to a data scientist.**

## Value argument — needs rewording
Muwafag's version: "replaces a department of up to 10 engineers at 5k SAR/month" — i.e.
600k SAR/year, so 30k is ~5% of the replaced cost.

**Problem:** these academies have *no* data staff at all. You cannot save money nobody is
spending. Reword from cost-replacement to capability-access:

> "A capability you could never afford to build, for about 5% of what building it would cost."

Aspirational, not cost-cutting. Same math, a sentence that survives scrutiny.

## SECOND CUSTOMER — players' agents / agencies
Muwafag: *"a big potential customer."* Treated properly here rather than as a footnote.

**What they want:** **scouting** primarily — finding and evaluating talent, and building the
case for a client. Opposition analysis is secondary.

**Why this is strategically interesting, not just an extra segment:**
1. **Much shorter sales cycle.** An agency is a small commercial business making its own
   buying decision — no governance committee, no grant cycle, no sporting director to
   convince. Days, not quarters.
2. **They are the natural buyer of the Verified Asset Ledger.** An agent's entire job is
   proving a player is worth more than the buying club currently believes. That is precisely
   what the Ledger produces.
3. **They may be a distribution channel, not only a customer.** An agent shopping a player
   with a Playbook-IQ ledger puts the platform in front of buying clubs — which is the
   "Trojan Horse" motion in the deck, but driven by agents rather than academies. Agents move
   between many clubs; an academy touches few.
4. **They already pay for data.** Agents buy Wyscout, Transfermarkt-type services and scouting
   reports today, so willingness to pay is established and comparables exist — which directly
   addresses the "no comparable pricing data" problem.

**Open questions**
- [ ] Is this a separate ICP with its own product, or the same platform with a scouting view?
- [ ] Do agents pay per player, per season, or per placement/success fee?
- [x] **Yes — agents need individual-player metrics, so this segment depends on Re-ID**,
      unlike the academy base tier. This confirms the sequencing: **academies first**
      (no Re-ID needed), **agents second**.
      *But note the wedge:* an agent cares about 1–3 named players, not all 22. Tracking two
      prospects through a match is a far smaller problem than full-squad Re-ID, and manual
      correction on two players is hours rather than months. So an agent offering could ship
      as a **manual service** well before the SaaS is ready — "send the match, name the
      player, we return the report," priced per player per match. No platform, no automation
      required. Revenue, a second validated segment, and real pricing comparables, all before
      Re-ID is solved. **Speculative, and a distraction if it pulls focus during Phase 1** —
      but the door exists.
- [ ] Are agents a customer, a channel, or both?
- [ ] Regulatory: FIFA-licensed agents in Saudi — how many, and are they reachable?

**Include at least one agent in the six discovery interviews.** Not as the beachhead — the
academies remain the Phase 1 focus — but because one conversation would resolve whether this
is a second business or a distraction, and agents are typically far easier to reach than
academy owners.

## Market vision
Start with academies; expand to pro clubs. That progression is what the TAM/SAM/SOM
structure in the deck is describing.

## Open, unvalidated — resolve through interviews
- [ ] Real willingness to pay. Nobody has been quoted.
- [ ] What competitors actually charge in this region.
- [ ] Exact wording, scope and deadline of the Ministry governance requirement.
- [ ] Who signs: auditor, governance lead, or technical director.
- [ ] Whether age-group teams really buy separately, or the academy buys once centrally.
- [ ] Whether a compliance artefact is itself a sellable deliverable.

---

## The real differentiator — tactical event notifications (8 Sep)
The live product (`app.playbookiq.ai`) already shows phase tabs (Build-up / Transition /
Finishing / **Retrait**) and fires in-context notifications such as **"RETRAIT → delay
attack."**

**The notification is the product; the metric is the engine.** A coach reads "delay attack"
and knows what to do. A game analyst — the competitor's user — understands it instantly.
xT never achieves that. This is the answer to the Hudl category problem: Hudl gives you
video to watch; Playbook-IQ tells you what happened and what was wrong with it.

**Hidden metric family (kept unpublished deliberately):** counter-attack efficiency —
time-sensitive metrics and percentages, to be developed with the science-lab partners.
"Retrait / delay attack" is the defensive counterpart already visible in the UI.

**Validation so far:** the two youth-league experts *loved the notifications.*

### Two consequences
1. **UI hierarchy is inverted.** The Performance Metrics table leads with xT, xP, SP, BP, PR
   — the industry-standard metrics Playbook-IQ would be *competing* on — while the
   differentiator is a small popup. Flip it. Same inversion exists in the deck.
2. **Engineering priority may be wrong.** The notifications are currently hand-labelled but
   "easy to automate," and phase/event detection runs off ball position and team shape —
   **no Re-ID required.** Automating the notification layer likely beats perfecting Re-ID as
   the next build. Re-ID gates individual-player metrics (premium tier); notifications gate
   the base product.

### IP protection — practical
Metrics and formulas are very hard to patent. Realistic protection:
- **Code** is automatically copyrighted.
- **Method** as a **trade secret** — which requires acting like it is one: NDA before
  detailed demos, documented internally, access controlled.
- **Trademark** Playbook-IQ and the metric names via **SAIP** (Saudi Authority for
  Intellectual Property). Cheap, fast, and the one reliable protection for a *named* metric.
- This is what **PDD Ask #2 (regulatory support)** is for — raise it with the program.
- **Do not let secrecy block the six interviews.** The label is already visible to anyone
  who sees a demo. A one-line NDA is enough; showing the notification fire is worth far more
  than the risk.
