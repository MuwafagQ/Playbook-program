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

## THE BUYER — this changes the whole sale
The Saudi **Ministry of Sport gives academies millions of riyals and requires them to
implement data-analytics / advanced-analysis tools. It is a governance requirement, and
compliance is rewarded financially.**

**Therefore the buyer is the internal auditors and the governance team — not the technical
director, not the coach.**

Consequences:
1. The job-to-be-done is **"satisfy the governance requirement and unlock the reward"**, not
   "win more matches." Football performance is the proof, compliance is the purchase reason.
2. The budget is **externally funded and earmarked**. You are not competing for the academy's
   own discretionary money.
3. Price against **the size of the grant**, not against what a club feels it can afford. This
   dissolves most of the pricing anxiety.
4. The pitch needs a **compliance artefact**: something an auditor can file as evidence that
   an advanced analytics tool is in use. That may be as valuable as the metrics themselves.
5. None of this is in the deck. **It should be on the problem slide.**

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

## Second market — players' agents
Agents want **scouting** (find and evaluate talent) primarily; opposition analysis secondary.
Potentially a much shorter sales cycle than an academy and a natural fit for the Verified
Asset Ledger. Not the beachhead — but worth one interview.

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
