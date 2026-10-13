# Final pass file

Builds the match pass CSV from the Pass Review app's data. It uses the same name rules as the app.

- `resolve.py` applies the app's name rules:
  - your box fixes;
  - swap tails taking their partner's name;
  - track-group names;
  - tag renames per segment;
  - substitutions;
  - typos (A66→A65, A52→A62, B8→B21).
- `final.py <dir with resolve.py> <out.csv>` writes one row per pass with:
  - shirt numbers for the passer and receiver;
  - kick and touch frames;
  - board clock, from the nearest board reading in the same half;
  - outcome and a to_check flag;
  - both teams' match momentum at the kick (`momentum_A`, `momentum_B`).
- `left.py <dir>` lists the passes still without a shirt number.

`PASS_DATA` points to the folder holding:
- the app database pulls: pieces, names, reviews, aliases, cuts, clock, and `moments/` for momentum;
- the published site files: pass_pieces, review_index, relay_A, cuts_auto, crossings.

For a new match, update the substitutions, typos and dropped clock readings at the top of `resolve.py` and `final.py`.

Momentum: each mark `{f, team, phase}` (from the app's Momentum mode) holds until the next mark in the same half. The other team gets the opposite phase:
- Buildup → Retreat
- Finishing → Retreat
- Transition → Transition
- Retreat → Buildup
