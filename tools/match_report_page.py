"""Report page from tools/match_report.py outputs.

  python -m tools.match_report_page --rep <match_report out dir> --passes passes_final.csv \
      --index review_index.json --template review/match_report_template.html --out review/match_report_<match>.html
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

FPS = 30


def build(rep: str, passes: str, index: str, pitch: str) -> dict:
    e = pd.read_csv(f"{rep}/events.csv", keep_default_na=False)
    C = pd.read_csv(f"{rep}/counters.csv"); T = pd.read_csv(f"{rep}/possessions.csv")
    ph = pd.read_csv(f"{rep}/phases_per_frame.csv.gz", keep_default_na=False)
    S = pd.read_csv(f"{rep}/team_speed_per_second.csv", index_col=0)
    idx = json.load(open(index))["segments"]
    P = pd.read_csv(passes, keep_default_na=False)
    live = ph[(ph.A != "") & (ph.A != "Dead ball")]
    teams = {}
    for t, k in (("A", "0"), ("B", "1")):
        p = P[P.passer_team == t]; tp = T[T.team == t]; c = C[C.team == t]
        teams[t] = {"passes": len(p), "complete_pct": round(100 * (p.outcome == "complete").mean(), 1),
                    "possession_pct": round(100 * (live.possession == t).mean(), 1), "possessions": len(tp),
                    "passes_per_poss": round(tp.passes.mean(), 1), "sec_per_pass": round(tp.sec_per_pass.median(), 2),
                    "progress_mps": round(tp.progress_mps.median(), 2), "regains": len(c),
                    "counters": int((c.label == "counter").sum()), "possible": int((c.label == "possible").sum()),
                    "counters_final_third": int(((c.label == "counter") & (c.reached == "final third")).sum()),
                    "run_speed": round(float(S[k].mean()), 2),
                    "phase_share": {a: round(100 * b, 1) for a, b in live[t].value_counts(normalize=True).items()}}
    segs = []
    for g in idx:
        a, b = g["seg_start"], g["seg_start"] + g["frames"]
        row = {"segment": g["name"], "label": g["name"][3:].replace("_", " ")}
        sp = S[(S.index * FPS >= a) & (S.index * FPS < b)]
        for t, k in (("A", "0"), ("B", "1")):
            kf = P.kick_frame.astype(int)
            pp = P[(P.passer_team == t) & (kf >= a) & (kf < b)]
            poss_s = (ph.possession.values[a:b] == t).sum() / FPS
            tp = T[(T.team == t) & (T.start_frame >= a) & (T.start_frame < b)]
            row[t] = {"passes": len(pp), "passes_per_poss_min": round(len(pp) / (poss_s / 60), 1) if poss_s > 30 else None,
                      "run_speed": round(float(sp[k].mean()), 2) if len(sp) else None,
                      "progress_mps": round(float(tp.progress_mps.median()), 2) if tp.progress_mps.notna().any() else None,
                      "counters": int(((C.team == t) & (C.label == "counter") & (C.regain_frame >= a) & (C.regain_frame < b)).sum())}
        segs.append(row)
    S["min"] = (S.index // 60).astype(int)
    m = S.groupby("min")[["0", "1"]].mean().round(2)
    keep = ["team", "clock", "segment", "kind", "won_in", "first_pass_s", "first_pass_gain_m", "carry_m", "progress_m", "speed_mps",
            "opp_visible", "opp_goal_side", "mates_ahead", "label", "reached", "passes", "ended", "goal", "regain_frame"]
    goals = P[P.notes.str.lower().str.startswith("goal") & ~P.notes.str.lower().str.startswith("goal kick")]
    agree = json.load(open(f"{rep}/phase_agreement.json"))["agreement_team_A"]
    return {"teams": teams, "segments": segs,
            "speed_minute": [{"min": int(i), "A": None if np.isnan(r["0"]) else r["0"], "B": None if np.isnan(r["1"]) else r["1"]} for i, r in m.iterrows()],
            "counters": json.loads(C[C.label != "none"][keep].sort_values("regain_frame").to_json(orient="records")),
            "agreement": agree, "n_passes": len(P), "n_regains": len(C), "pitch": pitch,
            "seg_bounds": [[g["name"], g["seg_start"], g["frames"]] for g in idx],
            "goals": [{"team": r.passer_team, "clock": r.kick_clock} for r in goals.itertuples()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True); ap.add_argument("--passes", required=True); ap.add_argument("--index", required=True)
    ap.add_argument("--template", required=True); ap.add_argument("--out", required=True); ap.add_argument("--pitch", default="105x68")
    a = ap.parse_args()
    d = build(a.rep, a.passes, a.index, a.pitch)
    json.dump(d, open(f"{a.rep}/report_data.json", "w"), separators=(",", ":"))
    html = open(a.template).read().replace("__DATA__", json.dumps(d, separators=(",", ":")))
    open(a.out, "w").write(html)
    print("teams", {t: {k: v for k, v in x.items() if k != "phase_share"} for t, x in d["teams"].items()})
    print("phases", {t: x["phase_share"] for t, x in d["teams"].items()}, "counters listed", len(d["counters"]))


if __name__ == "__main__":
    main()
