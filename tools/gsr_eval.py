"""Score SoccerNet Game State Reconstruction predictions as a ladder, not one number.

  python -m tools.gsr_eval --gt data/SoccerNetGS/valid --pred runs/ours/sngs_pred [--pred runs/baseline/pred]

--gt   a SoccerNetGS split folder: one sub-folder per clip holding Labels-GameState.json
--pred a folder of <clip>.json files in the challenge submission format
       ({"predictions": [...]}, see sn-gamestate/ChallengeRules.md). Several allowed.

Each rung adds one requirement, so the drop between rungs shows where points are lost:
  1 image_hota   boxes matched by IoU in the image; no attributes     -> detection + tracking
  2 pitch        positions matched in pitch space (5 m tolerance)     -> + homography
  3 +role        role must match (player / goalkeeper / referee / other)
  4 +team        team side must match (left / right)
  5 gs_hota      jersey number must match = the official GS-HOTA

Uses SoccerNet's own scorer (sn-trackeval, `pip install sn-trackeval`), configured as
tracklab's gs_hota evaluator.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

LADDER = [
    ("image_hota", dict(EVAL_SPACE="image", USE_ROLES=False, USE_TEAMS=False, USE_JERSEY_NUMBERS=False)),
    ("pitch", dict(EVAL_SPACE="pitch", USE_ROLES=False, USE_TEAMS=False, USE_JERSEY_NUMBERS=False)),
    ("+role", dict(EVAL_SPACE="pitch", USE_ROLES=True, USE_TEAMS=False, USE_JERSEY_NUMBERS=False)),
    ("+team", dict(EVAL_SPACE="pitch", USE_ROLES=True, USE_TEAMS=True, USE_JERSEY_NUMBERS=False)),
    ("gs_hota", dict(EVAL_SPACE="pitch", USE_ROLES=True, USE_TEAMS=True, USE_JERSEY_NUMBERS=True)),
]


def _seq_info(gt_split: Path, seqs: list[str]) -> dict[str, int]:
    out = {}
    for s in seqs:
        labels = json.loads((gt_split / s / "Labels-GameState.json").read_text())
        out[s] = len(labels["images"])
    return out


def _summary(res: dict) -> dict:
    """Combined-over-sequences numbers, as percentages (tracklab reports the same)."""
    comb = res["COMBINED_SEQ"]
    cls = comb.get("cls_comb_det_av") or comb.get("person") or next(iter(comb.values()))
    hota, clear, ident = cls["HOTA"], cls["CLEAR"], cls["Identity"]
    return {
        "HOTA": 100 * float(np.mean(hota["HOTA"])),
        "DetA": 100 * float(np.mean(hota["DetA"])),
        "AssA": 100 * float(np.mean(hota["AssA"])),
        "LocA": 100 * float(np.mean(hota["LocA"])),
        "MOTA": 100 * float(clear["MOTA"]),
        "IDF1": 100 * float(ident["IDF1"]),
    }


def evaluate(gt_split: str, pred_dirs: dict[str, str], seqs: list[str] | None = None,
             rungs=LADDER) -> dict[str, dict[str, dict]]:
    """Returns {tracker_name: {rung: {HOTA, DetA, AssA, LocA, MOTA, IDF1}}}."""
    import trackeval

    gt_split_p = Path(gt_split).resolve()
    if seqs is None:
        seqs = sorted(p.name for p in gt_split_p.iterdir() if (p / "Labels-GameState.json").is_file())
    for name, d in pred_dirs.items():
        missing = [s for s in seqs if not (Path(d) / f"{s}.json").is_file()]
        if missing:
            raise FileNotFoundError(f"{name}: no predictions for {missing[:5]}{'...' if len(missing) > 5 else ''}")
    split = gt_split_p.name
    info = _seq_info(gt_split_p, seqs)
    out: dict[str, dict[str, dict]] = {n: {} for n in pred_dirs}
    with tempfile.TemporaryDirectory() as tmp:
        trk_root = Path(tmp) / "trackers"
        for name, d in pred_dirs.items():
            dst = trk_root / f"SoccerNetGS-{split}" / name
            dst.mkdir(parents=True)
            for s in seqs:
                shutil.copy(Path(d) / f"{s}.json", dst / f"{s}.json")
        for rung, flags in rungs:
            cfg = trackeval.datasets.SoccerNetGS.get_default_dataset_config()
            cfg.update({
                "GT_FOLDER": str(gt_split_p.parent), "SPLIT_TO_EVAL": split,
                "GT_LOC_FORMAT": "{gt_folder}/{seq}/Labels-GameState.json",
                "TRACKERS_FOLDER": str(trk_root), "TRACKER_SUB_FOLDER": "",
                "OUTPUT_FOLDER": str(Path(tmp) / "out" / rung), "SEQ_INFO": info,
                "DO_PREPROC": False, "PRINT_CONFIG": False, "TRACKERS_TO_EVAL": list(pred_dirs),
                **flags,
            })
            eval_cfg = trackeval.Evaluator.get_default_eval_config()
            eval_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
                             "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                             "BREAK_ON_ERROR": True, "TIME_PROGRESS": False})
            metrics_cfg = {"METRICS": {"HOTA", "CLEAR", "Identity"}, "PRINT_CONFIG": False, "THRESHOLD": 0.5}
            metrics = [trackeval.metrics.HOTA(metrics_cfg), trackeval.metrics.CLEAR(metrics_cfg),
                       trackeval.metrics.Identity(metrics_cfg)]
            with contextlib.redirect_stdout(io.StringIO()):
                dataset = trackeval.datasets.SoccerNetGS(cfg)
                res, _ = trackeval.Evaluator(eval_cfg).evaluate([dataset], metrics)
            for name in pred_dirs:
                out[name][rung] = _summary(res["SoccerNetGS"][name])
    return out


def tracklab_state_to_predictions(pklz: str, out_dir: str, split_prefix: str = "SNGS") -> list[str]:
    """Write <clip>.json prediction files from a TrackLab tracker state (.pklz).

    Mirrors tracklab's SoccerNetGameState.soccernet_encoding for 'object' rows: detections
    without a track id, image box or pitch position are dropped (so they are not counted
    as false positives), NaN attributes become null, and no confidence is written.
    """
    import pickle  # noqa: F401  (pandas.read_pickle)
    import zipfile

    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(pklz) as z:
        vids = sorted(n[:-4] for n in z.namelist() if n.endswith(".pkl") and not n.endswith("_image.pkl"))
        for vid in vids:
            with z.open(f"{vid}.pkl") as f:
                det = pd.read_pickle(f)
            det = det.dropna(subset=["track_id", "bbox_ltwh", "bbox_pitch"], how="any")
            preds = []
            for idx, r in det.iterrows():
                def val(key):
                    v = r.get(key)
                    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
                x, y, w, h = (float(v) for v in r["bbox_ltwh"])
                preds.append({
                    "id": str(idx), "image_id": str(r["image_id"]), "video_id": str(r["video_id"]),
                    "track_id": int(r["track_id"]), "supercategory": "object",
                    "category_id": float(val("category_id") or 1.0),
                    "attributes": {"role": val("role"), "jersey": val("jersey_number"), "team": val("team")},
                    "bbox_image": {"x": x, "y": y, "w": w, "h": h},
                    "bbox_pitch": {k: float(v) for k, v in r["bbox_pitch"].items()},
                })
            name = f"{split_prefix}-{vid}"
            (out / f"{name}.json").write_text(json.dumps({"predictions": preds}))
            written.append(name)
    return written


def format_table(results: dict[str, dict[str, dict]], metric: str = "HOTA") -> str:
    names = list(results)
    rungs = [r for r, _ in LADDER if r in next(iter(results.values()))]
    lines = ["| rung | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for r in rungs:
        lines.append(f"| {r} | " + " | ".join(f"{results[n][r][metric]:.1f}" for n in names) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", action="append", default=[], help="prediction folder; name=path allowed")
    p.add_argument("--seqs", nargs="*", default=None, help="clips to score (default: all in --gt)")
    p.add_argument("--json", default=None, help="also write the full results here")
    p.add_argument("--from-state", action="append", default=[],
                   help="name=tracker_state.pklz: convert a TrackLab state to predictions first")
    a = p.parse_args(argv)
    preds = {}
    for item in a.from_state:
        name, _, pklz = item.partition("=")
        conv = Path(pklz).with_suffix("") / "pred"
        tracklab_state_to_predictions(pklz, str(conv))
        preds[name] = str(conv)
    for item in a.pred:
        name, _, path = item.partition("=") if "=" in item else (Path(item).name, "", item)
        preds[name or Path(path).name] = path
    res = evaluate(a.gt, preds, a.seqs)
    print(format_table(res, "HOTA"))
    print("\nDetA / AssA per rung:")
    for n, rows in res.items():
        print(f"  {n}: " + ", ".join(f"{r} {v['DetA']:.1f}/{v['AssA']:.1f}" for r, v in rows.items()))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
