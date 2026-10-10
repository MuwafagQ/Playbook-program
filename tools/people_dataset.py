"""People-only copy of a COCO detection dataset (Roboflow export layout), for training an
open RF-DETR people model (goalkeeper / player / referee; the ball has its own model).

  python -m tools.people_dataset --src data/v3 --dst data/v3_people

Class ids are fixed to the pipeline's: 1 goalkeeper, 2 player, 3 referee (id 0 is the
Roboflow-style super-category), so the trained model's class ids need no mapping.
Images are symlinked, not copied. Images without people keep an empty annotation list.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ANN = "_annotations.coco.json"
PEOPLE = {"goalkeeper": 1, "player": 2, "referee": 3}
OUT_CATEGORIES = [{"id": 0, "name": "people", "supercategory": "none"}] + [
    {"id": i, "name": n, "supercategory": "people"} for n, i in PEOPLE.items()]


def people_split(src_dir, dst_dir) -> dict:
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    coco = json.loads((src / ANN).read_text())
    remap = {c["id"]: PEOPLE[c["name"].lower()] for c in coco["categories"] if c["name"].lower() in PEOPLE}
    anns = []
    for a in coco["annotations"]:
        if a["category_id"] in remap:
            anns.append({**a, "id": len(anns), "category_id": remap[a["category_id"]]})
    for im in coco["images"]:
        link = dst / im["file_name"]
        if not link.exists():
            os.symlink((src / im["file_name"]).resolve(), link)
    (dst / ANN).write_text(json.dumps({"images": coco["images"], "annotations": anns, "categories": OUT_CATEGORIES}))
    counts = {n: sum(a["category_id"] == i for a in anns) for n, i in PEOPLE.items()}
    return {"images": len(coco["images"]), **counts}


def people_dataset(src_root, dst_root) -> dict:
    return {split: people_split(Path(src_root) / split, Path(dst_root) / split)
            for split in ("train", "valid", "test") if (Path(src_root) / split / ANN).exists()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    a = p.parse_args(argv)
    print(json.dumps(people_dataset(a.src, a.dst), indent=1))


if __name__ == "__main__":
    main()
