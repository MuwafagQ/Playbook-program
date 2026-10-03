"""Tests for tools/cvat_convert.py."""
from __future__ import annotations

import json
import zipfile

import cv2
import numpy as np
import pandas as pd

from tools import benchmark as bm
from tools import cvat_convert as cc

XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations><version>1.1</version>
<meta><job><size>61</size><original_size><width>320</width><height>180</height></original_size></job></meta>
<track id="0" label="Player">
  <box frame="0" keyframe="1" outside="0" occluded="0" xtl="10" ytl="20" xbr="30" ybr="80" z_order="0"/>
  <box frame="30" keyframe="0" outside="0" occluded="0" xtl="40" ytl="20" xbr="60" ybr="80" z_order="0"/>
  <box frame="60" keyframe="1" outside="0" occluded="0" xtl="70" ytl="20" xbr="90" ybr="80" z_order="0"/>
</track>
<track id="1" label="Referee">
  <box frame="0" keyframe="1" outside="0" occluded="0" xtl="100" ytl="20" xbr="120" ybr="80" z_order="0"/>
  <box frame="30" keyframe="1" outside="0" occluded="0" xtl="100" ytl="20" xbr="120" ybr="80" z_order="0"/>
  <box frame="60" keyframe="1" outside="1" occluded="0" xtl="100" ytl="20" xbr="120" ybr="80" z_order="0"/>
</track>
<track id="2" label="Pitch_lines"><skeleton frame="0" keyframe="1" outside="0" occluded="0" z_order="0"/></track>
</annotations>"""


def test_parse_select_coco_and_zip(tmp_path):
    x = tmp_path / "annotations.xml"
    x.write_text(XML)
    boxes = cc.parse_boxes(str(x))
    assert len(boxes) == 5  # the 'outside' referee box and the skeleton are dropped
    assert set(boxes.label) == {"Player", "Referee"}
    # frame 30 has an interpolated player box, so only 0 and 60 are fully hand-drawn
    assert cc.choose_training_frames(boxes, min_gap=30) == [0, 60]
    coco = cc.to_coco(boxes, [0, 60], "clipA", cc.frame_size(str(x)))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    assert sorted(names[a["category_id"]] for a in coco["annotations"]) == ["player", "player", "referee"]
    assert coco["images"][0]["width"] == 320

    tr = cc.to_tracks_csv(boxes)
    assert set(tr.notes) == {"cvat_keyframe", "cvat_interp"}

    video = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10, (320, 180))
    for f in range(61):
        vw.write(np.full((180, 320, 3), f, np.uint8))
    vw.release()
    assert cc.extract_zip(str(video), coco, str(tmp_path / "t.zip")) == 2
    with zipfile.ZipFile(tmp_path / "t.zip") as z:
        img = cv2.imdecode(np.frombuffer(z.read("clipA__f0000060.jpg"), np.uint8), cv2.IMREAD_COLOR)
        assert abs(int(img.mean()) - 60) <= 1  # the right frame was extracted
        assert json.loads(z.read("_annotations.coco.json"))["images"][1]["extra"]["frame"] == 60


def test_real_exports_score_perfectly_against_themselves():
    for z in ("data/cvat/cvat_annotation1.zip", "data/cvat/cvat_annotation2.zip"):
        tr = cc.to_tracks_csv(cc.parse_boxes(z))
        res, _, _ = bm.score_against_gt(tr, tr, classes=(1, 2, 3), offset_search=0)
        assert res["IDF1"] == 1.0 and res["id_switches"] == 0 and res["warnings"] == []
