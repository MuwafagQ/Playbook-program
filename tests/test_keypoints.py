"""Pitch keypoint model: dataset re-split / keypoint order, and the model adapter."""
from __future__ import annotations

import json

import cv2
import numpy as np
import supervision as sv

from tools.keypoint_dataset import ANN, FLIP_PAIRS, LABELS, keypoint_dataset, split_of


def _export(root, split, names, kp_names):
    d = root / split
    d.mkdir(parents=True)
    images, anns = [], []
    for i, n in enumerate(names):
        cv2.imwrite(str(d / n), np.zeros((10, 10, 3), np.uint8))
        images.append({"id": i, "file_name": n, "width": 1920, "height": 1080})
        kps = [0] * (3 * len(kp_names))
        for j, name in enumerate(kp_names):  # visible point at x = its label number
            if name.isdigit() and int(name) in (5, 6, 7, 8, 9):
                kps[3 * j:3 * j + 3] = [float(int(name)), 1.0, 2]
        anns.append({"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0,
                     "keypoints": kps, "num_keypoints": 5})
    cats = [{"id": 0, "name": "pitch-field", "supercategory": "none"},
            {"id": 1, "name": "pitch", "supercategory": "pitch-field", "keypoints": kp_names, "skeleton": []}]
    (d / ANN).write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))


def test_split_by_match():
    assert split_of("match_video_11_clip1_029m42s_01_jpg.rf.abc.jpg") == "valid"
    assert split_of("match_video_12_x.jpg") == "valid"
    assert split_of("match_video_1_x.jpg") == "train"          # not match 11
    assert split_of("hilal_hazm_sr6w_jpg.rf.x.jpg") == "test"
    assert split_of("frame_0019_jpg.rf.x.jpg") == "train"


def test_keypoint_dataset_resplits_and_reorders(tmp_path):
    shuffled = [str(i) for i in range(0, 34)][::-1]  # Roboflow's 34 slots, reversed order
    _export(tmp_path / "src", "train", ["match_video_11_a.jpg", "frame_0001.jpg"], shuffled)
    _export(tmp_path / "src", "valid", ["frame_0002.jpg", "hilal_hazm_b.jpg"], shuffled)
    stats = keypoint_dataset(tmp_path / "src", tmp_path / "dst")
    assert stats.pop("split_by") == "match"
    assert {s: v["images"] for s, v in stats.items()} == {"train": 2, "valid": 1, "test": 1}
    tr = json.loads((tmp_path / "dst/train" / ANN).read_text())
    cat = [c for c in tr["categories"] if c.get("keypoints")][0]
    assert cat["keypoints"] == LABELS
    k = tr["annotations"][0]["keypoints"]
    assert len(k) == 96
    for lab in (5, 6, 7, 8, 9):                         # label L sits at index L-1
        assert k[3 * (lab - 1)] == float(lab) and k[3 * (lab - 1) + 2] == 2
    assert (tmp_path / "dst/valid/match_video_11_a.jpg").exists()


def test_flip_pairs_mirror_the_pitch():
    from sports.configs.soccer import SoccerPitchConfiguration

    cfg = SoccerPitchConfiguration()
    pos = {int(l): np.asarray(v, float) for l, v in zip(cfg.labels, cfg.vertices)}  # look up by LABEL
    L = max(p[0] for p in pos.values())
    for a, b in zip(FLIP_PAIRS[::2], FLIP_PAIRS[1::2]):          # 0-based over labels 1..32
        assert np.allclose([L - pos[a + 1][0], pos[a + 1][1]], pos[b + 1])
    unpaired = set(range(32)) - set(FLIP_PAIRS)
    assert {i + 1 for i in unpaired} == {15, 16, 17, 18}         # the halfway-line points
    assert all(np.isclose(pos[i + 1][0], L / 2) for i in unpaired)


def test_field_model_adapter_feeds_homography_input():
    from vision.detect import infer_field_keypoints
    from vision.field_model import FieldModel

    class Fake:
        def predict(self, rgb, threshold=0.3):
            kp = sv.KeyPoints(xy=np.array([[[float(i), 2.0 * i] for i in range(32)]], np.float32))
            kp.detection_confidence = np.array([0.9])
            kp.keypoint_confidence = np.linspace(0, 1, 32)[None]
            return [kp for _ in rgb]
    kp = infer_field_keypoints(FieldModel(Fake()), np.zeros((1080, 1920, 3), np.uint8), 0.3)
    assert kp.xy.shape == (1, 32, 2)
    assert kp.class_id[0].tolist() == list(range(1, 33))   # vertex labels, not model indices
    assert np.isclose(kp.confidence[0, -1], 1.0) and np.allclose(kp.xy[0, 4], [4, 8])


def test_keypoint_dataset_keeps_roboflow_split_without_our_valid_frames(tmp_path):
    names = [str(i) for i in range(0, 34)]
    _export(tmp_path / "src", "train", ["frame_0001.jpg", "match_video_3_a.jpg"], names)
    _export(tmp_path / "src", "valid", ["frame_0002.jpg"], names)
    stats = keypoint_dataset(tmp_path / "src", tmp_path / "dst")
    assert stats.pop("split_by") == "roboflow"
    assert {s: v["images"] for s, v in stats.items()} == {"train": 2, "valid": 1, "test": 0}


def test_keypoint_dataset_reads_roboflow_zero_padded_names(tmp_path):
    # the real Roboflow export: "01".."09", and 14 / 19 listed last
    names = [f"{i:02d}" for i in list(range(1, 14)) + [15, 16, 17, 18] + list(range(20, 33)) + [14, 19]]
    _export(tmp_path / "src", "train", ["frame_0001.jpg"], names)
    _export(tmp_path / "src", "valid", ["frame_0002.jpg"], names)
    keypoint_dataset(tmp_path / "src", tmp_path / "dst")
    k = json.loads((tmp_path / "dst/train" / ANN).read_text())["annotations"][0]["keypoints"]
    for lab in (5, 6, 7, 8, 9):                         # "05".."09" must not be dropped
        assert k[3 * (lab - 1)] == float(lab) and k[3 * (lab - 1) + 2] == 2


def test_field_model_roboflow_order_and_stretch():
    from vision.field_model import ROBOFLOW_KP_ORDER, FieldModel, contrast_stretch

    assert sorted(ROBOFLOW_KP_ORDER) == list(range(1, 33)) and ROBOFLOW_KP_ORDER[13] == 15
    assert ROBOFLOW_KP_ORDER[30:] == [14, 19]
    img = np.repeat(np.linspace(80, 120, 100).astype(np.uint8)[None, :, None], 3, axis=2)  # dull gradient
    out = contrast_stretch(img)
    assert out.min() == 0 and out.max() == 255

    class Fake:
        def predict(self, rgb, threshold=0.3):
            kp = sv.KeyPoints(xy=np.zeros((1, 32, 2), np.float32))
            kp.detection_confidence = np.array([0.9]); kp.keypoint_confidence = np.ones((1, 32))
            return [kp for _ in rgb]
    res = FieldModel(Fake(), labels=ROBOFLOW_KP_ORDER, stretch=True).infer(np.zeros((20, 20, 3), np.uint8))
    names = [k.class_name for k in res[0].predictions[0].keypoints]
    assert names[13] == "15" and names[30] == "14" and names[31] == "19"
