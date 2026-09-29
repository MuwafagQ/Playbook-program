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
            if name in ("5", "6", "7", "8", "9"):
                kps[3 * j:3 * j + 3] = [float(name), 1.0, 2]
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
    from tools.manual_calib import pitch_vertices

    v = pitch_vertices()
    L = v[:, 0].max()
    pairs = list(zip(FLIP_PAIRS[::2], FLIP_PAIRS[1::2]))
    for a, b in pairs:
        assert np.allclose([L - v[a, 0], v[a, 1]], v[b])
    unpaired = set(range(32)) - set(FLIP_PAIRS)
    assert all(np.isclose(v[i, 0], L / 2) for i in unpaired)  # only halfway-line points map to themselves


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
