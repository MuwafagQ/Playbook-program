"""Record a run with fake models, replay it from the cache, and require identical tracks."""
from __future__ import annotations

import json
import sys
import types

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv

W, H, N, START = 640, 360, 45, 5
PLAYERS = [((200, 80, 60), 60), ((200, 80, 60), 180), ((60, 60, 200), 300), ((60, 60, 200), 420)]


def _box(i, f):
    _, x0 = PLAYERS[i]
    x = x0 + 2 * f
    y = 150 + 40 * (i % 2)
    return np.array([x - 12, y - 50, x + 12, y], dtype=np.float32)


def _make_video(path, scale=1.0):
    w, h = int(W * scale), int(H * scale)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (w, h))
    for f in range(START + N):
        img = np.full((H, W, 3), (40, 140, 40), np.uint8)
        for i, (color, _) in enumerate(PLAYERS):
            x1, y1, x2, y2 = _box(i, f).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        vw.write(cv2.resize(img, (w, h)) if scale != 1.0 else img)
    vw.release()


def _fake_detect(model, img, conf, upscale):
    f = int(img[0, 0, 0])  # frame index is stamped into the top-left pixel below
    boxes = [_box(i, f) for i in range(len(PLAYERS))]
    boxes.append(np.array([300 + f, 250, 308 + f, 258], dtype=np.float32))
    return sv.Detections(
        xyxy=np.stack(boxes),
        confidence=np.array([0.9, 0.85, 0.8, 0.88, 0.5], dtype=np.float32),
        class_id=np.array([2, 2, 2, 2, 0], dtype=np.int32),
    )


def _fake_keypoints(model, img, conf):
    from sports.configs.soccer import SoccerPitchConfiguration

    cfg = SoccerPitchConfiguration()
    idx = [0, 5, 24, 29, 13, 16]
    verts = np.asarray(cfg.vertices, dtype=np.float32)[idx]
    xy = np.stack([20 + verts[:, 0] * 0.05, 20 + verts[:, 1] * 0.045], axis=1)
    kp = sv.KeyPoints(xy=xy.reshape(1, -1, 2).astype(np.float32))
    kp.class_id = np.array([int(cfg.labels[i]) for i in idx], dtype=np.int32).reshape(1, -1)
    kp.confidence = np.full((1, len(idx)), 0.9, dtype=np.float32)
    return kp


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test")
    monkeypatch.setenv("TRACKER_TYPE", "botsort")
    monkeypatch.setenv("BOTSORT_WITH_REID", "false")
    monkeypatch.setenv("PREPROCESS_ENABLED", "false")
    monkeypatch.chdir(tmp_path)  # no .env here: defaults + env vars above
    fake_models = types.ModuleType("vision.models")
    fake_models.load_roboflow_models = lambda **kw: (object(), object())
    monkeypatch.setitem(sys.modules, "vision.models", fake_models)
    import main as m

    monkeypatch.setattr(m, "infer_players_and_ball_upscaled", _fake_detect)
    monkeypatch.setattr(m, "infer_field_keypoints", _fake_keypoints)
    monkeypatch.setattr(m, "recover_ball_in_roi", lambda *a, **k: sv.Detections.empty())
    # Stamp the frame index into a pixel the fake detector can read back.
    real_enhance = m.enhance_frame

    def stamped(frame, enabled=True):
        out = real_enhance(frame, enabled=enabled).copy()
        out[0, 0] = stamped.next_idx
        stamped.next_idx += 1
        return out

    monkeypatch.setattr(m, "enhance_frame", stamped)
    return m, stamped


def _tracks(path):
    return pd.read_csv(path).sort_values(["frame", "class_id", "x1"]).reset_index(drop=True)


def test_record_then_replay_is_identical(pipeline, tmp_path):
    m, stamped = pipeline
    video = tmp_path / "v.avi"
    _make_video(video)

    stamped.next_idx = START
    m.main(str(video), out_dir=str(tmp_path / "rec"), start_frame=START, end_frame=START + N - 1,
           record_cache=str(tmp_path / "cache"), write_video=False)
    meta = json.loads((tmp_path / "cache" / "meta.json").read_text())
    assert (meta["start_frame"], meta["frames"], meta["width"]) == (START, N, W)

    # Replay: the fake models must not be called at all.
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called during replay"))
    m.infer_players_and_ball_upscaled = boom
    m.infer_field_keypoints = boom
    stamped.next_idx = START
    m.main(str(video), out_dir=str(tmp_path / "rep"), replay_cache=str(tmp_path / "cache"), write_video=False)

    rec, rep = _tracks(tmp_path / "rec" / "per_frame_tracks.csv"), _tracks(tmp_path / "rep" / "per_frame_tracks.csv")
    assert len(rec) > 0 and rec["frame"].min() == START and rec["frame"].max() == START + N - 1
    pd.testing.assert_frame_equal(rec, rep)

    kpi = json.loads((tmp_path / "rec" / "kpi_summary.json").read_text())
    assert kpi["time_ms_per_frame_model_detect"] >= 0 and "time_ms_per_frame_tracker_botsort" in kpi
    assert json.loads((tmp_path / "rep" / "kpi_summary.json").read_text())["replay"] is True


def test_replay_from_half_res_proxy(pipeline, tmp_path):
    m, stamped = pipeline
    video, proxy = tmp_path / "v.avi", tmp_path / "proxy.avi"
    _make_video(video)
    # Proxy clip: only the window, at half resolution (frame 0 = START).
    cap = cv2.VideoCapture(str(video))
    vw = cv2.VideoWriter(str(proxy), cv2.VideoWriter_fourcc(*"FFV1"), 25, (W // 2, H // 2))
    for f in range(START + N):
        ok, img = cap.read()
        if f >= START:
            vw.write(cv2.resize(img, (W // 2, H // 2)))
    vw.release()

    stamped.next_idx = START
    m.main(str(video), out_dir=str(tmp_path / "rec"), start_frame=START, end_frame=START + N - 1,
           record_cache=str(tmp_path / "cache"), write_video=False)
    stamped.next_idx = START
    m.main(None, out_dir=str(tmp_path / "rep"), replay_cache=str(tmp_path / "cache"),
           replay_video=str(proxy), write_video=False)

    rec, rep = _tracks(tmp_path / "rec" / "per_frame_tracks.csv"), _tracks(tmp_path / "rep" / "per_frame_tracks.csv")
    assert rep["frame"].min() == START and rep["frame"].max() == START + N - 1
    assert len(rep) == len(rec)
    assert (rec["display_track_id"].to_numpy() == rep["display_track_id"].to_numpy()).mean() > 0.95


def test_replay_with_pitch_smoothing(pipeline, tmp_path):
    m, stamped = pipeline
    video = tmp_path / "v.avi"
    _make_video(video)
    stamped.next_idx = START
    m.main(str(video), out_dir=str(tmp_path / "rec"), start_frame=START, end_frame=START + N - 1,
           record_cache=str(tmp_path / "cache"), write_video=False)
    stamped.next_idx = START
    m.main(str(video), out_dir=str(tmp_path / "rep"), replay_cache=str(tmp_path / "cache"),
           write_video=False, smooth_pitch=True)
    rep = pd.read_csv(tmp_path / "rep" / "per_frame_tracks.csv")
    assert {"x_m_raw", "y_m_raw"} <= set(rep.columns)
    people = rep[rep.class_id == 2].dropna(subset=["x_m", "x_m_raw"])
    assert len(people) > 0
    # Static camera. The fixture detects every 2nd frame (FAST_MODE), so players move in
    # 80 cm stair-steps (2 px/frame at 20 cm/px); smoothing straightens the steps but must
    # never move anyone by more than half a step.
    moved = abs(people.x_m - people.x_m_raw)
    assert moved.quantile(0.95) < 40 and abs(people.y_m - people.y_m_raw).max() < 5
    kpi = json.loads((tmp_path / "rep" / "kpi_summary.json").read_text())
    assert kpi["pitch_smoothed"] is True
