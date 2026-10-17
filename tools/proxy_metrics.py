"""Quality numbers for a run without ground truth (new footage).

  python tools/proxy_metrics.py name=runs/a/per_frame_tracks.csv name2=... 

Compare against runs on the CVAT clips, where the true scores are known: new IDs per minute
(ID fragmentation), image jumps (an ID moving more than a body height in one frame), share of
frames with a detected (not interpolated) ball, and ball jumps (>120 px in one frame).
Assumes 30 fps."""
import sys, json
import numpy as np, pandas as pd
rows = {}
for spec in sys.argv[1:]:
    name, path = spec.split('=')
    t = pd.read_csv(path, low_memory=False)
    fps = 30.0
    n_frames = t.frame.nunique(); minutes = n_frames / fps / 60
    p = t[t.class_id.isin([1, 2, 3])]
    per_frame = p.groupby('frame').size()
    ids = p.groupby('track_id').frame.agg(['min', 'max', 'count'])
    late_births = int((ids['min'] > t.frame.min() + fps).sum())          # IDs that start after the first second
    # image jumps: an ID moving more than its body height between consecutive frames
    p = p.sort_values(['track_id', 'frame'])
    cx = (p.x1 + p.x2) / 2; cy = p.y2; h = (p.y2 - p.y1)
    same = (p.track_id.values[1:] == p.track_id.values[:-1]) & (np.diff(p.frame.values) == 1)
    d = np.hypot(np.diff(cx.values), np.diff(cy.values))
    jumps = int((same & (d > h.values[1:])).sum())
    b = t[(t.class_id == 0) & t.x1.notna()].drop_duplicates('frame').sort_values('frame')
    real = b[~b.ball_interpolated.astype(bool)] if 'ball_interpolated' in b else b
    bc = np.c_[(real.x1 + real.x2) / 2, (real.y1 + real.y2) / 2]
    bj = int(((np.diff(real.frame.values) == 1) & (np.hypot(*np.diff(bc, axis=0).T) > 120)).sum())
    rows[name] = {
        'people_per_frame': round(float(per_frame.mean()), 1),
        'ids_total': int(len(ids)), 'max_people_in_a_frame': int(per_frame.max()),
        'new_ids_after_1s_per_min': round(late_births / minutes, 1),
        'image_jumps_per_min': round(jumps / minutes, 1),
        'ball_real_share': round(len(real) / n_frames, 3),
        'ball_any_share': round(len(b) / n_frames, 3),
        'ball_jumps_per_min': round(bj / minutes, 1),
    }
print(pd.DataFrame(rows).T.to_string())
