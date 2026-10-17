import sys, json, pandas as pd, numpy as np
sys.path.insert(0, '/home/user/Playbook-program')
from tools.benchmark import score_against_gt, _iou_matrix
from tools.ball_eval import ball_eval
gt_path, fps = sys.argv[1], float(sys.argv[2])
GT = pd.read_csv(gt_path)
def roles(pred):
    ok = tot = 0; per = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    P = pred[pred.class_id.isin([1, 2, 3])]
    pf = {f: g for f, g in P.groupby('frame')}
    for f, g in GT[GT.class_id.isin([1, 2, 3])].groupby('frame'):
        p = pf.get(f)
        if p is None: continue
        iou = _iou_matrix(g[['x1','y1','x2','y2']].to_numpy(float), p[['x1','y1','x2','y2']].to_numpy(float))
        for i, c in enumerate(g.class_id.to_numpy()):
            j = iou[i].argmax()
            if iou[i, j] >= 0.5:
                per[c][1] += 1; per[c][0] += int(p.class_id.to_numpy()[j] == c)
    return {n: round(per[c][0] / max(per[c][1], 1), 3) for c, n in [(1, 'gk'), (2, 'player'), (3, 'referee')]}
out = {}
for spec in sys.argv[3:]:
    name, path = spec.split('=')
    pred = pd.read_csv(path, low_memory=False)
    a, _, _ = score_against_gt(pred, GT, classes=(2,), fps=fps)
    b, _, _ = score_against_gt(pred, GT, classes=(1, 2, 3), fps=fps)
    bl, _ = ball_eval(pred, GT)
    r = {'player_IDF1': a['IDF1'], 'player_switches': a['id_switches'], 'player_recall': round(a['detection_recall'], 3),
         'people_IDF1': b['IDF1'], 'people_switches': b['id_switches'], 'role_correct': roles(pred),
         'ball_hit': round(bl['hit_rate'], 3), 'ball_at_feet': round(bl['wrong_at_feet_rate'], 3)}
    out[name] = r
    print(f'{name:10s}', r)
print(json.dumps(out))
