"""Add frames to the Roboflow keypoint project (by image URL) with pitch points pre-labelled by the
current keypoint model, for a person to correct in Roboflow.

  python tools/roboflow_prelabel_keypoints.py selection.json all log.json

selection.json: [{file, url, new_split, group}, ...] (frames already in the workspace; their
source.roboflow.com URL). Needs ROBOFLOW_API_KEY. Cost: hosted inference (~0.4 s per frame) + upload.
Roboflow maps COCO keypoints by position onto ids 0..33 of this project; slots 0 and 33 are unused,
so they are sent as "not present" (otherwise they appear as visible points at 0,0).
"""
import os, sys, json, time, requests
KEY = os.environ['ROBOFLOW_API_KEY']
PROJ = 'football-field-detection-f07vi-it2xv'
MODEL = 'football-field-detection-f07vi-it2xv/10'
BATCH = 'our_matches_kp_v11'
KP_CONF = 0.5

def predict(url):
    for a in range(4):
        try:
            r = requests.post(f'https://serverless.roboflow.com/{MODEL}', params={'api_key': KEY, 'image': url, 'confidence': 0.3}, timeout=120)
            r.raise_for_status(); return r.json()
        except Exception as e:
            if a == 3: raise
            time.sleep(2 ** a)

def coco_for(name, pred):
    W, H = pred['image']['width'], pred['image']['height']
    kps = [0] * 96; n = 0
    if pred['predictions']:
        p = max(pred['predictions'], key=lambda p: p['confidence'])
        for k in p['keypoints']:
            v = int(k['class'])
            if 1 <= v <= 32 and k['confidence'] >= KP_CONF:
                kps[3 * (v - 1):3 * v] = [round(k['x'], 1), round(k['y'], 1), 2]; n += 1
    xs = [kps[i] for i in range(0, 96, 3) if kps[i + 2]]; ys = [kps[i + 1] for i in range(0, 96, 3) if kps[i + 2]]
    box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)] if xs else [0, 0, W, H]
    return n, {'images': [{'id': 0, 'file_name': name, 'width': W, 'height': H}],
               'categories': [{'id': 1, 'name': 'pitch', 'supercategory': 'none',
                               'keypoints': [str(i) for i in range(0, 34)], 'skeleton': []}],
               'annotations': [{'id': 0, 'image_id': 0, 'category_id': 1, 'bbox': box, 'area': box[2] * box[3],
                                'iscrowd': 0, 'num_keypoints': n, 'keypoints': [0, 0, 0] + kps + [0, 0, 0]}]}
    # Roboflow maps COCO keypoints by position onto ids 0..33 of this project; slots 0 and 33 are unused

def upload(x):
    r = requests.post(f'https://api.roboflow.com/dataset/{PROJ}/upload',
                      params={'api_key': KEY, 'name': x['file'], 'split': x['new_split'], 'image': x['url'], 'batch': BATCH,
                              'tag': [x['group'], 'our_match', 'prelabelled_check']}, timeout=120)
    return r.status_code, r.json()

def annotate(image_id, name, coco):
    r = requests.post(f'https://api.roboflow.com/dataset/{PROJ}/annotate/{image_id}',
                      params={'api_key': KEY, 'name': name.rsplit('.', 1)[0] + '.json', 'overwrite': 'true'}, data=json.dumps(coco),
                      headers={'Content-Type': 'text/plain'}, timeout=120)
    return r.status_code, r.text[:300]

if __name__ == '__main__':
    sel = json.load(open(sys.argv[1])); which = sys.argv[2]
    todo = sel[int(which):int(which) + 1] if which.isdigit() else sel
    log = []
    for x in todo:
        pred = predict(x['url']); n, coco = coco_for(x['file'], pred)
        s, up = upload(x); iid = up.get('id')
        a = annotate(iid, x['file'], coco) if iid else None
        log.append({'file': x['file'], 'split': x['new_split'], 'points': n, 'upload': [s, up], 'annotate': a, 'server_time': pred.get('time')})
        print(x['file'], x['new_split'], n, 'pts', s, up.get('id'), up.get('duplicate'), a, flush=True)
    json.dump(log, open(sys.argv[3], 'w'), indent=1)
