"""Open RF-DETR people model (trained by notebooks/train_people_rfdetr.ipynb) behind the
same `.infer(image, confidence)` interface as a Roboflow `inference` model, so the pipeline
uses it unchanged (PLAYER_MODEL_PATH). Class ids are the pipeline's: 1 goalkeeper,
2 player, 3 referee. It has no ball class: use it with the ball model (BALL_MODEL_ENABLED).
"""
from __future__ import annotations

NAMES = {1: "goalkeeper", 2: "player", 3: "referee"}


def _to_result(det, width: int, height: int) -> dict:
    preds = []
    for (x1, y1, x2, y2), c, k in zip(det.xyxy.tolist(), det.confidence.tolist(), det.class_id.tolist()):
        if int(k) not in NAMES:
            continue
        preds.append({"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "width": x2 - x1, "height": y2 - y1,
                      "confidence": float(c), "class_id": int(k), "class": NAMES[int(k)]})
    return {"image": {"width": width, "height": height}, "predictions": preds}


class PeopleModel:
    def __init__(self, model):
        self.model = model  # an rfdetr model (or anything with .predict(rgb_images, threshold))

    def infer(self, images, confidence: float = 0.3):
        batch = images if isinstance(images, list) else [images]
        rgb = [im[:, :, ::-1].copy() for im in batch]
        out = self.model.predict(rgb, threshold=confidence)
        out = out if isinstance(out, list) else [out]
        return [_to_result(d, im.shape[1], im.shape[0]) for d, im in zip(out, batch)]


def load_people_model(path: str) -> PeopleModel:
    from rfdetr.detr import RFDETR

    model = RFDETR.from_checkpoint(path, trust_checkpoint=True)
    try:
        model.optimize_for_inference()
    except Exception as e:  # optional speed-up; not available on every setup
        print(f"[people-model] optimize_for_inference skipped: {e}")
    return PeopleModel(model)
