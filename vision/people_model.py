"""Open RF-DETR people model (trained by notebooks/train_people_rfdetr.ipynb) behind the
same `.infer(image, confidence)` interface as a Roboflow `inference` model, so the pipeline
uses it unchanged (PLAYER_MODEL_PATH). Class ids are the pipeline's: 1 goalkeeper,
2 player, 3 referee. It has no ball class: use it with the ball model (BALL_MODEL_ENABLED).
"""
from __future__ import annotations

NAMES = {1: "goalkeeper", 2: "player", 3: "referee"}


# rfdetr numbers classes from 0 and skips the dataset's super-category (id 0), so its
# class k is dataset category k + 1: 0 goalkeeper, 1 player, 2 referee.
CLASS_OFFSET = 1


def _to_result(det, width: int, height: int, offset: int = CLASS_OFFSET) -> dict:
    preds = []
    for (x1, y1, x2, y2), c, k in zip(det.xyxy.tolist(), det.confidence.tolist(), det.class_id.tolist()):
        k = int(k) + offset
        if k not in NAMES:
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


def load_people_model(path: str, accel: str = "none") -> PeopleModel:
    from vision.fast_rfdetr import load_rfdetr

    return PeopleModel(load_rfdetr(path, accel, max_batch=4))
