
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
from ..config import LAYOUT_MODEL
from ..utils import detect_device

@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple  # x0, y0, x1, y1

class LayoutPredictor:
    def __init__(self, model_path: Path | None = None, device: str | None = None):
        self.model_path = model_path or LAYOUT_MODEL
        self.device = device or detect_device()
        self._model = None
        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Layout model not found at {self.model_path}. Run 'python3 -m scitrans_lm setup --yolo'.")
        # Attempt to load with ultralytics; if placeholder, this will likely fail at inference, which is fine until setup completes.
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
        except Exception as e:
            # Defer heavy errors to runtime; allow pipeline to continue if user wants to translate without detection.
            self._model = None

    def detect(self, image_path: str, conf: float = 0.25) -> List[Detection]:
        if self._model is None:
            return []
        try:
            results = self._model.predict(image_path, conf=conf, device=self.device, imgsz=1280, verbose=False)
            dets: List[Detection] = []
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls = int(b.cls[0].item())
                    label = r.names.get(cls, str(cls))
                    xyxy = b.xyxy[0].tolist()
                    score = float(b.conf[0].item())
                    dets.append(Detection(label=label, score=score, bbox=tuple(xyxy)))
            return dets
        except Exception:
            return []

