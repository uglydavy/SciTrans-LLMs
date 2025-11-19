
from __future__ import annotations
from pathlib import Path

def train(data_yaml: str, epochs: int = 10, model: str = "yolov8n.pt", out_dir: str = "runs/layout") -> None:
    """Train a YOLO model for document layout detection (stub)."""
    try:
        from ultralytics import YOLO
    except Exception:
        print("Ultralytics not installed. Please `pip install ultralytics` and ensure torch is installed.")
        return
    y = YOLO(model)
    y.train(data=data_yaml, epochs=epochs, imgsz=1280, project=out_dir, name="doclayout")
    print("Training complete. Copy best.pt to data/layout/layout_model.pt")


if __name__ == "__main__":
    print("This is a stub. Provide a dataset YAML and run training.")
