from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dataclasses_json import dataclass_json
from ultralytics import YOLO


@dataclass_json
@dataclass
class Detection:
    class_name: str
    similarity: float
    coords: np.ndarray  # (4, 2) array of box corners


def to_predictions(results):
    detections = []
    for result in results:
        names = result.names
        container = getattr(result, "boxes", None)
        if hasattr(result, "obb") and result.obb is not None:
            container = result.obb

        boxes = (
            container.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
            if hasattr(container, "xyxyxyxy")
            else container.xyxy.cpu().numpy()
        )
        confs = container.conf.cpu().numpy()
        classes = container.cls.cpu().numpy().astype(int)
        detections.extend(
            Detection(
                class_name=names[cls_id],
                similarity=float(conf),
                coords=box,
            )
            for box, conf, cls_id in zip(boxes, confs, classes)
        )
    return detections


def plot(img, detections):
    for det in detections:
        pts = det.coords.astype(int).reshape((-1, 1, 2))
        color = (0, 255, 0)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        label = f"{det.class_name} {det.similarity:.2f}"
        cv2.putText(
            img,
            label,
            tuple(pts[0][0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def main():
    versions = ["yolo11x", "yolo11m", "yolo11n"]
    for version in versions:
        model = YOLO(f"{version}-obb.pt")

        for file in Path("datasets/test-new").glob("*.png"):
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = model.predict(source=img, conf=0.05, max_det=100)
            predictions = to_predictions(results)
            plot(img, predictions)

            cv2.imwrite(f"{version}-{file.stem}-predicted.png", img)
            cv2.imshow("yolo", img)
            cv2.waitKey()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
