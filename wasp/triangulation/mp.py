from dataclasses import dataclass
from typing import List, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np


def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x, y = int(landmark[0] * image.shape[1]), int(
            landmark[1] * image.shape[0]
        )
        cv2.circle(
            image, (x, y), 2, (255, 0, 0), -1
        )  # Mark face landmarks in blue


def draw_iris_landmarks(image, landmarks):
    for x, y in landmarks:
        x, y = int(x * image.shape[1]), int(y * image.shape[0])
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Mark iris in green


def draw_face_bounding_box(image, bbox):
    x, y, w, h = bbox
    cv2.rectangle(
        image, (x, y), (x + w, y + h), (0, 0, 255), 2
    )  # Draw face bounding box in red


def draw_bbbox_and_landmarks(image, annotation):
    draw_face_bounding_box(image, annotation.bbox)
    draw_landmarks(image, annotation.landmarks)
    draw_iris_landmarks(image, annotation.leye)
    draw_iris_landmarks(image, annotation.reye)


Number = Union[int, float]
AbsoluteXYXY = Tuple[Number, Number, Number, Number]


@dataclass
class Annotation:
    bbox: AbsoluteXYXY
    landmarks: List[
        Tuple[Number, Number]
    ]  # 5 keypoints mix of iris and face mesh
    leye: Tuple[Tuple[Number, Number], ...]  # All iris landmarks for left eye
    reye: Tuple[Tuple[Number, Number], ...]  # All iris landmarks for right eye


class ExportModelMediaPipe:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.face_detection = mp.solutions.face_detection.FaceDetection()
        self.relevant_indices = [468, 473, 1, 61, 291]
        self.left_iris_indices = [
            468,
            469,
            470,
            471,
            472,
        ]  # Left eye iris landmarks
        self.right_iris_indices = [
            473,
            474,
            475,
            476,
            477,
        ]  # Right eye iris landmarks

    def infer_on_rgb(self, image: np.ndarray) -> List[Annotation]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_detections = self.face_detection.process(image_rgb)
        face_mesh_results = self.face_mesh.process(image_rgb)

        annotations: list[Annotation] = []
        if face_detections is None or face_detections.detections is None:
            return []

        detections = [
            max(
                face_detections.detections,
                key=lambda detection: detection.score[0],  # type: ignore
                default=None,  # type: ignore
            )
        ]
        if not detections or detections[0] is None:
            return []

        annotations = []
        for detection in detections:
            ih, iw, _ = image.shape
            if not face_mesh_results.multi_face_landmarks or detection is None:
                continue

            bbox = detection.location_data.relative_bounding_box
            bbox = (
                int(bbox.xmin * iw),
                int(bbox.ymin * ih),
                int(bbox.xmin * iw + bbox.width * iw),
                int(bbox.ymin * ih + bbox.height * ih),
            )
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                keypoints = [
                    (
                        face_landmarks.landmark[i].x * iw,
                        face_landmarks.landmark[i].y * ih,
                    )
                    for i in self.relevant_indices
                ]
                leye_points = tuple(
                    (
                        face_landmarks.landmark[i].x * iw,
                        face_landmarks.landmark[i].y * ih,
                    )
                    for i in self.left_iris_indices
                )
                reye_points = tuple(
                    (
                        face_landmarks.landmark[i].x * iw,
                        face_landmarks.landmark[i].y * ih,
                    )
                    for i in self.right_iris_indices
                )
                annotations.append(
                    Annotation(
                        bbox=bbox,
                        landmarks=keypoints,
                        leye=leye_points,
                        reye=reye_points,
                    )
                )
        return annotations


def main():
    image = cv2.imread("tests/assets/lenna.png")
    model = ExportModelMediaPipe()
    predictions = model.infer_on_rgb(image)

    for annotation in predictions:
        draw_bbbox_and_landmarks(image, annotation)

    cv2.imshow("Face, Iris, and Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
