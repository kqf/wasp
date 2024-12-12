import json
from dataclasses import dataclass
from typing import Callable

import cv2

from wasp.tracker.data import Annotation
from wasp.tracker.filter import KalmanFilter


@dataclass
class Segment:
    start_frame: int
    stop_frame: int
    last_frame: int
    bbox: tuple[float, float, float, float]
    tracker: Callable
    name: str

    def within(self, frame_count):
        return self.start_frame <= frame_count < self.stop_frame


SEGMENTS = {
    "sky": Segment(
        120,
        600,
        last_frame=580,
        bbox=(1031.0, 721.0, 200.0, 138.0),
        tracker=cv2.legacy.TrackerMOSSE_create,
        name="sky",
    ),
    "sky-slimmer": Segment(
        120,
        600,
        last_frame=580,
        bbox=(1048, 744, 160, 96),
        tracker=cv2.legacy.TrackerMOSSE_create,
        name="sky-slimmer",
    ),
    "mixed": Segment(
        580,
        667,
        last_frame=667,
        # roi=(897, 449, 32, 18),
        bbox=(896, 446, 16, 17),
        tracker=cv2.legacy.TrackerMOSSE_create,
        name="mixed",
    ),
    "field1": Segment(
        689,
        910,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(609, 470, 20, 21),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="field1",
    ),
    "field2": Segment(
        952,
        1100,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1325, 125, 61, 25),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="field2",
    ),
    "field3": Segment(
        1063,
        1139,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1325, 125, 61, 25),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="field3",
    ),
    "field4": Segment(
        1333,
        1370,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1325, 125, 61, 25),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="field4",
    ),
    "forest1": Segment(
        1404,
        1623,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(438, 908, 103, 49),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="forest1",
    ),
    "forest2": Segment(
        1623,
        3000,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(438, 908, 103, 49),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="forest2",
    ),
    "forest3": Segment(
        1746,
        2014,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1718, 546, 48, 22),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="forest3",
    ),
    "field5": Segment(
        2052,
        2161,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1718, 546, 48, 22),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="field5",
    ),
    "field6": Segment(
        2362,
        2484,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1718, 546, 48, 22),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="field6",
    ),
    "field7-sky": Segment(
        2529,
        2640,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1718, 546, 48, 22),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="field7-sky",
    ),
    "sky3": Segment(
        2574,
        2640,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(671, 8, 27, 17),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="sky3",
    ),
    "sky4": Segment(
        2760,
        3235,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(671, 8, 27, 17),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="sky4",
    ),
    "sky4-mixed": Segment(
        2760,
        3424,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(671, 8, 27, 17),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="sky4-mixed",
    ),
    "sky5": Segment(
        3633,
        3700,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(671, 8, 27, 17),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="sky5",
    ),
    "sky6": Segment(
        3633,
        3700,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1159, 939, 63, 21),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="sky6",
    ),
    "field8": Segment(
        4145,
        4424,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1159, 939, 63, 21),
        tracker=cv2.legacy.TrackerBoosting_create,
        name="field8",
    ),
}


def main():
    cap = cv2.VideoCapture("test.mov")
    segment = SEGMENTS["field8"]
    tracker = segment.tracker()
    bbox = segment.bbox
    frame_count = -1
    kalman_filter = None
    prev_frame = None

    with open("annotations.json", "r") as f:
        annotations: list[Annotation] = Annotation.schema().load(
            json.load(f), many=True
        )
    print(annotations)

    output_annotations: list[Annotation] = []

    output = cv2.VideoWriter(
        f"mosse-{segment.name}.mp4",
        cv2.VideoWriter_fourcc(*"H264"),
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    while frame_count < segment.stop_frame:
        ret, frame = cap.read()
        frame_count += 1

        if not segment.within(frame_count):
            continue

        if frame_count == segment.start_frame:
            tracker.init(frame, segment.bbox)
            kalman_filter = KalmanFilter(segment.bbox)

        kalman_filter.correct(bbox)
        success, bbox = tracker.update(frame)
        bbox = tuple(map(int, bbox))
        x, y, w, h = bbox

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not success:
            cv2.putText(
                frame,
                "Tracking failed",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        kx, ky, kw, kh = kalman_filter.predict()
        cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), (255, 0, 0), 2)

        if prev_frame is None:
            prev_frame = frame

        combined_frame = frame
        output.write(frame)
        cv2.imshow("tracking", combined_frame)
        cv2.waitKey()
        prev_frame = frame

        annotation = Annotation(
            label="tracker",
            segment=segment.name,
            xxyy=(x, y, x + w, y + h),
        )
        output_annotations.append(annotation)

    with open("annotations-updated.json", "w") as f:
        json.dump(
            Annotation.schema().dump(output_annotations, many=True),
            f,
            indent=4,
        )

    output.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
