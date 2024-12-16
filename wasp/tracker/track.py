import json
from dataclasses import dataclass

import cv2
from dataclasses_json import dataclass_json

# from wasp.tracker.boundaries import visualize_features
from wasp.tracker.filter import KalmanFilter


@dataclass_json
@dataclass
class Segment:
    start_frame: int
    stop_frame: int
    last_frame: int
    bbox: tuple[float, float, float, float]
    tracker: str
    name: str

    def within(self, frame_count):
        return self.start_frame <= frame_count < self.stop_frame


def overlay_bbox_on_frame_simple(frame, bbox, max_size=256, o_x=40):
    x, y, w, h = bbox
    roi = frame[y : y + h, x : x + w]
    w, h = roi.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_roi = cv2.resize(roi, (new_width, new_height))
    frame_height, frame_width = frame.shape[:2]
    o_y = frame_height - new_height - 10
    frame[o_y : o_y + new_height, o_x : o_x + new_width] = resized_roi


def overlay_template(frame, roi, score, max_size=256, o_x=40):
    w, h = roi.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_roi = cv2.resize(roi, (new_width, new_height))
    cv2.putText(
        resized_roi,
        f"score: {score:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        4,
    )
    frame_height, frame_width = frame.shape[:2]
    o_y = frame_height - new_height - 10
    frame[o_y : o_y + new_height, o_x : o_x + new_width] = resized_roi


def overlay_bbox_on_frame(frame, bbox, max_size=256, o_x=40):
    x, y, w, h = bbox
    center_x, center_y = x + w // 2, y + h // 2
    new_w, new_h = 2 * w, 2 * h
    new_x, new_y = max(0, center_x - new_w // 2), max(0, center_y - new_h // 2)
    roi = frame[
        new_y : new_y + min(frame.shape[0] - new_y, new_h),
        new_x : new_x + min(frame.shape[1] - new_x, new_w),
    ]
    scale = min(max_size / roi.shape[1], max_size / roi.shape[0])
    nroi = cv2.resize(
        roi,
        (int(roi.shape[1] * scale), int(roi.shape[0] * scale)),
    )
    o_y = max(10, frame.shape[0] - nroi.shape[0] - 10)
    o_x = min(o_x, frame.shape[1] - nroi.shape[1])
    frame[o_y : o_y + nroi.shape[0], o_x : o_x + nroi.shape[1]] = nroi  # noqa
    return frame


TRACKERS = {
    "cv2.legacy.TrackerMOSSE_create",
    cv2.legacy.TrackerMOSSE_create,
    "cv2.legacy.TrackerBoosting_create",
    cv2.legacy.TrackerBoosting_create,
}

SEGMENTS = {
    "sky": Segment(
        120,
        600,
        last_frame=580,
        bbox=(1031.0, 721.0, 200.0, 138.0),
        tracker="cv2.legacy.TrackerMOSSE_create",
        name="sky",
    ),
    "sky-slimmer": Segment(
        120,
        600,
        last_frame=580,
        bbox=(1048, 744, 160, 96),
        tracker="cv2.legacy.TrackerMOSSE_create",
        # "cv2.TrackerCSRT_create",
        # "TemplateMatchingTracker",
        # "TemplateMatchingTrackerWithResize",
        name="sky-slimmer",
    ),
    "mixed": Segment(
        580,
        667,
        last_frame=667,
        # roi=(897, 449, 32, 18),
        bbox=(896, 446, 16, 17),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        # "cv2.TrackerCSRT_create",
        tracker="cv2.legacy.TrackerMOSSE_create",
        name="mixed",
    ),
    "field1": Segment(
        689,
        910,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(609, 470, 20, 21),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="field1",
    ),
    "field2": Segment(
        952,
        1100,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1325, 125, 61, 25),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="field2",
    ),
    "field3": Segment(
        1063,
        1139,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1325, 125, 61, 25),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="field3",
    ),
    "field4": Segment(
        1333,
        1370,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1325, 125, 61, 25),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="field4",
    ),
    "forest1": Segment(
        1404,
        1623,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(438, 908, 103, 49),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="forest1",
    ),
    "forest2": Segment(
        1623,
        3000,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(438, 908, 103, 49),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="forest2",
    ),
    "forest3": Segment(
        1746,
        2014,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1718, 546, 48, 22),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="forest3",
    ),
    "field5": Segment(
        2052,
        2161,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1718, 546, 48, 22),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="field5",
    ),
    "field6": Segment(
        2362,
        2484,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1718, 546, 48, 22),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="field6",
    ),
    "field7-sky": Segment(
        2529,
        2640,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1718, 546, 48, 22),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="field7-sky",
    ),
    "sky3": Segment(
        2574,
        2640,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(671, 8, 27, 17),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="sky3",
    ),
    "sky4": Segment(
        2760,
        3235,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(671, 8, 27, 17),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="sky4",
    ),
    "sky4-mixed": Segment(
        2760,
        3424,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(671, 8, 27, 17),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="sky4-mixed",
    ),
    "sky5": Segment(
        3633,
        3700,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(671, 8, 27, 17),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="sky5",
    ),
    "sky6": Segment(
        3633,
        3700,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1159, 939, 63, 21),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="sky6",
    ),
    "field8": Segment(
        4145,
        4424,
        last_frame=830,
        # roi=(897, 449, 32, 18),
        bbox=(1159, 939, 63, 21),
        # tracker="cv2.legacy.TrackerMOSSE_create",
        tracker="cv2.legacy.TrackerBoosting_create",
        name="field8",
    ),
}


def save_segments(segments, filename):
    data = {key: segment.to_dict() for key, segment in segments.items()}
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_segments(filename):
    with open(filename, "r") as json_file:
        data = json.load(json_file)
    return {key: Segment.from_dict(value) for key, value in data.items()}


def main():
    save_segments(SEGMENTS, "test.json")
    return
    cap = cv2.VideoCapture("test.mov")
    segment = SEGMENTS["sky"]
    tracker = TRACKERS[segment.tracker]()
    bbox = segment.bbox
    frame_count = -1
    kalman_filter = None
    # ellipse = None
    prev_frame = None  # Store the previous frame

    # output = cv2.VideoWriter(
    #     f"mosse-{segment.name}.mp4",
    #     cv2.VideoWriter_fourcc(*"H264"),
    #     cap.get(cv2.CAP_PROP_FPS),
    #     (
    #         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    #     ),
    # )
    while frame_count < segment.stop_frame:
        ret, frame = cap.read()
        frame_count += 1

        if not segment.within(frame_count):
            continue

        if frame_count == segment.start_frame:
            # bbox = cv2.selectROI("select the area", frame)
            # print(bbox)
            tracker.init(frame, segment.bbox)
            kalman_filter = KalmanFilter(segment.bbox)

        kalman_filter.correct(bbox)
        success, bbox = tracker.update(frame)
        bbox = tuple(map(int, bbox))
        x, y, w, h = bbox

        # Draw the original tracker's bounding box
        # frame, ellipse = visualize_features(frame, roi, ellipse)
        # overlay_bbox_on_frame_simple(frame, bbox)
        # overlay_template(frame, tracker.template, tracker.max_val, o_x=300)
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

        # combined_frame = cv2.hconcat([prev_frame, frame])
        combined_frame = frame
        # output.write(frame)
        cv2.imshow("tracking", combined_frame)
        cv2.waitKey()
        prev_frame = frame
        print(frame_count)

    # print(output.release())
    print("released")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
