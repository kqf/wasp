import cv2
from tracker.capture import video_dataset
from tracker.segments import load_segments

from wasp.tracker.filter import KalmanFilter


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
    "cv2.legacy.TrackerMOSSE_create": cv2.legacy.TrackerMOSSE_create,
    "cv2.legacy.TrackerBoosting_create": cv2.legacy.TrackerBoosting_create,
}


def draw_bbox(frame, xywh, color=(0, 255, 0)):
    bbox = tuple(map(int, xywh))
    x, y, w, h = bbox
    return cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def main():
    segment = load_segments("wasp/tracker/segments.json")["sky"]
    tracker = None
    frames = video_dataset(
        iname="test.mov",
        start=segment.start_frame,
        final=segment.stop_frame,
        label="test-annotations.json",
    )
    bbox = segment.bbox
    kalman_filter = None
    prev_frame = None
    for frame in frames:
        if tracker is None:
            tracker = TRACKERS[segment.tracker]()
            # bbox = cv2.selectROI("select the area", frame)
            # print(bbox)
            tracker.init(frame, segment.bbox)
            kalman_filter = KalmanFilter(segment.bbox)

        kalman_filter.correct(bbox)
        success, bbox = tracker.update(frame)
        draw_bbox(frame, bbox)

        # Draw the original tracker's bounding box
        # frame, ellipse = visualize_features(frame, roi, ellipse)
        # overlay_bbox_on_frame_simple(frame, bbox)
        # overlay_template(frame, tracker.template, tracker.max_val, o_x=300)

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

        draw_bbox(frame, kalman_filter.predict(), (255, 0, 0))

        if prev_frame is None:
            prev_frame = frame

        # combined_frame = cv2.hconcat([prev_frame, frame])
        combined_frame = frame
        # output.write(frame)
        cv2.imshow("tracking", combined_frame)
        cv2.waitKey()
        prev_frame = frame
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
