import cv2


def main():
    cap = cv2.VideoCapture("test.mov")
    roi = 1031, 721, 200, 138
    tracker = cv2.legacy.TrackerMOSSE_create()

    start_frame, stop_frame = 120, 1000
    frame_count = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f"Current frame count {frame_count}", roi)

        if frame_count < start_frame or frame_count > stop_frame:
            continue

        if frame_count == start_frame:
            tracker.init(frame, roi)

        success, roi = tracker.update(frame)
        # success = False

        # if not success:
        # tracker.init(frame, roi_old)

        if success:
            (x, y, w, h) = tuple(map(int, roi))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(
                frame,
                "Tracking failed",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Object Tracking", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
