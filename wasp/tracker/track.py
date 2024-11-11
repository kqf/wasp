import cv2


def main():
    cap = cv2.VideoCapture("sample.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        exit()

    roi = cv2.selectROI(
        "Select ROI", frame, fromCenter=False, showCrosshair=True
    )
    cv2.destroyWindow("Select ROI")

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, roi)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, roi = tracker.update(frame)

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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
