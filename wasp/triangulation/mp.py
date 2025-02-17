import cv2
import mediapipe as mp


def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x, y = int(landmark.x * image.shape[1]), int(
            landmark.y * image.shape[0]
        )
        cv2.circle(
            image, (x, y), 2, (255, 0, 0), -1
        )  # Mark face landmarks in blue


def draw_iris_landmarks(image, landmarks):
    iris_indices = [
        468,
        469,
        470,
        471,
        472,
        473,
        474,
        475,
    ]  # Indices for iris landmarks
    for i in iris_indices:
        landmark = landmarks[i]
        x, y = int(landmark.x * image.shape[1]), int(
            landmark.y * image.shape[0]
        )
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Mark iris in green


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True
    )  # Enables iris tracking

    image = cv2.imread("tests/assets/lenna.png")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_mesh_results = face_mesh.process(image_rgb)
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            draw_landmarks(
                image, face_landmarks.landmark
            )  # Draw facial landmarks
            draw_iris_landmarks(
                image, face_landmarks.landmark
            )  # Draw iris landmarks

    cv2.imshow("Face and Iris Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
