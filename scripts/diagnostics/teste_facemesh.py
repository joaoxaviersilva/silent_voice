import cv2

from vision.detector import FaceMeshDetector

cap = cv2.VideoCapture(0)

detector = FaceMeshDetector()

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    pts = detector.extract(frame)

    if pts is not None:
        for x, y in pts:
            cv2.circle(
                frame,
                (int(x), int(y)),
                2,
                (0, 255, 0),
                -1,
            )

    cv2.imshow("FaceMesh", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
