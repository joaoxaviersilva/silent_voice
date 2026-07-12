import cv2
import mediapipe as mp
import numpy as np


class FaceMeshDetector:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Landmarks da boca (MediaPipe FaceMesh)
        self.idx = [
            61, 146, 91, 181, 84,
            17, 314, 405, 321, 375,
            291, 308, 324, 318, 402,
            317, 14, 87, 178, 88,
            95, 185, 40, 39, 37,
            0, 267, 269, 270, 409,
            415, 310, 311, 312, 13,
            82, 81, 42, 183, 78,
        ]

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        pts = []

        for idx in self.idx:
            lm = face.landmark[idx]
            pts.append([
                lm.x * w,
                lm.y * h,
            ])

        return np.array(pts, dtype=np.float32)