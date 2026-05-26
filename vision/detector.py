import cv2
import mediapipe as mp
import numpy as np


class FaceMeshDetector:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        self.idx = list(range(61, 88))

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None

        face = res.multi_face_landmarks[0]
        h, w, _ = frame.shape

        pts = []
        for i in self.idx:
            lm = face.landmark[i]
            pts.append([lm.x * w, lm.y * h])

        return np.array(pts)
