import numpy as np


def crop_mouth(frame, landmarks):
    x_min = int(np.min(landmarks[:, 0]))
    x_max = int(np.max(landmarks[:, 0]))
    y_min = int(np.min(landmarks[:, 1]))
    y_max = int(np.max(landmarks[:, 1]))

    return frame[y_min:y_max, x_min:x_max]
