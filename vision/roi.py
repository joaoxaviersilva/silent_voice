import cv2
import numpy as np


def crop_mouth(frame, landmarks, padding=10, output_size=(128, 128)):
    """
    Recorta a região da boca utilizando os landmarks do FaceMesh.

    Args:
        frame: imagem da câmera.
        landmarks: pontos da boca.
        padding: margem ao redor da boca.
        output_size: tamanho final do ROI.

    Returns:
        Região da boca redimensionada.
    """

    h, w = frame.shape[:2]

    x_min = max(0, int(np.min(landmarks[:, 0])) - padding)
    x_max = min(w, int(np.max(landmarks[:, 0])) + padding)

    y_min = max(0, int(np.min(landmarks[:, 1])) - padding)
    y_max = min(h, int(np.max(landmarks[:, 1])) + padding)

    roi = frame[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return None

    roi = cv2.resize(roi, output_size)

    return roi