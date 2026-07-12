import numpy as np


def normalize(landmarks):
    """
    Centraliza e normaliza os landmarks da boca.
    """

    center = np.mean(landmarks, axis=0)
    landmarks = landmarks - center

    scale = np.max(np.linalg.norm(landmarks, axis=1))

    if scale < 1e-6:
        scale = 1.0

    return landmarks / scale


def extract_sequence(sequence):
    """
    Extrai features geométricas e temporais da sequência
    de landmarks da boca.

    Cada frame gera um vetor com:
        - altura da boca
        - largura da boca
        - MAR (Mouth Aspect Ratio)
        - velocidade da altura
        - velocidade da largura
        - velocidade do MAR
    """

    features = []

    previous = None

    for landmarks in sequence:

        landmarks = normalize(landmarks)

        # Cantos da boca
        left = landmarks[0]
        right = landmarks[10]

        # Centro do lábio superior
        upper = landmarks[3]

        # Centro do lábio inferior
        lower = landmarks[17]

        # Distâncias principais
        mouth_width = np.linalg.norm(left - right)
        mouth_height = np.linalg.norm(upper - lower)

        # Mouth Aspect Ratio
        mar = mouth_height / (mouth_width + 1e-6)

        current = np.array(
            [
                mouth_height,
                mouth_width,
                mar,
            ],
            dtype=np.float32,
        )

        if previous is None:
            velocity = np.zeros_like(current)
        else:
            velocity = current - previous

        feature = np.concatenate(
            [
                current,
                velocity,
            ]
        )

        features.append(feature)

        previous = current

    return np.array(features, dtype=np.float32)
