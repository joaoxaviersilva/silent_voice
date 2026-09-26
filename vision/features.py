import numpy as np


LEFT_MOUTH_CORNER = 0
RIGHT_MOUTH_CORNER = 10
EPSILON = 1e-6


def normalize(landmarks):
    """
    Centraliza, redimensiona e alinha os landmarks da boca.

    Isso reduz diferenças causadas pela posição da pessoa,
    distância da câmera e inclinação da cabeça.
    """

    points = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "Os landmarks devem possuir o formato (N, 2)."
        )

    left_corner = points[LEFT_MOUTH_CORNER]
    right_corner = points[RIGHT_MOUTH_CORNER]

    center = (
        left_corner + right_corner
    ) / 2.0

    translated_points = points - center

    mouth_width = np.linalg.norm(
        right_corner - left_corner
    )

    if mouth_width < EPSILON:
        mouth_width = 1.0

    angle = np.arctan2(
        right_corner[1] - left_corner[1],
        right_corner[0] - left_corner[0],
    )

    cosine = np.cos(-angle)
    sine = np.sin(-angle)

    rotation_matrix = np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=np.float32,
    )

    aligned_points = (
        translated_points
        @ rotation_matrix.T
    )

    normalized_points = (
        aligned_points / mouth_width
    )

    return normalized_points.astype(
        np.float32
    )


def extract_sequence(sequence):
    """
    Converte uma sequência de landmarks em features temporais.

    Cada frame contém:
    - coordenadas normalizadas da boca;
    - velocidade dos landmarks em relação ao frame anterior.
    """

    if not sequence:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    features = []
    previous_shape = None

    for landmarks in sequence:
        normalized_landmarks = normalize(
            landmarks
        )

        current_shape = (
            normalized_landmarks
            .flatten()
            .astype(np.float32)
        )

        if previous_shape is None:
            velocity = np.zeros_like(
                current_shape
            )
        else:
            velocity = (
                current_shape
                - previous_shape
            )

        frame_features = np.concatenate(
            [
                current_shape,
                velocity,
            ]
        ).astype(np.float32)

        features.append(
            frame_features
        )

        previous_shape = current_shape

    return np.asarray(
        features,
        dtype=np.float32,
    )