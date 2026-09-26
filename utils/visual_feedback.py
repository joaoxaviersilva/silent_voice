import cv2
import numpy as np


def success(
    message="Frase reconhecida!",
    duration_ms=800,
):
    if duration_ms < 1:
        raise ValueError(
            "A duracao deve ser maior que zero."
        )

    window_name = "Silent Voice"

    image = np.zeros(
        (300, 500, 3),
        dtype=np.uint8,
    )

    image[:] = (0, 255, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    text_size, _ = cv2.getTextSize(
        message,
        font,
        font_scale,
        thickness,
    )

    text_width, text_height = text_size

    position_x = (image.shape[1] - text_width) // 2
    position_y = (image.shape[0] + text_height) // 2

    cv2.putText(
        image,
        message,
        (position_x, position_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    cv2.imshow(window_name, image)
    cv2.waitKey(duration_ms)
    cv2.destroyWindow(window_name)