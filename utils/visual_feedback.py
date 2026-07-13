import cv2
import numpy as np


def success():
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    img[:] = (0, 255, 0)
    cv2.imshow("OK", img)
    cv2.waitKey(500)
    cv2.destroyWindow("OK")
