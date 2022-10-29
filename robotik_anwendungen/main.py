from robotik_anwendungen.camera import IPCamera
import cv2 as cv
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    camera = IPCamera()
    camera.start()
    cv.namedWindow('Camera Stream')
    while True:
        image = camera.get_current_frame()
        cv.imshow('Camera Stream', image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
    exit(0)
