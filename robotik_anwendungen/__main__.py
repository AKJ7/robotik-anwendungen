import cv2 as cv
import logging
import time
from robotik_anwendungen.camera import IPCamera
from robotik_anwendungen.detector import Detector

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    camera = IPCamera()
    camera.start()
    cv.namedWindow('Camera Stream', cv.WINDOW_NORMAL)
    previous_image = cv.imread('../test/01_24_2022_03_11_08.jpg')
    glass_image = cv.imread('../test/01_24_2022_03_12_09.jpg')
    fps = 0
    duration = 0
    detector = Detector(previous_image, glass_image)
    time_start = time.time()
    while True:
        image = camera.get_current_frame()
        feature_image, levels = detector.get_level(image)
        duration = time.time() - time_start
        fps = 1 // duration
        beer_level, foam_level = levels[0] / detector.glass_height, levels[1] / detector.glass_height
        cv.putText(feature_image, f'FPS: {fps:.0f}', (360, 420), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(feature_image, f'Foam:{foam_level * 100: .2f}%', (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        cv.putText(feature_image, f'Beer:{beer_level * 100: .2f}%   `12', (15, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv.imshow('Camera Stream', feature_image)
        if (cv.waitKey(1) & 0xFF == 27) or cv.getWindowProperty('Camera Stream', cv.WND_PROP_VISIBLE) < 1:
            logger.info('Stopping')
            camera.stop()
            break
        time_start = time.time()
    cv.destroyAllWindows()
    exit(0)
