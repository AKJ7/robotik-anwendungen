import cv2 as cv
from threading import Thread
import warnings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class IPCamera:
    def __init__(self, url='http://192.168.178.20:8080/video'):
        self.url = url
        self.video_capture = cv.VideoCapture(self.url)
        self.current_frame = self.video_capture.read()[1]
        self.worker_thread = None
        self.stop_state = True

    def start(self):
        self.stop_state = False
        self.worker_thread = Thread(target=self._update_frame, args=())
        self.worker_thread.start()

    def _update_frame(self):
        while True:
            try:
                if self.stop_state is True:
                    logger.info('Breaking Working thread')
                    return
                self.current_frame = self.video_capture.read()[1]
            except Exception as e:
                warnings.warn(str(e))

    def get_current_frame(self):
        return self.current_frame

    def save_current_frame(self, path: Path):
        logger.info(f'Saving image to {path.absolute()}')
        cv.imwrite(str(path.absolute()), self.current_frame)

    def stop(self):
        logger.info('Stopping working thread')
        if self.worker_thread is not None:
            self.stop_state = True
            self.worker_thread.join()
