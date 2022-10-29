from robotik_anwendungen.camera import IPCamera
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    camera = IPCamera()
    filename = f'{time.strftime("%m_%d_%Y_%H_%M_%S")}.jpg'
    camera.save_current_frame(Path(f'{filename}'))
    logger.info(f'Successfully saved image at: {filename}')
