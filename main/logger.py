import logging
from dataclasses import dataclass


@dataclass
class Logger:
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %I:%M:%S %p')
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        if not self.logger.hasHandlers():
            self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger