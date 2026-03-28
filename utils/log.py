import logging


class Logger:
    def __init__(self):
        self.debug = self.setup_custom_logger("Debug")

    @staticmethod
    def setup_custom_logger(name) -> logging.Logger:
        formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        file_handler = logging.FileHandler("exec.log")
        file_handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            logger.addHandler(handler)
            logger.addHandler(file_handler)
        return logger
