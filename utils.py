from loguru import logger
def setup_logger(file_name):
    logger.remove()
    logger.add(f"./logs/{file_name}-debug.log", format="{time} {level} {message}", level="DEBUG", enqueue=True)
    logger.add(f"./logs/{file_name}-info.log", format="{time} {level} {message}", level="INFO", enqueue=True, backtrace=True)
    logger.add(f"./logs/{file_name}-error.log", format="{time} {level} {message}", level="ERROR", enqueue=True, backtrace=True, diagnose=True)
