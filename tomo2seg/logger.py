import logging
from logging import Formatter, StreamHandler
import sys


logger = logging.getLogger("tomo2seg")

logger.handlers = []
logger.propagate = False

fmt = "[%(asctime)s.%(msecs)03d] %(name)s :: %(levelname)s :: {%(filename)s:%(funcName)s:%(lineno)03d}" 
fmt += "\n%(message)s"
date_fmt = "%Y-%m-%d::%H:%M:%S"
formatter = Formatter(fmt, datefmt=date_fmt)

stdout_handler = StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)

logger.setLevel(logging.INFO)