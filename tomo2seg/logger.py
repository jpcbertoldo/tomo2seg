import logging
import sys
from logging import Formatter, Logger, StreamHandler
from pathlib import Path
from pprint import PrettyPrinter


def get_formatter():
    fmt = "%(levelname)s::%(name)s::{%(filename)s:%(funcName)s:%(lineno)03d}::[%(asctime)s.%(msecs)03d]"
    fmt += "\n%(message)s\n"
    date_fmt = "%Y-%m-%d::%H:%M:%S"
    return Formatter(fmt, datefmt=date_fmt)


def add_file_handler(logger_: Logger, file: Path) -> None:
    logspath = str(file.absolute())
    fh = logging.FileHandler(logspath)
    fh.setFormatter(get_formatter())
    logger_.addHandler(fh)

    logger_.info(f"Added a new file handler to the logger. {logspath=}")


def dict2str(dic: dict) -> str:
    return PrettyPrinter(indent=4, compact=False).pformat(dic)


logger = logging.getLogger("tomo2seg")

logger.handlers = []
logger.propagate = False

stdout_handler = StreamHandler(sys.stdout)
stdout_handler.setFormatter(get_formatter())

logger.addHandler(stdout_handler)

logger.setLevel(logging.INFO)
