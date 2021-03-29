from dataclasses import dataclass
from functools import wraps
from pathlib import Path


def mkdir_ok(property_):
    """
    Make sure that a directory returned from a property exists.
    """

    @wraps(property_)
    def wrapper(self) -> Path:
        dir_: Path = property_(self)
        dir_.mkdir(exist_ok=True)
        return dir_

    return wrapper


@dataclass
class BaseOutputs:

    root_dir: Path

    def __post_init__(self):

        assert self.root_dir.is_dir()
