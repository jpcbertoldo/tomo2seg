import logging
import os
import sys
from pathlib import Path

VOLUME_GITIGNORE = """
!*.metadata.yml
!*.raw.info
!*ground-truth-analysis*
*ground-truth-analysis*/*
!*ground-truth-analysis*/*.png
"""

ESTIMATION_VOLUME_GITIGNORE = """
!*.metadata.yml
!*confusion-matrix.png
!*volumetric-fraction.png
!*.classification-report-table.exact.csv
!*.classification-report-table.human.detail.txt
!*.classification-report-table.human.simple.txt
!debug_figs/
"""


def is_estimation_volume(path: Path) -> bool:
    return (
        path.is_dir() and
        "vol=" in path.name and
        "set=" in path.name and
        "model=" in path.name and
        "runid=" in path.name
    )


def is_volume(path: Path) -> bool:
    return (
        path.is_dir() and
        path.name.count(".") == 1 and
        (path.name + ".metadata.yml") in os.listdir(path) and
        (path.name + ".raw") in os.listdir(path)
    )


def is_model(path: Path) -> bool:
    return (
        path.is_dir() and
        "variables" in (path_contents := os.listdir(path)) and
        "assets" in path_contents and
        "saved_model.pb" in path_contents
    )


loglevel = sys.argv[1] if len(sys.argv) > 1 else logging.INFO

fmt = "%(levelname)s::%(name)s::{%(filename)s:%(funcName)s:%(lineno)03d}::[%(asctime)s.%(msecs)03d]"
fmt += "\n%(message)s\n"
date_fmt = "%Y-%m-%d::%H:%M:%S"

logger = logging.getLogger("unignore")

logger.handlers = []
logger.propagate = False

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

logger.addHandler(stdout_handler)

logger.setLevel(loglevel)

data_dir = Path("../data").resolve()

logger.debug(f"{data_dir=}")

assert data_dir.exists()

models_dir = data_dir / "models"

logger.debug(f"{models_dir=}")

assert models_dir.exists()

logger.info("Finding volumes, estimation volumes, and models to unignore.")

data_dir_ls = [
    data_dir / fname
    for fname in os.listdir(data_dir)
]

estimation_volumes = [
    path
    for path in data_dir_ls
    if is_estimation_volume(path)
]

volumes = [
    path
    for path in data_dir_ls
    if path not in estimation_volumes
    and is_volume(path)
]

master_model_dirs = os.listdir(models_dir)

models = [
    path
    for master_model_dir in master_model_dirs
    if (master_model_dir_path := models_dir / master_model_dir).is_dir()
    for fname in os.listdir(master_model_dir_path)
    if is_model(path := master_model_dir_path / fname)
]


with (data_dir / ".gitignore").open("w") as data_gitignore:

    data_gitignore.writelines([
        "!models/\n",
        "models/*\n",
    ])

    logger.info("Unignoring volume files.")

    for vol in volumes:

        logger.debug(f"Unignore {vol.name=}")

        data_gitignore.writelines([
            f"!{vol.name}/\n",
            f"{vol.name}/*\n",
        ])

        with (vol / ".gitignore").open("w") as vol_gitignore:
            vol_gitignore.write(VOLUME_GITIGNORE)

    logger.info("Unignoring estimation volume files.")

    for est_vol in estimation_volumes:

        logger.debug(f"Unignore {est_vol.name=}")

        data_gitignore.writelines([
            f"!{est_vol.name}/\n",
            f"{est_vol.name}/*\n",
        ])

        with (est_vol / ".gitignore").open("w") as est_vol_gitignore:
            est_vol_gitignore.write(ESTIMATION_VOLUME_GITIGNORE)


