"""Deeposlandia package
"""

from configparser import ConfigParser
import logging
import os
import sys

import daiquiri

__version__ = "0.5"

daiquiri.setup(
    level=logging.INFO,
    outputs=(
        daiquiri.output.Stream(
            formatter=daiquiri.formatter.ColorFormatter(
                fmt=(
                    "%(asctime)s :: %(levelname)s :: %(module)s :: "
                    "%(funcName)s : %(color)s%(message)s%(color_stop)s"
                )
            )
        ),
    ),
)
logger = daiquiri.getLogger("root")

AVAILABLE_MODELS = ("featdet", "semseg")

_DEEPOSL_CONFIG = os.getenv("DEEPOSL_CONFIG")
_DEEPOSL_CONFIG = (
    _DEEPOSL_CONFIG if _DEEPOSL_CONFIG is not None else "config.ini"
)
config = ConfigParser()
if os.path.isfile(_DEEPOSL_CONFIG):
    config.read(_DEEPOSL_CONFIG)
else:
    logger.error("No file config.ini!")
    sys.exit(1)
