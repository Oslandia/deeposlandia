"""Deeposlandia package
"""

from configparser import ConfigParser
import logging
import os

import daiquiri

__version__ = "0.6.3.post1"


# Do not log Tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure the logger
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

# Deeposlandia supports feature detection (featdet) and semantic segmentation (semseg)
AVAILABLE_MODELS = ("featdet", "semseg")

# Configuration file handling
_DEEPOSL_CONFIG = os.getenv("DEEPOSL_CONFIG")
_DEEPOSL_CONFIG = (
    _DEEPOSL_CONFIG if _DEEPOSL_CONFIG is not None else "config.ini"
)
config = ConfigParser()
if os.path.isfile(_DEEPOSL_CONFIG):
    config.read(_DEEPOSL_CONFIG)
else:
    raise FileNotFoundError("No file config.ini!")
