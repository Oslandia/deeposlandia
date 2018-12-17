"""Deeposlandia package
"""

import logging

import daiquiri

__version__ = '0.4'

daiquiri.setup(level=logging.INFO,outputs=(
    daiquiri.output.Stream(formatter=daiquiri.formatter.ColorFormatter(
        fmt=("%(asctime)s :: %(levelname)s :: %(module)s :: "
             "%(funcName)s : %(color)s%(message)s%(color_stop)s"))),
))
logger = daiquiri.getLogger("root")
