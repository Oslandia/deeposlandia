"""Launch the webapp in debug mode.

WARNING: only for development purpose!!
"""

import os
import sys

import daiquiri

from deeposlandia import utils
from deeposlandia.webapp.main import app, config


logger = daiquiri.getLogger(__name__)


if "symlink" not in config.sections():
    logger.error("Config.ini file does not contain any 'symlink' section.")
    sys.exit(1)
for link_name, path in config.items("symlink"):
    utils.create_symlink(os.path.join(app.static_folder, link_name), path)

app.config["DEBUG"] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.run(host="0.0.0.0", port=7897, debug=True)
