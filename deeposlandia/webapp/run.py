"""Launch the webapp in debug mode.

WARNING: only for development purpose!!
"""

from configparser import ConfigParser
import os
import sys

from deeposlandia import utils
from deeposlandia.webapp.main import app

config = ConfigParser()
CONFIG_FILENAME = "config.ini"
if os.path.isfile(CONFIG_FILENAME):
    config.read(CONFIG_FILENAME)
else:
    utils.logger.error("No file config.ini!")
    sys.exit(1)
if not "symlink" in config.sections():
    utils.logger.error("Config.ini file does not contain any 'symlink' section.")
    sys.exit(1)
for link_name, path in config.items("symlink"):
    utils.create_symlink(os.path.join(app.static_folder, link_name), path)

app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.run(host="0.0.0.0", port=7897, debug=True)
