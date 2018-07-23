"""Launch the webapp in debug mode.

WARNING: only for development purpose!!
"""

import os
from configparser import ConfigParser

from deeposlandia import utils
from deeposlandia.webapp import app

config = ConfigParser()
config.read("config.ini")
for link_name, path in config.items("symlink"):
    utils.create_symlink(os.path.join(app.static_folder, link_name), path)

app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.run(host="0.0.0.0", port=7897, debug=True)
