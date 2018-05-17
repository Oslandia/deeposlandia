"""Launch the webapp in debug mode.

WARNING: only for development purpose!!
"""

from deeposlandia.webapp import app

app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.run(port=7987, debug=True)
