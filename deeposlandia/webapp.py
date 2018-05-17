"""Flask web application for deeposlandia
"""

import os
import numpy as np

import daiquiri
import logging

from flask import Flask, render_template, abort, request, jsonify


daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger("deeposlandia-webapp")


app = Flask(__name__)
app.config['ERROR_404_HELP'] = False
app.config['SWAGGER_UI_DOC_EXPANSION'] = 'list'

MODELS = ('feature_detection', 'semantic_segmentation')
DATASETS = ('mapillary', 'shapes')

def check_model(model):
    if model not in MODELS:
        abort(404, "Model {} not found".format(model))

def check_dataset(dataset):
    if dataset not in DATASETS:
        abort(404, "Dataset {} not found".format(dataset))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/request/')
def swagger_ui():
    return render_template("swagger-ui.html")


@app.route("/<string:model>/<string:dataset>")
def predictor_view(model, dataset):
    check_model(model)
    check_dataset(dataset)
    if dataset == "shapes":
        image_id = np.random.randint(0, 5000)
        filename = os.path.join("shape_{:05d}.png".format(image_id))
        print(filename)
        return render_template('shape_predictor.html',
                               model=model,
                               image_name=os.path.join("images", filename))
    else:
        return render_template('predictor.html', model=model, dataset=dataset)
