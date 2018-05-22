"""Flask web application for deeposlandia
"""

import daiquiri
from flask import (abort, Flask, jsonify, redirect,
                   render_template, request, send_from_directory, url_for)
import logging
import numpy as np
import os
from werkzeug.utils import secure_filename

from deeposlandia import utils
from deeposlandia.inference import predict

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
        filename = os.path.join("sample_image", "shape_example.png")
        return render_template('shape_predictor.html',
                               model=model,
                               image_name=filename)
    else:
        filename = os.path.join("sample_image", "example.jpg")
        return render_template('predictor.html', model=model,
                               dataset=dataset, example_image=filename)


@app.route("/_shape_prediction")
def shape_prediction():
    filename = request.args.get('img')
    filename = os.path.join("deeposlandia", filename[1:])
    model = request.args.get('model')
    utils.logger.info("file: {}, dataset: shapes, model: {}".format(filename, model))
    predictions = predict([filename], "shapes", model)
    predictions[filename] = {k: 100*round(predictions[filename][k], 2)
                             for k in predictions[filename]}
    return jsonify(predictions)
