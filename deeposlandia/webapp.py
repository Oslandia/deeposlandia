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

MODELS = ('feature_detection', 'semantic_segmentation')
DATASETS = ('mapillary', 'shapes')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = '/tmp/deeposlandia/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger("deeposlandia-webapp")

app = Flask(__name__)
app.config['ERROR_404_HELP'] = False
app.config['SWAGGER_UI_DOC_EXPANSION'] = 'list'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def check_model(model):
    if model not in MODELS:
        abort(404, "Model {} not found".format(model))

def check_dataset(dataset):
    if dataset not in DATASETS:
        abort(404, "Dataset {} not found".format(dataset))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")


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


@app.route("/shape_prediction")
def shape_prediction():
    filename = request.args.get('img')
    filename = os.path.join("deeposlandia", filename[1:])
    model = request.args.get('model')
    utils.logger.info("file: {}, dataset: shapes, model: {}".format(filename, model))
    predictions = predict([filename], "shapes", model)
    predictions[filename] = {k: 100*round(predictions[filename][k], 2)
                             for k in predictions[filename]}
    return jsonify(predictions)


@app.route("/prediction")
def prediction():
    filename = request.args.get('img').split("/")[-1]
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dataset = request.args.get('dataset')
    model = request.args.get('model')
    utils.logger.info("file: {}, dataset: {}, model: {}".format(filename, dataset, model))
    predictions = predict([full_filename], dataset, model, aggregate=True)
    predictions[full_filename] = {k: 100*round(predictions[full_filename][k], 2)
                             for k in predictions[full_filename]}
    return jsonify(predictions)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/<string:model>/<string:dataset>", methods=['GET', 'POST'])
def upload_image(model, dataset):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_filename)
            return render_template('predictor.html', model="feature_detection",
                                   dataset="mapillary", image_name=filename)
