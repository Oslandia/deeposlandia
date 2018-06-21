"""Flask web application for deeposlandia
"""

import daiquiri
from flask import (abort, Flask, jsonify, redirect,
                   render_template, request, send_from_directory, url_for)
import logging
import os
import random
from werkzeug.utils import secure_filename

from deeposlandia import utils
from deeposlandia.inference import predict

MODELS = ('feature_detection', 'semantic_segmentation')
DATASETS = ('mapillary', 'shapes')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
PROJECT_FOLDER = '/tmp/deeposlandia'
UPLOAD_FOLDER = os.path.join(PROJECT_FOLDER, 'uploads/')
PREDICT_FOLDER = os.path.join(PROJECT_FOLDER, 'predicted/')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

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


@app.route("/predictor_demo/<string:model>/<string:dataset>")
def predictor_demo(model, dataset):
    check_model(model)
    check_dataset(dataset)
    if dataset == "shapes":
        sample_filename = os.path.join("sample_image", "shape_example.png")
        return render_template('shape_demo.html', model=model,
                               sample_filename=sample_filename)
    else:
        sample_filename = os.path.join("sample_image", "example.jpg")
        return render_template('mapillary_demo.html', model=model,
                               sample_filename=sample_filename)


@app.route("/predictor")
def predictor():
    filename = os.path.join("sample_image", "example.jpg")
    return render_template("predictor.html", example_image=filename)


@app.route("/demo_prediction")
def demo_prediction():
    filename = request.args.get('img')
    filename = os.path.join("deeposlandia", filename[1:])
    dataset = request.args.get('dataset')
    model = request.args.get('model')
    utils.logger.info("file: {}, dataset: {}, model: {}".format(filename,
                                                                dataset,
                                                                model))
    if model == "feature_detection":
        predictions = predict([filename], dataset, model)
        predictions[filename] = {k: 100*round(predictions[filename][k], 2)
                                 for k in predictions[filename]}
        return jsonify(predictions)
    elif model == "semantic_segmentation":
        predictions = predict([filename], dataset, model,
                              output_dir=PREDICT_FOLDER)
        return jsonify(predictions)
    else:
        utils.logger.error(("Unknown model. Please choose "
                            "'feature_detection' or 'semantic_segmentation'."))
        return ""


@app.route("/prediction")
def prediction():
    filename = request.args.get('img').split("/")[-1]
    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dataset = request.args.get('dataset')
    model = request.args.get('model')
    utils.logger.info("file: {}, dataset: {}, model: {}".format(dataset, model,
                                                                filename))
    predictions = predict([filename], "mapillary", "semantic_segmentation",
                          aggregate=False, output_dir=PREDICT_FOLDER)
    return jsonify(predictions)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/predictor", methods=['GET', 'POST'])
def upload_image():
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
            return render_template('predictor.html', image_name=filename)

@app.route("/mapillary_image_selector")
def mapillary_image_selector():
    dataset = request.args.get('dataset')
    utils.logger.info("DATASET: {}".format(dataset))
    folder = os.path.join("deeposlandia", "static", dataset, "images")
    utils.logger.info("FOLDER: {}".format(folder))
    filename = random.choice(os.listdir(folder)).split(".")[0]
    utils.logger.info("FILENAME: {}".format(filename))
    return jsonify(image_name=filename)
