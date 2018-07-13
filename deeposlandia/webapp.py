"""Flask web application for deeposlandia
"""

import daiquiri
from flask import (abort, Flask, jsonify, redirect,
                   render_template, request, send_from_directory, url_for)
import json
import logging
import numpy as np
import os
from PIL import Image
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
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def check_model(model):
    """Check if `model` is valid, *i.e.* equal to `feature_detection` or
    `semantic_segmentation`

    Parameters
    ----------
    model : str
        String to verify
    """
    if model not in MODELS:
        abort(404, "Model {} not found".format(model))

def check_dataset(dataset):
    """Check if `dataset` is valid, *i.e.* equal to `shapes` or
    `mapillary`

    Parameters
    ----------
    dataset : str
        String to verify
    """
    if dataset not in DATASETS:
        abort(404, "Dataset {} not found".format(dataset))

def allowed_file(filename):
    """Check if `filename` is really an image file name on the file system,
    *i.e.* composed of at least one '.' character, and which ends with an
    allowed extensions (namely, `jpg`, `jpeg` or `png`)

    Parameters
    ----------
    filename : str
        String to verify
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Route to application home page

    Returns
    -------
    Jinja template
        Template for application home page
    """
    return render_template("index.html")


@app.route("/predictor_demo/<string:model>/<string:dataset>")
def predictor_demo(model, dataset):
    """Route to a demo page dedicated to `model` and ̀dataset`

    Parameters
    ----------
    model : str
        Considered research problem (either `feature_detection` or `semantic_segmentation`)
    dataset : str
        Considered dataset (either `shapes` or `mapillary`)

    Returns
    -------
    Jinja template
        Template for demo web page fed with the specified model and an image
    filename
    """
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
    """Route to the deep learning predictor web page, that considers
    uploaded-by-client images

    Returns
    -------
    Jinja template
        Template for predictor web page fed with an image filename
    """
    filename = os.path.join("sample_image", "example.jpg")
    return render_template("predictor.html", example_image=filename)


@app.route("/demo_prediction")
def demo_prediction():
    """Route to a jsonified version of deep learning model predictions, for
    demo case

    Returns
    -------
    dict
        Deep learning model prediction for demo page (depends on the chosen
    model, either feature detection or semantic segmentation)
    """
    filename = request.args.get('img')
    filename = os.path.join("deeposlandia", filename[1:])
    dataset = request.args.get('dataset')
    agg_value = dataset == "mapillary"
    model = request.args.get('model')
    utils.logger.info("file: {}, dataset: {}, model: {}".format(filename,
                                                                dataset,
                                                                model))
    if model == "feature_detection":
        predictions = predict([filename], dataset, model, aggregate=agg_value)
        return jsonify(predictions)
    elif model == "semantic_segmentation":
        predictions = predict([filename], dataset, model, aggregate=agg_value,
                              output_dir=PREDICT_FOLDER)
        return jsonify(predictions)
    else:
        utils.logger.error(("Unknown model. Please choose "
                            "'feature_detection' or 'semantic_segmentation'."))
        return ""


@app.route("/prediction")
def prediction():
    """Route to a jsonified version of deep learning model predictions, for
    client tool

    Returns
    -------
    dict
        Deep learning model predictions
    """
    filename = os.path.basename(request.args.get('img'))
    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dataset = request.args.get('dataset')
    model = request.args.get('model')
    utils.logger.info("file: {}, dataset: {}, model: {}".format(dataset, model,
                                                                filename))
    predictions = predict([filename], "mapillary", "semantic_segmentation",
                          aggregate=True, output_dir=PREDICT_FOLDER)
    return jsonify(predictions)


@app.route('/uploads/<filename>')
def send_image(filename):
    """Route to uploaded-by-client images

    Returns
    -------
    file
        Image file on the server (see Flask documentation)
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/predictor", methods=['GET', 'POST'])
def upload_image():
    """Route to deep learning predictor that takes as an input a uploaded-by-client
    image (which is saved on the server); if the uploaded file is not valid,
    the method does a simple redirection

    Returns
    -------
    Jinja template
        Template for predictor web page fed with the uploaded image

    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            utils.logger.info('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            utils.logger.info('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_filename)
            return render_template('predictor.html', image_name=filename)


@app.route("/demo_image_selector")
def demo_image_selector():
    """Route to a random server file with corresponding label information

    Returns
    -------
    dict
        Dictionary that contains four items: `image_name` is the short name of
    an image that will be displayed by the app, `image_file` is the relative
    path of the image on the server, `label_file` is the relative path of its
    labelled version on the server, and `labels` is a dictionary that
    summarizes the label information for displyaing purpose
    """
    dataset = request.args.get('dataset')
    dataset_code = dataset + "_agg" if dataset == "mapillary" else dataset
    server_folder = os.path.join(app.static_folder, dataset_code, "images")
    client_folder = os.path.join(app.static_url_path, dataset_code, "images")
    filename = np.random.choice(os.listdir(server_folder))
    image_file = os.path.join(client_folder, filename)
    label_file = image_file.replace("images", "labels")
    if dataset == "mapillary" or dataset == "mapillary_agg":
        label_file = label_file.replace(".jpg", ".png")
    server_label_filename = os.path.join("deeposlandia", label_file[1:])
    server_label_image = np.array(Image.open(server_label_filename))
    size_aggregation = "224_aggregated" if dataset == "mapillary" else "64_full"
    with open(os.path.join("data", dataset, "preprocessed", size_aggregation
                           , "testing.json")) as fobj:
        config = json.load(fobj)
    actual_labels = np.unique(server_label_image.reshape([-1, 3]), axis=0)
    printed_labels = {item['category']: utils.RGBToHTMLColor(item['color'])
                      for item in config['labels']
                      if item['color'] in actual_labels}
    return jsonify(image_name=filename,
                   image_file=image_file,
                   label_file=label_file,
                   labels=printed_labels)
