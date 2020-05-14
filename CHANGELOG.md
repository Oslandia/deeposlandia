# Changelog

Pour suivre l'évolution des différentes versions.
Le format de ce fichier est basé sur Keep a Changelog,
et ce projet respecte le Semantic Versioning.
Les sections conserveront leur nom en anglais.

## Unreleased

## v0.6.3.post1 (2020-05-14)

*A nicer Pypi description*

### Fixed

- From `https://github.com/Oslandia/deeposlandia/blob/master/images/...` to `https://github.com/Oslandia/deeposlandia/raw/master/images/...`

## v0.6.3 (2020-05-14)

*A nice Pypi description*

### Added

- `long_description_content_type` argument in `setup()` function.

### Modified

- From relative image paths to absolute image paths, in `README.md`.

## v0.6.2 (2020-05-14)

*Postprocessing improvement*

### Added

- `--nb-tiles-per-image` as a new argument for `datagen` command.
- A progress bar for inference processes (#153)

### Changed

- `utils.prepare_output_folder()` returns now a dictionary of all useful output paths
- Some dependency updates (Tensorflow, opencv, pillow, keras, daiquiri)
- The preprocessing has been modified for geographic datasets: `-t`, `-v` and `-T` now
  refer to raw images, the amount of preprocessed tiles being obtained by a combination
  of `--nb-tiles-per-image` and these last arguments.
- The tile grid becomes optional for postprocessing (#155).

### Fixed

- Draws *without replacement* instead of *with replacement* in the case of preprocessing
  of geographic dataset testing images (`np.random.choice` wrong parameterization). #146

### Security

- `pillow` was updated to `7.1.1` (moderate severity vulnerability alert for
  `pillow<6.2.2`)

### Removed

- `sys.exit` statements (#150)

## v0.6.1 (2020-04-01)

*Packaging clean-up*

When preparing a major release, or an old release, you necessarily forget details.

### Changed

- Package version 0.5 -> 0.6.1
- Long description

## v0.6 (2020-04-01)

*Georeferenced dataset post-processing*

This release essentially copes with the georeferenced dataset, one may now post-process
the results, so as to visualize labelled masks as raster. A vectorized version of each
prediction is also available.

As another major evolution, `deeposlandia` now has a Command-Line Interface (CLI). The
available commands are `datagen`, `train`, `infer` and `postprocess` respectively for
generating preprocessed datasets, training neural networks, doing inference and
post-processing neural network outputs.

### Added

- Set up a Command-Line Interface (#90).
- Consider `RGBA` images and warns the user as this format is not handled by the web app
  (#107).
- Consider geometric treatments in a dedicated module, add vector-to-raster and
  raster-to-vector transformation steps ; save postprocessed images as vector and raster
  files (#119).
- Postprocess aerial images so as to produce predicted rasters (#118, #126, #127).
- Add missing test files for Tanzania dataset.
- Some information about GDPR in the web app (#113).
- Improve unit tests dedicated to georeferenced data processing (#104).

### Changed

- Label folders are standardized (`labels`), in particular this folder name replaces `gt`
  for `Aerial` dataset (#139).
- Always use the best existing model, instead of parametrizing the access to the model
  (#135).
- Broken images are considered, hence not serialized onto the file system (#129).
- The georeferenced aerial datasets are updated and factorized into a generic
  `GeoreferencedDataset` class, the test files are updated accordingly (#128).
- Deep learning model are now known as `featdet` and `semseg` instead of
  `feature_detection` and `semantic_segmentation` (#133).
- Update the training metric history when using a existing trained model (#102).
- Move the documentation to a dedicated folder.
- Some code cleaning operations, using `black` and `flake8` (#120).
- Update dependencies, especially `Tensorflow`, due to vulnerability issues.
- Fix the unit tests for Tanzania dataset population (#111).
- The process quantity is an argument of `populate()` functions, in order to implement
  multiprocessing (#110).
- Logger syntax has been refactored (%-format) (#103).

### Removed

- The concept of "agregated dataset" is removed, as we consider a home-made Mapillary
  dataset version. As a consequence, some input/output folder paths have been updated
  (#134).
- The hyperparameter optimization script (`paramoptim.py`) has been removed, `train.py`
  can handle several value for each parameter (#125).

## v0.5 (2019-01-24)

*Georeferenced datasets and web application*

Some new datasets focusing on building footprint detection have been introduced in the
framework, namely Inria Aerial Image dataset and Open AI Tanzania dataset.

Some new state-of-the-art deep neural network architectures have been implemented to
enrich the existing collection, and design more sophisticated models.

Furthermore a bunch of Jupyter notebooks has been written to make the framework usage
easier, and clarify deep learning pipelines, from dataset description to model training
and inference.

And last but not least, a light Flask Web application has been developed to showcase some
deep learning predictions. Oslandia hosts this Web app at
http://data.oslandia.io/deeposlandia.

## v0.4 (2018-05-03)

*Train convolutional neural networks with Keras API*

This new release is characterized by the transition from the TensorFlow library to the
Keras library so as to train neural networks and predict image labels.

Additionally, the code has been structured in a production-like purpose:

- the program modules have been moved to a deeposlandia repository;
- a tests repository contains a bunch of tests that guarantee the code validity;
- a setup.py file summarizes the project description and target. Some complements may
  arise in order to publish the project on Pypi.

## v0.3.2 (2018-03-28)

*Validate and test the trained a wider range of TensorFlow models*

In this patch, a more mature code is provided:

- Dataset handling is factorized, we can now consider Mapillary or shape datasets
  starting from a common Dataset basis
- Model handling is factorized, we can generate feature detection models or semantic
  segmentation models, with common behaviors (basic layer creation, for instance)
- Some state-of-the-art models have been implemented (VGG, Inception)
- A base of code has been deployed for considering Keras API (the switch from TensorFlow
  to Keras will be the object of a next minor release)

## v0.3.1 (2018-03-13)

*Validate and test the trained model (Minor README fixes)*

Fix the 0.3 release with minor changes around README.md file (picture updates,
essentially).

## v0.3 (2018-03-13)

*Validate and test the trained model*

- Add a single-batched validation phase during training process, the corresponding
  metrics are logged onto Tensorboard so as to be compared with training metrics (same
  graphs) ;
- Add a model inference module, that call the test() method of
  ConvolutionalNeuralNetwork: it takes a trained model as an input, and infer label
  occurrences on a image testing set ;
- Manage the Tensorboard monitoring in a more clever way ;
- Add the possibility to gather similar labels for Mapillary dataset: by aggregating
  them, the number of labels decreases and the model may become easier to
  train. :warning: With this new feature, the dataset structure in json files has been
  modified: the labels keys are now dictionaries (instead of a lists) that link class ids
  (keys) and label occurrences (values), for each image.

## v0.2 (2018-01-17)

*Object-oriented convolutional neural network*

This new release provide an improved version of the project by considering
object-oriented programming.

- The project is structured around Dataset and ConvolutionalNeuralNetwork classes. These
  classes are written in dedicated modules.
- As a consequence, the main module contains only program-specific code (argument
  handling).
- A second dataset has been introduced to the project (geometric shapes), so as to make
  development easier and more reliable.

## v0.1 (2017-12-19)

*Street-scene object detection*

This repository runs a convolutional neural network on Mapillary Vistas Dataset, so as to
detect a range of street-scene objects (car, pedestrian, street-lights, pole,
... ). Developments are still under progress, as the model is unable to provide a
satisfying detection yet.
