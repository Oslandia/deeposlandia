% Author: R. Delhome
% Date: 18/12/11

# How to add a dataset?

Some steps have to be accomplished:

## Prepare the data repository

Following folders must be created and maintained:

+ `data/dataset/input/training/images` must contain raw training images
+ `data/dataset/input/training/labels` must contain training labels, either
  as images or text files (`json` and `geojson` are possible)
+ `data/dataset/input/validation/images` must contain raw validation images
+ `data/dataset/input/validation/labels` must contain validation images
+ `data/dataset/input/testing/images` must contain testing images
+ `data/dataset/preprocessed/` will contain preprocessed material (images and labels) that will be used by neural network models
+ `data/dataset/output` will contain neural network outputs (trained models)

## Generate pre-processed data

- Create a class that inheritates from `Dataset` (for a sake of
  clarity, declare it in a dedicated module) so as to describe the new dataset
- Define the class generator by defining labels on the `Dataset` manner
- Define a `populate` method in which images are preprocessed, and exploitable
  images and labels are generated on the file system (image files with a fixed
  square size).
- Add the new module as a dependency in `datagen.py`
- Manage the new dataset creation in `datagen.py` (*hint*: search for all
  occurrence of `aerial` or `mapillary` to know the accurate place)
- Add the dataset name to `AVAILABLE_DATASETS` variable in
  `deeposlandia/datasets/__init__.py`

## Test

+ Consider a little sample of your data (less than 5Mo), and reproduce the
  previous steps in `tests/data` folder
+ Write unit tests for :
    - dataset handling (see `tests/test_dataset.py` for examples)
	- generator verification (see `tests/test_generator.py` for examples)

## Model training

- Train a neural network model with the new created dataset:
  + use `paramoptim.py` for exploring several hyperparametrization and store
    the best model in `data/dataset/output/semantic_segmentation/checkpoint/`.
  + alternatively use the simpler `train.py` to train a single model. In such a
    case, you will have to manually copy the trained model from the instance
    folder to the global checkpoint folder and to create a `json` file that
    summarizes the model training parameters.

As an example that illustrates the required trained model files, in the
`aerial` dataset case we have:
- `data/aerial/output/semantic_segmentation/checkpoints/best-model-250.h5`
  that contains the trained model weights
- `data/aerial/output/semantic_segmentation/checkpoints/best-instance-250.json`
  that contains a single dictionary with values of validation accuracy
  (`val_acc`), batch size (`batch_size`), network, dropout, learning rate
  (`learning_rate`) and learning rate decay (`learning_rate_decay`).

```
{"val_acc": 0.9586366659402847, "batch_size": 20, "network": "unet", "dropout": 1.0, "learning_rate": 0.001, "learning_rate_decay": 1e-05}
```

## Display result onto the web application

- Link the app `static` folder to image repository:
  + in a development environment, update the `config.ini` and
    `config.ini.sample` files: they will manage a symbolic link creation
    towards the images
  + in a production environment: in your app repository, add a bunch of images
    into a dedicated folder that contains `images` and `labels` subfolders
- Add the dataset name in `webapp/main.py` docstring (*hint*: search for all
  occurrence of `aerial` or `mapillary` to know the accurate places)
- Specify the depicted image size for the dataset in `webapp/main.py` (see
  `recover_image_info` method)
- Create a new `html` web page dedicated to the new dataset, on the model of
  previous datasets
- Refer to this webpage by updating `index.html`
