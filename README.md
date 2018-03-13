# Motivation

## Mapillary dataset

In this project we use a set of images provided
by [Mapillary](https://www.mapillary.com/), in order to investigate on the
presence of some typical street-scene objects (vehicles, roads,
pedestrians...). Mapillary released this dataset on July 2017, it
is [available on its website](https://www.mapillary.com/dataset/vistas)
and may be downloaded freely for a research purpose.

As inputs, Mapillary provides a bunch of street scene images of various sizes
in a `images` repository, and the same images after filtering process in
`instances` and `labels` repositories. The latter is crucial, as the filtered
images are actually composed of pixels in a reduced set of colors. Actually,
there is one color per object types; and 66 object types in total.

![Example of image, with its filtered version](./images/MVD_M2kh294N9c72sICO990Uew.png)

There are 18000 images in the training set, 2000 images in the validation set,
and 5000 images in the testing set. The testing set is proposed only for a
model test purpose, it does not contain filtered versions of images.

## Shape dataset

To complete the project, and make the test easier, a randomly-generated shape model is also
available. In this dataset, some simple coloured geometric shapes are inserted into each picture,
on a total random mode. There can be one rectangle, one circle and/or one triangle per image, or
neither of them. Their location into each image is randomly generated (they just can't be too close
to image borders). The shape and background colors are randomly generated as well.

The picture below shows an example of image generated in this way:

![Example of shape image](./images/shape_00000.png)

# Dependencies

This project needs to load the following Python dependencies:

+ cv2
+ logging
+ matplotlib
+ numpy
+ pandas
+ PIL
+ tensorflow

These dependencies are stored in `requirements.txt` located at the project root. As a remark, the
code has been run with Python3 (version 3.5).

# Content

The project contains some Python materials designed to illustrate the Tensorflow library (snippets
and notebooks)

+ [article](./article) contains the original text of articles that have been published
  on [Oslandia blog](http://oslandia.com/en/blog/) on this topic
+ [images](./images) contains some example images to illustrate the Mapillary dataset as well as
  some preprocessing analysis results
+ [notebooks](./notebooks) contains some Jupyter notebooks that aim at describing data or basic
  neural network construction
+ [sources](./sources) contains Python modules that train a convolutional neural network based on
  the Mapillary street image dataset

Additionally, running the code may generate extra repositories:

+ [checkpoints](./checkpoints) refers to trained model back-ups, they are
  organized with respect to models
+ [graphs](./graphs) is organized like `checkpoints` repository, it contains
  `Tensorflow` graphs corresponding to each neural network
+ [chronos](./chronos) allows to store some training execution times, if wanted

These repository are located at the data repository root.

# Running the code

This project supposes that you have downloaded the Mapillary image dataset. The
following program calls are supposed to be made from the `source` repository.

## Printing Mapillary glossary

First of all, the Mapillary glossary can be printed for information purpose
with the following command:

```
python3 train.py -g -d mapillary -s 256 -dp ./any-data-path
```

The `-g` argument makes the program recover the data glossary that corresponds to the dataset
indicated by `-d` command (the program expects `mapillary` or `shapes`). By default, the program
will look for the glossary in `../data` repository (*i.e.* it hypothesizes that the data repository
is at the project root, or that a symbolic link points to it). This behavior may be changed through
`-dp` argument. By default, the image characteristics are computed starting from resized images of
512 * 512 pixels, that can be modified with the `-s` argument.

As an easter-egg feature, label popularity (proportion of images where the label appears in the
dataset) is also printed for each label.

## Model training

Then the model training itself may be undertaken:

```
python3 train.py -dp ../data -d mapillary -n mapcnn -s 512 -e 5
```

In this example, 512*512 images will be exploited (either after a
pre-processing step for `mapillary` dataset, or after random image generations
for `shape` dataset). A network called `mapcnn` will be built (`cnnmapil` is
the default value). The network name is useful for checkpoints, graphs and
results naming. Here the training will take place for five epoches, as
indicated by the `-e` argument. One epoch refers to the scan of every training
image.

Some other arguments may be parametrized for running this program:
+ `-a`: aggregate labels (*e.g.* `car`, `truck` or `caravan`... into a `vehicle` labels)
+ `-b`: indicate the batch size (number of images per training batch, 20 by
  default)
+ `-c`: indicates if training time must be measured
+ `-do`: percentage of dropped out neurons during training process
+ `-h`: show the help message
+ `-it`: number of training images (default to 18000, according to the Mapillary dataset)
+ `-iv`: number of validation images (default to 200, regarding computing memory limitation, as
  validation is done at once)
+ `-l`: IDs of considered labels during training (between 1 and 65 if
  `mapillary` dataset is considered)
+ `-ls`: log periodicity during training (print dashboard on log each `ss`
  steps)
+ `-m`: monitoring level on TensorBoard, either 0 (no monitoring), 1 (monitor main scalar tensor),
  2 (monitor all scalar tensors), or 3 (full-monitoring, including histograms and images, mainly
  for a debugging purpose)
+ `-ns`: neural network size for feature detection problem, either `small` (default value), or
  `medium`, the former being composed of 3 convolution+pooling operation and 1 fully-connected
  layer, whilst the latter is composed of 6 convolution+pooling operation plus 2 fully-connected
  layers.
+ `-r`: decaying learning rate components; can be one floating number (constant
  learning rate) or three ones (starting learning rate, decay steps and decay
  rate) if learning rate has to decay during training process
+ `-ss`: back-up periodicity during training (back-up the TensorFlow model into a `checkpoints`
  sub-directory each `ss` steps)
+ `-t`: training limit, measured as a number of iteration; overload the epoch
  number if specified
+ `-vs`: validation periodicity during training (run the validation phase on the whole validation
  dataset each `ss` steps)

## Model testing

Trained models may be tested after the training process. Once a model is trained, a checkpoint
structure is recorded in `<datapath>/<dataset>/checkpoints/<network-name>`. It is the key for
inference, as the model state after training is stored into it.

The model testing is done as follows:

```
python3 test.py -dp ../data -d mapillary -n mapcnn_256_small -i 1000 -b 100 -ls 100
```

+ `-b`: testing image batch size (default to 20)
+ `-d`: dataset (either `mapillary` or `shapes`)
+ `-dp`: data path in which the data are stored onto the computer (the dataset content is located
  at `<datapath>/<dataset>`)
+ `-i`: number of testing images (default to 5000, according to the Mapillary dataset)
+ `-ls`: log periodicity during training (print dashboard on log each `ss`
  steps)
+ `-n`: instance name, under the format `<netname>_<imsize>_<netsize>`, that allows to recover the
  model trained with the network name `<netname>`, image size of `<imsize>*<imsize>` pixels and a
  neural network of size `<netsize>` (either `small` or `medium`).

# TensorBoard

The model monitoring is ensured through Tensorboard usage. For more details
about this tool and downloading instructions, please check on the
corresponding [Github project](https://github.com/tensorflow/tensorboard) or
the
[TensorFlow documentation](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

The network graph is created under `<datapath>/<dataset>/graph/<network-name>` (*e.g.*
`../data/mapillary/graph/mapcnn`).

To check the training process, a simple command must be done on your command prompt:

```
tensorboard --port 6006 --logdir=<datapath>/<dataset>/graph/<network-name>
```

Be careful, if the path given to `--logdir` argument do not correspond to those created within the
training, the Tensorboard dashboard won't show anything. As a remark, several run can be showed at
the same time; in such a case, `--logdir` argument is composed of several path separated by commas,
and graph instances may be named as follows:

```
tensorboard --port 6006 --logdir=n1:<datapath>/<dataset>/graph/<network-name-1>,n2:<datapath>/<dataset>/graph/<network-name-2>
```

An example of visualization for scalar variables (*e.g.* loss, learning rate,
true positives...) is provided in the following figure:

[-> tensorboard example](./images/tensorboard_example.png)

___

Oslandia, March 2018
