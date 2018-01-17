# Motivation

In this project we use a set of images provided
by [Mapillary](https://www.mapillary.com/), in order to investigate on the
presence of some typical street-scene objects (vehicles, roads,
pedestrians...). Mapillary released this dataset recently, it
is [still available on its website](https://www.mapillary.com/dataset/vistas)
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

# Dependencies

This project needs to load the following Python dependencies:

+ cv2
+ logging
+ matplotlib
+ numpy
+ pandas
+ PIL
+ tensorflow

As a remark, the code has been run with Python3 (version 3.5).

# Content

Contain some Python materials designed to illustrate the Tensorflow library
(snippets and notebooks)

+ [sources](./sources) contains Python modules that train a convolutional
  neural network based on the Mapillary street image dataset
+ [images](./images) contains some example images to illustrate the Mapillary
  dataset as well as some preprocessing analysis results

Additionally, running the code may generate extra repositories:

+ [checkpoints](./checkpoints) refers to trained model back-ups, they are
  organized with respect to models
+ [graphs](./graphs) is organized like `checkpoints` repository, it contains
  `Tensorflow` graphs corresponding to each neural network

# Running the code

This project supposes that you have downloaded the Mapillary image dataset. The
following program calls are supposed to be made from the `source` repository.

First of all, the Mapillary glossary can be printed for information purpose
with the following command:

```
python3 train.py -g -d mapillary -dp ../data
```

The `-g` argument makes the program recover the data glossary that corresponds
to the dataset indicated by `-d` command (the program expects `mapillary` or
`shape`). By default, the program will look for the glossary in `../data`
repository (*i.e.* it hypothesizes that the data repository is at the project
root, or that a symbolic link points to it). This behavior may be changed
through `-dp` argument.

Then the model training itself may be undertaken:

```
python3 train.py -d mapillary -dp ../data -n mapcnn -s 512 -e 5
```

In this example, 512*512 images will be exploited (either after a
pre-processing step for `mapillary` dataset, or after random image generations
for `shape` dataset). A network called `mapcnn` will be built (`cnnmapil` is
the default value). The network name is useful for checkpoints, graphs and
results naming. Here the training will take place for five epoches, as
indicated by the `-e` argument. One epoch refers to the scan of every training
image.

Some other arguments may be parametrized for running this program:
+ `-h`: show the help message
+ `-b`: indicate the batch size (number of images per training batch, 20 by
  default)
+ `-dn`: dataset name (`training`, `validation` or `testing`), useful for image
  storage on file system
+ `-do`: percentage of dropped out neurons during training process
+ `-l`: IDs of considered labels during training (between 1 and 65 if
  `mapillary` dataset is considered)
+ `-r`: decaying learning rate components; can be one floating number (constant
  learning rate) or three ones (starting learning rate, decay steps and decay
  rate) if learning rate has to decay during training process
+ `-ss`: log periodicity during training (print dashboard on log each `ss`
  steps)
+ `-t`: training limit, measured as a number of iteration; overload the epoch
  number if specified

# TensorBoard

The model monitoring is ensured through Tensorboard usage. For more details
about this tool and downloading instructions, please check on the
corresponding [Github project](https://github.com/tensorflow/tensorboard) or
the
[TensorFlow documentation](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

The network graph is created under `<datapath>/graph/<network-name>` (*e.g.*
`../data/mapcnn`).

To check the training process, a simple command must be done on your command prompt:

```
tensorboard --logdir <datapath>/graph/<network-name> --port 6006
```

Be careful, if the path given to `--logdir` argument do not correspond to those
created within the training, the Tensorboard dashboard won't show anything.

An example of visualization for scalar variables (*e.g.* loss, learning rate,
true positives...) is provided in the following figure:

[-> tensorboard example](./images/tensorboard_example.png)

___

Oslandia, December 2017
