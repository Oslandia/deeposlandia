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

+ matplotlib
+ numpy
+ pandas
+ PIL
+ sklearn.metrics
+ tensorflow
+ tensorflow.python.framework

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
+ [models](./models) contains a set of `.json` configuration files that
  summarize each model through their main hyperparameter (layer number, types,
  and shapes)
+ [results](./results) gathers some `.csv` files that describe the model
  training evolution, epoch after epoch; more than 200 features are recorded in
  each file, this high number being essentially due to the label quantity (66
  labels, each of which being evaluated through accuracy, precision and recall
  of predictions)

# Running the code

This project supposes that you have downloaded the Mapillary image dataset. The
following program calls are supposed to be made from the `source` repository.

First of all, the Mapillary glossary can be printed for information purpose
with the following command:

```
python3 main.py -g -d ../data
```

The `-g` argument makes the program recover the data glossary within the
repository indicated by `-d` command. By default, the program will look at
`../data` (it hypothesizes that the data repository is at the project root, or
that a symbolic link points to it).

As a prerequisite of the training, a data preprocessing step is applied on the
bunch of images (`-p` argument), so as to normalize image file names and image
sizes. The desired size is given after the `-s` argument, under the format `-s
<width> <height>`:

```
python3 main.py -p -d ../data -s 512 512
```

Then the model training itself may be undertaken:

```
python3 main.py -d ../data -s 512 512 -n cnn_mapil -c 2 -f 1 -e 5
```

In this example, the 512*512 images produced by previous command will be
exploited. A network called `cnn_mapil` will be built (`cnn_mapil` is the
default value), it will be composed of three convolutional layers followed by
two fully-connected layers (respectively arguments `-c` and `-f`, with default
values of 2 and 1). The network name and the layer quantities are useful for
checkpoints and results naming. Here the training will take place for five
epoches, as indicated by the `-e` argument. One epoch refers to the scan of
every training image.

Some other arguments may be parametrized for running this program:
+ `-h`: show the help message
+ `-b`: indicate the batch size (number of images per training batch)
+ `-do`: percentage of dropped out neurons during training process
+ `-l`: IDs of considered labels during training (between 1 and 66 if Mapillary
  data are considered); if 1, the problem becomes a mono-label problem, the
  `softmax` activation function is preferred to the `sigmoid` one in order to
  compute final logits
+ `-m`: training mode (either `training`, `testing` or `both`)
+ `-r`: decaying learning rate components (starting learning rate, decay steps
  and decay rate)
+ `-ss`: log periodicity during training (print dashboard on log each `ss`
  steps)
+ `-t`: training limit, measured as a number of iteration; overload the epoch
  number if specified
+ `-w`: weighting policy to apply on label contributions to loss (either `base`
  if no weights, `global` if contributions are weighted according to global
  label popularity, `batch` if they are weighted according to label popularity
  within each batch, `centeredglobal` if global weighting with less weight for
  medium-popularity labels, `centeredbatch` if batch weighting with less weight
  for medium-popularity labels)

___

Oslandia, December 2017
