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

+ itertools
+ json
+ logging
+ math
+ matplotlib
+ numpy
+ os
+ pandas
+ PIL
+ sklearn.metrics
+ sys
+ tensorflow
+ tensorflow.python.framework
+ time

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

This project supposes that you have downloaded the Mapillary image dataset.

First of all, a data preprocessing step is applied on the bunch of images, so
as to normalize image file names and image sizes:

```bash python3 ./data_preprocessing.py ```

Then some neural network model may be generated before launching the
convolutional neural network training:

```bash python3 ./cnn_instance_building.py ```

And finally the model training itself may be undertaken:

```bash python3 ./cnn_train.py ```

Every module call is supposed to be made from the `source` repository.

# License

The MIT License

Copyright (c) 2017 Oslandia http://oslandia.com

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

___

by Raphael Delhome, Oslandia

September 2017
