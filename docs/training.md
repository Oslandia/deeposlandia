The model training may be undertaken as follows:

```
python deeposlandia/train.py -M feature_detection -D mapillary -s 512 -e 5
```

In this example, 512 * 512 Mapillary images will be exploited from training a
feature detection model. Here the training will take place for five epoches. An
inference step is always undertaken at the end of the training.

Here comes the parameter handled by this program:
+ `-a`: aggregate labels (*e.g.* `car`, `truck` or `caravan`... into a
  `vehicle` labels); do nothing if applied to `shapes` dataset.
+ `-b`: indicate the batch size (number of images per training batch, 50 by
  default).
+ `-D`: dataset (either `mapillary` or `shapes`).
+ `-d`: percentage of dropped out neurons during training process. Default
  value=1.0, no dropout.
+ `-e`: number of epochs (one epoch refers to the scan of every training
  image). Default value=0, the model is not trained, inference is done starting
  from the last trained model.
+ `-h`: show the help message.
+ `-ii`: number of testing images (default to 5000, according to the Mapillary
  dataset).
+ `-it`: number of training images (default to 18000, according to the
  Mapillary dataset).
+ `-iv`: number of validation images (default to 2000, according to the
  Mapillary dataset).
+ `L`: starting learning rate. Default to 0.001.
+ `l`: learning rate decay (according to
  the [Adam optimizer definition](https://keras.io/optimizers/#adam)). Default
  to 1e-4.
+ `-M`: considered research problem, either `feature_detection` (determining if
  some labelled objects are on an image) or `semantic_segmentation`
  (classifying each pixel of an image).
+ `-N`: neural network architecture, either `simple` (default value), or
  `vgg16` for the feature detection problem, `simple` is the only handled
  architecture for semantic segmentation.
+ `-n`: neural network name, used for checkpoint path naming. Default to `cnn`.
+ `-p`: path to datasets, on the file system. Default to `./data`.
+ `-s`: image size, in pixels (height = width). Default to 256.
