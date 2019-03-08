Trained models may be tested after the training process. Once a model is
trained, a checkpoint structure is recorded in
`<datapath>/<dataset>/output/<problem>/checkpoints/<instance-name>`. It is the
key point for inference.

The model testing is done as follows:

```
python deeposlandia/inference.py -D shapes -i ./data/shapes/preprocessed/64_full/testing/images/shape_00000.png
```

In this example, a label prediction will be done on a single image, for
`shapes` dataset in the feature detection case. The trained model will be
recovered by default in `<datapath>/<dataset>/output/<problem>/checkpoints/`,
by supposing that an optimized model (*e.g.* regarding hyperparameters) has
been produced. If the hyperparameters are specified (training batch size,
dropout rate, starting learning rate, learning rate decay, model architecture
and even model name), knowing that the image size is given by the first tested
image, the trained model is recovered in
`<datapath>/<dataset>/output/<problem>/checkpoints/<instance>/`, where
`<instance>` is defined as:

```
<model_name>-<image_size>-<network_architecture>-<batch_size>-<aggregation_mode>-<dropout>-<start_lr>-<lr_decay>
```

If no trained model can be found in the computed path, the label prediction is
done from scratch (and will be rather inaccurate...).

The list of handled parameters is as follows:
+ `-a`: aggregate labels. Used to point out the accurate configuration file, so
  as to get the number of labels in the dataset.
+ `-b`: training image batch size. Default to `None` (aims at identifying
  trained model).
+ `-D`: dataset (either `mapillary` or `shapes`)
+ `-d`: percentage of dropped out neurons during training process. Default to
  `None` (aims at identifying trained model).
+ `-i`: path to tested images, may handle regex for multi-image selection.
+ `L`: starting learning rate. Default to `None` (aims at identifying trained
  model).
+ `l`: learning rate decay (according to
  the [Adam optimizer definition](https://keras.io/optimizers/#adam)). Default
  to `None` (aims at identifying trained model).
+ `-M`: considered research problem, either `feature_detection` (determining if
  some labelled objects are on an image) or `semantic_segmentation`
  (classifying each pixel of an image).
+ `-N`: trained model neural network architecture. Default to `None` (aims at
  identifying trained model).
+ `-n`: neural network name. Default to `None` (aims at identifying trained model).
+ `-p`: path to datasets, on the file system. Default to `./data`.
