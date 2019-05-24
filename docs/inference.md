# Model inference

Trained models may be tested after the training process. Once a model is
trained, a checkpoint structure is recorded in
`<datapath>/<dataset>/output/<problem>/checkpoints/` folder.

The model testing is done as follows:

```
deepo infer -D shapes -M featdet -i ./data/shapes/preprocessed/64/testing/images/shape_00000.png
```

In this example, a label prediction will be done on a single image, for
`shapes` dataset in the feature detection case. The trained model will be
recovered in `<datapath>/<dataset>/output/<problem>/checkpoints/`,
by supposing that an optimized model (*e.g.* regarding hyperparameters) has
been produced. If the hyperparameters are specified (training batch size,
dropout rate, starting learning rate, learning rate decay, model architecture
and even model name), knowing that the image size is given by the first tested
image, the trained model is recovered in
`<datapath>/<dataset>/output/<problem>/checkpoints/<instance>/`, where
`<instance>` is defined as:

```
<model_name>-<image_size>-<network_architecture>-<batch_size>-<dropout>-<start_lr>-<lr_decay>
```

If no trained model can be found in the computed path, the label prediction is
done from scratch (and will be rather inaccurate...).

See more details on this command by running `deepo infer -h`.
