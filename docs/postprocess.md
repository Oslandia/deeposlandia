# Postprocess geographic dataset predictions

Producing neural network prediction for geographic dataset is interesting however the
analysis can go further, by converting the predicted raster as vectorized data. For
example one may run the following command :

```
deepo postprocess -P data -D tanzania -s 384 -b 2 -i grid_034 -g
```

It will predict semantic segmentation labels on a file named `grid_034.tif`
(georeferenced image) located at `./data/tanzania/input/testing/images/`, by splitting it
in 384x384 pixelled tiles, and making predictions in 2-image batchs. The `-g` argument,
if specified, adds a postprocessing grid that materializes the tile within the original
image.

Model checkpoints, as well as predicted rasters and vectors are stored at
`./data/tanzania/output/semseg/`.

More details on the CLI is provided by running `deepo postprocess -h`.
