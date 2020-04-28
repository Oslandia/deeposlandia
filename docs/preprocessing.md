# Data preprocessing

Preprocessed versions of raw datasets has to be generated before any neural
network training:

```
deepo datagen -D mapillary -s 224 -P ./any-data-path -t 18000 -v 2000 -T 5000
```

This command will generates a set of 224 * 224 images based on Mapillary
dataset. The raw dataset must be in `./any-data-path/input`, and the
preprocessed dataset will be stored in `./any-data-path/preprocessed/224`.

The `-t`, `-v` and `-T` arguments refer respectively to training, validation and testing
image quantities **in the raw input folder**. The previous example correspond to the raw
Mapillary dataset size. In the Mapillary case, these amounts correspond to the preprocessed datasets as well: each image is modified in order to fit the required size.

For `aerial` and `tanzania` datasets, the amount of raw images are far smaller. However
one can design as many tiles as desired by exploiting the high resolution of these
images: one original image may represent dozens of billions of pixels, hence the
preprocessing step consists in building smaller tiles as cropped versions of the original
images. By default, one generates 1000 tiles per image, this behavior may be modified
with the `--nb-tiles-per-image` parameter. Two use cases are distinguished:
- for the training/validation cases, one picks random (x, y)-coordinates to generate the
  tiles;
- for the testing case, one cuts out the raw images following a regular grid, with an
  implicit objective: rebuilt a predicted version of the whole raw image (see
  [`postprocess.md`](./docs/postprocess.md) for more details).

In the shape datase case, this preprocessing step generates a bunch of images
from scratch.

As an easter-egg feature, label popularity is also printed by this command
(proportion of images where each label appears in the preprocessed dataset).

See more details with `deepo datagen -h`.
