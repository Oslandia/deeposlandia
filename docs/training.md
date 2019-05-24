# Model training

The model training may be undertaken as follows:

```
deepo train -M featdet -D mapillary -s 512 -e 5 -t 1000
```

In this example, 512 * 512 Mapillary images will be exploited for training a
feature detection model, using 1000 images of the training set. Here the
training will take place for five epoches.

Don't hesitate to `deepo train -h` for more details on available parameters.

# Hyperparameter optimization

A more complete hyperparameter analysis can be undertaken with the same
command, by passing lists to some of the parameters:
+ network architecture
+ batch size
+ dropout rate
+ learning rate
+ learning rate decay

Considering several options makes the program iterate over all possibilities,
and launch as many training processes as the number of parameter combinations. As an example:

```
deepo train -M featdet -D mapillary -s 512 -e 5 -t 1000 -L 0.01 0.001
```

will run two training processes, with learning rates respectively equal to 0.01
and 0.001 (all unspecified parameters takes their default value).
