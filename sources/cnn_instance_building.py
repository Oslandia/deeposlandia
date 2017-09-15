import itertools
import json
import numpy as np
import pandas as pd

import utils

# Basic definitions
conv_layer_1_depth = [8, 16]
conv_layer_2_depth = [12, 24]
conv_layer_3_depth = [16, 32]
conv_layer_kernel = [4, 8]
pool_layer_strides = [[1, 2, 2, 1], [1, 4, 4, 1]]
fullconn_layer_1_depth = [512, 1024]
fullconn_layer_2_depth = [256, 512]

model_shapes = pd.DataFrame({"n_conv":[1, 2, 3, 2, 3],
                             "n_pool":[1, 2, 3, 2, 3],
                             "n_fullconn":[1, 1, 1, 2, 2]})

# Build the seminal experimental design
param_table = []
for it, model in model_shapes.iterrows():
    cur_param = []
    if model["n_conv"] == 1:
        cur_param = cur_param + [[0, 1], [None], [None]]
    elif model["n_conv"] == 2:
        cur_param = cur_param + [[0, 1], [0, 1], [None]]
    elif model["n_conv"] == 3:
        cur_param = cur_param + [[0, 1], [0, 1], [0, 1]]
    if model["n_pool"] == 1:
        cur_param = cur_param + [[0, 1], [None], [None]]
    elif model["n_pool"] == 2:
        cur_param = cur_param + [[0, 1], [0, 1], [None]]
    elif model["n_pool"] == 3:
        cur_param = cur_param + [[0, 1], [0, 1], [0, 1]]
    if model["n_fullconn"] == 1:
        cur_param = cur_param + [[0, 1], [None]]
    elif model["n_fullconn"] == 2:
        cur_param = cur_param + [[0, 1], [0, 1]]
    param_table.append(cur_param)
param_table = pd.DataFrame(param_table)

# Expand the experimental design to a understandable list of instances
experimental_design = []
for i, row in param_table.iterrows():
    conv1, conv2, conv3, pool1, pool2, pool3, fc1, fc2 = row
    g = itertools.product(conv1, conv2, conv3, pool1, pool2, pool3, fc1, fc2)
    experimental_design = experimental_design + list(g)

# Select only a limited set of instances (identic levels for parameter of the
# same layer type)
ed = experimental_design
conv_filter = [ed[index][:3].count(0) == 0 or ed[index][:3].count(1) == 0
               for index in range(len(ed))]
ed = list(itertools.compress(ed, conv_filter))
print(len(ed))
pool_filter = [ed[index][3:6].count(0) == 0 or ed[index][3:6].count(1) == 0
               for index in range(len(ed))]
ed = list(itertools.compress(ed, pool_filter))
print(len(ed))
fc_filter = [ed[index][6:].count(0) == 0 or ed[index][6:].count(1) == 0
             for index in range(len(ed))]
ed = list(itertools.compress(ed, fc_filter))
print(len(ed))

# Write a configuration file (json format) for each instance
utils.make_dir("../models")
for config in experimental_design:
    config_dict = dict()
    n_conv = n_pool = n_fullconn = 1
    # conv1
    config_dict["conv1"] = {"depth": conv_layer_1_depth[config[0]],
                            "kernel_size": conv_layer_kernel[config[0]],
                            "strides": [1, 1, 1, 1]}
    # conv2
    if not config[1] is None:
        n_conv = 2
        config_dict["conv2"] = {"depth": conv_layer_2_depth[config[1]],
                                "kernel_size": conv_layer_kernel[config[1]],
                                "strides": [1, 1, 1, 1]}
    # conv3
    if not config[2] is None:
        n_conv = 3
        config_dict["conv3"] = {"depth": conv_layer_3_depth[config[2]],
                                "kernel_size": conv_layer_kernel[config[2]],
                                "strides": [1, 1, 1, 1]}
    # pool1
    config_dict["pool1"] = {"kernel_size": pool_layer_strides[config[3]],
                            "strides": pool_layer_strides[config[3]]}
    # pool2
    if not config[4] is None:
        n_pool = 2
        config_dict["pool2"] = {"kernel_size": pool_layer_strides[config[4]],
                                "strides": pool_layer_strides[config[4]]}
    # pool3
    if not config[5] is None:
        n_pool = 3
        config_dict["pool3"] = {"kernel_size": pool_layer_strides[config[5]],
                                "strides": pool_layer_strides[config[5]]}
    # fullconn1
    config_dict["fullconn1"] = {"depth": fullconn_layer_1_depth[config[6]]}
    # fullconn2
    if not config[7] is None:
        n_fullconn = 2
        config_dict["fullconn2"] = {"depth": fullconn_layer_2_depth[config[7]]}

    instance_name = ('../models/cnn_mapil_' + str(n_conv) + '_' +
                     str(config[0]) + '_' + str(n_pool) + '_' +
                     str(config[3]) + '_' + str(n_fullconn) + '_' +
                     str(config[6]))
    with open(instance_name+'.json', 'w', encoding='utf-8') as output_file:
        json.dump(config_dict, output_file, indent=4)
