#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""/**
 *   This script aims to test a neural network model in order to detect
 *   objects from street-scene (based on Mapillary Vistas dataset)

 *   Raphael Delhome - september 2017
 *
 *   This library is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU Library General Public
 *   License as published by the Free Software Foundation; either
 *   version 2 of the License, or (at your option) any later version.
 *   
 *   This library is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *   Library General Public License for more details.
 *   You should have received a copy of the GNU Library General Public
 *   License along with this library; if not, see <http://www.gnu.org/licenses/>
 */

"""

import argparse
import os
import pandas as pd
import sys

from dataset import Dataset, ShapeDataset
from model import ConvolutionalNeuralNetwork
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Convolutional Neural Netw"
                                                  "ork on street-scene images"))
    parser.add_argument('-a', '--aggregate-label', action='store_true',
                        help="Aggregate some labels")
    parser.add_argument('-b', '--batch-size', required=False, type=int,
                        nargs='?', default=20,
                        help=("Number of images that must be contained "
                              "into a single batch"))
    parser.add_argument('-d', '--dataset', required=True, nargs='?',
                        help="Dataset type (either mapillary or shape")
    parser.add_argument('-dp', '--datapath', required=False,
                        default="../data", nargs='?',
                        help="Relative path towards data directory")
    parser.add_argument('-i', '--nb-testing-image', type=int, default=5000,
                        help=("Number of testing images"))
    parser.add_argument('-l', '--label-list', required=False, nargs="+",
                        default=-1, type=int,
                        help=("List of label indices that "
                              "will be considered during testing process"))
    parser.add_argument('-M', '--model',
                        help=("Research problem that is addressed, "
                              "either 'feature_detection' or "
                              "'semantic_segmentation'"))
    parser.add_argument('-ls', '--log-step', nargs="?",
                        default=10, type=int,
                        help=("Log periodicity during testing process"))
    parser.add_argument('-n', '--name', default="cnnmapil", nargs='?',
                        help=("Model name that will be used for results, "
                              "checkout and graph storage on file system"
                              "expected format: "
                              "<instance-name>_<image-size>_<network-size>_"
                              "<batch-size>_<full/aggregated>_<dropout-rate>_"
                              "<learning-rate>"))
    args = parser.parse_args()

    # instance name decomposition (instance name = name + image size + network size)
    _, image_size, network_size, _, aggregate_value, _, _ = args.name.split('_')
    image_size = int(image_size)

    if image_size > 1024:
        utils.logger.error(("Unsupported image size. Please provide a "
                            "reasonable image size (less than 1024)"))
        sys.exit(1)

    if not network_size in ["small", "medium"]:
        utils.logger.error(("Unsupported network size. "
                            "Please choose 'small' or 'medium'."))
        sys.exit(1)

    if not args.model in ["feature_detection", "semantic_segmentation"]:
        utils.logger.error("Unsupported model. Please consider ")
        utils.logger.utils(("'feature_detection' or 'semantic_segmentation'"))
        sys.exit(1)

    # Data path and repository management
    folders = utils.prepare_folders(args.datapath, args.dataset, aggregate_value,
                                    image_size, args.model)

    # Dataset creation
    if args.dataset == "mapillary":
        config_name = "config.json" if aggregate_value == 'full' else "config_aggregate.json"
        config_path = os.path.join(folders["input"], config_name)
        testing_dataset = Dataset(image_size, config_path)
    elif args.dataset == "shapes":
        testing_dataset = ShapeDataset(image_size, 3)
    else:
        utils.logger.error("Unsupported dataset type. Please choose 'mapillary' or 'shape'")
        sys.exit(1)

    # Dataset populating/loading (depends on the existence of a specification file)
    if os.path.isfile(folders["testing_config"]):
        testing_dataset.load(folders["testing_config"], args.nb_testing_image)
    else:
        input_image_dir = os.path.join(folders["input"], "testing")
        testing_dataset.populate(folders["prepro_testing"], input_image_dir,
                                 nb_images=args.nb_testing_image, labelling=False)
        testing_dataset.save(folders["testing_config"])

    # Glossary management (are all the labels required?)
    if args.label_list == -1:
        label_list = testing_dataset.label_ids
    else:
        label_list = args.label_list
        if sum([l>=testing_dataset.get_nb_class() for l in args.label_list]) > 0:
            utils.logger.error(("Unsupported label list. Please enter a list of integers comprised"
                                "between 0 and {}".format(nb_labels)))
            sys.exit(1)

    # Convolutional Neural Network creation
    utils.logger.info(("{} classes in the dataset glossary, {} being focused "
                       "").format(testing_dataset.get_nb_class(), len(label_list)))
    cnn = ConvolutionalNeuralNetwork(network_name=args.name, image_size=image_size,
                                     nb_channels=3, nb_labels=len(label_list),
                                     netsize=network_size)
    cnn.test(testing_dataset, labels=label_list, batch_size=min(args.batch_size, args.nb_testing_image),
             log_step=args.log_step, backup_path=folders["output"])

    sys.exit(0)
