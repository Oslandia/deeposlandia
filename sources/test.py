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
    parser.add_argument('-b', '--batch-size', required=False, type=int,
                        nargs='?', default=20,
                        help=("Number of images that must be contained "
                              "into a single batch"))
    parser.add_argument('-d', '--dataset', required=True, nargs='?',
                        help="Dataset type (either mapillary or shape")
    parser.add_argument('-dp', '--datapath', required=False,
                        default="../data", nargs='?',
                        help="Relative path towards data directory")
    parser.add_argument('-i', '--nb-testing-image', type=int,
                        help=("Number of testing images"))
    parser.add_argument('-l', '--label-list', required=False, nargs="+",
                        default=-1, type=int,
                        help=("List of label indices that "
                              "will be considered during testing process"))
    parser.add_argument('-ls', '--log-step', nargs="?",
                        default=10, type=int,
                        help=("Log periodicity during testing process"))
    parser.add_argument('-n', '--name', default="cnnmapil", nargs='?',
                        help=("Model name that will be used for results, "
                              "checkout and graph storage on file system"))
    parser.add_argument('-ns', '--network-size', default="small",
                        help=("Neural network size, either 'small' or 'medium'"
                              "('small' refers to 3 conv/pool blocks and 1 "
                              "fully-connected layer, and 'medium' refers to 6"
                              "conv/pool blocks and 2 fully-connected layers)"))
    parser.add_argument('-s', '--image-size', nargs="?",
                        default=512, type=int,
                        help=("Desired size of images (width = height)"))
    args = parser.parse_args()

    if args.image_size > 1024:
        utils.logger.error(("Unsupported image size. Please provide a "
                            "reasonable image size (less than 1024)"))
        sys.exit(1)

    # Data path and repository management
    dataset_repo = os.path.join(args.datapath, args.dataset)
    testing_name = "testing_" + str(args.image_size)
    utils.make_dir(dataset_repo)
    utils.make_dir(os.path.join(dataset_repo, testing_name))
    testing_filename = os.path.join(dataset_repo, testing_name + '.json')

    # Dataset creation
    if args.dataset == "mapillary":
        testing_dataset = Dataset(args.image_size, os.path.join(args.datapath, args.dataset, "config.json"))
    elif args.dataset == "shapes":
        testing_dataset = ShapeDataset(args.image_size, 3)
    else:
        utils.logger.error("Unsupported dataset type. Please choose 'mapillary' or 'shape'")
        sys.exit(1)

    # Dataset populating/loading (depends on the existence of a specification file)
    if os.path.isfile(testing_filename):
        testing_dataset.load(testing_filename)
    else:
        testing_dataset.populate(os.path.join(args.datapath, args.dataset, testing_name),
                                 nb_images=args.nb_testing_image)
        testing_dataset.save(testing_filename)

    # Glossary management (are all the labels required?)
    if args.label_list == -1:
        label_list = list(testing_dataset.class_info.keys())
    else:
        label_list = args.label_list
        if sum([l>=testing_dataset.get_nb_class() for l in args.label_list]) > 0:
            utils.logger.error(("Unsupported label list. Please enter a list of integers comprised"
                                "between 0 and {}".format(nb_labels)))
            sys.exit(1)

    # Convolutional Neural Network creation
    utils.logger.info(("{} classes in the dataset glossary, {} being focused "
                       "").format(testing_dataset.get_nb_class(), len(label_list)))
    cnn = ConvolutionalNeuralNetwork(network_name=args.name, image_size=args.image_size,
                                     nb_channels=3, nb_labels=len(label_list),
                                     netsize=args.network_size)

    cnn.test(testing_dataset, labels=label_list, batch_size=args.batch_size,
             log_step=args.log_step, backup_path=dataset_repo)

    sys.exit(0)
