#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
/**
 *   This script aims to train a neural network model in order to read street
 *   scene images produced by Mapillary (https://www.mapillary.com/dataset/vistas)

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

from dataset import Dataset
from model import ConvolutionalNeuralNetwork
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Convolutional Neural Netw"
                                                  "ork on street-scene images"))
    parser.add_argument('-b', '--batch-size', required=False, type=int,
                        nargs='?', default=20,
                        help=("The number of images that must be contained "
                              "into a single batch"))
    parser.add_argument('-c', '--nbconv', required=False, type=int,
                        nargs='?', default=2,
                        help=("The number of convolutional layers "
                              "that must be inserted into the network"))
    parser.add_argument('-d', '--datapath', required=False,
                        default="../data", nargs='?',
                        help="""The relative path towards data directory""")
    parser.add_argument('-dn', '--dataset-name', required=False,
                        default="training_dataset", nargs='?',
                        help=("The json dataset filename, "
                              "without its extension"))
    parser.add_argument('-do', '--dropout', required=False,
                        default=2.0/3, nargs='?',
                        help=("The percentage of dropped out neurons "
                              "during training"))
    parser.add_argument('-e', '--nb-epochs', required=False, type=int,
                        default=5, nargs='?',
                        help=("The number of training epochs (one epoch means "
                              "scanning each training image once)"))
    parser.add_argument('-f', '--nbfullyconn', required=False, type=int,
                        nargs='?', default=1,
                        help=("The number of fully-connected layers "
                              "that must be inserted into the network"))
    parser.add_argument('-g', '--glossary-printing', action="store_true",
                        help=("True if the program must only "
                              "print the glossary, false otherwise)"))
    parser.add_argument('-l', '--label-list', required=False, nargs="+",
                        default=-1, type=int,
                        help=("The list of label indices that "
                              "will be considered during training process"))
    parser.add_argument('-ls', '--log-step', nargs="?",
                        default=10, type=int,
                        help=("The log periodicity during training process"))
    parser.add_argument('-m', '--mode', required=False, default="train",
                        nargs='?', help=("The network running mode"
                                         "('train', 'test', 'both')"))
    parser.add_argument('-n', '--name', default="cnn_mapil", nargs='?',
                        help=("The model name that will be used for results, "
                              "checkout and graph storage on file system"))
    parser.add_argument('-p', '--prepare-data', action="store_true",
                        help=("True if the data must be prepared, "
                              "false otherwise"))
    parser.add_argument('-r', '--learning-rate', required=False, nargs="+",
                        default=[0.01, 1000, 0.95], type=float,
                        help=("List of learning rate components (starting LR, "
                              "decay steps and decay rate)"))
    parser.add_argument('-s', '--image-size', nargs="?",
                        default=512, type=int,
                        help=("The desired size of images (width = height)"))
    parser.add_argument('-ss', '--save-step', nargs="?",
                        default=100, type=int,
                        help=("The save periodicity during training process"))
    parser.add_argument('-t', '--training-limit', default=None, type=int,
                        help=("Number of training iteration, "
                              "if not specified the model run during "
                              "nb-epochs * nb-batchs iterations"))
    parser.add_argument('-w', '--weights', default=["base"], nargs='+',
                        help=("The weight policy to apply on label "
                              "contributions to loss: either 'base' "
                              "(default case), 'global', 'batch', "
                              "'centeredglobal', 'centeredbatch'"))
    args = parser.parse_args()

    if args.image_size < 256 or args.image_size > 2048:
        utils.logger.error(("Unsupported image size. Please provide a "
                            "reasonable image size (between 256 and 2048"))
        sys.exit(1)

    if args.mode not in ["train", "test", "both"]:
        utils.logger.error(("Unsupported running mode. "
                            "Please choose amongst 'train', 'test' or 'both'."))
        sys.exit(1)

    weights = ["base", "global", "batch", "centeredbatch", "centeredglobal"] 
    if sum([w in weights for w in args.weights]) != len(args.weights):
        utils.logger.error(("Unsupported weighting policy. Please choose "
                            "amongst 'base', 'global', 'batch', "
                            "'centeredglobal' or 'centeredbatch'."""))
        utils.logger.info("'base': Regular weighting scheme...")
        utils.logger.info(("'global': Label contributions to loss are "
                           "weighted with respect to label popularity "
                           "within the dataset (decreasing weights)..."))
        utils.logger.info(("'batch': Label contributions to loss are weighted "
                           "with respect to label popularity within the "
                           "dataset (convex weights with min at 50%)..."))
        utils.logger.info(("'centeredbatch': Label contributions to loss are "
                           "weighted with respect to label popularity within "
                           "each batch (decreasing weights)..."))
        utils.logger.info(("'centeredglobal': Label contributions to loss are "
                           "weighted with respect to label popularity within "
                           "each batch (convex weights with min at 50%)..."))
        sys.exit(1)

    if len(args.learning_rate) != 1 and len(args.learning_rate) != 3:
        utils.logger.error(("There must be 1 or 3 learning rate component(s) "
                            "(start, decay steps and decay rate"
                            "; actually, there is/are {}"
                            "").format(len(args.learning_rate)))
        sys.exit(1)

    dataset_filename = os.path.join(args.datapath, args.dataset_name+'.json')
    d = Dataset(args.image_size, os.path.join(args.datapath, "config.json"))
    if os.path.isfile(dataset_filename):
        d.load(dataset_filename)
    else:
        d.populate(os.path.join(args.datapath, "training"))
        d.save(dataset_filename)

    if args.label_list == -1:
        label_list = list(d.class_info.keys())
    else:
        label_list = args.label_list
        if sum([l>=d.get_nb_class() for l in args.label_list]) > 0:
            utils.logger.error(("Unsupported label list. "
                                "Please enter a list of integers comprised"
                                "between 0 and {}".format(nb_labels)))
            sys.exit(1)

    if args.glossary_printing:
        glossary = pd.DataFrame(d.class_info).T
        glossary["popularity"] = d.get_class_popularity()
        utils.logger.info("Data glossary:\n{}".format(d.class_info))
        sys.exit(0)

    utils.logger.info(("{} classes in the dataset glossary, {} being focused "
                       "").format(d.get_nb_class(), len(label_list)))
    utils.logger.info(("{} images in the training"
                       "set").format(d.get_nb_images()))

    cnn = ConvolutionalNeuralNetwork(network_name=args.name,
                                     image_size=args.image_size,
                                     nb_channels=3,
                                     batch_size=args.batch_size,
                                     nb_labels=len(label_list),
                                     learning_rate=args.learning_rate)

    cnn.train(d, label_list,
              nb_epochs=args.nb_epochs, nb_iter=args.training_limit,
              log_step=args.log_step, save_step=args.save_step)
    
    sys.exit(0)
