#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
/**
 *   Raphael Delhome - december 2017
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

import json
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import time

import dataset
from cnn_model import ConvolutionalNeuralNetwork
import utils

class SemanticSegmentationModel(ConvolutionalNeuralNetwork):

    def __init__(self, network_name="mapillary", image_size=512, nb_channels=3,
                 nb_labels=65, netsize="small", learning_rate=[1e-3],
                 monitoring_level=1):
        """
        """
        ConvolutionalNeuralNetwork.__init__(self, network_name, image_size, nb_channels, nb_labels, monitoring_level)
        self._Y = tf.placeholder(tf.int8, name='Y',
                                 shape=[None, self._image_size,
                                        self._image_size, self._nb_channels])
        self.add_layers()
        self.compute_loss()
        self.optimize(learning_rate)
        self._cm = self.compute_dashboard(self._Y, self._Y_pred)

    def add_layers(self):
        """
        """
        pass

    def compute_loss(self):
        """
        """
        pass

    def optimize(self):
        """
        """
        pass

    def define_batch(self):
        """
        """
        pass

    def train(self):
        """
        """
        pass

    def validate(self):
        """
        """
        pass

    def test(self):
        """
        """
        pass
