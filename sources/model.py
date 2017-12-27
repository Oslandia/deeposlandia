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


class ConvolutionalNeuralNetwork(object):

    def __init__(self, batch_size=128, num_labels=65, learning_rate=1e-3):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels

    def get_batch_size(self):
        return _batch_size
    
    def get_learning_rate(self):
        return _learning_rate
    
    def get_num_labels(self):
        return _num_labels
    
