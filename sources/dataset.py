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

class Dataset(object):

    def __init__(self, image_size, glossary_filename):
        """
        """
        self._image_size = image_size
        self.class_info = defaultdict()
        self.build_glossary(glossary_filename)
        self.image_info = defaultdict()

    def get_class(self, class_id):
        """
        """
        if not class_id in self.class_info.keys():
            print("Class {} not in the dataset glossary".format(class_id))
            return None
        return self.class_info[class_id]
    
    def get_image(self, image_id):
        """
        """
        if not image_id in self.image_info.keys():
            print("Image {} not in the dataset".format(image_id))
            return None
        return self.image_info[image_id]

    def get_nb_class(self):
        """
        """
        return len(self.class_info)

    def build_glossary(self, config_filename):
        """Read the Mapillary glossary stored as a json file at the data repository
        root

        """
        with open(config_filename) as config_file:
            glossary = json.load(config_file)
        if "labels" not in glossary.keys():
            print("There is no 'label' key in the provided glossary.")
            return None
        for lab_id, label in enumerate(glossary["labels"]):
            if label["evaluate"]:
                name_items = label["name"].split('--')
                category = '-'.join(name_items[:-1])
                self.add_class(lab_id, name_items[-1], label["color"], category)

    def add_class(self, class_id, class_name, color, category=None):
        """
        """
        if class_id in self.class_info.keys():
            print("Class {} already stored into the class set.".format(class_id))
            return None
        self.class_info[class_id] = {"name": class_name,
                                     "category": category,
                                     "color": color}
