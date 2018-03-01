"""Image resizing and aggregation labels

Can pass this script with the GNU command 'parallel' in order to process the
images in parallel. Let image you have a text file `image.list`:

cat image.list | parallel "python aggregate_label.py -s=256 -k={}"

Or alternatively, without this text file:

ls <image_repository> | sed -e 's/\.jpg$//' | parallel "python aggregate_label.py -s=256 -k={}"
"""

import os
import json

import numpy as np
import pandas as pd

from PIL import Image

import utils


DATASET = 'mapillary'


def _split_object_label(df):
    """Split 'object' label into three sub-labels

    - object
    - vehicle
    - traffic

    XXX : split vehicle into 2 or 4 sub-labels (motorcycle, bike, car, bus, etc.)?
      - bus, car, caravan, truck
      - motorcycle, bicycle
      - car
      - others

    :param df: pandas dataframe - Mapillary labels
    :return: pandas dataframe - modified Mapillary labels
    """
    df = df.copy()
    mask_vehicle = df['name'].str.split('--').apply(lambda x: 'vehicle' in x)
    df.loc[mask_vehicle, 'group_label'] = 'vehicle'
    mask_traffic = df['readable'].str.lower().apply(lambda x: 'traffic' in x)
    df.loc[mask_traffic, 'group_label'] = 'traffic'
    return df


def _split_construction_label(df):
    """Split 'construction' label into three sub-labels

    - construction
    - flat
    - barrier

    :param df: pandas dataframe - Mapillary labels
    :return: pandas dataframe - modified Mapillary labels
    """
    df = df.copy()
    mask_barrier = df['name'].str.split('--').apply(lambda x: 'barrier' in x)
    df.loc[mask_barrier, 'group_label'] = 'barrier'
    mask_flat = df['name'].str.split('--').apply(lambda x: 'flat' in x)
    df.loc[mask_flat, 'group_label'] = 'flat'
    return df


def read_config(datadir):
    """Read the mapillary configuration JSON file

    :param datadir: string - path to data repository
    :return: dict - Mapillary glossary
    """
    with open(os.path.join(datadir, DATASET, 'config.json'), 'r') as fobj:
        return json.load(fobj)


def config_as_dataframe(config):
    """JSON labels data into a DataFrame.

    Add some metadata. Group some labels (in order to have less classes)

    :param config: dict - Mapillary glossary
    :return: pandas dataframe - Mapillary labels
    """
    df = pd.DataFrame(config['labels'])
    df['id'] = range(df.shape[0])
    df['family'] = df['name'].str.split('--').apply(lambda x: x[0])
    df['group_label'] = df['family']
    df = _split_object_label(df)
    df = _split_construction_label(df)
    return df


def resize(key, size, datadir):
    """
    :param key: string - image name, without its extension
    :param size: integer - image size, in pixels
    :param datadir: string - path to data repository
    Returns
    -------
    Two resized images (train and labels)
    """
    train_fname = os.path.join(datadir, DATASET, 'training', 'images', key + '.jpg')
    label_fname = os.path.join(datadir, DATASET, 'training', 'labels', key + '.png')
    # open original images
    train_img = Image.open(train_fname)
    label_img = Image.open(label_fname)

    # resize images (size*larger_size or larger_size*size)
    img_in = utils.resize_image(train_img, size)
    img_out = utils.resize_image(label_img, size)

    # crop images to get size * size dimensions
    crop_pix = np.random.randint(0, 1 + max(img_in.size) - size)
    final_img_in = utils.mono_crop_image(img_in, crop_pix)
    final_img_out = utils.mono_crop_image(img_out, crop_pix)
    return final_img_in, final_img_out


def group_image_label(image, df):
    """Group the labels

    :param image: PIL.Image
    :param df: DataFrame - grouped labels

    Returns
    -------
    PIL.Image
    """
    # turn all label ids into the lowest digits/label id according to its "group"
    # (manually built)
    a = np.array(image)
    for label, grp in df.groupby("group_label")["id"]:
        keep_label_id = min(grp)
        for label_id in grp[1:]:
            mask = a == label_id
            a[mask] = keep_label_id
    return Image.fromarray(a, mode=image.mode)


def save_images(train_img, label_img, key, datadir, size):
    """
    :param train_img: PIL.Image - training image
    :param label_img: PIL.Image - labelled training image
    :param key: string - image name, without its extension
    :param size: integer - image size, in pixels
    :param datadir: string - path to data repository
    """
    train_dir = os.path.join(
        datadir, DATASET, 'training_label_aggregate', 'images_' + str(size))
    label_dir = os.path.join(
        datadir, DATASET, 'training_label_aggregate', 'labels_' + str(size))
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    train_img.save(os.path.join(train_dir, key + '.jpg'))
    label_img.save(os.path.join(label_dir, key + '.png'))


def main(key, size, datadir):
    """
    :param key: string - image name, without its extension
    :param size: integer - image size, in pixels
    :param datadir: string - path to data repository
    """
    config = read_config(datadir)
    df = config_as_dataframe(config)
    df.to_csv(os.path.join(datadir, DATASET, 'config_agg_' + str(size) + '.csv'), index=False)
    train_img, label_img = resize(key, size, datadir)
    modified_label_img = group_image_label(label_img, df)
    save_images(train_img, modified_label_img, key, datadir, size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=("Aggregate labels on a training image"))
    parser.add_argument('-dp', '--datapath', required=False,
                        default="../data", nargs='?',
                        help="Relative path towards data directory")
    parser.add_argument('-s', '--size', type=int, required=True,
                        help="Desired size of images")
    parser.add_argument('-k', '--key', required=True, help='The key/name of the image')
    args = parser.parse_args()
    main(args.key, args.size, args.datapath)
