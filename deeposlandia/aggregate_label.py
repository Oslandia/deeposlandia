"""Script to aggregate a few Mapillary labels
"""

import os
import json

import pandas as pd


DATASET = 'mapillary'
# column name which contains the name of the aggregated label family
KEY_AGG = 'agg_label'


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

    Parameters
    ----------
    df : pandas dataframe
        Mapillary labels

    Returns
    -------
    pandas dataframe
        Modified Mapillary labels
    """
    df = df.copy()
    mask_vehicle = df['name'].str.split('--').apply(lambda x: 'vehicle' in x)
    df.loc[mask_vehicle, KEY_AGG] = 'vehicle'
    mask_traffic = df['readable'].str.lower().apply(lambda x: 'traffic' in x)
    df.loc[mask_traffic, KEY_AGG] = 'traffic'
    return df


def _split_construction_label(df):
    """Split 'construction' label into three sub-labels

    - construction
    - flat
    - barrier

    Parameters
    ----------
    df : pandas dataframe
        Mapillary labels

    Returns
    -------
    pandas dataframe
        Modified Mapillary labels
    """
    df = df.copy()
    mask_barrier = df['name'].str.split('--').apply(lambda x: 'barrier' in x)
    df.loc[mask_barrier, KEY_AGG] = 'barrier'
    mask_flat = df['name'].str.split('--').apply(lambda x: 'flat' in x)
    df.loc[mask_flat, KEY_AGG] = 'flat'
    return df


def read_config(datadir):
    """Read the mapillary configuration JSON file

    Parameters
    ----------
    datadir : string
        Path to data repository

    Returns
    -------
    dict
        Mapillary glossary
    """
    with open(os.path.join(datadir, DATASET, 'config.json'), 'r') as fobj:
        return json.load(fobj)


def config_as_dataframe(config):
    """JSON labels data into a DataFrame.

    Add some metadata. Group some labels (in order to have less classes)

    Parameters
    ----------
    config : dict
        Mapillary glossary

    Returns
    -------
    pandas dataframe
        Mapillary labels
    """
    df = pd.DataFrame(config['labels'])
    df['id'] = range(df.shape[0])
    df['family'] = df['name'].str.split('--').apply(lambda x: x[0])
    df[KEY_AGG] = df['family']
    df = _split_object_label(df)
    df = _split_construction_label(df)
    return df


def aggregate_config(config, df):
    """Aggregate the labels from the original configuration

    Parameters
    ----------
    config : dict
    df : DataFrame

    Returns
    -------
    dict to serialize
    """
    assert len(config['labels']) == df.shape[0], "should have the same size"
    result = {'folder_structure': config['folder_structure']}
    result['labels'] = []
    for key, group in df.groupby(KEY_AGG):
        label_id = int(group['id'].min())  # int conversion from int64 for JSON serialization
        for _, label in group.iterrows():
            if label['id'] == label_id:
                d = config['labels'][label_id]
                d['id'] = label_id
                d['group_name'] = label[KEY_AGG]
                d['aggregate'] = []
            else:
                d['aggregate'].append((label['id'], label['name']))
        result['labels'].append(d)
    return result


def main(datadir):
    """Generate a new config.json file with aggregated labels.

    Parameters
    ----------
    datadir : str

    Returns
    -------
    dict
    """
    config = read_config(datadir)
    df = config_as_dataframe(config)
    agg_config = aggregate_config(config, df)
    return agg_config


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=("Aggregate labels on a training image"))
    parser.add_argument('-dp', '--datapath', required=False,
                        default="data", nargs='?',
                        help="Relative path towards data directory")
    parser.add_argument('-s', '--save', required=False,
                        default="config_aggregate.json",
                        help="Name of the output JSON file.")
    args = parser.parse_args()
    label_aggregated = main(args.datapath)
    with open(os.path.join(args.datapath, 'mapillary', args.save), 'w') as fobj:
        print("write the file '{}'".format(os.path.join(args.datapath, 'mapillary', args.save)))
        json.dump(label_aggregated, fobj)
