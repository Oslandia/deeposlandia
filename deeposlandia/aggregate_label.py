"""Script to aggregate a few Mapillary labels
"""

import os
import json

import pandas as pd
import seaborn as sns

from deeposlandia import utils


def set_label_color(nb_colors):
    """Set a color for each aggregated label with seaborn palettes

    Parameters
    ----------
    nb_colors : int
        Number of label to display
    """
    palette = sns.hls_palette(nb_colors, 0.01, 0.6, 0.75)
    return ([int(255 * item) for item in color] for color in palette)


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
    df = pd.DataFrame(config["labels"])
    df["id"] = range(df.shape[0])
    df["family"] = df["name"].str.split("--").apply(lambda x: x[0])
    df["label"] = df["name"].str.split("--").apply(lambda x: x[-1])
    df["new_label"] = df["name"].str.split("--").apply(lambda x: x[-2])
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
    assert len(config["labels"]) == df.shape[0], "should have the same size"
    result = {"folder_structure": config["folder_structure"]}
    result["labels"] = []
    nb_labels = df.new_label.nunique()
    intgen = iter(range(nb_labels))
    palette = set_label_color(nb_labels)
    for key, group in df.groupby("new_label"):
        d = {
            "id": next(intgen),
            "family": group.family.iloc[0],
            "name": group.new_label.iloc[0],
            "contains": group.label.str.cat(sep="..."),
            "contains_id": list(group.id),
            "color": next(palette),
            "evaluate": any(group.evaluate),
        }
        result["labels"].append(d)
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
    config = utils.read_config(os.path.join(datadir, "config.json"))
    df = config_as_dataframe(config)
    agg_config = aggregate_config(config, df)
    return agg_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=("Aggregate labels on a training image")
    )
    parser.add_argument(
        "-dp",
        "--datapath",
        required=False,
        default="data",
        nargs="?",
        help="Relative path towards data directory",
    )
    parser.add_argument(
        "-s",
        "--save",
        required=False,
        default="config_aggregate.json",
        help="Name of the output JSON file.",
    )
    args = parser.parse_args()
    label_aggregated = main(args.datapath)
    with open(os.path.join(args.datapath, args.save), "w") as fobj:
        print(
            "write the file '{}'".format(
                os.path.join(args.datapath, args.save)
            )
        )
        json.dump(label_aggregated, fobj)
