import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--train')
def train_model(name, url, out_dir):
    """Train a ML model using a dataset and save the model to disk.

    Parameters
    ----------
    train: str
        Path to train dataset in disk.

    Returns
    -------
    None
    """
    pass


if __name__ == '__main__':
    train_model()
