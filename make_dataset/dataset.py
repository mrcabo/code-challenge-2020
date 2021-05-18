import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def chunk(s):
    return s.value_counts()


def agg(s):
    return s.apply(lambda s: s.groupby(level=-1).sum())


def finalize(s):
    level = list(range(s.index.nlevels - 1))
    return (
        s.groupby(level=level)
        .apply(lambda s: s.reset_index(level=level, drop=True).idxmax())
    )


def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train.parquet/'
    out_test = outdir / 'test.parquet/'
    flag = outdir / '.SUCCESS'

    train.to_parquet(str(out_train))
    test.to_parquet(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):
    """Processes csv file and saves a curated dataset to disk.

    Parameters
    ----------
    in-csv: str
        path to csv file in local disk
    out_dir:
        directory where files should be saved to.

    Returns
    -------
    None
    """
    log = logging.getLogger('make-dataset')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Connect to the dask cluster
    log.info(f'Starting make_datasets with in_csv: {in_csv} and out_dir: {out_dir}')
    log.info('Connecting to cluster')
    c = Client('dask-scheduler:8786')

    # load data as a dask Dataframe if you have trouble with dask
    # please fall back to pandas or numpy
    log.info('Reading csv file')
    ddf = dd.read_csv(in_csv, blocksize=1e6)
   
    log.info('ouput dataframe head')
    log.info(ddf.head())
    log.info('Trace 1')
    # we set the index so we can properly execute loc below
    ddf = ddf.set_index('Unnamed: 0')

    # trigger computation
    n_samples = len(ddf)

    # Fill NaN values with new 'Unknown' category
    ddf['country'] = ddf['country'].fillna('Unknown')
    ddf['province'] = ddf['province'].fillna('Unknown')
    ddf['taster_name'] = ddf['taster_name'].fillna('Unknown')
    log.info('Trace 2')
    # Fill region_1 missing values using the 'province' column.
    # Most common value for each province will be used. Rest are labeled Unknown
    mode = dd.Aggregation('mode', chunk, agg, finalize)
    most_common_region = ddf.groupby(
        ['province']).agg({'region_1': mode}).compute()
    ddf['region_1'] = ddf.apply(lambda x: most_common_region.loc[x.province, 'region_1']
                                if x.province in most_common_region['region_1'].index
                                else 'Unknown', axis=1).where(ddf['region_1'].isna(), ddf['region_1'])
    log.info('Trace 3')
    # We fill price values with the province's average price. If that is
    # not available, we use the global average price
    mean_prices = ddf.groupby(['province'])['price'].mean().compute()
    global_mean = ddf['price'].mean().compute()
    mean_prices = mean_prices.fillna(global_mean)
    ddf['price'] = ddf.apply(lambda x: mean_prices[x['province']], axis=1, meta=(
        'x', 'f8')).where(ddf['price'].isna(), ddf['price'])
    # Drop this columns as explained in notebook
    ddf = ddf.drop(['description', 'designation', 'region_2',
                    'taster_twitter_handle', 'title'], axis=1)

    # Encode categorical values using one-hot encoding.
    # This results in >6k columns. Maybe we'll need to change the encoding type
    # for some features such as 'winery' with so many unique values.
    # Also, I think this should be done in the model task.
    ddf = ddf.categorize()
    # encoder = DummyEncoder()
    # ddf = encoder.fit_transform(ddf)
    
    # # Normalize price values
    # scaler = StandardScaler()
    # ddf['price'] = scaler.fit_transform(ddf[['price']]).price
    log.info('dataset processed')

    # split dataset into train test feel free to adjust test percentage
    idx = np.arange(n_samples)
    test_idx = idx[:n_samples // 10]
    test = ddf.loc[test_idx]

    train_idx = idx[n_samples // 10:]
    train = ddf.loc[train_idx]

    # This also shuffles the data. Not sure if csv was shuffled before..
    # train, test = ddf.random_split([0.9, 0.1], shuffle=True)

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
