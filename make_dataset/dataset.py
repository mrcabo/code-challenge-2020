import click
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ddf = pd.read_csv(in_csv)

    # we set the index so we can properly execute loc below
    ddf = ddf.set_index('Unnamed: 0')

    # Fill NaN values with new 'Unknown' category
    ddf['country'] = ddf['country'].fillna('Unknown')
    ddf['province'] = ddf['province'].fillna('Unknown')
    ddf['taster_name'] = ddf['taster_name'].fillna('Unknown')

    # Fill region_1 missing values using the 'province' column.
    # Most common value for each province will be used. Rest are labeled Unknown
    most_common_region = (ddf.groupby('province')['region_1']
                          .apply(lambda x: x.mode()))

    ddf['region_1'] = ddf.apply(lambda x: most_common_region.loc[x.province][0]
                                if x.province in most_common_region
                                else 'Unknown', axis=1).where(ddf['region_1'].isna(), ddf['region_1'])

    # We fill price values with the province's average price. If that is
    # not available, we use the global average price
    mean_prices = ddf.groupby(['province'])['price'].mean()
    global_mean = ddf['price'].mean()
    mean_prices = mean_prices.fillna(global_mean)
    ddf['price'] = ddf.apply(lambda x: mean_prices[x['province']], axis=1).where(
        ddf['price'].isna(), ddf['price'])

    # Drop this columns as explained in notebook
    ddf = ddf.drop(['description', 'designation', 'region_2',
                    'taster_twitter_handle', 'title'], axis=1)

    train, test = train_test_split(ddf, test_size=0.1,
                                   random_state=0, shuffle=True)

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
