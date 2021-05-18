# %%
from dask_ml.datasets import make_regression
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import Categorizer, DummyEncoder, StandardScaler
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, progress
import dask as d
import pandas as pd


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


#### Remove NaNs ####
# %%
in_csv = '/home/diego/Coding/code-challenge-2020/data_root/raw/wine_dataset.csv'
# client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
ddf = dd.read_csv(in_csv, blocksize=1e6)
ddf = ddf.set_index('Unnamed: 0')

ddf['country'] = ddf['country'].fillna('Unknown')
ddf['province'] = ddf['province'].fillna('Unknown')
ddf['taster_name'] = ddf['taster_name'].fillna('Unknown')

# fill missing values using the 'province' column. Most common value for each province will be used.rest are labeled as Unknown.
mode = dd.Aggregation('mode', chunk, agg, finalize)
most_common_region = ddf.groupby(
    ['province']).agg({'region_1': mode}).compute()
ddf['region_1'] = ddf.apply(lambda x: most_common_region.loc[x.province, 'region_1']
                            if x.province in most_common_region['region_1'].index
                            else 'Unknown', axis=1).where(ddf['region_1'].isna(), ddf['region_1'])

mean_prices = ddf.groupby(['province'])['price'].mean().compute()
global_mean = ddf['price'].mean().compute()
mean_prices = mean_prices.fillna(global_mean)
ddf['price'] = ddf.apply(lambda x: mean_prices[x['province']], axis=1, meta=(
    'x', 'f8')).where(ddf['price'].isna(), ddf['price'])
ddf = ddf.drop(['description', 'designation', 'region_2',
                'taster_twitter_handle', 'title'], axis=1)
ddf.head()


# %%
# Useful if we use sklearn pipelines
# ddf.dtypes
# ce = Categorizer()
# ce.fit_transform(ddf).dtypes

# ddf.categorize(columns=['country', 'province', 'region_1'])
ddf = ddf.categorize()

# # %%
# scaler = StandardScaler()
# ddf['price'] = scaler.fit_transform(ddf[['price']]).price

# # %%
# encoder = DummyEncoder()
# ddf = encoder.fit_transform(ddf)

# %%
train, test = ddf.random_split([0.9, 0.1], shuffle=True)

# y = ddf[['points']]
# X = ddf.drop('points', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
# %%
train.head()
# %%
