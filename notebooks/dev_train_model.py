
# %%
# import logging
from operator import length_hint
# from pathlib import Path
from distributed import Client
# import numpy as np
import dask.dataframe as dd
# import click
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
# from make_dataset.dataset import chunk, agg, finalize, _save_datasets
from dask_ml.preprocessing import Categorizer, StandardScaler, DummyEncoder
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
# from dask_ml.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import set_config


ddf = dd.read_parquet('/home/diego/Coding/code-challenge-2020/data_root/processed/train.parquet',
                      engine='pyarrow')


# %%
X, y = ddf.drop(['points'], axis=1), ddf[['points']]
X.head()

# # %%
# scaler = StandardScaler()
# enc = DummyEncoder()
# cat = Categorizer()
# df2 = ddf
# df2['price'] = scaler.fit_transform(df2[['price']]).price
# df2 = cat.fit_transform(df2)
# df2 = enc.fit_transform(df2)
# df2.head()
# df2_X, df2_y = df2.drop(['points'], axis=1), df2[['points']]
# model = LinearRegression()

# %%
# Prepare data
set_config(display='diagram')  # Allows us to visualize pipeline
num_proc = make_pipeline(StandardScaler())
cat_proc = make_pipeline(Categorizer(), DummyEncoder())
cat_cols = X.columns.to_list()
cat_cols.remove('price')
preprocessor = make_column_transformer((num_proc, ['price']),
                                       (cat_proc, cat_cols))
model = make_pipeline(preprocessor, LinearRegression())
model

# %%
model.fit(X, y)

# %%
cross_val_scores = cross_val_score(model, X.compute(), y.compute(), cv=5)
# cross_val_scores = cross_val_score(model, X, y, cv=5)  # Using dask dataframes

# sklearn cross_val_score doesn't work with dask dataframes. I could create a
# simple k-fold cross validation for loop but lets leave it unless I manage to
# make docker-compose work properly

# %%
model.score(X, y)
