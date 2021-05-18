# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Data exploration

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import dask.dataframe as dd
from dask.distributed import Client, progress
import dask as d

# %% [markdown]
# ## Create dask dataframe

# %%
in_csv = '/home/diego/Coding/code-challenge-2020/data_root/raw/wine_dataset.csv'

client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')


# %%

ddf = dd.read_csv(in_csv, blocksize=1e6)
ddf = ddf.set_index('Unnamed: 0')
ddf.head()
#ddf.dtypes

# %% [markdown]
# ## Preprocess dataset. Dealing with NaN values

# %%
# There is very few complete rows
print("Original size: {}".format(len(ddf), len(ddf.columns)))
print("Number of rows with 0 NaN values: {}".format(len(ddf.dropna())))


# %%
df2 = ddf.isna().sum().compute()
print("Ratio of NaNs in each column:\n\n{}".format(df2/len(ddf)))

g = sns.barplot(x=df2.index, y=df2)
plt.setp(g.get_xticklabels(), rotation=45, ha='right')
g.set(title="Number of NaN per categorie")

# %% [markdown]
# Region_2 column should be dropped directly. There is too little realible data.
# 

# %%
print("Number of different designations: {}".format(len(ddf.groupby(['designation']).size())))

# %% [markdown]
# Too many different designations and too many NaN values. Better to drop column.
# 
# Finally, the description column would need some NLP processing so we will drop it for now.
# %% [markdown]
# ### Dealing with NaN values in categorical columns
# %% [markdown]
# We can see that if the information about a country is missing, then the province and region values are missing as well. Thus, is not possible to fill country values. We will use an 'Unknown' type to fill in missing values.

# %%
# If country is missing, province, and regions are missing as well.
len(ddf.loc[(ddf['country'].isnull()) & (~ddf['province'].isnull())])
len(ddf.loc[(ddf['province'].isnull()) & (~ddf['country'].isnull())]) # Cant guess province based on country
ddf['country'] = ddf['country'].fillna('Unknown')
ddf['province'] = ddf['province'].fillna('Unknown')

# %% [markdown]
# We try to see if we can fill some NaN values of region_1 with information from region_2, since region_2 contains region_1. For example using most common value for a specific region_2 to fill region_1 missing value. However there is 100% overlap of missing values. Maybe in bigger dataset but for now is not an option.
# 
# However we can try to fill missing values using the 'province' column. Most common value for each province will be used.

# %%
# The custom aggregate mode function works, but since there is so many NaN values in region_1, the result is not great
source = pd.DataFrame({'Country' : ['USA', 'USA', 'Russia','USA'], 
                  'City' : ['New-York', 'Jacksonville', 'Sankt-Petersburg', 'New-York'],
                  'Short name' : ['NY','New','Spb','NY']})
source
source.groupby(['Country'])['City'].aggregate(pd.Series.mode)

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

mode = dd.Aggregation('mode', chunk, agg, finalize)
most_common_region=ddf.groupby(['province']).agg({'region_1': mode}).compute()


ddf['region_1']=ddf.apply(lambda x: most_common_region.loc[x.province, 'region_1'] if x.province in most_common_region['region_1'].index else 'Unknown', axis=1).where(ddf['region_1'].isna(), ddf['region_1'])

# %% [markdown]
# We should try to merge twitter handle and taster name since they are equivalent. However, there is no additional information in taster_twitter_handle column. It is possible that in the bigger dataset this merge can be done more effectively. But for now we will drop the 'taster_twitter_handle' column since it has more NaN values.

# %%
len(ddf[ddf['taster_name'].isnull()  & ~ddf['taster_twitter_handle'].isnull()].compute())
ddf['taster_name'] = ddf['taster_name'].fillna('Unknown')

# %% [markdown]
# ### Filling price data

# %%
ddf.loc[ddf['province'] == 'Aconcagua Valley'].compute()


# %%
mean_prices = ddf.groupby(['province'])['price'].mean().compute()
global_mean = ddf['price'].mean().compute()
mean_prices = mean_prices.fillna(global_mean)
mean_prices


# %%
ddf.loc[ddf['price'].isnull()].compute()
ddf.loc[ddf['province'] == 'Aconcagua Valley'].compute()

#ddf['price']=ddf.apply(lambda x: x['price']+100, axis=1)
ddf['price']=ddf.apply(lambda x: mean_prices[x['province']], axis=1, meta=('x', 'f8')).where(ddf['price'].isna(),ddf['price'])


ddf.loc[ddf['province'] == 'Aconcagua Valley'].compute()


# %%
len(ddf['title'].unique())


# %%
df2 = ddf.drop(['description', 'designation','region_2', 'taster_twitter_handle'], axis=1)
print("Ratio of NaNs in each column:\n\n{}".format(df2.isna().sum().compute()/len(ddf)))
df2.head()


