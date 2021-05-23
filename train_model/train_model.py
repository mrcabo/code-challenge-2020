import click
import numpy as np
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from joblib import dump

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--in-data')
@click.option('--out-dir')
def train_model(in_data, out_dir):
    """Train a ML model using a dataset and save the model to disk.

    Parameters
    ----------
    in-data: str
        Path to train dataset in disk.
    out-dir: str
        Path where model will be saved.

    Returns
    -------
    None
    """
    log = logging.getLogger('train-model')
    log.info("Reading data")

    df = pd.read_parquet(in_data, engine='pyarrow')
    X, y = df.drop(['points'], axis=1), df[['points']]

    # Create preprocessor pipeline (scales numerical features and encodes categorical features)
    num_proc = make_pipeline(StandardScaler())
    cat_proc = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
    numerical_columns = ['price']
    categorical_columns = X.columns.drop('price').to_list()
    preprocessor = make_column_transformer((num_proc, numerical_columns),
                                           (cat_proc, categorical_columns))

    # Create regressor. This can be done without gridsearch if it gets too slow.
    svr_pipeline = make_pipeline(preprocessor, SVR())
    param_grid = [
        {
            'svr__kernel': ['rbf'],
            'svr__gamma': np.logspace(-3, 1, 10),
            'svr__C': [1, 10, 100]
        },
    ]
    grid_search = GridSearchCV(svr_pipeline, param_grid)
    log.info("Training model..")

    # Train on dataset
    grid_search.fit(X, y.values.ravel())
    log.info(f"Best parameter set is: {grid_search.best_params_}")
    log.info(f"Best score is: {grid_search.best_score_}")
    model = grid_search.best_estimator_

    # Save model to disk
    out_path = str(Path(out_dir) / 'model.joblib')
    log.info("Save model")

    dump(model, out_path)


if __name__ == '__main__':
    train_model()
