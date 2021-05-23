import logging
from pathlib import Path

import click
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--in-data')
@click.option('--in-model')
@click.option('--out-dir')
def evaluate_model(in_data, in_model, out_dir):
    """Evaluate the trained model on the test dataset. Then, produce a report 
    and save that to disk.

    Parameters
    ----------
    in-data: str
        Path to test dataset in disk.
    in-model: str
        Path to trained model stored in disk.
    out-dir: str
        Path where results will be saved.

    Returns
    -------
    None
    """
    log = logging.getLogger('evaluate-model')
    log.info("Reading data")
    df = pd.read_parquet(in_data, engine='pyarrow')
    X, y = df.drop(['points'], axis=1), df[['points']]
    model = load(in_model)
    pd.set_option('display.precision', 2)
    # X.describe(include="all")
    
    # Metrics
    d = {
        'R2': [r2_score(y, model.predict(X))],
        'MSE': [mean_squared_error(y, model.predict(X))],
        'MAE': [mean_absolute_error(y, model.predict(X))]
    }
    df = pd.DataFrame(d, index=['svr'])
    log.info(df)

    # The permutation feature importance 
    r = permutation_importance(model, X, y,
                               n_repeats=30,
                               random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            log.info(f"{X.columns[i]:<14}"
                     f"{r.importances_mean[i]:.3f}"
                     f" ± {r.importances_std[i]:.3f}")

    # Plots
    out_path = Path(out_dir)

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(y, model.predict(X))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
    plt.title('Model predictions vs Ground truth')
    plt.ylabel('Model predictions')
    plt.xlabel('Truths')
    plt.savefig(out_path/'predict_vs_truth.png')

    partial_dependence = plot_partial_dependence(model, X, ['price']).figure_
    partial_dependence.savefig(out_path/'partial_dependence.png')

    flag = out_path / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    evaluate_model()
