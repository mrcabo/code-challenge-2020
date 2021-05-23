#' % FIR filter design with Python and SciPy
#' % Matti Pastell
#' % 15th April 2013

#' # Introduction

#' In this report we present the results after 
#' training a ML model on the wine dataset.
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence

in_model = '/home/diego/Coding/code-challenge-2020/data_root/model/model.joblib'
in_data = '/home/diego/Coding/code-challenge-2020/data_root/processed/test.parquet'
out_dir = '/home/diego/Coding/code-challenge-2020/data_root/report'
# Load test dataset and model.
df = pd.read_parquet(in_data, engine='pyarrow')
X, y = df.drop(['points'], axis=1), df[['points']]
model = load(in_model)
pd.set_option('display.precision', 2)
X.describe(include="all")

#' # Metrics

#' The first thing we would like to know is the accuracy of our predictions. 
#' In order to measure it, we will focus on 3 distinct metrics: R2 score, 
#' Mean Squared Error and Mean Absolute Error.

d = {
    'R2': [r2_score(y, model.predict(X))],
    'MSE': [mean_squared_error(y, model.predict(X))],
    'MAE': [mean_absolute_error(y, model.predict(X))]
}
df = pd.DataFrame(d, index=['svr'])
df

#' We can see right away that the $R^{2}$ score is not great. The $R^{2}$ score or 
#' [Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) 
#' is commonly used to measure the accuracy of regression models. It provides 
#' a measure of how well observed outcomes are replicated by the model, 
#' based on the proportion of total variation of outcomes explained by the model.
#' As a reference, a baseline model which always predicts the mean of the observed data $\bar{y}$, will have $R^{2}=0$.
#' 
#' The [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error), 
#' which tells us the arithmetic average of the absolute errors between the 
#' predictions and the real data, is around 2. If we remember that the $y$ values 
#' range from 0 to 100, we could say that the model gives quite good predictions. 
#' However we have to look at the [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error).
#' The MSE squares the error between the predictions and the real data. It is 
#' much more sensitive to outliers than the MAE. This makes sense since we saw 
#' some pretty big outliers in the dataset. (It's harder for the model to learn to predict $1300 wine bottles when there is only one datapoint.)

#' # Permutation Importance

#' The [permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) 
#' is defined to be the decrease in a model score when a single feature value is 
#' randomly shuffled. It can gives us an idea of which variable are more relevant 
#' when calculating predictions.

r = permutation_importance(model, X, y,
                           n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{X.columns[i]:<14}"
              f"{r.importances_mean[i]:.3f}"
              f" ± {r.importances_std[i]:.3f}")

#' # Plotting predictions vs labels $y$

fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(y, model.predict(X))
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
# plt.text(3, 20, string_score)
plt.title('Ridge model')
plt.ylabel('Model predictions')
plt.xlabel('Truths')
plt.show()

#' # Partial dependence plots

#' Intuitively, we can interpret [partial dependence plots](https://scikit-learn.org/stable/modules/partial_dependence.html#partial-dependence-plots) 
#' as the expected target response as a function of the input features of interest

plot_partial_dependence(model, X, ['price'])  
