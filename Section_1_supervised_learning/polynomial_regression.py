import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Assign the data to predictor and outcome variables
train_data = pd.read_csv("data/polynomial_regression_data")
X = train_data["Var_X"].values.reshape((20, 1))
y = train_data["Var_Y"]


# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!
degree_range = range(18, 23)
_error = []
for degree in degree_range:
    # Create polynomial features
    poly_feat = PolynomialFeatures(degree=degree)
    X_poly = poly_feat.fit_transform(X)

    # Make and fit the polynomial regression model
    poly_model = LinearRegression().fit(X_poly, y)

    _pred = poly_feat.fit_transform(X)
    _error.append(((poly_model.predict(_pred) - y) ** 2).mean(axis=0))

plt.plot( degree_range, _error)
# plt.scatter(X, y, zorder=3)
plt.show()

