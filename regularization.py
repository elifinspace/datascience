
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
# Assign the data to predictor and outcome variables
train_data = pd.read_csv("data/regularization.csv", header=None)
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# scale added
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.fit(X_scaled, y).coef_
print(reg_coef)