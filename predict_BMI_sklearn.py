"""1. Load the data

The data is in the file called "bmi_and_life_expectancy.csv".
Use pandas read_csv to load the data into a dataframe (don't forget to import pandas!)
Assign the dataframe to the variable bmi_life_data.
2. Build a linear regression model

Create a regression model using scikit-learn's LinearRegression and assign it to bmi_life_model.
Fit the model to the data.
3. Predict using the model

Predict using a BMI of 21.07931 and assign it to the variable laos_life_exp."""

import pandas as pd
from sklearn.linear_model import LinearRegression
# Assign the dataframe to this variable.
bmi_life_data = pd.read_csv("data/bmi_and_life_expectancy.csv", sep=",")

# Make and fit the linear regression model
model = LinearRegression()
bmi_life_model = model.fit(bmi_life_data[["BMI"]], bmi_life_data[["Life expectancy"]])  # bmi_life_data[:, 2] is BMI

# Make a prediction using the model
laos_life_exp = bmi_life_model.predict([[21.07931]])

