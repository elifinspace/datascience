from sklearn.ensemble import AdaBoostClassifier

# model variable is a decision tree model that 
# has been fitted to the data x_values and y_values. The functions fit and predict work exactly as before
model = AdaBoostClassifier()
model.fit(x_train, y_train)
model.predict(x_test)
# base_estimator: The model utilized for the weak learners 
# (Warning: Don't forget to import the model that you decide to use for the weak learner).
# n_estimators: The maximum number of weak learners used.
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=4)
