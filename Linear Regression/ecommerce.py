"""
Linear Regression (Multivariate)
@author Vivek Kumar
"""

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Read data from file
customers = pd.read_csv('Ecommerce Customers')

# Input Features
X = customers[[
    'Avg. Session Length', 'Time on App', 'Time on Website',
    'Length of Membership'
]]

# Output Feature
y = customers[['Yearly Amount Spent']]

# Split data into training and testing set
# test size = 0.3 and random_state=101
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# Create an instance of Linear Regression Model
lm = LinearRegression()

# Train/fit lm on the training data
lm.fit(X_train, y_train)

# Print the coefficients of the model (i.e θ values)
print('Coefficients of Model - {}'.format(lm.coef_))

# Predict test values
predictions = lm.predict(X_test)

# Plot real test values vs predicted values
plt.plot(y_test, predictions)
plt.xlabel('Y Test Values')
plt.ylabel('Predicted values')
plt.show()

# Evaluate performance of model
print('Mean Absolute Error: {}'.format(
    metrics.mean_absolute_error(y_test, predictions)))
print('Mean Squared Error: {}'.format(
    metrics.mean_squared_error(y_test, predictions)))
print('Mean Squared Error: {}'.format(
    np.sqrt(metrics.mean_squared_error(y_test, predictions))))
print('Expalined Variance Score: {}'.format(
    metrics.explained_variance_score(y_test, predictions)))

# Plot a histogram of residuals and make sure it looks uniformly distributed
sns.distplot((y_test - predictions), bins=50)
