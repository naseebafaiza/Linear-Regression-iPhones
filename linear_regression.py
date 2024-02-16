
########### the first part does linear regression on each feature, calculating RSME values for each feature ###########
########### we want to then sort out the features in increasing order of RSME to see the impact of simple linear regression ###########

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score

iPhone_file_path = 'iPhone.csv'
iPhone_file_data = pd.read_csv(iPhone_file_path)

# Removing the non-numerical (categorial) NAME column
iPhone_numerical_data = iPhone_file_data.drop(columns=['NAME'])

# store RSME results in a dictionary
rmse_by_feature = {}

# for loop through each column
for feature in iPhone_numerical_data.columns[:-1]:
    X = iPhone_numerical_data[[feature]] # independent - features
    y = iPhone_numerical_data['CO2E'] # target variable - CO2E
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # initialize 
    linear_regression_model = LinearRegression()

    # fit model on training data
    linear_regression_model.fit(X_train, y_train)

    # Predict on the testing set
    y_predict = linear_regression_model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))

    # store in dictionary
    rmse_by_feature[feature] = rmse

items = rmse_by_feature.items()

# We define a function that takes an item and returns its second element (the RMSE score)
def getRMSE(item):
    return item[1] # gets the RMSE, in this case

sorted_rmse_scores = sorted(items, key = getRMSE) # so it is sorted in terms of RMSE
sorted_rmse_scores_dictionary = dict(sorted_rmse_scores)
print(f'Sorted dictionary with RMSE scores: {sorted_rmse_scores_dictionary}')

########### now, we want to perform cross validation on the 5 most important features, get the average RMSE out of those 5 important features ###########

def calculate_cv_rmse(feature):
    X = iPhone_numerical_data[[feature]] # Independent - feature
    y = iPhone_numerical_data['CO2E'] # Target variable - CO2E

    model = LinearRegression()

    # Perform 5-fold cross-validation and compute the scores
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

    # Calculate RMSE for each fold and take the average
    rmse_scores = np.sqrt(-scores) # Convert MSE to RMSE
    average_rmse = np.mean(rmse_scores) # Average RMSE across folds

    return average_rmse

top_5_features = list(sorted_rmse_scores_dictionary.keys())[:5]
cross_val_rmse_results = {}
for feature in top_5_features:
    cross_val_rmse_results[feature] = calculate_cv_rmse(feature)

# Sort the cross-validated RMSE values in ascending order to see the impact
sorted_cv_rmse_results = dict(sorted(cross_val_rmse_results.items(), key=lambda item: item[1]))
print(f'\nSorted dictionary with RMSE scores AFTER cross validation: {sorted_cv_rmse_results}')

########### MULTIPLE LINEAR REGRESSION ###########

# Independent variables (all features)
X = iPhone_numerical_data.drop('CO2E', axis=1)
y = iPhone_numerical_data['CO2E']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing set
y_predict = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

print(f'\nRMSE for Multiple Linear Regression: {rmse}')

########### MULTIPLE LINEAR REGRESSION AFTER 5 FOLD CROSS VALIDATION ###########

# Re-perform 5-fold cross-validation and compute the scores
five_fold_CV_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
five_fold_CV_scores_RMSE = np.sqrt(-five_fold_CV_scores)  # Convert MSE to RMSE
five_fold_CV_scores_AVERAGE_RMSE = np.mean(five_fold_CV_scores_RMSE)  # Average RMSE across folds

print(f'\nAverage RMSE after 5 fold cross validation on all features: {five_fold_CV_scores_AVERAGE_RMSE}')