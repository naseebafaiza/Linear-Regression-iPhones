# iPhone CO2E Prediction Analysis

## Overview
This project involves linear regression analysis on various iPhone features to predict CO2E emissions. The analysis includes both simple and multiple linear regression models, with and without cross-validation, to understand the impact of each feature on prediction accuracy.

## Simple Linear Regression
- Each feature is individually used to predict CO2E, calculating RMSE for each.
- Features are sorted by RMSE to identify the most predictive ones.

## Cross-validation on Top 5 Features
- 5-fold cross-validation is performed on the 5 features with the lowest RMSE to refine their accuracy.

## Multiple Linear Regression
- All features are used together to predict CO2E, showing the combined effect on prediction accuracy.
- RMSE is calculated to evaluate the model's performance.

## Results
- **Simple Linear Regression**: Identified Storage as the most accurate predictor.
- **Cross-validation**: Refined accuracy, with Storage still being the most significant.
- **Multiple Linear Regression**: Demonstrated a lower RMSE, indicating improved prediction accuracy when using all features.
- **5-Fold Cross-Validation on Multiple Linear Regression**: Provided a more realistic estimate of the model's predictive accuracy, adjusting for potential overfitting.

## Conclusion
The analysis highlights the importance of combining features for predicting CO2E emissions accurately, with multiple linear regression offering a comprehensive approach.

