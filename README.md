# credit-risk-classification

## Overview of the Analysis
The goal of this analysis was to assess loan risk by predicting whether a borrower is likely to default on a loan.
Using machine learning, we aimed to classify loans as either high-risk (1) or healthy (0) based on financial data.

The dataset contained financial information such as
* the size of the loan
* its interest rate
* the borrower's income
* the debt to income ratio
* the number of accounts the borrower held
* derogatory marks against the borrower
* the total debt

Our target variable was "loan_status", where:
0 represents a healthy loan (low risk).
1 represents a high-risk loan (likely default).

Distribution of Target Variable (loan_status)
The dataset (77,536 data points) was split into training and testing sets. The training set was used to build an initial logistic regression model (Logistic Regression Model 1) using the LogisticRegression module from scikit-learn. Logistic Regression Model 1 was then applied to the testing dataset. 
0 (Healthy Loans): 75,036
1 (High-Risk Loans): 2,500
This indicates that the dataset is imbalanced, with far more healthy loans than high-risk loans.

This intial model was drawing from a dataset that had 75,036 low-risk loan data points and 2,500 high-risk data points. To resample the training data and ensure that the logistic regression model had an equal number of data points to draw from, the training set data was resampled with the RandomOverSampler module from imbalanced-learn. This generated 56,277 data points for both low-risk (0) and high-risk (1) loans, based on the original dataset.


Stages of the Machine Learning Process

Data Preprocessing
Loaded the dataset and separated features (X) and labels (y).
Split the data into training (75%) and testing (25%) sets.

Model Selection & Training
Chose Logistic Regression, a common classification algorithm.
Trained the model using the training data.

Model Evaluation
Made predictions on the testing set.
Evaluated performance using accuracy, precision, recall, and F1-score.

## Results
<strong>Logistic Regression Model 1:</strong>
Metric		Class 0 (Healthy)	Class 1 (High-Risk)
Precision	1.00				0.87
Recall		1.00				0.95
F1-score	1.00				0.91
Accuracy	99%					-

<strong>Logistic Regression Model 2:</strong>
Metric		Class 0 (Healthy)	Class 1 (High-Risk)
Precision	1.00				0.87
Recall		1.00				1.00
F1-score	1.00				0.93
Accuracy	100%					-

## Summary
Best Performing Model: Logistic Regression

The overall accuracy of model 1 is 99%, meaning the model correctly predicts loan status most of the time.
High recall (0.95) for high-risk loans suggests the model is effective at identifying borrowers likely to default.
If minimizing loan defaults is the top priority, recall for high-risk loans (1) is key. The model captures 95% of high-risk loans, making it a strong choice.
If reducing false positives (incorrectly labeling healthy loans as high-risk) is a concern, then further tuning may be required to improve precision for class 1.

Logistic Regression Model 2 is less likely to predict false negative results. However, based on the confusion matrices for each model, Logistic Regression Model 2 predicted slightly more false positives (low-risk when the actual was high-risk). 

Recommendation: Use Logistic Regression, but consider additional strategies to further refine predictions.