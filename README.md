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

## Instructions
For detailed instructions, refer to the [Instructions PDF](instructions.pdf).

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
Model 1 Observations: 
* The model performed extremely well in predicting healthy loans (0), with nearly perfect precision and recall.
* However, it still missed 5% of high-risk loans (1), meaning some defaults were misclassified as safe loans.

Model 2 Observations: 
* Recall for 1 (high-risk loans) improved from 95% to 100%, meaning the model correctly identified all default loans in the test set.
* Precision for 1 stayed the same (0.87), meaning some false positives (loans incorrectly labeled as high-risk) still exist.
* The overall accuracy increased slightly to 100%, though this may be an overfit due to oversampling. 

Recommendation: Use Logistic Regression, but consider additional strategies to further refine predictions. The oversampled model is preferable if the primary concern is identifying all high-risk loans (1), since it achieves 100% recall—meaning no defaults were misclassified as safe loans.
However, the false positive rate (precision) for 1 remains the same, meaning some healthy loans are still incorrectly labeled as high-risk.
* If misidentifying safe loans as high-risk is not a major issue, the oversampled model is the better choice.
* If misclassifying healthy loans as high-risk could create business problems (e.g., rejecting good applicants), then the original model may be preferable.