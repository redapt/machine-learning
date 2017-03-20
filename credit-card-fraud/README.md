# Predicting Credit Card Fraud

In this article, I will be using the free, open source "[Credit Card Fraud Detection](https://www.kaggle.com/dalpozz/creditcardfraud)" dataset found on Kaggle.

The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced; the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Due to confidentiality issues, the authors of the dataset cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are "Time" and "Amount". Feature "Time" contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature "Amount" is the transaction amount (this feature can be used for example-dependant cost-senstive learning). Feature "Class" is the response variable and it takes value 1 in case of fraud and 0 otherwise (note: I will be renaming this feature to "Fraud" later on in the analysis).

The goal for this analysis is to predict credit card fraud in the transactional data. I will be using TensorFlow to build the predictive model, and t-distributed stochastic neighbor embedding (t-SNE) to visualize the dataset in two dimensions.

I will also use [Amazon Machine Learning](https://aws.amazon.com/machine-learning/) (Amazon ML) tools and services and provide comparisons against TensorFlow and t-SNE. If I have time, I will also use the [Google Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/) (Google ML).

## Exploring the data

Below are what the first 5 entries (or lines) in the dataset with the V3..V27 feature/columns removed for formatting reasons:

| Time | V1        | V2        | .. | V28       | Amount | Class |
|------|-----------|-----------|----|-----------|--------|-------|
| 0.0  | -1.359807 | -0.072781 | .. | -0.021053 | 149.62 | 0     |
| 0.0  |  1.191857 |  0.266151 | .. |  0.014724 |   2.69 | 0     |
| 1.0  | -1.358354 | -1.340163 | .. | -0.059752 | 378.66 | 0     |
| 1.0  | -0.966272 | -0.185226 | .. |  0.061458 | 123.50 | 0     |
| 2.0  | -1.158233 |  0.877737 | .. |  0.215153 |  69.99 | 0     |


# Glossary
TensorFlow is an open source software library for machine learning across a range of tasks, and developed by Google to meet their needs for systems capable of building and training neural networks to detect and decipher patterns and correlations, analogous to the learning and reasoning which humans use.

t-distributed stochastic neighbor embedding (t-SNE) is a machine learning algorithm for dimensionality reduction. It is a nonlinear dimensionality reduction technique that is particularly well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot. Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points.

# Sources
* Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. "Calibrating Probability with Undersampling for Unbalanced Classification". _In Symposium on Computational Intelligence and Data Mining (CIDM)_, IEEE, 2015.
* [Credit Card Fraud Detection](https://www.kaggle.com/dalpozz/creditcardfraud) &mdash; Anonymized credit card transactions labeled as fraudulent or genuine
