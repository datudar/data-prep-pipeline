## Data Preprocessing Pipeline

This is a preprocessing [pipeline](/data_preprocessing.py) for handling heterogeneous data such as binary, categorical, and numerical data. 

Note: The steps in this particular pipeline are purely for demonstration purposes, so it is highly recommended you modify the pipeline to suit the needs of your analysis.

### Features

- **Binary features**
	* These are features that have two categories labeled either 1 or 0
	* We keep these features as they are
- **Categorical features**
	* **Numerical categories**: These are features that have **at least three** numerical categories and have no order
	* **Textual categories**: These are features that have **at least three** textual categories
	* We want to transform them into dummy variables of ones and zeros
	* Due to multi-collinearity concerns, we also drop one of the dummy variables so that we are left with n-1 dummies
- **Numerical features**
	* These features are numerical such as integers and floats
	* We want to apply some normalization technique on these values

### Implementation

The example [data file](/input/data_example.csv) contains ten samples with one target column, labeled "y", and eight feature columns of various data types. The pipeline reads in theis data and performs a few basic data preprocessing steps.

First, the data is "upsampled" as it is intentionally imbalanced (i.e., there are only a couple examples of the positive class). Then, we feed the upsampled data through a pipeline which performs: **1) feature selection**, **2) imputation** of missing values, **3) feature engineering** by creating dummy factors and adding polynomial and interaction features, and **4) transformation** using normalization scaling. The final data will contain thirteen samples with sixteen features. The output, y and X, can then be fed directly into a machine learning library such as scikit-learn.



