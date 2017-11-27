## Data Preprocessing Pipeline

This is a [preprocessing pipeline](/data_preprocessing.py) for handling heterogeneous data such as binary, categorical, and numerical data in tabular form. The [example data](/input/data_example.csv) contains ten samples with one target column, labeled "y", and eight feature columns.

The data is intentionally imbalanced (i.e., there are just a couple of examples of the positive class, "y" = 1), so we first "upsample" the positive class. Then, we feed the upsampled data through a pipeline which performs: **feature selection**, **imputation** of missing values, **feature engineering** by adding polynomial and interaction features, and **transformation** using normalization scaling. The steps in this particular pipeline are purely for demonstration purposes, so it is highly recommended you modify the pipeline to suit the needs of your analysis. 

The final data will contain thirteen samples with sixteen feature columns. The output, y and X, can then be fed directly into a machine learning library such as Scikit-learn.

### Binary features

- These are features that have two categories labeled either 1 or 0
- We keep these features as they are

### Categorical features

- **Numerical categories**: These are features that have **at least three** numerical categories and have no order
- **Textual categories**: These are features that have **at least three** textual categories
- We want to transform them into dummy variables of ones and zeros
- Due to multi-collinearity concerns, we also drop one of the dummy variables so that we are left with n-1 dummies

### Numerical features

- These features are numerical such as integers and floats
- We want to apply some normalization technique on these values

