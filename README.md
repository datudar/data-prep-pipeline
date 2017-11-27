## Data Preprocessing Pipeline

This is a preprocessing pipeline for handling heterogeneous data such as binary, categorical, and numerical data.

The [example data file](/input/data_example.csv) contains ten samples with one target column (y) and eight feature columns (X). The data is intentionally imbalanced (i.e., few samples where y=1),so the data is first "upsampled" before it is run through the pipeline. 

The basic steps in the pipeline are:

 1. feature selection
 2. imputation
 3. feature engineering
 4. transformation

After the data has been upsampled and fed through pipeline, the final data will contain thirteen samples with sixteen feature columns. The output, y and X, can then be fed directly into machine learning library fo your choosing, such as Scikit-learn.

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

