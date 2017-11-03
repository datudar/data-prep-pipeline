## <p align="center">Data Preprocessing Pipeline</p>

This is a data preprocessing tool for handling heterogeneous data such as binary, categorical (both numerical and textual), and numerical data.

The output of this pipeline is a NumPy array that you can feed directly into a Scikit-learn algorithm.

The basic steps in the pipeline are:

 1. feature selection
 2. imputation of missing values
 3. transformation

### Binary features

- These are features that have two categories labeled either 1 or 0
- We keep these features as they are

### Categorical features

- **Numerical categories**: These are features that have **at least three** numerical categories and have no order.
- **Textual categories**: These are features that have **at least three** textual categories
- We want to transform them into dummy variables of ones and zeros
- Due to multi-collinearity concerns, we also drop one of the dummy variables so that we are left with n-1 dummies

### Numerical features

- These features that are numerical such as integers and floats
- We want to apply some normalization technique on these values

