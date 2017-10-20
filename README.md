## Data Preprocessing Pipeline

As data scientists, a lot of our time is spent cleaning and transforming data in some meaningful way. This is a data preprocessing pipeline is a helpful tool for handling heterogeneous data such as binary, categorical, and numerical data. The basic steps in the pipeline are:

 1. feature selection
 2. imputation of missing values
 3. transformation

The output of the pipeline is a NumPy array that one can feed directly into a Scikit-Learn algorithm.

#### Binary features
- These are features that have two categories labeled either 1 or 0
- We want to keep the features as they are

#### Numerical categorical features
- These are features that have **at least three** numerical categories
- We want to transform these features into dummy variables of ones and zeros
- Also, due to multicollinearty concerns, we would also like to drop one of the dummy variable so that we are left with n-1 dummies

#### Textual categorical features
- These are features that have **at least three** textual categories
- We want to transform them into dummy variables of ones and zeros

#### Numerical features
- These features that are numerical such as integers and floats
- We want to apply some normalization technique on these values

