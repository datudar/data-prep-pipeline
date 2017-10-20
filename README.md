## Data Preprocessing Pipeline

This is a data preprocessing pipeline that handles heterogeneous data such as binary, categorical, and numerical data. The basic steps in the pipeline are:

 1. feature selection
 2. imputation
 3. transformation

The output if the pipeline is a NumPy array that one can feed directly into a Scikit Learn algorithm.

#### Binary features
- these are features that have two categories labeled either 1 or 0
- we want to keep the features as they are

#### Numerical categorical features
- these are features that have **at least three** numerical categories
- we want to transform these features into dummy variables of ones and zeros

#### Textual categorical features
- these are features that have **at least three** textual categories
- we want to transform them into dummy variables of ones and zeros

#### Numerical features
- these features that are numerical such as integers and floats
- we want to normalize these values

