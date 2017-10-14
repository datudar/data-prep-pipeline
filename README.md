## Data Preprocessing Pipeline

This is a data preprocessing "pipeline" that handles binary, categorical, and numerical data. It selects features from a sample data file ("Data_example.csv"), imputes their missing values, and performs a transformation.

The final data is an array which you can feed directly into the machine learning algorithm of your choice.

### Binary features
- these are features that have two categories labeled either 1 or 0
- we want to keep the features as they are

### Numerical categorical features
- these are features that have **at least three** numerical categories
- we want to transform these features into dummy variables of ones and zeros

### Textual categorical features
- these are features that have **at least three** textual categories
- we want to tranform them into dummy variables of ones and zeros

### Numerical features
- these features that are numerical such as integers and floats
- we want to normalize these values

