## Data Preprocessing Pipeline

This is a preprocessing [pipeline](/data_preprocessing.py) for handling heterogeneous data such as binary, categorical, and numerical data. 

Note: The steps in this particular pipeline are purely for demonstration purposes, so it is highly recommended you modify the pipeline to suit the needs of your analysis.

### Features

The example [data file](/input/data_example.csv) contains ten samples with one target column, labeled "y", and eight feature columns of various data types. The following are characteristics of the three data categories:

- **Binary features**: columns 1 and 2
	* These are features that have two categories: the positive class or the negative class,  labeled either 1 or 0, respectively
	* We keep these features as they are
- **Categorical features**:
	* **Numerical categories**: columns 3 and 4
		* These are features that have **at least three** numerical categories and have no order
	* **Textual categories**: columns 5 and 6 
		* These are features that have **at least three** textual categories
	* We want to transform them into dummy variables of ones and zeros
	* Due to multi-collinearity concerns, we also drop one of the dummy variables so that we are left with n-1 dummies
- **Numerical features**: columns 7 and 8
	* These features are numerical such as integers and floats
	* We want to apply some normalization technique on these values

### Implementation

The pipeline reads in the example data and performs the following data preprocessing steps:

First, the data is "upsampled" as it is intentionally imbalanced (i.e., there are only a couple of examples of the positive class). Then, we feed the upsampled data through a pipeline which performs: 1) **feature selection**, 2) **imputation** of missing values, 3) **feature engineering** by creating dummy factors and adding polynomial and interaction features, and 4) **transformation** using normalization scaling.

The final data will contain thirteen samples with sixteen features, and the output, y and X, can be fed directly into a machine learning library such as scikit-learn.



