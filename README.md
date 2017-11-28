## Data Preprocessing Pipeline

This is a preprocessing [pipeline](/data_preprocessing.py) for handling heterogeneous data such as **binary**, **categorical**, and **numerical** data. 

Note: The steps in this particular pipeline are purely for demonstration purposes, so it is highly recommended you modify the pipeline to suit the needs of your analysis.

### Data

The example [data file](/input/data_example.csv) contains ten made-up samples of one target column (labeled "y") and eight feature columns of various data categories.

#### Target
- The target column has two categories: the positive class and the negative class, which are labeled 1 and 0, respectively

#### Features
- **Binary**: columns 1 and 2
- **Categorical**:
	* **Numerical categories**: columns 3 and 4
		* These are features that have **at least three** numerical categories and have no order
	* **Textual categories**: columns 5 and 6 
		* These are features that have **at least three** textual categories
	* We want to transform them into dummy variables of ones and zeros
	* Due to multi-collinearity concerns, we also drop one of the dummy variables so that we are left with n-1 dummy variables
- **Numerical**: columns 7 and 8
	* These features are typically integers or floats
	* We want to apply a normalization technique on these values

### Implementation

The pipeline reads in the data and performs a few basic preprocessing steps. First, the data is "upsampled" as it is intentionally imbalanced (i.e., there are only two examples of the positive class). Then, it is fed through a pipeline which performs: 

1. **imputation** of missing values
2. **feature engineering** by creating dummy variables, adding polynomial features, and adding interaction features
3. **transformation** using normalization scaling
4. **feature selection** by placing a minimum requirement for the variability of the feature

Finally, the pipeline outputs thirteen samples with sixteen features. The output, y and X, are now ready for further analysis.