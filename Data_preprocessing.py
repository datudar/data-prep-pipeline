#==============================================================================
# Import packages
#==============================================================================

import numpy as np
import pandas as pd
# Transformer to select a subset of the Pandas DataFrame columns
from sklearn.base import BaseEstimator, TransformerMixin    
# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
# Data preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

#==============================================================================
# Custom classes and functions
#==============================================================================

# Class to select columns since Scikit-Learn doesn't handle DataFrames yet
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Function to calculate the most frequent label in a feature
def impute_txtcat_feature(f):
    return df[f].fillna(df[f].value_counts().index[0], inplace=True)

# Class to remove (extraneous) dummy feature
class FeatureDropFirst(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass    
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, 1:]

#==============================================================================
# Data import
#==============================================================================

ID = 'id'
DATA_FILE = "Data_example.csv"

df = pd.read_csv(DATA_FILE, index_col=ID, header=0)

# Select features: binary ('bin'), numerical categorical ('numcat'),
# textual categorical ('strcat'), and numerical ('num'). 
#
# bin: 1. these are features that have two categories labeled either 1 or 0
# ===  2. we want to keep the columns as they are
#
# numcat: 1. these are features that have at least three numerical categories
# ======  2. we want to tranform them into dummy variables
#
# txtcat: 1. these are features that have at least three textual categories
# ======  2. we want to tranform them into dummy variables
#
# num: 1. these features that are numerical such as integers and floats
# ===  2. we want to normalize these values

bin_features = [f for f in df.columns if f[3:len(f)] == 'bin']
numcat_features = [f for f in df.columns if f[3:len(f)] == 'numcat']
# Note: for textual categorical features, I only know how to apply it
# on one column at a time. In a future version, I will try to make it 
# work on a list of features all in one pipeline
txtcat_features = 'x5_txtcat'
num_features = [f for f in df.columns if f[3:len(f)] == 'num']

#==============================================================================
# Preprocessing pipeline
#==============================================================================

# 1. Select features
# 2. Impute missing values with the median
bin_pipeline = Pipeline([
        ('selector', FeatureSelector(bin_features)),
        ('imputer', Imputer(missing_values=np.nan, strategy='median', axis=0)),
    ])

# 1. Select features
# 2. Impute missing values with the median
# 3. Create binary features for each category
# 4. Due to ulti-collinearity, remove one feature to retain n-1 dummy features
# Note: An alternative is use get_dummies in pandas with drop_first=True
numcat_pipeline = Pipeline([
        ('selector', FeatureSelector(numcat_features)),
        ('imputer', Imputer(missing_values=np.nan, strategy='median', axis=0)),
        ('encoder', OneHotEncoder(sparse=False)),
        ('remover', FeatureDropFirst()),        
    ])

# 1. Select features
# 2. Impute missing values with the most frequent category
# 3. Create binary features for each category
# 4. Due to multi-collinearity concerns, remove the firt dummy feature to retain n-1 dummy features
# Note: An alternative method is to use pandas.get_dummies(data, drop_first=True)
txtcat_pipeline = Pipeline([
            ('selector', FeatureSelector(txtcat_features)),
            ('imputer', impute_txtcat_feature(txtcat_features)),
            ('binarizer', LabelBinarizer()),
            ('remover', FeatureDropFirst()),
        ])
  
# 1. Select features
# 2. Impute missing values with the mean
# 3. Scale the values using standard normalization: (x-mean(x))/stdev(x)
num_pipeline = Pipeline([
        ('selector', FeatureSelector(num_features)),
        ('imputer', Imputer(missing_values=np.nan, strategy='mean', axis=0)),
        ('normalizer', StandardScaler()),
    ])

# Combine all pipelines into a single pipeline
full_pipeline = FeatureUnion(transformer_list=[
        ("bin_pipeline", bin_pipeline),
        ("numcat_pipeline", numcat_pipeline),
        ("txtcat_pipeline", txtcat_pipeline),
        ("num_pipeline", num_pipeline),
    ])

# Execute entire pipeline. If the output is a sparse matrix,
# then convert it to a dense matrix using the toarray method.
try:
    final_data = full_pipeline.fit_transform(df).toarray()
except AttributeError:
    final_data = full_pipeline.fit_transform(df)