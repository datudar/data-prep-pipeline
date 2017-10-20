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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#==============================================================================
# Custom transformer classes
#==============================================================================

# Class to select columns
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, x):
        return self
    def transform(self, x):
        return x[self.attribute_names].values

# Class to impute textual category
class ImputerTextualCategory(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  
    def fit(self, df, y=None):
        return self
    def transform(self, x): 
        return pd.DataFrame(x).apply(lambda x: x.fillna(x.value_counts().index[0]))

# Class to encode lables across multiple columns
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, astype=int):
        self.astype = astype
    def fit(self, x, y=None):
        return self
    def transform(self, x, y=None):
        if self.astype == int:
            return pd.DataFrame(x).apply(LabelEncoder().fit_transform)
        else:
            return pd.DataFrame(x).apply(LabelEncoder().fit_transform).astype(str)
                            
# Class for one-hot encoding of textual categorical values and optionally
# drop (extraneous) dummy feature
class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, drop_first=False):
        self.drop_first = drop_first
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return pd.get_dummies(x, drop_first=self.drop_first)
    
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
# ======  2. we want to transform them into dummy variables
#
# txtcat: 1. these are features that have at least three textual categories
# ======  2. we want to transform them into dummy variables
#
# num: 1. these features that are numerical such as integers and floats
# ===  2. we want to normalize these values

bin_features = [f for f in df.columns if f[3:len(f)] == 'bin']
numcat_features = [f for f in df.columns if f[3:len(f)] == 'numcat']
txtcat_features = [f for f in df.columns if f[3:len(f)] == 'txtcat']
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
# 3. Encode each feature and set the type to string so that the GetDummies class
#    (which uses pandas.get_dummies) can transform labels into dummy variables 
# 4. Create one-hot encoding for each feature and (due to multi-collinearity concerns)
#    remove the first dummy feature to retain n-1 dummy features
numcat_pipeline = Pipeline([
        ('selector', FeatureSelector(np.array(numcat_features))),
        ('imputer', Imputer(missing_values=np.nan, strategy='median', axis=0)),
        ('labelencoder', MultiColumnLabelEncoder(astype=str)),
        ('getdummies', GetDummies(drop_first=True)),
    ])

# 1. Select features
# 2. Impute missing values with the most frequent category
# 3. Create one-hot encoding for each feature and (for multi-collinearity concerns)
#    remove the first dummy feature to retain n-1 dummy features
txtcat_pipeline = Pipeline([
        ('selector', FeatureSelector(np.array(txtcat_features))),
        ('imputer', ImputerTextualCategory()),
        ('getdummies', GetDummies(drop_first=True)),
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
    final_data = pd.DataFrame(full_pipeline.fit_transform(df).toarray())
except AttributeError:
    final_data = pd.DataFrame(full_pipeline.fit_transform(df))

#==============================================================================
# The End
#==============================================================================