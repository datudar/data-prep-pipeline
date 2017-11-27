#==============================================================================
# Import packages
#==============================================================================

import numpy as np
import pandas as pd
# Utilities
from sklearn.utils import resample
# Transformer to select a subset of the Pandas DataFrame columns
from sklearn.base import BaseEstimator, TransformerMixin    
# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
# Data preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
# Feature selection
from sklearn.feature_selection import VarianceThreshold

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

# Class to encode labels across multiple columns
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
# drop the first dummy feature (if multi-collinearity is a concern)
class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, drop_first=False):
        self.drop_first = drop_first
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return pd.get_dummies(x, drop_first=self.drop_first)

#==============================================================================
# Initialization Settings
#==============================================================================

ID = 'id'
Y = 'y'
DIR = "input"
DATAFILE = "{0}/data_example.csv".format(DIR)
NTRAINROWS = None # Number of rows of data file to read; None reads all rows
UPSAMPLEPCT = .4 # Percent of samples to have positive class; 0 <= pct < 1
SEED = 42 # Seed state for reproducibility
VARTHRESHOLD = .001 # Minimum variability allowed for features

#==============================================================================
# Data import
#==============================================================================

df = pd.read_csv(DATAFILE, index_col=ID, header=0, nrows=NTRAINROWS)

# Separate majority and minority classes
df_majority = df[df[Y]==0]
df_minority = df[df[Y]==1]

# Upsample minority class with replacement
df_minority_sampled = resample(df_minority, 
                               replace=True,
                               n_samples=int(UPSAMPLEPCT*df_majority.shape[0]/(1-UPSAMPLEPCT)),
                               random_state=SEED)
 
# Combine majority class with upsampled minority class
df_sampled = pd.concat([df_majority, df_minority_sampled])

# Shuffle all the samples
df_sampled = resample(df_sampled, replace=False, random_state=SEED)

# Separate y and X variables
y = df_sampled[Y]
X = df_sampled.loc[:, df_sampled.columns != Y]

#==============================================================================
# Preprocessing pipeline
#==============================================================================

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

bin_features = [f for f in X.columns if f[3:len(f)] == 'bin']
numcat_features = [f for f in X.columns if f[3:len(f)] == 'numcat']
txtcat_features = [f for f in X.columns if f[3:len(f)] == 'txtcat']
num_features = [f for f in X.columns if f[3:len(f)] == 'num']

# 1. Select features
# 2. Impute missing values with the median
bin_pipeline = Pipeline([
        ('selector', FeatureSelector(bin_features)),
        ('imputer', Imputer(missing_values=np.nan, strategy='median', axis=0)),
        ('threshold', VarianceThreshold(VARTHRESHOLD)),
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
        ('threshold', VarianceThreshold(VARTHRESHOLD)),
    ])

# 1. Select features
# 2. Impute missing values with the most frequent
# 3. Create one-hot encoding for each feature and (for multi-collinearity concerns)
#    remove the first dummy feature to retain n-1 dummy features
txtcat_pipeline = Pipeline([
        ('selector', FeatureSelector(np.array(txtcat_features))),
        ('imputer', ImputerTextualCategory()),
        ('getdummies', GetDummies(drop_first=True)),
        ('threshold', VarianceThreshold(VARTHRESHOLD)),
    ])
  
# 1. Select features
# 2. Impute missing values with the mean
# 3. Scale the values using standard normalization: (x-mean(x))/stdev(x)
num_pipeline = Pipeline([
        ('selector', FeatureSelector(num_features)),
        ('imputer', Imputer(missing_values=np.nan, strategy='mean', axis=0)),
        ('poly', PolynomialFeatures(2, interaction_only=False)),
        ('normalizer', StandardScaler()),
        ('threshold', VarianceThreshold(VARTHRESHOLD)),
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
    X = pd.DataFrame(full_pipeline.fit_transform(X).toarray())
except AttributeError:
    X = pd.DataFrame(full_pipeline.fit_transform(X))

#==============================================================================
# The End
#==============================================================================