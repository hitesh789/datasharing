%cd C:\Users\bama6012\Desktop\Python My study\data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"
data=pd.read_csv('adult.csv',header=None, index_col=False,
names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
'marital-status', 'occupation', 'relationship', 'race', 'gender',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
'income'])

# For illustration purposes, we only select some of the columns
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week','occupation', 'income']]

# One-Hot-Encoding (Dummy Variables)-------------------------------------------
print("Original features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Features after get_dummies:\n", list(data_dummies.columns))

data_dummies.head()

features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']

# Extract NumPy arrays
X = features.values
y = data_dummies['income_ >50K'].values
print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))

# create a DataFrame with an integer feature and a categorical string feature
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],'Categorical Feature': ['socks', 'fox', 'socks', 'box']})

# Using get_dummies will only encode the string feature and will not change the integer feature
pd.get_dummies(demo_df)

"""
If you want dummy variables to be created for the “Integer Feature” column, you can
explicitly list the columns you want to encode using the columns parameter. Then,
both features will be treated as categorical
"""
demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])

#----------------------------------------------------------------------------------Automatic Feature Selection
# There are three basic strategies: 
# 1) univariate statistics
# 2) Model-based selection 
# 3) iterative selection

# Univariate Statistics--------------------------------------------------------
"""
In univariate statistics, we compute whether there is a statistically significant relationship
between each feature and the target. Then the features that are related with the
highest confidence are selected. In the case of classification, this is also known as
analysis of variance (ANOVA). A key property of these tests is that they are univariate,
meaning that they only consider each feature individually.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()

# get deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

# use f_classif (the default) and SelectPercentile to select 50% of features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)

# transform training set
X_train_selected = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

"""
As you can see, the number of features was reduced from 80 to 40 (50 percent of the
original number of features). We can find out which features have been selected using
the get_support method, which returns a Boolean mask of the selected features
"""
mask = select.get_support()
print(mask)

# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

from sklearn.linear_model import LogisticRegression
# transform test data
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(
lr.score(X_test_selected, y_test)))

# Model-Based Feature Selection------------------------------------------------
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),threshold="median")
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))

mask = select.get_support()
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("Test score: {:.3f}".format(score))

# Iterative Feature Selection--------------------------------------------------
# Recursive feature elimination (RFE)
"""
which starts with all features,
builds a model, and discards the least important feature according to the
model. Then a new model is built using all but the discarded feature, and so on until
only a prespecified number of features are left. For this to work, the model used for
selection needs to provide some way to determine feature importance, as was the case
for the model-based selection. Here, we use the same random forest model that we
used earlier, and get the results
"""
from sklearn.feature_selection import RFE
select=RFE(RandomForestClassifier(n_estimators=100,random_state=42),n_features_to_select=40)

select.fit(X_train,y_train)

# visualize the selected features:
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

X_train_rfe= select.transform(X_train)
X_test_rfe= select.transform(X_test)
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Test score: {:.3f}".format(score))
print("Test score: {:.3f}".format(select.score(X_test, y_test)))