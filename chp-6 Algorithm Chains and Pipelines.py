import mglearn
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# load and split data
cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)

# compute min max scaling for the data
scaler=MinMaxScaler().fit(X_train)

# rescale the training data
X_train_scaled=scaler.transform(X_train)

# learn an SVM on the scaled training data
svm=SVC()
svm.fit(X_train_scaled,y_train)

# scale the test data and score the scaled data
X_test_scaled=scaler.transform(X_test)

print('test score : {}'.format(svm.score(X_test_scaled,y_test)))

#--------------------------------------------------------------------------Parameter Selection with Preprocessing
from sklearn.model_selection import GridSearchCV

# for illustration purposes only, don't use this code!
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid=GridSearchCV(SVC(),param_grid=param_grid,cv=5)
grid.fit(X_train_scaled,y_train)

print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best set score: {:.2f}".format(grid.score(X_test_scaled, y_test)))
print("Best parameters: ", grid.best_params_)

mglearn.plots.plot_improper_processing()

#------------------------------------------------------------------------------Building Pipelines
from sklearn.pipeline import Pipeline
pipe=Pipeline([('scaler',MinMaxScaler()),('svm',SVC())])
pipe.fit(X_train,y_train)

print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))

# Using Pipelines in Grid Searches---------------------------------------------
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))
grid_scores=pd.DataFrame(grid.grid_scores_)

mglearn.plots.plot_proper_processing()

# Convenient Pipeline Creation with make_pipeline------------------------------
from sklearn.pipeline import make_pipeline

# Standard Synatx
pipe_long=Pipeline([('scaler',MinMaxScaler()),('svm',SVC(C=100))])

# abbreviated syntax
pipe_short=make_pipeline(MinMaxScaler(),SVC(C=100))

"""
The pipeline objects pipe_long and pipe_short do exactly the same thing, but
pipe_short has steps that were automatically named. We can see the names of the
steps by looking at the steps attribute
"""

print('long pipe line steps :\n{}'.format(pipe_long.steps))
print('short pipe line steps :\n{}'.format(pipe_short.steps))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe=make_pipeline(StandardScaler(),PCA(n_components=2),StandardScaler())

print("Pipeline steps:\n{}".format(pipe.steps))

# Accessing Step Attributes
# fit the pipeline defined before to the cancer dataset
pipe.fit(cancer.data)

# extract the first two principal components from the "pca" step
components = pipe.named_steps["pca"].components_
print("components.shape: {}".format(components.shape))

# Accessing Attributes in a Grid-Searched Pipeline
from sklearn.linear_model import LogisticRegression
pipe=make_pipeline(StandardScaler(),LogisticRegression())
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best estimator:\n{}".format(grid.best_estimator_))

"""
This best_estimator_ in our case is a pipeline with two steps, standardscaler and
logisticregression. To access the logisticregression step, we can use the
named_steps attribute of the pipeline, as explained earlier:
"""
print("Logistic regression step:\n{}".format(grid.best_estimator_.named_steps["logisticregression"]))

print("Logistic regression coefficients:\n{}".format(grid.best_estimator_.named_steps["logisticregression"].coef_))

# Grid-Searching Preprocessing Steps and Model Parameters----------------------
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge,Lasso

boston=load_boston()
X_train, X_test, y_train, y_test=train_test_split(boston.data,boston.target,random_state=0)

from sklearn.preprocessing import PolynomialFeatures

pipe=make_pipeline(StandardScaler(),
                   PolynomialFeatures(),
                   Ridge())

param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1),vmin=0, cmap="viridis")
plt.xlabel("ridge__alpha")
plt.ylabel("polynomialfeatures__degree")
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])),
param_grid['polynomialfeatures__degree'])
plt.colorbar()

print("Best parameters: {}".format(grid.best_params_))
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))

# Let’s run a grid search without polynomial features for comparison
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Score without poly features: {:.2f}".format(grid.score(X_test, y_test)))

#-----------------------------------------------------------------------------Grid-Searching Which Model To Use

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

param_grid = [{'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
                              'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                              'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'classifier': [RandomForestClassifier(n_estimators=100)],
                              'preprocessing': [None], 
                              'classifier__max_features': [1, 2, 3]},
              {'classifier':[DecisionTreeClassifier()],
                               'preprocessing':[None]}]

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))
