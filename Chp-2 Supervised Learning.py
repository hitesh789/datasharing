# reset the environment
%reset -f

%cd C:\Users\bama6012\Desktop\desk\Python My study\Py Codes-Introduction to Machine Learning Book

data='C:/Users/bama6012/Desktop/desk/Python My study/data/'

#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

# Some Sample Datasets-----------------------------------------------------------------------------------------
# generate dataset
X,y=mglearn.datasets.make_forge()

#plot dataset
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print('cancer.keys() : \n{}'.format(cancer.keys()))
print('shape of cancer data : {}'.format(cancer['data'].shape))
print('sample counts per class : \n{}'.format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))

from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))

X,y=mglearn.datasets.load_extended_boston()
print('X.shape : {}'.format(X.shape))

#--------------------------------------------------k-Nearest Neighbors
# Predictions made by the one-nearest-neighbor model on the forge dataset
mglearn.plots.plot_knn_classification(n_neighbors=1)
# Predictions made by the three-nearest-neighbors model on the forge dataset
mglearn.plots.plot_knn_classification(n_neighbors=3)

# Lets build a model--------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X,y=mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print('Test set predictions : {}'.format(clf.predict(X_test)))
print('Test set Accuracy : {:.2f}'.format(clf.score(X_test,y_test)))

# Analyzing KNeighborsClassifier-------------------------------------------------------------------------
# Decision boundaries created by the nearest neighbors model for different values
# of n_neighbors
fig,axes=plt.subplots(1,3,figsize=(10,3))
for n_neighbors,ax in zip([1,3,9],axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)

"""less neighbors more complex,more neighbors simple model"""

"""
Let’s investigate whether we can confirm the connection between model complexity
and generalization that we discussed earlier. We will do this on the real-world Breast
Cancer dataset. We begin by splitting the dataset into a training and a test set. Then
we evaluate training and test set performance with different numbers of neighbors.
"""
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,
                                               stratify=cancer.target,random_state=66)
training_accuracy=[]
test_accuracy=[]

# Try n_neighbors from 1 to 10
neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings,training_accuracy,label='training accuracy')
plt.plot(neighbors_settings,test_accuracy,label='test accuracy')
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

# -------------------------------------------------k-neighbors regression
mglearn.plots.plot_knn_regression(n_neighbors=1)
"""Again, we can use more than the single closest neighbor for regression. When using
multiple nearest neighbors, the prediction is the average, or mean, of the relevant
neighbors"""
mglearn.plots.plot_knn_regression(n_neighbors=3)

# let us build a knn regression model
from sklearn.neighbors import KNeighborsRegressor
X,y=mglearn.datasets.make_wave(n_samples=40)

# split the data into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# instantiate the model and set the number of neighbors to consider to 3
reg=KNeighborsRegressor(n_neighbors=3)

# fit the model using the training data and training targets
reg.fit(X_train,y_train)

print('test set predictions : \n{}'.format(reg.predict(X_test)))
"""
We can also evaluate the model using the score method, which for regressors returns
the R2 score. The R2 score, also known as the coefficient of determination, is a measure
of goodness of a prediction for a regression model, and yields a score between 0
and 1. A value of 1 corresponds to a perfect prediction, and a value of 0 corresponds
to a constant model that just predicts the mean of the training set responses, y_train
"""
print('Test set R^2 : {:.2f}'.format(reg.score(X_test,y_test)))

# Analyzing KNeighborsRegressor------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # make predictions using 1, 3, or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
    n_neighbors, reg.score(X_train, y_train),
    reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                    "Test data/target"], loc="best")

#------------------------------------------------------------------------------Linear Models
# Linear models for regression-------------------------------------------------
mglearn.plots.plot_linear_regression_wave()

# Linear regression (aka ordinary least squares)-------------------------------
from sklearn.linear_model import LinearRegression
X,y=mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
lr=LinearRegression().fit(X_train,y_train)
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

# Let’s look at the training set and test set performance
# R^2 value
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

#Lets build a Linear regression on Boston dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X,y=mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
Linreg = LinearRegression()
lr = Linreg.fit(X_train, y_train)
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# Ridge regression-------------------------------------------------------------
from sklearn.linear_model import Ridge
ridge=Ridge().fit(X_train,y_train)
print('Training set score : {}'.format(ridge.score(X_train,y_train)))
print('Test set score : {}'.format(ridge.score(X_test,y_test)))

"""
The Ridge model makes a trade-off between the simplicity of the model (near-zero
coefficients) and its performance on the training set. How much importance the
model places on simplicity versus training set performance can be specified by the
user, using the alpha parameter. In the previous example, we used the default parameter
alpha=1.0. There is no reason why this will give us the best trade-off, though.
The optimum setting of alpha depends on the particular dataset we are using.
Increasing alpha forces coefficients to move more toward zero, which decreases
training set performance but might help generalization. For example
"""
ridge10=Ridge(alpha=10).fit(X_train,y_train)
print('Training set score : {}'.format(ridge10.score(X_train,y_train)))
print('Test set score : {}'.format(ridge10.score(X_test,y_test)))

train_acc=[]
test_acc=[]
for i in range(1,11):
    ridge10=Ridge(alpha=i).fit(X_train,y_train)
    train_acc.append(ridge10.score(X_train,y_train))
    test_acc.append(ridge10.score(X_test,y_test))

plt.plot(neighbors_settings,train_acc,label='training accuracy')
plt.plot(neighbors_settings,test_acc,label='test_accuracy')
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

"""Decreasing alpha allows the coefficients to be less restricted, meaning we move right
in Figure 2-1. For very small values of alpha, coefficients are barely restricted at all,
and we end up with a model that resembles LinearRegression"""

# Comparing coefficient magnitudes for ridge regression with different values
# of alpha and linear regression
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

# Lasso Regression ------------------------------------------------------------
from sklearn.linear_model import Lasso
lasso=Lasso().fit(X_train,y_train)
print('Training set score : {}'.format(lasso.score(X_train,y_train)))
print('Test set score : {}'.format(lasso.score(X_test,y_test)))
print('Number of features used : {}'.format(np.sum(lasso.coef_!=0)))    

# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter
lasso001=Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train)
print('Training set score : {}'.format(lasso001.score(X_train,y_train)))
print('Test set score : {}'.format(lasso001.score(X_test,y_test)))
print('Number of features used : {}'.format(np.sum(lasso001.coef_!=0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

# Comparing coefficient magnitudes for lasso regression with different values
# of alpha and ridge regression
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")

#--------------------------------------------------------------classifiers
# Logistic regression-------------------Linear SVC

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X,y=mglearn.datasets.make_forge()

fig,axes=plt.subplots(1,2,figsize=(10,3))

for model , ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf=model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=False,eps=0.5,ax=ax,alpha=0.7)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
axes[0].legend


from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,
                                               stratify=cancer.target,random_state=42)
logreg=LogisticRegression().fit(X_train,y_train)
print('Training set score : {}'.format(logreg.score(X_train,y_train)))
print('Test set score : {}'.format(logreg.score(X_test,y_test)))

"""
For LogisticRegression and LinearSVC the trade-off parameter that determines the
strength of the regularization is called C, and higher values of C correspond to less regularization.
In other words, when you use a high value for the parameter C, Logis
ticRegression and LinearSVC try to fit the training set as best as possible, while with
low values of the parameter C, the models put more emphasis on finding a coefficient
vector (w) that is close to zero.
"""

"""
The default value of C=1 provides quite good performance, with 95% accuracy on
both the training and the test set. But as training and test set performance are very
close, it is likely that we are underfitting. Let’s try to increase C to fit a more flexible
model:
"""

"""
The C parameter tells the SVM optimization how much you want to avoid misclassifying each 
training example. For large values of C, the optimization will choose a smaller-margin hyperplane 
if that hyperplane does a better job of getting all the training points classified correctly. 
Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating 
hyperplane, even if that hyperplane misclassifies more points. For very tiny values of C, you should 
get misclassified examples, often even if your training data is linearly separable.
"""

"""
 When you are using Gaussian RBF kernel, 
your separating surface will be based on a combination of bell-shaped surfaces centered at each 
support vector. The width of each bell-shaped surface will be inversely proportional to γγ(gamma). 
If this width is smaller than the minimum pair-wise distance for your data, you essentially 
have overfitting. If this width is larger than the maximum pair-wise distance for your data, 
all your points fall into one class and you don't have good performance either.
So the optimal width should be somewhere between these two extremes.
"""

logreg100=LogisticRegression(C=100).fit(X_train,y_train)
print('Training set score : {}'.format(logreg100.score(X_train,y_train)))
print('Test set score : {}'.format(logreg100.score(X_test,y_test)))

"""
Using C=100 results in higher training set accuracy, and also a slightly increased test
set accuracy, confirming our intuition that a more complex model should perform
better.
We can also investigate what happens if we use an even more regularized model than
the default of C=1, by setting C=0.01:
"""

logreg001=LogisticRegression(C=0.01).fit(X_train,y_train)
print('Training set score : {}'.format(logreg001.score(X_train,y_train)))
print('Test set score : {}'.format(logreg001.score(X_test,y_test)))

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()

"""
As LogisticRegression applies an L2 regularization by default,
the result looks similar to that produced by Ridge
"""
"""
If we desire a more interpretable model, using L1 regularization might help, as it limits
the model to using only a few features. Here is the coefficient plot and classification
accuracies for L1 regularization
"""

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
    C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
    C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)
"""
As you can see, there are many parallels between linear models for binary classification
and linear models for regression. As in regression, the main difference between
the models is the penalty parameter, which influences the regularization and
whether the model will use all available features or select only a subset.
"""

# multiple classifier----------------------------------------------------------
# Linear models for multiclass classification----------------------------------
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1],y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])

linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)
"""
We see that the shape of the coef_ is (3, 2), meaning that each row of coef_ contains
the coefficient vector for one of the three classes and each column holds the
coefficient value for a specific feature (there are two in this dataset). The intercept_
is now a one-dimensional array, storing the intercepts for each class.
Let’s visualize the lines given by the three binary classifiers
"""

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.ylim(-10, 15)
    plt.xlim(-10, 8)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
    'Line class 2'], loc=(1.01, 0.3))

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.8)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
            'Line class 2'], loc=(1.01, 0.3))
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    
"""
Linear models are very fast to train, and also fast to predict. They scale to very large
datasets and work well with sparse data. If your data consists of hundreds of thousands
or millions of samples, you might want to investigate using the solver='sag'
option in LogisticRegression and Ridge, which can be faster than the default on
large datasets. Other options are the SGDClassifier class and the SGDRegressor
class, which implement even more scalable versions of the linear models described
here.
"""
##### Large values for alpha or small values for C mean simple models,more regularization

#------------------------------------------------Method Chaining--------------------------
# instantiate model and fit it in one line
from sklearn.datasets import load_breast_cancer
lbc=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(lbc.data,lbc.target,random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
"""Here, we used the return value of fit (which is self) to assign the trained model to
the variable logreg. This concatenation of method calls (here __init__ and then fit)
is known as method chaining."""
"""
Another common application of method chaining in
scikit-learn is to fit and predict in one line
"""
logreg=LogisticRegression()
ypred=logreg.fit(X_train,y_train).predict(X_test)

"""Finally, you can even do model instantiation, fitting, and predicting in one line"""
ypred=LogisticRegression().fit(X_train,y_train).predict(X_test)
"""
This very short variant is not ideal, though. A lot is happening in a single line, which
might make the code hard to read. Additionally, the fitted logistic regression model
isn’t stored in any variable, so we can’t inspect it or use it to predict on any other data.
"""
#-------------------------------------------------------------------------------------------
# Naive Bayes Classifiers------------------------------------------------------
import numpy as np
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])

y = np.array([0, 1, 0, 1])

counts={}
for label in np.unique(y):
    # iterate over each class
    # count (sum) entries of 1 per feature
    counts[label]=X[y==label].sum(axis=0)
print('Feature counts : {}'.format(counts))

# Decision Trees---------------------------------------------------------------
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/bin'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/bin/dot.exe'
mglearn.plots.plot_animal_tree()

from sklearn.tree import DecisionTreeClassifier

cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,
                                               stratify=cancer.target,random_state=42)
tree=DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
print('Accuracy on training set:{:.3f}'.format(tree.score(X_train,y_train)))
print('Accuracy on test set:{:.3f}'.format(tree.score(X_test,y_test)))

"""
Limiting the
depth of the tree decreases overfitting. This leads to a lower accuracy on the training
set, but an improvement on the test set
"""
tree=DecisionTreeClassifier(max_depth=4,random_state=0)
tree.fit(X_train,y_train)
print('Accuracy on training set:{:.3f}'.format(tree.score(X_train,y_train)))
print('Accuracy on test set:{:.3f}'.format(tree.score(X_test,y_test)))

"""
Analyzing decision trees
We can visualize the tree using the export_graphviz function from the tree module.
This writes a file in the .dot file format, which is a text file format for storing graphs.
We set an option to color the nodes to reflect the majority class in each node and pass
the class and features names so the tree can be properly labeled
"""
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
feature_names=cancer.feature_names, impurity=False, filled=True)

import graphviz

with open('tree.dot') as f:
    dot_graph=f.read()
graphviz.Source(dot_graph)

#Feature importance in trees
print('important features are : {}'.format(tree.feature_importances_))

def plot_feature_importances_cancer(model):
    n_features=cancer.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center',color='red')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    
plot_feature_importances_cancer(tree)

tree = mglearn.plots.plot_tree_not_monotone()

import pandas as pd
ram_prices=pd.read_csv(data+'ram_price.csv',index_col=0)

plt.semilogy(ram_prices.date,ram_prices.price)
plt.xlabel('year')
plt.ylabel('price in $/Mbyte')

# Decision tree regressor

from sklearn.tree import DecisionTreeRegressor
#use historical data to forecast prices after the year 2000
data_train=ram_prices[ram_prices['date']<2000]
data_test=ram_prices[ram_prices.date>=2000]

# predict prices based on date
X_train=data_train.date[:,np.newaxis]
# we use a log-transform to get a simpler relationship of data to target
y_train=np.log(data_train.price)

tree=DecisionTreeRegressor().fit(X_train,y_train)
linear_reg=LinearRegression().fit(X_train,y_train)

# predict on all data
X_all=ram_prices.date[:,np.newaxis]

pred_tree=tree.predict(X_all)
pred_lr=linear_reg.predict(X_all)

#undo log-transform
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()

# Analyzing Random Forests 

#Let’s apply a random forest consisting of five trees to the
#two_moons dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42)

forest=RandomForestClassifier(n_estimators=5,random_state=2)
forest.fit(X_train,y_train)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

plot_feature_importances_cancer(tree)
plot_feature_importances_cancer(forest)

#Gradient boosted regression trees (gradient boosting machines)---------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

"""
As the training set accuracy is 100%, we are likely to be overfitting. To reduce overfitting,
we could either apply stronger pre-pruning by limiting the maximum depth or
lower the learning rate
"""
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
plot_feature_importances_cancer(gbrt)

# Kernelized Support Vector Machines------------------------------------------------------------------------
#Linear models and nonlinear features

"""
low c value will increase the regularization
high value of gamma makes you fit a complex curve which may result in over fitting"""

X, y = make_blobs(centers=4, random_state=8)
y = y % 2

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

from sklearn.svm import LinearSVC
linear_svm=LinearSVC().fit(X,y)

mglearn.plots.plot_2d_separator(linear_svm,X)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')

# add the squared first feature
X_new = np.hstack([X, X[:, 1:] ** 2])
from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
# visualize in 3D
ax = Axes3D(figure, elev=-152, azim=-26)
# plot first all the points with y == 0, then all with y == 1
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# show linear decision boundary
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature0 ** 2")

ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

from sklearn.svm import SVC
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plot support vectors
sv = svm.support_vectors_
# class labels of support vectors are given by the sign of the dual coefficients
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#Tuning SVM parameters
"""The gamma parameter is the one shown in the formula given in the previous section,
which controls the width of the Gaussian kernel. It determines the scale of what it
means for points to be close together. The C parameter is a regularization parameter,
similar to that used in the linear models. It limits the importance of each point (or
more precisely, their dual_coef_)"""

"""
C is the cost of classification as correctly stated by Dima.

A large C gives you low bias and high variance. Low bias because you penalize the cost of missclasification a lot.
A small C gives you higher bias and lower variance.

Gamma is the parameter of a Gaussian Kernel (to handle non-linear classification). Check this points:

They are not linearly separable in 2D so you want to transform them to a higher dimension where they 
will be linearly sepparable. Imagine "raising" the green points, then you can sepparate them from 
the red points with a plane (hyperplane)

To "raise" the points you use the RBF kernel, gamma controls the shape of the "peaks" where you raise the 
points. A small gamma gives you a pointed bump in the higher dimensions, a large gamma gives you a softer, 
broader bump.

So a small gamma will give you low bias and high variance while a large gamma will give you higher bias and 
low variance.

You usually find the best C and Gamma hyper-parameters using Grid-Search.
"""
                                                                                
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
            ncol=4, loc=(.9, 1.2))

"""
Let’s apply the RBF kernel SVM to the Breast Cancer dataset. By default, C=1 and
gamma=1/n_features
"""
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
svc = SVC()
svc.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")

"""
From this plot we can determine that features in the Breast Cancer dataset are of
completely different orders of magnitude. This can be somewhat of a problem for
other models (like linear models), but it has devastating effects for the kernel SVM.
Let’s examine some ways to deal with this issue.
"""

# Preprocessing data for SVMs
"""
One way to resolve this problem is by rescaling each feature so that they are all
approximately on the same scale. A common rescaling method for kernel SVMs is to
scale the data such that all features are between 0 and 1. We will see how to do this
using the MinMaxScaler preprocessing method in Chapter 3, where we’ll give more
details. For now, let’s do this “by hand”
"""
# compute the minimum value per feature on the training set
min_on_training=X_train.min(axis=0)
# compute the range of each feature (max - min) on the training set
range_on_training = (X_train - min_on_training).max(axis=0)

# subtract the min, and divide by range
# afterward, min=0 and max=1 for each feature
X_train_scaled=(X_train-min_on_training)/range_on_training
print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
print("Maximum for each feature\n {}".format(X_train_scaled.max(axis=0)))

# use THE SAME transformation on the test set,
# using min and range of the training set (see Chapter 3 for details)
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC()
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
"""
Scaling the data made a huge difference! Now we are actually in an underfitting
regime, where training and test set performance are quite similar but less close to
100% accuracy. From here, we can try increasing either C or gamma to fit a more complex
model. For example:
""""
svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))    
"""Here, increasing C allows us to improve the model significantly, resulting in 97.2% accuracy"""

# Genetic Algorithms-----------------------------------------------------------
"""
1) Initialisation
2) Fitness Function
3) Selection
4) cross over
5) Mutation
"""
"""
The off-springs thus produced are again validated using our fitness function, and if considered fit then will replace the less fit chromosomes from the population.

But the question is how we will get to know that we have reached our best possible solution?

So basically there are different termination conditions, which are listed below:

There is no improvement in the population for over x iterations.
We have already predefined an absolute number of generation for our algorithm.
When our fitness function has reached a predefined value.
"""
# installing DEAP, update_checker and tqdm 
#!pip install deap update_checker tqdm

# installling TPOT
#!pip install TPOT

# Now lets implement ga using a dataset

# import basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deap,update_checker,tqdm

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# import train and test data
train=pd.read_csv(data+'Train_UWu5bXk.csv')
test=pd.read_csv(data+'Test_u94Q5KV.csv')

# pre processing---------------

# Mean imputations
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)
test['Item_Weight'].fillna((test['Item_Weight'].mean()), inplace=True)

### reducing fat content to only two categories
train['Item_Fat_Content']=train['Item_Fat_Content'].replace(['LF','low fat','reg'],['Low Fat','Low Fat','Regular'])
train['Item_Fat_Content'].value_counts()

test['Item_Fat_Content']=test['Item_Fat_Content'].replace(['LF','low fat','reg'],['Low Fat','Low Fat','Regular'])
test['Item_Fat_Content'].value_counts()

train['Outlet_Establishment_Year'].value_counts()
train['Outlet_Establishment_Year'] = 2013 - train['Outlet_Establishment_Year'] 
test['Outlet_Establishment_Year'] = 2013 - test['Outlet_Establishment_Year'] 

train['Outlet_Size'].value_counts(dropna=False)

train['Outlet_Size'].fillna('Small',inplace=True)
test['Outlet_Size'].fillna('Small',inplace=True)

train['Item_Visibility'].value_counts(dropna=False)
train['Item_Visibility'] = np.sqrt(train['Item_Visibility'])
test['Item_Visibility'] = np.sqrt(test['Item_Visibility'])

col = ['Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Fat_Content']
test['Item_Outlet_Sales'] = 0
combi = train.append(test)
number = preprocessing.LabelEncoder()
for i in col:
 combi[i] = number.fit_transform(combi[i].astype('str'))
 combi[i] = combi[i].astype('object')
train = combi[:train.shape[0]]
test = combi[train.shape[0]:]
test.drop('Item_Outlet_Sales',axis=1,inplace=True)

## removing id variables 
tpot_train = train.drop(['Outlet_Identifier','Item_Type','Item_Identifier'],axis=1)
tpot_test = test.drop(['Outlet_Identifier','Item_Type','Item_Identifier'],axis=1)
target = tpot_train['Item_Outlet_Sales']
tpot_train.drop('Item_Outlet_Sales',axis=1,inplace=True)

# finally building model using tpot library
from tpot import TPOTRegressor

X_train, X_test, y_train, y_test = train_test_split(tpot_train, target,train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export(data+'tpot_boston_pipeline.py')

## predicting using tpot optimised pipeline
tpot_pred = tpot.predict(tpot_test)
sub1 = pd.DataFrame(data=tpot_pred)

#sub1.index = np.arange(0, len(test)+1)
sub1 = sub1.rename(columns = {'0':'Item_Outlet_Sales'})
sub1['Item_Identifier'] = test['Item_Identifier']
sub1['Outlet_Identifier'] = test['Outlet_Identifier']
sub1.columns = ['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier']
sub1 = sub1[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
sub1.to_csv('tpot.csv',index=False)