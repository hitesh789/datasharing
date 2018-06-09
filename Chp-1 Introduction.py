# reset the environment
%reset -f
#%clear

# setting up your current working directory
%cd C:\Users\bama6012\Desktop\desk\Python My study
%pwd

# import a csv in pandas dataframe format
import pandas as pd
df=pd.read_csv('C:/Users/bama6012/Desktop/desk/Desktop/datasets/cars.csv',index_col=0)

# To convert a pandas dataframe into a numpy array
array_df=df.values
array_df[0,]

# --------------------Numpy
#Intro to Numpy
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))

# --------------------Scipy
"""
The most important part of SciPy for us is scipy.sparse: 
This provides sparse matrices, which are another representation that is used for data in scikitlearn.
Sparse matrices are used whenever we want to store a 2D array that contains
mostly zeros
"""
from scipy import sparse

#Create a 2D NumPy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print('eye:\n{}'.format(eye))

# Convert the NumPy array to a SciPy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix=sparse.csr_matrix(eye)
print('\nScipy sparse matrix:\n{}'.format(sparse_matrix))

"""
Usually it is not possible to create dense representations of sparse data (as they would
not fit into memory), so we need to create sparse representations directly. Here is a
way to create the same sparse matrix as before, using the COO format:
"""
data=np.ones(4)
row_indices=np.arange(4)
col_indices=np.arange(4)
eye_coo=sparse.coo_matrix((data,(row_indices,col_indices)))
print('\ncoo representation:\n{}'.format(eye_coo))

#--------------------Matplotlib
import matplotlib.pyplot as plt
#%matplotlib notebook
#%matplotlib inline
x=np.linspace(-10,10,100)
y=np.sin(x)
plt.plot(x,y,marker='x')

#-------------------Pandas
# out of the scope for this book
import pandas as pd

# create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
'Location' : ["New York", "Paris", "Berlin", "London"],
'Age' : [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)


"""
Throughout the book we make ample use of NumPy, matplotlib
and pandas. All the code will assume the following imports:
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# Versions Used in this Book
import sys
print("Python version: {}".format(sys.version))
import pandas as pd
print("pandas version: {}".format(pd.__version__))
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))
import numpy as np
print("NumPy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import IPython
print("IPython version: {}".format(IPython.__version__))
import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))

# A First Application: Classifying Iris Species--------------------------------------------------------------
"""
Meet the Data:
--------------
The data we will use for this example is the Iris dataset, a classical dataset in machine
learning and statistics. It is included in scikit-learn in the datasets module. We
can load it by calling the load_iris function.
"""
from sklearn.datasets import load_iris
iris_dataset=load_iris()
print("\nKeys of iris dataset:\n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print(iris_dataset['DESCR'])
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target: {}".format(iris_dataset['target']))
print("Target:\n{}".format(iris_dataset['target']))

#------ Measuring Success: Training and Testing Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

# use test_size=0.30 for getting 30% data as test data),in default train=0.75,test=0.25
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

#------ First Things First: Look at Your Data
"""to get a pair plot pandas had implemented it"""
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)

# create a scatter matrix from the dataframe, color by y_train
import mglearn
grr=pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(20,20),marker='o',\
                               hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

#------Building Your First Model: k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

#------Making Predictions
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction=knn.predict(X_new)
print('prediction : {}'.format(prediction))
print('predicted target name : {}'.format(iris_dataset['target_names'][prediction]))

#------Evaluating the Model
y_pred=knn.predict(X_test)
print('Test set predictions : {}'.format(y_pred))
print('Test set predictions names : {}'.format(iris_dataset['target_names'][y_pred]))

print('Test set score : {:.2f}'.format(np.mean(y_pred==y_test)))
"""We can also use the score method of the knn object, which will compute the test set accuracy for us"""
print('Test set score : {:.2f}'.format(knn.score(X_test,y_test)))
