#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[307]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[308]:


testdf = pd.read_excel ("D://test.xlsx")


# In[309]:


testdf


# In[310]:


testdf.head()


# In[311]:


testdf=testdf.dropna()


# In[312]:


testdf[["class"]]=testdf[["class"]].apply(lambda col:pd.Categorical(col).codes)


# In[313]:


testdf


# In[314]:


traindf = pd.read_excel ("D://train.xlsx")


# In[315]:


traindf.head()


# In[316]:


traindf=traindf.dropna()


# In[317]:


traindf[["class"]]=traindf[["class"]].apply(lambda col:pd.Categorical(col).codes)


# In[318]:


traindf.head(40)


# # implementing perceptron

# In[319]:


class Perceptron:
    def __init__(self, max_iters=20):
        self.max_iters = max_iters
    
    def fit(self, X, y):
        # Bookkeeping.
        X, y = np.asarray(X), np.asarray(y)
        iters = 20
        
        # Insert a bias column.
        X = np.concatenate((X, np.asarray([[1] * X.shape[0]]).T), axis=1)
        
        # Initialize random weights.
        ω = np.random.random(X.shape[1])        
        
        # Train as many rounds as allotted, or until fully converged.
        
        for _ in range(self.max_iters):
            y_pred_all = []
            for idx in range(X.shape[0]):
                traindf, testdf = X[idx], y[idx]
                y_pred = int(np.sum(ω * traindf) >= 0.5)
                if y_pred == testdf:
                    pass
                elif y_pred == 0 and testdf == 1:
                    ω = ω + traindf
                elif y_pred == 1 and testdf == 0:
                    ω = ω - traindf
                
                y_pred_all.append(y_pred)
            
            iters += 1
            if np.equal(np.array(y_pred_all), y).all():
                break
                
        self.iters, self.ω = iters, ω
        
    def predict(self, X):
        # Inject the bias column.
        X = np.asarray(X)
        X = np.concatenate((X, np.asarray([[1] * X.shape[0]]).T), axis=1)
        
        return (X @ self.ω > 0.5).astype(int)


# In[320]:


clf = Perceptron()
clf.fit([[0], [1], [2]], [0, 0, 1])


# In[321]:


clf.iters


# In[322]:


clf.predict([[0], [1], [2]])


# In[323]:


clf = Perceptron()
clf.fit([[0], [1], [2]], [0, 1, 0])


# In[324]:


clf.iters


# In[325]:


clf.predict([[0], [1], [2]])


# In[326]:


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=20):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):

            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)


# In[327]:



if __name__ == "__main__":

    import matplotlib.pyplot as plt
 

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=20)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))


# In[328]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()


# In[329]:


perceptron = Perceptron()


# In[330]:


traindf.plot.scatter('f1', 'f2')


# In[331]:


plt.scatter('f1', 'f2', data=testdf);


# In[332]:


plt.scatter('f1', 'class', data=testdf);


# In[333]:


i=testdf['class'].head(29)


# In[334]:


j=traindf['class'].head(29)


# In[335]:


plt.scatter(i,j);


# In[336]:


l=traindf.iloc[20:50]


# In[337]:


l


# In[338]:


y_pred


# # node class declaration

# In[385]:


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
           # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value


# In[ ]:





# # tree class

# In[391]:


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
       
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            #self.print_tree
            self.print_tree(tree.right, indent + indent)
    #fitting
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


# In[ ]:





# # training and testing the model

# In[392]:


X = traindf.iloc[:, :-1].values
Y = traindf.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)


# # fitting the model

# In[393]:


classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()


# In[394]:


Y_pred = classifier.predict(X_test) 
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred,20)


# In[395]:


accuracy_score(y_pred, y_test, 20)


# # print(accuracy_score(Y_pred_test, Y_test))
# 

# In[396]:


#perceptron.fit(X_train, y_train, 20, 0.0001)


# In[397]:


###Y_pred_test = perceptron.predict(X_test)
#print(accuracy_score(Y_pred_test, Y_test))


# In[398]:


testdf_transposed = testdf.T


# In[399]:


testdf_transposed


# In[ ]:


traindf_transposed = traindf.T


# In[ ]:



traindf_transposed


# In[344]:


accuracy_score(y_pred, y_test, [0.01, 0.1, 1.0, 10.0, 100.0])


# In[ ]:




