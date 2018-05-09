import pandas as pd
from sklearn.tree import DecisionTreeRegressor,export_graphviz
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('bikes.csv')
dataset.head()
Y = dataset['count']
dates  = dataset['date']
dataset =dataset.drop(['count','date'],axis=1)
dataset.shape

def MSE(predicted,Y):
    s=0
    for i in range(len(predicted)):
        s+=(predicted[i]-Y[i])**2
    s=s/len(predicted)
    return s
depth = []
maxnodes = list(range(10,101,10))
for i in range(3,10):
    clf = DecisionTreeRegressor(max_depth=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=clf, X=dataset, y=Y, cv=10, n_jobs=4)
    depth.append(scores.mean())

# print(depth)
print(depth)

maxdepth = list(range(3,10))

plt.plot(maxdepth,depth)
plt.xlabel('Hyperparameter Maxdepth')
plt.ylabel('Mean Sqaure Error of 10cv(MSE)')
plt.title('Calculating Maxdepth hyperparameter with Least MSE ')
print('Hence, maxdepth is either 3 or 5')

depth = []
maxnodes = list(range(10,101,10))
for i in maxnodes:
    clf = DecisionTreeRegressor(max_depth=5,max_leaf_nodes=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=clf, X=dataset, y=Y, cv=10, n_jobs=4)
    depth.append(scores.mean())



plt.plot(maxnodes,depth)
plt.xlabel('Hyperparameter Max Leaf nodes')
plt.ylabel('Mean Sqaure Error of 10cv(MSE)')
plt.title('Calculating Maxdepth hyperparameter with Least MSE for Max Depth 5 ')
print('Hence minimum value of MSE for Max Leaf nodes is obtained at 30')
MSEfor5 = min(depth)

model = DecisionTreeRegressor(max_depth=5,max_leaf_nodes=30)
# Perform 7-fold cross validation 
model.fit(dataset,Y)
predicted = model.predict(dataset)
MSE5 = MSE(predicted,Y)

depth = []
maxnodes = list(range(10,101,10))
for i in maxnodes:
    clf = DecisionTreeRegressor(max_depth=3,max_leaf_nodes=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=clf, X=dataset, y=Y, cv=10, n_jobs=4)
    depth.append(scores.mean())


model = DecisionTreeRegressor(max_depth=3,max_leaf_nodes=20)
# Perform 7-fold cross validation 
model.fit(dataset,Y)
predicted = model.predict(dataset)
MSE3 = MSE(predicted,Y)

print('Minimum MSE for Max Depth 5 and MaxLeaf Node 30 is '+str(MSE5))
print('Minimum MSE for Max Depth 3 and MaxLeafNode 20 is '+str(MSE3))

model = DecisionTreeRegressor(max_depth=5,max_leaf_nodes=30)
# Perform 7-fold cross validation 
model.fit(dataset,Y)

n_nodes = model.tree_.node_count
children_left = model.tree_.children_left
children_right = model.tree_.children_right
feature = model.tree_.feature
threshold = model.tree_.threshold


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print('The number of leaf nodes are '+str(sum(is_leaves)))


predicted = model.predict(dataset)
print("The MSE value of part(i) is")
print(MSE(predicted,Y))

from graphviz import Source
from IPython.display import SVG
graph = Source(export_graphviz(model, out_file=None, feature_names=dataset.columns))
SVG(graph.pipe(format='svg'))

x= model.feature_importances_
def  imp(clf,feature_names):
    featureImp = clf.feature_importances_
    num = 0
    for i in range(len(featureImp)):
        print (feature_names[i], " : " ,featureImp[i])
        if (featureImp[i] > 0):
            num = num + 1
    print (" The more the value of a feature, more important it is.")
    print (" The number of importance feature is ",num)
imp(model,dataset.columns)

for i in range(len(dataset['month'])):
    if dataset['month'][i]==1 or dataset['month'][i]==2:
        dataset['month'][i] = 1
    elif dataset['month'][i]==3 or dataset['month'][i]==4 or dataset['month'][i]==11 or dataset['month'][i]==12:
        dataset['month'][i]=3
    else:
        dataset['month'][i]=2

depth = []
maxnodes = list(range(10,101,10))
for i in range(3,10):
    clf = DecisionTreeRegressor(max_depth=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=clf, X=dataset, y=Y, cv=10, n_jobs=4)
    depth.append(scores.mean())

# print(depth)
print(depth)

maxdepth = list(range(3,10))

plt.plot(maxdepth,depth)
plt.xlabel('Hyperparameter Maxdepth')
plt.ylabel('Mean Sqaure Error of 10cv(MSE)')
plt.title('Calculating Maxdepth hyperparameter with Least MSE ')
print('Hence, maxdepth is either 3 or 5')

depth = []
maxnodes = list(range(10,101,10))
for i in maxnodes:
    clf = DecisionTreeRegressor(max_depth=5,max_leaf_nodes=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=clf, X=dataset, y=Y, cv=10, n_jobs=4)
    depth.append(scores.mean())

plt.plot(maxnodes,depth)
plt.xlabel('Hyperparameter Max Leaf nodes')
plt.ylabel('Mean Sqaure Error of 10cv(MSE)')
plt.title('Calculating Maxdepth hyperparameter with Least MSE for Max Depth 5 ')
print('Hence minimum value of MSE for Max Leaf nodes is obtained at 30')


depth = []
maxnodes = list(range(10,101,10))
for i in maxnodes:
    clf = DecisionTreeRegressor(max_depth=3,max_leaf_nodes=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=clf, X=dataset, y=Y, cv=10, n_jobs=4)
    depth.append(scores.mean())


plt.plot(maxnodes,depth)
plt.xlabel('Hyperparameter Max Leaf nodes')
plt.ylabel('Mean Sqaure Error of 10cv(MSE)')
plt.title('Calculating Maxdepth hyperparameter with Least MSE for Max Depth 3 ')

model = DecisionTreeRegressor(max_depth=5,max_leaf_nodes=20)
# Perform 7-fold cross validation 
model.fit(dataset,Y)
predicted = model.predict(dataset)
MSE5 = MSE(predicted,Y)
model = DecisionTreeRegressor(max_depth=3,max_leaf_nodes=20)
model.fit(dataset,Y)
predicted=model.predict(dataset)
MSE3 = MSE(predicted,Y)

print('MSE value for maxdepth 3 and max_leaf_node 20 is '+str(MSE3))
print('MSE value for maxdepth 5 and max_leaf_node 20 is '+str(MSE5))

print('Hence maxdepth is 5 and max_leaf_node is 20')
print('MSE is '+str(MSE5))

print('Previous MSE was 376820.1402192555 and now '+str(MSE5))
print('difference = '+str(376820.1402192555 - MSE5))
print('MSE value was lesser before, it didn''t improve')