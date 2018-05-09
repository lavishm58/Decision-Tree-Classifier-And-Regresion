import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

trainX = pd.read_csv('P1_data/trainX.csv',header=None)
trainY = pd.read_csv('P1_data/trainY.csv',header=None)

testX = pd.read_csv('P1_data/testX.csv',header=None)
testY = pd.read_csv('P1_data/testY.csv',header=None)


print('Shape of training dataset '+ str(trainX.shape))
print('Shape of training label dataset '+ str(trainY.shape))
print('Shape of test dataset '+ str(testX.shape))


def MSE(predicted,Y):
    s=0
    for i in range(len(predicted)):
        s+=(predicted[i]-Y[i])**2
    s=s/len(predicted)
    return s
depth = []
maxnodes = list(range(10,101,10))
validationX = trainX[0:int(0.3*len(trainX))]
validationY = trainY[0:int(0.3*len(trainY))]
validationY = validationY.values.tolist()

trainnewX = trainX[int(0.3*len(trainX)):]
trainnewY = trainY[int(0.3*len(trainY)):]
for i in range(3,10):
    clf = DecisionTreeClassifier(max_depth=i)
# Perform 7-fold cross validation 
    clf.fit(trainnewX,trainnewY)
    predicted = clf.predict(validationX)
    depth.append(MSE(predicted,validationY))
# print(depth)


maxdepth = list(range(3,10))

plt.plot(maxdepth,depth)
plt.xlabel('Hyperparameter Maxdepth')
plt.ylabel('Mean Sqaure Error of 10cv(MSE)')
plt.title('Calculating Maxdepth hyperparameter with Least MSE ')
print('Hence, maxdepth is 7')	

depth = []
maxnodes = list(range(10,101,10))
for i in maxnodes:
    clf = DecisionTreeClassifier(max_depth=7,max_leaf_nodes=i)
# Perform 7-fold cross validation 
    clf.fit(trainnewX,trainnewY)
    predicted = clf.predict(validationX)
    depth.append(MSE(predicted,validationY))
# print(depth)	

maxdepth = list(range(3,10))

plt.plot(maxnodes,depth)
plt.xlabel('Hyperparameter Maxdepth')
plt.ylabel('Mean Sqaure Error of 10cv(MSE)')
plt.title('Calculating Maxdepth hyperparameter with Least MSE ')
print('Hence, max_leaf_nodes are 90')

model = DecisionTreeClassifier(max_depth=7,max_leaf_nodes=80)
model.fit(trainX,trainY)
print(model)	

print('                     Classification Report')
predicted = model.predict(testX)
print(metrics.classification_report(testY,predicted))
print('Confusion Matrix')
print(metrics.confusion_matrix(testY,predicted))

print('(b)Total no of nodes')
print(model.tree_.node_count)

n_nodes = model.tree_.node_count
children_left = model.tree_.children_left
children_right = model.tree_.children_right
feature = model.tree_.feature
threshold = model.tree_.threshold

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
print('(c)Total Number of leaf nodes')
print(sum(is_leaves))

from graphviz import Source
from IPython.display import SVG
graph = Source(export_graphviz(model, out_file=None, feature_names=trainX.columns))
SVG(graph.pipe(format='svg'))

def accuracy(predicted,trainY):
    s=0
    for i in range(len(trainY)):
        #print(predicted[i],trainY[i])
        if(predicted[i] == trainY[i]):
            s+=1
    return float(s)/float(len(trainY))


traininacc = []
testacc = []
datasetsize = list(range(1,11))
for i in range(1,11):
    trainnewX = trainX[0:int(i*0.1*len(trainX))]
    trainnewY = trainY[0:int(i*0.1*len(trainX))]
    model = DecisionTreeClassifier()
    model.fit(trainnewX,trainnewY)
    trainnewY = trainnewY.values.tolist()

    predicted = model.predict(trainnewX)

    traininacc.append(accuracy(predicted,trainnewY))
    predicted = model.predict(testX)
    testnewY = testY.values.tolist()
    testacc.append(accuracy(predicted,testnewY))


plt.plot(datasetsize,traininacc)
plt.plot(datasetsize,testacc)
plt.xlabel('Dataset Size')
plt.ylabel('Training/Test Accuracy')
plt.title('Accuracy vs Dataset S')
