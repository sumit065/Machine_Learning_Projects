import pandas as pd
import numpy as np
import sys

df = pd.read_csv(sys.argv[1], header = None)
df_test = pd.read_csv(sys.argv[2], header = None)

#constants
last_col = len(df.columns) - 1
num_labels = 10
c = 500
max_iter = 20
batch_size = 100
eta = 0.0001

#segregate data class-wise
labelwise_data = {}
for i in range(0, num_labels):
    labelwise_data[i] = np.array(df.loc[df[last_col] == i])
    

def binarySVCFit(class1_data, class2_data, label1, label2):
    #label1 < label2 always
    #prepare xtrain, ytrain
    x = np.append(class1_data, class2_data, axis = 0)
    for i in range(0, len(x)):
        x[i][784] = -1 if x[i][784] == label1 else 1

    np.random.shuffle(x)
        
    ytrain = x.T[-1]  #n*1 1-D array
    xtrain = (x.T[:-1].T)/255.0  #n*m
    del(x)
    m = xtrain.shape[1]
    n = xtrain.shape[0]
        
    w = np.zeros(m)
    b = 0
    
    for _ in range(0, max_iter): 
        bstart = 0
        bend = batch_size
        while bstart < n:
            ti = (ytrain[bstart:bend]*(np.dot(xtrain[bstart:bend], w) + b) <= 1).astype(int)  #bool b*1
            lamda = c/len(ti)
            
            tempw = lamda*np.dot(xtrain[bstart:bend].T , (ytrain[bstart:bend]*ti))
            tempb = lamda*np.sum(ytrain[bstart:bend]*ti)
        
            w = w - eta*(w - tempw)
            b = b + eta*tempb
            
            bstart += batch_size
            bend += batch_size
            bend = min(n,bend)
            
    return  w, b

#generate weights for every class pair
classPair_weights = {}
for i in range(0,num_labels):
    print(i)
    for j in range(i+1, num_labels):
        classPair_weights[(i,j)] = binarySVCFit(labelwise_data[i], labelwise_data[j], i , j)

        
def predict(xtest):
    yprd = np.zeros(len(xtest), dtype = 'int')
    for row in range(0, len(xtest)):
        votes = np.zeros(num_labels)
        for key in classPair_weights.keys():
            s = key[0]; l = key[1]
            w, b = classPair_weights[key]
            pred = s if (np.dot(w,xtest[row]) + b < 0) else l
            votes[pred] += 1
        yprd[row] = np.argmax(votes)
    return yprd

xtest = np.array(df_test.iloc[:, :-1])
ypred = predict(xtest)

with open(sys.argv[3], 'w') as f:
    for x in ypred:
        f.write(str(x) + '\n')














            