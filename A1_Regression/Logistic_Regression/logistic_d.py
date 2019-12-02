import numpy as np
import pandas as pd
import time
import sys


start_time = time.time()

def predict(x_test, w): #w is  m x k ; x is n x m
    z = np.dot(x_test,w)
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

def cross_entropy_loss_gradient(y_pred, y_train, x_train):
    return np.dot(x_train.T, (y_pred - y_train))/(2*len(y_train))

def cross_entropy_loss(y_pred, y_train):
    return -0.5* (np.mean(np.sum( np.log(y_pred) * y_train, axis=1)))

def gradient_descent(x_train, y_train): 
    
    eta = 100
    max_iter = 1000000
    w = np.zeros([len(x_train[0]),len(y_train[0])])  # w is m x k
    loss_gradient = cross_entropy_loss_gradient(predict(x_train, w),y_train, x_train) 
   
    while itr < max_iter:
        if time.time() - start_time >= 120:
            break
        eta = seed_eta/np.sqrt(itr)

        w = w - np.dot(eta, loss_gradient)
        y_pred = predict(x_train, w)
        loss = cross_entropy_loss(y_pred, y_train)
        loss_gradient = cross_entropy_loss_gradient(y_pred,y_train, x_train)       
        itr = itr + 1
        
    return w

df_raw = pd.read_csv(sys.argv[1], header = None)
df_raw_test = pd.read_csv(sys.argv[2], header = None)

df = pd.get_dummies(df_raw.drop(columns = [len(df_raw.columns)-1]))
df_test = pd.get_dummies(df_raw_test)


df_label = pd.get_dummies(df_raw[len(df_raw.columns)-1])

X = np.array(df)
X = np.append(X,np.ones([len(X),1]),axis=1)

X_test = np.array(df_test)
X_test = np.append(X_test,np.ones([len(X_test),1]),axis=1)

Y = np.array(df_label)

W = gradient_descent(X,Y)

Y_pred = predict(X_test, W)
Y_pred = np.argmax(Y_pred, axis= 1)

#write output
label_arr = df_label.columns.tolist()
with open(sys.argv[3], 'w') as f:
     for item in Y_pred:
              f.write("%s\n" % label_arr[item])

with open(sys.argv[4], 'w') as f:
    # item = W[-1]
    # string = str(item[0]) + "," +str(item[2]) + ","+str(item[4]) + ","+str(item[1]) + ","+str(item[3])
    # f.write("%s\n" % string)
    for item in W:
        string = str(item[0]) + "," +str(item[2]) + ","+str(item[4]) + ","+str(item[1]) + ","+str(item[3])
        f.write("%s\n" % string)













