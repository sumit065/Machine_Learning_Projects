import numpy as np
import pandas as pd
import sys

def predict(x_test, w): #w is  m x k ; x is n x m
    z = np.dot(x_test,w)
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

def cross_entropy_loss_gradient(y_pred, y_train, x_train):
    return np.dot(x_train.T, (y_pred - y_train))/(2*len(y_train))

def cross_entropy_loss(y_pred, y_train):
    return -0.5* (np.mean(np.sum( np.log(y_pred) * y_train, axis=1)))


def mini_batch_gd(x_train, y_train, params_arr):
    
    w = np.zeros([len(x_train[0]),len(y_train[0])])
    eta = 0.1 ; alpha = 0.3 ; beta = 0.5 ; seed_eta = 1; epsilon = 0.0000001
        
    batch_size = int(params_arr[-1])
    max_iter = int(params_arr[-2])
    
    if params_arr[0] == '1':
            eta = float(params_arr[1])
    if params_arr[0] == '2':
            seed_eta = float(params_arr[1])

    i = 0; j = batch_size-1
    
    y_pred = predict(x_train[i:j], w)
    loss_gradient = cross_entropy_loss_gradient(y_pred,y_train[i:j], x_train[i:j]) 
    total_loss = cross_entropy_loss(y_pred, y_train[i:j])
    prev_mean = total_loss
    
    iterations = 0
    for it in range(0, max_iter):
        i = 0; j = batch_size-1
        
        while i < len(x_train):
            
            if params_arr[0] == '2':
                eta = seed_eta/np.sqrt(itr)
            elif params_arr[0] == '3':
                #edit here
                eta = eta
            w = w - np.dot(eta, loss_gradient)

            i = i+batch_size 
            j = j+batch_size
          
            y_pred = predict(x_train[i:j], w)
            if len(y_pred) == 0:
                break
            loss = cross_entropy_loss(y_pred, y_train[i:j])
            total_loss += loss
            
            loss_gradient = cross_entropy_loss_gradient(y_pred,y_train[i:j], x_train[i:j])
            
        if prev_mean - total_loss/(len(x_train)/batch_size) < epsilon:
            break
        prev_mean = total_loss/(len(x_train)/batch_size)
        total_loss = 0
        iterations = it+1
    print (iterations)
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

arr = []
with open(sys.argv[3], 'r') as f:
    for line in f:
            line = line.strip()
            arr.append(line)

W = mini_batch_gd(X,Y, arr)

Y_pred = predict(X_test, W)
Y_pred = np.argmax(Y_pred, axis= 1)

#write output
label_arr = df_label.columns.tolist()
with open(sys.argv[4], 'w') as f:
     for item in Y_pred:
              f.write("%s\n" % label_arr[item])

with open(sys.argv[5], 'w') as f:
    # item = W[-1]
    # string = str(item[0]) + "," +str(item[2]) + ","+str(item[4]) + ","+str(item[1]) + ","+str(item[3])
    # f.write("%s\n" % string)
    for item in W:
        string = str(item[0]) + "," +str(item[2]) + ","+str(item[4]) + ","+str(item[1]) + ","+str(item[3])
        f.write("%s\n" % string)













