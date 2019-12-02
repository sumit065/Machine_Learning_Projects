import pandas as pd
import numpy as np
import sys
from sklearn import linear_model

def LinearRegression(train_data_path, test_data_path,outfile,weightfile):
    df = pd.read_csv(train_data_path, header = None) 
    
    n = len(df)
    m = len(df.columns)
    X = np.array(df.drop(columns = [m-1]))
    Y = np.array([df[m-1]]).T 
    
    ones_col = np.ones([n,1]) 
    X = np.append(X,ones_col,axis=1)
   
    W = np.dot(np.linalg.inv(np.dot(X.T, X)),np.dot(X.T, Y))
    
    df_test = pd.read_csv(test_data_path, header = None)
    
    X_test = np.array(df_test)
    X_test = np.append(X_test, np.ones([len(df_test),1]), axis = 1)
    Y_predicted = np.dot(X_test,W)
    
    with open(outfile, 'w') as f:
        for item in Y_predicted:
            f.write("%s\n" % item[0])
        
    with open(weightfile, 'w') as f:
    #write intercept in first line
        f.write("%s\n" % W[-1][0])
        for i in range (0,len(W)-1):
            f.write("%s\n" % W[i][0])
        

def RidgeRegression(train_data_path, test_data_path, regularization, outfile, weightfile):
    df = pd.read_csv(train_data_path, header = None) 
    
    n = len(df)
    m = len(df.columns)
    X = np.array(df.drop(columns = [m-1]))
    Y = np.array([df[m-1]]).T 
    
    ones_col = np.ones([n,1]) 
    X = np.append(X,ones_col,axis=1)

    df_test = pd.read_csv(test_data_path, header = None)
    X_test = np.array(df_test)
    X_test = np.append(X_test,np.ones([len(df_test),1]), axis = 1)	
        
    validation_error_per_lambda = []
    folds = []
    j =0; k = 10
    for _ in range (0,k): #runs k times
        folds.append(j)
        j = j + n/k
    folds.append(n-1) #append the last element
    folds = [int(i) for i in folds]
    
    lambda_val = []
    with open(regularization, 'r') as f:
        for line in f:
            line = line.strip()
            if(len(line) != 0):
                lambda_val.append(float(line))
            

    for l in lambda_val:
        errors = np.empty(k)
        for i in range(0,k):        
            train_features = np.append(X[0:folds[i]],X[folds[i+1]:], axis=0)
            validation_features = X[folds[i]:folds[i+1]]
            train_target = np.append(Y[0:folds[i]],Y[folds[i+1]:], axis=0)
            validation_target = Y[folds[i]:folds[i+1]]
            W_ridge = np.dot(np.linalg.inv(np.dot(train_features.T, train_features) + np.dot(l,np.identity(m))),np.dot(train_features.T, train_target))
            validation_set_prediction = np.dot(validation_features, W_ridge)
            errors[i] = ((np.linalg.norm(validation_set_prediction - validation_target))**2)/(2*len(validation_target))
        validation_error_per_lambda.append(np.mean(errors))
    
    min_lmda = lambda_val[np.argmin(validation_error_per_lambda)]
    W_ridge_optimum = np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(min_lmda,np.identity(m))),np.dot(X.T, Y))
    Y_predicted = np.dot(X_test,W_ridge_optimum)
    
    with open(outfile, 'w') as f:
        for item in Y_predicted:
            f.write("%s\n" % item[0])
        
    with open(weightfile, 'w') as f:
        #write intercept in first line
        f.write("%s\n" % W_ridge_optimum[-1][0])
        for i in range (0,len(W_ridge_optimum)-1):
            f.write("%s\n" % W_ridge_optimum[i][0])
    
    print (min_lmda)   
    
def add_feature(dtf, k,m):
    for i in range (0,m):
        for j in range (2,k+1):
            dtf[len(dtf.columns)] = dtf[i]**j
    
def LassoRegression(train_data_path, test_data_path, outfile):
    df = pd.read_csv(train_data_path, header = None)  
    n = len(df)
    Y = np.array([df[len(df.columns)-1]]).T   #transpose of X is X.T
    df = df.drop(columns = [len(df.columns)-1])
    m  = len(df.columns)
    
    add_feature(df,2,m)
    X = np.array(df)
    ones_col = np.ones([n,1]) 
    X = np.append(X,ones_col,axis=1)
    
    clf = linear_model.LassoLars(alpha=0.3, fit_intercept=False, max_iter=2000)
    clf.fit(X, Y)
    W = clf.coef_
    W.shape = [len(W),1]
    
    df_test = pd.read_csv(test_data_path, header = None)
    add_feature(df_test,2,m)
    X_test = np.array(df_test)
    X_test = np.append(X_test,np.ones([len(df_test),1]), axis = 1)
    
    Y_pred = np.dot(X_test,W)
    Y_pred.shape = [1,len(Y_pred)]
    Y_pred[0] = np.array([x if x >= 0 else 0 for x in Y_pred[0]])
    
    with open(outfile, 'w') as f:
        for item in Y_pred[0]:
            f.write("%s\n" % item)
        
    
if sys.argv[1] == 'a':
    LinearRegression(sys.argv[2], sys.argv[3],sys.argv[4],sys.argv[5])
elif sys.argv[1] == 'b':
    RidgeRegression(sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5], sys.argv[6])
elif sys.argv[1] == 'c':
    LassoRegression(sys.argv[2], sys.argv[3], sys.argv[4])
    










