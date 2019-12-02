import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid
import sys

df = pd.read_csv(sys.argv[1] , header = None)

params_arr = []
with open(sys.argv[2], 'r') as f:
    for line in f:
            line = line.strip()
            params_arr.append(line)
            
def init_params(architecture):
    layer_sizes = architecture['layer_sizes']
    weights = {}
    
    for i in range(1, len(layer_sizes)):
        w = np.zeros([layer_sizes[i-1]+1, layer_sizes[i]])
        weights[i] = w  #w.shape
    return weights

def bce_loss(y_pred, y_target):
    ones = np.ones(y_pred.shape)
    p1 = y_target*np.log(y_pred)
    p2 = (ones - y_target)*np.log(ones - y_pred)
    return -(np.sum(p1+p2))/len(y_pred)

def softmax(x):
    z = np.exp(x - np.max(x))
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

def fwd_prop(weights, x_train):  #weights dict
    activations_dict = {}
    n = len(x_train)
    ones_col = np.ones([n,1])
    
    z = x_train
    for i in range(1,len(weights)+1): #for four layers including o/p it run four times
        x_prime = np.append(ones_col,z,axis=1)
        if i != len(weights):
            z = sigmoid(np.dot(x_prime,weights[i])) #z is n*k
        else:
            z = softmax(np.dot(x_prime,weights[i]))
        activations_dict[i] = z
    return activations_dict

def get_dL_ds(weights_dict, activations_dict, y_target):
    train_size = len(y_target)
    ds_dict = {}   #size is the number of hidden layers
    i = len(activations_dict)   #weights_dict and activations_dict have the same size
                                #activations_dict is one size larger than ds_dict
    y_pred = activations_dict[i]  #n*k ;  ds : n*kl
        
    stored_ds = (y_pred-y_target)/float(train_size)
    i = i-1
       
    while i >= 1:
        z_cur = activations_dict[i]
        w = ((weights_dict[i+1])[1:]).T   #now w is k[i+1]*k[i]
        ds_dict[i] = (np.dot(stored_ds, w))*z_cur*(1-z_cur)
        stored_ds = ds_dict[i]
        i = i-1
    return ds_dict
    
def backprop_wt_update(ds_dict, activations_dict, weights_dict, y_pred, y_train, x_train, eta):
    #wt update for op layers
    train_size = float(len(x_train))
        
    i = len(weights_dict)

    #wt update for all hidden layers
    while i>1:
        if i == len(weights_dict):
            cur_ds = (y_pred - y_train)/train_size
        else:
            cur_ds = ds_dict[i]
            
        z_prev_layer = np.append(np.ones([len(activations_dict[i-1]),1]),activations_dict[i-1], axis =1) #n*k[i-1]        
        delta_w = np.dot(z_prev_layer.T,cur_ds)
        weights_dict[i] = weights_dict[i] - delta_w*eta
        
        i = i- 1
    
    #wt update for first hidden layer
    #x_train does not have has ones col added to it
    z_prev_layer = np.append(np.ones([len(x_train),1]), x_train,axis =1)
    cur_ds = ds_dict[i]
    delta_w = np.dot(z_prev_layer.T,cur_ds)
    weights_dict[i] = weights_dict[i] - delta_w*eta
    
    return weights_dict
    
def mini_batch_gd(x_train, y_train, params_arr):
    
    n = len(x_train); m = len(x_train[0]); t = len(y_train[0])
    
    hidden_layer = list(np.array((params_arr[-1]).strip().split(" "), dtype = int))
    architecture = {
        "train_size": n,
        "features": m,
        "layer_sizes": [m] + hidden_layer + [t],  #1st is input layer, last is hidden layer
        "no_of_hidden_layers": len(hidden_layer)
    }
    
    batch_size = int(params_arr[-2])
    max_iter = int(params_arr[-3])
   
    w = init_params(architecture)
            
    if params_arr[0] == '1':
            eta = float(params_arr[1])
    if params_arr[0] == '2':
            seed_eta = float(params_arr[1])
    
    iterations = 0
    iter_c = 0
    break_loop = False
    for it in range(0, max_iter):
        l = 0; j = batch_size
        if break_loop:
            break
        while l < len(x_train):            
            if params_arr[0] == '2':
                eta = seed_eta/np.sqrt(iter_c +1)
           
            activations_dict = fwd_prop(w,x_train[l:j])
            y_pred = activations_dict[len(activations_dict)]
            deltas_dict = get_dL_ds(w, activations_dict, y_train[l:j])
            w = backprop_wt_update(deltas_dict, activations_dict, w,y_pred, y_train[l:j], x_train[l:j], eta )
                
            l = l+batch_size 
            j = j+batch_size
            iter_c += 1
            
            if iter_c == max_iter:
                with open(sys.argv[3], 'w') as f: 
                    for item in w:
                        mat = w[item]
                        for row in mat:
                            for i in row:
                                f.write("%s\n" % i)
                break_loop = True
                break
                      
            if len(y_pred) == 0:
                break
    print (iter_c)
    return w    
       


X_train = np.array(df.drop(columns = [len(df.columns)-1]))
Y_train = np.array( pd.get_dummies(df[len(df.columns)-1]) )

mini_batch_gd(X_train, Y_train, params_arr)