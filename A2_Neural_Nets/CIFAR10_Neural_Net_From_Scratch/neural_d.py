import pandas as pd
import numpy as np
from skimage.filters import gabor
import sys
import time

start_time = time.time()

df = pd.read_csv(sys.argv[1] , header = None)
df_test = pd.read_csv(sys.argv[2], header =None)

params_arr = ['1','0.1' , '1000', '500','128 128' ]
            
def init_params(architecture):
    layer_sizes = architecture['layer_sizes']
    weights = {}
    np.random.seed(4)
    for i in range(1, len(layer_sizes)):
        normalization_k = np.sqrt(2)/np.sqrt(layer_sizes[i-1])
        w = np.random.standard_normal(size = (layer_sizes[i-1]+1, layer_sizes[i]))*normalization_k
        w[0] = np.zeros(layer_sizes[i])
        weights[i] = w  #w.shape
    return weights


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def softmax(x):
    z = np.max(x, axis=1)
    z = z[:, np.newaxis]
    return (np.exp(np.subtract(x, z).T) / np.sum(np.exp(np.subtract(x, z)), axis=1)).T
    
    
def relu(d):
    return d*(d>0)

def delta_relu_from_activation(d):
    return np.greater(d,0).astype(float)


def fwd_prop(weights, x_train):  #weights dict
    activations_dict = {}
    n = len(x_train)
    ones_col = np.ones([n,1])
    
    z = x_train
    for i in range(1,len(weights)+1): #for four layers including o/p it run four times
        x_prime = np.append(ones_col,z,axis=1)
        if i != len(weights):
            #z = sigmoid(np.dot(x_prime,weights[i])) #z is n*k
            z = relu(np.dot(x_prime,weights[i])) #z is n*k
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
        ds_dict[i] = (np.dot(stored_ds, w))*delta_relu_from_activation(z_cur)
        #ds_dict[i] = (np.dot(stored_ds, w))*z_cur*(1-z_cur)
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
    
    l = 0 
    j = batch_size
    
    for it in range(0, max_iter):
        if l >= len(x_train):
            l = 0; j = batch_size
            print (it)
        
        if time.time() - start_time > 580:
            break
        if params_arr[0] == '2':
            eta = seed_eta/np.sqrt(it +1)
       
        activations_dict = fwd_prop(w,x_train[l:j])
        y_pred = activations_dict[len(activations_dict)]
        deltas_dict = get_dL_ds(w, activations_dict, y_train[l:j])
        #print deltaz_dict
        w = backprop_wt_update(deltas_dict, activations_dict, w,y_pred, y_train[l:j], x_train[l:j], eta )
            
        l = l+batch_size 
        j = j+batch_size
    return w    
       

X_train = np.array(df.drop(columns = [len(df.columns)-1]))
Y_train = np.array( pd.get_dummies(df[len(df.columns)-1]))

image_arr_train = np.reshape(X_train, [len(X_train),32,32])

X_test =np.array(df_test.drop(columns = [len(df_test.columns)-1]))
image_arr_test = np.reshape(X_test, [len(X_test),32,32])

for i in range (0,len(image_arr_train)):
    image_arr_train[i] = gabor(image_arr_train[i], frequency=0.6)[0]
    
for i in range (0,len(image_arr_test)):
    image_arr_test[i] = gabor(image_arr_test[i], frequency=0.6)[0]
    
X_train = np.reshape(image_arr_train, [len(image_arr_train), 1024])
X_test = np.reshape(image_arr_test, [len(image_arr_test), 1024])


wt = mini_batch_gd(X_train, Y_train, params_arr)
ac_dict = fwd_prop(wt, X_test)
Y_pred = ac_dict[len(ac_dict)]
Y_pred = np.argmax(Y_pred, axis =1) 

with open(sys.argv[3], 'w') as f: 
    for item in Y_pred:
         f.write("%s\n" % str(item))
