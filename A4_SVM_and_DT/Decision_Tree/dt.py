import pandas as pd
import numpy as np
import sys

train_file = sys.argv[1]
validation_file = sys.argv[2]
test_file = sys.argv[3]
validation_output_file = sys.argv[4]
test_output_file = sys.argv[5]

df = pd.read_csv(train_file)
df_val = pd.read_csv(validation_file)
df_test = pd.read_csv(test_file)

    
xtrain = np.array(df.iloc[:, :len(df.columns)-1])
ytrain = np.array(df.iloc[:,len(df.columns)-1])
ytrain = ytrain.reshape(ytrain.shape[0], 1)

xval = np.array(df_val.iloc[:, :len(df_val.columns)-1])
yval = np.array(df_val.iloc[:,len(df_val.columns)-1])

xtest = np.array(df_test.iloc[:, :len(df_test.columns)-1])

def get_gini_index(split_set, classes):
    num_examples = float(sum([len(x) for x in split_set]))
    gini = 0.0
    for split_dataset in split_set:
        size = float(len(split_dataset))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in split_dataset].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / num_examples)
    return gini

def test_split(index, value, dataset):
    categorical_vars = [1,3,5,6,7,8,9,13]
    left, right = list(), list()
    if index not in categorical_vars:
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
    else:
        for row in dataset:
            if row[index] == value:
                left.append(row)
            else:
                right.append(row)
    return left, right

def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    columns = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    for index in columns:
        dts_np = np.array(dataset)
        #print (index, end = ' ')
        if index == 0:
            unique_val = range(np.min(dts_np.T[0]) , np.max(dts_np.T[0]))
        elif index == 2:
            unique_val = range(0, np.max(dts_np.T[2]) - np.min(dts_np.T[2]), 100000)
        else:
            unique_val = list(set(dts_np.T[index]))
                    
        for val in unique_val:
            groups = test_split(index, val, dataset)
            gini = get_gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, val , gini, groups
    return {'index':best_index, 'value':best_value, 'groups':best_groups}

def convert_to_terminal_node(dataf):
    outcomes = [row[-1] for row in dataf]
    return max(set(outcomes), key=outcomes.count)

def split_node(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = convert_to_terminal_node(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = convert_to_terminal_node(left), convert_to_terminal_node(right)
        return
    
    if len(left) <= min_size:
        node['left'] = convert_to_terminal_node(left)
    else:
        node['left'] = get_best_split(left)
        split_node(node['left'], max_depth, min_size, depth+1)

    if len(right) <= min_size:
        node['right'] = convert_to_terminal_node(right)
    else:
        node['right'] = get_best_split(right)
        split_node(node['right'], max_depth, min_size, depth+1)
 
def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split_node(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']



max_depth = 9; min_size = 500
dataset = np.append(xtrain, ytrain, axis = 1)

tree = build_tree(dataset, max_depth, min_size)
with open(validation_output_file, 'w') as f:
    for row_index in range(0, len(xval)):
        prediction = predict(tree, xval[row_index])    
        f.write(str(prediction) + '\n')
        
with open(test_output_file, 'w') as f:
    for r_index in range(0, len(xtest)):
        prediction = predict(tree, xtest[r_index])    
        f.write(str(prediction) + '\n')

















