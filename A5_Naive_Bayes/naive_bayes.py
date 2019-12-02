import pandas as pd
import numpy as np
import re
import sys

df = pd.read_csv(sys.argv[1])
df_test = pd.read_csv(sys.argv[2])

c_neg = 0; c_pos = 0
k = 2
n = len(df)

c_neg = 0; c_pos = 0
k = 2
n = len(df)

p = re.compile('\w+')
dict_pos = {}; dict_neg = {}
for i in range(0, len(df)):
    s = p.findall(df.loc[i, 'review'])
    if df.loc[i, 'sentiment'] == 'positive':
        c_pos += 1
        for key in s:
            key = key.lower()
            if key in dict_pos:
                dict_pos[key] += 1
            else:
                dict_pos[key] = 1            
    elif df.loc[i, 'sentiment'] == 'negative':
        c_neg += 1
        for key in s:
            key = key.lower()
            if key in dict_neg:
                dict_neg[key] += 1
            else:
                dict_neg[key] = 1
        
prob_cpos = c_pos/n
prob_cneg = c_neg/n

words_pos = 0; words_neg = 0
for x in dict_pos.values():
    words_pos += x
for r in dict_neg.values():
    words_neg += r
len_dict_pos = len(dict_pos)
len_dict_neg = len(dict_neg)


ypred = []
for i in range(0, len(df_test)):
    l = set(p.findall(df_test.iloc[i,0]))
    xpos = 0; xneg = 0
    for word in l:
        word = word.lower()
        if word in dict_pos:
            xpos += np.log((1+ dict_pos[word])/(len_dict_pos + words_pos+ len_dict_neg))
        else:
            xpos += np.log(1/(len_dict_pos + words_pos+ len_dict_neg))
        if word in dict_neg:
            xneg += np.log((1+ dict_neg[word])/(len_dict_neg + words_neg + len_dict_pos))
        else:
            xneg += np.log(1/(len_dict_neg + words_neg + len_dict_pos))
            
            
    xpos += np.log(prob_cpos)
    xneg += np.log(prob_cneg)
    if xpos > xneg:
        ypred.append(1)
    else:
        ypred.append(0)
    
with open(sys.argv[3], 'w') as f:
    for x in ypred:
        f.write(str(x) + '\n')
