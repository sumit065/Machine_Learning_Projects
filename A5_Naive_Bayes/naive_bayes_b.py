#stemming leads to decrease in accuracy (85 to 80)
#lemmatization might be an option
#stopwords do increase the accuracy from 84 to 85

import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer 
import sys

df = pd.read_csv(sys.argv[1])
df_test = pd.read_csv(sys.argv[2])

c_neg = 0; c_pos = 0
k = 2
n = len(df)

stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
             'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them'
             , 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
             'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
             'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
             'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
             'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
             'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
             'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
             "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
             "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
             'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
             'wouldn', "wouldn't"}
             

ps = PorterStemmer()
p = re.compile('\w+')
#p = re.compile('\w+\'*[a-zA-Z]*')
dict_pos = {}; dict_neg = {}
for i in range(0, len(df)):
    s = p.findall(df.loc[i, 'review'])
    
    if i%1000 == 0:
         print(i)
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
                
                
                
#process stopwords
for word in stopwords:
    if word in dict_pos:
        del(dict_pos[word])
    if word in dict_neg:
        del(dict_neg[word])

  
#uncomment to enable stemming  
  
# K = list(dict_pos.keys())
# for w in K:
#     x = ps.stem(w)
#     if w != x:
#         val = dict_pos[w]
#         del(dict_pos[w])
#         if x in dict_pos:
#             dict_pos[x] += val
#         else:
#             dict_pos[x] = val
            
# K = list(dict_neg.keys())
# for w in K:
#     x = ps.stem(w)
#     if w != x:
#         val = dict_neg[w]
#         del(dict_neg[w])
#         if x in dict_neg:
#             dict_neg[x] += val
#         else:
#             dict_neg[x] = val
        
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
    if i%1000 == 0:
        print(i)
    l = set(p.findall(df_test.iloc[i,0])) - set(stopwords)
    xpos = 0; xneg = 0
    for word in l:
        word = word.lower()
        if word in dict_pos:
            xpos += np.log((1+ dict_pos[word])/(len_dict_pos + words_pos + len_dict_neg))
        else:
            xpos += np.log(1/(len_dict_pos + words_pos + len_dict_neg))
        if word in dict_neg:
            xneg += np.log((1+ dict_neg[word])/(len_dict_neg + words_neg + len_dict_pos))
        else:
            xneg += np.log(1/(len_dict_neg + words_neg+ len_dict_pos))
            
            
    xpos += np.log(prob_cpos)
    xneg += np.log(prob_cneg)
    if xpos > xneg:
        ypred.append(1)
    else:
        ypred.append(0)
    
with open(sys.argv[3], 'w') as f:
    for x in ypred:
        f.write(str(x) + '\n')
