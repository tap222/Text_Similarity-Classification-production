import pandas as pd
import logging
import json
import numpy as np
import re
import string
from sklearn.externals import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import custom_token
import pickle

root_dir = os.getcwd() 
modelname = input("Type model name : ")
filename = input("Type file name : ")

test=pd.read_excel(filename)
test['Description']=test['Ticket_Description'].astype('str')

##Filling the Missing vaues
COMMENT = 'Description'
test[COMMENT].fillna("unknown", inplace=True)

import glob
path = root_dir + '\\' +  modelname + '\\'
file = glob.glob(path +'/*.json.gz_model.pkl')


category_names =  []

for i in range(len(file)):
    basename = os.path.basename(file[i])
    category_names.append(basename.split('.json.gz_model.pkl')[0])

preds = np.zeros((len(test), len(category_names)))

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "custom_token"
        return super().find_class(module, name)
        
for index,name in enumerate(category_names):
    print('fit', name)
    model_loc = root_dir + '\\' + modelname + '\\' + str(name)+ ".json.gz_model.pkl"
    vec_loc = root_dir + '\\' + modelname + '\\' + str(name) + ".json.gz_vector.pkl"
    r_loc = root_dir + '\\' + modelname + '\\' +  str(name) + ".json.gz_r.npz"
    re_tok = re.compile(f'([{string.punctuation}“”¨_«»®´·º½¾¿¡§£₤‘’])')
    def tokenize(s): return re_tok.sub(r' \1 ', s).split()
    # unpickle my model
    with open(model_loc, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        estimator = unpickler.load()
        
    with open(vec_loc, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        vec = unpickler.load()
        
    r = sparse.load_npz(r_loc)
    review = vec.transform(test[COMMENT])
    preds[:,index] = estimator.predict_proba(review.multiply(r))[:,1]

#Submission File 
submission = pd.DataFrame(preds, columns = category_names)
submission['Confidence_level'] = submission[category_names].max(axis=1)
submission['Pred_Intent'] = submission[category_names].idxmax(axis=1)
submission[['Intent Level1', 'Intent Level2']] = submission.Pred_Intent.str.split("__",expand=True,)
submission = submission[['Confidence_level','Pred_Intent','Intent Level1','Intent Level2']]
test_pred = pd.concat([test,submission], axis=1, sort=False)
test_pred.to_excel(str(modelname) + '__' + 'output.xlsx', index=False)
