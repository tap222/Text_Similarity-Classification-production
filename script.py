import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.externals import joblib
from scipy import sparse
import gzip, re, string
import pickle

import warnings
warnings.filterwarnings('ignore')

# modelname=str(input())
modelname ='------'

root_dir = 'C:\\Users\\tamohant\\Desktop\\Auto_synthesis_Training_data\\' 

import os
if not os.path.exists(root_dir + modelname):
    os.makedirs(root_dir + modelname)

# Reading the file 
train = pd.read_excel('----.xlsx',delimiter=',',encoding='latin-1')
train['Description']=train['Short Description'].astype('str')

#Label columns
# train['Intent'] = train[['Level1', 'Level2','Level3']].agg('-'.join, axis=1).astype('category')
# train['Intent'] = train[['Level2','Level3']].agg('-'.join, axis=1).astype('category')

train['Intent Level2'] =  train['Intent Level2'].astype('str')
train['Intent Level3'] =  train['Intent Level3'].astype('str')

train.dropna(subset = ['Intent Level2', 'Intent Level3'])

train['Intent'] = train[['Intent Level2','Intent Level3']].agg('-'.join, axis=1).astype('category')
train['Intent']= train['Intent'].astype('category')
label_cols =  train['Intent'].cat.categories.tolist()

###Handling target values
train = pd.concat([train,pd.get_dummies(train['Intent'])],axis=1)
train.drop(['Intent'],axis=1, inplace=True)

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

#Logistic Regression
def get_mdl(y):
    y = y.values
    r = sparse.csr_matrix(np.log(pr(1,y) / pr(0,y)))
    m = LogisticRegression(C=4,solver='sag',dual=False,class_weight="balanced",n_jobs=-1)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


#Prediction

for name in label_cols:
    COMMENT = 'Description'
    train[COMMENT].fillna("unknown", inplace=True)
    re_tok = re.compile(f'([{string.punctuation}“”¨_«»®´·º½¾¿¡§£₤‘’])')
    def tokenize(s): return re_tok.sub(r' \1 ', s).split()
    vec = TfidfVectorizer(ngram_range=(1,5),
                          stop_words="english",
                          analyzer='char',
                          tokenizer=tokenize,
                          min_df=3, max_df=0.9, 
                          strip_accents='unicode',
                          use_idf=1,
                          smooth_idf=1, 
                          sublinear_tf=1)
    vec.fit(train[COMMENT])
    x = vec.transform(train[COMMENT])
    print('fit', name)
    m,r = get_mdl(train[name])
    model_loc = root_dir + modelname + '\\' + str(name).replace('/','or') + ".json.gz_model.pkl"
    joblib.dump(m, model_loc)
    r_loc = root_dir + modelname + '\\' +  str(name).replace('/','or') + ".json.gz_r.npz"
    sparse.save_npz(r_loc, r)
    vec_loc = root_dir + modelname + '\\' + str(name).replace('/','or') + ".json.gz_vector.pkl"
    joblib.dump(vec, vec_loc,protocol=pickle.HIGHEST_PROTOCOL) 