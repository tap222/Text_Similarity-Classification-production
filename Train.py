from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import argparse
import os

import warnings
warnings.filterwarnings('ignore')


root_dir = os.getcwd()

modelname = input("Type model name : ")
filename = input("Type file name : ")
import os
if not os.path.exists(root_dir + modelname):
	os.makedirs(root_dir + modelname)

# Reading the file 
train = pd.read_excel(filename)
train['Description']=train['Ticket_Description'].astype('str')

train['Level1'] =  train['Level1'].astype('str')
train['Level2'] =  train['Level2'].astype('str')

train.dropna(subset = ['Level1', 'Level2'])

train['Intent'] = train[['Level1','Level2']].agg('__'.join, axis=1).astype('category')
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
	model_loc = root_dir + '\\' + modelname + '\\' + str(name).replace('/','or') + ".json.gz_model.pkl"
	pickle.dump(m, open(model_loc,'wb'))
	r_loc = root_dir + '\\' + modelname + '\\' +  str(name).replace('/','or') + ".json.gz_r.npz"
	sparse.save_npz(r_loc, r)
	vec_loc = root_dir + '\\' + modelname + '\\' + str(name).replace('/','or') + ".json.gz_vector.pkl"
	pickle.dump(vec, open(vec_loc,'wb'))
