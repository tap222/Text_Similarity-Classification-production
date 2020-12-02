from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import os
import sys
import utils
import traceback
import re, string
from scipy import sparse
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

def traindata(pData, pDesc, pLevel1, pLevel2, pFromDir, pToDir):
    try:
        pData[pDesc]= pData[pDesc].astype('str')
        pData[pLevel1] =  pData[pLevel1].astype('str')
        pData[pLevel2] =  pData[pLevel2].astype('str')
        pData.dropna(subset = [pLevel1, 'Level2'])
        pData['Intent'] = pData[[pLevel1,pLevel2]].agg('__'.join, axis=1).astype('category')
        pLabel =  pData['Intent'].cat.categories.tolist()
        
    except Exception as e:
        print('*** ERROR[001]: Error in Training: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        utils.movefile(pFromDir, pToDir)
        return(-1)
    return pData, pLabel

# ###########################################################################################################################
# # Author      : Tapas Mohanty                                                                                        
# # Functionality : Create Model for given Training ID using TfidfVectorizer and custom Tokenizer
# ###########################################################################################################################
    
def pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


def get_mdl(x, y):
    y = y.values
    r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
    m = LogisticRegression(C=4,solver='sag',dual=False,class_weight="balanced",n_jobs=-1)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

def vector_trans(pData, pDesc, pModelName, pRootDir, pFromDir, pToDir):
    try:
        pData[pDesc].fillna("unknown", inplace=True)
        print('Started vector for Sample ')
        vec = TfidfVectorizer(ngram_range=(1,5),
                              stop_words="english",
                              analyzer='char',
                              tokenizer=tokenize,
                              min_df=3, max_df=0.9, 
                              strip_accents='unicode',
                              use_idf=1,
                              smooth_idf=1, 
                              sublinear_tf=1)
        vec.fit(pData[pDesc])
        x = vec.transform(pData[pDesc])
        if not os.path.exists(pRootDir + '\\' +  str(pModelName) +  '\\' + str(pModelName[6:]) + '_Vector'):
            os.makedirs(pRootDir + '\\' +   str(pModelName) +  '\\' + str(pModelName[6:]) + '_Vector')
        vec_loc = pRootDir + '\\' + str(pModelName)  + '\\' + str(pModelName[6:]) + '_Vector'  +'\\' + str(pModelName[6:])  +  ".vector.pkl"
        pickle.dump(vec, open(vec_loc,'wb'))
        print('completed vector for Sample ')
        
    except OSError as e:
        raise(e)
        print(traceback.format_exc())
        print("*** ERROR[002]: %s " % (e.strerror))
        utils.movefile(pFromDir, pToDir)
        return(-1)      
    return x,vec
    
def tokenize(s): 
    re_tok = re.compile(f'([{string.punctuation}“”¨_«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()  
    
def createModel(pData, pDesc, pLevel1, pLevel2, pModelName, pRootDir, nTickets, pFromDir, pToDir):
    try:
        x,vec = vector_trans(pData, pDesc, pModelName, pRootDir, pFromDir, pToDir)
        print('Number of Tickets for training :', len(pData)) 
        pTrainData, __ = traindata(pData, pDesc, pLevel1, pLevel2, pFromDir, pToDir)      
        pTrainData['Intent']= pTrainData['Intent'].astype('category')
        pLabel = [k for k in pTrainData['Intent'].value_counts().keys() if pTrainData['Intent'].value_counts()[k] > int(nTickets)]
        pTrainData = pd.concat([pTrainData,pd.get_dummies(pTrainData['Intent'])],axis=1)
        pTrainData.drop(['Intent'],axis=1, inplace=True)
        
        for index,name in enumerate(pLabel):
            print('Creating vector for intent: ', name)
            m,r = get_mdl(x, pTrainData[name])
            pFolderName = ['_Csr_matrix','_Model']       
            for foldername in pFolderName:
                if not os.path.exists(pRootDir + '\\' +  str(pModelName) + '\\' + str(pModelName[6:]) + foldername):
                    os.makedirs(pRootDir + '\\' +   str(pModelName)+ '\\' + str(pModelName[6:]) + foldername)
                if foldername == '_Model':
                    model_loc = pRootDir + '\\' +  str(pModelName) + '\\' + str(pModelName[6:]) +  foldername + '\\' + str(name).replace('/','or') + ".model.pkl"
                    pickle.dump(m, open(model_loc,'wb'))                
                else:
                    r_loc = pRootDir + '\\' +  str(pModelName) + '\\' + str(pModelName[6:]) +  foldername + '\\' + str(name).replace('/','or') + ".npz" 
                    sparse.save_npz(r_loc, r)   

    except OSError as e:
        raise(e)
        print(traceback.format_exc())
        print("*** ERROR[003] : %s" % (e.strerror))
        utils.movefile(pFromDir, pToDir)
        return(-1)  
    return(0)   