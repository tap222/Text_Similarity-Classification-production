import os
import sys
import glob
import utils
import pickle
import random
import traceback
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack 
###########################################################################################################################
# # Author      : Tapas Mohanty                                                                                        
# # Functionality : Importing the pickle files for model, vector from the model folder
# ###########################################################################################################################

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "custom_token"
        return super().find_class(module, name)
        
def loadTfidfFile(pRootDir, pModelName):
    vec_loc = pRootDir + '\\' + str(pModelName)  + '\\' + str(pModelName[6:]) + '_Vector'  +'\\' + str(pModelName[6:])  +  ".vector.pkl"
    with open(vec_loc, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        vec = unpickler.load()
    return vec

def loadEncFile(pRootDir, pModelName):
    enc_loc = pRootDir + '\\' + str(pModelName)  + '\\' + str(pModelName[6:]) + '_Ohe_encode'  +'\\' + str(pModelName[6:])  +  ".ohe.pkl"
    with open(enc_loc, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        enc = unpickler.load()
    return enc
    
def loadcsr_matrix(pRootDir, pModelName, pIntent):
    r_loc = pRootDir + '\\' +  str(pModelName) + '\\' + str(pModelName[6:]) +  '_Csr_matrix' + '\\' +  str(pIntent) + ".npz"
    r = sparse.load_npz(r_loc)
    return r
    
def loadmodel(pRootDir, pModelName, pIntent):
    model_loc = pRootDir + '\\' +  str(pModelName) + '\\' + str(pModelName[6:]) +  '_Model' + '\\' + str(pIntent) + ".model.pkl"
    with open(model_loc, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        model = unpickler.load()
    return model

def categories(pRootDir, pModelName):
    path = pRootDir + '\\' +  str(pModelName) + '\\' + str(pModelName[6:]) +  '_Model' + '\\'
    file = glob.glob(path +'/*.model.pkl')
    category_names =  []
    for i in range(len(file)):
        basename = os.path.basename(file[i])
        category_names.append(basename.split('.model.pkl')[0])
    return category_names

def featuresconcat(pData, features, encoder, pVec):
    try:
        Test_encoded = encoder.transform(pData[features])
        test_x = hstack([pVec, Test_encoded]).tocsr()
        
    except Exception as e:
        print('*** ERROR[001]: Error in Training: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        return(-1)
        sys.exit(-1)
    return test_x   
    
# ###########################################################################################################################
# # Author      : Tapas Mohanty                                                                                        
# # Functionality : Find intent for the tickets which has low thershold value by using NB-SVM and Logistic Regression
# ###########################################################################################################################
def intentpred(pData, pDesc, pTh, pThSim, pTicketId, pLevel1, pLevel2, pModelName, pRootDir, pFromDir, pToDir, sim, features, pFeatures):

    try:
        if 'Confidence_Level' not in pData:
            pData['Confidence_Level'] = float(pThSim + 1)   
        if sim:           
            pDataTh = pData[np.where(pData['Confidence_Level'] < float(pThSim), True, False)]
        else:
            pDataTh = pData
        print('Length of file for Prediction after similarity:', pDataTh.shape[0])      
        if len(pDataTh) > 0:
            pDataTh[pDesc].fillna("unknown", inplace=True)     
            oCategoryNames = categories(pRootDir, pModelName)
            pDataTh[pTicketId] = pDataTh[pTicketId].astype('category') 
            preds = np.zeros((len(pDataTh), len(oCategoryNames)))
            
            vec = loadTfidfFile(pRootDir, pModelName)
            
            if pFeatures:
                oTktVec = vec.transform(pDataTh[pDesc].astype(str))
                encoder = loadEncFile(pRootDir, pModelName)
                tkt_desc = featuresconcat(pDataTh, features, encoder, pVec = oTktVec)
            else:
                tkt_desc = vec.transform(pDataTh[pDesc].astype(str))

            for index,name in enumerate(oCategoryNames):
                print('Calculating prediction of intent', name)
                estimator = loadmodel(pRootDir, pModelName, name)           
                r = loadcsr_matrix(pRootDir, pModelName, name)
                preds[:,index] = estimator.predict_proba(tkt_desc.multiply(r))[:,1]      
            pintentdf = pd.DataFrame(preds, columns = oCategoryNames)
            pintentdf['Confidence_Level'] = pintentdf[oCategoryNames].max(axis=1)
            pintentdf['Intent'] = pintentdf[oCategoryNames].idxmax(axis=1)
            pintentdf['Intent']= np.where(pintentdf['Confidence_Level'] > float(pTh), pintentdf['Intent'] , 'Others') 
            pDataTh.reset_index(drop=True, inplace=True)
            pintentdf.reset_index(drop=True, inplace=True) 
            pintentdf = pd.concat([pDataTh[pTicketId], pintentdf],axis=1)  
            pintentdf = pintentdf[[pTicketId,'Confidence_Level','Intent']]
            pData.loc[pData[pTicketId].isin(pintentdf[pTicketId]), ['Confidence_Level', 'Intent']] = pintentdf[['Confidence_Level', 'Intent']].values
            pData[['Level1','Level2']] = pData.Intent.str.split("__",expand=True,)
        else:
            pData['Confidence_Level'] = pData['Confidence_Level'].astype('float')
            
        pData[['Level1', 'Level2']] = pData.Intent.str.split("__",expand=True,)       
 
    except Exception as e:
        print(e)
        print('*** ERROR[001]: intentpred ***', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        utils.movefile(pFromDir, pToDir)
        return(-1, pData)
        sys.exit(-1)        
    return(0, pData, oCategoryNames, vec)
