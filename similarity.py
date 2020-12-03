import sys
import utils
import traceback
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from polyfuzz import PolyFuzz

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : calculate the sparse matrix for similarity calculation
###########################################################################################################################

def awesome_cossim_top(A, B, ntop, pFromDir, pToDir, lower_bound=0):
    try:
        # force A and B as a CSR matrix.
        # If they have already been CSR, there is no overhead
        A = A.tocsr()
        B = B.tocsr()
        M, _ = A.shape
        _, N = B.shape
     
        idx_dtype = np.int32
     
        nnz_max = M*ntop
     
        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)
    except Exception as e:
        print('*** ERROR[001]: Error in similarity calculating matrix: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        utils.movefile(pFromDir, pToDir)
        return(-1)

    return csr_matrix((data,indices,indptr),shape=(M,N))

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Transformation of train data i.e. clubbing level 1 and level2 into one column name intent.
###########################################################################################################################

def traindata(pData, pDesc, pLevel1, pLevel2, pFromDir, pToDir):
    try:
        pData[pDesc]= pData[pDesc].astype('str')
        pData[pLevel1] =  pData[pLevel1].astype('str')
        pData[pLevel2] =  pData[pLevel2].astype('str')
        pData.dropna(subset = [pLevel1, 'Level2'])
        pData['Intent'] = pData[[pLevel1,pLevel2]].agg('__'.join, axis=1).astype('category')
        pLabel =  pData['Intent'].cat.categories.tolist()
        
    except Exception as e:
        print('*** ERROR[002]: Error in similarity transform train data: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        utils.movefile(pFromDir, pToDir)
        return(-1)
    return pData, pLabel
 
###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Finding the similarity between two text
###########################################################################################################################
 
def similaritymain(pTrainData, pTestData, pLevel1, pLevel2, pDesc, pFromDir, pToDir):
    try:
        pMatches, pTestData['Intent'], pTestData['Confidence_Level'] = [],'Nan','Nan'
        pTrainData, __ = traindata(pTrainData, pDesc, pLevel1, pLevel2)
        pTrainDataDesc = pd.DataFrame(pTrainData[pDesc])
        pTrainDataDescUnq = pTrainDataDesc[pDesc].unique()
        vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', lowercase=False)
        tfidf = vectorizer.fit_transform(pTrainDataDescUnq)
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
        queryTFIDF_ = vectorizer.transform(pTestData[pDesc].values)
        distances, indices = nbrs.kneighbors(queryTFIDF_)
        pTestDataDescList = list(pTestData[pDesc].values) #need to convert back to a list

        for i,j in enumerate(indices):
          pTemp = [distances[i][0], pTrainDataDesc.values[j][0][0],pTestDataDescList[i]]
          pMatches.append(pTemp)
        pMatchesDf = pd.DataFrame(pMatches, columns=['Confidence_Level','Matched name','Original name'])
        
        for i in range(len(pTestData)):
            pTestData['Intent'][i] = pTrainData[np.where(pTrainData[pDesc] == pMatchesDf['Matched name'][i], True , False)]['Intent'].values[0]
            pTestData['Confidence_Level'][i] = pMatchesDf['Confidence_Level'][i]
    except Exception as e:
        print('*** ERROR[003]: Error in similarity main function: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        utils.movefile(pFromDir, pToDir)
        return(-1)
    return(0, pTestData)

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Fiding the similarity between two text using polyfuzz
# Comments      : https://github.com/MaartenGr/PolyFuzz
###########################################################################################################################

def similaritypolymain(pTrainData, pTestData, pLevel1, pLevel2, pDesc, pFromDir, pToDir):
    try:
        pTrainData = pTrainData[pTrainData[pDesc].notna()]
        pTestData = pTestData[pTestData[pDesc].notna()]
        pTestData['Intent'], pTestData['Confidence_Level'] = 'Nan','Nan'
        pTrainData, __ = traindata(pTrainData, pDesc, pLevel1, pLevel2, pFromDir, pToDir)
        pTrainDataDesc = pd.DataFrame(pTrainData[pDesc])
        pTrainDataDescUnq = pTrainDataDesc[pDesc].unique().tolist()
        pTestDataDescList = list(pTestData[pDesc].values) #need to convert back to a list
        model = PolyFuzz("TF-IDF")
        model.match(pTestDataDescList, pTrainDataDescUnq)
        pMatchesDf = model.get_matches()
        for i in range(len(pTestData)):
            if pMatchesDf['To'][i] != None:
                pTestData['Intent'][i] = pTrainData[np.where(pTrainData[pDesc] == pMatchesDf['To'][i], True , False)]['Intent'].values[0]
            pTestData['Confidence_Level'][i] = pMatchesDf['Similarity'][i]
            
    except Exception as e:
        print('*** ERROR[004]: Error in similarity poly main function: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        utils.movefile(pFromDir, pToDir)
        return(-1)
    return(0, pTestData)    