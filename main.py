import os
import sys
import test
import train
import utils
import config
import tarfile
import datetime
import traceback
import similarity
import pandas as pd
import preprocessing
import visualization

import warnings
warnings.filterwarnings("ignore")

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Run the main file for training
###########################################################################################################################

def maintrain(pData, pDesc, pLevel1, pLevel2, pModelName, pRootDir, nTickets, pTrainDir, pFailedDir):
    if not set([pDesc, pLevel1, pLevel2]).issubset(pData.columns):
        utils.movefile(pTrainDir, pFailedDir)
        __, pFailedData = utils.Filelist(pFailedDir)
        print('*** ERROR[001]: Loading XLS - Could be due to using non-standard template ***', str(pFailedData.columns))
        return(-1, pData)
    try:
       pData = pData.dropna(subset=[pDesc, pLevel1, pLevel2], how='any')
       train.createModel(pData, pDesc, pLevel1, pLevel2, pModelName, pRootDir, nTickets)
    
    except Exception as e:
        print('*** ERROR[002]: Error in Train main function: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        return(-1)
    return(0)
    
###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Run the main file for testing
###########################################################################################################################

def maintest(pData, pDesc, pTh, pThSim, pTicketId, pLevel1, pLevel2, pModelName, pRootDir, pTestDir, pFailedDir):
    if not set([pDesc, pTicketId]).issubset(pData.columns):
        utils.movefile(pTrainDir, pFailedDir)
        __, pFailedData = utils.Filelist(pFailedDir)
        print('*** ERROR[003]: Loading XLS - Could be due to using non-standard template ***', str(pFailedData.columns))
        return(-1, pData)
    try:
        pData = pData.dropna(subset=[pDesc, pTicketId], how='any')
        _, TestOutputData, pClassNames, pVec = test.intentpred(pData, pDesc, pTh, pThSim, pTicketId, pLevel1, pLevel2, pModelName, pRootDir)
    except Exception as e:
        print('*** ERROR[004]: Error in Test main function: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        return(-1)
    return(0, TestOutputData, pClassNames, pVec)

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Variables declared used in the script
###########################################################################################################################

pRootDir = config.pRootDir
pAccountName = config.pAccountName
pTrainingFileName = config.pTrainingFileName
pDesc = config.pDesc
pLevel1 = config.pLevel1
pLevel2 = config.pLevel2
pTh = config.pTh
pTicketId = config.pTicketId
pTestFileName = config.pTestFileName
pTrainDir = config.pTrainDir
pTestDir = config.pTestDir
nTickets = config.nTickets
Idx = config.Idx
viz = config.viz
nTopKeywrd = config.nTopKeywrd
pTrainingDataDir = config.pTrainingDataDir
pThSim = config.pThSim
pFailedDir = config.pFailedDir

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Run the main flow for both training the data and testing the data.
###########################################################################################################################

if __name__ == "__main__":
    if config.Train:
        pTrainingFiles, pData = utils.Filelist(pTrainDir)
        if len(pTrainingFiles) > 0: 
            pTrainingData = pData
            utils.setupFile(pRootDir, pAccountName) 
            if not os.path.exists(pRootDir +  '\\' + 'traindata' +  '\\' + str(pAccountName[6:]) ):
                os.makedirs(pRootDir + '\\' + 'traindata' + '\\' + str(pAccountName[6:]))
            pTrainingData.to_excel(os.path.join(pRootDir  + '\\' + 'traindata' + '\\' + str(pAccountName[6:]), pTrainingFileName + '.xlsx'), index = False)      
        else:
            print('No files in the directory')
            sys.exit(-1)
        if config.preprocessing:
            print('*************************Training Preprocess Started***********************************')
            # _, pTrainingPreprocesData = preprocess(pTrainingData, pTktDesc, pCol)
            _, pTrainingData = preprocessing.preprocess(pTrainingData, pDesc)
            pDesc = 'Sample'
            print('*************************Training Preprocess Completed*********************************')

        print('*************************Training Started***********************************************')
        maintrain(pTrainingData, pDesc, pLevel1, pLevel2, pAccountName, pRootDir, nTickets, pTrainDir, pFailedDir)  
        print('*************************Training Completed*********************************************')
    
    if config.Test:
        pTestFiles, pData = utils.Filelist(pTestDir)
        if len(pTestFiles) > 0: 
            pTestingData = pData
            if not os.path.exists(pRootDir +  '\\' + 'testdata' +  '\\' + str(pAccountName[6:]) ):
                os.makedirs(pRootDir + '\\' + 'testdata' + '\\' + str(pAccountName[6:]) )
            pTestingData.to_excel(os.path.join(pRootDir  + '\\' + 'testdata' + '\\' + str(pAccountName[6:]), pTestFileName + '.xlsx'), index = False)         
        else:
            print('No files in the directory')
            sys.exit(-1)
        
        if config.preprocessing:
            print('*************************Testing Preprocess Started***********************************')     
            # _, pTestingPreprocesData = preprocess(pTestingData, pDesc, pCol)
            pDesc = config.pDesc
            _, pTestingData = preprocessing.preprocess(pTestingData, pDesc)
            pDesc = 'Sample'
            print('*************************Testing Preprocess Completed*********************************')  

        print('*************************Testing Similarity Started***************************************')
        if config.sim:
            pTrainingFiles, pTrainingData = utils.Filelist(pTrainingDataDir)
            pDesc = config.pDesc
            if len(pTrainingFiles) > 0:
                # __, pTestingData = similarity.similaritymain(pTrainingData, pTestingData, pLevel1, pLevel2, pDesc)  
                __, pTestingData = similarity.similaritypolymain(pTrainingData, pTestingData, pLevel1, pLevel2, pDesc)  
            else:
                print('No Training File present to compare skipping similarity')
                pass
        print('*************************Testing Similarity Completed*************************************')     
        
        print('*************************Testing Started***********************************************')
        pDesc = 'Sample'
        _, pTestOutputData, pClassNames, pVec = maintest(pTestingData, pDesc, pTh, pThSim, pTicketId, pLevel1, pLevel2, pAccountName, pRootDir, pTestDir, pFailedDir)
        print('*************************Testing Completed*********************************************')
        print('*************************Visualization Started*********************************************')
        if config.viz:
            visualization.eli5visual(pTestOutputData, pDesc, Idx, pAccountName, pVec, nTopKeywrd, pRootDir)
        if not os.path.exists(pRootDir +  '\\' + 'output' +  '\\' + str(pAccountName[6:])):
            os.makedirs(pRootDir + '\\' + 'output' + '\\' + str(pAccountName[6:]))           
        pTestOutputData.to_excel(os.path.join(pRootDir  + '\\' + 'output' + '\\' + str(pAccountName[6:]), pTestFileName + '__' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + '__' + 'output' + '.xlsx'), index = False) 
        print('*************************Visualization Completed*********************************************')