import train
import test
import traceback
import config
import os
import pandas as pd
import sys
import tarfile
import datetime
import utils
import preprocessing

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Run the main file for training
###########################################################################################################################

def maintrain(pData, pDesc, pLevel1, pLevel2, pModelName, pRootDir):
    if not set([pDesc, pLevel1, pLevel2]).issubset(pData.columns):
        print('*** ERROR[001]: Loading XLS - Could be due to using non-standard template ***', str(pData.columns))
        return(-1, pData)
    try:
       train.createModel(pData, pDesc, pLevel1, pLevel2, pModelName, pRootDir)
    
    except Exception as e:
        print('*** ERROR[002]: Error in Train main function: ', sys.exc_info()[0],str(e))
        return(-1)
    return(0)
    
###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Run the main file for testing
###########################################################################################################################

def maintest(pData, pDesc, pTh, pTicketId, pLevel1, pLevel2, pModelName, pRootDir):
    if not set([pDesc, pTicketId]).issubset(pData.columns):
        print('*** ERROR[003]: Loading XLS - Could be due to using non-standard template ***', str(pData.columns))
        return(-1, pData)
    try:
        _, TestOutputData = test.intentpred(pData, pDesc, pTh, pTicketId, pLevel1, pLevel2, pModelName, pRootDir)
    
    except Exception as e:
        print('*** ERROR[004]: Error in Test main function: ', sys.exc_info()[0],str(e))
        return(-1)
    return(0, TestOutputData)

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Variables declared used in the script
###########################################################################################################################

pRootDir = config.pRootDir
pModelName = config.pModelName
pTrainingFileName = config.pTrainingFileName
pDesc = config.pDesc
pLevel1 = config.pLevel1
pLevel2 = config.pLevel2
pTh = config.pTh
pTicketId = config.pTicketId
pTestFileName = config.pTestFileName
pTrainDir = config.pTrainDir
pTestDir = config.pTestDir

if __name__ == "__main__":
    if config.Train:
        pTrainingFiles, pData = utils.Filelist(pTrainDir)
        if len(pTrainingFiles) > 0: 
            pTrainingData = pData
            utils.setupFile(pRootDir, pModelName) 
            if not os.path.exists(pRootDir +  '\\' + 'traindata' +  '\\' + str(pModelName[6:]) ):
                os.makedirs(pRootDir + '\\' + 'traindata' + '\\' + str(pModelName[6:]) )
            pTrainingData.to_excel(os.path.join(pRootDir  + '\\' + 'traindata' + '\\' + str(pModelName[6:]), pTrainingFileName + '.xlsx'), index = False)      
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
        maintrain(pTrainingData, pDesc, pLevel1, pLevel2, pModelName, pRootDir)  
        print('*************************Training Completed*********************************************')
    
    if config.Test:
        pTestFiles, pData = utils.Filelist(pTestDir)
        if len(pTestFiles) > 0: 
            pTestingData = pData
            if not os.path.exists(pRootDir +  '\\' + 'testdata' +  '\\' + str(pModelName[6:]) ):
                os.makedirs(pRootDir + '\\' + 'testdata' + '\\' + str(pModelName[6:]) )
            pTestingData.to_excel(os.path.join(pRootDir  + '\\' + 'testdata' + '\\' + str(pModelName[6:]), pTestFileName + '.xlsx'), index = False)         
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
        
        print('*************************Testing Started***********************************************')
        _, pTestOutputData = maintest(pTestingData, pDesc, pTh, pTicketId, pLevel1, pLevel2, pModelName, pRootDir)
        pTestOutputData.to_excel(os.path.join(pRootDir  + '\\' + 'output', pTestFileName + '__' + 'ouput' + '.xlsx'), index = False) 
        print('*************************Testing Completed*********************************************')