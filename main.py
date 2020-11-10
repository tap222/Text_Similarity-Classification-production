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

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Run the main file for training
###########################################################################################################################

def maintrain(pData, pDesc, pLevel1, pLevel2, pModelName, pRootDir):
    if not set([config.pDesc, config.pLevel1, config.pLevel2]).issubset(pData.columns):
        print('*** ERROR[010]: Loading XLS - Could be due to using non-standard template ***', str(pData.columns))
        return(-1, pData)
    try:
       train.createModel(pData, pDesc, pLevel1, pLevel2, pModelName, pRootDir)
    
    except Exception as e:
        print('*** ERROR[001]: Error in Training: ', sys.exc_info()[0],str(e))
        return(-1)
    return(0)
    
###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Run the main file for testing
###########################################################################################################################

def maintest(pData, pDesc, pTh, pTicketId, pModelName, pRootDir):
    if not set([config.pDesc, config.pTicketId]).issubset(pData.columns):
        print('*** ERROR[010]: Loading XLS - Could be due to using non-standard template ***', str(pData.columns))
        return(-1, pData)
    try:
        _, TestOutputData = test.intentpred(pData, pDesc, pTh,pTicketId, pModelName, pRootDir)
    
    except Exception as e:
        print('*** ERROR[002]: Error in Training: ', sys.exc_info()[0],str(e))
        return(-1)
    return(0, TestOutputData)


if __name__ == "__main__":
    if config.Train:
        pTrainingFiles, pData = utils.Filelist(config.pTrainDir)
        if len(pTrainingFiles) > 0: 
            pTrainingData = pData
            utils.setupFile(config.pRootDir, pDataFolder = config.pModelName) 
            if not os.path.exists(config.pRootDir +  '\\' + 'traindata' +  '\\' + str(config.pModelName[6:]) ):
                os.makedirs(config.pRootDir + '\\' + 'traindata' + '\\' + str(config.pModelName[6:]) )
            pTrainingData.to_excel(os.path.join(config.pRootDir  + '\\' + 'traindata' + '\\' + str(config.pModelName[6:]), config.pTrainingFileName + '.xlsx'), index = False)      
        else:
            print('No files in the directory')
            sys.exit(-1)
        print('*************************Training Started***********************************************')
        maintrain(pTrainingData, config.pDesc, config.pLevel1, config.pLevel2, config.pModelName, config.pRootDir)  
        print('*************************Training Completed*********************************************')
    if config.Test:
        pTestFiles, pData = utils.Filelist(config.pTestDir)
        if len(pTestFiles) > 0: 
            pTestingData = pData
            if not os.path.exists(config.pRootDir +  '\\' + 'testdata' +  '\\' + str(config.pModelName[6:]) ):
                os.makedirs(config.pRootDir + '\\' + 'testdata' + '\\' + str(config.pModelName[6:]) )
            pTestingData.to_excel(os.path.join(config.pRootDir  + '\\' + 'testdata' + '\\' + str(config.pModelName[6:]), config.pTestFileName + '.xlsx'), index = False)         
        else:
            print('No files in the directory')
            sys.exit(-1)
        print('*************************Testing Started***********************************************')
        _, pTestOutputData = maintest(pTestingData, config.pDesc, config.pTh, config.pTicketId, config.pModelName, config.pRootDir)
        pTestOutputData.to_excel(os.path.join(config.pRootDir  + '\\' + 'output', config.pTestFileName + '__' + 'ouput' + '.xlsx'), index = False) 
        print('*************************Testing Completed*********************************************')