import traceback
import os
import pandas as pd
import sys
import tarfile
import datetime
import shutil

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Collate multiple file into one
###########################################################################################################################

def Filelist(pDir, pSheetName):
    try: 
        pData = pd.DataFrame() 
        pFiles, pAppendData = [],[]
        for file in os.listdir(pDir):
            pFiles.append(file)  
            
        if len(pFiles) > 0:       
            for file in os.listdir(os.path.join(pDir)):
                if pSheetName != None:
                    pDataFile = pd.read_excel(os.path.join(pDir, file), sheet_name = str(pSheetName))
                else:
                    pDataFile = pd.read_excel(os.path.join(pDir, file))
                pAppendData.append(pDataFile)   
            pData = pd.concat(pAppendData)
        
        # for root, dirs, files in os.walk(pDir):
            # for file in files:
                # os.remove(os.path.join(root, file))
            
    except Exception as e:
        print(traceback.format_exc())
        print("*** ERROR[003]: %s : %s" % (pDir , e.strerror)) 
    return pFiles, pData 

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : create excel file and add into .tar for each folder
###########################################################################################################################

def setupFile(pRootdir, pDataFolder):
    try:
        pDataFiles = []
        moduledirectory = os.path.join(pRootdir,pDataFolder)
        
        if not os.path.exists(moduledirectory):
            os.makedirs(moduledirectory)
        
        os.chdir(moduledirectory)
        files = os.listdir(moduledirectory)
        for filename in files:
            if(os.path.exists(os.path.join(moduledirectory, filename))):
                pDataFiles.append(os.path.join(moduledirectory, filename))
            else:
                continue  
        os.chdir(pRootdir) 
        file_name = pRootdir + '\\archive\\' + 'data' + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")+".tar.gz"
        tar = tarfile.open(file_name, "w:gz")   
        for filename in pDataFiles:
            os.chdir(os.path.dirname(filename))  
            for name in os.listdir('.'):            
                print('file added :', os.path.basename(filename))
                tar.add(name)
                shutil.rmtree(os.path.join(moduledirectory,name))
        tar.close()
        os.chdir(pRootdir) 
        os.chdir('../')    
        
    except OSError as e:
        raise(e)
        print(traceback.format_exc())
        print("*** ERROR[004]: %s : %s" % (pDataFolder , e.strerror))
        return(-1)          
    return(0)
    
def movefile(pFromDir, pToDir):
    try:
        for root, dirs, files in os.walk(pFromDir):
            for file in files:
                shutil.move(root + '//' + file, pToDir)

    except OSError as e:
        raise(e)
        print(traceback.format_exc())
        print("*** ERROR[004]: %s : %s" % (pDataFolder , e.strerror))
        return(-1)          
    return(0)
