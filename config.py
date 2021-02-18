import os

pTh = 0.1 # Thershold below intent is categorzied other
pDesc = 'Ticket_Description' # Ticket Description column name for training and testing
pTicketId = 'Ticket_No' # colum name for ticket column
pAccountName = 'model\\'+'Test' #model Name 
pRootDir = 'C:\\Users\\tamohant\\Desktop\\Auto_synthesis_Training_data\\AutoSynthesisLite\\data' # Directory of the data file
pTrainingDataDir = os.path.join(pRootDir  + '\\' + 'traindata' + '\\' + str(pAccountName[6:]))
pLevel1 = 'Level1' # Column name of Level1
pLevel2 = 'Level2' # column name of Level2
pTrainingFileName = str(pAccountName[6:]) + 'TrainData' # name for training file
pTestFileName = str(pAccountName[6:]) + 'TestData' # name for testing file
pTrainDir = os.path.join(pRootDir,'inputtraindata') # Training file directory
pTestDir = os.path.join(pRootDir,'inputtestdata') # Testing file Directory
pFailedDir = os.path.join(pRootDir,'faileddata')
Train = True # Boolean whether to run the script for training or not
Test = True # Boolean whether to run scrip for testing of not 
sim = True # Boolean whether to run scrip for Similarity or not
preprocessing = True
viz = False #Boolean whether to run the script of visualization
Idx = [100,120,150,300] # Row no. to display for keyword weightage
nTopKeywrd = 10 #Top keywords to represent in weight calculation for an intent
nTickets = 3
pThSim = 0.60
pSheetName = 'Sheet1'
Nbest = 2 # Number next best parameter
ewords = False
features = ['Category','Subcategory']