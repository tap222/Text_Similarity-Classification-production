import os
pModelName = 'model\\'+'TestModel' #model Name 
pTh = 0.3 # Thershold below intent is categorzied other
pDesc = 'Ticket_Description' # Ticket Description column name for training and testing
pRootDir = 'C:\\Users\\tamohant\\Desktop\\Auto_synthesis_Training_data\\AutoSynthesisTesting\\data' # Directory of the data file
pLevel1 = 'Level1' # Column name of Level1
pLevel2 = 'Level2' # column name of Level2
pTrainingFileName = 'TrainData' # name for training file
pTestFileName = 'TestData' # name for testing file
pTrainDir = os.path.join(pRootDir,'inputtraindata') # Training file directory
pTestDir = os.path.join(pRootDir,'inputtestdata') # Testing file Directory
Train = True # Boolean whether to run the script for training or not
Test = True # Boolean whether to run scrip for testing of not 
pTicketId = 'Ticket_No' # colum name for ticket column
