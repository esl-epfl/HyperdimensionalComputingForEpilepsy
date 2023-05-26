__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

from HDfunctionsLib import *
from parametersSetup import *

###################################
#CHBMIT dataset
Dataset='01_CHBMIT'
folderIn = '../../../../../databases/medical/chb-mit/edf/'
GeneralParams.patients  =['01','02','03','04','05','06','07','08','09','10','11', '12','13','14','15','16','17','18','19','20','21','22','23','24']
GeneralParams.patients  =['01','02','03']

####################################################################
#CREATING SUBSETS OF DATABASE (AND IN CSV FORMAT INSTEAD OF EDF)
# #Just convert original edf data to csv data
# folderOut = '01_datasetProcessed_Raw/'
# extractEDFdataToCSV_originalData(folderIn, folderOut, SigInfoParams, patients)

# Creating database with only subset of non-seizure data
print('factor 1')
factor=1 #defines how many times more non-seizure data should be in respect to seizure
folderOut = '01_datasetProcessed_' + 'MoreNonSeizure_Fact'+str(factor)
extractEDFdataToCSV_MoreNonSeizThenSeizData(folderIn, folderOut, SigInfoParams, GeneralParams.patients, factor)

factor=5 #defines how many times more non-seizure data should be in respect to seizure
folderOut = '01_datasetProcessed_' + 'MoreNonSeizure_Fact'+str(factor)
extractEDFdataToCSV_MoreNonSeizThenSeizData(folderIn, folderOut, SigInfoParams, GeneralParams.patients, factor)

factor=10 #defines how many times more non-seizure data should be in respect to seizure
folderOut = '01_datasetProcessed_' + 'MoreNonSeizure_Fact'+str(factor)
extractEDFdataToCSV_MoreNonSeizThenSeizData(folderIn, folderOut, SigInfoParams, GeneralParams.patients, factor)