__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

'''script that does several things (comment out things not needed) 
- converts edf files to csv files without any changes in data 
- calculates features for each input file and saves them 
    - possible to choose which features (e.g. meanAmpl, lineLength, FrequencyFeatues, ZeroCross (not used now)
- performas subselection of data or reorders is 
    - subselection x1 - contains the same amount of seizure and non seizure data (non seizure randomly selected from files with no seizure)
    - subselection x10 - the same as x1, but contains 10 times more non-seizure data then seizure 
    - all data StoS - rearanges all data so that each file contains data from end of previous seizure to beginning of current seizure 
    - all data fixed size files - rearanges all data to contain fixes amount of data (e.g. 1h or 4h) 
- also plots labels for any rearangement - this is useful to check if rearangement was done correctly and no data was lost 
    
'''

from parametersSetup import *
from VariousFunctionsLib import *

#########################################################################
#CHBMIT
Dataset='01_CHBMIT' #'01_CHBMIT', '01_iEEG_Bern'
GeneralParams.patients  =['01','02','03','04','05','06','07','08','09','10','11', '12','13','14','15','16','17','18','19','20','21','22','23','24']
createFolderIfNotExists( '../'+Dataset)

# OTHER PARAMETERS
GeneralParams.PersGenApproach='personalized' #'personalized', 'generalized'
datasetFiltering='1to30Hz'  # 'Raw', 'MoreNonSeizure_Fact10' #depends if we want all data or some subselection
FeaturesParams.winLen = 4  # in seconds, window length on which to calculate features
FeaturesParams.winStep = 0.5  # in seconds, step of moving window length
repearabilityVer='' #fore testing repeatablity, otherwise put ''
keepOriginalFileNames=1

# DEFINING INPUT/OUTPUT FOLDERS
folderInEDF = '../../../../databases/medical/chb-mit/edf/' #location on server so that we dont have to download to each persons folder
folderInCSV = '../'+Dataset+'/01_datasetProcessed_Raw/' #where to save filtered raw data
createFolderIfNotExists(folderInCSV)
folderInfo= '../'+Dataset+'/01_SeizureInfoOriginalData/' #folder to save results of basic analysis about seizures
createFolderIfNotExists(folderInfo)
folderOutFeatures= '../'+Dataset+'/02_Features_' +datasetFiltering+'_'+str(FeaturesParams.winLen )+'_'+ str(FeaturesParams.winStep)  #where to save calculated features for each original file
createFolderIfNotExists(folderOutFeatures)


# #####################################################################
# # CONVERT EDF TO CSV DATA
# # convert original edf data to csv data
extractEDFdataToCSV_originalData_gzip(folderInEDF, folderInCSV, DatasetPreprocessParams, GeneralParams.patients , keepOriginalFileNames)
#
# #########################################################################
# # #EXPORTING SEIZURE STRUCTURE PER FILE
analyseSeizureDurations(folderInEDF, folderInfo, GeneralParams.patients)
#
# # CALCULATE RANGES OF SIGNAL VALUES
# #percntile 5-95 of values - this is needed for ZeroCross features
# calculateStatisticsOnSignalValues(folderInCSV, folderInfo, GeneralParams)

# #########################################################################
# # CALCULATE FEATURES FOR EACH FILE
FeaturesParams.featNames = np.array( ['MeanAmpl', 'LineLength', 'Frequency'])
sigRanges=readDataFromFile(folderInfo + '/AllSubj_DataRange5to95Percentile.csv')
calculateFeaturesPerEachFile_gzip(ZeroCrossFeatureParams.EPS_thresh_arr, folderInCSV, folderOutFeatures, GeneralParams,DatasetPreprocessParams, FeaturesParams,  sigRanges)

# ##########################################################################
# # PERFORMING DATA SELECTION - NOT USING ALL DATA
# # CONCATENATE FEATURES TO FILES SO THAT MORE NONSEIZRUE DATA BY SOME FACTOR
factor=10 #1, 10
folderOutFeatures= '../'+Dataset+'/02_Features_' +datasetFiltering+'_'+str(FeaturesParams.winLen )+'_'+ str(FeaturesParams.winStep)
folderOutRearangedData= '../'+Dataset+'/04_RearangedData_MergedFeatures_' +datasetFiltering+'_'+str(FeaturesParams.winLen )+'_'+ str(FeaturesParams.winStep) +'_' +'-'.join(FeaturesParams.featNames)+'_Fact'+str(factor) +repearabilityVer#where to save calculated features for each original file
createFolderIfNotExists(folderOutRearangedData)
concatenateAllFeatures_moreNonseizureForFactor_gzip(folderOutFeatures, folderOutRearangedData,GeneralParams, DatasetPreprocessParams, FeaturesParams,  factor)
plotRearangedDataLabelsInTime(folderOutRearangedData,  GeneralParams,PostprocessingParams, FeaturesParams)

#
# #########################################################################
# USING ALL DATA - NOT USED IN THIS PAPER

# REORDERING DATA TO ROLLING BASIS SEIZURE TO SEIZURE APPROACH
# use actually features calculated for the whole database
# and save as new files where each file contains data from begining of last siezure to start of new one
# but exclude windows that are 1 min before seizure and ? time after seizure
folderOutFeatures= '../'+Dataset+'/02_Features_' +datasetFiltering+'_'+str(FeaturesParams.winLen )+'_'+ str(FeaturesParams.winStep)
folderOutRearangedData= '../'+Dataset+'/04_RearangedData_MergedFeatures_' +datasetFiltering+'_'+str(FeaturesParams.winLen )+'_'+ str(FeaturesParams.winStep) +'_' +'-'.join(FeaturesParams.featNames)+'_AllDataStoS'+repearabilityVer
createFolderIfNotExists(folderOutRearangedData)
concatenateFeatureFilesStoSApproach_gzip(folderOutFeatures, folderOutRearangedData, GeneralParams, FeaturesParams)
plotRearangedDataLabelsInTime(folderOutRearangedData,  GeneralParams,PostprocessingParams, FeaturesParams)

# CONCATENATE FEAUTURS TO FILES WITH EQUAL AMOUNT OF DATA
FeaturesParams.featNames = np.array( ['MeanAmpl', 'LineLength', 'Frequency'])
windowSize=4*60*60 #in seconds  # or 4*60*60 - for 1h or 4h files
folderOutFeatures= '../'+Dataset+'/02_Features_' +datasetFiltering+'_'+str(FeaturesParams.winLen )+'_'+ str(FeaturesParams.winStep)
folderOutRearangedData= '../'+Dataset+'/04_RearangedData_MergedFeatures_' +datasetFiltering+'_'+str(FeaturesParams.winLen )+'_'+ str(FeaturesParams.winStep) +'_' +'-'.join(FeaturesParams.featNames)+'_AllDataWin'+str(windowSize)+'s'
createFolderIfNotExists(folderOutRearangedData)
concatenateFeatures_allDataInEqualWindows_gzip(folderOutFeatures, folderOutRearangedData, GeneralParams,  FeaturesParams,  windowSize)
plotRearangedDataLabelsInTime(folderOutRearangedData,  GeneralParams,PostprocessingParams, FeaturesParams)

# ADAPTED WITH FIRST FILE CONTAINING SEIZURE FOR SURE
windowSize=4*60*60 #in seconds
numHours = 1 #at least 6h in first file
folderOutRearangedData= '../'+Dataset+'/04_RearangedData_MergedFeatures_' +datasetFiltering+'_'+str(FeaturesParams.winLen )+'_'+ str(FeaturesParams.winStep) +'_' +'-'.join(FeaturesParams.featNames)+'_AllDataWin'+str(windowSize)+'s'
folderOutRearangedData2= '../'+Dataset+'/04_RearangedData_MergedFeatures_' +datasetFiltering+'_'+str(FeaturesParams.winLen )+'_'+ str(FeaturesParams.winStep) +'_' +'-'.join(FeaturesParams.featNames)+'_AllDataWin'+str(windowSize)+'s_1File'+str(numHours)+'h'
concatenateFeatures_allDataInEqualWindows_FirstFileNeedsSeizure(folderOutRearangedData, folderOutRearangedData2,GeneralParams, numHours)
plotRearangedDataLabelsInTime(folderOutRearangedData2,  GeneralParams,PostprocessingParams, FeaturesParams)

