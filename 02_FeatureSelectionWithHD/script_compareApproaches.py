__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

''' script to compare different approaches 
- compares performances of different encoding 
	- loads performances for different encoding approaches and plots them in a simple way to visually compare 
- compares memory and computational complexity of different encoding approaches 
	- calculates and plots memory and number of operations needed for each endcoding approach 
- creates table with all results on feature selection 
	- for all three approaches of feature selection 
	- reports improvements on performance and number of features chosen
'''

from HDfunctionsLib import *
from parametersSetup import *

# ###################################
# SETUP DATASET USED
# CHBMIT
Dataset = '01_CHBMIT'  # '01_CHBMIT', '01_iEEG_Bern'
GeneralParams.patients = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']


# PARAMETERS
GeneralParams.PersGenApproach = 'personalized'  # 'personalized', 'generalized'
datasetFiltering = '1to30Hz'  # 'Raw', 'MoreNonSeizure_Fact10' #depends if we want all data or some subselection
GeneralParams.CVtype = 'LeaveOneOut'  # 'LeaveOneOut', 'RollingBase'
datasetPreparationType = 'Fact10'  # 'Fact1', 'Fact10' ,'AllDataWin3600s', 'AllDataWin3600s_1File6h', 'AllDataStoS'  # !!!!!!
HDtype='StdHD' #'StdHD', 'OnlineHD'

# features used
FeaturesParams.featNames = np.array(['MeanAmpl', 'LineLength', 'Frequency'])
FeaturesUsed = '-'.join(FeaturesParams.featNames)
AllExistingFeatures = FeaturesParams.allFeatTypesName
FeaturesParams.featNames = constructAllfeatNames(FeaturesParams)  # feat used here
totalNumFeat = len(FeaturesParams.allFeatNames)  # all features that exist in original data
FeaturesParams.featNorm = 'Norm&Discr'  # 'Norm', 'Norm&Discr'  #!!! HAS TO BE DISCRETIZED
FeaturesParams.winLen = 4  # in seconds, window length on which to calculate features
FeaturesParams.winStep = 0.5  # in seconds, step of moving window length

# HD COMPUTING PARAMS
torch.cuda.set_device(HDParams.CUDAdevice)
HDParams.vectorTypeLevel = 'scaleNoRand1'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
HDParams.vectorTypeFeat = 'random'
HDParams.roundingTypeForHDVectors = 'inSteps'  # 'inSteps','onlyOne','noRounding'
HDParams.D = 1000
HDParams.bindingFeatures = 'FeatAppend'

# DEFINING INPUT/OUTPUT FOLDERS
folderInEDF = '../../../../databases/medical/chb-mit/edf/'  # location on server so that we dont have to download to each persons folder
folderInCSV = '../' + Dataset + '/01_datasetProcessed_Raw/'  # where to save filtered raw data
createFolderIfNotExists(folderInCSV)
folderInfo = '../' + Dataset + '/01_SeizureInfoOriginalData/'  # folder to save results of basic analysis about seizures
createFolderIfNotExists(folderInfo)
folderOutFeatures = '../' + Dataset + '/02_Features_' + datasetFiltering + '_' + str(FeaturesParams.winLen) + '_' + str( FeaturesParams.winStep)  # where to save calculated features for each original file
createFolderIfNotExists(folderOutFeatures)
folderOutRearangedData = '../' + Dataset + '/04_RearangedData_MergedFeatures_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str( FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType

# CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
folderOutPredictions0 = '../' + Dataset + '/05_Predictions_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str( FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType
createFolderIfNotExists(folderOutPredictions0)
if (HDParams.HDvecType == 'bin'):
    folderOutPredictions0 = folderOutPredictions0 + '/' + str(  GeneralParams.PersGenApproach) + '_' + GeneralParams.CVtype
elif (HDParams.HDvecType == 'bipol'):
    folderOutPredictions0 = folderOutPredictions0 + '/' + str( GeneralParams.PersGenApproach) + '_' + GeneralParams.CVtype + '_bipolarVec/'
createFolderIfNotExists(folderOutPredictions0)

# ## CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
folderOutParams = FeaturesUsed + '_' + str( HDParams.numSegmentationLevels) + '_' + HDParams.bindingFeatures + '_D' + str( HDParams.D)  # not all parameters are saved here (but coudl be if needed)
folderOutPredictions = folderOutPredictions0 + '/' + folderOutParams + '_RF_StdHD_OnlHD/'
createFolderIfNotExists(folderOutPredictions)
folderOutPredictionsPlot = folderOutPredictions + '/PredictionsComparison/'
createFolderIfNotExists(folderOutPredictionsPlot)
if (HDParams.bindingFeatures == 'FeatAppend'):
    folderOutPredictionsPerFeat = folderOutPredictions + '/PerFeat/'
    createFolderIfNotExists(folderOutPredictionsPerFeat)
    # OUTPUT FOR THIS VOTING TYPE
    folderOutPredictionsVoting = folderOutPredictions + '/Voting_ConfPerFeature_'+HDtype+'/'
    createFolderIfNotExists(folderOutPredictionsVoting)


# # ####################################################################################
# COMPARING PERFORMANCE  OF DIFFERENT ENCODINGS
folderOut='../01_CHBMIT/06_Comparison_BindingApproaches/'
createFolderIfNotExists(folderOut)

#### FOR D=19000
folderInArray=['../01_CHBMIT/05_Predictions_1to30Hz_4_0.5_MeanAmpl-LineLength-Frequency_Fact10/personalized_LeaveOneOut/MeanAmpl-LineLength-Frequency_20_FeatxVal_D19000_RF_StdHD_OnlHD/',
               '../01_CHBMIT/05_Predictions_1to30Hz_4_0.5_MeanAmpl-LineLength-Frequency_Fact10/personalized_LeaveOneOut/MeanAmpl-LineLength-Frequency_20_FeatxChxVal_D19000_RF_StdHD_OnlHD/',
               '../01_CHBMIT/05_Predictions_1to30Hz_4_0.5_MeanAmpl-LineLength-Frequency_Fact10/personalized_LeaveOneOut/MeanAmpl-LineLength-Frequency_20_ChxFeatxVal_D19000_RF_StdHD_OnlHD/',
               '../01_CHBMIT/05_Predictions_1to30Hz_4_0.5_MeanAmpl-LineLength-Frequency_Fact10/personalized_LeaveOneOut/MeanAmpl-LineLength-Frequency_20_ChFeatCombxVal_D19000_RF_StdHD_OnlHD/',
               '../01_CHBMIT/05_Predictions_1to30Hz_4_0.5_MeanAmpl-LineLength-Frequency_Fact10/personalized_LeaveOneOut/MeanAmpl-LineLength-Frequency_20_FeatAppend_D1000_RF_StdHD_OnlHD/']
folderInNames=['FeatxVal','FeatxChxVal','ChxFeatxVal', 'ChFeatCombxVal','FeatAppend']
outputName='D=1000_Fact10_Appended'
plot_performanceComparison_FromDifferentFolders(folderInArray, folderInNames, folderOut,  GeneralParams.patients ,outputName,  'Appended', 'StdHD')
plot_performanceComparison_FromDifferentFolders(folderInArray, folderInNames, folderOut,  GeneralParams.patients ,outputName,  'Appended', 'OnlineHD')
outputName='D=1000_Fact10_Avrg'
plot_performanceComparison_FromDifferentFolders(folderInArray, folderInNames, folderOut,  GeneralParams.patients ,outputName,  'Avrg', 'StdHD')
plot_performanceComparison_FromDifferentFolders(folderInArray, folderInNames, folderOut,  GeneralParams.patients ,outputName,  'Avrg', 'OnlineHD')
# #compare predictions in time
folderOutPredictionsInTime=folderOut+'/PredictionsInTime_D=1000'
func_plotPredictionsForDifferentBinding(folderInArray, folderInNames,GeneralParams, 'StdHD', folderOutPredictionsInTime)
func_plotPredictionsForDifferentBinding(folderInArray, folderInNames,GeneralParams, 'OnlineHD', folderOutPredictionsInTime)


###########################################################################################
#### COMPARE MEMORY AND TIME FOR DIFFERENT ENCODING APPROACHES
folderOut='../01_CHBMIT/06_Comparison_MemoryAndTime/'
createFolderIfNotExists(folderOut)
approachNames = ['FeatxVal', 'FeatxChxVal', 'ChxFeatxVal', 'ChFeatCombxVal', 'FeatAppend']
numCh=18
numFeat=19
numFeatVal=20

#CALCULATE MEMORY
memoryForInitVectors=[ numFeat+numFeatVal, numFeat+numFeatVal+ numCh, numFeat+numFeatVal+ numCh, numFeat*numCh+numFeatVal, (numFeat+ numCh+ numFeatVal)/numFeat]
memoryForInitVectors=memoryForInitVectors/np.max(memoryForInitVectors) #normalize to have relative values

#CALCULATE NUMBER OF OPERATIONS
numOperations=[numFeat*numCh, numFeat*numCh+numFeat, numFeat*numCh+numCh, numFeat*numCh, numCh ]
numOperations=numOperations/np.max(numOperations)

# PLOTTING FOR ALL SUBJ
fontSizeNum = 12
fig1 = plt.figure(figsize=(14, 2), constrained_layout=False)
gs = GridSpec(1, 2, figure=fig1)
fig1.subplots_adjust(wspace=0.15, hspace=0.15)
# fig1.suptitle('Memory and computational complexity')
xValues = np.arange(0, len(approachNames))
ax1 = fig1.add_subplot(gs[0, 0])
ax1.bar(xValues, memoryForInitVectors, color='salmon', width = 0.5)
ax1.set_xticks(xValues)
ax1.set_xticklabels(approachNames, fontsize=fontSizeNum * 0.8)
ax1.set_xlabel('Approaches')
ax1.set_ylabel('Rel. value')
ax1.set_title('Memory needed to store all vectors')
ax1.grid()
ax1 = fig1.add_subplot(gs[0, 1])
ax1.bar(xValues, numOperations, color='salmon', width = 0.5)
ax1.set_xticks(xValues)
ax1.set_xticklabels(approachNames, fontsize=fontSizeNum * 0.8)
ax1.set_xlabel('Approaches')
ax1.set_ylabel('Rel. value')
# ax1.set_title('Time for encoding and training')
ax1.set_title('Number of operations for encoding')
ax1.grid()
fig1.show()
fig1.savefig(folderOut + '/AllSubj_MemoryAndTime.png', bbox_inches='tight')
fig1.savefig(folderOut + '/AllSubj_MemoryAndTime.svg', bbox_inches='tight')
plt.close(fig1)


##########################################################################################
### CREATE TABLE WITH RESULTS ON IMPROVEMENT FOR DIFFERENT FEATURE SELECTION APROACHES
HDtype='OnlineHD'
folderList=['Voting_FeatSel_FeatPerformance_PerFeat_'+HDtype+'/ThrOptimization_F1E_noSmooth',
            'Voting_FeatSel_FeatPerformance_PerFeat_'+HDtype+'/ThrOptimization_F1DE_noSmooth',
            'Voting_FeatSel_FeatConfidence_PerFeat_'+HDtype+'/ThrOptimization_F1E_noSmooth',
            'Voting_FeatSel_FeatConfidence_PerFeat_'+HDtype+'/ThrOptimization_F1DE_noSmooth',
            'Voting_FeatSel_OptimalFeatOrder_PerFeat_'+HDtype+'/ThrOptimization_F1E_noSmooth',
            'Voting_FeatSel_OptimalFeatOrder_PerFeat_'+HDtype+'/ThrOptimization_F1DE_noSmooth']

for foldIndx, fold in enumerate(folderList):
    inputName = folderOutPredictions+ fold + '/AllSubj_' + HDtype + '_OptimalThresholds.csv'
    optThr_allSubj=readDataFromFile(inputName)
    inputName = folderOutPredictions+ fold +  '/AllSubj_' + HDtype + '_PerfImprovTrain.csv'
    perfImprovTrain_allSubj=readDataFromFile(inputName)
    perfImprovTrain_allSubj_mean=np.nanmean(perfImprovTrain_allSubj,0)
    inputName = folderOutPredictions+ fold +  '/AllSubj_' + HDtype + '_PerfImprovTest.csv'
    perfImprovTest_allSubj=readDataFromFile(inputName)
    perfImprovTest_allSubj_mean = np.nanmean(perfImprovTest_allSubj, 0)

    # dataLine=np.hstack(( [np.mean(optThr_allSubj[:,0])] ,  100*perfImprovTrain_allSubj_mean[[2,7,8,18+2,18+7,18+8,27+2,27+7,27+8]], 100*perfImprovTest_allSubj_mean[[2,7,8,18+2,18+7,18+8,27+2,27+7,27+8]] ))
    dataLine = np.hstack(([np.mean(optThr_allSubj[0,:])], 100 * perfImprovTrain_allSubj_mean[[2, 7, 18 + 2, 18 + 7]], 100*perfImprovTest_allSubj_mean[[2, 7, 18 + 2, 18 + 7]] ))

    if (foldIndx==0):
        table=dataLine
    else:
        table=np.vstack((table, dataLine))

outputName = folderOutPredictions + '/FeatSelection_PerfImprovmentsTable.csv'
saveDataToFile(table, outputName, '.csv')
outputName = folderOutPredictions + '/FeatSelection_PerfImprovmentsTable.txt'
np.savetxt(outputName, table, delimiter=",", fmt='%.2f')