__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

'''
- performs different feature selection 
	- based on parameter VotingParam.approach 
	- options are: FeatPerformance, FeatConfidence and OptimalFeatOrder
- outputs predictions and performance when adding one by one feature (or certain perfcentages of features) that are sorted based on VotingParam.approach from above 
- plots:
	- performance when increasing number of features (average over all subj)
	- optimal order of features (average over all subj)
	- optimal number of features chosen per subject and performance improvement (drop) with that number of features vs when using all features  
'''

from HDfunctionsLib import *
from parametersSetup import *

###################################
# SETUP DATASET USED
# CHBMIT
Dataset = '01_CHBMIT'  # '01_CHBMIT', '01_iEEG_Bern'
GeneralParams.patients = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
# create folder for dataset if it doesnt exist
createFolderIfNotExists('../' + Dataset)

# PARAMETERS
GeneralParams.PersGenApproach = 'personalized'  # 'personalized', 'generalized'
datasetFiltering = '1to30Hz'  # 'Raw', 'MoreNonSeizure_Fact10' #depends if we want all data or some subselection
GeneralParams.CVtype = 'LeaveOneOut'  # 'LeaveOneOut', 'RollingBase'
datasetPreparationType = 'Fact10'  # 'Fact1', 'Fact10' ,'AllDataWin3600s', 'AllDataWin3600s_1File6h', 'AllDataStoS'  # !!!!!!
HDtype = 'OnlineHD'  # 'StdHD', 'OnlineHD'

# FEATURE SELECTION PARAMETERS - IMPORTANT!!!
class VotingParam:
    approach = 'FeatPerformance' #'FeatConfidence', 'FeatPerformance', 'OptimalFeatOrder'
    selectionStyle='PerFeat' #'Perc', 'PerFeat'
    confThreshArray = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    confThreshArray = [0.05, 0.1, 0.15, 0.2, 0.25,  0.3, 0.35,  0.4, 0.45,  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

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
folderOutFeatures = '../' + Dataset + '/02_Features_' + datasetFiltering + '_' + str(FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep)  # where to save calculated features for each original file
createFolderIfNotExists(folderOutFeatures)
folderOutRearangedData = '../' + Dataset + '/04_RearangedData_MergedFeatures_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType

# CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
folderOutPredictions0 = '../' + Dataset + '/05_Predictions_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType
createFolderIfNotExists(folderOutPredictions0)
if (HDParams.HDvecType == 'bin'):
    folderOutPredictions0 = folderOutPredictions0 + '/' + str( GeneralParams.PersGenApproach) + '_' + GeneralParams.CVtype
elif (HDParams.HDvecType == 'bipol'):
    folderOutPredictions0 = folderOutPredictions0 + '/' + str(GeneralParams.PersGenApproach) + '_' + GeneralParams.CVtype + '_bipolarVec/'
createFolderIfNotExists(folderOutPredictions0)

# FOLDER FOR DIVERGENCE
folderOutFeatDiffMeas= '../' + Dataset + '/05_Predictions_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType
createFolderIfNotExists(folderOutFeatDiffMeas)
folderOutFeatDiverg = folderOutFeatDiffMeas + '/FeatureDivergence/'
createFolderIfNotExists(folderOutFeatDiverg)

# ## CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
folderOutParams = FeaturesUsed + '_' + str(HDParams.numSegmentationLevels) + '_' + HDParams.bindingFeatures + '_D' + str( HDParams.D)  # not all parameters are saved here (but coudl be if needed)
folderOutPredictions = folderOutPredictions0 + '/' + folderOutParams + '_RF_StdHD_OnlHD/'
createFolderIfNotExists(folderOutPredictions)
folderOutPredictionsPlot = folderOutPredictions + '/PredictionsComparison/'
createFolderIfNotExists(folderOutPredictionsPlot)
if (HDParams.bindingFeatures == 'FeatAppend'):
    folderOutPredictionsPerFeat = folderOutPredictions + '/PerFeat/'
    createFolderIfNotExists(folderOutPredictionsPerFeat)
    # OUTPUT FOR THIS VOTING TYPE
    folderOutPredictionsVoting = folderOutPredictions + '/Voting_FeatSel_' + VotingParam.approach+ '_'+VotingParam.selectionStyle+'_'+HDtype + '/'
    createFolderIfNotExists(folderOutPredictionsVoting)

if (VotingParam.selectionStyle=='Perc'):
    numThr = len(VotingParam.confThreshArray)
else:
    numThr=len(FeaturesParams.featNames)
# feature indexes to keep
HDParams.numFeat = len(FeaturesParams.featNames)

##############################################################
## FEATURE SELECTION
# various postpocessing parameters
seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
toleranceFP_bef = int(PostprocessingParams.toleranceFP_befSeiz / FeaturesParams.winStep)
toleranceFP_aft = int(PostprocessingParams.toleranceFP_aftSeiz / FeaturesParams.winStep)

# distSeparability_AllSubj = np.zeros((len(GeneralParams.patients), HDParams.numFeat + 1))
featureOrders_allSubj=np.zeros((len(GeneralParams.patients), HDParams.numFeat))
performancesAll_Train_AllSubj = np.zeros((numThr + 1, 9, len(GeneralParams.patients)))
performancesAll_Test_AllSubj = np.zeros((numThr + 1, 9, len(GeneralParams.patients)))
# if GeneralParams.PersGenApproach == 'personalized':
for patIndx, pat in enumerate(GeneralParams.patients):
    filesAll = np.sort(glob.glob(folderOutPredictionsPerFeat + '/*Subj' + pat + '*' + HDtype + '_PerFeat_TrainPredictions.csv.gz'))
    numFiles = len(filesAll)
    print('-- Patient:', pat, 'NumSeizures:', numFiles)

    # distSeparability_ThisSubj = np.zeros((numFiles, HDParams.numFeat + 1))
    featureOrders_ThisSubj = np.zeros((numFiles, HDParams.numFeat ))
    performancesAll_Train_ThisSubj = np.zeros((numThr + 1, 36, numFiles))
    performancesAll_Test_ThisSubj = np.zeros((numThr + 1, 36, numFiles))

    # load all files only once and mark where each file starts
    (dataAll, labelsAll, startIndxOfFiles) = concatenateDataFromFiles(filesAll)

    #GO THROUGH ALL CROSS-VALIDATIONS
    for cv in range(numFiles):
        filesToTestOn = list(filesAll[cv].split(" "))
        pom, fileName1 = os.path.split(filesToTestOn[0])
        fileName2 = fileName1[0:-32]

        # load predictions
        fileInName = folderOutPredictionsPerFeat + '/' + fileName2 + '_PerFeat_TrainPredictions.csv.gz'
        data0 = readDataFromFile(fileInName)
        predLabels_train = data0[:, 0:-1]
        label_train = data0[:, -1]
        fileInName = folderOutPredictionsPerFeat + '/' + fileName2 + '_PerFeat_TestPredictions.csv.gz'
        data0 = readDataFromFile(fileInName)
        predLabels_test = data0[:, 0:-1]
        label_test = data0[:, -1]

        # load distances from seiz and non seiz class
        fileInName = folderOutPredictionsPerFeat + '/' + fileName2 + '_PerFeat_DistancesFromSeiz_Train.csv.gz'
        distFromS_train = readDataFromFile(fileInName)
        fileInName = folderOutPredictionsPerFeat + '/' + fileName2 + '_PerFeat_DistancesFromNonSeiz_Train.csv.gz'
        distFromNS_train = readDataFromFile(fileInName)
        fileInName = folderOutPredictionsPerFeat + '/' + fileName2 + '_PerFeat_DistancesFromSeiz_Test.csv.gz'
        distFromS_test = readDataFromFile(fileInName)
        fileInName = folderOutPredictionsPerFeat + '/' + fileName2 + '_PerFeat_DistancesFromNonSeiz_Test.csv.gz'
        distFromNS_test = readDataFromFile(fileInName)

        # calculate probabilities per feature
        probabilityLabels_train = calculateProbabilitiesPerFeature(distFromS_train, distFromNS_train, predLabels_train)
        probabilityLabels_test = calculateProbabilitiesPerFeature(distFromS_test, distFromNS_test, predLabels_test)


        #measure of quality of features - confidence, performance of Kl value
        if ( VotingParam.approach=='FeatConfidence'):
            fileInName = folderOutPredictionsPerFeat + '/' + fileName2 + '_PerFeat_TrainConfidence.csv.gz'
            featureQualityMeasure = readDataFromFile(fileInName)
            # featureQualityMeasure=np.mean(np.abs(featureQualityMeasure),0)
            #calculate confidence only when correctly voting
            featureQualityMeasure=calculateConfOnlyWhenCorreclyVoting(featureQualityMeasure, label_train)
        elif ( VotingParam.approach=='FeatPerformance'):
            fileInName = folderOutPredictionsPerFeat + '/' + fileName2 + '_PerFeat_TrainPerformance.csv.gz'
            featureQualityMeasure = readDataFromFile(fileInName)
            featureQualityMeasure=featureQualityMeasure[0:-1,7] #2 for F1E, 7 for F1DE

        elif (VotingParam.approach=='OptimalFeatOrder'):
            fileInName = folderOutFeatDiffMeas + '/FeatureOptimalOrder_'+HDtype+'_F1DE/' + fileName2 + '_TrainOptimalFeatureOrdering.csv.gz'
            featureQualityMeasure = readDataFromFile(fileInName)[1,:]
            featureQualityMeasure=np.abs(featureQualityMeasure-HDParams.numFeat) #to reverse that bigger number means better

        #feature order
        sortIndx=np.argsort(-featureQualityMeasure)
        orderedIndx=np.arange(1,HDParams.numFeat+1,1)
        np.put(featureOrders_ThisSubj[cv,:], sortIndx, orderedIndx)

        # GET FINAL PREDICTION FROM INDIVIDUAL CH PREDICTIONS
        predictionsFinal_allThresh_test = givePredictionsBasedOnqualityOfEachFeature_moreThresh(predLabels_test, distFromS_test, distFromNS_test,    featureQualityMeasure,VotingParam)
        predictionsFinal_allThresh_train = givePredictionsBasedOnqualityOfEachFeature_moreThresh(predLabels_train,  distFromS_train, distFromNS_train,featureQualityMeasure,VotingParam)

        performancesAll_Train = np.ones((numThr + 1, 36)) * np.nan
        performancesAll_Test = np.ones((numThr + 1, 36)) * np.nan
        thrNames = []
        for th in range(numThr):
            if (VotingParam.selectionStyle=='Perc'):
                thrNames.append('Thr=' + str(VotingParam.confThreshArray[th]))
            else:
                thrNames.append(str(th+1))
            (performancesAll_Train[th, :], _, _, _) = calculatePerformanceAfterVariousSmoothing(predictionsFinal_allThresh_train[:, th], label_train,probabilityLabels_train[:, th],
                                                                                   toleranceFP_bef, toleranceFP_aft, numLabelsPerHour,
                                                                                   seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
            (performancesAll_Test[th, :], _, _, _) = calculatePerformanceAfterVariousSmoothing(predictionsFinal_allThresh_test[:, th], label_test,probabilityLabels_test[:, th],
                                                                                   toleranceFP_bef, toleranceFP_aft, numLabelsPerHour,
                                                                                   seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)

        # add as the last performance with whole vector used
        (performancesAll_Train[th+ 1, :], _, _, _) = calculatePerformanceAfterVariousSmoothing(predLabels_train[:, -1], label_train, probabilityLabels_train[:, -1],
            toleranceFP_bef, toleranceFP_aft, numLabelsPerHour,seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx,PostprocessingParams.bayesProbThresh)
        (performancesAll_Test[th+ 1, :], _, _, _) = calculatePerformanceAfterVariousSmoothing(predLabels_test[:, -1], label_test, probabilityLabels_test[:, -1],
            toleranceFP_bef, toleranceFP_aft, numLabelsPerHour, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
        thrNames.append('WholeVec')

        # saving
        outputName = folderOutPredictionsVoting + '/' + fileName2 + '_PerformancePerFeat_Train.csv'
        saveDataToFile(performancesAll_Train, outputName, 'gzip')
        outputName = folderOutPredictionsVoting + '/' + fileName2 + '_PerformancePerFeat_Test.csv'
        saveDataToFile(performancesAll_Test, outputName, 'gzip')

        # SAVING TO CALCULATE AVRG FOR THIS SUBJ
        performancesAll_Train_ThisSubj[:, :, cv] = performancesAll_Train
        performancesAll_Test_ThisSubj[:, :, cv] = performancesAll_Test

    # save for this subj
    dataToSave = np.zeros((numThr + 1, 72))
    dataToSave[:, 0:36] = np.nanmean(performancesAll_Train_ThisSubj, 2)
    dataToSave[:, 36:] = np.nanstd(performancesAll_Train_ThisSubj, 2)
    outputName = folderOutPredictionsVoting + '/Subj' + pat + '_' + HDtype + '_PerformancePerFeat_Train.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')
    dataToSave = np.zeros((numThr + 1, 72))
    dataToSave[:, 0:36] = np.nanmean(performancesAll_Test_ThisSubj, 2)
    dataToSave[:, 36:] = np.nanstd(performancesAll_Test_ThisSubj, 2)
    outputName = folderOutPredictionsVoting + '/Subj' + pat + '_' + HDtype + '_PerformancePerFeat_Test.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')

    outputName = folderOutPredictionsVoting + '/Subj' + pat + '_' + HDtype + '_FeatureOrders.csv'
    saveDataToFile(featureOrders_ThisSubj, outputName, 'gzip')


###########################################################################################
## CALCULATE AND PLOT FOR EACH SUBJECT AND AVERAGES
# GeneralParams.patients  =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

thrNames = []
for th in range(numThr):
    if (VotingParam.selectionStyle == 'Perc'):
        thrNames.append('Thr=' + str(VotingParam.confThreshArray[th]))
    else:
        thrNames.append(str(th + 1))
thrNames.append('WholeVec')
performancesAll_Train_AllSubj = np.zeros((numThr + 1, 36, len(GeneralParams.patients)))
performancesAll_Test_AllSubj = np.zeros((numThr + 1, 36, len(GeneralParams.patients)))
featureOrders_allSubj=np.zeros((len(GeneralParams.patients), HDParams.numFeat))
# if GeneralParams.PersGenApproach == 'personalized':
for patIndx, pat in enumerate(GeneralParams.patients):
    filesAll = np.sort( glob.glob(folderOutPredictionsPerFeat + '/*Subj' + pat + '*' + HDtype + '_PerFeat_TrainPredictions.csv.gz'))
    numFiles = len(filesAll)
    print('-- Patient:', pat, 'NumSeizures:', numFiles)
    fileInName = folderOutPredictionsVoting + '/Subj' + pat + '_' + HDtype + '_PerformancePerFeat_Train.csv'
    performancesAll_Train_AllSubj[:, :, patIndx] = readDataFromFile(fileInName)[:, 0:36]
    fileInName = folderOutPredictionsVoting + '/Subj' + pat + '_' + HDtype + '_PerformancePerFeat_Test.csv'
    performancesAll_Test_AllSubj[:, :, patIndx] = readDataFromFile(fileInName)[:, 0:36]
    fileInName = folderOutPredictionsVoting + '/Subj' + pat + '_' + HDtype + '_FeatureOrders.csv'
    featureOrders=readDataFromFile(fileInName)
    featureOrders_allSubj[patIndx,:]=np.mean(featureOrders,0)

# save avrg of all subj  subj
dataToSave = np.zeros((numThr + 1, 72))
dataToSave[:, 0:36] = np.nanmean(performancesAll_Train_AllSubj, 2)
dataToSave[:, 36:] = np.nanstd(performancesAll_Train_AllSubj, 2)
outputName = folderOutPredictionsVoting + '/AllSubj_' + HDtype + '_PerformancePerFeat_Train.csv'
saveDataToFile(dataToSave, outputName, 'gzip')
dataToSave = np.zeros((numThr + 1, 72))
dataToSave[:, 0:36] = np.nanmean(performancesAll_Test_AllSubj, 2)
dataToSave[:, 36:] = np.nanstd(performancesAll_Test_AllSubj, 2)
outputName = folderOutPredictionsVoting + '/AllSubj_' + HDtype + '_PerformancePerFeat_Test.csv'
saveDataToFile(dataToSave, outputName, 'gzip')

outputName = folderOutPredictionsVoting + '/AllSubj_' + HDtype + '_FeatureOrders.csv'
saveDataToFile(featureOrders_allSubj, outputName, 'gzip')


##CONDENSED PLOT FOR PAPER VERSION 2 - performance and feature order on the same plot
sindx=0
fig1 = plt.figure(figsize=(20, 3), constrained_layout=False)
gs = GridSpec(1, 3, figure=fig1)
fig1.subplots_adjust(wspace=0.15, hspace=0.15)
xValues = np.arange(sindx, numThr , 1)
perfNames = ['F1score episodes',  'F1DEgmean']
perfIndxs=[2,7]
numPerf = len(perfNames)
for perfIndx, perf in enumerate(perfIndxs):
    ax1 = fig1.add_subplot(gs[0, perfIndx])
    ax1.errorbar(xValues, np.nanmean(performancesAll_Train_AllSubj[sindx:-1, perf, :], 1),
                 yerr=np.nanstd(performancesAll_Train_AllSubj[sindx:-1, perf, :], 1), color='k')
    ax1.errorbar(xValues, np.nanmean(performancesAll_Train_AllSubj[sindx:-1, 18+perf, :], 1),
                 yerr=np.nanstd(performancesAll_Train_AllSubj[sindx:-1, 18+perf, :], 1), color='coral')
    ax1.legend(['Raw', 'Postprocessed'])
    ax1.errorbar(xValues, np.nanmean(performancesAll_Test_AllSubj[sindx:-1, perf, :], 1),
                 yerr=np.nanstd(performancesAll_Test_AllSubj[sindx:-1, perf, :], 1), color='k', linestyle='--')
    ax1.errorbar(xValues, np.nanmean(performancesAll_Test_AllSubj[sindx:-1, 18+perf, :], 1),
                 yerr=np.nanstd(performancesAll_Test_AllSubj[sindx:-1, 18+perf, :], 1), color='coral', linestyle='--')
    ax1.set_xticks(xValues)
    ax1.set_ylim([0.2,1.1])
    if (VotingParam.selectionStyle == 'Perc'):
        ax1.set_xticklabels(thrNames[sindx:-1], fontsize=12 * 0.8, rotation=45)
    else:
        ax1.set_xticklabels(thrNames[sindx:-1], fontsize=12 * 0.8)
        ax1.set_xlabel('Number of features', fontsize=12)
    ax1.set_title(perfNames[perfIndx])
    ax1.grid()
xValues = np.arange(1, HDParams.numFeat +1, 1)
ax1 = fig1.add_subplot(gs[0, 2])
ax1.boxplot(featureOrders_allSubj, medianprops=dict(color='red', linewidth=1),
                boxprops=dict(linewidth=1), capprops=dict(linewidth=1), whiskerprops=dict(linewidth=1), showfliers=False)
ax1.set_title('Order of feature per importance')
ax1.grid()
ax1.set_xticks(xValues)
ax1.set_xticklabels(FeaturesParams.featNames, fontsize=11 * 0.8, rotation=45)
# ax1.grid()
fig1.show()
fig1.savefig(folderOutPredictionsVoting + '/AllSubj_' + HDtype + '_PredictionsWithVoting_forPaper2.png', bbox_inches='tight')
fig1.savefig(folderOutPredictionsVoting + '/AllSubj_' + HDtype + '_PredictionsWithVoting_forPaper2.svg', bbox_inches='tight')
plt.close(fig1)


#################
# CALCULATE OPTIMAL THRESHOLD BASED ON TRAIN AND MEASURE TEST PERFORMANCE FOR IT
PerfOptimized='F1DE_noSmooth'
PerfOptimizedIndx=7

folderOutPredictionsVotingOptim = folderOutPredictionsVoting + '/ThrOptimization_'+PerfOptimized
createFolderIfNotExists(folderOutPredictionsVotingOptim)
optThr_allSubj = np.zeros((2, len(GeneralParams.patients)))
perfImprovTrain_allSubj = np.zeros((len(GeneralParams.patients), 36, 2))
perfImprovTest_allSubj = np.zeros((len(GeneralParams.patients), 36, 2))

for patIndx, pat in enumerate(GeneralParams.patients):
    filesAll = np.sort(glob.glob( folderOutPredictionsVoting + '/*Subj' + pat + '_cv*' + HDtype + '_PerformancePerFeat_Train.csv.gz'))
    optThr = np.zeros((len(filesAll)))
    perfImprovTrain = np.zeros((len(filesAll), 36))
    perfImprovTest = np.zeros((len(filesAll), 36))
    for fIndx in range(len(filesAll)):
        performance_train = readDataFromFile(filesAll[fIndx])#[:, 0:9]
        performance_test = readDataFromFile(filesAll[fIndx][0:-12] + 'Test.csv.gz')#[:, 0:9]
        pom, fileName1 = os.path.split(filesAll[fIndx])
        fileName2 = os.path.splitext(fileName1)[0][0:-38]

        # find max perf on train
        # maxIndx = np.argmax(performance_train[0:-1, PerfOptimizedIndx])  # F1E #not taking last into account as it is perf with the whole vector
        maxPerf= np.max(performance_train[0:-1, PerfOptimizedIndx])
        maxIndx=np.where(performance_train[0:-1, PerfOptimizedIndx] > maxPerf-0.01)[0][0]
        optPerfTrain = np.vstack((performance_train[maxIndx, :], performance_train[-1, :]))  # last is when using whole vector
        optPerfTest = np.vstack( (performance_test[maxIndx, :], performance_test[-1, :]))  # last is when using whole vector
        outputName = folderOutPredictionsVotingOptim + '/' + fileName2 + '_OptimalTrainAndTestPerformance.csv'
        saveDataToFile(np.vstack((optPerfTrain, optPerfTest)), outputName, 'gzip')
        if (VotingParam.selectionStyle=='Perc'):
            optThr[fIndx] = VotingParam.confThreshArray[maxIndx]
        else:
            optThr[fIndx] = maxIndx+1 #numFeatures
        perfImprovTrain[fIndx, :] = performance_train[maxIndx, :] - performance_train[-1, :]
        perfImprovTest[fIndx, :] = performance_test[maxIndx, :] - performance_test[-1, :]

    outputName = folderOutPredictionsVotingOptim + '/Subj' + pat + '_' + HDtype + '_OptimalThresholds.csv'
    saveDataToFile(optThr, outputName, 'gzip')
    outputName = folderOutPredictionsVotingOptim + '/Subj' + pat + '_' + HDtype + '_PerfImprovTrain.csv'
    saveDataToFile(perfImprovTrain, outputName, 'gzip')
    outputName = folderOutPredictionsVotingOptim + '/Subj' + pat + '_' + HDtype + '_PerfImprovTest.csv'
    saveDataToFile(perfImprovTest, outputName, 'gzip')

    # mean for all subj
    optThr_allSubj[:, patIndx] = [np.mean(optThr), np.std(optThr)]
    perfImprovTrain_allSubj[patIndx, :, 0] = np.mean(perfImprovTrain, 0)
    perfImprovTrain_allSubj[patIndx, :, 1] = np.std(perfImprovTrain, 0)
    perfImprovTest_allSubj[patIndx, :, 0] = np.mean(perfImprovTest, 0)
    perfImprovTest_allSubj[patIndx, :, 1] = np.std(perfImprovTest, 0)

outputName = folderOutPredictionsVotingOptim + '/AllSubj_' + HDtype + '_OptimalThresholds.csv'
saveDataToFile(optThr_allSubj, outputName, 'gzip')
outputName = folderOutPredictionsVotingOptim + '/AllSubj_' + HDtype + '_PerfImprovTrain.csv'
saveDataToFile(perfImprovTrain_allSubj[:, :, 0], outputName, 'gzip')
outputName = folderOutPredictionsVotingOptim + '/AllSubj_' + HDtype + '_PerfImprovTest.csv'
saveDataToFile(perfImprovTest_allSubj[:, :, 0], outputName, 'gzip')

# Plotting
fig1 = plt.figure(figsize=(20,2), constrained_layout=False) #5,10
gs = GridSpec(1, 3, figure=fig1)
fig1.subplots_adjust(wspace=0.3, hspace=0.3)
# fig1.suptitle('Comparing features '+ fileName2)
xValues = np.arange(1, len(GeneralParams.patients) + 1, 1)
ax1 = fig1.add_subplot(gs[0, 0])
ax1.errorbar(xValues, optThr_allSubj[0, :], yerr=optThr_allSubj[1, :], fmt='k', linestyle='None', marker='D', markersize=5.0)
ax1.plot(xValues, np.ones(len(xValues)) * np.mean(optThr_allSubj[0, :]), 'k--', linewidth=2)
ax1.set_xlabel('Subjects', fontsize=12)
ax1.set_xticks(xValues)
ax1.set_xticklabels(xValues.astype(int), fontsize=12 * 0.8)
ax1.set_title('Optimal threshold, mean='+'{:.2f}'.format(np.mean(optThr_allSubj[0, :])))
ax1.grid()
ax1 = fig1.add_subplot(gs[0, 1])
ax1.errorbar(xValues, perfImprovTrain_allSubj[:, 2, 0], yerr=perfImprovTrain_allSubj[:, 2, 1], fmt='k', linestyle='None',  marker='D',markersize=5.0, )
ax1.errorbar(xValues, perfImprovTest_allSubj[:, 2, 0], yerr=perfImprovTest_allSubj[:, 2, 1], color='orangered', linestyle='None', marker='D', markersize=5.0)
ax1.legend(['Train', 'Test'])
ax1.plot(xValues, np.ones(len(xValues)) * np.mean(perfImprovTrain_allSubj[:, 2, 0]), 'k--', linewidth=2)
ax1.plot(xValues, np.ones(len(xValues)) * np.mean(perfImprovTest_allSubj[:, 2, 0]), color='orangered', linestyle='--', linewidth=2)
ax1.set_xlabel('Subjects', fontsize=12)
ax1.set_xticks(xValues)
ax1.set_xticklabels(xValues.astype(int), fontsize=12 * 0.8)
ax1.set_title('F1E performance improvement, mean='+'{:.2f}'.format(100*np.mean(perfImprovTest_allSubj[:, 2, 0]) )+'%')
ax1.grid()
ax1 = fig1.add_subplot(gs[0, 2])
ax1.errorbar(xValues, perfImprovTrain_allSubj[:, 2, 0], yerr=perfImprovTrain_allSubj[:, 7, 1], fmt='k', linestyle='None', marker='D', markersize=5.0)
ax1.errorbar(xValues, perfImprovTest_allSubj[:, 2, 0], yerr=perfImprovTest_allSubj[:, 7, 1], color='orangered', linestyle='None', marker='D',markersize=5.0)
ax1.legend(['Train', 'Test'])
ax1.plot(xValues, np.ones(len(xValues)) * np.mean(perfImprovTrain_allSubj[:, 7, 0]), 'k--', linewidth=2)
ax1.plot(xValues, np.ones(len(xValues)) * np.mean(perfImprovTest_allSubj[:, 7, 0]), color='orangered', linestyle='--', linewidth=2)
ax1.set_xlabel('Subjects', fontsize=12)
ax1.set_xticks(xValues)
ax1.set_xticklabels(xValues.astype(int), fontsize=12 * 0.8)
ax1.set_title('F1DE performance improvement, mean='+'{:.2f}'.format(100*np.mean(perfImprovTest_allSubj[:, 7, 0])) +'%')
ax1.grid()
# ax1 = fig1.add_subplot(gs[0, 3])
# ax1.errorbar(xValues, perfImprovTrain_allSubj[:, 18+8, 0], yerr=perfImprovTrain_allSubj[:, 18+8, 1], fmt='k', linestyle='None',  marker='D',markersize=5.0)
# ax1.errorbar(xValues, perfImprovTest_allSubj[:, 18+8, 0], yerr=perfImprovTest_allSubj[:, 18+8, 1], color='orangered', linestyle='None', marker='D', markersize=5.0)
# ax1.legend(['Train', 'Test'])
# ax1.plot(xValues, np.ones(len(xValues)) * np.mean(perfImprovTrain_allSubj[:, 18+8, 0]), 'k--', linewidth=2)
# ax1.plot(xValues, np.ones(len(xValues)) * np.mean(perfImprovTest_allSubj[:, 18+8, 0]), color='orangered', linestyle='--', linewidth=2)
# ax1.set_xlabel('Subjects')
# ax1.set_xticks(xValues)
# ax1.set_xticklabels(xValues.astype(int), fontsize=12 * 0.8)
# ax1.set_title('NumFP improvement, mean='+'{:.2f}'.format(np.mean(perfImprovTest_allSubj[:, 18+8, 0]) ))
# ax1.grid()
fig1.show()
fig1.savefig(folderOutPredictionsVotingOptim + '/AllSubj_' + HDtype + '_OptimizedThreshold_Performances.png', bbox_inches='tight')
fig1.savefig(folderOutPredictionsVotingOptim + '/AllSubj_' + HDtype + '_OptimizedThreshold_Performances.svg', bbox_inches='tight')
plt.close(fig1)
