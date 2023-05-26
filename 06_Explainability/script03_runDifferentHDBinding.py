__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

''' 
script that runs HD training using different encoding approaches
- encoding is defined in variable HDParams.bindingFeatures
- for every encoding it runs standard HD (StdHD), online HD (OnlineHD) and random forest (RD) approach
- it saves predictions in time for each cross-validation
- for FeatAppend approach it also saves predictions per feature in time (as well as distances from classes etc)
- later it is used to also analyse feature qualities
    - separability of model vectors for seizure and non-seizure class of each feature
    - average confidences per feature
    - average performance per feature
    - correlation between those values 
- uses files that are prepared using script_prepareDatasets
    - with parameter "datasetPreparationType" possible to choose which dataset to use
 - possible to perform training on 2 ways
    - leave one file out - train on all but that one (this doens keep time information into account - that some data was before another one) 
    - rolling base approach - uses all previous files to train for the current file and tests on current file    

'''

from HDfunctionsLib import *
from parametersSetup import *

#########################################################################
###################################
# SETUP DATASET USED
# CHBMIT
Dataset = '01_CHBMIT'  # '01_CHBMIT', '01_iEEG_Bern'
GeneralParams.patients = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
createFolderIfNotExists('../' + Dataset)

# PARAMETERS
GeneralParams.PersGenApproach = 'personalized'  # 'personalized', 'generalized'
datasetFiltering = '1to30Hz'  # 'Raw', 'MoreNonSeizure_Fact10' #depends if we want all data or some subselection
GeneralParams.CVtype = 'LeaveOneOut'  # 'LeaveOneOut', 'RollingBase'
datasetPreparationType = 'Fact10'  # 'Fact1', 'Fact10' ,'AllDataWin3600s', 'AllDataWin3600s_1File6h', 'AllDataStoS'  # !!!!!!

# features used
FeaturesParams.featNames = np.array(['MeanAmpl', 'LineLength', 'Frequency'])
FeaturesUsed = '-'.join(FeaturesParams.featNames)
AllExistingFeatures = FeaturesParams.allFeatTypesName
FeaturesParams.featNames = constructAllfeatNames(FeaturesParams)  # feat used here
FeaturesParams.featNames=  ['mean_ampl', 'line_length','p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot'] #to use less features
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

# IMPORTANT PARAMETERS!!!! - PARAMETERS THAT DEFINE HD ENCODING
# for feature selection choose HDParams.bindingFeatures = 'ChAppend', and for channel selection choose HDParams.bindingFeatures = 'FeatAppend'
# if one of appending approaches is chosen then D defines D per ch/feat, not total vector D
HDParams.D = 1000 #1000 for FeatAppend and 19000 for all others
HDParams.bindingFeatures = 'FeatAppend'  # 'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend', 'ChAppend'
#sets various names to be able to use the same script for channel and feature selection
if (HDParams.bindingFeatures == 'FeatAppend'):
    FeatChType='PerFeat'
    elemNames= FeaturesParams.featNames
else:  #'ChAppend'
    FeatChType='PerCh'
    elemNames =DatasetPreprocessParams.channelNamesToKeep


# ##################################################################
# DEFINING INPUT/OUTPUT FOLDERS
folderOutRearangedData = '../' + Dataset + '/04_RearangedData_MergedFeatures_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str( FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType

# CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
folderOutPredictions0 = '../' + Dataset + '/05_Predictions_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str( FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType # +'-ALLFEAT'
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
elif (HDParams.bindingFeatures == 'ChAppend'):
    folderOutPredictionsPerFeat = folderOutPredictions + '/PerCh/'
    createFolderIfNotExists(folderOutPredictionsPerFeat)

# #################################################################
# TRAINING
# feature indexes to keep
HDParams.numFeat = len(FeaturesParams.featNames)
featIndxs = np.zeros((len(FeaturesParams.featNames)))
for f in range(len(FeaturesParams.featNames)):
    featIndxs[f] = int(np.where(np.asarray(FeaturesParams.allFeatNames) == FeaturesParams.featNames[f])[0][0])
featIndxs = featIndxs.astype(int)

# various postpocessing parameters
seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
toleranceFP_bef = int(PostprocessingParams.toleranceFP_befSeiz / FeaturesParams.winStep)
toleranceFP_aft = int(PostprocessingParams.toleranceFP_aftSeiz / FeaturesParams.winStep)


## TRAINING ON ONE BY ONE SUBJECT
AllSubjTimes = np.zeros((len(GeneralParams.patients), 3, 2))  # 3 approaches
for patIndx, pat in enumerate(GeneralParams.patients):
    filesAll = np.sort(glob.glob(folderOutRearangedData + '/*Subj' + pat + '*.csv.gz'))
    print('-- Patient:', pat, 'NumSeizures:', len(filesAll))

    # load all files only once and mark where each file starts
    (dataAll, labelsAll, startIndxOfFiles) = concatenateDataFromFiles(filesAll)

    # remove nan and inf from matrix
    dataAll[np.where(np.isinf(dataAll))] = np.nan
    col_mean = np.nanmean(dataAll, axis=0)
    inds = np.where(np.isnan(dataAll))
    dataAll[inds] = np.take(col_mean, inds[1])
    # if still somewhere nan replace with 0
    dataAll[np.where(np.isnan(dataAll))] = 0

    # keep only features of interest
    numCh = round(len(dataAll[0, :]) / totalNumFeat)
    HDParams.numCh=numCh
    featIndxsAllCh = []
    for ch in range(numCh):
        featIndxsAllCh = np.hstack((featIndxsAllCh, featIndxs + ch * totalNumFeat))
    featIndxsAllCh = featIndxsAllCh.astype(int)
    dataAll = dataAll[:, featIndxsAllCh]

    if (GeneralParams.CVtype == 'LeaveOneOut'):
        numCV = len(filesAll)
    else:  # rolling base
        numCV = len(filesAll) - 1
    ThisSubjTimes = np.zeros((numCV, 3))  # 10 approaches

    # RUN FOR AL CROSS-VALIDATIONS
    for cv in range(numCV):
        fileName2 = 'Subj' + pat + '_cv' + str(cv).zfill(3)

        # create train and test data - LEAVE ONE OUT
        if (GeneralParams.CVtype == 'LeaveOneOut'):
            if (cv == 0):
                dataTest = dataAll[0:startIndxOfFiles[cv], :]
                label_test = labelsAll[0:startIndxOfFiles[cv]]
                dataTrain = dataAll[startIndxOfFiles[cv]:, :]
                label_train = labelsAll[startIndxOfFiles[cv]:]
                dataSource_test = (cv + 1) * np.ones((startIndxOfFiles[cv]))  # value 1 means then file after the train one
            else:
                dataTest = dataAll[startIndxOfFiles[cv - 1]:startIndxOfFiles[cv], :]
                label_test = labelsAll[startIndxOfFiles[cv - 1]:startIndxOfFiles[cv]]
                dataTrain = dataAll[0:startIndxOfFiles[cv - 1], :]
                label_train = labelsAll[0:startIndxOfFiles[cv - 1]]
                dataTrain = np.vstack((dataTrain, dataAll[startIndxOfFiles[cv]:, :]))
                label_train = np.hstack((label_train, labelsAll[startIndxOfFiles[cv]:]))
                dataSource_test = (cv + 1) * np.ones( (startIndxOfFiles[cv] - startIndxOfFiles[cv - 1]))  # value 1 means then file after the train one
        else:  # RollingBase
            dataTest = dataAll[startIndxOfFiles[cv]:startIndxOfFiles[cv + 1], :]  # test data comes from only one file after this CV
            label_test = labelsAll[startIndxOfFiles[cv]:startIndxOfFiles[cv + 1]]
            dataTrain = dataAll[0:startIndxOfFiles[cv], :]
            label_train = labelsAll[0:startIndxOfFiles[cv]]
            dataSource_test = (cv + 1) * np.ones((startIndxOfFiles[cv + 1] - startIndxOfFiles[cv]))  # value 1 means then file after the train one

        # normalize and discretize data if needed (for HD has to be normalized and discretized )
        if (FeaturesParams.featNorm == 'Norm&Discr'):
            # normalizing data and discretizing
            (data_train_Norm, data_test_Norm, data_train_Discr,
             data_test_Discr) = normalizeAndDiscretizeTrainAndTestData(dataTrain, dataTest,  HDParams.numSegmentationLevels)
            data_train_ToTrain = data_train_Discr.astype(int)
            data_test_ToTrain = data_test_Discr.astype(int)
        elif (FeaturesParams.featNorm == 'Norm'):
            # normalizing data and discretizing
            (data_train_Norm, data_test_Norm, data_train_Discr,
             data_test_Discr) = normalizeAndDiscretizeTrainAndTestData(dataTrain, dataTest,  HDParams.numSegmentationLevels)
            data_train_ToTrain = data_train_Norm
            data_test_ToTrain = data_test_Norm
        else:
            data_train_ToTrain = dataTrain
            data_test_ToTrain = dataTest

        ##########################################################
        ## STANDARD ML LEARNING - RANDOM FOREST
        t0 = time.time()
        StandardMLParams.modelType = 'RandomForest'
        MLstdModel = train_StandardML_moreModelsPossible(data_train_ToTrain, label_train, StandardMLParams)
        ThisSubjTimes[cv, 0] = time.time() - t0
        # testing
        (predLabels_test, probabLab_test, acc_test, accPerClass_test) = test_StandardML_moreModelsPossible( data_test_ToTrain, label_test, MLstdModel)
        (predLabels_train, probabLab_train, acc_train, accPerClass_train) = test_StandardML_moreModelsPossible( data_train_ToTrain, label_train, MLstdModel)
        print('RF acc_train: ', acc_train, 'acc_test: ', acc_test)

        # perform smoothing
        (performanceTrain0, yPredTrain_MovAvrgStep1, yPredTrain_MovAvrgStep2, yPredTrain_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels_train, label_train,probabLab_train, toleranceFP_bef, toleranceFP_aft,numLabelsPerHour,
                                                                             seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx,PostprocessingParams.bayesProbThresh)
        dataToSave = np.vstack((label_train, probabLab_train, predLabels_train, yPredTrain_MovAvrgStep1,  yPredTrain_MovAvrgStep2, yPredTrain_SmoothBayes)).transpose()
        outputName = folderOutPredictions + '/' + fileName2 + '_RF_TrainPredictions.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')

        (performanceTest0, yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2, yPredTest_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels_test, label_test, probabLab_test, toleranceFP_bef, toleranceFP_aft,
                                                                            numLabelsPerHour,seizureStableLenToTestIndx,seizureStablePercToTest,  distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
        dataToSave = np.vstack((label_test, probabLab_test, predLabels_test, yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2, yPredTest_SmoothBayes)).transpose()  # added from which file is specific part of test set
        outputName = folderOutPredictions + '/' + fileName2 + '_RF_TestPredictions.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')

        ##########################################################
        # HD APPROACHES
        # initializing model and then training
        # model = HD_classifier_GeneralAndNoCh(HDParams, len(data_train_ToTrain[0, :]), HDParams.HDvecType)
        # if more different encodings wants to be tested
        model = HD_classifier_GeneralWithChCombinations(HDParams, len(DatasetPreprocessParams.channelNamesToKeep), HDParams.HDvecType)
        storeModel_InitializedVectors_GeneralWithChCombinations(model, folderOutPredictions, fileName2)

        ################
        # STANDARD SINGLE PASS 2 CLASS LEARNING
        t0 = time.time()
        (ModelVectors, ModelVectorsNorm, numAddedVecPerClass, numLabels) = trainModelVecOnData(data_train_ToTrain,label_train, model, HDParams, HDParams.HDvecType)
        ThisSubjTimes[cv, 1] = time.time() - t0
        (predLabelsTrain_2class, predLabelsTest_2class, allInfo_Train, allInfo_Test)=testAndSavePredictionsForHD(folderOutPredictions, fileName2, model, ModelVectorsNorm, HDParams, PostprocessingParams, FeaturesParams,
                                    data_train_ToTrain, label_train, data_test_ToTrain, label_test, 'StdHD')
        #for Feat append - saving predictions and performance per feature
        if (HDParams.bindingFeatures == 'FeatAppend' or HDParams.bindingFeatures == 'ChAppend'):
            testAndSavePredictionsForHD_PerAppended(folderOutPredictionsPerFeat, fileName2, model, ModelVectorsNorm, HDParams, data_train_ToTrain, label_train,data_test_ToTrain, label_test, 'StdHD', FeatChType)

        #################
        # ONLINE HD
        ItterType = 'AddAndSubtract'  # to use latter for multi class only addandsubtract
        t0 = time.time()
        (ModelVectors_OnlineHDAddSub, ModelVectorsNorm_OnlineHDAddSub, numAddedVecPerClass_OnlineHDAddSub, weights, allInfo_Train) = onlineHD_ModelVecOnData(data_train_ToTrain, label_train, model, HDParams, 'AddAndSubtract',HDParams.HDvecType)
        ThisSubjTimes[cv, 2] = time.time() - t0
        (predLabelsTrain_OnlineHD, predLabelsTest_OnlineHD, allInfo_testTrain, allInfoTestTest)=testAndSavePredictionsForHD(folderOutPredictions, fileName2, model, ModelVectorsNorm_OnlineHDAddSub, HDParams, PostprocessingParams, FeaturesParams,
                                    data_train_ToTrain, label_train, data_test_ToTrain, label_test, 'OnlineHD')
        #for Feat append - saving predictions and performance per feature
        if (HDParams.bindingFeatures == 'FeatAppend' or HDParams.bindingFeatures == 'ChAppend'):
            testAndSavePredictionsForHD_PerAppended(folderOutPredictionsPerFeat, fileName2, model, ModelVectorsNorm_OnlineHDAddSub, HDParams, data_train_ToTrain, label_train,data_test_ToTrain, label_test, 'OnlineHD', FeatChType)

        ##########################################################
        # SAVE PREDICTIONS FOR ALL APPROACHES
        dataToSave_train = np.vstack((label_train, predLabelsTrain_2class, predLabelsTrain_OnlineHD)).transpose()
        outputName = folderOutPredictions + '/' + fileName2 + '_AllApproaches_TrainPredictions.csv'
        saveDataToFile(dataToSave_train, outputName, 'gzip')
        dataToSave_test = np.vstack((label_test, predLabelsTest_2class, predLabelsTest_OnlineHD)).transpose()
        outputName = folderOutPredictions + '/' + fileName2 + '_AllApproaches_TestPredictions.csv'
        saveDataToFile(dataToSave_test, outputName, 'gzip')

        # SAVE  MODEL VECTORS
        # standard learning
        outputName = folderOutPredictions + '/' + fileName2 + '_StdHD_ModelVecsNorm.csv'  # first nonSeiz, then Seiz
        saveDataToFile(ModelVectorsNorm.transpose(), outputName, 'gzip')
        outputName = folderOutPredictions + '/' + fileName2 + '_StdHD_ModelVecs.csv'  # first nonSeiz, then Seiz
        saveDataToFile(ModelVectors.transpose(), outputName, 'gzip')
        outputName = folderOutPredictions + '/' + fileName2 + '_StdHD_AddedToEachSubClass.csv'
        saveDataToFile(numAddedVecPerClass, outputName, 'gzip')
        # Online HD AddSub
        outputName = folderOutPredictions + '/' + fileName2 + '_OnlineHD_ModelVecsNorm.csv'  # first nonSeiz, then Seiz
        saveDataToFile(ModelVectorsNorm_OnlineHDAddSub.transpose(), outputName, 'gzip')
        outputName = folderOutPredictions + '/' + fileName2 + '_OnlineHD_ModelVecs.csv'  # first nonSeiz, then Seiz
        saveDataToFile(ModelVectors_OnlineHDAddSub.transpose(), outputName, 'gzip')
        outputName = folderOutPredictions + '/' + fileName2 + '_OnlineHD_AddedToEachSubClass.csv'
        saveDataToFile(numAddedVecPerClass_OnlineHDAddSub, outputName, 'gzip')

    #save time for different approahes of this subject
    outputName = folderOutPredictions + '/Subj'+ pat+'_TimeForDiffApproaches.csv'
    saveDataToFile(ThisSubjTimes, outputName, 'gzip')
    # average times for this subj
    AllSubjTimes[patIndx, :, 0] = np.nanmean(ThisSubjTimes, 0)
    AllSubjTimes[patIndx, :, 1] = np.nanstd(ThisSubjTimes, 0)

    # saving perfomance for all subj
    meanStd = ['_mean', '_std']
    for ni, meanStdVal in enumerate(meanStd):
        outputName = folderOutPredictions + '/AllSubj_TimeForDiffApproaches' + meanStdVal + '.csv'
        saveDataToFile(AllSubjTimes[:, :, ni], outputName, 'gzip')

# # # ########################################################################################
# # PLOTTING BASED ON CALCULATED PREDICTIONS
# #GeneralParams.patients =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
# GeneralParams.patients =['01','02','03','04','05','06','07','08','09','10','11']
#
# ## CALCULATE PERFORMANCE BASED ON PREDICTIONS (rerun for al subjects again)
# func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderOutPredictions, GeneralParams, PostprocessingParams,FeaturesParams, 'RF')
# func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderOutPredictions, GeneralParams, PostprocessingParams, FeaturesParams, 'StdHD')
# func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderOutPredictions, GeneralParams, PostprocessingParams,FeaturesParams, 'OnlineHD')
#
# ## MEASURE PERFORMANCE WHEN APPENDING TEST DATA and plot appended predictions in time
# func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderOutPredictions, GeneralParams,PostprocessingParams, FeaturesParams, 'RF')
# func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderOutPredictions, GeneralParams, PostprocessingParams, FeaturesParams, 'StdHD')
# func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderOutPredictions, GeneralParams, PostprocessingParams, FeaturesParams, 'OnlineHD')
#
# # ## PLOT COMPARISON BETWEEN RF, STD HD and ONLINE HD PERFORMANCE
# plot_performanceComparison_RFvsSTDHDandONLHD(folderOutPredictions, folderOutPredictionsPlot, GeneralParams.patients, 'Train')
# plot_performanceComparison_RFvsSTDHDandONLHD(folderOutPredictions, folderOutPredictionsPlot, GeneralParams.patients, 'Test')
# # ## plot predictions in time of all models
# modelsList = ['RF', 'StdHD', 'OnlineHD']
# func_plotPredictionsOfDifferentModels(modelsList, GeneralParams, folderOutPredictions, folderOutPredictionsPlot)

######################################################################################
# ADDITIONAL PLOTS FOR FEATURE APPENDING
# analysing quality of features
#   - separability of model vectors for seizure and non-seizure class of each feature
#	- average confidences per feature
#	- average performance per feature
if (HDParams.bindingFeatures == 'FeatAppend' or HDParams.bindingFeatures == 'ChAppend'):
    HDParams.numFeat = len(FeaturesParams.featNames)
    HDParams.numCh= len(DatasetPreprocessParams.channelNamesToKeep)

    # #FOR PAPER: plots correlations between different measures from HD vectors of indiv channels/features - separability, performance, correlations per channel/feature
    # plotPerAppendingComparison_ForPaper(folderOutPredictionsPerFeat, HDParams,  FeaturesParams,DatasetPreprocessParams,  'StdHD', FeatChType)
    # plotPerAppendingComparison_ForPaper(folderOutPredictionsPerFeat, HDParams,  FeaturesParams, DatasetPreprocessParams, 'OnlineHD', FeatChType)
    # # #old, and per inividual subjects
    # # func_analysePerAppending(folderOutPredictions, folderOutPredictionsPerFeat, GeneralParams, PostprocessingParams, FeaturesParams, HDParams, DatasetPreprocessParams, 'StdHD', FeatChType)
    # # func_analysePerAppending(folderOutPredictions, folderOutPredictionsPerFeat, GeneralParams, PostprocessingParams, FeaturesParams, HDParams, DatasetPreprocessParams, 'OnlineHD', FeatChType)

    #FOR PAPER: plot confidences per element in time
    func_plotConfidencesPerElement_Appending(folderOutPredictionsPerFeat, GeneralParams, PostprocessingParams, FeaturesParams,  'StdHD', FeatChType, elemNames)
    func_plotConfidencesPerElement_Appending(folderOutPredictionsPerFeat, GeneralParams, PostprocessingParams, FeaturesParams, 'OnlineHD', FeatChType, elemNames)