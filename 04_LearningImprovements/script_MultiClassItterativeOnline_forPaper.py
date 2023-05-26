
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"


from HDfunctionsLib import *
from parametersSetup import *

####################################

#SETUPS
# type of HD vectors
HDParams.HDvecType= 'bin'  #'bin', 'bipol' #binary 0,1, bipolar -1,1
HDParams.D=10000
GeneralParams.plottingON=0
GeneralParams.PersGenApproach='personalized' #'personalized', 'generalized'
datasetPreparationType='MoreNonSeizure_Fact10'  # 'MoreNonSeizure_Fact10_v1' , 'MoreNonSeizure_Fact10'
torch.cuda.set_device(HDParams.CUDAdevice)

#MULTI CLASS PARAMS
numSteps = 10
groupingThresh = 0.95
optType = 'F1DEgmean'  # 'simpleAcc', 'F1DEgmean' which performance to check if saturates
#subClassReductApproachType = 'clustering'  # 'removing', 'clustering' #DOING BOTH
perfDropThr=0.03 #0.01, 0.02, 0.03
#ITTERATIVE LEARNING
#ItterType='AddAndSubtract'  #'AddAndSubtract', 'AddOnly'  #DOING BOTH
ItterFact=1
ItterImprovThresh=0.01 #if in threec consecutive runs not bigger improvement then this then stop
savingStepData=1 #whether to save improvements per each itteration

## SETUP DATASET USED
Dataset='01_CHBMIT'
patients =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
patients =['01','02','03']
GeneralParams.patients =patients

# DEFINING INPUT/OUTPUT FOLDERS
folderIn = '01_datasetProcessed_'+datasetPreparationType+'/'
folderOut0='03_Predictions_' +datasetPreparationType #+'_run2/'
createFolderIfNotExists(folderOut0)
if (HDParams.HDvecType=='bin'):
    folderOut0=folderOut0 +'/'+ str(GeneralParams.PersGenApproach)+'/'
elif (HDParams.HDvecType == 'bipol'):
    folderOut0=folderOut0 +'/'+ str(GeneralParams.PersGenApproach)+'_bipolarVec/'
createFolderIfNotExists(folderOut0)
folderFeaturesOut0='02_features_'+datasetPreparationType
createFolderIfNotExists(folderFeaturesOut0)
folderFeaturesOut0=folderFeaturesOut0 +'/'+ str(GeneralParams.PersGenApproach)+'/'
createFolderIfNotExists(folderFeaturesOut0)

# TESTING STANDARD ML FEATURES - 45 FEAT
HDParams.HDapproachON=1
SegSymbParams.symbolType ='StandardMLFeatures' #'StandardMLFeatures', 'ReducedAndZCfeatures
SegSymbParams.numSegLevels=20
SegSymbParams.segLenSec = 8 #8 # length of EEG sements in sec
SegSymbParams.slidWindStepSec = 1 #1  # step of slidin window to extract segments in sec
HDParams.vectorTypeLevel = 'scaleNoRand1'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
HDParams.vectorTypeFeat='random'
HDParams.numFeat= 45 #45,25
HDParams.roundingTypeForHDVectors='inSteps' #'inSteps','onlyOne','noRounding'
HDParams.bindingFeatures='FeatxVal' #'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend1000'
HDParams.D=10000
# # # # # TESTING FEAT APPEND
# HDParams.bindingFeatures='FeatAppend' #'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend1000'
# HDParams.D=222
HDParams.ItterativeRelearning='on'
# HDParams.VotingType='ConfVoting' # 'ConfVoting', 'StdVoting'

#################################################################
## CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
if (SegSymbParams.symbolType == 'CWT'):
    folderOutName = SegSymbParams.symbolType + '_' + str(SegSymbParams.CWTlevel)
    folderOutNameFeat= SegSymbParams.symbolType
elif (SegSymbParams.symbolType == 'Entropy'):
    folderOutName = SegSymbParams.symbolType + '_' + SegSymbParams.entropyType + '_' + str(SegSymbParams.numSegLevels)
    folderOutNameFeat = SegSymbParams.symbolType+ '_' + SegSymbParams.entropyType
elif (SegSymbParams.symbolType == 'Amplitude'):
    folderOutName = SegSymbParams.symbolType + '_' + str(SegSymbParams.numSegLevels) + '_' + str(SegSymbParams.amplitudeRangeFactor) + '_' + SegSymbParams.amplitudeBinsSpacing
    folderOutNameFeat = SegSymbParams.symbolType
elif (SegSymbParams.symbolType == 'AllFeatures'):
    folderOutName = SegSymbParams.symbolType + '_' + str(SegSymbParams.CWTlevel) + '_' + SegSymbParams.entropyType + '_' + str(
        SegSymbParams.numSegLevels) + '_' + SegSymbParams.amplitudeBinsSpacing + '_' + HDParams.bindingFeatures + '_FEATvec' + HDParams.vectorTypeFeat
    folderOutNameFeat = SegSymbParams.symbolType + '_' + SegSymbParams.entropyType
elif (SegSymbParams.symbolType == 'StandardMLFeatures' or SegSymbParams.symbolType =='ReducedAndZCfeatures' ):
    folderOutName = SegSymbParams.symbolType +'_'+ str(HDParams.numFeat)+ '_' + str(SegSymbParams.numSegLevels) + '_numFeat' + str(
        HDParams.numFeat) + '_' + HDParams.bindingFeatures + '_FEATvec' + HDParams.vectorTypeFeat
    folderOutNameFeat = SegSymbParams.symbolType + '_'+ str(HDParams.numFeat)


folderOutName = folderOutName + '_' + str(SegSymbParams.segLenSec) + '_' + str(
    SegSymbParams.slidWindStepSec) + 's' + '_' + HDParams.similarityType + '_RND' + HDParams.roundingTypeForHDVectors + '_CHVect' + HDParams.vectorTypeCh + '_LVLVect' + HDParams.vectorTypeLevel+'_D'+str(HDParams.D)
folderOutNameFeat =folderOutNameFeat+ '_' + str(SegSymbParams.segLenSec) + '_' + str(SegSymbParams.slidWindStepSec) + 's'

#CREATING OUTPUT FOLDER!!!
folderOutName=folderOutName +'_AllApproaches' #!!!!!!!!!!!

folderOut_ML = folderOut0 + folderOutName
createFolderIfNotExists(folderOut_ML)
folderOut_ML = folderOut_ML + '/' + optType + '_' + str(perfDropThr) + '_' + str(numSteps) + '_' + str(groupingThresh)  # doing both clusteering and removal and addAndSubtract and AddOnly
createFolderIfNotExists(folderOut_ML)
print('FOLDER OUT:', folderOut_ML)
folderFeaturesOut = folderFeaturesOut0 + folderOutNameFeat
createFolderIfNotExists(folderFeaturesOut)
print('FOLDER OUT FEATURES:', folderFeaturesOut)
folderOutPredictionsPlot = folderOut_ML+'/Plots_predictions'
createFolderIfNotExists(folderOutPredictionsPlot)

#CALCULATING SOME PARAMETERS
seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
seizureStablePercToTest = GeneralParams.seizureStablePercToTest
distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz / SegSymbParams.slidWindStepSec)

################################################################
## CALCULATING FEATURES FOR EACH FILE
numFiles = len(np.sort(glob.glob(folderFeaturesOut + '/*chb' + '*.csv')))
if (numFiles==0):
    print('EXTRACTING FEATURES!!!')
    func_calculateFeaturesForInputFiles(SigInfoParams, SegSymbParams, GeneralParams, HDParams, folderIn, folderFeaturesOut)

################################################################
## TRAINING
AllSubjResRF_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResRF_test = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjRes_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjRes_test = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMulti_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMulti_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResMultiRedRemov_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMultiRedRemov_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResMultiRedClust_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMultiRedClust_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResItterRemov_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResItterRemov_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResItterClust_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResItterClust_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResItter2class_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResItter2class_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResItter2classAddOnly_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResItter2classAddOnly_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjOnlineHDAddSub_train= np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjOnlineHDAddSub_test= np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjOnlineHDAdd_train= np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjOnlineHDAdd_test= np.zeros((len(GeneralParams.patients), 33, 2))
AllSubj_OptimalResultsRemov_train= np.zeros((len(GeneralParams.patients),34, 2)) #3+3+1+9+9+9
AllSubj_OptimalResultsRemov_test= np.zeros((len(GeneralParams.patients),34, 2))
AllSubj_OptimalResultsClust_train= np.zeros((len(GeneralParams.patients),34, 2)) #3+3+1+9+9+9
AllSubj_OptimalResultsClust_test= np.zeros((len(GeneralParams.patients),34, 2))
AllSubjTimes = np.zeros((len(GeneralParams.patients), 10,2))  # 10 approaches
for patIndx, pat in enumerate(GeneralParams.patients):
    numFiles = len(np.sort(glob.glob(folderFeaturesOut + '/*chb' + pat + '*.csv')))
    print('-- Patient:', pat, 'NumSeizures:', numFiles)

    AllResRF_train=np.zeros((numFiles,33))
    AllResRF_test = np.zeros((numFiles, 33))
    AllRes_train=np.zeros((numFiles,33))
    AllRes_test = np.zeros((numFiles, 33))
    AllResMulti_train = np.zeros((numFiles, 33))
    AllResMulti_test = np.zeros((numFiles, 33))
    AllResMultiRedRemov_train = np.zeros((numFiles, 33))
    AllResMultiRedRemov_test = np.zeros((numFiles, 33))
    AllResMultiRedClust_train = np.zeros((numFiles, 33))
    AllResMultiRedClust_test = np.zeros((numFiles, 33))
    AllResItterRemov_train = np.zeros((numFiles, 33))
    AllResItterRemov_test = np.zeros((numFiles, 33))
    AllResItterClust_train = np.zeros((numFiles, 33))
    AllResItterClust_test = np.zeros((numFiles, 33))
    AllResItter2class_train = np.zeros((numFiles, 33))
    AllResItter2class_test = np.zeros((numFiles, 33))
    AllResItter2classAddOnly_train = np.zeros((numFiles, 33))
    AllResItter2classAddOnly_test = np.zeros((numFiles, 33))
    AllResOnlineHD_AddSub_train= np.zeros((numFiles, 33))
    AllResOnlineHD_AddSub_test= np.zeros((numFiles, 33))
    AllResOnlineHD_Add_train= np.zeros((numFiles, 33))
    AllResOnlineHD_Add_test= np.zeros((numFiles, 33))
    OptimalValuesRemov_train= np.zeros((numFiles, 34))
    OptimalValuesRemov_test = np.zeros((numFiles, 34))
    OptimalValuesClust_train= np.zeros((numFiles, 34))
    OptimalValuesClust_test = np.zeros((numFiles, 34))
    ThisSubjTimes=np.zeros((numFiles, 10)) #10 approaches
    for cv in range(numFiles):
        # creates list of files to train and test on
        filesToTrainOn = []
        for fIndx, fileName in enumerate(np.sort(glob.glob(folderFeaturesOut + '/*chb' + pat + '*.csv'))):
            if (fIndx != cv):
                filesToTrainOn.append(fileName)
            else:
                filesToTestOn = list(fileName.split(" "))

        pom, fileName1 = os.path.split(filesToTestOn[0])
        fileName2 = os.path.splitext(fileName1)[0]

        # concatenating data from more files
        (dataTrain, label_train)=concatenateDataFromFiles(filesToTrainOn)
        (dataTest, label_test) = concatenateDataFromFiles(filesToTestOn)

        # normalizing data and discretizing
        (data_train_Norm, data_test_Norm, data_train_Discr, data_test_Discr)=normalizeAndDiscretizeTrainAndTestData(dataTrain, dataTest, SegSymbParams.numSegLevels)
        data_train_Discr=data_train_Discr.astype(int)
        data_test_Discr = data_test_Discr.astype(int)

        ## STANDARD ML LEARNING - RANDOM FOREST
        MLstdModel = train_StandardML_moreModelsPossible(data_train_Discr, label_train, StandardMLParams)
        (predLabelsTrain_RF, probabilityLabelsTrain_RF)= test_StandardML_moreModelsPossible_v2(data_train_Discr, MLstdModel)
        (predLabelsTest_RF, probabilityLabelsTest_RF) = test_StandardML_moreModelsPossible_v2(data_test_Discr, MLstdModel)
        #calcualte performance
        AllResRF_train[cv,0:6] = np.hstack((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)) #not applicable for RF, but just to keep the same dimensions
        AllResRF_test[cv, 0:6] = np.hstack((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)) #not applicable for RF, but just to keep the same dimensions
        AllResRF_train[cv,6:15] = performance_all9(predLabelsTrain_RF, label_train, toleranceFP_bef, toleranceFP_aft,  numLabelsPerHour)
        AllResRF_test[cv,6:15] = performance_all9(predLabelsTest_RF, label_test, toleranceFP_bef, toleranceFP_aft,  numLabelsPerHour)
        (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabelsTrain_RF, seizureStableLenToTestIndx, seizureStablePercToTest,  distanceBetweenSeizuresIndx)
        AllResRF_train[cv,15:24] = performance_all9(yPred_SmoothOurStep1, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        AllResRF_train[cv,24:33] = performance_all9(yPred_SmoothOurStep2, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabelsTest_RF, seizureStableLenToTestIndx,seizureStablePercToTest,  distanceBetweenSeizuresIndx)
        AllResRF_test[cv,15:24] = performance_all9(yPred_SmoothOurStep1, label_test, toleranceFP_bef, toleranceFP_aft,  numLabelsPerHour)
        AllResRF_test[cv,24:33] = performance_all9(yPred_SmoothOurStep2, label_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        #save predictions
        dataToSave_train=np.vstack((label_train, predLabelsTrain_RF, probabilityLabelsTrain_RF)).transpose()
        outputName = folderOut_ML + '/' + fileName2 + '_RF_TrainPredictions.csv'
        np.savetxt(outputName, dataToSave_train, delimiter=",")
        dataToSave_test=np.vstack((label_test, predLabelsTest_RF, probabilityLabelsTest_RF)).transpose()
        outputName = folderOut_ML + '/' + fileName2 + '_RF_TestPredictions.csv'
        np.savetxt(outputName, dataToSave_test, delimiter=",")

        ############################################
        # HD APPROACHES
        # initializing model and then training
        model = HD_classifier_GeneralAndNoCh(SigInfoParams, SegSymbParams, HDParams, HDParams.numFeat*len(SigInfoParams.chToKeep), HDParams.HDvecType)
        #model = HD_classifier_GeneralWithChCombinations(SigInfoParams, SegSymbParams, HDParams, len(SigInfoParams.chToKeep), HDParams.HDvecType)

        ################
        #STANDARD SINGLE PASS 2 CLASS LEARNING
        t0 = time.time()
        (ModelVectors, ModelVectorsNorm, numAddedVecPerClass, numLabels) = trainModelVecOnData(data_train_Discr, label_train, model, HDParams, HDParams.HDvecType)
        ThisSubjTimes[cv,0] = time.time()-t0
        (AllRes_train[cv,:], AllRes_test[cv,:], predLabelsTrain_2class, predLabelsTest_2class)= testModelsAndReturnAllPerformances_2class(data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                  ModelVectorsNorm, HDParams, GeneralParams, SegSymbParams,  HDParams.HDvecType)
        print('2 CLASS acc_train: ', AllRes_train[cv,2], 'acc_test: ', AllRes_test[cv,2])

        #################
        # ONLY ITTERATIVE WITH 2 CLASSES
        #ADD AND SUBTRACT
        ItterType='AddAndSubtract'
        folderOut_Itter2class = folderOut_ML + '/Itterative2class_' + ItterType + '_' + str(ItterFact) + '_' + str(ItterImprovThresh) + '_' + optType + '/'
        ModelVectorsNorm_Seiz_Itter2class =np.copy(np.reshape(ModelVectorsNorm[1, :], (1, HDParams.D)))
        ModelVectorsNorm_NonSeiz_Itter2class=np.copy(np.reshape(ModelVectorsNorm[0, :], (1, HDParams.D)))
        ModelVectors_Seiz_Itter2class =np.copy(np.reshape(ModelVectors[1, :], (1, HDParams.D)))
        ModelVectors_NonSeiz_Itter2class=np.copy(np.reshape(ModelVectors[0, :], (1, HDParams.D)))
        numAddedVecPerClass_Seiz_Itter2class=np.copy([numAddedVecPerClass[1]])
        numAddedVecPerClass_NonSeiz_Itter2class = np.copy([numAddedVecPerClass[0]])
        t0 = time.time()
        (ModelVectorsNorm_Seiz_Itter2class, ModelVectorsNorm_NonSeiz_Itter2class, ModelVectors_Seiz_Itter2class,
         ModelVectors_NonSeiz_Itter2class, numAddedVecPerClass_Seiz_Itter2class,
         numAddedVecPerClass_NonSeiz_Itter2class) = totalRetrainingAsLongAsNeeded( data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                                                   ModelVectorsNorm_Seiz_Itter2class,ModelVectorsNorm_NonSeiz_Itter2class,
                                                                                   ModelVectors_Seiz_Itter2class, ModelVectors_NonSeiz_Itter2class,
                                                                                   numAddedVecPerClass_Seiz_Itter2class, numAddedVecPerClass_NonSeiz_Itter2class,
                                                                                   HDParams,GeneralParams, SegSymbParams,  optType, ItterType, ItterFact, ItterImprovThresh,
                                                                                   savingStepData, folderOut_Itter2class, fileName2,  HDParams.HDvecType)
        ThisSubjTimes[cv, 1] = time.time() - t0
        #measure performance
        (AllResItter2class_train[cv,:], AllResItter2class_test[cv,:], predLabelsTrain_Itter2class, predLabelsTest_Itter2class)=testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                     ModelVectorsNorm_Seiz_Itter2class, ModelVectorsNorm_NonSeiz_Itter2class, HDParams,  GeneralParams, SegSymbParams,  HDParams.HDvecType)
        print('2CLASS AFTER ITTERATIVE acc_train: ', AllResItter2class_train[cv,2], 'acc_test: ', AllResItter2class_test[cv,2],  'numSubClass_Seiz', 1, 'numSubClass_NonSeiz', 1)

        #ADD ONLY
        ItterType='AddOnly'
        folderOut_Itter2class = folderOut_ML + '/Itterative2class_' + ItterType + '_' + str(ItterFact) + '_' + str(ItterImprovThresh) + '_' + optType + '/'
        ModelVectorsNorm_Seiz_Itter2classAddOnly =np.copy(np.reshape(ModelVectorsNorm[1, :], (1, HDParams.D)))
        ModelVectorsNorm_NonSeiz_Itter2classAddOnly=np.copy(np.reshape(ModelVectorsNorm[0, :], (1, HDParams.D)))
        ModelVectors_Seiz_Itter2classAddOnly =np.copy(np.reshape(ModelVectors[1, :], (1, HDParams.D)))
        ModelVectors_NonSeiz_Itter2classAddOnly=np.copy(np.reshape(ModelVectors[0, :], (1, HDParams.D)))
        numAddedVecPerClass_Seiz_Itter2classAddOnly=np.copy([numAddedVecPerClass[1]])
        numAddedVecPerClass_NonSeiz_Itter2classAddOnly = np.copy([numAddedVecPerClass[0]])
        t0 = time.time()
        (ModelVectorsNorm_Seiz_Itter2classAddOnly, ModelVectorsNorm_NonSeiz_Itter2classAddOnly, ModelVectors_Seiz_Itter2classAddOnly,
         ModelVectors_NonSeiz_Itter2classAddOnly, numAddedVecPerClass_Seiz_Itter2classAddOnly,
         numAddedVecPerClass_NonSeiz_Itter2classAddOnly) = totalRetrainingAsLongAsNeeded( data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                                                   ModelVectorsNorm_Seiz_Itter2classAddOnly,ModelVectorsNorm_NonSeiz_Itter2classAddOnly,
                                                                                   ModelVectors_Seiz_Itter2classAddOnly, ModelVectors_NonSeiz_Itter2classAddOnly,
                                                                                   numAddedVecPerClass_Seiz_Itter2classAddOnly, numAddedVecPerClass_NonSeiz_Itter2classAddOnly,
                                                                                   HDParams,GeneralParams, SegSymbParams,  optType, ItterType, ItterFact, ItterImprovThresh,
                                                                                   savingStepData, folderOut_Itter2class, fileName2,  HDParams.HDvecType)
        ThisSubjTimes[cv, 2] = time.time() - t0
        #measure performance
        (AllResItter2classAddOnly_train[cv,:], AllResItter2classAddOnly_test[cv,:], predLabelsTrain_Itter2classAddOnly, predLabelsTest_Itter2classAddOnly)=testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                     ModelVectorsNorm_Seiz_Itter2classAddOnly, ModelVectorsNorm_NonSeiz_Itter2classAddOnly, HDParams,  GeneralParams, SegSymbParams,  HDParams.HDvecType)
        print('2CLASS AFTER ITTERATIVE ADD ONLY acc_train: ', AllResItter2classAddOnly_train[cv,2], 'acc_test: ', AllResItter2classAddOnly_test[cv,2],  'numSubClass_Seiz', 1, 'numSubClass_NonSeiz', 1)



        ItterType = 'AddAndSubtract' # to use latter for multi class only addandsubtract
        #################
        # ONLINE HD - SINGLE PASS BUT WEIGHTNIHG SAMPLES BEFORE ADDING
        #ADDANDSUBTRACT
        #train
        t0 = time.time()
        (ModelVectors_OnlineHDAddSub, ModelVectorsNorm_OnlineHDAddSub, numAddedVecPerClass_OnlineHDAddSub, weights) =onlineHD_ModelVecOnData(data_train_Discr, label_train, model, HDParams, 'AddAndSubtract',  HDParams.HDvecType)
        ThisSubjTimes[cv, 8] = time.time() - t0
        #performance
        (AllResOnlineHD_AddSub_train[cv,:], AllResOnlineHD_AddSub_test[cv,:], predLabelsTrain_OnlineHDAddSub, predLabelsTest_OnlineHDAddSub)= testModelsAndReturnAllPerformances_2class(data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                  ModelVectorsNorm_OnlineHDAddSub, HDParams, GeneralParams, SegSymbParams,  HDParams.HDvecType)
        print('ONLINE HD AddAndSub acc_train: ', AllResOnlineHD_AddSub_train[cv,2], 'acc_test: ', AllResOnlineHD_AddSub_test[cv,2])
        #Save weights
        folderOut_OnlineHD= folderOut_ML + '/OnlineHD/'
        createFolderIfNotExists(folderOut_OnlineHD)
        outputName = folderOut_OnlineHD + '/' + fileName2 + '_Weights_Add&Subtract.csv'
        np.savetxt(outputName, weights, delimiter=",")

        #ADD ONLY
        #train
        t0 = time.time()
        (ModelVectors_OnlineHDAdd, ModelVectorsNorm_OnlineHDAdd, numAddedVecPerClass_OnlineHDAdd, weights) =onlineHD_ModelVecOnData(data_train_Discr, label_train, model,  HDParams, 'AddOnly',  HDParams.HDvecType)
        ThisSubjTimes[cv,9] = time.time() - t0
        #performance
        (AllResOnlineHD_Add_train[cv,:], AllResOnlineHD_Add_test[cv,:], predLabelsTrain_OnlineHDAdd, predLabelsTest_OnlineHDAdd)= testModelsAndReturnAllPerformances_2class(data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                  ModelVectorsNorm_OnlineHDAdd, HDParams, GeneralParams, SegSymbParams,  HDParams.HDvecType)
        print('ONLINE HD AddOnly acc_train: ', AllResOnlineHD_Add_train[cv,2], 'acc_test: ', AllResOnlineHD_Add_test[cv,2])
        #save weights
        outputName = folderOut_OnlineHD + '/' + fileName2 + '_Weights_AddOnly.csv'
        np.savetxt(outputName, weights, delimiter=",")

        #################
        #MULTICLASS LEARNING WITH ITTERATIVE AT THE END
        t0 = time.time()
        (ModelVectorsMulti_Seiz, ModelVectorsMultiNorm_Seiz, ModelVectorsMulti_NonSeiz, ModelVectorsMultiNorm_NonSeiz,
         numAddedVecPerClassMulti_Seiz, numAddedVecPerClassMulti_NonSeiz) =trainModelVecOnData_Multiclass(data_train_Discr, label_train, model, HDParams, HDParams.HDvecType)
        ThisSubjTimes[cv, 3] = time.time() - t0
        #performance on training and test dataset
        (AllResMulti_train[cv,:], AllResMulti_test[cv,:], predLabelsTrain_Multi, predLabelsTest_Multi)=testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                     ModelVectorsMultiNorm_Seiz, ModelVectorsMultiNorm_NonSeiz, HDParams,  GeneralParams, SegSymbParams, HDParams.HDvecType)
        print('MULTI CLASS acc_train: ', AllResMulti_train[cv,2], 'acc_test: ', AllResMulti_test[cv,2],  'numSubClass_Seiz', len(numAddedVecPerClassMulti_Seiz), 'numSubClass_NonSeiz', len(numAddedVecPerClassMulti_NonSeiz))

        ################
        #ANALYSE REMOVING LESS CROWDED SUBCLASSES
        ModelVectorsMultiNorm_Seiz_OptimalRemov =np.copy(ModelVectorsMultiNorm_Seiz)
        ModelVectorsMultiNorm_NonSeiz_OptimalRemov=np.copy(ModelVectorsMultiNorm_NonSeiz)
        ModelVectorsMulti_Seiz_OptimalRemov =np.copy(ModelVectorsMulti_Seiz)
        ModelVectorsMulti_NonSeiz_OptimalRemov=np.copy(ModelVectorsMulti_NonSeiz)
        numAddedVecPerClass_Seiz_OptimalRemov=np.copy(numAddedVecPerClassMulti_Seiz)
        numAddedVecPerClass_NonSeiz_OptimalRemov=np.copy(numAddedVecPerClassMulti_NonSeiz)
        t0 = time.time()
        (OptimalValuesRemov_train[cv,:], OptimalValuesRemov_test[cv,:], ModelVectorsMulti_Seiz_OptimalRemov, ModelVectorsMulti_NonSeiz_OptimalRemov, ModelVectorsMultiNorm_Seiz_OptimalRemov, ModelVectorsMultiNorm_NonSeiz_OptimalRemov, numAddedVecPerClass_Seiz_OptimalRemov,
         numAddedVecPerClass_NonSeiz_OptimalRemov)=reduceNumSubclasses_removingApproach(data_train_Discr, label_train,data_test_Discr, label_test, model,  HDParams, ModelVectorsMulti_Seiz_OptimalRemov, ModelVectorsMulti_NonSeiz_OptimalRemov,
                                                                                   ModelVectorsMultiNorm_Seiz_OptimalRemov, ModelVectorsMultiNorm_NonSeiz_OptimalRemov, numAddedVecPerClass_Seiz_OptimalRemov, numAddedVecPerClass_NonSeiz_OptimalRemov,
                                                                                   numSteps, optType, perfDropThr,  GeneralParams, SegSymbParams, folderOut_ML, fileName2, HDParams.HDvecType)
        ThisSubjTimes[cv, 4] = time.time() - t0
        # performance on training and test dataset
        (AllResMultiRedRemov_train[cv, :], AllResMultiRedRemov_test[cv, :], predLabelsTrain_MultiRedRemov, predLabelsTest_MultiRedRemov) = testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
            ModelVectorsMultiNorm_Seiz_OptimalRemov, ModelVectorsMultiNorm_NonSeiz_OptimalRemov, HDParams, GeneralParams, SegSymbParams, HDParams.HDvecType)
        print('MULTI CLASS REDUCED REMOV acc_train: ', AllResMultiRedRemov_train[cv, 2], 'acc_test: ', AllResMultiRedRemov_test[cv, 2], 'numSubClass_Seiz', len(ModelVectorsMulti_Seiz_OptimalRemov[:,0]), 'numSubClass_NonSeiz',  len(ModelVectorsMulti_NonSeiz_OptimalRemov[:,0]))


        #CLUSTERING WAY OF REDUCING SUBCLASSES
        ModelVectorsMultiNorm_Seiz_OptimalClust =np.copy(ModelVectorsMultiNorm_Seiz)
        ModelVectorsMultiNorm_NonSeiz_OptimalClust=np.copy(ModelVectorsMultiNorm_NonSeiz)
        ModelVectorsMulti_Seiz_OptimalClust =np.copy(ModelVectorsMulti_Seiz)
        ModelVectorsMulti_NonSeiz_OptimalClust=np.copy(ModelVectorsMulti_NonSeiz)
        numAddedVecPerClass_Seiz_OptimalClust=np.copy(numAddedVecPerClassMulti_Seiz)
        numAddedVecPerClass_NonSeiz_OptimalClust=np.copy(numAddedVecPerClassMulti_NonSeiz)
        t0 = time.time()
        (OptimalValuesClust_train[cv,:], OptimalValuesClust_test[cv,:], ModelVectorsMulti_Seiz_OptimalClust, ModelVectorsMulti_NonSeiz_OptimalClust, ModelVectorsMultiNorm_Seiz_OptimalClust,ModelVectorsMultiNorm_NonSeiz_OptimalClust, numAddedVecPerClass_Seiz_OptimalClust,
         numAddedVecPerClass_NonSeiz_OptimalClust) = reduceNumSubclasses_clusteringApproach(data_train_Discr, label_train,data_test_Discr,label_test, model, HDParams, ModelVectorsMulti_Seiz_OptimalClust, ModelVectorsMulti_NonSeiz_OptimalClust,
                                                                                     ModelVectorsMultiNorm_Seiz_OptimalClust, ModelVectorsMultiNorm_NonSeiz_OptimalClust,  numAddedVecPerClass_Seiz_OptimalClust,  numAddedVecPerClass_NonSeiz_OptimalClust,
                                                                                     numSteps, optType, perfDropThr, groupingThresh,  GeneralParams, SegSymbParams, folderOut_ML, fileName2, HDParams.HDvecType)
        ThisSubjTimes[cv, 5] = time.time() - t0
        # performance on training and test dataset
        (AllResMultiRedClust_train[cv, :], AllResMultiRedClust_test[cv, :], predLabelsTrain_MultiRedClust, predLabelsTest_MultiRedClust) = testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
            ModelVectorsMultiNorm_Seiz_OptimalClust, ModelVectorsMultiNorm_NonSeiz_OptimalClust, HDParams, GeneralParams, SegSymbParams, HDParams.HDvecType)
        print('MULTI CLASS REDUCED CLUSTER  acc_train: ', AllResMultiRedClust_train[cv, 2], 'acc_test: ', AllResMultiRedClust_test[cv, 2], 'numSubClass_Seiz', len(ModelVectorsMulti_Seiz_OptimalClust[:,0]), 'numSubClass_NonSeiz',  len(ModelVectorsMulti_NonSeiz_OptimalClust[:,0]))

        #################
        # RETRAINING WITH EXISTING MODELS AND SUBCLASSES
        folderOut_ItterMultiClass = folderOut_ML + '/ItterativeMultiClassRemov_' + ItterType + '_' + str(ItterFact) + '_' + str(ItterImprovThresh) + '_' + optType + '/'
        ModelVectorsNorm_Seiz_RetrainedRemov =np.copy(ModelVectorsMultiNorm_Seiz_OptimalRemov)
        ModelVectorsNorm_NonSeiz_RetrainedRemov=np.copy(ModelVectorsMultiNorm_NonSeiz_OptimalRemov)
        ModelVectors_Seiz_RetrainedRemov =np.copy(ModelVectorsMulti_Seiz_OptimalRemov)
        ModelVectors_NonSeiz_RetrainedRemov=np.copy(ModelVectorsMulti_NonSeiz_OptimalRemov)
        numAddedVecPerClass_Seiz_RetrainedRemov=np.copy(numAddedVecPerClass_Seiz_OptimalRemov)
        numAddedVecPerClass_NonSeiz_RetrainedRemov=np.copy(numAddedVecPerClass_NonSeiz_OptimalRemov)
        t0 = time.time()
        (ModelVectorsNorm_Seiz_RetrainedRemov, ModelVectorsNorm_NonSeiz_RetrainedRemov, ModelVectors_Seiz_RetrainedRemov,ModelVectors_NonSeiz_RetrainedRemov, numAddedVecPerClass_Seiz_RetrainedRemov,
         numAddedVecPerClass_NonSeiz_RetrainedRemov) = totalRetrainingAsLongAsNeeded( data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                                                 ModelVectorsNorm_Seiz_RetrainedRemov,  ModelVectorsNorm_NonSeiz_RetrainedRemov,
                                                                                 ModelVectors_Seiz_RetrainedRemov,ModelVectors_NonSeiz_RetrainedRemov,
                                                                                 numAddedVecPerClass_Seiz_RetrainedRemov,  numAddedVecPerClass_NonSeiz_RetrainedRemov,
                                                                                 HDParams, GeneralParams, SegSymbParams, optType, ItterType, ItterFact, ItterImprovThresh, savingStepData, folderOut_ItterMultiClass, fileName2, HDParams.HDvecType)
        ThisSubjTimes[cv, 6] = time.time() - t0
        # performance on training and test dataset
        (AllResItterRemov_train[cv, :], AllResItterRemov_test[cv, :], predLabelsTrain_ItterMultiRemov, predLabelsTest_ItterMultiRemov) = testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
            ModelVectorsNorm_Seiz_RetrainedRemov, ModelVectorsNorm_NonSeiz_RetrainedRemov, HDParams, GeneralParams, SegSymbParams, HDParams.HDvecType)
        print('MULTI CLASS ITTER REMOV acc_train: ', AllResItterRemov_train[cv, 2], 'acc_test: ', AllResItterRemov_test[cv, 2], 'numSubClass_Seiz', len(ModelVectorsMulti_Seiz_OptimalRemov[:,0]), 'numSubClass_NonSeiz',  len(ModelVectorsMulti_NonSeiz_OptimalRemov[:,0]))

        folderOut_ItterMultiClass = folderOut_ML + '/ItterativeMultiClassClust_' + ItterType + '_' + str(ItterFact) + '_' + str(ItterImprovThresh) + '_' + optType + '/'
        ModelVectorsNorm_Seiz_RetrainedClust =np.copy(ModelVectorsMultiNorm_Seiz_OptimalClust)
        ModelVectorsNorm_NonSeiz_RetrainedClust=np.copy(ModelVectorsMultiNorm_NonSeiz_OptimalClust)
        ModelVectors_Seiz_RetrainedClust =np.copy(ModelVectorsMulti_Seiz_OptimalClust)
        ModelVectors_NonSeiz_RetrainedClust=np.copy(ModelVectorsMulti_NonSeiz_OptimalClust)
        numAddedVecPerClass_Seiz_RetrainedClust=np.copy(numAddedVecPerClass_Seiz_OptimalClust)
        numAddedVecPerClass_NonSeiz_RetrainedClust=np.copy(numAddedVecPerClass_NonSeiz_OptimalClust)
        t0 = time.time()
        (ModelVectorsNorm_Seiz_RetrainedClust, ModelVectorsNorm_NonSeiz_RetrainedClust, ModelVectors_Seiz_RetrainedClust,ModelVectors_NonSeiz_RetrainedClust, numAddedVecPerClass_Seiz_RetrainedClust,
         numAddedVecPerClass_NonSeiz_RetrainedClust) = totalRetrainingAsLongAsNeeded( data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                                                      ModelVectorsNorm_Seiz_RetrainedClust, ModelVectorsNorm_NonSeiz_RetrainedClust,
                                                                                      ModelVectors_Seiz_RetrainedClust, ModelVectors_NonSeiz_RetrainedClust,
                                                                                      numAddedVecPerClass_Seiz_RetrainedClust, numAddedVecPerClass_NonSeiz_RetrainedClust,
                                                                                 HDParams, GeneralParams, SegSymbParams, optType, ItterType, ItterFact, ItterImprovThresh, savingStepData, folderOut_ItterMultiClass, fileName2, HDParams.HDvecType)
        ThisSubjTimes[cv, 7] = time.time() - t0
        # performance on training and test dataset
        (AllResItterClust_train[cv, :], AllResItterClust_test[cv, :], predLabelsTrain_ItterMultiClust, predLabelsTest_ItterMultiClust) = testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
            ModelVectorsNorm_Seiz_RetrainedClust, ModelVectorsNorm_NonSeiz_RetrainedClust, HDParams, GeneralParams, SegSymbParams, HDParams.HDvecType)
        print('MULTI CLASS ITTER CLUST acc_train: ', AllResItterClust_train[cv, 2], 'acc_test: ', AllResItterClust_test[cv, 2], 'numSubClass_Seiz', len(ModelVectorsMulti_Seiz_OptimalClust[:,0]), 'numSubClass_NonSeiz',  len(ModelVectorsMulti_NonSeiz_OptimalClust[:,0]))



        #SAVE PREDICTIONS FOR ALL APPROACHES
        dataToSave_train=np.vstack((label_train, predLabelsTrain_2class, predLabelsTrain_Itter2class,predLabelsTrain_Itter2classAddOnly,   predLabelsTrain_Multi, predLabelsTrain_MultiRedRemov, predLabelsTrain_MultiRedClust, predLabelsTrain_ItterMultiRemov, predLabelsTrain_ItterMultiClust, predLabelsTrain_OnlineHDAdd, predLabelsTrain_OnlineHDAddSub)).transpose()
        outputName = folderOut_ML + '/' + fileName2 + '_AllApproaches_TrainPredictions.csv'
        np.savetxt(outputName, dataToSave_train, delimiter=",")
        dataToSave_test=np.vstack((label_test, predLabelsTest_2class, predLabelsTest_Itter2class, predLabelsTest_Itter2classAddOnly,  predLabelsTest_Multi, predLabelsTest_MultiRedRemov,predLabelsTest_MultiRedClust, predLabelsTest_ItterMultiRemov, predLabelsTest_ItterMultiClust, predLabelsTest_OnlineHDAdd, predLabelsTest_OnlineHDAddSub)).transpose()
        outputName = folderOut_ML + '/' + fileName2 + '_AllApproaches_TestPredictions.csv'
        np.savetxt(outputName, dataToSave_test, delimiter=",")

        #plot predictions for test
        # approachNames = ['2C', '2Citter', 'MC', 'MCred', 'MCredItter']
        # approachIndx = [1, 2, 4, 6, 8]
        approachNames = ['2C', '2Citter','2CitterAdd', 'MC', 'MCredRem','MCredClust', 'MCredItterRem','MCredItterClust', 'OnlineAdd', 'OnlineAddSub']
        approachIndx = [1, 2, 4, 6, 8, 10,12, 14, 16, 17]
        func_plotRawSignalAndPredictionsOfDiffApproaches_thisFile(fileName2, dataToSave_test,dataToSave_train, approachNames, approachIndx, folderIn, folderOutPredictionsPlot, SigInfoParams, GeneralParams, SegSymbParams)


        #SAVE  MODEL VECTORS
        #standard learning
        outputName = folderOut_ML + '/' + fileName2 + '_StandardLearning_ModelVecsNorm.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, ModelVectorsNorm.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_StandardLearning_ModelVecs.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, ModelVectors.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_StandardLearning_AddedToEachSubClass.csv'
        np.savetxt(outputName, numAddedVecPerClass, delimiter=",")
        #Online HD AddSub
        outputName = folderOut_ML + '/' + fileName2 + '_OnlineHDAddSub_ModelVecsNorm.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, ModelVectorsNorm_OnlineHDAddSub.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_OnlineHDAddSub_ModelVecs.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, ModelVectors_OnlineHDAddSub.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_OnlineHDAddSub_AddedToEachSubClass.csv'
        np.savetxt(outputName, numAddedVecPerClass_OnlineHDAddSub, delimiter=",")
        #standard learning
        outputName = folderOut_ML + '/' + fileName2 + '_OnlineHDAdd_ModelVecsNorm.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, ModelVectorsNorm_OnlineHDAdd.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_OnlineHDAdd_ModelVecs.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, ModelVectors_OnlineHDAdd.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_OnlineHDAdd_AddedToEachSubClass.csv'
        np.savetxt(outputName, numAddedVecPerClass_OnlineHDAdd, delimiter=",")
        #itterative 2class AddAndSubtract
        dataToSave = np.ones((2, HDParams.D)) * np.nan
        dataToSave[0, :] = ModelVectorsNorm_NonSeiz_Itter2class
        dataToSave[1, :] = ModelVectorsNorm_Seiz_Itter2class
        outputName = folderOut_ML + '/' + fileName2 + '_Itter2class_ModelVecsNorm.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, dataToSave, delimiter=",")
        dataToSave = np.ones((2, HDParams.D)) * np.nan
        dataToSave[0, :] = ModelVectors_NonSeiz_Itter2class
        dataToSave[1, :] = ModelVectors_Seiz_Itter2class
        outputName = folderOut_ML + '/' + fileName2 + '_Itter2class_ModelVecs.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, dataToSave, delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_Itter2class_AddedToEachSubClass.csv'
        np.savetxt(outputName, [numAddedVecPerClass_Seiz_Itter2class, numAddedVecPerClass_NonSeiz_Itter2class], delimiter=",")
        #itterative 2class AddOnly
        dataToSave = np.ones((2, HDParams.D)) * np.nan
        dataToSave[0, :] = ModelVectorsNorm_NonSeiz_Itter2classAddOnly
        dataToSave[1, :] = ModelVectorsNorm_Seiz_Itter2classAddOnly
        outputName = folderOut_ML + '/' + fileName2 + '_Itter2classAddOnly_ModelVecsNorm.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, dataToSave, delimiter=",")
        dataToSave = np.ones((2, HDParams.D)) * np.nan
        dataToSave[0, :] = ModelVectors_NonSeiz_Itter2classAddOnly
        dataToSave[1, :] = ModelVectors_Seiz_Itter2classAddOnly
        outputName = folderOut_ML + '/' + fileName2 + '_Itter2classAddOnly_ModelVecs.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, dataToSave, delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_Itter2classAddOnly_AddedToEachSubClass.csv'
        np.savetxt(outputName, [numAddedVecPerClass_Seiz_Itter2classAddOnly, numAddedVecPerClass_NonSeiz_Itter2classAddOnly], delimiter=",")
        #multiclass
        numSubClass_Seiz=  len(numAddedVecPerClassMulti_Seiz)
        numSubClass_NonSeiz = len(numAddedVecPerClassMulti_NonSeiz)
        maxLen=np.max([numSubClass_Seiz,numSubClass_NonSeiz ] )
        dataToSave=np.ones((2,maxLen))*np.nan
        dataToSave[0,0:numSubClass_Seiz]=numAddedVecPerClassMulti_Seiz[0:numSubClass_Seiz]
        dataToSave[1, 0:numSubClass_NonSeiz] = numAddedVecPerClassMulti_NonSeiz[0:numSubClass_NonSeiz]
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClass_AddedToEachSubClass.csv'
        np.savetxt(outputName, dataToSave.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClass_SeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_Seiz.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClass_NonSeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_NonSeiz.transpose(), delimiter=",")
        #multiclass reduced with removal
        numSubClassMultiRedRemov_Seiz=len(ModelVectorsMulti_Seiz_OptimalRemov[:,0])
        numSubClassMultiRedRemov_NonSeiz = len(ModelVectorsMulti_NonSeiz_OptimalRemov[:, 0])
        dataToSave=np.ones((2,maxLen))*np.nan
        dataToSave[0,0:numSubClassMultiRedRemov_Seiz]=numAddedVecPerClass_Seiz_OptimalRemov[0:numSubClassMultiRedRemov_Seiz]
        dataToSave[1, 0:numSubClassMultiRedRemov_NonSeiz] = numAddedVecPerClass_NonSeiz_OptimalRemov[0:numSubClassMultiRedRemov_NonSeiz]
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassReducedRemov_AddedToEachSubClass.csv'
        np.savetxt(outputName, dataToSave.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassReducedRemov_SeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_Seiz_OptimalRemov.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassReducedRemov_NonSeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_NonSeiz_OptimalRemov.transpose(), delimiter=",")
        #multiclass reduced with clustering
        numSubClassMultiRedClust_Seiz=len(ModelVectorsMulti_Seiz_OptimalClust[:,0])
        numSubClassMultiRedClust_NonSeiz = len(ModelVectorsMulti_NonSeiz_OptimalClust[:, 0])
        dataToSave=np.ones((2,maxLen))*np.nan
        dataToSave[0,0:numSubClassMultiRedClust_Seiz]=numAddedVecPerClass_Seiz_OptimalClust[0:numSubClassMultiRedClust_Seiz]
        dataToSave[1, 0:numSubClassMultiRedClust_NonSeiz] = numAddedVecPerClass_NonSeiz_OptimalClust[0:numSubClassMultiRedClust_NonSeiz]
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassReducedClust_AddedToEachSubClass.csv'
        np.savetxt(outputName, dataToSave.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassReducedClust_SeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_Seiz_OptimalClust.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassReducedClust_NonSeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_NonSeiz_OptimalClust.transpose(), delimiter=",")
        #multi class with itterative with removal
        dataToSave=np.ones((2,maxLen))*np.nan
        dataToSave[0,0:numSubClassMultiRedRemov_Seiz]=numAddedVecPerClass_Seiz_RetrainedRemov[0:numSubClassMultiRedRemov_Seiz]
        dataToSave[1, 0:numSubClassMultiRedRemov_NonSeiz] = numAddedVecPerClass_NonSeiz_RetrainedRemov[0:numSubClassMultiRedRemov_NonSeiz]
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassRedItterRemov_AddedToEachSubClass.csv'
        np.savetxt(outputName, dataToSave.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassRedItterRemov_SeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsNorm_Seiz_RetrainedRemov.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassRedItterRemov_NonSeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsNorm_NonSeiz_RetrainedRemov.transpose(), delimiter=",")
        #multi class with itterative with clustering
        dataToSave=np.ones((2,maxLen))*np.nan
        dataToSave[0,0:numSubClassMultiRedClust_Seiz]=numAddedVecPerClass_Seiz_RetrainedClust[0:numSubClassMultiRedClust_Seiz]
        dataToSave[1, 0:numSubClassMultiRedClust_NonSeiz] = numAddedVecPerClass_NonSeiz_RetrainedClust[0:numSubClassMultiRedClust_NonSeiz]
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassRedItterClust_AddedToEachSubClass.csv'
        np.savetxt(outputName, dataToSave.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassRedItterClust_SeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsNorm_Seiz_RetrainedClust.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassRedItterClust_NonSeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsNorm_NonSeiz_RetrainedClust.transpose(), delimiter=",")

    #saving performance per subj
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsRemov_Train.csv'
    np.savetxt(outputName, OptimalValuesRemov_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsRemov_Test.csv'
    np.savetxt(outputName, OptimalValuesRemov_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsClust_Train.csv'
    np.savetxt(outputName, OptimalValuesClust_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsClust_Test.csv'
    np.savetxt(outputName, OptimalValuesClust_test, delimiter=",")

    outputName = folderOut_ML + '/Subj' + pat + '_RandForest_TrainRes.csv'
    np.savetxt(outputName, AllResRF_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_RandForest_TestRes.csv'
    np.savetxt(outputName, AllResRF_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_StandardLearning_TrainRes.csv'
    np.savetxt(outputName, AllRes_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_StandardLearning_TestRes.csv'
    np.savetxt(outputName, AllRes_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_Itter2class_TrainRes.csv'
    np.savetxt(outputName, AllResItter2class_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_Itter2class_TestRes.csv'
    np.savetxt(outputName, AllResItter2class_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_Itter2classAddOnly_TrainRes.csv'
    np.savetxt(outputName, AllResItter2classAddOnly_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_Itter2classAddOnly_TestRes.csv'
    np.savetxt(outputName, AllResItter2classAddOnly_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_TrainRes.csv'
    np.savetxt(outputName, AllResMulti_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_TestRes.csv'
    np.savetxt(outputName, AllResMulti_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReducedRemov_TrainRes.csv'
    np.savetxt(outputName, AllResMultiRedRemov_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReducedRemov_TestRes.csv'
    np.savetxt(outputName, AllResMultiRedRemov_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassRedItterRemov_TrainRes.csv'
    np.savetxt(outputName, AllResItterRemov_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassRedItterRemov_TestRes.csv'
    np.savetxt(outputName, AllResItterRemov_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReducedClust_TrainRes.csv'
    np.savetxt(outputName, AllResMultiRedClust_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReducedClust_TestRes.csv'
    np.savetxt(outputName, AllResMultiRedClust_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassRedItterClust_TrainRes.csv'
    np.savetxt(outputName, AllResItterClust_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassRedItterClust_TestRes.csv'
    np.savetxt(outputName, AllResItterClust_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_OnlineHDAddSub_TrainRes.csv'
    np.savetxt(outputName, AllResOnlineHD_AddSub_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_OnlineHDAddSub_TestRes.csv'
    np.savetxt(outputName, AllResOnlineHD_AddSub_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_OnlineHDAdd_TrainRes.csv'
    np.savetxt(outputName, AllResOnlineHD_Add_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_OnlineHDAdd_TestRes.csv'
    np.savetxt(outputName, AllResOnlineHD_Add_test, delimiter=",")

    #plot performances for this subj for each approach and all cv
    ApproachName=[ 'StandardLearning' , 'Itter2class', 'Itter2classAddOnly', 'MultiClassLearning', 'MultiClassRedRemov', 'MultiClassRedClust', 'MultiClassRedItterRemov','MultiClassRedItterClust', 'OnlineHDAddSub', 'OnlineHDAdd']
    AppShortNames=[ '2C','2Ci','2Cia','MC','MCrr','MCrc', 'MCrir','MCric', 'OnlAS', 'OnlA']
    AppLineStyle=[ 'k','k--','k:','m','r','r--', 'r:', 'm', 'm--', 'm:']
    performancessAll = np.dstack((AllRes_train,AllResItter2class_train, AllResItter2classAddOnly_train, AllResMulti_train, AllResMultiRedRemov_train,AllResMultiRedClust_train,  AllResItterRemov_train,  AllResItterClust_train,AllResOnlineHD_AddSub_train,AllResOnlineHD_Add_train))
    func_plotPerformancesOfDiffApproaches_thisSubj(pat, 'TrainRes', performancessAll, folderOutPredictionsPlot,  ApproachName, AppShortNames, AppLineStyle)
    performancessAll = np.dstack((AllRes_test, AllResItter2class_test,AllResItter2classAddOnly_test, AllResMulti_test, AllResMultiRedRemov_test,AllResMultiRedClust_test,  AllResItterRemov_test,  AllResItterClust_test, AllResOnlineHD_AddSub_test, AllResOnlineHD_Add_test))
    func_plotPerformancesOfDiffApproaches_thisSubj(pat, 'TestRes', performancessAll, folderOutPredictionsPlot,  ApproachName, AppShortNames, AppLineStyle)

    #average times for this subj
    AllSubjTimes[patIndx,:,0]  = np.nanmean(ThisSubjTimes,0)
    AllSubjTimes[patIndx, :, 1] = np.nanstd(ThisSubjTimes, 0)
    #save avrg for this subj
    AllSubjResRF_train[patIndx,:,0]  = np.nanmean(AllResRF_train,0)
    AllSubjResRF_test[patIndx,:,0]  = np.nanmean(AllResRF_test,0)
    AllSubjRes_train[patIndx,:,0]  = np.nanmean(AllRes_train,0)
    AllSubjRes_test[patIndx,:,0]  = np.nanmean(AllRes_test,0)
    AllSubjResMulti_train[patIndx,:,0]  = np.nanmean(AllResMulti_train,0)
    AllSubjResMulti_test[patIndx,:,0] = np.nanmean(AllResMulti_test,0)
    AllSubjResMultiRedRemov_train[patIndx,:,0]  = np.nanmean(AllResMultiRedRemov_train,0)
    AllSubjResMultiRedRemov_test[patIndx,:,0] = np.nanmean(AllResMultiRedRemov_test,0)
    AllSubjResItterRemov_train[patIndx,:,0]  = np.nanmean(AllResItterRemov_train,0)
    AllSubjResItterRemov_test[patIndx,:,0] = np.nanmean(AllResItterRemov_test,0)
    AllSubjResMultiRedClust_train[patIndx,:,0]  = np.nanmean(AllResMultiRedClust_train,0)
    AllSubjResMultiRedClust_test[patIndx,:,0] = np.nanmean(AllResMultiRedClust_test,0)
    AllSubjResItterClust_train[patIndx,:,0]  = np.nanmean(AllResItterClust_train,0)
    AllSubjResItterClust_test[patIndx,:,0] = np.nanmean(AllResItterClust_test,0)
    AllSubjResItter2class_train[patIndx,:,0]  = np.nanmean(AllResItter2class_train,0)
    AllSubjResItter2class_test[patIndx,:,0] = np.nanmean(AllResItter2class_test,0)
    AllSubjResItter2classAddOnly_train[patIndx,:,0]  = np.nanmean(AllResItter2classAddOnly_train,0)
    AllSubjResItter2classAddOnly_test[patIndx,:,0] = np.nanmean(AllResItter2classAddOnly_test,0)
    AllSubjOnlineHDAddSub_train[patIndx,:,0] = np.nanmean(AllResOnlineHD_AddSub_train,0)
    AllSubjOnlineHDAddSub_test[patIndx, :,0] = np.nanmean(AllResOnlineHD_AddSub_test, 0)
    AllSubjOnlineHDAdd_train[patIndx,:,0] = np.nanmean(AllResOnlineHD_Add_train,0)
    AllSubjOnlineHDAdd_test[patIndx, :,0] = np.nanmean(AllResOnlineHD_Add_test, 0)
    AllSubjResRF_train[patIndx,:,1]  = np.nanstd(AllResRF_train,0)
    AllSubjResRF_test[patIndx,:,1]  = np.nanstd(AllResRF_test,0)
    AllSubjRes_train[patIndx,:,1]  = np.nanstd(AllRes_train,0)
    AllSubjRes_test[patIndx,:,1]  = np.nanstd(AllRes_test,0)
    AllSubjResMulti_train[patIndx,:,1]  = np.nanstd(AllResMulti_train,0)
    AllSubjResMulti_test[patIndx,:,1] = np.nanstd(AllResMulti_test,0)
    AllSubjResMultiRedRemov_train[patIndx,:,1]  = np.nanstd(AllResMultiRedRemov_train,0)
    AllSubjResMultiRedRemov_test[patIndx,:,1] = np.nanstd(AllResMultiRedRemov_test,0)
    AllSubjResItterRemov_train[patIndx,:,1]  = np.nanstd(AllResItterRemov_train,0)
    AllSubjResItterRemov_test[patIndx,:,1] = np.nanstd(AllResItterRemov_test,0)
    AllSubjResMultiRedClust_train[patIndx,:,1]  = np.nanstd(AllResMultiRedClust_train,0)
    AllSubjResMultiRedClust_test[patIndx,:,1] = np.nanstd(AllResMultiRedClust_test,0)
    AllSubjResItterClust_train[patIndx,:,1]  = np.nanstd(AllResItterClust_train,0)
    AllSubjResItterClust_test[patIndx,:,1] = np.nanstd(AllResItterClust_test,0)
    AllSubjResItter2class_train[patIndx,:,1]  = np.nanstd(AllResItter2class_train,0)
    AllSubjResItter2class_test[patIndx,:,1] = np.nanstd(AllResItter2class_test,0)
    AllSubjResItter2classAddOnly_train[patIndx,:,1]  = np.nanstd(AllResItter2classAddOnly_train,0)
    AllSubjResItter2classAddOnly_test[patIndx,:,1] = np.nanstd(AllResItter2classAddOnly_test,0)
    AllSubj_OptimalResultsRemov_train[patIndx,:,1] = np.nanstd(OptimalValuesRemov_train,0)
    AllSubj_OptimalResultsRemov_test[patIndx, :,1] = np.nanstd(OptimalValuesRemov_test, 0)
    AllSubj_OptimalResultsClust_train[patIndx,:,1] = np.nanstd(OptimalValuesClust_train,0)
    AllSubj_OptimalResultsClust_test[patIndx, :,1] = np.nanstd(OptimalValuesClust_test, 0)
    AllSubjOnlineHDAddSub_train[patIndx,:,1] = np.nanstd(AllResOnlineHD_AddSub_train,0)
    AllSubjOnlineHDAddSub_test[patIndx, :,1] = np.nanstd(AllResOnlineHD_AddSub_test, 0)
    AllSubjOnlineHDAdd_train[patIndx,:,1] = np.nanstd(AllResOnlineHD_Add_train,0)
    AllSubjOnlineHDAdd_test[patIndx, :,1] = np.nanstd(AllResOnlineHD_Add_test, 0)

    #saving perofmance for all subj
    meanStd=['_mean', '_std']
    for ni, meanStdVal in enumerate(meanStd):
        outputName = folderOut_ML + '/AllSubj_TimeForDiffApproaches'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjTimes[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsRemov_Train'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubj_OptimalResultsRemov_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsRemov_Test'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubj_OptimalResultsRemov_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsClust_Train'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubj_OptimalResultsClust_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsClust_Test'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubj_OptimalResultsClust_test[:,:,ni] , delimiter=",")

        outputName = folderOut_ML + '/AllSubj_RandForest_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResRF_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_RandForest_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResRF_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_StandardLearning_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjRes_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_StandardLearning_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjRes_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMulti_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMulti_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassReducedRemov_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMultiRedRemov_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassReducedRemov_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMultiRedRemov_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassRedItterRemov_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResItterRemov_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassRedItterRemov_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResItterRemov_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassReducedClust_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMultiRedClust_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassReducedClust_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMultiRedClust_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassRedItterClust_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResItterClust_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassRedItterClust_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResItterClust_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_Itter2class_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResItter2class_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_Itter2class_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResItter2class_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_Itter2classAddOnly_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResItter2classAddOnly_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_Itter2classAddOnly_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResItter2classAddOnly_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_OnlineHDAddSub_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjOnlineHDAddSub_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_OnlineHDAddSub_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjOnlineHDAddSub_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_OnlineHDAdd_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjOnlineHDAdd_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_OnlineHDAdd_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjOnlineHDAdd_test[:,:,ni] , delimiter=",")


#####################################################################
#CALCUALTING AVRG FOR ALL SUBJ (USEFUL IF THINGS RESTARTED FOR ONLY SOME SUBJECTS)
AllSubjResRF_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResRF_test = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjRes_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjRes_test = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMulti_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMulti_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResMultiRedRemov_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMultiRedRemov_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResMultiRedClust_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMultiRedClust_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResItterRemov_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResItterRemov_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResItterClust_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResItterClust_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResItter2class_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResItter2class_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResItter2classAddOnly_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResItter2classAddOnly_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjOnlineHDAddSub_train= np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjOnlineHDAddSub_test= np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjOnlineHDAdd_train= np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjOnlineHDAdd_test= np.zeros((len(GeneralParams.patients), 33, 2))
AllSubj_OptimalResultsRemov_train= np.zeros((len(GeneralParams.patients),34, 2)) #3+3+1+9+9+9
AllSubj_OptimalResultsRemov_test= np.zeros((len(GeneralParams.patients),34, 2))
AllSubj_OptimalResultsClust_train= np.zeros((len(GeneralParams.patients),34, 2)) #3+3+1+9+9+9
AllSubj_OptimalResultsClust_test= np.zeros((len(GeneralParams.patients),34, 2))

for patIndx, pat in enumerate(GeneralParams.patients):
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsRemov_Train.csv'
    reader = csv.reader(open(outputName, "r"))
    OptimalValuesRemov_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsRemov_Test.csv'
    reader = csv.reader(open(outputName, "r"))
    OptimalValuesRemov_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsClust_Train.csv'
    reader = csv.reader(open(outputName, "r"))
    OptimalValuesClust_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsClust_Test.csv'
    reader = csv.reader(open(outputName, "r"))
    OptimalValuesClust_test = np.array(list(reader)).astype("float")

    outputName = folderOut_ML + '/Subj' + pat + '_RandForest_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResRF_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_RandForest_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResRF_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_StandardLearning_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllRes_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_StandardLearning_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllRes_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMulti_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMulti_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReducedRemov_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMultiRedRemov_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReducedRemov_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMultiRedRemov_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassRedItterRemov_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResItterRemov_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassRedItterRemov_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResItterRemov_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReducedClust_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMultiRedClust_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReducedClust_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMultiRedClust_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassRedItterClust_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResItterClust_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassRedItterClust_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResItterClust_test = np.array(list(reader)).astype("float")

    outputName = folderOut_ML + '/Subj' + pat + '_Itter2class_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResItter2class_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_Itter2class_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResItter2class_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_Itter2classAddOnly_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResItter2classAddOnly_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_Itter2classAddOnly_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResItter2classAddOnly_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_OnlineHDAddSub_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResOnlineHD_AddSub_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_OnlineHDAddSub_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResOnlineHD_AddSub_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_OnlineHDAdd_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResOnlineHD_Add_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_OnlineHDAdd_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResOnlineHD_Add_test = np.array(list(reader)).astype("float")

    #save avrg for this subj
    AllSubjResRF_train[patIndx,:,0]  = np.nanmean(AllResRF_train,0)
    AllSubjResRF_test[patIndx,:,0]  = np.nanmean(AllResRF_test,0)
    AllSubjRes_train[patIndx,:,0]  = np.nanmean(AllRes_train,0)
    AllSubjRes_test[patIndx,:,0]  = np.nanmean(AllRes_test,0)
    AllSubjResMulti_train[patIndx,:,0]  = np.nanmean(AllResMulti_train,0)
    AllSubjResMulti_test[patIndx,:,0] = np.nanmean(AllResMulti_test,0)
    AllSubjResMultiRedRemov_train[patIndx, :, 0] = np.nanmean(AllResMultiRedRemov_train, 0)
    AllSubjResMultiRedRemov_test[patIndx, :, 0] = np.nanmean(AllResMultiRedRemov_test, 0)
    AllSubjResItterRemov_train[patIndx, :, 0] = np.nanmean(AllResItterRemov_train, 0)
    AllSubjResItterRemov_test[patIndx, :, 0] = np.nanmean(AllResItterRemov_test, 0)
    AllSubjResMultiRedClust_train[patIndx, :, 0] = np.nanmean(AllResMultiRedClust_train, 0)
    AllSubjResMultiRedClust_test[patIndx, :, 0] = np.nanmean(AllResMultiRedClust_test, 0)
    AllSubjResItterClust_train[patIndx, :, 0] = np.nanmean(AllResItterClust_train, 0)
    AllSubjResItterClust_test[patIndx, :, 0] = np.nanmean(AllResItterClust_test, 0)
    AllSubjResItter2class_train[patIndx,:,0]  = np.nanmean(AllResItter2class_train,0)
    AllSubjResItter2class_test[patIndx,:,0] = np.nanmean(AllResItter2class_test,0)
    AllSubjResItter2classAddOnly_train[patIndx,:,0]  = np.nanmean(AllResItter2classAddOnly_train,0)
    AllSubjResItter2classAddOnly_test[patIndx,:,0] = np.nanmean(AllResItter2classAddOnly_test,0)
    AllSubjOnlineHDAddSub_train[patIndx,:,0]  = np.nanmean(AllResOnlineHD_AddSub_train,0)
    AllSubjOnlineHDAddSub_test[patIndx,:,0] = np.nanmean(AllResOnlineHD_AddSub_test,0)
    AllSubjOnlineHDAdd_train[patIndx,:,0]  = np.nanmean(AllResOnlineHD_Add_train,0)
    AllSubjOnlineHDAdd_test[patIndx,:,0] = np.nanmean(AllResOnlineHD_Add_test,0)
    AllSubj_OptimalResultsRemov_train[patIndx,:,0] = np.nanmean(OptimalValuesRemov_train,0)
    AllSubj_OptimalResultsRemov_test[patIndx, :,0] = np.nanmean(OptimalValuesRemov_test, 0)
    AllSubj_OptimalResultsClust_train[patIndx,:,0] = np.nanmean(OptimalValuesClust_train,0)
    AllSubj_OptimalResultsClust_test[patIndx, :,0] = np.nanmean(OptimalValuesClust_test, 0)
    AllSubjResRF_train[patIndx,:,1]  = np.nanstd(AllResRF_train,0)
    AllSubjResRF_test[patIndx,:,1]  = np.nanstd(AllResRF_test,0)
    AllSubjRes_train[patIndx,:,1]  = np.nanstd(AllRes_train,0)
    AllSubjRes_test[patIndx,:,1]  = np.nanstd(AllRes_test,0)
    AllSubjResMulti_train[patIndx,:,1]  = np.nanstd(AllResMulti_train,0)
    AllSubjResMulti_test[patIndx,:,1] = np.nanstd(AllResMulti_test,0)
    AllSubjResMultiRedRemov_train[patIndx, :, 1] = np.nanstd(AllResMultiRedRemov_train, 0)
    AllSubjResMultiRedRemov_test[patIndx, :, 1] = np.nanstd(AllResMultiRedRemov_test, 0)
    AllSubjResItterRemov_train[patIndx, :, 1] = np.nanstd(AllResItterRemov_train, 0)
    AllSubjResItterRemov_test[patIndx, :, 1] = np.nanstd(AllResItterRemov_test, 0)
    AllSubjResMultiRedClust_train[patIndx, :, 1] = np.nanstd(AllResMultiRedClust_train, 0)
    AllSubjResMultiRedClust_test[patIndx, :, 1] = np.nanstd(AllResMultiRedClust_test, 0)
    AllSubjResItterClust_train[patIndx, :, 1] = np.nanstd(AllResItterClust_train, 0)
    AllSubjResItterClust_test[patIndx, :, 1] = np.nanstd(AllResItterClust_test, 0)
    AllSubjResItter2class_train[patIndx,:,1]  = np.nanstd(AllResItter2class_train,0)
    AllSubjResItter2class_test[patIndx,:,1] = np.nanstd(AllResItter2class_test,0)
    AllSubjResItter2classAddOnly_train[patIndx,:,1]  = np.nanstd(AllResItter2classAddOnly_train,0)
    AllSubjResItter2classAddOnly_test[patIndx,:,1] = np.nanstd(AllResItter2classAddOnly_test,0)
    AllSubjOnlineHDAddSub_train[patIndx,:,1]  = np.nanstd(AllResOnlineHD_AddSub_train,0)
    AllSubjOnlineHDAddSub_test[patIndx,:,1] = np.nanstd(AllResOnlineHD_AddSub_test,0)
    AllSubjOnlineHDAdd_train[patIndx,:,1]  = np.nanstd(AllResOnlineHD_Add_train,0)
    AllSubjOnlineHDAdd_test[patIndx,:,1] = np.nanstd(AllResOnlineHD_Add_test,0)
    AllSubj_OptimalResultsRemov_train[patIndx,:,1] = np.nanstd(OptimalValuesRemov_train,0)
    AllSubj_OptimalResultsRemov_test[patIndx, :,1] = np.nanstd(OptimalValuesRemov_test, 0)
    AllSubj_OptimalResultsClust_train[patIndx,:,1] = np.nanstd(OptimalValuesClust_train,0)
    AllSubj_OptimalResultsClust_test[patIndx, :,1] = np.nanstd(OptimalValuesClust_test, 0)

#saving perofmance for all subj
meanStd=['_mean', '_std']
for ni, meanStdVal in enumerate(meanStd):
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsRemov_Train' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubj_OptimalResultsRemov_train[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsRemov_Test' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubj_OptimalResultsRemov_test[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsClust_Train' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubj_OptimalResultsClust_train[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsClust_Test' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubj_OptimalResultsClust_test[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_RandForest_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResRF_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_RandForest_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResRF_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_StandardLearning_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjRes_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_StandardLearning_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjRes_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResMulti_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResMulti_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassReducedRemov_TrainRes' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubjResMultiRedRemov_train[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassReducedRemov_TestRes' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubjResMultiRedRemov_test[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassRedItterRemov_TrainRes' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubjResItterRemov_train[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassRedItterRemov_TestRes' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubjResItterRemov_test[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassReducedClust_TrainRes' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubjResMultiRedClust_train[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassReducedClust_TestRes' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubjResMultiRedClust_test[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassRedItterClust_TrainRes' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubjResItterClust_train[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassRedItterClust_TestRes' + meanStdVal + '.csv'
    np.savetxt(outputName, AllSubjResItterClust_test[:, :, ni], delimiter=",")
    outputName = folderOut_ML + '/AllSubj_Itter2class_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResItter2class_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_Itter2class_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResItter2class_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_Itter2classAddOnly_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResItter2classAddOnly_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_Itter2classAddOnly_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResItter2classAddOnly_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_OnlineHDAddSub_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjOnlineHDAddSub_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_OnlineHDAddSub_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjOnlineHDAddSub_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_OnlineHDAdd_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjOnlineHDAdd_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_OnlineHDAdd_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjOnlineHDAdd_test[:,:,ni] , delimiter=",")

#mean of all subj
TotalMean_RF_train=np.zeros((2,33))
TotalMean_RF_test=np.zeros((2,33))
TotalMean_2class_train=np.zeros((2,33))
TotalMean_2class_test=np.zeros((2,33))
TotalMean_Multi_train=np.zeros((2,33))
TotalMean_Multi_test=np.zeros((2,33))
TotalMean_MultiRedRemov_train=np.zeros((2,33))
TotalMean_MultiRedRemov_test=np.zeros((2,33))
TotalMean_ItterRemov_train=np.zeros((2,33))
TotalMean_ItterRemov_test=np.zeros((2,33))
TotalMean_MultiRedClust_train=np.zeros((2,33))
TotalMean_MultiRedClust_test=np.zeros((2,33))
TotalMean_ItterClust_train=np.zeros((2,33))
TotalMean_ItterClust_test=np.zeros((2,33))
TotalMean_Itter2class_train=np.zeros((2,33))
TotalMean_Itter2class_test=np.zeros((2,33))
TotalMean_Itter2classAddOnly_train=np.zeros((2,33))
TotalMean_Itter2classAddOnly_test=np.zeros((2,33))
TotalMean_OnlineHDAddSub_train=np.zeros((2,33))
TotalMean_OnlineHDAddSub_test=np.zeros((2,33))
TotalMean_OnlineHDAdd_train=np.zeros((2,33))
TotalMean_OnlineHDAdd_test=np.zeros((2,33))
TotalMean_RF_train[0,:]  = np.nanmean(AllSubjResRF_train[:,:,0],0)
TotalMean_RF_test[0,:]  = np.nanmean(AllSubjResRF_test[:,:,0],0)
TotalMean_2class_train[0,:]  = np.nanmean(AllSubjRes_train[:,:,0],0)
TotalMean_2class_test[0,:]  = np.nanmean(AllSubjRes_test[:,:,0],0)
TotalMean_Multi_train[0,:] = np.nanmean(AllSubjResMulti_train[:,:,0],0)
TotalMean_Multi_test[0,:] = np.nanmean(AllSubjResMulti_test[:,:,0],0)
TotalMean_MultiRedRemov_train[0,:]  = np.nanmean(AllSubjResMultiRedRemov_train[:,:,0],0)
TotalMean_MultiRedRemov_test[0,:] = np.nanmean(AllSubjResMultiRedRemov_test[:,:,0],0)
TotalMean_ItterRemov_train[0,:]  = np.nanmean(AllSubjResItterRemov_train[:,:,0],0)
TotalMean_ItterRemov_test[0,:] = np.nanmean(AllSubjResItterRemov_test[:,:,0],0)
TotalMean_MultiRedClust_train[0,:]  = np.nanmean(AllSubjResMultiRedClust_train[:,:,0],0)
TotalMean_MultiRedClust_test[0,:] = np.nanmean(AllSubjResMultiRedClust_test[:,:,0],0)
TotalMean_ItterClust_train[0,:]  = np.nanmean(AllSubjResItterClust_train[:,:,0],0)
TotalMean_ItterClust_test[0,:] = np.nanmean(AllSubjResItterClust_test[:,:,0],0)
TotalMean_Itter2class_train[0,:]  = np.nanmean(AllSubjResItter2class_train[:,:,0],0)
TotalMean_Itter2class_test[0,:] = np.nanmean(AllSubjResItter2class_test[:,:,0],0)
TotalMean_Itter2classAddOnly_train[0,:]  = np.nanmean(AllSubjResItter2classAddOnly_train[:,:,0],0)
TotalMean_Itter2classAddOnly_test[0,:] = np.nanmean(AllSubjResItter2classAddOnly_test[:,:,0],0)
TotalMean_OnlineHDAddSub_train[0,:]  = np.nanmean(AllSubjOnlineHDAddSub_train[:,:,0],0)
TotalMean_OnlineHDAddSub_test[0,:] = np.nanmean(AllSubjOnlineHDAddSub_test[:,:,0],0)
TotalMean_OnlineHDAdd_train[0,:]  = np.nanmean(AllSubjOnlineHDAdd_train[:,:,0],0)
TotalMean_OnlineHDAdd_test[0,:] = np.nanmean(AllSubjOnlineHDAdd_test[:,:,0],0)
TotalMean_RF_train[1,:]  = np.nanstd(AllSubjResRF_train[:,:,0],0)
TotalMean_RF_test[1,:]  = np.nanstd(AllSubjResRF_test[:,:,0],0)
TotalMean_2class_train[1,:]  = np.nanstd(AllSubjRes_train[:,:,0],0)
TotalMean_2class_test[1,:]  = np.nanstd(AllSubjRes_test[:,:,0],0)
TotalMean_Multi_train[1,:] = np.nanstd(AllSubjResMulti_train[:,:,0],0)
TotalMean_Multi_test[1,:] = np.nanstd(AllSubjResMulti_test[:,:,0],0)
TotalMean_MultiRedRemov_train[1,:]  = np.nanstd(AllSubjResMultiRedRemov_train[:,:,0],0)
TotalMean_MultiRedRemov_test[1,:] = np.nanstd(AllSubjResMultiRedRemov_test[:,:,0],0)
TotalMean_ItterRemov_train[1,:]  = np.nanstd(AllSubjResItterRemov_train[:,:,0],0)
TotalMean_ItterRemov_test[1,:] = np.nanstd(AllSubjResItterRemov_test[:,:,0],0)
TotalMean_MultiRedClust_train[1,:]  = np.nanstd(AllSubjResMultiRedClust_train[:,:,0],0)
TotalMean_MultiRedClust_test[1,:] = np.nanstd(AllSubjResMultiRedClust_test[:,:,0],0)
TotalMean_ItterClust_train[1,:]  = np.nanstd(AllSubjResItterClust_train[:,:,0],0)
TotalMean_ItterClust_test[1,:] = np.nanstd(AllSubjResItterClust_test[:,:,0],0)
TotalMean_Itter2class_train[1,:]  = np.nanstd(AllSubjResItter2class_train[:,:,0],0)
TotalMean_Itter2class_test[1,:] = np.nanstd(AllSubjResItter2class_test[:,:,0],0)
TotalMean_Itter2classAddOnly_train[1,:]  = np.nanstd(AllSubjResItter2classAddOnly_train[:,:,0],0)
TotalMean_Itter2classAddOnly_test[1,:] = np.nanstd(AllSubjResItter2classAddOnly_test[:,:,0],0)
TotalMean_OnlineHDAddSub_train[1,:]  = np.nanstd(AllSubjOnlineHDAddSub_train[:,:,0],0)
TotalMean_OnlineHDAddSub_test[1,:] = np.nanstd(AllSubjOnlineHDAddSub_test[:,:,0],0)
TotalMean_OnlineHDAdd_train[1,:]  = np.nanstd(AllSubjOnlineHDAdd_train[:,:,0],0)
TotalMean_OnlineHDAdd_test[1,:] = np.nanstd(AllSubjOnlineHDAdd_test[:,:,0],0)
outputName = folderOut_ML + '/AllSubjAvrg_RandForest_TrainRes.csv'
np.savetxt(outputName, TotalMean_RF_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_RandForest_TestRes.csv'
np.savetxt(outputName, TotalMean_RF_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_StandardLearning_TrainRes.csv'
np.savetxt(outputName, TotalMean_2class_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_StandardLearning_TestRes.csv'
np.savetxt(outputName, TotalMean_2class_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassLearning_TrainRes.csv'
np.savetxt(outputName, TotalMean_Multi_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassLearning_TestRes.csv'
np.savetxt(outputName, TotalMean_Multi_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassReducedRemov_TrainRes.csv'
np.savetxt(outputName, TotalMean_MultiRedRemov_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassReducedRemov_TestRes.csv'
np.savetxt(outputName, TotalMean_MultiRedRemov_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassRedItterRemov_TrainRes.csv'
np.savetxt(outputName, TotalMean_ItterRemov_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassRedItterRemov_TestRes.csv'
np.savetxt(outputName, TotalMean_ItterRemov_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassReducedClust_TrainRes.csv'
np.savetxt(outputName, TotalMean_MultiRedClust_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassReducedClust_TestRes.csv'
np.savetxt(outputName, TotalMean_MultiRedClust_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassRedItterClust_TrainRes.csv'
np.savetxt(outputName, TotalMean_ItterClust_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassRedItterClust_TestRes.csv'
np.savetxt(outputName, TotalMean_ItterClust_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_Itter2class_TrainRes.csv'
np.savetxt(outputName, TotalMean_Itter2class_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_Itter2class_TestRes.csv'
np.savetxt(outputName, TotalMean_Itter2class_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_Itter2classAddOnly_TrainRes.csv'
np.savetxt(outputName, TotalMean_Itter2classAddOnly_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_Itter2classAddOnly_TestRes.csv'
np.savetxt(outputName, TotalMean_Itter2classAddOnly_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_OnlineHDAddSub_TrainRes.csv'
np.savetxt(outputName, TotalMean_OnlineHDAddSub_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_OnlineHDAddSub_TestRes.csv'
np.savetxt(outputName, TotalMean_OnlineHDAddSub_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_OnlineHDAdd_TrainRes.csv'
np.savetxt(outputName, TotalMean_OnlineHDAdd_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_OnlineHDAdd_TestRes.csv'
np.savetxt(outputName, TotalMean_OnlineHDAdd_test, delimiter=",")


######################################################################
######################################################################
#GeneralParams.patients =['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
folderPlotsForPaper=folderOut_ML+'/PlotsForPaper/'
createFolderIfNotExists(folderPlotsForPaper)

# PLOT COMPARISON HD AND RF
plot_perfHDvsRF_onlyTest(folderOut_ML, folderPlotsForPaper,GeneralParams.patients)


# # PLOTTING GRAPHS FOR ITTERATIVE LEARNING
plotItterativeApproachGraphs_violin_onlyTest(folderOut_ML, folderPlotsForPaper)
# #run stat analysis
# runStatistics_ItterativeApproach(folderOut_ML,  'Train')
runStatistics_ItterativeApproach(folderOut_ML,  'Test')

# #PLOTTING GRAPHS FOR MULTICLASS LEARNING
plotMultiClassApproachGraphs_violin_withoutMCc(folderOut_ML, folderPlotsForPaper)
# #run stat analysis
# runStatistics_MultiClassApproach(folderOut_ML,  'Train')
runStatistics_MultiClassApproach(folderOut_ML,  'Test')

##PLOTTING GRAPHS FOR ONLINEHD
plotOnlineHDApproach_weightsAndPerfomance_onlyTest(folderOut_ML, folderPlotsForPaper, GeneralParams)

#FINAL COMPARISON OF ALL APPROACHES
#plotAllApproachGraps_trainAndTest(folderOut_ML, folderPlotsForPaper, GeneralParams)
plotAllApproachGraps_onlyTest(folderOut_ML, folderPlotsForPaper, GeneralParams)

