
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"


from HDfunctionsLib import *
from parametersSetup import *
from scipy import interpolate

#################################################################################
#SETUPS
GeneralParams.plottingON=0
GeneralParams.PersGenApproach='personalized'
datasetPreparationType='MoreNonSeizure_Fact5'  # 'MoreNonSeizure_Fact5' , 'MoreNonSeizure_Fact10'
torch.cuda.set_device(HDParams.CUDAdevice)
HDParams.D=10000

optType = 'F1DEgmean'  # 'simpleAcc', 'F1DEgmean'
#MULTI CLASS PARAMS
numSteps = 10
groupingThresh = 0.95
subClassReductApproachType = 'clustering'  # 'removing', 'clustering'
perfDropThr=0.03 #0.01, 0.02, 0.03
#ITTERATIVE LEARNING
ItterType='AddAndSubtract'  #'AddAndSubtract', 'AddOnly'
ItterFact=1
ItterImprovThresh=0.01 #if in threec consecutive runs not bigger improvement then this then stop
savingStepData=1 #whether to save improvements per each itteration

#DATASET
Dataset='01_CHBMIT'
GeneralParams.patients =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
GeneralParams.patients =['01','02','03']


# DEFINING INPUT/OUTPUT FOLDERS
folderIn = '01_datasetProcessed_'+datasetPreparationType+'/'
folderOut0= '03_predictions_' +datasetPreparationType
createFolderIfNotExists(folderOut0)
# folderOut0=folderOut0 +'/'+ str(GeneralParams.PersGenApproach)+'/'
# createFolderIfNotExists(folderOut0)
folderFeaturesOut='02_features_'+datasetPreparationType
createFolderIfNotExists(folderFeaturesOut)
# folderFeaturesOut0=folderFeaturesOut0 +'/'+ str(GeneralParams.PersGenApproach)+'/'
# createFolderIfNotExists(folderFeaturesOut0)

# FEATURS USED -  STANDARD ML FEATURES - 45 FEAT
HDParams.HDapproachON=1
SegSymbParams.symbolType ='StandardMLFeatures'
HDParams.numFeat=45
SegSymbParams.numSegLevels=20 #num dicretized windows for feature values
SegSymbParams.segLenSec = 8 #8 # length of EEG sements in sec
SegSymbParams.slidWindStepSec = 1 #1  # step of sliding window to extract segments in sec
HDParams.vectorTypeLevel = 'scaleNoRand1'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
HDParams.vectorTypeFeat='random'
HDParams.roundingTypeForHDVectors='inSteps' #'inSteps','onlyOne','noRounding'
HDParams.bindingFeatures='FeatxVal' #'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend1000'
HDParams.D=10000
#HDParams.ItterativeRelearning='on'

#calculating various parameters
seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
seizureStablePercToTest = GeneralParams.seizureStablePercToTest
distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz / SegSymbParams.slidWindStepSec)


# #saving parameters to folder name
# folderOutName = SegSymbParams.symbolType +'_'+ str(HDParams.numFeat)+ '_' + str(SegSymbParams.numSegLevels) + '_numFeat' + str(
#     HDParams.numFeat) + '_' + HDParams.bindingFeatures + '_FEATvec' + HDParams.vectorTypeFeat
# folderOutNameFeat = SegSymbParams.symbolType + '_'+ str(HDParams.numFeat)
# folderOutName = folderOutName + '_' + str(SegSymbParams.segLenSec) + '_' + str(
#     SegSymbParams.slidWindStepSec) + 's' + '_' + HDParams.similarityType + '_RND' + HDParams.roundingTypeForHDVectors + '_CHVect' + HDParams.vectorTypeCh + '_LVLVect' + HDParams.vectorTypeLevel+'_D'+str(HDParams.D)
# folderOutNameFeat =folderOutNameFeat+ '_' + str(SegSymbParams.segLenSec) + '_' + str(SegSymbParams.slidWindStepSec) + 's'
# folderOutName=folderOutName+'_MultiClassPaper'
# folderFeaturesOut = folderFeaturesOut0 + folderOutNameFeat
# folderOut_ML = folderOut0 + folderOutName
# createFolderIfNotExists(folderOut_ML)

#final folder to store data to
folderOut_ML =folderOut0 +'/'+optType+'_'+ str(perfDropThr) +'_'+ str(numSteps)
createFolderIfNotExists(folderOut_ML)
print('FOLDER OUT:', folderOut_ML)
print('FOLDER OUT FEATURES:', folderFeaturesOut)
folderOutPredictionsPlot = folderOut_ML+'/Plots_predictions'
createFolderIfNotExists(folderOutPredictionsPlot)


#################################################################################
## CALCULATING FEATURES FOR EACH FILE
numFiles = len(np.sort(glob.glob(folderFeaturesOut + '/*chb' + '*.csv')))
if (numFiles==0):
    print('EXTRACTING FEATURES!!!')
    func_calculateFeaturesForInputFiles(SigInfoParams, SegSymbParams, GeneralParams, HDParams, folderIn, folderFeaturesOut)


#################################################################################
## TRAINING
AllSubjRes_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjRes_test = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMulti_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMulti_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResMultiRed_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMultiRed_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResMultiClust_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMultiClust_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubj_OptimalResultsReduced_train= np.zeros((len(GeneralParams.patients),34, 2)) #3+3+1+9+9+9
AllSubj_OptimalResultsReduced_test= np.zeros((len(GeneralParams.patients),34, 2))
AllSubj_OptimalResultsClustered_train= np.zeros((len(GeneralParams.patients),34, 2)) #3+3+1+9+9+9
AllSubj_OptimalResultsClustered_test= np.zeros((len(GeneralParams.patients),34, 2))

# go through each subject for personalized approach
for patIndx, pat in enumerate(GeneralParams.patients):
    numFiles = len(np.sort(glob.glob(folderFeaturesOut + '/*chb' + pat + '*.csv')))
    print('-- Patient:', pat, 'NumSeizures:', numFiles)

    # go through leave-one-out cross-validations for this subject
    AllRes_train=np.zeros((numFiles,33))
    AllRes_test = np.zeros((numFiles, 33))
    AllResMulti_train = np.zeros((numFiles, 33))
    AllResMulti_test = np.zeros((numFiles, 33))
    AllResMultiRed_train = np.zeros((numFiles, 33))
    AllResMultiRed_test = np.zeros((numFiles, 33))
    AllResMultiClust_train = np.zeros((numFiles, 33))
    AllResMultiClust_test = np.zeros((numFiles, 33))
    OptimalValues_train_red= np.zeros((numFiles, 34))
    OptimalValues_test_red = np.zeros((numFiles, 34))
    OptimalValues_train_clust= np.zeros((numFiles, 34))
    OptimalValues_test_clust = np.zeros((numFiles, 34))
    OptimalValuesClustered_train= np.zeros((numFiles, 34))
    OptimalValuesClustered_test = np.zeros((numFiles, 34))
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
        (dataTrain, label_train)= concatenateDataFromFiles(filesToTrainOn)
        (dataTest, label_test) = concatenateDataFromFiles(filesToTestOn)

        # normalizing data and discretizing
        (data_train_Norm, data_test_Norm, data_train_Discr, data_test_Discr)=normalizeAndDiscretizeTrainAndTestData(dataTrain, dataTest, SegSymbParams.numSegLevels)
        data_train_Discr=data_train_Discr.astype(int)
        data_test_Discr = data_test_Discr.astype(int)

        # INITIALIZING HD VECTORS
        model = HD_classifier_GeneralAndNoCh(SigInfoParams, SegSymbParams, HDParams, HDParams.numFeat*len(SigInfoParams.chToKeep))
        #model = HD_classifier_GeneralWithChCombinations(SigInfoParams, SegSymbParams, HDParams, len(SigInfoParams.chToKeep))

        #################
        #STANDARD SINGLE PASS 2 CLASS LEARNING
        #learn on trainin set
        (ModelVectors, ModelVectorsNorm, numAddedVecPerClass, numLabels) = trainModelVecOnData(data_train_Discr, label_train, model, HDParams)
        #measure performance on test set
        (AllRes_train[cv,:], AllRes_test[cv,:], predLabelsTrain_2class, predLabelsTest_2class)= testModelsAndReturnAllPerformances_2class(data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                  ModelVectorsNorm, HDParams, GeneralParams, SegSymbParams)
        print('2 CLASS acc_train: ', AllRes_train[cv,2], 'acc_test: ', AllRes_test[cv,2])

        #################
        #MULTICLASS LEARNING
        # learn on trainin set
        (ModelVectorsMulti_Seiz, ModelVectorsMultiNorm_Seiz, ModelVectorsMulti_NonSeiz, ModelVectorsMultiNorm_NonSeiz,
         numAddedVecPerClassMulti_Seiz, numAddedVecPerClassMulti_NonSeiz) =trainModelVecOnData_Multiclass(data_train_Discr, label_train, model, HDParams)
        #measure performance  on test set
        (AllResMulti_train[cv,:], AllResMulti_test[cv,:], predLabelsTrain_Multi, predLabelsTest_Multi)=testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
                                                     ModelVectorsMultiNorm_Seiz, ModelVectorsMultiNorm_NonSeiz, HDParams,  GeneralParams, SegSymbParams)
        print('MULTI CLASS acc_train: ', AllResMulti_train[cv,2], 'acc_test: ', AllResMulti_test[cv,2],  'numSubClass_Seiz', len(numAddedVecPerClassMulti_Seiz), 'numSubClass_NonSeiz', len(numAddedVecPerClassMulti_NonSeiz))


        #################
        #ANALYSE REMOVING LESS CROWDED SUBCLASSES
        #REMOVING
        subClassReductApproachType = 'removing'
        (OptimalValues_train_red[cv,:], OptimalValues_test_red[cv,:], ModelVectorsMulti_Seiz_red, ModelVectorsMulti_NonSeiz_red, ModelVectorsMultiNorm_Seiz_red, ModelVectorsMultiNorm_NonSeiz_red, numAddedVecPerClass_Seiz_red,
         numAddedVecPerClass_NonSeiz_red)=reduceNumSubclasses_removingApproach(data_train_Discr, label_train,data_test_Discr, label_test,  model, HDParams, ModelVectorsMulti_Seiz, ModelVectorsMulti_NonSeiz,
                                                                                   ModelVectorsMultiNorm_Seiz, ModelVectorsMultiNorm_NonSeiz, numAddedVecPerClassMulti_Seiz, numAddedVecPerClassMulti_NonSeiz,
                                                                                   numSteps, optType, perfDropThr,  GeneralParams, SegSymbParams, folderOut_ML, fileName2)
        # performance on training and test dataset
        (AllResMultiRed_train[cv, :], AllResMultiRed_test[cv, :], predLabelsTrain_MultiRed, predLabelsTest_MultiRed) = testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
            ModelVectorsMultiNorm_Seiz_red, ModelVectorsMultiNorm_NonSeiz_red, HDParams, GeneralParams, SegSymbParams)
        print('MULTI CLASS REDUCED acc_train: ', AllResMultiRed_train[cv, 2], 'acc_test: ', AllResMultiRed_test[cv, 2], 'numSubClass_Seiz', len(ModelVectorsMulti_Seiz_red[:,0]), 'numSubClass_NonSeiz',  len(ModelVectorsMulti_NonSeiz_red[:,0]))

        #CLUSTERING
        subClassReductApproachType = 'clustering'
        (OptimalValues_train_clust[cv,:], OptimalValues_test_clust[cv,:], ModelVectorsMulti_Seiz_clust, ModelVectorsMulti_NonSeiz_clust, ModelVectorsMultiNorm_Seiz_clust,ModelVectorsMultiNorm_NonSeiz_clust, numAddedVecPerClass_Seiz_clust,
         numAddedVecPerClass_NonSeiz_clust) = reduceNumSubclasses_clusteringApproach(data_train_Discr, label_train,data_test_Discr,label_test, model, HDParams, ModelVectorsMulti_Seiz, ModelVectorsMulti_NonSeiz,
                                                                                     ModelVectorsMultiNorm_Seiz, ModelVectorsMultiNorm_NonSeiz,  numAddedVecPerClassMulti_Seiz,  numAddedVecPerClassMulti_NonSeiz,
                                                                                     numSteps, optType, perfDropThr, groupingThresh,  GeneralParams, SegSymbParams, folderOut_ML, fileName2)
        # performance on training and test dataset
        (AllResMultiClust_train[cv, :], AllResMultiClust_test[cv, :], predLabelsTrain_MultiClust, predLabelsTest_MultiClust) = testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test, model,
            ModelVectorsMultiNorm_Seiz_clust, ModelVectorsMultiNorm_NonSeiz_clust, HDParams, GeneralParams, SegSymbParams)
        print('MULTI CLASS CLUSTER acc_train: ', AllResMultiClust_train[cv, 2], 'acc_test: ', AllResMultiClust_test[cv, 2], 'numSubClass_Seiz', len(ModelVectorsMulti_Seiz_clust[:,0]), 'numSubClass_NonSeiz',  len(ModelVectorsMulti_NonSeiz_clust[:,0]))


        #SAVE PREDICTIONS FOR ALL APPROACHES
        dataToSave_train=np.vstack((label_train, predLabelsTrain_2class,   predLabelsTrain_Multi, predLabelsTrain_MultiRed, predLabelsTrain_MultiClust  )).transpose()
        outputName = folderOut_ML + '/' + fileName2 + '_AllApproaches_TrainPredictions.csv'
        np.savetxt(outputName, dataToSave_train, delimiter=",")
        dataToSave_test=np.vstack((label_test, predLabelsTest_2class,   predLabelsTest_Multi, predLabelsTest_MultiRed,  predLabelsTest_MultiClust )).transpose()
        outputName = folderOut_ML + '/' + fileName2 + '_AllApproaches_TestPredictions.csv'
        np.savetxt(outputName, dataToSave_test, delimiter=",")
        #plot predictions for test
        approachNames = ['2C', 'MC', 'MCred', 'MCclust']
        approachIndx = [1, 2, 4, 6]
        func_plotRawSignalAndPredictionsOfDiffApproaches_thisFile(fileName2, dataToSave_test,dataToSave_train, approachNames, approachIndx, folderIn, folderOutPredictionsPlot, SigInfoParams, GeneralParams, SegSymbParams)

        #SAVE  MODEL VECTORS
        #standard learning
        outputName = folderOut_ML + '/' + fileName2 + '_StandardLearning_ModelVecsNorm.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, ModelVectorsNorm.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_StandardLearning_ModelVecs.csv' #first nonSeiz, then Seiz
        np.savetxt(outputName, ModelVectors.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_StandardLearning_AddedToEachSubClass.csv'
        np.savetxt(outputName, numAddedVecPerClass, delimiter=",")
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
        #multiclass reduced
        numSubClassMultiRed_Seiz=len(ModelVectorsMulti_Seiz_red[:,0])
        numSubClassMultiRed_NonSeiz = len(ModelVectorsMulti_NonSeiz_red[:, 0])
        dataToSave=np.ones((2,maxLen))*np.nan
        dataToSave[0,0:numSubClassMultiRed_Seiz]=numAddedVecPerClass_Seiz_red[0:numSubClassMultiRed_Seiz]
        dataToSave[1, 0:numSubClassMultiRed_NonSeiz] = numAddedVecPerClass_NonSeiz_red[0:numSubClassMultiRed_NonSeiz]
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassReduced_AddedToEachSubClass.csv'
        np.savetxt(outputName, dataToSave.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassReduced_SeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_Seiz_red.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassReduced_NonSeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_NonSeiz_red.transpose(), delimiter=",")
        #multiclass clustered
        numSubClassMultiClust_Seiz=len(ModelVectorsMulti_Seiz_clust[:,0])
        numSubClassMultiClust_NonSeiz = len(ModelVectorsMulti_NonSeiz_clust[:, 0])
        dataToSave=np.ones((2,maxLen))*np.nan
        dataToSave[0,0:numSubClassMultiClust_Seiz]=numAddedVecPerClass_Seiz_clust[0:numSubClassMultiClust_Seiz]
        dataToSave[1, 0:numSubClassMultiClust_NonSeiz] = numAddedVecPerClass_NonSeiz_clust[0:numSubClassMultiClust_NonSeiz]
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassClustered_AddedToEachSubClass.csv'
        np.savetxt(outputName, dataToSave.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassClustered_SeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_Seiz_clust.transpose(), delimiter=",")
        outputName = folderOut_ML + '/' + fileName2 + '_MultiClassClustered_NonSeizModelVecs.csv'
        np.savetxt(outputName, ModelVectorsMultiNorm_NonSeiz_clust.transpose(), delimiter=",")


    #saving performance per subj
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsReduced_Train.csv'
    np.savetxt(outputName, OptimalValues_train_red, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsReduced_Test.csv'
    np.savetxt(outputName, OptimalValues_test_red, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsClustered_Train.csv'
    np.savetxt(outputName, OptimalValues_train_clust, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsClustered_Test.csv'
    np.savetxt(outputName, OptimalValues_test_clust, delimiter=",")

    outputName = folderOut_ML + '/Subj' + pat + '_StandardLearning_TrainRes.csv'
    np.savetxt(outputName, AllRes_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_StandardLearning_TestRes.csv'
    np.savetxt(outputName, AllRes_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_TrainRes.csv'
    np.savetxt(outputName, AllResMulti_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_TestRes.csv'
    np.savetxt(outputName, AllResMulti_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReduced_TrainRes.csv'
    np.savetxt(outputName, AllResMultiRed_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReduced_TestRes.csv'
    np.savetxt(outputName, AllResMultiRed_test, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassClustered_TrainRes.csv'
    np.savetxt(outputName, AllResMultiClust_train, delimiter=",")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassClustered_TestRes.csv'
    np.savetxt(outputName, AllResMultiClust_test, delimiter=",")


    #plot performances for this subj for each approach and all cv
    performancessAll=np.dstack((AllRes_train,AllResMulti_train, AllResMultiRed_train,AllResMultiClust_train  ))
    func_plotPerformancesOfDiffApproaches_thisSubj_multiClassPaper(pat, 'TrainRes', performancessAll, folderOutPredictionsPlot)
    performancessAll = np.dstack((AllRes_test, AllResMulti_test, AllResMultiRed_test, AllResMultiClust_test))
    func_plotPerformancesOfDiffApproaches_thisSubj_multiClassPaper(pat, 'TestRes', performancessAll, folderOutPredictionsPlot)

    #save avrg for this subj
    AllSubjRes_train[patIndx,:,0]  = np.nanmean(AllRes_train,0)
    AllSubjRes_test[patIndx,:,0]  = np.nanmean(AllRes_test,0)
    AllSubjResMulti_train[patIndx,:,0]  = np.nanmean(AllResMulti_train,0)
    AllSubjResMulti_test[patIndx,:,0] = np.nanmean(AllResMulti_test,0)
    AllSubjResMultiRed_train[patIndx,:,0]  = np.nanmean(AllResMultiRed_train,0)
    AllSubjResMultiRed_test[patIndx,:,0] = np.nanmean(AllResMultiRed_test,0)
    AllSubjResMultiClust_train[patIndx,:,0]  = np.nanmean(AllResMultiClust_train,0)
    AllSubjResMultiClust_test[patIndx,:,0] = np.nanmean(AllResMultiClust_test,0)
    AllSubj_OptimalResultsReduced_train[patIndx,:,0] = np.nanmean(OptimalValues_train_red,0)
    AllSubj_OptimalResultsReduced_test[patIndx, :,0] = np.nanmean(OptimalValues_test_red, 0)
    AllSubj_OptimalResultsClustered_train[patIndx,:,0] = np.nanmean(OptimalValues_train_clust,0)
    AllSubj_OptimalResultsClustered_test[patIndx, :,0] = np.nanmean(OptimalValues_test_clust, 0)
    AllSubjRes_train[patIndx,:,1]  = np.nanstd(AllRes_train,0)
    AllSubjRes_test[patIndx,:,1]  = np.nanstd(AllRes_test,0)
    AllSubjResMulti_train[patIndx,:,1]  = np.nanstd(AllResMulti_train,0)
    AllSubjResMulti_test[patIndx,:,1] = np.nanstd(AllResMulti_test,0)
    AllSubjResMultiRed_train[patIndx,:,1]  = np.nanstd(AllResMultiRed_train,0)
    AllSubjResMultiRed_test[patIndx,:,1] = np.nanstd(AllResMultiRed_test,0)
    AllSubjResMultiClust_train[patIndx,:,1]  = np.nanstd(AllResMultiClust_train,0)
    AllSubjResMultiClust_test[patIndx,:,1] = np.nanstd(AllResMultiClust_test,0)
    AllSubj_OptimalResultsReduced_train[patIndx,:,1] = np.nanstd(OptimalValues_train_red,0)
    AllSubj_OptimalResultsReduced_test[patIndx, :,1] = np.nanstd(OptimalValues_test_red, 0)
    AllSubj_OptimalResultsClustered_train[patIndx,:,1] = np.nanstd(OptimalValues_train_clust,0)
    AllSubj_OptimalResultsClustered_test[patIndx, :,1] = np.nanstd(OptimalValues_test_clust, 0)

    #saving perofmance for all subj
    meanStd=['_mean', '_std']
    for ni, meanStdVal in enumerate(meanStd):
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsReduced_Train'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubj_OptimalResultsReduced_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsReduced_Test'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubj_OptimalResultsReduced_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsClustered_Train'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubj_OptimalResultsClustered_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsClustered_Test'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubj_OptimalResultsClustered_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_StandardLearning_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjRes_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_StandardLearning_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjRes_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMulti_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassLearning_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMulti_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassReduced_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMultiRed_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassReduced_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMultiRed_test[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassClustered_TrainRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMultiClust_train[:,:,ni] , delimiter=",")
        outputName = folderOut_ML + '/AllSubj_MultiClassClustered_TestRes'+meanStdVal+'.csv'
        np.savetxt(outputName, AllSubjResMultiClust_test[:,:,ni] , delimiter=",")

######################################################################################

#CALCUALTING AVRG FOR ALL SUBJ (USEFUL IF THINGS RESTARTED FOR ONLY SOME SUBJECTS)
AllSubjRes_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjRes_test = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMulti_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMulti_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResMultiRed_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMultiRed_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubjResMultiClust_train = np.zeros((len(GeneralParams.patients), 33, 2))
AllSubjResMultiClust_test = np.zeros((len(GeneralParams.patients),33, 2))
AllSubj_OptimalResultsRed_train= np.zeros((len(GeneralParams.patients),34, 2)) #3+3+1+9+9+9
AllSubj_OptimalResultsRed_test= np.zeros((len(GeneralParams.patients),34, 2))
AllSubj_OptimalResultsClust_train= np.zeros((len(GeneralParams.patients),34, 2)) #3+3+1+9+9+9
AllSubj_OptimalResultsClust_test= np.zeros((len(GeneralParams.patients),34, 2))
for patIndx, pat in enumerate(GeneralParams.patients):
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsReduced_Train.csv'
    reader = csv.reader(open(outputName, "r"))
    OptimalValuesRed_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsReduced_Test.csv'
    reader = csv.reader(open(outputName, "r"))
    OptimalValuesRed_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsClustered_Train.csv'
    reader = csv.reader(open(outputName, "r"))
    OptimalValuesClust_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassLearning_OptimalResultsClustered_Test.csv'
    reader = csv.reader(open(outputName, "r"))
    OptimalValuesClust_test = np.array(list(reader)).astype("float")

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
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReduced_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMultiRed_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassReduced_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMultiRed_test = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassClustered_TrainRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMultiClust_train = np.array(list(reader)).astype("float")
    outputName = folderOut_ML + '/Subj' + pat + '_MultiClassClustered_TestRes.csv'
    reader = csv.reader(open(outputName, "r"))
    AllResMultiClust_test = np.array(list(reader)).astype("float")


    #save avrg for this subj
    AllSubjRes_train[patIndx,:,0]  = np.nanmean(AllRes_train,0)
    AllSubjRes_test[patIndx,:,0]  = np.nanmean(AllRes_test,0)
    AllSubjResMulti_train[patIndx,:,0]  = np.nanmean(AllResMulti_train,0)
    AllSubjResMulti_test[patIndx,:,0] = np.nanmean(AllResMulti_test,0)
    AllSubjResMultiRed_train[patIndx,:,0]  = np.nanmean(AllResMultiRed_train,0)
    AllSubjResMultiRed_test[patIndx,:,0] = np.nanmean(AllResMultiRed_test,0)
    AllSubjResMultiClust_train[patIndx,:,0]  = np.nanmean(AllResMultiClust_train,0)
    AllSubjResMultiClust_test[patIndx,:,0] = np.nanmean(AllResMultiClust_test,0)
    AllSubj_OptimalResultsRed_train[patIndx,:,0] = np.nanmean(OptimalValuesRed_train,0)
    AllSubj_OptimalResultsRed_test[patIndx, :,0] = np.nanmean(OptimalValuesRed_test, 0)
    AllSubj_OptimalResultsClust_train[patIndx,:,0] = np.nanmean(OptimalValuesClust_train,0)
    AllSubj_OptimalResultsClust_test[patIndx, :,0] = np.nanmean(OptimalValuesClust_test, 0)
    AllSubjRes_train[patIndx,:,1]  = np.nanstd(AllRes_train,0)
    AllSubjRes_test[patIndx,:,1]  = np.nanstd(AllRes_test,0)
    AllSubjResMulti_train[patIndx,:,1]  = np.nanstd(AllResMulti_train,0)
    AllSubjResMulti_test[patIndx,:,1] = np.nanstd(AllResMulti_test,0)
    AllSubjResMultiRed_train[patIndx,:,1]  = np.nanstd(AllResMultiRed_train,0)
    AllSubjResMultiRed_test[patIndx,:,1] = np.nanstd(AllResMultiRed_test,0)
    AllSubjResMultiClust_train[patIndx,:,1]  = np.nanstd(AllResMultiClust_train,0)
    AllSubjResMultiClust_test[patIndx,:,1] = np.nanstd(AllResMultiClust_test,0)
    AllSubj_OptimalResultsRed_train[patIndx,:,1] = np.nanstd(OptimalValuesRed_train,0)
    AllSubj_OptimalResultsRed_test[patIndx, :,1] = np.nanstd(OptimalValuesRed_test, 0)
    AllSubj_OptimalResultsClust_train[patIndx,:,1] = np.nanstd(OptimalValuesClust_train,0)
    AllSubj_OptimalResultsClust_test[patIndx, :,1] = np.nanstd(OptimalValuesClust_test, 0)

#saving perofmance for all subj
meanStd=['_mean', '_std']
for ni, meanStdVal in enumerate(meanStd):
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsReduced_Train'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubj_OptimalResultsRed_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsReduced_Test'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubj_OptimalResultsRed_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsClustered_Train'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubj_OptimalResultsClust_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_OptimalResultsClustered_Test'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubj_OptimalResultsClust_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_StandardLearning_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjRes_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_StandardLearning_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjRes_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResMulti_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassLearning_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResMulti_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassReduced_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResMultiRed_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassReduced_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResMultiRed_test[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassClustered_TrainRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResMultiClust_train[:,:,ni] , delimiter=",")
    outputName = folderOut_ML + '/AllSubj_MultiClassClustered_TestRes'+meanStdVal+'.csv'
    np.savetxt(outputName, AllSubjResMultiClust_test[:,:,ni] , delimiter=",")


#mean of all subj
TotalMean_2class_train=np.zeros((2,33))
TotalMean_2class_test=np.zeros((2,33))
TotalMean_Multi_train=np.zeros((2,33))
TotalMean_Multi_test=np.zeros((2,33))
TotalMean_MultiRed_train=np.zeros((2,33))
TotalMean_MultiRed_test=np.zeros((2,33))
TotalMean_MultiClust_train=np.zeros((2,33))
TotalMean_MultiClust_test=np.zeros((2,33))
TotalMean_2class_train[0,:]  = np.nanmean(AllSubjRes_train[:,:,0],0)
TotalMean_2class_test[0,:]  = np.nanmean(AllSubjRes_test[:,:,0],0)
TotalMean_Multi_train[0,:] = np.nanmean(AllSubjResMulti_train[:,:,0],0)
TotalMean_Multi_test[0,:] = np.nanmean(AllSubjResMulti_test[:,:,0],0)
TotalMean_MultiRed_train[0,:]  = np.nanmean(AllSubjResMultiRed_train[:,:,0],0)
TotalMean_MultiRed_test[0,:] = np.nanmean(AllSubjResMultiRed_test[:,:,0],0)
TotalMean_MultiClust_train[0,:]  = np.nanmean(AllSubjResMultiClust_train[:,:,0],0)
TotalMean_MultiClust_test[0,:] = np.nanmean(AllSubjResMultiClust_test[:,:,0],0)

TotalMean_2class_train[1,:]  = np.nanstd(AllSubjRes_train[:,:,0],0)
TotalMean_2class_test[1,:]  = np.nanstd(AllSubjRes_test[:,:,0],0)
TotalMean_Multi_train[1,:] = np.nanstd(AllSubjResMulti_train[:,:,0],0)
TotalMean_Multi_test[1,:] = np.nanstd(AllSubjResMulti_test[:,:,0],0)
TotalMean_MultiRed_train[1,:]  = np.nanstd(AllSubjResMultiRed_train[:,:,0],0)
TotalMean_MultiRed_test[1,:] = np.nanstd(AllSubjResMultiRed_test[:,:,0],0)
TotalMean_MultiClust_test[1,:] = np.nanstd(AllSubjResMultiClust_test[:,:,0],0)
TotalMean_MultiClust_train[1,:]  = np.nanstd(AllSubjResMultiClust_train[:,:,0],0)

outputName = folderOut_ML + '/AllSubjAvrg_StandardLearning_TrainRes.csv'
np.savetxt(outputName, TotalMean_2class_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_StandardLearning_TestRes.csv'
np.savetxt(outputName, TotalMean_2class_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassLearning_TrainRes.csv'
np.savetxt(outputName, TotalMean_Multi_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassLearning_TestRes.csv'
np.savetxt(outputName, TotalMean_Multi_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassReduced_TrainRes.csv'
np.savetxt(outputName, TotalMean_MultiRed_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassReduced_TestRes.csv'
np.savetxt(outputName, TotalMean_MultiRed_test, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassClustered_TrainRes.csv'
np.savetxt(outputName, TotalMean_MultiClust_train, delimiter=",")
outputName = folderOut_ML + '/AllSubjAvrg_MultiClassClustered_TestRes.csv'
np.savetxt(outputName, TotalMean_MultiClust_test, delimiter=",")


######################################################################################
######################################################################################
######################################################################################
#PLOT PREDICTIONS, PERFORMANCE PER SUBJ AND MODEL
folderOutPredictionsPlots=folderOut_ML+'/Plots_predictions'
createFolderIfNotExists(folderOutPredictionsPlots)

######################################################################################
#PLOTS FOR THE MULTICLASS PAPER - ONE SET OF PARAMETERS (that are set at the beginnign of file)

# plot comparison between 2C, MC, MCred, MCclust performance for this setup
#funct_plotPerformancesForMultiClassPaper_SingleParamsSetup(folderOut_ML)
dataToPlotMean_train=np.dstack((TotalMean_2class_train,TotalMean_Multi_train, TotalMean_MultiRed_train, TotalMean_MultiClust_train))
dataToPlotMean_test=np.dstack((TotalMean_2class_test,TotalMean_Multi_test, TotalMean_MultiRed_test, TotalMean_MultiClust_test))
xLabNames = ['2C', 'MC', 'MCred', 'MCclust']
func_plotAllPerformancesForManyApproaches(dataToPlotMean_train, dataToPlotMean_test, xLabNames, folderOut_ML)

#plot percentage of data per subclasses
GeneralParams.patients =['01','02','03','06', '07'] #plot only for some subjects
func_plotNumDataPerSubclasses_forMultiClassPaper( folderOut_ML, folderOutPredictionsPlot, GeneralParams)

# plotting numsbclasses and performances when itteratively removing or clustering subclasses
folderInRemov=folderOut0 +'/F1DEgmean_0.03_10/ItterativeRemovingSubclasses_numSteps10'
folderInClust=folderOut0 +'/F1DEgmean_0.03_10/ItterativeClusteringSubclasses_numSteps10_PercThr0.95'
func_plotWhenItterativelyRemovingSubclasses_forMultiClassPaper(folderInRemov, folderInClust, folderOut_ML, GeneralParams, numSteps)

######################################################################################
# PLOTTING COMPARISONS BETWEEN DIFFERENT FACTORS
folderPlots = '04_PlotsForPaper/'
createFolderIfNotExists(folderPlots)

datasetPreparationTypeArray=['MoreNonSeizure_Fact1', 'MoreNonSeizure_Fact5', 'MoreNonSeizure_Fact10']
factNames=['Fact1', 'Fact5','Fact10']
folderOutList= []
for foldI, foldN in enumerate(datasetPreparationTypeArray):
    folderOutList.append('03_predictions_' +foldN +'/'+optType+'_'+ str(perfDropThr) +'_'+ str(numSteps) )
# #plot errorbars
# funct_plotPerformancesForMultiClassPaper_ComparisonSeveralParamsSetup(folderOutList, folderPlots)
#plot boxplot only for test smooth
funct_plotPerformancesForMultiClassPaper_ComparisonSeveralParamsSetup_boxplot(folderOutList, folderPlots)

# plotting 6 performances of Fac1, 5, 10 for 2c, MC, MCred and MCclust
funct_plotPerformancesForMultiClassPaper_ComparisonSeveralParamsSetup_graph2(folderOutList, folderPlots, factNames)

#plotting perf imrov and num subclasses after MCred for Fact1, 5, 10
funct_plotPerformancesForMultiClassPaper_ComparisonSeveralParamsSetup_graph3(folderOutList, folderPlots, factNames)
funct_plotPerformancesForMultiClassPaper_ComparisonSeveralParamsSetup_graph3_boxplot(folderOutList, folderPlots, factNames)

