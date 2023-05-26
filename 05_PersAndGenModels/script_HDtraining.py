__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

''' script that performs HD computing (and RF) training and testing of seizures detection
- uses files that are prepared using script_prepareDatasets_...
    - with parameter "datasetPreparationType" possible to choose which dataset to use
- uses 3 different ML models 
    - standard Random Forest (RF) model with 100 trees 
    - standard HD computing where vectors of all samples from the same class are accumulated 
    - online HD computing which used weighted approach so that if current sample vector is alsoready similar it is multiplied with lower weight 
        (this usually helps to prevent majority class dominating model vectors) 
- possible to perform training on 2 ways
    - leave one file out - train on all but that one (this doesn't keep time information into account - that some data was before another one) 
    - rolling base approach - uses all previous files to train for the current file and tests on current file 
- it can perform personalized training or load already trained HD models (e.g. 'generalized', 'NSpers_Sgen', 'NSgen_Spers'
- script saves predictions (raw and also after different predictions smoothing processes) 
- also calculates performance and plots per subject and in average of all subjects 
- in the end compares prediction and performance between different models
'''

import warnings
warnings.filterwarnings('ignore')

from HDfunctionsLib import *
from PersGenVectLib import *
import torch
torch.manual_seed(0)

# Set up parameters
import baseParams
params = baseParams.getArgs(['datasetParams','generalParams'])
from generalParams import *
from datasetParams import *

##########################
# VARIOUS PARAMETERS TO SET
GeneralDatasetParams.itterativeLearning=0 #if we want to pass iteratively through dataset and train
GeneralDatasetParams.persGenApproach = 'personalized'  # ['personalized', 'generalized', 'NSpers_Sgen','NSgen_Spers']
suffixName = '_OnlFact1&'+str(HDParams.onlineFNSfact)+'/' #defines with what factor false non-seizures will be multiplied
stdHDName='ClassicHD' #'StdHD', 'ClassicHD'  # just names how to save StdHD or OnlineHD
onlHDName='OnlineHDAddSub' #'OnlHD', 'OnlineHDAddSub'
methodLists = [stdHDName, onlHDName]
similarityType = 'hamming'  # 'hamming', 'cosine'

# #########################
# # DEFINE INPUT OUTPUT FOLDERS
#
# ## REPOMSE DATASET
# Dataset='01_Repomse'
# folderBase = '../' +Dataset
# createFolderIfNotExists((folderBase))
# folderFeaturesIn= folderBase+ '/02_Features/'

## CHBMIT DATASET
Dataset='01_CHBMIT'
GeneralDatasetParams.datasetPreparationType = 'Fact10_TheSameFile'# 'AllDataWin3600s_1File6h'  # 'Fact1', 'Fact10' ,'Fact10_TheSameFile','AllDataWin3600s', 'AllDataWin3600s_1File6h', 'AllDataStoS'  # !!!!!!
GeneralDatasetParams.CVType = 'RollingBase'  # 'LeaveOneOut' 'RollingBase'
folderBase = '../' +Dataset +'_'+ GeneralDatasetParams.datasetPreparationType
createFolderIfNotExists((folderBase))
folderFeaturesIn= '../../10_datasets/CHBMIT/04_RearangedData_MergedFeatures_' + GeneralDatasetParams.datasetFiltering + '_' + str(FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep) +\
                  '_' + FeaturesParams.allFeatTypesName + '_' + GeneralDatasetParams.datasetPreparationType # where to save calculated features for each original file
subjAll= [ '01', '02', '03', '05', '06','07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']


dataOutFolder = folderBase +  '/04_ResultsPERSGEN_' + IOParams.outFolderParameters+'_D'+str(HDParams.D)
if 'Time' in HDParams.bindingMethod:
    dataOutFolder=dataOutFolder+'_withTime_Random'
    # dataOutFolder=dataOutFolder+HDParams.bindingMethod
    HDParams.timeStepsWinShift=np.array(HDParams.timeStepsInSec)/FeaturesParams.winStep
if (GeneralDatasetParams.CVType== 'LeaveOneOut'):
    dataOutFolder=dataOutFolder+'_L10'
createFolderIfNotExists(dataOutFolder)
folderSimilarities = dataOutFolder+'/Similarities_' + similarityType +suffixName+ '/'
createFolderIfNotExists(folderSimilarities)
if not os.path.exists(dataOutFolder): os.makedirs(dataOutFolder, exist_ok=True)
folderGenVectors=dataOutFolder+'/HD_ModelVectors' + suffixName + '/'
createFolderIfNotExists(folderGenVectors)
folderOutPredictions=dataOutFolder+'/Approach_'+GeneralDatasetParams.persGenApproach+ suffixName+'/'
createFolderIfNotExists(folderOutPredictions)

# Save params to output folder
with open(dataOutFolder + '/params', 'wb') as f:
    pickle.dump(params, f)

#########################
# EXTRACT NUMBER OF PATIENTS
GeneralDatasetParams.patients =[]
if (Dataset=='01_Repomse'):
    subjAll = np.sort(glob.glob(folderFeaturesIn + '/Pat*/' ))
    for patIndx, patfold in enumerate(subjAll):
        path, pat = os.path.split(patfold[:-1])
        GeneralDatasetParams.patients.append(pat)
elif (Dataset=='01_CHBMIT'):
    for patIndx, pat in enumerate(subjAll):
        GeneralDatasetParams.patients.append('Subj'+ pat)


# #################################################################
# INITIALIZING THINGS
# feature indexes to keep
AllExistingFeatures = FeaturesParams.allFeatTypesName
FeaturesParams.featNames = FeaturesParams.allFeatNames
totalNumFeat = len(FeaturesParams.allFeatNames)  # all features that exist in original data
HDParams.numFeat = len(FeaturesParams.featNames)*18 # TODO FIGURE OUT NAMES AND NUMBER OF FEATURES
featIndxs = np.where(np.in1d(FeaturesParams.allFeatNames, FeaturesParams.featNames))[0]
HDParams.numClasses = 2

#generate initalized feature and value vectors as building vectors for model vectors
folderOutHDvectors = dataOutFolder + '/HD_BaseVectors/'
createFolderIfNotExists(folderOutHDvectors)
#if fors the first time running this set of parameters - initialize them and save them
if not os.path.exists(folderOutHDvectors+'HDvec_features.tensor'):
    if 'Time' in HDParams.bindingMethod:
        classicModel = HD_classifier_TimeTracking(HDParams)
        onlineModel = HD_classifier_TimeTracking(HDParams)
    else:
        classicModel = HD_classifier_General(HDParams)
        onlineModel = HD_classifier_General(HDParams)
    torch.save(classicModel.proj_mat_features , folderOutHDvectors+'HDvec_features.tensor')
    torch.save(classicModel.proj_mat_FeatVals, folderOutHDvectors + 'HDvec_values.tensor')
    onlineModel.proj_mat_features = torch.load( folderOutHDvectors+'HDvec_features.tensor')
    onlineModel.proj_mat_FeatVals = torch.load(folderOutHDvectors + 'HDvec_values.tensor')
    classicModel.modelVectorClasses = torch.tensor([0, 1]).to(device=classicModel.device)
    onlineModel.modelVectorClasses = torch.tensor([0, 1]).to(device=classicModel.device)
# if already exists for this set of parameters, load them
# e.g. they were crated when running personalized models, but now for generalized we need to use the same
# to make results comparable
else:
    if 'Time' in HDParams.bindingMethod:
        classicModel = HD_classifier_TimeTracking(HDParams)
        onlineModel = HD_classifier_TimeTracking(HDParams)
    else:
        classicModel = HD_classifier_General(HDParams)
        onlineModel = HD_classifier_General(HDParams)
    classicModel.proj_mat_features = torch.load( folderOutHDvectors+'HDvec_features.tensor')
    classicModel.proj_mat_FeatVals = torch.load(folderOutHDvectors + 'HDvec_values.tensor')
    onlineModel.proj_mat_features = torch.load( folderOutHDvectors+'HDvec_features.tensor')
    onlineModel.proj_mat_FeatVals = torch.load(folderOutHDvectors + 'HDvec_values.tensor')
    classicModel.modelVectorClasses = torch.tensor([0, 1]).to(device=classicModel.device)
    onlineModel.modelVectorClasses = torch.tensor([0, 1]).to(device=classicModel.device)

## Find all subjects that are already processed - but only if personalized
if (GeneralDatasetParams.persGenApproach == 'personalized' ):
    ProcessedFiles=np.sort(glob.glob(folderOutPredictions + '/SubjPat'+ '*'+ 'onlineModel'))
    # ProcessedSubj=[x[3:10] for x in ProcessedFiles]
    ProcessedSubj=[]
    for file in ProcessedFiles:
        folder, fileName= os.path.split(file)
        if (fileName[4:10] not in ProcessedSubj):
            ProcessedSubj.append(fileName[4:10])
    NonProcessedSubj=set(GeneralDatasetParams.patients)- set(ProcessedSubj)
    NonProcessedSubj= list(NonProcessedSubj)
    NonProcessedSubj.sort()
else:
    NonProcessedSubj=GeneralDatasetParams.patients


##############################################################
## TRAINING
for patIndx, pat in enumerate(NonProcessedSubj):
    if (Dataset == '01_Repomse'):
        filesAll0 = np.sort(glob.glob(folderFeaturesIn+'/' + pat + '/'+ pat+ '*'))
        filesAllPic= np.sort(glob.glob(folderFeaturesIn+'/' + pat + '/'+ pat+ '*.png'))
        filesAll=np.array(list(set(filesAll0)-set(filesAllPic)))
    elif (Dataset=='01_CHBMIT'):
        filesAll = np.sort(glob.glob(folderFeaturesIn + '/' + pat + '*.csv.gz'))
    print('-- Patient:', pat, 'NumSeizures:', len(filesAll))

    # load all files only once and mark where each file starts (normalization of feature values will be done later)
    if (FeaturesParams.normPerFile==0): #dont normalize per file
        (dataAll, labelsAll, startIndxOfFiles) = concatenateDataFromFiles(filesAll, Dataset[3:])
    # load data one by one file, while normalizing feature values per file
    else:
        if FeaturesParams.featNorm != "None":
            (dataAll, labelsAll, startIndxOfFiles) = concatenateDataFromFiles_withNormPerFile(filesAll, HDParams, FeaturesParams, Dataset[3:])

    # keep only features of interest
    numCh = round(len(dataAll[0, :]) / totalNumFeat)
    featIndxsAllCh = np.hstack([featIndxs + ch * totalNumFeat for ch in range(numCh)])
    dataAll = dataAll[:, featIndxsAllCh].contiguous()
    labelsAll = labelsAll.contiguous()

    # Leave one out or rolling base
    numCV = len(filesAll) if (GeneralDatasetParams.CVType == 'LeaveOneOut') else len(filesAll) - 1
    ThisSubjTimes = np.zeros((numCV, 3))  # save time for execution of each CV, and for RF, StdHD, OnlHD
    HDParams.numFeat = dataAll.shape[1]


    ## DEFINE MODEL VECTORS (only once for all cv of the same subject)
    if GeneralDatasetParams.persGenApproach == 'personalized' :
        #training from 0
        classicModel.modelVectors = torch.zeros((HDParams.numClasses, HDParams.HD_dim), device=HDParams.device)
        onlineModel.modelVectors = torch.zeros((HDParams.numClasses, HDParams.HD_dim), device=HDParams.device)
        if onlineModel.packed:
            onlineModel.modelVectorsNorm= torch.zeros((onlineModel.numClasses, math.ceil(onlineModel.HD_dim/32)), device = onlineModel.device, dtype=torch.int32) ## before was int32
            classicModel.modelVectorsNorm = torch.zeros((classicModel.numClasses, math.ceil(classicModel.HD_dim / 32)),  device=classicModel.device, dtype=torch.int32)
        else:
            onlineModel.modelVectorsNorm= torch.zeros((onlineModel.numClasses, onlineModel.HD_dim), device = onlineModel.device, dtype=torch.int8)
            classicModel.modelVectorsNorm = torch.zeros((classicModel.numClasses, classicModel.HD_dim), device=classicModel.device, dtype=torch.int8)
        classicModel.numAddedVecPerClass, onlineModel.numAddedVecPerClass = torch.zeros(HDParams.numClasses, device=HDParams.device), torch.zeros( HDParams.numClasses, device=HDParams.device)
    else :
        ## LOADING GENERALIZED MODELS  - MANUALLY UNCOMMENT WHICH ONE TO USE
        # load generalized from the same dataset, created using all subjects but this one
        with open(folderGenVectors + '/GenVectors_StdHD_Subj'+pat+'.pickle', 'rb') as file:
            data = pickle.load(file)
        classicModelGen_modelVectors =torch.from_numpy(data[0]).to(HDParams.device)
        classicModelGen_modelVectorsNorm =packVector(data[1], HDParams.HD_dim, HDParams.device)
        classicModelGen_numAddedVecPerClass = torch.from_numpy(data[2]).to(HDParams.device).squeeze()
        with open(folderGenVectors + '/GenVectors_OnlHD_Subj'+pat+'.pickle', 'rb') as file:
            data = pickle.load(file)
        onlineModelGen_modelVectors=torch.from_numpy(data[0]).to(HDParams.device)
        onlineModelGen_modelVectorsNorm = packVector(data[1], HDParams.HD_dim, HDParams.device)
        onlineModelGen_numAddedVecPerClass  = torch.from_numpy(data[2]).to(HDParams.device).squeeze()

        # #load generalized but one using all subjects
        # with open(folderGenVectors + '/GenVectors_WSub.pickle', 'rb') as file:
        #     data = pickle.load(file)
        # helper = np.array([data[0]['StdHD'][i] for i in [0, 1]])
        # classicModelGen_modelVectors =torch.from_numpy(helper.squeeze()).to(HDParams.device)
        # helper = np.array([data[1]['StdHD'][i] for i in [0, 1]])
        # classicModelGen_modelVectorsNorm = packVector(helper.squeeze(), HDParams.HD_dim, HDParams.device)
        # helper = np.array([data[2]['StdHD'][i] for i in [0, 1]])
        # classicModelGen_numAddedVecPerClass = torch.from_numpy(helper.squeeze()).to(HDParams.device)
        # helper = np.array([data[0]['OnlHD'][i] for i in [0, 1]])
        # onlineModelGen_modelVectors = torch.from_numpy(helper.squeeze()).to(HDParams.device)
        # helper = np.array([data[1]['OnlHD'][i] for i in [0, 1]])
        # onlineModelGen_modelVectorsNorm = packVector(helper.squeeze(), HDParams.HD_dim, HDParams.device)
        # helper = np.array([data[2]['OnlHD'][i] for i in [0, 1]])
        # onlineModelGen_numAddedVecPerClass = torch.from_numpy(helper.squeeze()).to(HDParams.device)
        #
        # #load one gneralizd from all subjects (but where for both classic and onl HD they are in the same file to load from)
        # with open(folderGenVectors + '/GenVectors_WeighedAdding.pickle', 'rb') as file:
        #     data = pickle.load(file)
        # classicModelGen_modelVectors =torch.from_numpy(np.array( [data[0]['StdHD'][i] for i in data[0]['StdHD'].keys()]).squeeze()).to(HDParams.device)
        # classicModelGen_modelVectorsNorm =packVector(np.array( [data[1]['StdHD'][i] for i in data[1]['StdHD'].keys()]).squeeze(), HDParams.HD_dim, HDParams.device)
        # classicModelGen_numAddedVecPerClass = torch.from_numpy(np.array( [data[2]['StdHD'][i] for i in data[2]['StdHD'].keys()]).squeeze()).to(HDParams.device).squeeze()
        # onlineModelGen_modelVectors=torch.from_numpy(np.array( [data[0]['OnlHD'][i] for i in data[0]['OnlHD'].keys()]).squeeze()).to(HDParams.device)
        # onlineModelGen_modelVectorsNorm = packVector(np.array( [data[1]['OnlHD'][i] for i in data[1]['OnlHD'].keys()]).squeeze(), HDParams.HD_dim, HDParams.device)
        # onlineModelGen_numAddedVecPerClass  = torch.from_numpy(np.array( [data[2]['OnlHD'][i] for i in data[2]['OnlHD'].keys()]).squeeze()).to(HDParams.device).squeeze()
        #
        #
        # # #load one gneralized for all subj from different datasets (also select dataset)
        # with open(folderGenVectors + '/REPOMSE_GenVectors_WeighedAdding.pickle', 'rb') as file:
        #     data = pickle.load(file)
        # # with open(folderGenVectors + '/CHBMIT_FACT10_GenVectors_WeighedAdding.pickle', 'rb') as file:
        # #     data = pickle.load(file)
        # # with open(folderGenVectors + '/CHBMIT_ALL_GenVectors_WeighedAdding.pickle', 'rb') as file:
        # #     data = pickle.load(file)
        # classicModelGen_modelVectors =torch.from_numpy(np.array( [data[0]['StdHD'][i] for i in data[0]['StdHD'].keys()]).squeeze()).to(HDParams.device)
        # classicModelGen_modelVectorsNorm =packVector(np.array( [data[1]['StdHD'][i] for i in data[1]['StdHD'].keys()]).squeeze(), HDParams.HD_dim, HDParams.device)
        # classicModelGen_numAddedVecPerClass = torch.from_numpy(np.array( [data[2]['StdHD'][i] for i in data[2]['StdHD'].keys()]).squeeze()).to(HDParams.device).squeeze()
        # onlineModelGen_modelVectors=torch.from_numpy(np.array( [data[0]['OnlHD'][i] for i in data[0]['OnlHD'].keys()]).squeeze()).to(HDParams.device)
        # onlineModelGen_modelVectorsNorm = packVector(np.array( [data[1]['OnlHD'][i] for i in data[1]['OnlHD'].keys()]).squeeze(), HDParams.HD_dim, HDParams.device)
        # onlineModelGen_numAddedVecPerClass  = torch.from_numpy(np.array( [data[2]['OnlHD'][i] for i in data[2]['OnlHD'].keys()]).squeeze()).to(HDParams.device).squeeze()

    # TRAIN AND TEST FOR EACH CV
    for cv in range(numCV):
        fileName2 = 'Subj' + pat + '_cv' + str(cv).zfill(3)
        print(fileName2)
        # create train and test data - LEAVE ONE OUT
        if (GeneralDatasetParams.CVType == 'LeaveOneOut'):
            classicModel.modelVectors = torch.zeros((HDParams.numClasses, HDParams.HD_dim), device=HDParams.device)
            onlineModel.modelVectors = torch.zeros((HDParams.numClasses, HDParams.HD_dim), device=HDParams.device)
            classicModel.numAddedVecPerClass, onlineModel.numAddedVecPerClass = torch.zeros(HDParams.numClasses,device=HDParams.device), torch.zeros( HDParams.numClasses, device=HDParams.device)

            testStart = 0 if cv == 0 else startIndxOfFiles[cv - 1]
            data_test = dataAll[testStart:startIndxOfFiles[cv], :].to(HDParams.device)
            label_test = labelsAll[testStart:startIndxOfFiles[cv]].to(HDParams.device)
            data_train = torch.cat((dataAll[:testStart], dataAll[startIndxOfFiles[cv]:])).to(HDParams.device)
            label_train = torch.cat((labelsAll[:testStart], labelsAll[startIndxOfFiles[cv]:])).to(HDParams.device)
        else:  # RollingBase
            trainStart = 0 if cv == 0 else startIndxOfFiles[cv - 1]
            data_test = dataAll[startIndxOfFiles[cv]:startIndxOfFiles[cv + 1], :].to( HDParams.device)  # test data comes from only one file after this CV
            label_test = labelsAll[startIndxOfFiles[cv]:startIndxOfFiles[cv + 1]].to(HDParams.device)
            data_train = dataAll[0:startIndxOfFiles[cv], :].to(HDParams.device)
            label_train = labelsAll[0:startIndxOfFiles[cv]].to(HDParams.device)

        # normalize and discretize data if needed (for HD has to be discretized too)
        if (FeaturesParams.normPerFile == 0):
            if FeaturesParams.featNorm == "None":
                dataTrain, dataTest = data_train, data_test
            else: #Normalize and discretize
                if FeaturesParams.featNormWith=='max':
                    #normalize with min and max - bad if many outliers
                    dataTrain, dataTest = normalizeAndDiscretizeTrainAndTestData(data_train, data_test,  HDParams.numSegmentationLevels, FeaturesParams.featNorm)
                else:
                    # normalize with percentile
                    dataTrain, dataTest= normalizeAndDiscretizeTrainAndTestData_withPercentile(data_train, data_test, HDParams.numSegmentationLevels, FeaturesParams.featNorm,FeaturesParams.featNormPercentile)
        else: #normalize per file
            dataTrain, dataTest = data_train, data_test

        ## STANDARD ML LEARNING - RANDOM FOREST
        if 'RF' in GeneralParams.testType:
            t0 = time.time()
            StandardMLParams.modelType = 'RandomForest'
            MLstdModel = train_StandardML_moreModelsPossible(dataTrain, label_train, StandardMLParams)
            ThisSubjTimes[cv, 0] = time.time() - t0
            # testing
            #             test_StandardML_moreModelsPossible(folderOutPredictions, fileName2, dataTrain, label_train, MLstdModel, 'Train', PostprocessingParams, FeaturesParams)
            test_StandardML_moreModelsPossible(folderOutPredictions, fileName2, dataTest, label_test, MLstdModel,'Test', PostprocessingParams, FeaturesParams)

        ############################################
        # HD APPROACHES
        # for HD computing and rolling base we dont need to retrain using previous data, but we can continue
        if (GeneralDatasetParams.CVType == 'RollingBase'):
            dataTrain = dataTrain[trainStart:]
            label_train = label_train[trainStart:]

        #setup correct models if it is not personalized
        if (GeneralDatasetParams.persGenApproach != 'personalized'):
            classicModel.modelVectors = classicModelGen_modelVectors
            classicModel.modelVectorsNorm = classicModelGen_modelVectorsNorm
            classicModel.numAddedVecPerClass = classicModelGen_numAddedVecPerClass
            onlineModel.modelVectors = onlineModelGen_modelVectors
            onlineModel.modelVectorsNorm = onlineModelGen_modelVectorsNorm
            onlineModel.numAddedVecPerClass = onlineModelGen_numAddedVecPerClass
        if (GeneralDatasetParams.persGenApproach == 'NSpers_Sgen' or GeneralDatasetParams.persGenApproach == 'NSgen_Spers'  ):
            #load pers model
            with open(dataOutFolder + '/Approach_personalized'+ suffixName+'/'+fileName2+'_classicModel', 'rb') as file:
                modelPers_classic = pickle.load(file)
            with open(dataOutFolder + '/Approach_personalized'+ suffixName+'/'+fileName2+'_onlineModel', 'rb') as file:
                modelPers_online = pickle.load(file)
            if (GeneralDatasetParams.persGenApproach == 'NSpers_Sgen'):
                IndxToCorr=0
            else:
                IndxToCorr=1
            classicModel.modelVectors[IndxToCorr]=modelPers_classic.modelVectors[IndxToCorr]
            classicModel.modelVectorsNorm[IndxToCorr] = modelPers_classic.modelVectorsNorm[IndxToCorr]
            classicModel.numAddedVecPerClass[IndxToCorr] = modelPers_classic.numAddedVecPerClass[IndxToCorr]
            onlineModel.modelVectors[IndxToCorr]=modelPers_online.modelVectors[IndxToCorr]
            onlineModel.modelVectorsNorm[IndxToCorr] = modelPers_online.modelVectorsNorm[IndxToCorr]
            onlineModel.numAddedVecPerClass[IndxToCorr] = modelPers_online.numAddedVecPerClass[IndxToCorr]

        ################
        maxDatasize=1000 #20000 # make smaller if no GPU space
        numDataItter=int(dataTrain.shape[0]/maxDatasize)+1
        numPerfPerThr_classic=np.zeros((100,100,4))
        numPerfPerThr_online=np.zeros((100,100,4))

        # STANDARD SINGLE PASS 2 CLASS LEARNING
        if 'ClassicHD' in GeneralParams.testType:
            torch.cuda.empty_cache()
            if (GeneralDatasetParams.persGenApproach=='personalized'):
                if (GeneralDatasetParams.itterativeLearning == 1):
                    perfPerItter=[]
                    continueItter=1
                    while (continueItter==1): #if std of last three values is less then 2%
                        print('Itter nr: ', str(len(perfPerItter)))
                        print('Perfs are: ', perfPerItter)
                        for i in range(numDataItter):
                            classicModel.trainModelVecOnData_withRetrainPossible( dataTrain[i * maxDatasize:(i + 1) * maxDatasize, :], label_train[i * maxDatasize:(i + 1) * maxDatasize],
                                                                                  GeneralDatasetParams.persGenApproach, HDParams.bindingMethod)

                            numPerfPerThr_classic = numPerfPerThr_classic + analyseOptimalConfidnce(classicModel,  dataTrain[i * maxDatasize:(i + 1) * maxDatasize, :], label_train[i * maxDatasize:(i + 1) * maxDatasize], HDParams.bindingMethod)
                        perf0=measurePerformance(classicModel, PostprocessingParams, FeaturesParams, dataTrain, label_train, HDParams.bindingMethod)
                        perfPerItter.append(perf0)
                        if (len(perfPerItter)>10): #too many itterations
                            continueItter=0
                        if (len(perfPerItter)>3): #not enough improvement
                            if (np.nanstd(perfPerItter[-3:])<0.02):
                                continueItter=0
                            if ((perfPerItter[-1]-perfPerItter[-2])<0):  #if it become worse
                                continueItter=0
                            if (np.nansum(perfPerItter)==0):
                                continueItter=0
                else:
                    for i in range(numDataItter):
                        if (len(label_train) - i * maxDatasize) > 1: #to prevent problm with dimnsions if only one point left
                            classicModel.trainModelVecOnData_withRetrainPossible(dataTrain[i*maxDatasize:(i+1)*maxDatasize,:], label_train[i*maxDatasize:(i+1)*maxDatasize], GeneralDatasetParams.persGenApproach, HDParams.bindingMethod)
                            numPerfPerThr_classic = numPerfPerThr_classic + analyseOptimalConfidnce(classicModel, dataTrain[  i * maxDatasize:(i + 1) * maxDatasize, :], label_train[ i * maxDatasize:( i + 1) * maxDatasize],HDParams.bindingMethod)

                # save model
                classicModel.save(f'{folderOutPredictions}/{fileName2}_classicModel')
                #compare separability of model vecotrs
                measureSeparabilityOfModelVecs(classicModel, HDParams)

                #analyse optimal confidences
                # numPerfPerThr_classic=numPerfPerThr_classic+ analyseOptimalConfidnce(classicModel, dataTrain, label_train, HDParams.bindingMethod)
                # chck the best probab thresholds
                if (len(torch.where((label_train == 1))[0]) > 0):  # if there is any seizre
                    sens = numPerfPerThr_classic[:, :, 0] / (numPerfPerThr_classic[:, :, 0] + numPerfPerThr_classic[:, :, 3])
                    prec = numPerfPerThr_classic[:, :, 1] / (numPerfPerThr_classic[:, :, 1] + numPerfPerThr_classic[:, :, 2])
                    balAcc = (sens + prec) / 2
                    if (np.nansum(balAcc)!=0):
                        thrTS_classic = np.where(balAcc == np.nanmax(balAcc))[0][0]
                        thrTNS_classic = np.where(balAcc == np.nanmax(balAcc))[1][0]
                        print('CV: ', cv, 'THR classic: ', thrTS_classic, thrTNS_classic, 'improv:', (balAcc[thrTS_classic, thrTNS_classic]-balAcc[50,50])*100)
                        HDParams.thrTS=thrTS_classic/100
                        HDParams.thrTNS=thrTNS_classic/100

            testAndSavePredictionsForHD(folderOutPredictions, fileName2, classicModel, PostprocessingParams,FeaturesParams, HDParams, dataTest, label_test, 'ClassicHD', 'Test', 0)
            HDParams.thrTS=0.5
            HDParams.thrTNS=0.5

        #################
        # ONLINE HD - SINGLE PASS BUT WEIGHTNIHG SAMPLES BEFORE ADDING
        if 'OnlineHD' in GeneralParams.testType:
            if (GeneralDatasetParams.persGenApproach == 'personalized'):
                if (GeneralDatasetParams.itterativeLearning == 1):
                    perfPerItter=[]
                    continueItter=1
                    while (continueItter==1): #if std of last three values is less then 2%
                        print('Itter nr: ', str(len(perfPerItter)))
                        print('Perfs are: ', perfPerItter)
                        for i in range(numDataItter):
                            if HDParams.batchSize == 1:
                                onlineHD_ModelVecOnData_withRetrainPossible(dataTrain[i*maxDatasize:(i+1)*maxDatasize,:], label_train[i*maxDatasize:(i+1)*maxDatasize], onlineModel,
                                                                            HDParams.onlineHDType, HDParams.bindingMethod, GeneralDatasetParams.persGenApproach)
                            else:
                                onlineHD_ModelVecOnData_batches_withRetrainPossible(dataTrain[i*maxDatasize:(i+1)*maxDatasize,:], label_train[i*maxDatasize:(i+1)*maxDatasize], onlineModel,
                                                                                    HDParams.onlineHDType, batchSize=HDParams.batchSize, type=GeneralDatasetParams.persGenApproach)

                            numPerfPerThr_online = numPerfPerThr_classic + analyseOptimalConfidnce(onlineModel,dataTrain[i*maxDatasize:(i+1)*maxDatasize,:], label_train[i*maxDatasize:(i+1)*maxDatasize], HDParams.bindingMethod)

                        perf0=measurePerformance(onlineModel, PostprocessingParams, FeaturesParams, dataTrain, label_train, HDParams.bindingMethod)
                        perfPerItter.append(perf0)
                        if (len(perfPerItter)>10): #too many itterations
                            continueItter=0
                        if (len(perfPerItter)>3): #not enough improvement
                            if (np.nanstd(perfPerItter[-3:])<0.02):
                                continueItter=0
                            if ((perfPerItter[-1]-perfPerItter[-2])<0):  #if it become worse
                                continueItter=0
                            if (np.nansum(perfPerItter)==0):
                                continueItter=0
                else:
                    for i in range(numDataItter):
                        if (len(label_train)- i * maxDatasize)>1: #to prevent problm with dimnsions if only one point left
                            if HDParams.batchSize == 1:
                                onlineHD_ModelVecOnData_withRetrainPossible(dataTrain[i * maxDatasize:(i + 1) * maxDatasize, :],label_train[i * maxDatasize:(i + 1) * maxDatasize], onlineModel,
                                                                            HDParams.onlineHDType,HDParams.bindingMethod,  GeneralDatasetParams.persGenApproach)
                            else:
                                onlineHD_ModelVecOnData_batches_withRetrainPossible( dataTrain[i * maxDatasize:(i + 1) * maxDatasize, :], label_train[i * maxDatasize:(i + 1) * maxDatasize], onlineModel,
                                                                                     HDParams.onlineHDType, batchSize=HDParams.batchSize, type=GeneralDatasetParams.persGenApproach)

                            numPerfPerThr_online = numPerfPerThr_online + analyseOptimalConfidnce(onlineModel, dataTrain[ i * maxDatasize:( i + 1) * maxDatasize,:],label_train[i * maxDatasize:(i + 1) * maxDatasize], HDParams.bindingMethod)

                # save model
                onlineModel.save(f'{folderOutPredictions}/{fileName2}_onlineModel')
                # compare separability of model vecotrs
                measureSeparabilityOfModelVecs(onlineModel, HDParams)

                # chcek the best probab thresholds
                if (len(torch.where((label_train == 1))[0]) > 0):  # if there is any seizre
                    sens = numPerfPerThr_online[:, :, 0] / (numPerfPerThr_online[:, :, 0] + numPerfPerThr_online[:, :, 3])
                    prec = numPerfPerThr_online[:, :, 1] / (numPerfPerThr_online[:, :, 1] + numPerfPerThr_online[:, :, 2])
                    balAcc = (sens + prec) / 2
                    if (np.nansum(balAcc)!=0):
                        thrTS_online = np.where(balAcc == np.nanmax(balAcc))[0][0]
                        thrTNS_online = np.where(balAcc == np.nanmax(balAcc))[1][0]
                        print('CV: ', cv, 'THR online: ', thrTS_online, thrTNS_online, 'improv:', (balAcc[thrTS_online, thrTNS_online]-balAcc[50,50])*100)
                        HDParams.thrTS=thrTS_online/100
                        HDParams.thrTNS=thrTNS_online/100

            testAndSavePredictionsForHD(folderOutPredictions, fileName2, onlineModel, PostprocessingParams, FeaturesParams, HDParams, dataTest, label_test, 'OnlineHD', 'Test', 0)
            HDParams.thrTS=0.5
            HDParams.thrTNS=0.5

        #################
        # SAVE  MODEL VECTORS
        # standard learning
        outputName = folderOutPredictions + '/' + fileName2 + '_ClassicHD_ModelVecsNorm.csv'  # first nonSeiz, then Seiz
        saveDataToFile(classicModel.modelVectorsNorm.transpose(1, 0), outputName, 'gzip')
        outputName = folderOutPredictions + '/' + fileName2 + '_ClassicHD_ModelVecs.csv'  # first nonSeiz, then Seiz
        saveDataToFile(classicModel.modelVectors.transpose(1, 0), outputName, 'gzip')
        outputName = folderOutPredictions + '/' + fileName2 + '_ClassicHD_AddedToEachSubClass.csv'
        saveDataToFile(classicModel.numAddedVecPerClass, outputName, 'gzip')
        # Online HD AddSub
        outputName = folderOutPredictions + '/' + fileName2 + '_OnlineHDAddSub_ModelVecsNorm.csv'  # first nonSeiz, then Seiz
        saveDataToFile(onlineModel.modelVectorsNorm.transpose(1, 0), outputName, 'gzip')
        outputName = folderOutPredictions + '/' + fileName2 + '_OnlineHDAddSub_ModelVecs.csv'  # first nonSeiz, then Seiz
        saveDataToFile(onlineModel.modelVectors.transpose(1, 0), outputName, 'gzip')
        outputName = folderOutPredictions + '/' + fileName2 + '_OnlineHDAddSub_AddedToEachSubClass.csv'
        saveDataToFile(onlineModel.numAddedVecPerClass, outputName, 'gzip')
#
# #####################################################################################

# CALCULATE PERFORMANCE BASED ON PREDICTIONS (rerun for all subjects again)
func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderOutPredictions, GeneralDatasetParams.patients,PostprocessingParams, FeaturesParams, 'RF')
func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams, FeaturesParams, 'ClassicHD')
func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams, FeaturesParams, 'OnlineHD')

# MEASURE PERFORMANCE WHEN APPENDING TEST DATA and plot appended predictions in time
func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderOutPredictions, GeneralDatasetParams.patients,PostprocessingParams, FeaturesParams, 'RF')
func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams, FeaturesParams, 'ClassicHD')
func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderOutPredictions, GeneralDatasetParams.patients,PostprocessingParams, FeaturesParams, 'OnlineHD')

# MEASURE PERFORMANCE WHEN APPENDING TEST DATA and plot appended predictions in time - WITH BAYES THRESHOLD OPTIMIZATION
func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending_withBayesThrOpt(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams, FeaturesParams, 'RF')
func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending_withBayesThrOpt(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams, FeaturesParams, 'ClassicHD')
func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending_withBayesThrOpt(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams, FeaturesParams, 'OnlineHD')


###################################
# PLOT COMPARISON BETWEEN RF, STD HD and ONLINE HD PERFORMANCE
# plot predictions in time of all models
modelsList = ['RF', 'ClassicHD', 'OnlineHD']
foldeInBayes=folderOutPredictions + '/PerformanceWithAppendedTests_Bthr'+str(PostprocessingParams.bayesProbThresh)+'/'
folderOutComparison=folderOutPredictions+ '/PredictionsComparison_Bthr'+str(PostprocessingParams.bayesProbThresh)+'/'
func_plotPredictionsOfDifferentModels(modelsList, GeneralDatasetParams.patients, foldeInBayes, f'{folderOutPredictions}/PredictionsComparison/')
#more zoomed in plot
func_plotPredictionsOfDifferentModels_v2(modelsList, GeneralDatasetParams.patients, foldeInBayes,folderOutComparison)


# # ## PLOT COMPARISON BETWEEN RF, STD HD and ONLINE HD AS BOXPLOTS
# for appended CV tests all together
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'Step2', 18, 'Appended', str(PostprocessingParams.bayesProbThresh))
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'Bayes', 27, 'Appended', str(PostprocessingParams.bayesProbThresh))
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'Step1', 9, 'Appended',str(PostprocessingParams.bayesProbThresh))
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'NoSmooth', 0, 'Appended',str(PostprocessingParams.bayesProbThresh))
# for averaging over all CV tests
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'Step2', 18, 'Avrg',str(PostprocessingParams.bayesProbThresh))
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'Bayes', 27, 'Avrg',str(PostprocessingParams.bayesProbThresh))
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'Step1', 9, 'Avrg',str(PostprocessingParams.bayesProbThresh))
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'NoSmooth', 0, 'Avrg',str(PostprocessingParams.bayesProbThresh))
#for optimized bayes
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'Step2', 18, 'Appended', 'Optimized')
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'Bayes', 27, 'Appended',  'Optimized')
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'Step1', 9, 'Appended', 'Optimized')
plot_RFvsSTDHDandONLHD(folderOutPredictions, GeneralDatasetParams.patients, PostprocessingParams,'Test', 'NoSmooth', 0, 'Appended', 'Optimized')


##################################
# PLOT COMPARISON OF PERS; GENERALIZED; PERS_GEN MODELS PREDICTIONS IN TIME
modelsList=['personalized', 'generalized', 'NSpers_Sgen','NSgen_Spers']
foldeOutComparisonModels=dataOutFolder + '/PersGenComparison_4types'
#comparison in time
func_plotPredictionsOfDifferentModels_persGen(modelsList,'ClassicHD', GeneralDatasetParams.patients, dataOutFolder)
func_plotPredictionsOfDifferentModels_persGen(modelsList,'OnlineHD', GeneralDatasetParams.patients, dataOutFolder)

##################################
# PLOT COMPARISON OF PERS; GENERALIZED; PERS_GEN MODELS PERFORMANCE AS BOXPLOTS
# CHBMIT models on Repomse database
modelsList=['personalized',  'generalized','generalized_CHBMITFact10', 'NSpers_Sgen_CHBMITFact10','NSgen_Spers_CHBMITFact10']
foldeOutComparisonModels=dataOutFolder + '/PersGenComparison_5types_CHBMITFact10'
modelsList=['personalized',  'generalized','generalized_CHBMITAll', 'NSpers_Sgen_CHBMITAll','NSgen_Spers_CHBMITAll']
foldeOutComparisonModels=dataOutFolder + '/PersGenComparison_5types_CHBMITAll'
# Repomse models on CHBMIT FACT10
modelsList=['personalized',  'generalized_CHBMIT','generalized_Repomse', 'NSpers_Sgen_Repomse','NSgen_Spers_Repomse']
foldeOutComparisonModels=dataOutFolder + '/PersGenComparison_5types_Repomse'
# rolling base
modelsList=['personalized',  'generalized_CHBMIT', 'generalized', 'NSpers_Sgen','NSgen_Spers']
foldeOutComparisonModels=dataOutFolder + '/PersGenComparison_5types_Repomse'

#comparison of performance - plortting avrage performance boxplots
print('Avrg Bthr 1.5')
plotPerfComp_PersGenModels_AllSmothingTypes(dataOutFolder,  modelsList, GeneralDatasetParams.patients, suffixName, 'Bthr'+str(PostprocessingParams.bayesProbThresh), 'Avrg', foldeOutComparisonModels)
print('Appended Bthr 1.5')
plotPerfComp_PersGenModels_AllSmothingTypes(dataOutFolder,  modelsList, GeneralDatasetParams.patients, suffixName, 'Bthr'+str(PostprocessingParams.bayesProbThresh), 'Appended', foldeOutComparisonModels)
print('Appended Bthr Optimized')
plotPerfComp_PersGenModels_AllSmothingTypes(dataOutFolder,  modelsList, GeneralDatasetParams.patients, suffixName, 'BthrOptimized', 'Appended', foldeOutComparisonModels)

