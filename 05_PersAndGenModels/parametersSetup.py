''' file with all parameters'''
import numpy as np
import pickle
import glob

Dataset='01_Repomse' #'01_CHBMIT', '01_iEEG_Bern'
patients =['01','02','03','04','05','06','07','08','09','10','11', '12','13','14','15','16','17','18','19','20','21','22','23','24']


def constructAllfeatNames(FeaturesParams ):
    allFeatNames=[]
    for fIndx, fName in enumerate(FeaturesParams.featNames):
        if (fName=='MeanAmpl'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_MeanAmpl)
        elif (fName=='LineLength'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_LL)
        elif (fName=='Frequency'):
            allFeatNames.extend(FeaturesParams.indivFeatNames_Freq)
        elif (fName=='ZeroCross'):
            for i in range(len(FeaturesParams.ZC_thresh_arr)):
                allFeatNames.extend(['ZC_thr_'+ str(FeaturesParams.ZC_thresh_arr[i])])
    return(allFeatNames)


class DatasetPreprocessParams:
    minNumSeiz=3

    samplFreq = 256  # sampling frequency of data
    desired_Fs = [256, 512, 1024] #which frew from orignal dat to keep, they will be resampled to 256

    seizLabelNames = ['crise', 'crise\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 ',
                    'crise 1', 'debut de crise', 'debut crise', 'crise1', 'début de crise', 'crise n°1', 'debut', 'crise']

    #channels to keep
    # channels_list = ["F7", "F8", "T3", "T4", 'T5', 'T6']
    # channels_list_mcn = ["F7", "F8", "T7", "T8", 'P7', 'P8']  # MCN nomenclature for EEG electrodes
    # channel_pairs = ["F7-T3", 'F8-T4', 'T3-T5', 'T4-T6']  # Desired channel pairs
    # channel_pairs_mcn = ["F7-T7", 'F8-T8', 'T7-P7', 'T8-P8']
    channels_list = ['FP1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'FZ', 'CZ', 'PZ']
    channels_list_mcn =  ['FP1', 'F7', 'T7', 'P7', 'O1', 'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8', 'FZ', 'CZ', 'PZ'] # MCN nomenclature for EEG electrodes
    channel_pairs = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',  'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',  'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FZ-CZ', 'CZ-PZ']
    # channel_pairs_mcn=  ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',  'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',     'FZ-CZ', 'CZ-PZ']

    #seizureLen ideal
    seizLen=1 #in min

    # #pre and post ictal data to be removed
    # PreIctalTimeToRemove=0 #in seconds #0, 60
    # PostIctalTimeToRemove=0 #in seconds #0, 600
    # PrePostRemoveTrain=0
    # PrePostRemoveTest=0
    #
    # #how to select and rearange data in files before feature extraction and training
    # FileRearangeAllData='AllData_FixedSize' #'AllData_FixedSize','AllData_StoS', 'SubselData_NonSeizAroundSeiz', 'SubselData_NonSeizRandom'
    # FileLen=60 #60, 240  in minutes - only for AllData_FixedSize needed
    # RatioNonSeizSeiz=1 #1, 10 - only for SubselData needed
    #
    # #filtering parameters
    # BorderFreqLow=1#Hz for the bandpass butterworth filter
    # BorderFreqHigh=30 #Hz for the bandpass butterworth filter
    #
    # #saving type
    # SaveType='gzip' #'csv, 'gzip'' #gzip saves a lot of memory so is recommended

class FeaturesParams:
    #window size and step in which is moved
    winLen= 4 #in seconds, window length on which to calculate features
    winStep=0.5 #in seconds, step of moving window length

    #normalization of feature values or not
    featNorm = 'Norm' #'Norm&Discr', 'Norm'

    #when we have more labels in one window, how final label is chosen
    LabelVotingType='atLeastOne' #'majority', 'allOne', 'atLeastOne'

    #features extracted from data
    # featNames = np.array( ['MeanAmpl', 'LineLength', 'Frequency', 'ZeroCross'])
    featNames = np.array(['MeanAmpl', 'LineLength', 'Frequency'])
    allFeatTypesName = '-'.join(featNames)
    indivFeatNames_MeanAmpl=['meanAmpl']
    indivFeatNames_LL=['lineLenth']
    indivFeatNames_Freq = ['p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel', 'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot']

    #ZC features params
    ZC_thresh_arr=[ 0.25, 0.50, 0.75, 1, 1.5]


FeaturesParams.allFeatNames=constructAllfeatNames(FeaturesParams )

class PostprocessingParams:
    #LABEL SMOOTHING PARAMETERS
    seizureStableLenToTest=5 #in seconds  - window for performin label voting
    seizureStablePercToTest=0.5 # 50% of 1 in last seizureStableLenToTest values that needs to be 1 to finally keep label 1
    distanceBetween2Seizures=30 #in seconds - if seizures are closer then this then they are merged
    timeBeforeSeizureConsideredAsSeizure=30 #in seconds - if seizure starts bit before true seizure to still consider ok
    numFPperDayThr=1 #for additional meausre of performance what number of FP seizures per days we consider ok

    toleranceFP_befSeiz=10 #in sec
    toleranceFP_aftSeiz=30 #in sec

    bayesProbThresh= 1.5 #smoothing with cummulative probabilities, threshold from Valentins paper

class GeneralParams:
    patients=patients  #on which subjects to train and test
    plottingON=0  #determines whether some additional plots are plotted
    PersGenApproach='personalized' #'personalized', 'generalized' approaches


class HDParams:
    HDapproachON=1  # whether to use HD (1) or standardML (0)
    HDvecType= 'bin'  #'bin', 'bipol' #binary 0,1, bipolar -1,1
    ItterativeRelearning='off'   # 'on', 'off'
    relearningImprovThresh=0.05
    VotingType='' # '' for closest distance, 'ConfVoting'

    # GENERAL PARAMETERS NEEDED FOR HD
    CUDAdevice = 0 #number of the GPU used
    D = 10000  #dimension of hypervectors
    numSegmentationLevels=20
    similarityType= 'hamming' #'hamming','cosine' #similarity measure used for comparing HD vectors
    vectorTypeCh= 'random' # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2' ... #defines how HD vectors are initialized
    vectorTypeLevel = 'scaleNoRand4'# 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'... #defines how HD vectors are initialized
    roundingTypeForHDVectors='inSteps' #'inSteps','onlyOne','noRounding' #defines how and when HD vectors are binarized
    vectorTypeFeat='random'
    bindingFeatures='ChxFeatxVal' #'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal' #defines how HD vectors encoded
    # normValuesForFeatures=np.zeros((2,numFeat))  #variable to keep track in real time


class StandardMLParams:
    modelType='RandomForest' #'KNN', 'SVM', 'DecisionTree', 'RandomForest','BaggingClassifier','AdaBoost'
    trainingDataResampling='NoResampling' #'NoResampling','ROS','RUS','TomekLinks','ClusterCentroids','SMOTE','SMOTEtomek'
    traininDataResamplingRatio=0.2
    samplingStrategy='default' # depends on resampling, but if 'default' then default for each resampling type, otherwise now implemented only for RUS if not default
    #KNN parameters
    KNN_n_neighbors=5
    KNN_metric='euclidean' #'euclidean', 'manhattan'
    #SVM parameters
    SVM_kernel = 'linear'  # 'linear', 'rbf','poly'
    SVM_C = 1  # 1,100,1000
    SVM_gamma = 'auto' # 0  # 0,10,100
    #DecisionTree and random forest parameters
    DecisionTree_criterion = 'gini'  # 'gini', 'entropy'
    DecisionTree_splitter = 'best'  # 'best','random'
    DecisionTree_max_depth = 0  # 0, 2, 5,10,20
    RandomForest_n_estimators = 100 #10,50, 100,250
    #Bagging, boosting classifier parameters
    Bagging_base_estimator='SVM' #'SVM','KNN', 'DecisionTree'
    Bagging_n_estimators = 100  # 10,50, 100,250


#zero cross feature  parameters
class ZeroCrossFeatureParams:
    EPS_thresh_arr=[16, 32, 64, 128, 256]
    buttFilt_order=4
    buttFilt_freq=20
    samplFreq=256
    winLen=4 # in sec
    winStep=0.5 # in sec
    minValueSignal = np.zeros((len(DatasetPreprocessParams.channel_pairs)))
    meanValueSignal = np.zeros((len(DatasetPreprocessParams.channel_pairs)))  # variable to keep track in real time about min and max value
    cntForLL = 0  # variable to keep track in real time about min and max value



#SAVING SETUP once again to update if new info
with open('../PARAMETERS.pickle', 'wb') as f:
    # pickle.dump([GeneralParams, DatasetPreprocessParams, FeaturesParams, PostprocessingParams, SegSymbParams, SigInfoParams, EEGfreqBands, StandardMLParams, FeaturesParams, ZeroCrossFeatureParams, patients], f)
    pickle.dump([GeneralParams, DatasetPreprocessParams, FeaturesParams, PostprocessingParams,  FeaturesParams, HDParams, StandardMLParams, ZeroCrossFeatureParams, patients], f)