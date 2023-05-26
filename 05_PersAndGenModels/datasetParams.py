'''
 parameters related to Datasets and specific epilepsy use case
'''

import sys
from baseParams import *


class GeneralDatasetParams(Params):
    def __init__(self, args={}):
        self.CVType = 'LeaveOneOut'  # 'LeaveOneOut' 'RollingBase'
        self.itterativeLearning=0 #0,1
        self.persGenApproach = 'personalized'  # ['personalized', 'generalized', 'gen-retrain-all', 'gen-retrain-seiz', 'gen-retrain-nonSeiz', 'NSpers_Sgen','NSgen_Spers', 'bothModels']
        self.datasetFiltering = '1to30Hz'  # 'Raw', 'MoreNonSeizure_Fact10' #depends if we want all data or some subselection
        self.datasetPreparationType ='' # 'AllDataWin3600s_1File6h'  # 'Fact1', 'Fact10' ,'Fact10_TheSameFile','AllDataWin3600s', 'AllDataWin3600s_1File6h', 'AllDataStoS'  # !!!!!!
        Params.__init__(self, args)

class DatasetPreprocessParams(Params):
    def __init__(self, args={}):

        self.samplFreq = 256  # sampling frequency of data
        self.desired_Fs = [256, 512, 1024] #which frew from orignal dat to keep, they will be resampled to 256

        self.seizLabelNames = ['crise', 'crise\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 ',
                        'crise 1', 'debut de crise', 'debut crise', 'crise1', 'début de crise', 'crise n°1', 'debut', 'crise']

        #channels to keep
        # channels_list = ["F7", "F8", "T3", "T4", 'T5', 'T6']
        # channels_list_mcn = ["F7", "F8", "T7", "T8", 'P7', 'P8']  # MCN nomenclature for EEG electrodes
        # channel_pairs = ["F7-T3", 'F8-T4', 'T3-T5', 'T4-T6']  # Desired channel pairs
        # channel_pairs_mcn = ["F7-T7", 'F8-T8', 'T7-P7', 'T8-P8']
        self.channels_list = ['FP1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'FZ', 'CZ', 'PZ']
        self.channels_list_mcn =  ['FP1', 'F7', 'T7', 'P7', 'O1', 'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8', 'FZ', 'CZ', 'PZ'] # MCN nomenclature for EEG electrodes
        self.channel_pairs = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',  'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',  'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FZ-CZ', 'CZ-PZ']
        # channel_pairs_mcn=  ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',  'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',     'FZ-CZ', 'CZ-PZ']

        #seizures params
        self.seizLen=1  #ideal seizure len [min]
        self.minNumSeiz=3

        # pre and post ictal data to be removed
        self.preIctalTimeToRemove = 0  # in seconds #0, 60
        self.postIctalTimeToRemove = 0  # in seconds #0, 600

        # filtering parameters
        self.borderFreqLow = 1  # Hz for the bandpass butterworth filter
        self.borderFreqHigh = 30  # Hz for the bandpass butterworth filter

        # saving type
        self.saveType = 'gzip'  # 'csv, 'gzip'' #gzip saves a lot of memory so is recommended
        Params.__init__(self, args)


class FeaturesParams(Params):

    def __init__(self, args={}):
        self.winLen = 4  # in seconds, window length on which to calculate features
        self.winStep = 0.5  # in seconds, step of moving window length
        # normalization of feature values or not
        self.normPerFile= 0 #0 - for all train data in one, 1 - per each individual file
        self.featNorm = 'Norm&Discr'  # Discr, Norm&Discr, Norm, None
        self.featNormWith='max' #'max', 'percentile'
        self.featNormPercentile=0.9
        # when we have more labels in one window, how final label is chosen
        self.labelVotingType = 'atLeastOne'  # 'majority', 'allOne', 'atLeastOne'
        # features extracted from data
        # self.featNames = ['MeanAmpl', 'LineLength', 'Frequency']
        self.featNames = ['MeanAmpl', 'LineLength', 'Frequency', 'ZeroCross']
        # self.featNames = ['ZeroCross']
        self.indivFeatNamesMeanAmpl = ['meanAmpl']
        self.indivFeatNamesLL = ['lineLength']
        self.indivFeatNamesFreq = ['p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel',
                                   'p_beta_rel', 'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa',
                                   'p_middle', 'p_beta', 'p_gamma', 'p_tot']
        # ZC features params
        # self.ZCThreshArr = [0.25, 0.50, 0.75, 1, 1.5] #relative values
        self.ZCThreshArr = [16, 32, 64, 128, 256] #absolute values

        Params.__init__(self, args)

        self.allFeatTypesName = '-'.join(self.featNames)
        self.constructAllfeatNames()

    def constructAllfeatNames(self):
        allFeatNames = []
        for fIndx, fName in enumerate(self.featNames):
            if (fName == 'MeanAmpl'):
                allFeatNames.extend(self.indivFeatNamesMeanAmpl)
            elif (fName == 'LineLength'):
                allFeatNames.extend(self.indivFeatNamesLL)
            elif (fName == 'Frequency'):
                allFeatNames.extend(self.indivFeatNamesFreq)
            elif (fName == 'ZeroCross'):
                allFeatNames.extend(['ZC_thr_' + str(0)])
                for i in range(len(self.ZCThreshArr)):
                    allFeatNames.extend(['ZC_thr_' + str(self.ZCThreshArr[i])])
        self.allFeatNames = allFeatNames


class PostprocessingParams(Params):

    def __init__(self, args={}):
        # LABEL SMOOTHING PARAMETERS
        self.seizureStableLenToTest = 5  # in seconds  - window for performin label voting
        self.seizureStablePercToTest = 0.5  # 50% of 1 in last seizureStableLenToTest values that needs to be 1 to finally keep label 1
        self.distanceBetween2Seizures = 5  # in seconds - if seizures are closer then this then they are merged #30
        self.timeBeforeSeizureConsideredAsSeizure = 5  # in seconds - if seizure starts bit before true seizure to still consider ok #30
        self.numFPperDayThr = 1  # for additional meausre of performance what number of FP seizures per days we consider ok

        self.toleranceFPBefSeiz = 5  # in sec #10
        self.toleranceFPAftSeiz = 10  # in sec #30

        self.bayesProbThresh = 1.5 #0, 1.5  # smoothing with cummulative probabilities, threshold from Valentins paper
        Params.__init__(self, args)



class IOParams(Params):
    def __init__(self, args={}, datasetFiltering='', winLen=1, winStep=1, allFeatTypesName='', datasetPreparationType='', normPerFile=0, normType='None', featNormWith='max', featNormPercentile=0.9):
        # rearranged features folder
        if (normPerFile==0):
            normName='TotNorm_'+ normType
        else:
            normName='PerFileNorm_'+ normType
        if normType!='None':
            normName=normName+'_'+featNormWith
        if featNormWith!='max':
            normName=normName+str(featNormPercentile)
        self.outFolderParameters = datasetFiltering + '_' + str(winLen) + '_' + str( winStep) + '_' + allFeatTypesName + '_' + datasetPreparationType+'_'+normName
        Params.__init__(self, args)


def generateParams(optList):
    global GeneralDatasetParams, DatasetPreprocessParams, FeaturesParams, PostprocessingParams, IOParams
    GeneralDatasetParams = GeneralDatasetParams(optList)
    DatasetPreprocessParams = DatasetPreprocessParams(optList)
    FeaturesParams = FeaturesParams(optList)
    PostprocessingParams = PostprocessingParams(optList)
    IOParams = IOParams(optList, GeneralDatasetParams.datasetFiltering, FeaturesParams.winLen, FeaturesParams.winStep,
                        FeaturesParams.allFeatTypesName, GeneralDatasetParams.datasetPreparationType, FeaturesParams.normPerFile, FeaturesParams.featNorm,FeaturesParams.featNormWith, FeaturesParams.featNormPercentile )
    return (GeneralDatasetParams, DatasetPreprocessParams, FeaturesParams, PostprocessingParams, IOParams)


paramModules.append(sys.modules[__name__])
