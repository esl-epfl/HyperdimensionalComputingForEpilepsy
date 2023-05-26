'''
parameters related to HD computing and machine learning models
'''

import sys, time
from baseParams import Params, paramModules, enum


class GeneralParams(Params):

    def __init__(self, args={}):
        self.dataSet = 'Repomse'  # '01_CHBMIT', '01_iEEG_Bern'
        self.testType = enum(['RF', 'ClassicHD', 'OnlineHD'], ['RF', 'ClassicHD', 'OnlineHD'])
        self.plottingON = enum(0, [0, 1])  # determines whether some additional plots are plotted
        self.outputFolder = time.strftime("%Y%m%d-%H%M%S")
        self.testRepeatability = 0
        self.parallelize = 1  # chose whether to parallelize feature extraction per file per core
        self.perc_cores = 0.5  # percetage of cores availables to be charged with data processing
        Params.__init__(self, args)


class HDParams(Params):

    def __init__(self, args={}):
        self.HDapproachON = 1  # whether to use HD (1) or standardML (0)
        self.HDvecType = 'bin'  # 'bin', 'bipol' #binary 0,1, bipolar -1,1
        self.iterativeRelearning = 'off'  # 'on', 'off'
        self.relearningImprovThresh = 0.05
        self.votingType = ''  # '' for closest distance, 'ConfVoting'
        self.featNorm = 'Norm&Discr'  # Discr
        # GENERAL PARAMETERS NEEDED FOR HD
        self.device = 0 # 'cuda'  # device to use (cpu, cuda)
        self.packed = True # True
        self.D =9984  #4800# 9984  10000 , 4992, 2496 # dimension of hypervectors - needs to be multiplier of 32 for Time and Append
        self.numClasses = 2
        self.numSegmentationLevels = 20
        self.similarityType = 'hamming'  # 'hamming','cosine' #similarity measure used for comparing HD vectors
        self.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2' ... #defines how HD vectors are initialized
        self.vectorTypeLevel = enum('random', ['random', 'sandwich', 'scaleNoRand1', 'scaleNoRand2', 'scaleRand1',
                                                     'scaleRand2','scaleWithRadius10'])  # defines how HD vectors are initialized
        self.roundingTypeForHDVectors = 'inSteps'  # 'inSteps','onlyOne','noRounding' #defines how and when HD vectors are binarized
        self.vectorTypeFeat = 'random'
        self.bindingMethod = 'FeatxVal'  # 'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal' #defines how HD vectors encoded
        # self.bindingMethod = 'Time&FeatxVal'  # 'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal' #defines how HD vectors encoded
        # #different time binding approaches
        # self.bindingMethod='TimeDiffAbsApp' #'TimeDiffAbsApp', 'TimeDiffRelApp', 'TimeValApp', 'TimeDiffAbsSum', 'TimeDiffAbsPerm', 'TimeValSum','TimeValPerm'
        # self.timeStepsInSec=[0, 10,30,60, 120] #if time taken into account
        self.timeStepsInSec = [0, 120]
        # self.timeStepsInSec = [0]

        # OnlineHD params
        self.onlineHDType = 'Always_AddAndSubtract'  # 'Always_AddAndSubtract', 'WrongOnly_AddAndSubtract'
        self.batchSize = 1
        self.onlineHDLearnRate = 1  # 0.5,0.2
        self.onlineFNSfact=1 #1,10,100,1000

        # Confidence thresholds to classify something seizure or nonseizure
        self.thrTS = 0.5
        self.thrTNS = 0.5

        if 'Time' in self.bindingMethod:
            self.HD_dim = self.D *len(self.timeStepsInSec)
        else:
            self.HD_dim=self.D
        Params.__init__(self, args)


class StandardMLParams(Params):

    def __init__(self, args={}):
        self.modelType = 'RandomForest'  # 'KNN', 'SVM', 'DecisionTree', 'RandomForest','BaggingClassifier','AdaBoost'
        self.trainingDataResampling = 'NoResampling'  # 'NoResampling','ROS','RUS','TomekLinks','ClusterCentroids','SMOTE','SMOTEtomek'
        self.samplingStrategy = 'default'  # depends on resampling, but if 'default' then default for each resampling type, otherwise now implemented only for RUS if not default
        # KNN parameters
        self.KNNNumNeighbors = 5
        self.KNNMetric = 'euclidean'  # 'euclidean', 'manhattan'
        # SVM parameters
        self.SVMKernel = 'linear'  # 'linear', 'rbf','poly'
        self.SVMC = 1  # 1,100,1000
        self.SVMGamma = 'auto'  # 0  # 0,10,100
        # DecisionTree and random forest parameters
        self.decisionTreeCriterion = 'gini'  # 'gini', 'entropy'
        self.decisionTreeSplitter = 'best'  # 'best','random'
        self.decisionTreeMaxDepth = 0  # 0, 2, 5,10,20
        self.randomForestNumEstimators = 100  # 10,50, 100,250
        # Bagging, boosting classifier parameters
        self.baggingBaseEstimator = 'SVM'  # 'SVM','KNN', 'DecisionTree'
        self.baggingNumEstimators = 100  # 10,50, 100,250
        Params.__init__(self, args)


# zero cross feature  parameters
class ZeroCrossFeatureParams(Params):
    def __init__(self, args={}):
        self.EPSThreshArr = [16, 32, 64, 128, 256]
        self.buttFiltOrder = 4
        self.buttFiltFreq = 20
        self.samplFreq = 256
        self.winLen = 4  # in sec
        self.winStep = 0.5  # in sec
        self.cntForLL = 0  # variable to keep track in real time about min and max value
        Params.__init__(self, args)


def generateParams(optList):
    global GeneralParams, StandardMLParams, HDParams, ZeroCrossFeatureParams
    GeneralParams = GeneralParams(optList)
    StandardMLParams = StandardMLParams(optList)
    HDParams = HDParams(optList)
    ZeroCrossFeatureParams = ZeroCrossFeatureParams(optList)
    return (GeneralParams, StandardMLParams, HDParams, ZeroCrossFeatureParams)


paramModules.append(sys.modules[__name__])
