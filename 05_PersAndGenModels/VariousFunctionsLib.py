''' library including various functions for HD project but not necessarily related to HD vectors'''
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

import os, pathlib
import threading
import math
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import glob
import csv
import multiprocessing as mp
import time
from math import ceil
import math
import sklearn
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pywt
from entropy import *
import scipy
import sys
import pyedflib
import MITAnnotation as MIT
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import scipy.io
import random
from PerformanceMetricsLib import *
from scipy import signal
from scipy import interpolate
import pandas as pd
import seaborn as sns
import scipy.io
import antropy as ant
import imblearn
from featuresLib import *


#
# ########################################################
# #global variables for multi-core operation
# number_free_cores = 0
# n_cores_semaphore = 1
#
# files_processed=[]
# n_files_per_patient=[]
# # number_free_cores = 0
# number_free_cores_pat = 0
# # n_cores_semaphore = 1
#
def createFolderIfNotExists(folderOut):
    ''' creates folder if doesnt already exist
    warns if creation failed '''
    if not os.path.exists(folderOut):
        try:
            os.mkdir(folderOut)
        except OSError:
            print("Creation of the directory %s failed" % folderOut)


#
# def smoothenLabels(prediction,  seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx):
#     ''' returns labels after two steps of postprocessing
#     first moving window with voting  - if more then threshold of labels are 1 final label is 1 otherwise 0
#     second merging seizures that are too close '''
#
#     #labels = labels.reshape(len(labels))
#     smoothLabelsStep1=np.zeros((len(prediction)))
#     smoothLabelsStep2=np.zeros((len(prediction)))
#     # try:
#     #     a=int(seizureStableLenToTestIndx)
#     # except:
#     #     print('error seizureStableLenToTestIndx')
#     #     print(seizureStableLenToTestIndx)
#     # try:
#     #     a=int(len(prediction))
#     # except:
#     #     print('error prediction')
#     #     print(prediction)
#     #first classifying as true 1 if at laest  GeneralParams.seizureStableLenToTest in a row is 1
#     for i in range(int(seizureStableLenToTestIndx), int(len(prediction))):
#         s= sum( prediction[i-seizureStableLenToTestIndx+1: i+1] )/seizureStableLenToTestIndx
#         try:
#             if (s>= seizureStablePercToTest):  #and prediction[i]==1
#                 smoothLabelsStep1[i]=1
#         except:
#             print('error')
#     smoothLabelsStep2=np.copy(smoothLabelsStep1)
#
#     #second part
#     prevSeizureEnd=-distanceBetweenSeizuresIndx
#     for i in range(1,len(prediction)):
#         if (smoothLabelsStep2[i] == 1 and smoothLabelsStep2[i-1] == 0):  # new seizure started
#             # find end of the seizure
#             j = i
#             while (smoothLabelsStep2[j] == 1 and j< len(smoothLabelsStep2)-1):
#                 j = j + 1
#             #if current seizure distance from prev seizure is too close merge them
#             if ((i - prevSeizureEnd) < distanceBetweenSeizuresIndx):  # if  seizure started but is too close to previous one
#                 #delete secon seizure
#                 #prevSeizureEnd = j
#                 #[i:prevSeizureEnd]=np.zeros((prevSeizureEnd-i-1)) #delete second seizure - this was before
#                 #concatenate seizures
#                 if (prevSeizureEnd<0): #if exactly first seizure
#                     prevSeizureEnd=0
#                 smoothLabelsStep2[prevSeizureEnd:j] = np.ones((j - prevSeizureEnd ))
#             prevSeizureEnd = j
#             i=prevSeizureEnd
#
#     return  (smoothLabelsStep2, smoothLabelsStep1)
#
#
def readEdfFile (fileName):
    ''' reads .edf file and returnes  data[numSamples, numCh], sampling frequency, names of channels'''
    f = pyedflib.EdfReader(fileName)
    n = f.signals_in_file
    channelNames = f.getSignalLabels()
    f.getSampleFrequency(0)
    samplFreq= data = np.zeros(( f.getNSamples()[0], n))
    for i in np.arange(n):
        data[:, i] = f.readSignal(i)
    return (data, samplFreq, channelNames)

# def writeToCsvFile( data, labels,  fileName):
#     ''' function taht aves data and labels in .csv file'''
#     outputName= fileName+'.csv'
#     myFile = open(outputName, 'w',newline='')
#     dataToWrite=np.column_stack((data, labels))
#     with myFile:
#         writer = csv.writer(myFile)
#         writer.writerows(dataToWrite)
#
def exportNonSeizFile(fileName, folderOut, pat, FileOutIndx, channelNamesToKeep):
    ''' load .edf file that doesn't contain seizure and export it as gzip file
    each column represents values of one channel and the last column is labels
    1 is seizure and 0 is non seizure
    also channels that are not in interest are removed '''
    allGood = 1
    (rec, samplFreq, channels) = readEdfFile(fileName)
    # take only the channels we need and in correct order
    try:
        chToKeepAndInCorrectOrder = [channels.index(channelNamesToKeep[i]) for i in range(len(channelNamesToKeep))]
    except:
        print('Sth wrong with the channels in a file: ', fileName)
        allGood = 0
    if (allGood == 1):
        newData = rec[1:, chToKeepAndInCorrectOrder]
        (lenSig, numCh) = newData.shape
        newLabel = np.zeros(lenSig)
        # saving
        fileNameOut = folderOut + '/Subj' + pat + '_f' + str(FileOutIndx).zfill(3)
        print(fileNameOut)
        saveDataToFile(np.hstack((newData, np.reshape(newLabel, (-1, 1)))), fileNameOut, 'gzip')
        FileOutIndx=FileOutIndx+1
    return(FileOutIndx)

def exportSeizFile(fileName, folderOut, pat, FileOutIndx, channelNamesToKeep):
    ''' load .edf file that contains seizure and export it as gzip file
    each column represents values of one channel and the last column is labels
    1 is seizure and 0 is non seizure
    also channels that are not in interest are removed '''
    allGood = 1
    fileName0 = os.path.splitext(fileName)[0]  # removing .seizures from the string
    # here replaced reading .hea files with .edf reading to avoid converting !!!
    (rec, samplFreq, channels) = readEdfFile(fileName)
    # take only the channels we need and in correct order
    try:
        chToKeepAndInCorrectOrder = [channels.index(channelNamesToKeep[i]) for i in   range(len(channelNamesToKeep))]
    except:
        print('Sth wrong with the channels in a file: ', fileName)
        allGood = 0
    if (allGood == 1):
        newData = rec[1:, chToKeepAndInCorrectOrder]
        (lenSig, numCh) = newData.shape
        newLabel = np.zeros(lenSig)
        # read times of seizures
        szStart = [a for a in MIT.read_annotations(fileName+'.seizures') if a.code == 32]  # start marked with '[' (32)
        szStop = [a for a in MIT.read_annotations(fileName+'.seizures') if a.code == 33]  # start marked with ']' (33)
        # for each seizure cut it out and save (with few parameters)
        numSeizures = len(szStart)
        for i in range(numSeizures):
            seizureLen = szStop[i].time - szStart[i].time
            newLabel[int(szStart[i].time):int(szStop[i].time)] = np.ones(seizureLen)
        # saving
        fileNameOut = folderOut + '/Subj' + pat + '_f' + str(FileOutIndx).zfill(3) + '_s'
        print(fileNameOut)
        saveDataToFile(np.hstack((newData, np.reshape(newLabel, (-1, 1)))), fileNameOut, 'gzip')
        FileOutIndx=FileOutIndx+1
    return(FileOutIndx)

def extractEDFdataToCSV_originalData_gzip(folderIn, folderOut, GeneralParams, GeneralCHBMITParams ,DatasetPreprocessParams):
    ''' converts data from edf format to csv using gzip compression
    20210705 UnaPale'''
    createFolderIfNotExists(folderOut)

    global number_free_cores
    print('Extracting .csv from CHB edf files')
    if GeneralParams.parallelize:
        n_cores = mp.cpu_count()
        n_cores = ceil(n_cores * GeneralParams.perc_cores)

        if n_cores > len(GeneralCHBMITParams.patients):
            n_cores = len(GeneralCHBMITParams.patients)

        print('Number of used cores: ' + str(n_cores))

        pool = mp.Pool(processes =n_cores)
        number_free_cores = n_cores

    for pat in GeneralCHBMITParams.patients:
        print('-- Patient:', pat)
        PATIENT = pat if len(sys.argv) < 2 else '{0:02d}'.format(int(sys.argv[1]))
        #number of Seiz and nonSeiz files
        SeizFiles=sorted(glob.glob(f'{folderIn}/chb{PATIENT}*.seizures'))
        AllFiles=sorted(glob.glob(f'{folderIn}/chb{PATIENT}*.edf'))

        # CREATE LIST OF FILES WITH SEIZURE AND WITH NON-SEIZURES
        # create lists with just names, to be able to compare them
        SeizFileNames = list()
        for fIndx, f in enumerate(SeizFiles):
            justName = os.path.split(f)[1][:-13]
            if (fIndx == 0):
                SeizFileNames = [justName]
            else:
                SeizFileNames.append(justName)
        NonSeizFileNames = list()
        NonSeizFileFullNames = list()
        for fIndx, f in enumerate(AllFiles):
            justName = os.path.split(f)[1][:-4]
            if (justName not in SeizFileNames):
                if (fIndx == 0):
                    NonSeizFileNames = [justName]
                    NonSeizFileFullNames = [f]
                else:
                    NonSeizFileNames.append(justName)
                    NonSeizFileFullNames.append(f)

        # #EXPORT  FILES
        # FileOutIndx=0
        # for fileIndx, fileName in enumerate(AllFiles):
        #     justName = os.path.split(fileName)[1][:-4]
        #     # pom, fileName1 = os.path.split(fileName)
        #     # fileName2 = os.path.splitext(fileName1)[0]
        #     if (justName not in SeizFileNames): #nonseizure file
        #         # fileNameOut = folderOut + '/Subj'+ pat + '_f'+ str(FileOutIndx).zfill(3)
        #         FileOutIndx=exportNonSeizFile(fileName, folderOut, pat, FileOutIndx, DatasetPreprocessParams)
        #     else: # seizure file
        #         # fileNameOut = folderOut + '/Subj'+ pat + '_f'+ str(FileOutIndx).zfill(3) + '_s'
        #         FileOutIndx= exportSeizFile(fileName, folderOut, pat, FileOutIndx, DatasetPreprocessParams)

        if GeneralParams.parallelize:
            pool.apply_async(extractDataLabels_CHB, args=(pat,  SeizFileNames, AllFiles, DatasetPreprocessParams.channelNamesToKeep, folderOut), callback=collect_result)
            number_free_cores = number_free_cores - 1
            if number_free_cores == -1:
                while number_free_cores == -1:  # synced in the callback
                    time.sleep(0.1)
                    pass
        else:
            extractDataLabels_CHB(pat,  SeizFileNames, AllFiles, DatasetPreprocessParams.channelNamesToKeep, folderOut)

    while number_free_cores < n_cores:  # wait till all subjects have their data processed
        time.sleep(0.1)
        pass

    if GeneralParams.parallelize:
        pool.close()
        pool.join()


# #callback for the apply_async process paralization
# def collect_result(result):
#     global number_free_cores
#     global n_cores_semaphore
#
#     while n_cores_semaphore==0: #block callback in case of multiple accesses
#         pass
#
#     if n_cores_semaphore:
#         n_cores_semaphore=0
#         number_free_cores = number_free_cores+1
#         n_cores_semaphore=1
#
def extractDataLabels_CHB(pat, SeizFileNames, AllFiles, channelNamesToKeep, folderOut):
    # EXPORT  FILES
    FileOutIndx = 0
    for fileIndx, fileName in enumerate(AllFiles):
        justName = os.path.split(fileName)[1][:-4]
        print(justName)
        # pom, fileName1 = os.path.split(fileName)
        # fileName2 = os.path.splitext(fileName1)[0]
        if (justName not in SeizFileNames):  # nonseizure file
            # fileNameOut = folderOut + '/Subj'+ pat + '_f'+ str(FileOutIndx).zfill(3)
            FileOutIndx = exportNonSeizFile(fileName, folderOut, pat, FileOutIndx, channelNamesToKeep)
        else:  # seizure file
            # fileNameOut = folderOut + '/Subj'+ pat + '_f'+ str(FileOutIndx).zfill(3) + '_s'
            FileOutIndx = exportSeizFile(fileName, folderOut, pat, FileOutIndx, channelNamesToKeep)

    return (pat)


def concatenateDataFromFiles(fileNames):
    '''load data from different files and concatenate'''
    dataAll = []
    for f, fileName in enumerate(fileNames):
        reader = csv.reader(open(fileName, "r"))
        data0 = list(reader)
        data = np.array(data0).astype("float")
        # separating to data and labels
        X = data[:, 0:-1]
        y = data[:, -1]

        if (dataAll == []):
            dataAll = X
            labelsAll = y.astype(int)
        else:
            dataAll = np.vstack((dataAll, X))
            labelsAll = np.hstack((labelsAll, y.astype(int)))

    return (dataAll, labelsAll)


def normalizeAndDiscretizeTrainAndTestData_withPercentile(data_train, data_test, numSegLevels, featNorm, percentile=0.9):
    ''' normalize train and test data using normalization values from train set
    also discretize values to specific number of levels  given by numSegLevels'''
    percentileThresh=torch.quantile(data_train, percentile, dim=0)

    # normalize and discretize train adn test data
    data_train_Norm = (data_train - data_train.min(dim=0)[0]) / (percentileThresh - data_train.min(dim=0)[0])
    data_test_Norm = (data_test - data_train.min(dim=0)[0]) / (percentileThresh - data_train.min(dim=0)[0])

    #remove bigger then 1 - put to 1
    data_train_Norm[data_train_Norm>1]=1.0
    data_test_Norm[data_test_Norm > 1] = 1.0

    #remove nans
    data_train_Norm.nan_to_num_(nan=torch.nan, posinf=torch.nan, neginf=torch.nan)
    col_mean = data_train_Norm.nanmean(dim=0)
    col_mean.nan_to_num_(nan=0)
    data_train_Norm=torch.where(data_train_Norm.isnan(), col_mean, data_train_Norm)

    data_test_Norm.nan_to_num_(nan=torch.nan, posinf=torch.nan, neginf=torch.nan)
    col_mean = data_test_Norm.nanmean(dim=0)
    col_mean.nan_to_num_(nan=0)
    data_test_Norm=torch.where(data_test_Norm.isnan(), col_mean, data_test_Norm)

    if featNorm == "Norm":
        return (data_train_Norm, data_test_Norm)
    elif featNorm == "Norm&Discr":
        # discretize
        data_train_Discr = torch.floor((numSegLevels - 1) * data_train_Norm)
        data_test_Discr = torch.floor((numSegLevels - 1) * data_test_Norm)

        # check for outliers
        data_test_Discr[data_test_Discr >= numSegLevels] = numSegLevels - 1
        data_test_Discr[data_test_Discr < 0] = 0
        data_test_Discr[torch.isnan(data_test_Discr)] = 0
        data_train_Discr[data_train_Discr >= numSegLevels] = numSegLevels - 1
        data_train_Discr[data_train_Discr < 0] = 0
        data_train_Discr[torch.isnan(data_train_Discr)] = 0

        # discr values to int
        data_train_Discr = data_train_Discr.to(int)
        data_test_Discr = data_test_Discr.to(int)

        return (data_train_Discr, data_test_Discr)
    else:
        raise TypeError(f'Unrecognized featNorm value:{featNorm}')

def normalizeAndDiscretizeTrainAndTestData(data_train, data_test, numSegLevels, featNorm):
    ''' normalize train and test data using normalization values from train set
    also discretize values to specific number of levels  given by numSegLevels'''

    # normalize and discretize train adn test data
    data_train_Norm = (data_train - data_train.min(dim=0)[0]) / (data_train.max(dim=0)[0] - data_train.min(dim=0)[0])
    data_train_Discr = torch.floor((numSegLevels - 1) * data_train_Norm)
    data_test_Norm = (data_test - data_train.min(dim=0)[0]) / (data_train.max(dim=0)[0] - data_train.min(dim=0)[0])
    data_test_Discr = torch.floor((numSegLevels - 1) * data_test_Norm)
    # check for outliers
    data_test_Discr[data_test_Discr >= numSegLevels] = numSegLevels - 1
    data_test_Discr[data_test_Discr < 0] = 0
    data_test_Discr[torch.isnan(data_test_Discr)] = 0
    data_train_Discr[data_train_Discr >= numSegLevels] = numSegLevels - 1
    data_train_Discr[data_train_Discr < 0] = 0
    data_train_Discr[torch.isnan(data_train_Discr)] = 0
    # discr values to int
    data_train_Discr = data_train_Discr.to(int)
    data_test_Discr = data_test_Discr.to(int)
    if featNorm == "Norm":
        return (data_train_Norm, data_test_Norm)
    elif featNorm == "Norm&Discr":
        return (data_train_Discr, data_test_Discr)
    else:
        raise TypeError(f'Unrecognized featNorm value:{featNorm}')
#
# def calculateAllPairwiseDistancesOfVectors_returnMatrix(VecMatrix1,VecMatrix2, vecType ):
#     ''' calculate pairwise distance between two matrixes columns '''
#     (a, b) = VecMatrix1.shape
#     # rearange to row be D and columns subclases
#     if (a < b):
#         VecMatrix1 = VecMatrix1.transpose()
#         VecMatrix2 = VecMatrix2.transpose()
#     (D, numClasses1) = VecMatrix1.shape
#     (D, numClasses2) = VecMatrix2.shape
#     distances = []
#     distMat=np.zeros((numClasses1, numClasses2))
#     for i in range(numClasses1):
#         for j in range( numClasses2):
#             # hamming distance
#             # vec_c = np.abs(VecMatrix1[:, i] - VecMatrix2[:, j])
#             # distances.append(np.sum(vec_c) / float(D))
#             # distMat [i,j]= np.sum(vec_c) / float(D)
#             dist=ham_dist_arr( VecMatrix1[:, i],VecMatrix2[:, j], D, vecType)
#             distances.append(dist)
#             distMat [i,j]= dist
#     return (distMat, distances)
#
# def ham_dist_arr( vec_a, vec_b, D, vecType='bin'):
#     ''' calculate relative hamming distance fur for np array'''
#     if (vecType=='bin'):
#         vec_c= np.abs(vec_a-vec_b)
#         rel_dist = np.sum(vec_c) / float(D)
#     elif (vecType=='bipol'):
#         vec_c= vec_a+vec_b
#         rel_dist = np.sum(vec_c==0) / float(D)
#     return rel_dist
#
#
# def func_plotRawSignalAndPredictionsOfDiffApproaches_thisFile(justName, predictions_test, predictions_train,  approachNames, approachIndx, folderInRawData, folderOut, SigInfoParams, GeneralParams, SegSymbParams):
#     seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
#     seizureStablePercToTest = GeneralParams.seizureStablePercToTest
#     distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
#
#
#     ## LOAD RAW DATA
#     reader = csv.reader(open(folderInRawData +'/' + justName+'.csv', "r"))
#     data = np.array(list(reader)).astype("float")
#     numCh=np.size(data,1)
#
#     # PLOTTING
#     fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
#     gs = GridSpec(2, 1, figure=fig1)
#     fig1.subplots_adjust(wspace=0.4, hspace=0.6)
#     fig1.suptitle(justName)
#
#     #plotting raw data
#     timeRaw=np.arange(0,len(data[:,0]))/256
#     ax1 = fig1.add_subplot(gs[0,0])
#     # plot all ch raw data
#     for ch in range(numCh-1):
#         sig=data[:,ch]
#         sigNorm=(sig-np.min(sig))/(np.max(sig)-np.min(sig))
#         ax1.plot(timeRaw,sigNorm+ch, 'k')
#     # plot true label
#     ax1.plot(timeRaw, data[:,numCh-1] *numCh, 'r')
#     ax1.set_ylabel('Channels')
#     ax1.set_xlabel('Time')
#     ax1.set_title('Raw data')
#     ax1.grid()
#     yTrueRaw=data[:,numCh-1]
#
#     #plotting predictions
#     yTrue=predictions_test[:,0]
#     # approachNames=['2C', '2Citter','MC', 'MCred', 'MCredItter']
#     # approachIndx=[1,2,4,6,8]
#     ax2 = fig1.add_subplot(gs[1,0])
#     for appIndx, app in enumerate(approachIndx):
#         yPred_NoSmooth=predictions_test[:,app]
#         (yPred_OurSmoothing_step2, yPred_OurSmoothing_step1) = smoothenLabels(yPred_NoSmooth, seizureStableLenToTestIndx, seizureStablePercToTest,distanceBetweenSeizuresIndx)
#         yPred_OurSmoothing_step1 = yPred_OurSmoothing_step1 * 0.4 + appIndx
#         yPred_OurSmoothing_step2 = yPred_OurSmoothing_step2 * 0.3 + appIndx
#         yPred_NoSmooth =yPred_NoSmooth * 0.5 + appIndx
#         time = np.arange(0, len(yTrue)) * 0.5
#         ax2.plot(time, yPred_NoSmooth, 'k', label='NoSmooth')
#         ax2.plot(time, yPred_OurSmoothing_step1, 'b', label='OurSmoothing_step1')
#         ax2.plot(time, yPred_OurSmoothing_step2, 'm', label='OurSmoothing_step2')
#         if (appIndx == 0):
#             ax2.legend()
#     #segmentedLabels = segmentLabels(yTrueRaw, SegSymbParams, SigInfoParams)
#     time = np.arange(0, len(yTrue)) * 0.5
#     ax2.plot(time, yTrue * len(approachNames), 'r')  # label='Performance')
#     ax2.set_yticks(np.arange(0,len(approachNames),1))
#     ax2.set_yticklabels(approachNames, fontsize=12 * 0.8)
#     ax2.set_xlabel('Time')
#     ax2.set_ylabel('Different models')
#     ax2.set_title('Predictions')
#     ax2.grid()
#     if (GeneralParams.plottingON == 1):
#         fig1.show()
#     fig1.savefig(folderOut + '/'+justName+'_RawDataPlot_Test.png')
#     plt.close(fig1)
#
#
#     #PLOTTING JUST PREDICTIONS AND LABELS FOR TRAIN, WITHOUR RAW DATA
#     # PLOTTING
#     fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
#     gs = GridSpec(1, 1, figure=fig1)
#     fig1.subplots_adjust(wspace=0.4, hspace=0.6)
#     fig1.suptitle(justName)
#     #plotting predictions
#     yTrue=predictions_train[:,0]
#     # approachNames=['2C', '2Citter','MC', 'MCred', 'MCredItter']
#     # approachIndx=[1,2,4,6,8]
#     ax2 = fig1.add_subplot(gs[0,0])
#     for appIndx, app in enumerate(approachIndx):
#         yPred_NoSmooth=predictions_train[:,app]
#         (yPred_OurSmoothing_step2, yPred_OurSmoothing_step1) = smoothenLabels(yPred_NoSmooth, seizureStableLenToTestIndx, seizureStablePercToTest,distanceBetweenSeizuresIndx)
#         yPred_OurSmoothing_step1 = yPred_OurSmoothing_step1 * 0.4 + appIndx
#         yPred_OurSmoothing_step2 = yPred_OurSmoothing_step2 * 0.3 + appIndx
#         yPred_NoSmooth =yPred_NoSmooth * 0.5 + appIndx
#         time = np.arange(0, len(yTrue)) * 0.5
#         ax2.plot(time, yPred_NoSmooth, 'k', label='NoSmooth')
#         ax2.plot(time, yPred_OurSmoothing_step1, 'b', label='OurSmoothing_step1')
#         ax2.plot(time, yPred_OurSmoothing_step2, 'm', label='OurSmoothing_step2')
#         if (appIndx == 0):
#             ax2.legend()
#     #segmentedLabels = segmentLabels(yTrueRaw, SegSymbParams, SigInfoParams)
#     time = np.arange(0, len(yTrue)) * 0.5
#     ax2.plot(time, yTrue * len(approachNames), 'r')  # label='Performance')
#     ax2.set_yticks(np.arange(0,len(approachNames),1))
#     ax2.set_yticklabels(approachNames, fontsize=12 * 0.8)
#     ax2.set_xlabel('Time')
#     ax2.set_ylabel('Different models')
#     ax2.set_title('Predictions')
#     ax2.grid()
#     if (GeneralParams.plottingON == 1):
#         fig1.show()
#     fig1.savefig(folderOut + '/'+justName+'_RawDataPlot_Train.png')
#     plt.close(fig1)
#
#
def train_StandardML_moreModelsPossible(X_train, y_train,  StandardMLParams):
    ''' functions that has many possible standard ML approaches that can be used to train
     exact model and its parameters are defined in StandardMLParams
     output is trained model'''
    #
    # if (np.size(X_train,0)==0):
    #         print('X train size 0 is 0!!', X_train.shape, y_train.shape)
    # if (np.size(X_train,1)==0):
    #         print('X train size 1 is 0!!', X_train.shape, y_train.shape)
    # col_mean = np.nanmean(X_train, axis=0)
    # inds = np.where(np.isnan(X_train))
    # X_train[inds] = np.take(col_mean, inds[1])
    # # if still somewhere nan replace with 0
    # X_train[np.where(np.isnan(X_train))] = 0
    # X_train=X_train

    #MLmodels.modelType = 'KNN'
    if (StandardMLParams.modelType=='KNN'):
        model = KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric)
        model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='SVM'):
        model = svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma)
        model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='DecisionTree'):
        if (StandardMLParams.decisionTreeMaxDepth==0):
            model = DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.decisionTreeCriterion, splitter=StandardMLParams.decisionTreeSplitter)
        else:
            model = DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.decisionTreeCriterion, splitter=StandardMLParams.decisionTreeSplitter,  max_depth=StandardMLParams.decisionTreeMaxDepth)
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='RandomForest'):
        if (StandardMLParams.decisionTreeMaxDepth == 0):
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.randomForestNumEstimators, criterion='gini',n_jobs=10 ) #, min_samples_leaf=10
        else:
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.randomForestNumEstimators, criterion=StandardMLParams.DecisionTree_criterion,  max_depth=StandardMLParams.decisionTreeMaxDepth) #, min_samples_leaf=10
        model.fit(X_train.cpu(), y_train.cpu())
    elif (StandardMLParams.modelType=='BaggingClassifier'):
        if (StandardMLParams.Bagging_base_estimator=='SVM'):
            model = BaggingClassifier(base_estimator=svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif  (StandardMLParams.Bagging_base_estimator=='KNN'):
            model = BaggingClassifier(base_estimator= KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif (StandardMLParams.Bagging_base_estimator == 'DecisionTree'):
            model = BaggingClassifier(DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.decisionTreeSplitter),
                n_estimators=StandardMLParams.Bagging_n_estimators, random_state=0)
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='AdaBoost'):
        if (StandardMLParams.Bagging_base_estimator=='SVM'):
            model = AdaBoostClassifier(base_estimator=svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif  (StandardMLParams.Bagging_base_estimator=='KNN'):
            model = AdaBoostClassifier(base_estimator= KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif (StandardMLParams.Bagging_base_estimator == 'DecisionTree'):
            model = AdaBoostClassifier(DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.decisionTreeSplitter),
                n_estimators=StandardMLParams.Bagging_n_estimators, random_state=0)
        model = model.fit(X_train, y_train)

    return (model)
#
#
# def createDataFramePerformance_onlyTest(data, indx, typeApproach,  numSubj):
#     dataAppend = np.vstack( (data[:, indx], np.repeat(typeApproach, numSubj))).transpose()
#     dataFrame=pd.DataFrame(dataAppend, columns=['Performance', 'Type'])
#     return (dataFrame)
#
def saveThreadFunc( data,  outputName, fType):
    ''' saves data to .csv or .gzip file, depending on type parameter'''
    if ('.csv' not in outputName):
        outputName= outputName+'.csv'
    df = pd.DataFrame(data=data if not torch.is_tensor(data) else data.cpu().numpy())
    if (fType=='gzip'):
        df.to_csv(outputName + '.gz', index=False, compression='gzip')
    else:
        df.to_csv(outputName, index=False)

threads = []
def saveDataToFile(data,  outputName, fType, join=False):
    global threads
    threads.append(threading.Thread(target=saveThreadFunc,args=(data,outputName,fType)))
    threads[-1].start()
    if(join):
        for thread in threads:
            thread.join()


def readDataFromFile(inputName):
    ''' read data from first npz, then csv.gz, then csv'''
    # Strip suffixes
    inputName = pathlib.Path(inputName.replace("".join(pathlib.Path(inputName).suffixes),""))
    # Check fist for npz, then csv.gz, then csv
    if os.path.exists(inputName.with_suffix('.npz')):
        return np.load(inputName.with_suffix('.npz'))['d']
    elif os.path.exists(inputName.with_suffix('.csv.gz')):
        return pd.read_csv(inputName.with_suffix('.csv.gz'), compression='gzip').to_numpy()
    elif os.path.exists(inputName.with_suffix('.csv')):
        return pd.read_csv(inputName.with_suffix('.csv')).to_numpy()
        data = torch.from_numpy(data)


def extractFeatAndLabelsFromFile(fileIn,folderOutFeatures,fileName2,borderFreqHigh, samplFreq, winLen, winStep, ZCThreshArr,featNames, pat, sigRanges):
    print('Processing file: ' + fileIn)

    if (borderFreqHigh==20):
        sos2=np.array([[1.71814708e-03,  3.43629417e-03,  1.71814708e-03 , 1.00000000e+00, -1.29030792e+00,  4.33611296e-01],[1.00000000e+00,  2.00000000e+00 , 1.00000000e+00 , 1.00000000e+00, -1.52048756e+00 , 7.18819902e-01],
                       [1.00000000e+00, - 2.00000000e+00 , 1.00000000e+00,  1.00000000e+00, -1.95186008e+00 , 9.52561419e-01],[1.00000000e+00 ,- 2.00000000e+00 , 1.00000000e+00  ,1.00000000e+00, -1.98210850e+00 , 9.82720454e-01]])
    elif (borderFreqHigh == 30):
        sos2= np.array([[ 0.00735282 , 0.01470564  ,0.00735282  ,1.     ,    -0.96372427,  0.25932529], [ 1.   ,       2.     ,     1.    ,      1.  ,       -1.19967297 , 0.61089997],
                    [ 1.      ,   -2.    ,      1.   ,       1.      ,   -1.95300151 , 0.95365993], [ 1.    ,     -2.      ,    1.     ,     1.      ,   -1.98163749  ,0.98224463]])
    else:
        print("ERROR: Butterworth filter not defined!")

    # TODO: STH IS NOT WORKING WIHT signa.butter if parameter='sos' but before it worked
    # ##butterworth filter initialization
    # sos2 = signal.butter(4, [DatasetPreprocessParams.bordFreqLow, DatasetPreprocessParams.borderFreqHigh], 'bandpass', fs=DatasetPreprocessParams.samplFreq, output='sos')

    # reading data
    data = readDataFromFile(fileIn)
    X = data[:, 0:-1]
    y = data[:, -1]
    (lenData, numCh) = X.shape
    labels = y[0:lenData - 2]
    index = np.arange(0, lenData - int(samplFreq * winLen), int(samplFreq * winStep))
    labelsSegm = calculateMovingAvrgMeanWithUndersampling(labels, int( samplFreq * winLen), int(samplFreq * winStep))
    labelsSegm = (labelsSegm > 0.5) * 1

    actualThrValues = np.zeros((numCh, len(ZCThreshArr)))
    for fIndx, fName in enumerate(featNames):
        print(fName)
        for ch in range(numCh):
            sig = X[:, ch]
            sigFilt = signal.sosfiltfilt(sos2, sig)  # filtering

            if (fName == 'ZeroCross'):
                (featVals, actualThrValues[ch, :]) = calulateZCfeaturesRelative_oneCh(np.copy(sigFilt), samplFreq, winLen, winStep, ZCThreshArr,sigRanges[int(pat) - 1, ch])
            else:
                featVals = calculateChosenMLfeatures_oneCh(np.copy(sigFilt), samplFreq, winLen, winStep, fName)

            if (ch == 0):
                AllFeatures = featVals
            else:
                AllFeatures = np.hstack((AllFeatures, featVals))

        # save for this file  features and labels
        outputName = folderOutFeatures + '/' + fileName2 + '_' + fName + '.csv'
        saveDataToFile(AllFeatures, outputName, 'gzip')
        if (fName == 'ZeroCross'):
            outputName = folderOutFeatures + '/' + fileName2 + '_ZCthreshValues.csv'
            saveDataToFile(actualThrValues, outputName, 'gzip')

    outputName = folderOutFeatures + '/' + fileName2 + '_Labels.csv'
    saveDataToFile(labelsSegm, outputName, 'gzip')
    return()


def calculateFeaturesPerEachFile_gzip(folderIn, folderOutFeatures, GeneralParams,GeneralCHBMITParams, DatasetPreprocessParams, FeaturesParams, sigRanges):
    '''function that loads one by one file, filters data and calculates different features, saves each of them in individual files so that
    later it can be chosen and combined '''
    # if (DatasetPreprocessParams.borderFreqHigh==20):
    #     sos2=np.array([[1.71814708e-03,  3.43629417e-03,  1.71814708e-03 , 1.00000000e+00, -1.29030792e+00,  4.33611296e-01],[1.00000000e+00,  2.00000000e+00 , 1.00000000e+00 , 1.00000000e+00, -1.52048756e+00 , 7.18819902e-01],
    #                    [1.00000000e+00, - 2.00000000e+00 , 1.00000000e+00,  1.00000000e+00, -1.95186008e+00 , 9.52561419e-01],[1.00000000e+00 ,- 2.00000000e+00 , 1.00000000e+00  ,1.00000000e+00, -1.98210850e+00 , 9.82720454e-01]])
    # elif (DatasetPreprocessParams.borderFreqHigh == 30):
    #     sos2= np.array([[ 0.00735282 , 0.01470564  ,0.00735282  ,1.     ,    -0.96372427,  0.25932529], [ 1.   ,       2.     ,     1.    ,      1.  ,       -1.19967297 , 0.61089997],
    #                 [ 1.      ,   -2.    ,      1.   ,       1.      ,   -1.95300151 , 0.95365993], [ 1.    ,     -2.      ,    1.     ,     1.      ,   -1.98163749  ,0.98224463]])
    # else:
    #     print("ERROR: Butterworth filter not defined!")
    #
    # # TODO: STH IS NOT WORKING WIHT signa.butter if parameter='sos' but before it worked
    # # ##butterworth filter initialization
    # # # sos = signal.butter(4, 20, 'low', fs=ZeroCrossFeatureParams.samplFreq, output='sos')
    # # #sos = signal.butter(4, [1, 30], 'bandpass', fs=ZeroCrossFeatureParams.samplFreq, output='sos')
    # # sos2 = signal.butter(4, [DatasetPreprocessParams.bordFreqLow, DatasetPreprocessParams.borderFreqHigh], 'bandpass', fs=DatasetPreprocessParams.samplFreq, output='sos')

    global files_processed
    global n_files_per_patient
    global number_free_cores

    print('Extracting features from all subjects files')

    # numFeat = len(ZeroCrossFeatureParams.EPS_thresh_arr) + 1

    if GeneralParams.parallelize:
        n_cores = mp.cpu_count()
        n_cores = ceil(n_cores * GeneralParams.perc_cores)
        print('Number of cores: ' + str(n_cores))

        pool = mp.Pool(n_cores)
        number_free_cores = n_cores

    files_processed = np.zeros(len(GeneralCHBMITParams.patients))
    n_files_per_patient = np.zeros(len(GeneralCHBMITParams.patients))



    # go through all patients
    for patIndx, pat in enumerate(GeneralCHBMITParams.patients):
        filesIn=np.sort(glob.glob(folderIn + '/*Subj' + pat + '*.csv.gz'))
        if (len(filesIn)==0):
            filesIn = np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv.gz'))
        numFiles=len(filesIn)
        n_files_per_patient[patIndx] = numFiles
        print('-- Patient:', pat, 'NumSeizures:', numFiles)

        for fileIndx, fileIn in enumerate(filesIn):
            pom, fileName1 = os.path.split(fileIn)
            fileName2 = fileName1[0:-7]
            print('File: ' + fileName2 + '  NFilesprocessed: ' + str(files_processed[patIndx]) + '  Out of: ' + str(n_files_per_patient[patIndx]))


            if GeneralParams.parallelize:
                # pool.apply_async(extractFeatAndLabelsFromFile, args=( fileIn, folderOutFeatures, fileName2, pat, DatasetPreprocessParams,
                #     FeaturesParams,  sigRanges),callback=collect_result_features)
                pool.apply_async(extractFeatAndLabelsFromFile, args=( fileIn, folderOutFeatures, fileName2,  DatasetPreprocessParams.borderFreqHigh,
                                                                      DatasetPreprocessParams.samplFreq, FeaturesParams.winLen, FeaturesParams.winStep,
                                                                      FeaturesParams.ZCThreshArr, FeaturesParams.featNames, pat, sigRanges),callback=collect_result_features)

                number_free_cores = number_free_cores - 1
                if number_free_cores == 0:
                    while number_free_cores == 0:  # synced in the callback
                        time.sleep(0.1)
                        pass
            else:
                # extractFeatAndLabelsFromFile(fileIn, folderOutFeatures, fileName2, pat, DatasetPreprocessParams,FeaturesParams,  sigRanges)
                extractFeatAndLabelsFromFile( fileIn, folderOutFeatures, fileName2,  DatasetPreprocessParams.borderFreqHigh,
                                                                      DatasetPreprocessParams.samplFreq, FeaturesParams.winLen, FeaturesParams.winStep,
                                                                      FeaturesParams.ZCThreshArr, FeaturesParams.featNames, pat, sigRanges)

    while number_free_cores < n_cores:  # wait till all subjects have their data processed
        time.sleep(0.1)
        pass

    if GeneralParams.parallelize:
        pool.close()
        pool.join()

            # # reading data
            # data=readDataFromFile(fileIn)
            # X = data[:, 0:-1]
            # y = data[:, -1]
            # (lenData, numCh) = X.shape
            # labels = y[0:lenData - 2]
            # index = np.arange(0, lenData - int(DatasetPreprocessParams.samplFreq * FeaturesParams.winLen), int(DatasetPreprocessParams.samplFreq * FeaturesParams.winStep))
            # labelsSegm = calculateMovingAvrgMeanWithUndersampling(labels, int( DatasetPreprocessParams.samplFreq * FeaturesParams.winLen), int( DatasetPreprocessParams.samplFreq * FeaturesParams.winStep))
            # labelsSegm = (labelsSegm > 0.5) * 1
            #
            #
            # actualThrValues = np.zeros((numCh, len(ZeroCrossFeatureParams.EPS_thresh_arr)))
            # for fIndx, fName in enumerate(FeaturesParams.featNames):
            #     print(fName)
            #     for ch in range(numCh):
            #         sig = X[:, ch]
            #         sigFilt = signal.sosfiltfilt(sos2, sig) #filtering
            #
            #         if (fName == 'ZeroCross'):
            #             (featVals, actualThrValues[ch,:])=calulateZCfeaturesRelative_oneCh(np.copy(sigFilt), DatasetPreprocessParams, FeaturesParams,sigRanges[int(pat) - 1, ch])
            #         else:
            #             featVals= calculateChosenMLfeatures_oneCh(np.copy(sigFilt), DatasetPreprocessParams, FeaturesParams, fName)
            #
            #         if (ch == 0):
            #             AllFeatures = featVals
            #         else:
            #             AllFeatures = np.hstack((AllFeatures, featVals))
            #
            #     # save for this file  features and labels
            #     outputName = folderOutFeatures + '/' + fileName2 + '_'+fName+'.csv'
            #     saveDataToFile(AllFeatures, outputName, 'gzip')
            #     if (fName == 'ZeroCross'):
            #         outputName = folderOutFeatures + '/' + fileName2 + '_ZCthreshValues.csv'
            #         saveDataToFile(actualThrValues, outputName, 'gzip')
            #
            # outputName = folderOutFeatures + '/' + fileName2 + '_Labels.csv'
            # saveDataToFile(labelsSegm, outputName, 'gzip')

# #callback for the apply_async process paralization
# def collect_result_features(result):
#     global files_processed
#     global number_free_cores
#     global n_cores_semaphore
#     global n_files_per_patient
#
#     while n_cores_semaphore==0: #block callback in case of multiple accesses
#         pass
#
#     if n_cores_semaphore:
#         n_cores_semaphore=0
#         files_processed[result] += 1
#         number_free_cores = number_free_cores+1
#         n_cores_semaphore=1
#
def calculateStatisticsOnSignalValues(folderIn, folderOut, GeneralParams):
    ''' function that analyses 5 to 95 percentile range of raw data
    but only form first 5h or data (otherwise it would not be fair)
    this values are needed if we use relative Zero-cross feature thresholds'''
    #save only file with first 5h of data
    for patIndx, pat in enumerate(GeneralParams.patients):
        allFiles = np.sort(glob.glob(folderIn + '/Subj' + pat + '*.csv.gz'))
        firstFileCreated = 0
        numFilesThisSubj = 0
        for fIndx, fileName in enumerate(allFiles):
            data0 = readDataFromFile(fileName)
            data = data0[:, 0:-1]
            label = data0[:, -1]
            pom, fileName1 = os.path.split(fileName)
            fileNameOut = os.path.splitext(fileName1)[0][0:5]
            if (firstFileCreated == 0):  # first file, append until at least one seizure
                if (fIndx == 0):
                    dataOut = data
                    labelOut = label
                else:
                    dataOut = np.vstack((dataOut, data))
                    labelOut = np.hstack((labelOut, label))
                if (fIndx >= 4): #at least 5h
                    firstFileCreated = 1
                    fileNameOut2 = folderOut + '/Subj' + pat +'_OnlyFirst5h.csv'
                    saveDataToFile(np.hstack((dataOut, labelOut.reshape((-1, 1)))), fileNameOut2, 'gzip')

    # calculate percentiles
    percentileRangesPerSubj = np.zeros((len(GeneralParams.patients), 2))
    for patIndx, pat in enumerate(GeneralParams.patients):
        fileIn = folderOut + '/Subj' + pat + '_OnlyFirst5h.csv'
        data = readDataFromFile(fileIn)
        if (patIndx == 0):
            percentileRangesPerSubj = np.zeros((len(GeneralParams.patients), len(data[0, :]) - 1))
        for ch in range(len(data[0, :]) - 1):
            percentileRangesPerSubj[patIndx, ch] = np.percentile(data[:, ch], 95) - np.percentile(data[:, ch], 5)
    outputName = folderOut + '/AllSubj_DataRange5to95Percentile.csv'
    saveDataToFile(percentileRangesPerSubj, outputName, 'gzip')

def mergeFeatFromTwoMatrixes(mat1, mat2, numCh):
    ''' merges two matrixes that contain features for all ch
    rearanges them so that in columns are fis all features of ch1, then all feat of ch2 etc'''
    numFeat1=int(len(mat1[0, :])/numCh)
    numFeat2=int(len(mat2[0, :])/numCh)
    numColPerCh=numFeat1+numFeat2
    matFinal=np.zeros((len(mat1[:,0]), numColPerCh*numCh))
    for ch in range(numCh):
        matFinal[:, ch*numColPerCh : ch*numColPerCh + numFeat1]=mat1[ :, ch*numFeat1: (ch+1)*numFeat1]
        matFinal[:,  ch * numColPerCh + numFeat1  : ch * numColPerCh + numColPerCh ] = mat2[:, ch * numFeat2: (ch + 1) * numFeat2]
    return matFinal

def loadAndConcatenateAllFeatFilesForThisFile(fileName, featNames, numCh):
    ''' concatenates files containing different features calculated in a way that output file contains
    first all features for ch1, chen all features for ch2 etc '''
    for featIndx, featName in enumerate(featNames):
        fileIn=fileName+ featName+ '.csv.gz'
        if (featIndx==0):
            mat1 = readDataFromFile(fileIn)
        else:
            mat2= readDataFromFile(fileIn)
            mat1= mergeFeatFromTwoMatrixes(mat1, mat2, numCh)
    return (mat1)

# def concatenateAllFeatures_moreNonseizureForFactor_gzip(folderIn, folderOut,GeneralParams,  FeaturesParams,   factor):
#     ''' loads original data (features calculated) and creates subselection of data that contains all seizures and "factor" times mroe non seizure data
#     it uses every seizure that exists and then randomly selects nonseizure data from different file and appends before and after seizure so that seizure is centered'''
#
#     createFolderIfNotExists(folderOut)
#     paieredFiles=[]
#     for patIndx, pat in enumerate(GeneralParams.patients):
#         seizFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_s_'+FeaturesParams.featNames[0]+'.csv.gz'))
#         allFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+FeaturesParams.featNames[0]+'.csv.gz'))
#         numLettersToRemove=-len(FeaturesParams.featNames[0])-7
#         nonSeizFiles=list(set(list(allFiles)) - set(list(seizFiles)))
#         print('-- Patient:', pat, 'NumSeizures:', len(seizFiles))
#         outputFileIndx=0
#         IndxNonSeizFile=0
#         #LOAD SEIZURE FILES
#         for fIndx, fileName in enumerate(seizFiles):
#             pom, fileName1 = os.path.split(fileName)
#             fileNameS = os.path.splitext(fileName1)[0][0:numLettersToRemove]
#
#             numCh = 18
#             data=loadAndConcatenateAllFeatFilesForThisFile(fileName[0:numLettersToRemove], FeaturesParams.featNames, numCh)
#             numFeat = int(len(data[0, :]))
#
#             fileName2=fileName[0:numLettersToRemove]+'Labels.csv'
#             labels = readDataFromFile(fileName2)
#
#             #find starts and stops of seizures
#             diffSig=np.diff(np.squeeze(labels))
#             szStart=np.where(diffSig==1)[0]
#             szStop= np.where(diffSig == -1)[0]
#
#             # for each seizure cut it out and save
#             numSeizures = len(szStart)
#             for i in range(numSeizures):
#                 #prepare where to save new cutout
#                 try:
#                     seizureLen = int(szStop[i]- szStart[i])
#                 except:
#                     seizureLen = int(len(diffSig)- szStart[i])
#                     print('prob with indx')
#                 newLabel = np.zeros(seizureLen * (factor + 1))  # both for seizure nad nonSeizure lavels
#                 newData = np.zeros((seizureLen * (factor + 1), numFeat))
#                 #save seizure part
#                 nonSeizLen = int(factor * seizureLen)
#                 newData[int(nonSeizLen / 2):int(nonSeizLen / 2) + seizureLen] = data[(szStart[i]): (szStart[i] + seizureLen), :]
#                 newLabel[int(nonSeizLen / 2):int(nonSeizLen / 2) + seizureLen] = np.ones(seizureLen)
#
#                 #LOAD NON SEIZRUE DATA
#                 for fns in range(IndxNonSeizFile,len(nonSeizFiles)):
#                     pom, fileName1 = os.path.split(nonSeizFiles[fns])
#                     fileNameNS = fileName1[0:numLettersToRemove-1]
#
#                     numCh = 18
#                     dataNS = loadAndConcatenateAllFeatFilesForThisFile(nonSeizFiles[fns][0:numLettersToRemove], FeaturesParams.featNames, numCh)
#                     numFeat = int(len(dataNS[0, :]))
#
#                     lenSigNonSeiz= len(dataNS[:,0])
#                     if (lenSigNonSeiz > nonSeizLen):
#                         # cut nonseizure part
#                         nonSeizStart = np.random.randint(lenSigNonSeiz - nonSeizLen - 1)
#                         nonSeizCutout = dataNS[nonSeizStart: nonSeizStart + nonSeizLen, :]
#                         newData[0:int(nonSeizLen / 2)] = nonSeizCutout[0:int(nonSeizLen / 2)]
#                         newData[int(nonSeizLen / 2) + seizureLen:] = nonSeizCutout[int(nonSeizLen / 2):]
#
#                         # SAVING TO CSV FILE
#                         fileNameOut = os.path.splitext(fileName1)[0][0:6]
#                         fileName3 = folderOut + '/' + fileNameOut  + '_f' + str(outputFileIndx).zfill(3) # 's' marks it is file with seizure
#                         # writeToCsvFile(newData, newLabel, fileName3)
#                         saveDataToFile(np.hstack((newData, np.reshape(newLabel, (-1,1)))), fileName3, 'gzip')
#
#                         print('PAIRED: ', fileNameS , '- ', fileNameNS)
#                         paieredFiles.append( fileNameOut  + '_cv' + str(outputFileIndx).zfill(3)  +' : ' +fileNameS + ' -- '+ fileNameNS)
#
#                         outputFileIndx=outputFileIndx+1
#                         IndxNonSeizFile = IndxNonSeizFile + 1
#
#
#                         #in cases when there is more seizure files then non seizure ones, we will not save this seizures
#                         # because there is no seizure file to randomly select from
#                         # thus we can start from firs non seizure file again
#                         # or if we want to be absolutely sure there is no overlap of non seizure files we can comment this,
#                         # but we will loose some seizures  (or we need to think of smarter way to do this matching)
#                         if (IndxNonSeizFile==len(nonSeizFiles)):
#                             IndxNonSeizFile=0
#
#                         break
#                     else:
#                         #fns = fns + 1
#                         print('not enough nonSeiz data in this file')
#
#     #save paired files
#     file= open(folderOut + '/PairedFiles.txt', 'w')
#     for i in range(len(paieredFiles)):
#         file.write(paieredFiles[i]+'\n')
#     file.close()
#
#
def concatenateAllFeatures_moreNonseizureForFactorFromTheSameFile_gzip(folderIn, folderOut,GeneralParams,  FeaturesParams,   factor):
    ''' loads original data (features calculated) and creates subselection of data that contains all seizures and "factor" times mroe non seizure data
    it uses every seizure that exists and then randomly selects nonseizure data from different file and appends before and after seizure so that seizure is centered'''

    createFolderIfNotExists(folderOut)
    paieredFiles=[]
    for patIndx, pat in enumerate(GeneralParams.patients):
        seizFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_s_'+FeaturesParams.featNames[0]+'.csv.gz'))
        allFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+FeaturesParams.featNames[0]+'.csv.gz'))
        numLettersToRemove=-len(FeaturesParams.featNames[0])-7
        nonSeizFiles=list(set(list(allFiles)) - set(list(seizFiles)))
        print('-- Patient:', pat, 'NumSeizures:', len(seizFiles))
        outputFileIndx=0
        IndxNonSeizFile=0
        #LOAD SEIZURE FILES
        for fIndx, fileName in enumerate(seizFiles):
            pom, fileName1 = os.path.split(fileName)
            fileNameS = os.path.splitext(fileName1)[0][0:numLettersToRemove]

            numCh = 18
            data=loadAndConcatenateAllFeatFilesForThisFile(fileName[0:numLettersToRemove], FeaturesParams.featNames, numCh)
            numFeat = int(len(data[0, :]))

            fileName2=fileName[0:numLettersToRemove]+'Labels.csv'
            labels = readDataFromFile(fileName2)

            #find starts and stops of seizures
            diffSig=np.diff(np.squeeze(np.append(labels, [0])))
            szStart=np.where(diffSig==1)[0]
            szStop= np.where(diffSig == -1)[0]
            # print('File:'+fileNameS, 'SzStart and stop', szStart, szStop)
            # print(diffSig)
            # for each seizure cut it out and save
            numSeizures = len(szStart)
            for i in range(numSeizures):
                #prepare where to save new cutout
                seizureLen = int(szStop[i]- szStart[i])
                newLabel = np.zeros(seizureLen * (factor + 1))  # both for seizure nad nonSeizure lavels
                newData = np.zeros((seizureLen * (factor + 1), numFeat))
                #save seizure part
                nonSeizLen = int(factor * seizureLen)
                winStart=np.max([0,szStart[i]-int(nonSeizLen/2)])
                winEnd=np.min([len(data), szStop[i]+int(nonSeizLen/2)])
                newData=data[winStart: winEnd,:]
                newLabel=np.zeros((len(newData[:,0])))
                actualSStartInLabels=szStart[i]-winStart
                newLabel[actualSStartInLabels:actualSStartInLabels+seizureLen]=np.ones(seizureLen)
                # newData[int(nonSeizLen / 2):int(nonSeizLen / 2) + seizureLen] = data[(szStart[i]): (szStart[i] + seizureLen), :]
                # newLabel[int(nonSeizLen / 2):int(nonSeizLen / 2) + seizureLen] = np.ones(seizureLen)

                # SAVING TO CSV FILE
                fileNameOut = os.path.splitext(fileName1)[0][0:6]
                fileName3 = folderOut + '/' + fileNameOut + '_f' + str(outputFileIndx).zfill( 3)  # 's' marks it is file with seizure
                # writeToCsvFile(newData, newLabel, fileName3)
                saveDataToFile(np.hstack((newData, np.reshape(newLabel, (-1, 1)))), fileName3, 'gzip')
                outputFileIndx = outputFileIndx + 1



def plotRearangedDataLabelsInTime(folderIn,  GeneralCHBMITParams,PostprocessingParams, FeaturesParams):
    ''' function that plots of all data of one subject in appended way
    this way it is possible to test if data rearanging worked and no data is lost'''
    folderOut=folderIn +'/LabelsInTime'
    createFolderIfNotExists(folderOut)

    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    for patIndx, pat in enumerate(GeneralCHBMITParams.patients):
        print('Subj ', pat)
        inputFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*.csv.gz'))
        numFiles = len(inputFiles)

        #concatinatin predictions so that predictions for one seizure are based on train set with all seizures before it
        for fIndx, fileName in enumerate(inputFiles):
            print('File ', fileName)
            data=readDataFromFile(fileName)
            if fIndx==0:
                labels = np.squeeze(data[:,-1])
                testIndx=np.ones(len(data[:,-1]))*(fIndx+1)
            else:
                labels = np.hstack((labels,  np.squeeze(data[:,-1])))
                testIndx= np.hstack((testIndx, np.ones(len(data[:,-1]))*(fIndx+1)))

        # print('Smoothing')
        # (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(labels, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx)

        print('Plotting')
        #Plot predictions in time
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(2, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.2, hspace=0.2)
        xValues = np.arange(0, len(labels), 1) / (60*60*2)
        ax1 = fig1.add_subplot(gs[0,0])
        ax1.plot(xValues, labels , 'r')
        ax1.set_ylabel('TrueLabel')
        ax1.set_title('Subj'+pat)
        ax1.grid()
        # ax1 = fig1.add_subplot(gs[1, 0])
        # ax1.plot(xValues, yPred_SmoothOurStep1, 'b')
        # ax1.set_ylabel('Step1')
        # ax1.grid()
        # ax1 = fig1.add_subplot(gs[2, 0])
        # ax1.plot(xValues, yPred_SmoothOurStep2, 'm')
        # ax1.set_ylabel('Step2')
        # ax1.grid()
        ax1 = fig1.add_subplot(gs[1, 0])
        ax1.plot(xValues, testIndx , 'k')
        ax1.set_ylabel('FileNr')
        ax1.grid()
        ax1.set_xlabel('Time [h]')
        fig1.show()
        fig1.savefig(folderOut + '/Subj' + pat + '_RawLabels.png', bbox_inches='tight')
        plt.close(fig1)


def normalizeData(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data
#
#
# def plot_RawSeizureData_thisFile(data, labels, samplFreq, fileNameFigName, fileNameTitle,folderOut):
#     path, fileName = os.path.split(fileNameFigName)
#
#     numCh = len(data[0, :])
#     xValues = np.arange(0, len(labels), 1) / (samplFreq)  # in time [s]
#
#     # Plot raw data in time
#     fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
#     gs = GridSpec(1, 1, figure=fig1)
#     fig1.subplots_adjust(wspace=0.2, hspace=0.2)
#
#     ax1 = fig1.add_subplot(gs[0, 0])
#     ax1.plot(xValues, labels * (numCh + 1), 'r')
#     ax1.set_title(fileNameTitle)
#
#     ticNames = []
#     for ch in range(numCh):
#         normData = normalizeData(data[:, ch])
#         ax1.plot(xValues, normData * 0.8 + ch, 'k')
#         ticNames.append('Ch' + str(ch))
#
#     ax1.set_yticks(np.arange(0, numCh, 1))
#     ax1.set_yticklabels(ticNames, fontsize=10)  # , rotation=45)
#     ax1.set_xlabel('Time [s]')
#     ax1.grid()
#     fig1.show()
#     fig1.savefig(folderOut +'/'+fileName + '_RawData.png', bbox_inches='tight')
#     plt.close(fig1)
#
#
# def plot_RawSeizureData(folderIn,  GeneralCHBMITParams,DatasetPreprocessParams, PostprocessingParams, FeaturesParams):
#     ''' function that plots of all data of one subject in appended way
#     this way it is possible to test if data rearanging worked and no data is lost'''
#     folderOut=folderIn +'/RawSeizData'
#     createFolderIfNotExists(folderOut)
#
#     for patIndx, pat in enumerate(GeneralCHBMITParams.patients):
#         print('Subj ', pat)
#         inputFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*.csv.gz'))
#         numFiles = len(inputFiles)
#         seizNum=0
#         #concatinatin predictions so that predictions for one seizure are based on train set with all seizures before it
#         for fIndx, fileName in enumerate(inputFiles):
#             print('File ', fileName)
#             data=readDataFromFile(fileName)
#             rawSig=data[:,:-1]
#             labels = np.squeeze(data[:, -1])
#
#             if (np.sum(labels)!=0):
#                 # #all data from file
#                 # plot_RawSeizureData_thisFile(rawSig, labels,DatasetPreprocessParams.samplFreq,fileName[:-7],'Subj'+pat+'_S'+str(seizNum), folderOut)
#
#                 #cutout aroung seizure
#                 #find start and stop of seiz
#                 diffLab = np.diff(labels)
#                 seizStart=np.where(diffLab==1)[0]
#                 seizStop=np.where(diffLab==-1)[0]
#                 if (len(seizStart)==0):
#                     seizStart=0
#                 if (len(seizStop)==0):
#                     seizStop=len(labels)
#                 for s in range(len(seizStart)):
#                     print('More seizures in file '+ fileName + 'NumSeiz: '+str(len(seizStart)))
#                     cutoutStart=int(seizStart[s] - DatasetPreprocessParams.samplFreq*10 )#10 sec before
#                     cutoutStop=int(seizStop[s]+ DatasetPreprocessParams.samplFreq*10)#10 sec after
#                     if (cutoutStart<0):
#                         cutoutStart=0
#                     if (cutoutStop>len(labels)):
#                         cutoutStop=len(labels)
#                     plot_RawSeizureData_thisFile(rawSig[cutoutStart:cutoutStop,:], labels[cutoutStart:cutoutStop],DatasetPreprocessParams.samplFreq,fileName[:-7]+'_cutout','Subj'+pat+'_S'+str(seizNum)+'_cutout', folderOut)
#
#                     seizNum=seizNum+1
#
#
#
# def removePreAndPostIctalData_StoS(data, labels, DatasetPreprocessParams, FeaturesParams):
#     labels2=np.append(labels, 0)
#     seizStarts=np.where(np.diff(labels2)==1)[0]
#     seizStops = np.where(np.diff(labels2) == -1)[0]
#     seizStops=seizStops[-len(seizStarts):]
#     lenBeforeSeizIndx=int(DatasetPreprocessParams.PreIctalTimeToRemove/FeaturesParams.winStep)
#     lenAfterSeizIndx = int(DatasetPreprocessParams.PostIctalTimeToRemove /FeaturesParams.winStep)
#     (dataNew, labelsNew)= removePreAndPostIctalAreas_StoS(data, labels, seizStarts, seizStops, lenBeforeSeizIndx, lenAfterSeizIndx)
#     return (dataNew, labelsNew)
#
# def removePreAndPostIctalData(data, labels, DatasetPreprocessParams, FeaturesParams):
#     labels2=np.append(labels, 0)
#     seizStarts=np.where(np.diff(labels2)==1)[0]
#     seizStops = np.where(np.diff(labels2) == -1)[0]
#     # if (seizStops[-1]==len(labels2)):
#     #     seizStops[-1]=seizStops[-1]-1
#     lenBeforeSeizIndx=int(DatasetPreprocessParams.PreIctalTimeToRemove/FeaturesParams.winStep)
#     lenAfterSeizIndx = int(DatasetPreprocessParams.PostIctalTimeToRemove /FeaturesParams.winStep)
#     (dataNew, labelsNew)= removePreAndPostIctalAreas(data, labels, seizStarts, seizStops, lenBeforeSeizIndx, lenAfterSeizIndx)
#     return (dataNew, labelsNew)
#
# def removePreAndPostIctalAreas(data, labels, szStart, szStop, lenBeforeSeizIndx, lenAfterSeizIndx):
#     ''' function that remores some amount of pre and post ictal data '''
#     keepIndxs=np.ones(len(labels))
#     for s in range(len(szStart)):
#         #pre seizure part
#         if (s == 0): #if first seizure, so no seizure before
#             keepIndxs[ np.max([0,szStart[s]-lenBeforeSeizIndx ]): szStart[s]]=0
#         elif (szStop[s - 1]  in range(szStart[s] - lenBeforeSeizIndx, szStart[s])): #previous seizure would we cut
#             keepIndxs[szStop[s - 1]: szStart[s]] = 0
#         else:  # seizure in the middle and all ok
#             keepIndxs[szStart[s]-lenBeforeSeizIndx : szStart[s]] = 0
#         #post seizure part
#         if (s == (len(szStart) - 1)): #if last seizure, so no seizure after
#             keepIndxs[szStop[s]: np.min([szStop[s]+lenAfterSeizIndx,len(labels)])] = 0
#         elif (szStart[s + 1]  in range(szStop[s], szStop[s] + lenAfterSeizIndx)): #if next seizure in postictal of this, dont cut
#             keepIndxs[szStop[s]: szStart[s+1]] = 0
#         else: #seizure in the middle and all ok
#             keepIndxs[szStop[s]: szStop[s]+lenAfterSeizIndx] = 0
#
#     pos=np.where(keepIndxs==1)[0]
#     try:
#         dataNew=data[pos, :]
#     except:
#         print('dsadas')
#     labelsNew=labels[pos]
#     return(dataNew, labelsNew)
#
def removePreAndPostIctalAreas_StoS(data, labels, szStart, szStop, lenBeforeSeizIndx, lenAfterSeizIndx):
    ''' function that remores some amount of pre and post ictal data '''
    keepIndxs=np.ones(len(labels))
    #post seizure part - actually beginning of this file
    keepIndxs[0:lenAfterSeizIndx] = 0

    # pre seizure part
    for s in range(len(szStart)):
        if (s == 0): #if first seizure, so no seizure before
            keepIndxs[ np.max([0,szStart[s]-lenBeforeSeizIndx ]): szStart[s]]=0
        elif (szStop[s - 1]  in range(szStart[s] - lenBeforeSeizIndx, szStart[s])): #previous seizure would we cut
            keepIndxs[szStop[s - 1]: szStart[s]] = 0
        else:  # seizure in the middle and all ok
            keepIndxs[szStart[s]-lenBeforeSeizIndx : szStart[s]] = 0

    pos=np.where(keepIndxs==1)[0]
    try:
        dataNew=data[pos, :]
    except:
        print('dsadas')
    labelsNew=labels[pos]
    return(dataNew, labelsNew)

# def concatenateFeatures_allData_gzip(folderIn, folderOut,GeneralParams, GeneralCHBMITParams, FeaturesParams,  maxWinLen, type):
#     ''' loads original files (with calculated features) and rearanges them in a way that each file contains equal amount of data
#      amount of data corronds to maxWinLen and is usually 1h or 4h '''
#
#     global number_free_cores
#     print('Extracting .csv from CHB edf files')
#     if GeneralParams.parallelize:
#         n_cores  = mp.cpu_count()
#         n_cores = ceil(n_cores*GeneralParams.perc_cores)
#
#         if n_cores > len(GeneralCHBMITParams.patients):
#             n_cores = len(GeneralCHBMITParams.patients)
#
#         print('Number of used cores: ' + str(n_cores))
#
#         pool = mp.Pool(n_cores)
#         number_free_cores = n_cores
#
#     lenBeforeSeiz=0 #60 # in sec
#     lenAfterSeiz= 0 #600 # in sec
#     lebBeforeSeizIndx=int(lenBeforeSeiz/FeaturesParams.winStep)
#     lenAfterSeizIndx = int(lenAfterSeiz / FeaturesParams.winStep)
#
#     createFolderIfNotExists(folderOut)
#     maxWinLenIndx=int(maxWinLen/FeaturesParams.winStep)
#     for patIndx, pat in enumerate(GeneralCHBMITParams.patients):
#         if GeneralParams.parallelize:
#             if (type=='StoS'):
#                 pool.apply_async(concatenateFeatureFilesStoSApproach_gzip_perSubj,
#                                  args=(folderIn, folderOut, pat, FeaturesParams.featNames, lebBeforeSeizIndx, lenAfterSeizIndx), callback=collect_result)
#             else:  # 'FixedWin'
#                 pool.apply_async(concatenateFeatures_allDataInEqualWindows_gzip_perSubj,
#                                  args=(folderIn, folderOut, pat, FeaturesParams.featNames, maxWinLenIndx, lebBeforeSeizIndx, lenAfterSeizIndx), callback=collect_result)
#             number_free_cores = number_free_cores - 1
#             if number_free_cores == 0:
#                 while number_free_cores == 0:  # synced in the callback
#                     time.sleep(0.1)
#                     pass
#         else:
#             if (type=='StoS'):
#                 concatenateFeatureFilesStoSApproach_gzip_perSubj(folderIn, folderOut, pat, FeaturesParams.featNames, lebBeforeSeizIndx, lenAfterSeizIndx)
#             else: #'FixedWin'
#                 concatenateFeatures_allDataInEqualWindows_gzip_perSubj(folderIn, folderOut, pat, FeaturesParams.featNames, maxWinLenIndx, lebBeforeSeizIndx, lenAfterSeizIndx)
#
#     while number_free_cores < n_cores:  # wait till all subjects have their data processed
#         time.sleep(0.1)
#         pass
#
#     if GeneralParams.parallelize:
#         pool.close()
#         pool.join()
#
# def concatenateFeatures_allDataInEqualWindows_gzip_perSubj(folderIn,folderOut,  pat, featNames, maxWinLenIndx, lebBeforeSeizIndx, lenAfterSeizIndx):
#     seizFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_s_'+featNames[0]+'.csv.gz'))
#     allFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+featNames[0]+'.csv.gz'))
#     numLettersToRemove=-len(featNames[0])-7
#     nonSeizFiles=list(set(list(allFiles)) - set(list(seizFiles)))
#     print('-- Patient:', pat, 'NumSeizures:', len(seizFiles))
#
#     IndxNonSeizFile=0
#     indxStart=0
#     dataMissing=maxWinLenIndx
#     newFileToSave=1
#     numFilesThisSubj=0
#     #LOAD ALL FILES ONE BY ONE
#     for fIndx, fileName in enumerate(allFiles):
#         pom, fileName1 = os.path.split(fileName)
#         fileNameOut =fileName1[0:numLettersToRemove-1]
#         print(fileNameOut)
#
#         numCh = 18
#         data = loadAndConcatenateAllFeatFilesForThisFile(fileName[0:numLettersToRemove], featNames, numCh)
#         numFeat = int(len(data[0, :]))
#
#         fileName2 = fileName[0:numLettersToRemove] + 'Labels.csv'
#         labels = readDataFromFile(fileName2)
#
#         #if there is seizure in file find start and stops
#         if (np.sum(labels)!=0):
#             diffSig=np.diff(np.squeeze(labels))
#             szStart=np.where(diffSig==1)[0]
#             szStop= np.where(diffSig == -1)[0]
#
#             #remove data after seizure (unless seizre within that period)
#             (data, labels)=removePreAndPostIctalAreas(data, labels, szStart, szStop, lebBeforeSeizIndx, lenAfterSeizIndx)
#
#             #update now position of seizrues
#             diffSig=np.diff(np.squeeze(labels))
#             szStart=np.where(diffSig==1)[0]
#             szStop= np.where(diffSig == -1)[0]
#             if (len(szStart)!=0 and len(szStop)==0):
#                 szStop=[len(labels)]
#
#         thisFileStillHasData=1
#         while (thisFileStillHasData==1):
#             #if enough data in file
#             if (indxStart + dataMissing< len(labels)):
#                 #check if we would cut seizure in half
#                 if (np.sum(labels)!=0):
#                     for s in range(len(szStart)):
#                         try:
#                             if ( szStart[s]<indxStart+dataMissing  and szStop[s]>indxStart+dataMissing ): #cut would be whenre seizure is
#                                 dataMissing=szStop[s]- indxStart #move cut to the end of the seizure
#                         except:
#                             print('error')
#
#                 if (newFileToSave==1):
#                     newData=data[indxStart:indxStart+dataMissing,:]
#                     newLabel=labels[indxStart:indxStart+dataMissing,:]
#                 else: #appending to existing file
#                     newData= np.vstack((newData,data[indxStart:indxStart+dataMissing,:]))
#                     newLabel =np.vstack((newLabel,labels[indxStart:indxStart+dataMissing,:]))
#                 #finished this new file to save
#                 fileNameOut2 = folderOut + '/Subj' + pat + '_f' + str(numFilesThisSubj).zfill(3)
#                 saveDataToFile(np.hstack((newData, newLabel)), fileNameOut2, 'gzip')
#                 numFilesThisSubj = numFilesThisSubj + 1
#                 newFileToSave = 1
#                 indxStart = indxStart+dataMissing #start where we stopped
#                 dataMissing = maxWinLenIndx
#                 thisFileStillHasData=1
#             else: #not enough data in file
#                 if (newFileToSave==1):
#                     newData=data[indxStart:,:] #read until the end of the file
#                     newLabel=labels[indxStart:,:]
#                 else: #appending to existing file
#                     newData= np.vstack((newData,data[indxStart:,:]))
#                     newLabel =np.vstack((newLabel,labels[indxStart:,:]))
#                 dataMissing = maxWinLenIndx - len(newLabel) #calculate how much data is missing
#                 indxStart = 0 #in next file start from beginning
#                 thisFileStillHasData=0 #this file has no more data, need to load new one
#                 newFileToSave=0
#     # save file - last file that will only contain nonseizure, but sometimes it is lot of data
#     fileNameOut2 = folderOut + '/Subj' + pat + '_f' + str(numFilesThisSubj).zfill(3)
#     saveDataToFile(np.hstack((newData, newLabel)), fileNameOut2, 'gzip')
#     return
#
#
# def concatenateFeatures_allDataInEqualWindows_FirstFileNeedsSeizure(folderIn, folderOut,GeneralParams,GeneralCHBMITParams, numHours):
#     '''loads fixed len data files and rearanges so that first file has to contain at least numHours of data and at least one seizure
#     if no seizure found it adds more and more files until seizure is included in the first file '''
#
#     global number_free_cores
#     print('Extracting .csv from CHB edf files')
#     if GeneralParams.parallelize:
#         n_cores  = mp.cpu_count()
#         n_cores = ceil(n_cores*GeneralParams.perc_cores)
#
#         if n_cores > len(GeneralCHBMITParams.patients):
#             n_cores = len(GeneralCHBMITParams.patients)
#
#         print('Number of used cores: ' + str(n_cores))
#
#         pool = mp.Pool(n_cores)
#         number_free_cores = n_cores
#
#     createFolderIfNotExists(folderOut)
#     for patIndx, pat in enumerate(GeneralCHBMITParams.patients):
#         if GeneralParams.parallelize:
#             pool.apply_async(concatenateFeatures_allDataInEqualWindows_FirstFileNeedsSeizure_perSubj,
#                              args=(folderIn, folderOut, pat, numHours), callback=collect_result)
#             number_free_cores = number_free_cores - 1
#             if number_free_cores == 0:
#                 while number_free_cores == 0:  # synced in the callback
#                     time.sleep(0.1)
#                     pass
#         else:
#             concatenateFeatures_allDataInEqualWindows_FirstFileNeedsSeizure_perSubj(folderIn, folderOut, pat, numHours)
#
#     while number_free_cores < n_cores:  # wait till all subjects have their data processed
#         time.sleep(0.1)
#         pass
#
#     if GeneralParams.parallelize:
#         pool.close()
#         pool.join()
#
#
# def concatenateFeatures_allDataInEqualWindows_FirstFileNeedsSeizure_perSubj(folderIn, folderOut, pat, numHours):
#     print('Subj:' + pat)
#     allFiles=np.sort(glob.glob(folderIn + '/*Subj' + pat + '*.csv.gz'))
#     firstFileCreated=0
#     numFilesThisSubj=0
#     for fIndx, fileName in enumerate(allFiles):
#         # reader = csv.reader(open(fileName, "r"))
#         # data0 = np.array(list(reader)).astype("float")
#         data0 = readDataFromFile(fileName)
#         data=data0[:,0:-1]
#         label=data0[:,-1]
#         pom, fileName1 = os.path.split(fileName)
#         fileNameOut = os.path.splitext(fileName1)[0][0:6]
#
#         if (firstFileCreated==0): #first file, append until at least one seizure
#             if (fIndx==0):
#                 dataOut=data
#                 labelOut=label
#             else:
#                 dataOut=np.vstack((dataOut,data))
#                 labelOut = np.hstack((labelOut, label))
#             if (np.sum(labelOut)>0 and fIndx>=numHours-1): #at least 6 h or at least 1 seizure in first file
#                 firstFileCreated=1
#                 fileNameOut2 = folderOut + '/' + fileNameOut + '_f' + str(numFilesThisSubj).zfill(3)
#                 # writeToCsvFile(dataOut, labelOut, fileNameOut2)
#                 saveDataToFile(np.hstack((dataOut, labelOut.reshape((-1,1)))), fileNameOut2, 'gzip')
#                 numFilesThisSubj = numFilesThisSubj + 1
#         else:  #not first file, just resave with different cv name
#             fileNameOut2 = folderOut + '/' + fileNameOut + '_f' + str(numFilesThisSubj).zfill(3)
#             # writeToCsvFile(data, label, fileNameOut2)
#             saveDataToFile(np.hstack((data,label.reshape((-1,1)))), fileNameOut2, 'gzip')
#             numFilesThisSubj = numFilesThisSubj + 1
#     return
#


def concatenateFeatureFilesStoSApproach_gzip(folderIn, folderOut, GeneralParams, FeaturesParams):
    ''' loads original files (with calculated features) and rearanges them in a way that each file contains data from end of previous seizure to beginning of new seizure
    only first file is from beginning to before first file and last file is until the end of all data available
    '''
    folderOut=folderOut+'/'
    createFolderIfNotExists(folderOut)
    winLen=4 #in sec
    winStep=0.5 #in sec
    lenBeforeSeiz=0 #60 # in sec
    lenAfterSeiz=0 #600 # in sec
    lebBeforeSeizIndx=int(lenBeforeSeiz/winStep)
    lenAfterSeizIndx = int(lenAfterSeiz / winStep)

    for patIndx, pat in enumerate(GeneralParams.patients):
        # seizFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_s_'+FeaturesParams.featNames[0]+'.csv.gz'))
        allFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+FeaturesParams.featNames[0]+'.csv.gz'))
        numLettersToRemove=-len(FeaturesParams.featNames[0])-7
        numFiles = len(allFiles)
        print('-- Patient:', pat, 'NumFiles:', numFiles)
        numSeiz=0
        newStart=1
        posStartNextTime=0
        for fIndx, fileName in enumerate(allFiles):
            pom, fileName1 = os.path.split(fileName)
            fileNameOut =fileName1[0:numLettersToRemove-1]
            print(fileNameOut)

            numCh = 18
            data = loadAndConcatenateAllFeatFilesForThisFile(fileName[0:numLettersToRemove], FeaturesParams, numCh)
            numFeat = int(len(data[0, :]))

            fileName2 = fileName[0:numLettersToRemove] + 'Labels.csv'
            labels = readDataFromFile(fileName2)

            if (np.sum(labels) == 0):  # if no seizures
                if (newStart == 1):
                    allData = data[posStartNextTime:, :]
                    allLabels = labels[posStartNextTime:, :]
                    newStart=0
                else:
                    allData = np.vstack((allData, data[posStartNextTime:, :]))
                    allLabels = np.vstack((allLabels, labels[posStartNextTime:, :]))
                posStartNextTime = 0
            elif (np.sum(labels) != 0):  # if  seizures

                differ=np.diff(np.squeeze(labels))
                seizStarts=np.where(differ==1)[0]
                seizStops = np.where(differ == -1)[0]
                print('NumSeiz in file=', len(seizStarts))
                for sIndx in range(len(seizStarts)):
                    seizIndxStart=int(seizStarts[sIndx])
                    seizIndxStop = int(seizStops[sIndx])
                    if (sIndx!=len(seizStarts)-1):
                        endIndxForNextFile=int(seizStarts[sIndx+1])
                    else:
                        endIndxForNextFile=len(labels)
                    startIndx=seizIndxStart-lebBeforeSeizIndx
                    stopIndx=seizIndxStop+lenAfterSeizIndx
                    #data to add
                    if (startIndx < 0 and newStart!=1):
                        allData = allData[0:startIndx, :]  # remove previous data
                        allLabels = allLabels[0:startIndx, :]
                    elif (startIndx >0 and newStart==0):
                        if (sIndx==0): #if first seizure in a file
                            #add non seizure data enough far from seizure start
                            allData = np.vstack((allData, data[posStartNextTime:startIndx, :]))
                            allLabels = np.vstack((allLabels, labels[posStartNextTime:startIndx, :]))
                        #add sizure data
                        allData = np.vstack((allData, data[seizIndxStart:seizIndxStop, :]))
                        allLabels = np.vstack((allLabels, labels[seizIndxStart:seizIndxStop, :]))
                    elif (startIndx > 0 and newStart == 1): #if first file and seizure
                        # add non seizure data enough far from seizure start
                        allData = data[posStartNextTime:startIndx, :]
                        allLabels = labels[posStartNextTime:startIndx, :]
                        # add sizure data
                        allData = np.vstack((allData, data[seizIndxStart:seizIndxStop, :]))
                        allLabels = np.vstack((allLabels, labels[seizIndxStart:seizIndxStop, :]))


                    #save file
                    # justName=os.path.split(fileName)[1]
                    fileNameOut = folderOut + '/Subj' + pat + '_f' + str(numSeiz).zfill(3)
                    saveDataToFile(np.hstack((allData, allLabels)), fileNameOut, 'gzip')
                    print('Saved file:', fileNameOut)
                    numSeiz=numSeiz +1

                    # start new data collection
                    if (stopIndx > len(labels)): #some amount of next file should not be used
                        posStartNextTime = stopIndx-len(labels)
                        newStart = 1
                    else: #part of this file should be used
                        posStartNextTime = 0
                        newStart = 0
                        allData = data[stopIndx:endIndxForNextFile, :]
                        allLabels =labels[stopIndx:endIndxForNextFile, :]

        # save file - last file that will only contain nonseizure, but sometimes it is lot of data
        # justName = os.path.split(fileName)[1]
        fileNameOut =folderOut + '/Subj' + pat + '_f' + str(numSeiz).zfill(3)
        saveDataToFile(np.hstack((allData, allLabels)), fileNameOut, 'gzip')
        print('Saved file:', fileNameOut)

def concatenateFeatures_allDataInEqualWindows_gzip(folderIn, folderOut,GeneralParams, FeaturesParams,  maxWinLen):
    ''' loads original files (with calculated features) and rearanges them in a way that each file contains equal amount of data
     amount of data corronds to maxWinLen and is usually 1h or 4h '''

    lenBeforeSeiz=0 #60 # in sec
    lenAfterSeiz= 0 #600 # in sec
    lebBeforeSeizIndx=int(lenBeforeSeiz/FeaturesParams.winStep)
    lenAfterSeizIndx = int(lenAfterSeiz / FeaturesParams.winStep)

    createFolderIfNotExists(folderOut)
    maxWinLenIndx=int(maxWinLen/FeaturesParams.winStep)
    for patIndx, pat in enumerate(GeneralParams.patients):
        seizFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_s_'+FeaturesParams.featNames[0]+'.csv.gz'))
        allFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+FeaturesParams.featNames[0]+'.csv.gz'))
        numLettersToRemove=-len(FeaturesParams.featNames[0])-7
        nonSeizFiles=list(set(list(allFiles)) - set(list(seizFiles)))
        print('-- Patient:', pat, 'NumSeizures:', len(seizFiles))

        IndxNonSeizFile=0
        indxStart=0
        dataMissing=maxWinLenIndx
        newFileToSave=1
        numFilesThisSubj=0
        #LOAD ALL FILES ONE BY ONE
        for fIndx, fileName in enumerate(allFiles):
            pom, fileName1 = os.path.split(fileName)
            fileNameOut =fileName1[0:numLettersToRemove-1]
            print(fileNameOut)

            numCh = 18
            data = loadAndConcatenateAllFeatFilesForThisFile(fileName[0:numLettersToRemove], FeaturesParams, numCh)
            numFeat = int(len(data[0, :]))

            fileName2 = fileName[0:numLettersToRemove] + 'Labels.csv'
            labels = readDataFromFile(fileName2)

            #if there is seizure in file find start and stops
            if (np.sum(labels)!=0):
                diffSig=np.diff(np.squeeze(labels))
                szStart=np.where(diffSig==1)[0]
                szStop= np.where(diffSig == -1)[0]

                #remove data after seizure (unless seizre within that period)
                (data, labels)=removePreAndPostIctalAreas(data, labels, szStart, szStop, lebBeforeSeizIndx, lenAfterSeizIndx)

                #update now position of seizrues
                diffSig=np.diff(np.squeeze(labels))
                szStart=np.where(diffSig==1)[0]
                szStop= np.where(diffSig == -1)[0]
                if (len(szStart)!=0 and len(szStop)==0):
                    szStop=[len(labels)]

            thisFileStillHasData=1
            while (thisFileStillHasData==1):
                #if enough data in file
                if (indxStart + dataMissing< len(labels)):
                    #check if we would cut seizure in half
                    if (np.sum(labels)!=0):
                        for s in range(len(szStart)):
                            try:
                                if ( szStart[s]<indxStart+dataMissing  and szStop[s]>indxStart+dataMissing ): #cut would be whenre seizure is
                                    dataMissing=szStop[s]- indxStart #move cut to the end of the seizure
                            except:
                                print('error')

                    if (newFileToSave==1):
                        newData=data[indxStart:indxStart+dataMissing,:]
                        newLabel=labels[indxStart:indxStart+dataMissing,:]
                    else: #appending to existing file
                        newData= np.vstack((newData,data[indxStart:indxStart+dataMissing,:]))
                        newLabel =np.vstack((newLabel,labels[indxStart:indxStart+dataMissing,:]))
                    #finished this new file to save
                    fileNameOut2 = folderOut + '/Subj' + pat + '_f' + str(numFilesThisSubj).zfill(3)
                    saveDataToFile(np.hstack((newData, newLabel)), fileNameOut2, 'gzip')
                    numFilesThisSubj = numFilesThisSubj + 1
                    newFileToSave = 1
                    indxStart = indxStart+dataMissing #start where we stopped
                    dataMissing = maxWinLenIndx
                    thisFileStillHasData=1
                else: #not enough data in file
                    if (newFileToSave==1):
                        newData=data[indxStart:,:] #read until the end of the file
                        newLabel=labels[indxStart:,:]
                    else: #appending to existing file
                        newData= np.vstack((newData,data[indxStart:,:]))
                        newLabel =np.vstack((newLabel,labels[indxStart:,:]))
                    dataMissing = maxWinLenIndx - len(newLabel) #calculate how much data is missing
                    indxStart = 0 #in next file start from beginning
                    thisFileStillHasData=0 #this file has no more data, need to load new one
                    newFileToSave=0
        # save file - last file that will only contain nonseizure, but sometimes it is lot of data
        fileNameOut2 = folderOut + '/Subj' + pat + '_f' + str(numFilesThisSubj).zfill(3)
        saveDataToFile(np.hstack((newData, newLabel)), fileNameOut2, 'gzip')


def concatenateFeatures_allDataInEqualWindows_FirstFileNeedsSeizure(folderIn, folderOut,GeneralParams, numHours):
    '''loads fixed len data files and rearanges so that first file has to contain at least numHours of data and at least one seizure
    if no seizure found it adds more and more files until seizure is included in the first file '''

    createFolderIfNotExists(folderOut)
    for patIndx, pat in enumerate(GeneralParams.patients):
        print('Subj:'+ pat)
        allFiles=np.sort(glob.glob(folderIn + '/*Subj' + pat + '*.csv.gz'))
        firstFileCreated=0
        numFilesThisSubj=0
        for fIndx, fileName in enumerate(allFiles):
            # reader = csv.reader(open(fileName, "r"))
            # data0 = np.array(list(reader)).astype("float")
            data0 = readDataFromFile(fileName)
            data=data0[:,0:-1]
            label=data0[:,-1]
            pom, fileName1 = os.path.split(fileName)
            fileNameOut = os.path.splitext(fileName1)[0][0:6]

            if (firstFileCreated==0): #first file, append until at least one seizure
                if (fIndx==0):
                    dataOut=data
                    labelOut=label
                else:
                    dataOut=np.vstack((dataOut,data))
                    labelOut = np.hstack((labelOut, label))
                if (np.sum(labelOut)>0 and fIndx>=numHours-1): #at least 6 h or at least 1 seizure in first file
                    firstFileCreated=1
                    fileNameOut2 = folderOut + '/' + fileNameOut + '_f' + str(numFilesThisSubj).zfill(3)
                    # writeToCsvFile(dataOut, labelOut, fileNameOut2)
                    saveDataToFile(np.hstack((dataOut, labelOut.reshape((-1,1)))), fileNameOut2, 'gzip')
                    numFilesThisSubj = numFilesThisSubj + 1
            else:  #not first file, just resave with different cv name
                fileNameOut2 = folderOut + '/' + fileNameOut + '_f' + str(numFilesThisSubj).zfill(3)
                # writeToCsvFile(data, label, fileNameOut2)
                saveDataToFile(np.hstack((data,label.reshape((-1,1)))), fileNameOut2, 'gzip')
                numFilesThisSubj = numFilesThisSubj + 1


#
#
# def concatenateFeatureFilesStoSApproach_gzip(folderIn, folderOut, GeneralParams, FeaturesParams):
#     ''' loads original files (with calculated features) and rearanges them in a way that each file contains data from end of previous seizure to beginning of new seizure
#     only first file is from beginning to before first file and last file is until the end of all data available
#     '''
#     folderOut=folderOut+'/'
#     createFolderIfNotExists(folderOut)
#     winLen=4 #in sec
#     winStep=0.5 #in sec
#     lenBeforeSeiz=0 #60 # in sec
#     lenAfterSeiz=0 #600 # in sec
#     lebBeforeSeizIndx=int(lenBeforeSeiz/winStep)
#     lenAfterSeizIndx = int(lenAfterSeiz / winStep)
#
#     for patIndx, pat in enumerate(GeneralParams.patients):
#         concatenateFeatureFilesStoSApproach_gzip_perSubj(folderIn, folderOut, pat, featNames, lebBeforeSeizIndx, lenAfterSeizIndx)
#         # seizFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_s_'+FeaturesParams.featNames[0]+'.csv.gz'))
#
# def concatenateFeatureFilesStoSApproach_gzip_perSubj(folderIn, folderOut, pat, featNames, lebBeforeSeizIndx, lenAfterSeizIndx):
#     allFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+featNames[0]+'.csv.gz'))
#     numLettersToRemove=-len(featNames[0])-7
#     numFiles = len(allFiles)
#     print('-- Patient:', pat, 'NumFiles:', numFiles)
#     numSeiz=0
#     newStart=1
#     posStartNextTime=0
#     for fIndx, fileName in enumerate(allFiles):
#         pom, fileName1 = os.path.split(fileName)
#         fileNameOut =fileName1[0:numLettersToRemove-1]
#         print(fileNameOut)
#
#         numCh = 18
#         data = loadAndConcatenateAllFeatFilesForThisFile(fileName[0:numLettersToRemove], featNames, numCh)
#         numFeat = int(len(data[0, :]))
#
#         fileName2 = fileName[0:numLettersToRemove] + 'Labels.csv'
#         labels = readDataFromFile(fileName2)
#
#         if (np.sum(labels) == 0):  # if no seizures
#             if (newStart == 1):
#                 allData = data[posStartNextTime:, :]
#                 allLabels = labels[posStartNextTime:, :]
#                 newStart=0
#             else:
#                 allData = np.vstack((allData, data[posStartNextTime:, :]))
#                 allLabels = np.vstack((allLabels, labels[posStartNextTime:, :]))
#             posStartNextTime = 0
#         elif (np.sum(labels) != 0):  # if  seizures
#
#             differ=np.diff(np.squeeze(labels))
#             seizStarts=np.where(differ==1)[0]
#             seizStops = np.where(differ == -1)[0]
#             print('NumSeiz in file=', len(seizStarts))
#             for sIndx in range(len(seizStarts)):
#                 seizIndxStart=int(seizStarts[sIndx])
#                 seizIndxStop = int(seizStops[sIndx])
#                 if (sIndx!=len(seizStarts)-1):
#                     endIndxForNextFile=int(seizStarts[sIndx+1])
#                 else:
#                     endIndxForNextFile=len(labels)
#                 startIndx=seizIndxStart-lebBeforeSeizIndx
#                 stopIndx=seizIndxStop+lenAfterSeizIndx
#                 #data to add
#                 if (startIndx < 0 and newStart!=1):
#                     allData = allData[0:startIndx, :]  # remove previous data
#                     allLabels = allLabels[0:startIndx, :]
#                 elif (startIndx >0 and newStart==0):
#                     if (sIndx==0): #if first seizure in a file
#                         #add non seizure data enough far from seizure start
#                         allData = np.vstack((allData, data[posStartNextTime:startIndx, :]))
#                         allLabels = np.vstack((allLabels, labels[posStartNextTime:startIndx, :]))
#                     #add sizure data
#                     allData = np.vstack((allData, data[seizIndxStart:seizIndxStop, :]))
#                     allLabels = np.vstack((allLabels, labels[seizIndxStart:seizIndxStop, :]))
#                 elif (startIndx > 0 and newStart == 1): #if first file and seizure
#                     # add non seizure data enough far from seizure start
#                     allData = data[posStartNextTime:startIndx, :]
#                     allLabels = labels[posStartNextTime:startIndx, :]
#                     # add sizure data
#                     allData = np.vstack((allData, data[seizIndxStart:seizIndxStop, :]))
#                     allLabels = np.vstack((allLabels, labels[seizIndxStart:seizIndxStop, :]))
#
#
#                 #save file
#                 # justName=os.path.split(fileName)[1]
#                 fileNameOut = folderOut + '/Subj' + pat + '_f' + str(numSeiz).zfill(3)
#                 saveDataToFile(np.hstack((allData, allLabels)), fileNameOut, 'gzip')
#                 print('Saved file:', fileNameOut)
#                 numSeiz=numSeiz +1
#
#                 # start new data collection
#                 if (stopIndx > len(labels)): #some amount of next file should not be used
#                     posStartNextTime = stopIndx-len(labels)
#                     newStart = 1
#                 else: #part of this file should be used
#                     posStartNextTime = 0
#                     newStart = 0
#                     allData = data[stopIndx:endIndxForNextFile, :]
#                     allLabels =labels[stopIndx:endIndxForNextFile, :]
#
#     # save file - last file that will only contain nonseizure, but sometimes it is lot of data
#     # justName = os.path.split(fileName)[1]
#     fileNameOut =folderOut + '/Subj' + pat + '_f' + str(numSeiz).zfill(3)
#     saveDataToFile(np.hstack((allData, allLabels)), fileNameOut, 'gzip')
#     print('Saved file:', fileNameOut)

# def loadDataFromFile(fileName):
#     data = torch.from_numpy(readDataFromFile(str(fileName)))
#     # if not os.path.exists(fileName.with_suffix('.npz')):
#     #     np.savez(fileName.with_suffix('.npz'), d=data)
#     # #         data= np.float32(data)
#     dataAll=data[:, 0:-1]
#     labelsAll=data[:, -1]
#     return (dataAll, labelsAll)
#
def loadDataFromFile_Repomse(fileName):
    data = pd.read_parquet(fileName)
    dataAll = torch.tensor(data.iloc[:, 1:].values) #.to_numpy()
    chNames = data.columns.tolist()[1:]
    labelsAll = torch.tensor(data['Labels'].values) #.to_numpy()
    # dataAll = (data.iloc[:, 1:]).to_numpy()
    # labelsAll = (data['Labels']).to_numpy()
    return (dataAll, labelsAll, chNames)
#
# def concatenateDataFromFiles(fileNames, dataset):
#     ''' loads and concatenates data from all files in file name list
#     creates array noting lengths of each file to know when new file started in appeded data'''
#     dataAll = []
#     labelsAll = []
#     startIndxOfFiles=np.zeros(len(fileNames),dtype=int)
#     lenPrevFile = 0
#     for f, fileName in enumerate(fileNames):
#         fileName = pathlib.Path(fileName.replace("".join(pathlib.Path(fileName).suffixes),""))
#         if (dataset=='Repomse'):
#             (data0, label0, _)=loadDataFromFile_Repomse(fileName)
#         else: #CHBMIT
#             data = torch.from_numpy(readDataFromFile(str(fileName)))
#             if not os.path.exists(fileName.with_suffix('.npz')):
#                 np.savez(fileName.with_suffix('.npz'),d=data)
#             data0=data[:,0:-1]
#             label0=data[:,-1]
#         dataAll.append(data0)
#         labelsAll.append(label0)
#         lenPrevFile += len(label0)
#         startIndxOfFiles[f]=lenPrevFile
#
#     dataAll = torch.vstack(dataAll)
#     labelsAll = torch.hstack(labelsAll).to(torch.int64)
#
#     # remove nan and inf from matrix
#     dataAll.nan_to_num_(nan=torch.nan, posinf=torch.nan, neginf=torch.nan)
#     col_mean = dataAll.nanmean(dim=0)
#     torch.where(dataAll.isnan(), col_mean, dataAll)
#     return (dataAll, labelsAll, startIndxOfFiles)
#
# def concatenateDataFromFiles_Repomse(fileNames):
#     ''' loads and concatenates data from all files in file name list
#     creates array noting lengths of each file to know when new file started in appeded data'''
#     dataAll = []
#     labelsAll = []
#     startIndxOfFiles=np.zeros(len(fileNames),dtype=int)
#     lenPrevFile = 0
#     for f, fileName in enumerate(fileNames):
#         fileName = pathlib.Path(fileName.replace("".join(pathlib.Path(fileName).suffixes),""))
#         data = torch.from_numpy(readDataFromFile(str(fileName)))
#         if not os.path.exists(fileName.with_suffix('.npz')):
#             np.savez(fileName.with_suffix('.npz'),d=data)
# #         data= np.float32(data)
#         dataAll.append(data[:,0:-1])
#         labelsAll.append(data[:,-1])
#         lenPrevFile += len(data[:, -1])
#         startIndxOfFiles[f]=lenPrevFile
#
#     dataAll = torch.vstack(dataAll)
#     labelsAll = torch.hstack(labelsAll).to(torch.int64)
#
#     # remove nan and inf from matrix
#     dataAll.nan_to_num_(nan=torch.nan, posinf=torch.nan, neginf=torch.nan)
#     col_mean = dataAll.nanmean(dim=0)
#     torch.where(dataAll.isnan(), col_mean, dataAll)
#     return (dataAll, labelsAll, startIndxOfFiles)
#
def concatenateDataFromFiles_withNormPerFile(fileNames, HDParams, FeaturesParams, dataset):
    ''' loads and concatenates data from all files in file name list
    creates array noting lengths of each file to know when new file started in appeded data
    also normalizes each feature individually for each loaded file '''
    dataAll = []
    labelsAll = []
    startIndxOfFiles=np.zeros(len(fileNames),dtype=int)
    lenPrevFile = 0
    for f, fileName in enumerate(fileNames):
        fileName = pathlib.Path(fileName.replace("".join(pathlib.Path(fileName).suffixes),""))

        if (dataset=='Repomse'):
            (data, label0, _)=loadDataFromFile_Repomse(fileName)
        else: #CHBMIT
            data0 = torch.from_numpy(readDataFromFile(str(fileName)))
            if not os.path.exists(fileName.with_suffix('.npz')):
                np.savez(fileName.with_suffix('.npz'),d=data0)
            data=data0[:,0:-1]
            label0=data0[:,-1]
#         data = torch.from_numpy(readDataFromFile(str(fileName)))
#         if not os.path.exists(fileName.with_suffix('.npz')):
#             np.savez(fileName.with_suffix('.npz'),d=data)
# #         data= np.float32(data)

        # remove nan and inf from matrix
        data.nan_to_num_(nan=torch.nan, posinf=torch.nan, neginf=torch.nan)
        col_mean = data.nanmean(dim=0)
        torch.where(data.isnan(), col_mean, data)

        if FeaturesParams.featNormWith == 'max':
            # normalize with min and max - bad if many outliers
            dataNorm, _ = normalizeAndDiscretizeTrainAndTestData(data[:,0:-1], data[:,0:-1],  HDParams.numSegmentationLevels,  FeaturesParams.featNorm)
        else:
            # normalize with percentile
            dataNorm, _ = normalizeAndDiscretizeTrainAndTestData_withPercentile(data[:,0:-1], data[:,0:-1], HDParams.numSegmentationLevels, FeaturesParams.featNorm, FeaturesParams.featNormPercentile)

        dataAll.append(dataNorm)
        labelsAll.append(label0)
        lenPrevFile += len(label0)
        startIndxOfFiles[f]=lenPrevFile

    dataAll = torch.vstack(dataAll)
    labelsAll = torch.hstack(labelsAll).to(torch.int64)
    return (dataAll, labelsAll, startIndxOfFiles)

# # def concatenateDataFromFiles(fileNames):
# #     ''' loads and concatenates data from all files in file name list
# #     creates array noting lengths of each file to know when new file started in appeded data'''
# #     dataAll = []
# #     startIndxOfFiles=np.zeros(len(fileNames))
# #     for f, fileName in enumerate(fileNames):
# #         data = readDataFromFile(fileName)
# #         data= np.float32(data)
# #
# #         if (dataAll == []):
# #             dataAll = data[:, 0:-1]
# #             labelsAll = data[:, -1].astype(int)
# #             lenPrevFile=int(len(data[:, -1]))
# #             startIndxOfFiles[f]=lenPrevFile
# #         else:
# #             dataAll = np.vstack((dataAll, data[:, 0:-1]))
# #             labelsAll = np.hstack((labelsAll, data[:, -1].astype(int)))
# #             # startIndxOfFiles[f]=int(lenPrevFile)
# #             lenPrevFile = lenPrevFile+ len(data[:, -1])
# #             startIndxOfFiles[f] = int(lenPrevFile)
# #     startIndxOfFiles = startIndxOfFiles.astype((int))
# #     return (dataAll, labelsAll, startIndxOfFiles)
#
def test_StandardML_moreModelsPossible(folder, fileName, data, trueLabels,  model, TrainTestType, PostprocessingParams, FeaturesParams, run_analysis=True):
    ''' gives predictions for standard machine learning models (not HD)
    returns predictions and probability
    calculates also simple accuracy and accuracy per class'''

    # number of clases
    (unique_labels, counts) = np.unique(trueLabels.cpu().numpy(), return_counts=True)
    numLabels = len(unique_labels)
    if (numLabels==1): #in specific case when in test set all the same label
        numLabels=2

    #PREDICT LABELS
    # X_test0 = np.float32(data)
    # X_test0[np.where(np.isinf(X_test0))] = np.nan
    # if (np.size(X_test0,0)==0):
    #         print('X test size 0 is 0!!', X_test0.shape)
    # if (np.size(X_test0,1)==0):
    #         print('X test size 1 is 0!!', X_test0.shape)
    # col_mean = np.nanmean(X_test0, axis=0)
    # inds = np.where(np.isnan(X_test0))
    # X_test0[inds] = np.take(col_mean, inds[1])
    # # if still somewhere nan replace with 0
    # X_test0[np.where(np.isnan(X_test0))] = 0
    # X_test=X_test0
    #
    # #calculate predictions
    # y_pred= model.predict(X_test)
    # y_probability = model.predict_proba(X_test)

    y_probability = torch.from_numpy(model.predict_proba(data.cpu())).to(data.device)
    y_pred= torch.argmax(y_probability,dim=1).to(data.device)

    selected_prob = y_probability.max(1)[0]
    # if torch.all(y_pred==0):
    #     print('no seiz predicted')
    # if torch.all(y_pred==1):
    #     print('no non seiz predicted')

    #calculate accuracy
    acc= ((y_pred == trueLabels).sum()/len(trueLabels)).item()

    # calculate performance and distances per class
    accPerClass=torch.zeros(numLabels)
    distFromCorr_PerClass = torch.zeros(numLabels)
    distFromWrong_PerClass = torch.zeros(numLabels)
    for l in range(numLabels):
        indx=torch.where(trueLabels==l)
        trueLabels_part=trueLabels[indx]
        predLab_part=y_pred[indx]
        diffLab = predLab_part - trueLabels_part
        indx2 = torch.where(diffLab == 0)
        if (len(indx[0])==0):
            accPerClass[l] = torch.nan
        else:
            accPerClass[l] = len(indx2[0]) / len(indx[0])

    print('Standard ML', TrainTestType + ':', acc)
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFPBefSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFPAftSeiz / FeaturesParams.winStep)

    (performance0, yPred_MovAvrgStep1, yPred_MovAvrgStep2,yPred_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(y_pred.to(data.device), trueLabels,  selected_prob.to(data.device),
                                                                         toleranceFP_bef, toleranceFP_aft,  numLabelsPerHour,
                                                                         seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
    dataToSave = torch.vstack((trueLabels, selected_prob, y_pred, yPred_MovAvrgStep1, yPred_MovAvrgStep2, yPred_SmoothBayes)).transpose(1,0)
    outputName = folder + '/' + fileName + '_RF_' + TrainTestType + 'Predictions.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')

    return y_pred

# def func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderIn, patients,PostprocessingParams, FeaturesParams , typeModel):
#     ''' goes through predictions of each file of each subject and appends them in time
#     plots predictions in time
#     also calculates average performance of each subject based on performance of whole appended predictions
#     also calculates average of all subjects
#     plots performances per subject and average of all subjects '''
#     # folderOut=folderIn +'/PerformanceWithAppendedTests/'
#     folderOut = folderIn + '/PerformanceWithAppendedTests_Bthr'+str(PostprocessingParams.bayesProbThresh)+'/'
#     createFolderIfNotExists(folderOut)
#
#     AllSubjDiffPerf_test = torch.zeros((len(patients), 4* 9))
#     # various postpocessing parameters
#     seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
#     seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
#     distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
#     numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
#     toleranceFP_bef = int(PostprocessingParams.toleranceFPBefSeiz / FeaturesParams.winStep)
#     toleranceFP_aft = int(PostprocessingParams.toleranceFPAftSeiz / FeaturesParams.winStep)
#
#     for patIndx, pat in enumerate(patients):
#         # filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+typeModel+'_TestPredictions.csv.gz'))
#         # if (typeModel=='RF'):
#         #     filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+typeModel+'_TestPredictions.csv.gz'))
#         # else:
#         #     filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_Test0Predictions.csv.gz'))
#         try:
#             filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_TestPredictions.csv.gz'))
#         except:
#             filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_Test0Predictions.csv.gz'))
#         if (len(filesAll) == 0):
#             filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_Test0Predictions.csv.gz'))
#         numFiles=len(filesAll)
#         print('APPENDING ' +typeModel + '--> Subj '+ pat + ' numFiles: ', numFiles)
#
#         if len(filesAll) == 0:
#             continue
#         d0,d1,d2,d3,d4,d5,d6 = [],[],[],[],[],[],[]
#         for cv in range(len(filesAll)):
#             fileName2 = 'Subj' + pat + '_cv' + str(cv).zfill(3)
#             data= torch.from_numpy(readDataFromFile(filesAll[cv]))
#             d0.append(data[:, 0])
#             d1.append(data[:, 1])
#             d2.append(data[:, 2])
#             d3.append(torch.ones(len(data[:, 0]))*(cv+1))
#             if 'Fact' in folderIn:
#                 d4.append(data[:, 3])
#                 d5.append(data[:, 4])
#                 d6.append(data[:, 5])
#         trueLabels_AllCV = torch.hstack(d0)
#         probabLabels_AllCV = torch.hstack(d1)
#         predLabels_AllCV=torch.hstack(d2)
#         dataSource_AllCV= torch.hstack(d3)
#         if 'Fact' in folderIn:
#             yPredTest_MovAvrgStep1_AllCV = torch.hstack(d4)
#             yPredTest_MovAvrgStep2_AllCV = torch.hstack(d5)
#             yPredTest_SmoothBayes_AllCV  = torch.hstack(d6)
#         #predictionsSmoothed = np.vstack((predictionsSmoothed,data[indxs, 3:-1]))
#
#         if (True or 'Fact' not in folderIn): #smooth and calculate performance
#             (performanceTest, yPredTest_MovAvrgStep1_AllCV, yPredTest_MovAvrgStep2_AllCV,yPredTest_SmoothBayes_AllCV) = calculatePerformanceAfterVariousSmoothing(predLabels_AllCV, trueLabels_AllCV,probabLabels_AllCV,
#                                                                                 toleranceFP_bef, toleranceFP_aft,
#                                                                                 numLabelsPerHour, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
#         else:
#             performanceTest = calculatePerformanceWithoutSmoothing(predLabels_AllCV, yPredTest_MovAvrgStep1_AllCV, yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV, trueLabels_AllCV, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
#
#         # # calculationg avrg for this subj over all CV
#         AllSubjDiffPerf_test[patIndx, :] =performanceTest
#
#         dataToSave = np.vstack((trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1_AllCV,yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV, dataSource_AllCV)).transpose()  # added from which file is specific part of test set
#         outputName = folderOut + '/Subj' + pat  + '_'+typeModel+'_Appended_TestPredictions.csv'
#         saveDataToFile(dataToSave, outputName, 'gzip')
#
#         smoothenLabels_Bayes_vizualize(predLabels_AllCV, probabLabels_AllCV, seizureStableLenToTestIndx,
#                                        PostprocessingParams.bayesProbThresh,
#                                        distanceBetweenSeizuresIndx, trueLabels_AllCV, folderOut,
#                                        'Subj' + pat  + '_'+typeModel+'_Appended_TestPredictions')
#
#         #plot predictions in time
#         # Plot predictions in time
#         fig1 = plt.figure(figsize=(12, 8), constrained_layout=False)
#         gs = GridSpec(6, 1, figure=fig1)
#         fig1.subplots_adjust(wspace=0.2, hspace=0.2)
#         xValues = torch.arange(0, len(trueLabels_AllCV), 1)
#         ax1 = fig1.add_subplot(gs[0, 0])
#         ax1.plot(xValues, predLabels_AllCV, 'k')
#         ax1.set_ylabel('NoSmooth')
#         ax1.set_title('Subj' + pat)
#         ax1.grid()
#         ax1 = fig1.add_subplot(gs[1, 0])
#         ax1.plot(xValues, yPredTest_MovAvrgStep1_AllCV * 0.8, 'b')
#         ax1.plot(xValues, yPredTest_MovAvrgStep2_AllCV, 'c')
#         ax1.set_ylabel('Step1&2')
#         ax1.grid()
#         ax1 = fig1.add_subplot(gs[2, 0])
#         probability_pos = np.where(predLabels_AllCV == 0, 1 - probabLabels_AllCV, probabLabels_AllCV)
#         ax1.plot(xValues, probability_pos, 'm')
#         ax1.set_ylabel('Probability')
#         ax1.set_ylim([0,1])
#         ax1.grid()
#         ax1 = fig1.add_subplot(gs[3, 0])
#         ax1.plot(xValues, yPredTest_SmoothBayes_AllCV, 'm')
#         ax1.set_ylabel('Bayes')
#         ax1.grid()
#         ax1 = fig1.add_subplot(gs[4, 0])
#         ax1.plot(xValues, trueLabels_AllCV, 'r')
#         ax1.set_ylabel('TrueLabel')
#         ax1.grid()
#         ax1 = fig1.add_subplot(gs[5, 0])
#         ax1.plot(xValues, dataSource_AllCV, 'k')
#         ax1.set_ylabel('FileNr')
#         ax1.grid()
#         ax1.set_xlabel('Time')
#         # if (GeneralParams.plottingON == 1):
#         fig1.show()
#         fig1.savefig(folderOut + '/Subj' + pat  + '_'+typeModel+'_Appended_TestPredictions.png', bbox_inches='tight')
#         plt.close(fig1)
#
#     # SAVE PERFORMANCE MEASURES FOR ALL SUBJ
#     outputName = folderOut + '/AllSubj_AppendedTest_' +typeModel+'_AllPerfMeasures.csv'
#     saveDataToFile(AllSubjDiffPerf_test, outputName, 'gzip')
#
#     #plot
#     fig1 = plt.figure(figsize=(10, 10), constrained_layout=False)
#     gs = GridSpec(3, 3, figure=fig1)
#     fig1.subplots_adjust(wspace=0.35, hspace=0.35)
#     #fig1.suptitle('All subject different performance measures ')
#     xValues = torch.arange(1, len(patients)+1, 1)
#     perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
#                  'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
#     numPerf = len(perfNames)
#     for perfIndx, perf in enumerate(perfNames):
#         ax1 = fig1.add_subplot(gs[perfIndx // 3, perfIndx % 3])
#         ax1.plot(xValues, AllSubjDiffPerf_test[:, 0 +perfIndx], 'k')
#         ax1.plot(xValues, AllSubjDiffPerf_test[:, 9 +perfIndx], 'b')
#         ax1.plot(xValues, AllSubjDiffPerf_test[:, 18 +perfIndx], 'c')
#         ax1.plot(xValues, AllSubjDiffPerf_test[:, 27 +perfIndx], 'm')
#         ax1.legend(['NoSmooth', 'Step1', 'Step2', 'Bayes'])
#         #plotting mean values
#         ax1.plot(xValues, torch.ones(len(xValues))*torch.nanmean(AllSubjDiffPerf_test[:, 0 +perfIndx]), 'k')
#         ax1.plot(xValues, torch.ones(len(xValues))*torch.nanmean(AllSubjDiffPerf_test[:, 9 +perfIndx]), 'b')
#         ax1.plot(xValues, torch.ones(len(xValues))*torch.nanmean(AllSubjDiffPerf_test[:, 18 +perfIndx]), 'c')
#         ax1.plot(xValues, torch.ones(len(xValues)) * torch.nanmean(AllSubjDiffPerf_test[:, 27 +perfIndx]), 'm')
#         ax1.set_xlabel('Time steps')
#         ax1.set_xticks(xValues)
#         ax1.set_xticklabels(patients, fontsize=8, rotation=45)
#         ax1.set_xlabel('Subjects')
#         ax1.set_title(perf)
#         ax1.grid()
#     # if (GeneralParams.plottingON == 1):
#     fig1.show()
#     fig1.savefig(folderOut + '/AllSubj_AppendedTest_' +typeModel+'_AllPerformanceMeasures.png', bbox_inches='tight')
#     plt.close(fig1)
#
def plot_PredictionsInTime_AllSmoothing(trueLabels_AllCV, predLabels_AllCV, probabLabels_AllCV, yPredTest_MovAvrgStep1_AllCV, yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV, dataSource_AllCV, folderOut, fileNameOut):
    # Plot predictions in time
    fig1 = plt.figure(figsize=(12, 8), constrained_layout=False)
    gs = GridSpec(6, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    xValues = torch.arange(0, len(trueLabels_AllCV), 1)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.plot(xValues, predLabels_AllCV, 'k')
    ax1.set_ylabel('NoSmooth')
    ax1.set_title(fileNameOut[1:7])
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 0])
    ax1.plot(xValues, yPredTest_MovAvrgStep1_AllCV * 0.8, 'b')
    ax1.plot(xValues, yPredTest_MovAvrgStep2_AllCV, 'c')
    ax1.set_ylabel('Step1&2')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[2, 0])
    probability_pos = np.where(predLabels_AllCV == 0, 1 - probabLabels_AllCV, probabLabels_AllCV)
    ax1.plot(xValues, probability_pos, 'm')
    ax1.set_ylabel('Probability')
    ax1.set_ylim([0, 1])
    ax1.grid()
    ax1 = fig1.add_subplot(gs[3, 0])
    ax1.plot(xValues, yPredTest_SmoothBayes_AllCV, 'm')
    ax1.set_ylabel('Bayes')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[4, 0])
    ax1.plot(xValues, trueLabels_AllCV, 'r')
    ax1.set_ylabel('TrueLabel')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[5, 0])
    ax1.plot(xValues, dataSource_AllCV, 'k')
    ax1.set_ylabel('FileNr')
    ax1.grid()
    ax1.set_xlabel('Time')
    # if (GeneralParams.plottingON == 1):
    fig1.show()
    fig1.savefig(folderOut + fileNameOut, bbox_inches='tight')
    plt.close(fig1)

def findOptThr_Bayes(predLabels, probabilityLabels, labelTrue,seizureStableLenToTestIndx, toleranceFP_bef, toleranceFP_aft):
    # convert probability to probability of pos
    probability_pos = torch.where(predLabels == 0, 1 - probabilityLabels, probabilityLabels)

    # first classifying as true 1 if at laest  GeneralParams.seizureStableLenToTest in a row is 1
    p = torch.nn.ConstantPad1d((seizureStableLenToTestIndx - 1, 0), 0)
    unfolded_probability = probability_pos.unfold(0, seizureStableLenToTestIndx, 1)
    conf_pos = unfolded_probability.prod(dim=1)
    conf_neg = (1 - unfolded_probability).prod(dim=1)
    conf = ((conf_pos + 0.00000001) / (conf_neg + 0.00000001)).log()


    #test different min thresholds for FP
    thrValues= np.arange(torch.min(conf)+ 0.5* (torch.max(conf)-torch.min(conf) ),torch.max(conf), 0.2)
    F1score, sensE, precisE=np.zeros(len(thrValues)), np.zeros(len(thrValues)), np.zeros(len(thrValues))
    for thrIndx, thr in enumerate(thrValues):
        # bayes smoothing
        # (yPred_SmoothBayes,_) = smoothenLabels_Bayes(predLabels, probabilityLabels, seizureStableLenToTestIndx, thr, 10, labelTrue)
        yPred_SmoothBayes = p(torch.where(conf >= thr, 1, 0))

        (sensE[thrIndx], precisE[thrIndx], F1score[thrIndx], totalFP) = performance_episodes(yPred_SmoothBayes, labelTrue, toleranceFP_bef, toleranceFP_aft)

    maxVal=np.nanmax(F1score)
    if np.isnan(maxVal):
        maxVal=0
        thrVal=1.5
    else:
        indxs=np.where(F1score == maxVal)[0]
        indx= indxs[int(len(indxs)/2)]
        # indx = np.where(F1score == maxVal)[0][-1]
        thrVal=thrValues[indx]
    # return (thrValues[np.argmax(F1score)], np.max(F1score))
    return(thrVal, maxVal)


def func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending_withBayesThrOpt(folderIn, patients,PostprocessingParams, FeaturesParams , typeModel):
    ''' goes through predictions of each file of each subject and appends them in time
    plots predictions in time
    also calculates average performance of each subject based on performance of whole appended predictions
    also calculates average of all subjects
    plots performances per subject and average of all subjects '''
    # folderOut=folderIn +'/PerformanceWithAppendedTests/'
    folderOut = folderIn + '/PerformanceWithAppendedTests_BthrOptimized/'
    createFolderIfNotExists(folderOut)

    AllSubjDiffPerf_test = torch.zeros((len(patients), 4* 9))
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFPBefSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFPAftSeiz / FeaturesParams.winStep)

    for patIndx, pat in enumerate(patients):
        try:
            filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_TestPredictions.csv.gz'))
        except:
            filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_Test0Predictions.csv.gz'))
        if (len(filesAll) == 0):
            filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_Test0Predictions.csv.gz'))
        numFiles=len(filesAll)
        print('APPENDING ' +typeModel + '--> Subj '+ pat + ' numFiles: ', numFiles)

        if len(filesAll) == 0:
            continue
        d0,d1,d2,d3,d4,d5,d6 = [],[],[],[],[],[],[]
        predLabels_AllCV, probabLabels_AllCV, trueLabels_AllCV, dataSource_AllCV, bayesThr_AllCV= torch.zeros((0)), torch.zeros((0)), torch.zeros((0)), torch.zeros((0)), torch.zeros((0))
        yPredTest_MovAvrgStep1_AllCV, yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV = torch.zeros((0)), torch.zeros((0)), torch.zeros((0))
        bayesProbThresh=PostprocessingParams.bayesProbThresh
        for cv in range(len(filesAll)):
            fileName2 = 'Subj' + pat + '_cv' + str(cv).zfill(3)
            data= torch.from_numpy(readDataFromFile(filesAll[cv]))

            if (cv!=0 ):
                if (torch.sum(trueLabels_AllCV)>0): #at least one seizure
                    #optimize bayes thr based on previous CV
                    (bayesProbThresh, perfValue)= findOptThr_Bayes(predLabels_AllCV, probabLabels_AllCV, trueLabels_AllCV, seizureStableLenToTestIndx,  toleranceFP_bef, toleranceFP_aft)

            (performanceTest, yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2,yPredTest_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(data[:, 2],
                                        data[:, 0],  data[:, 1], toleranceFP_bef, toleranceFP_aft , numLabelsPerHour, seizureStableLenToTestIndx,  seizureStablePercToTest,distanceBetweenSeizuresIndx,  bayesProbThresh)

            yPredTest_MovAvrgStep1_AllCV=torch.hstack((yPredTest_MovAvrgStep1_AllCV,yPredTest_MovAvrgStep1 ))
            yPredTest_MovAvrgStep2_AllCV=torch.hstack((yPredTest_MovAvrgStep2_AllCV, yPredTest_MovAvrgStep2))
            yPredTest_SmoothBayes_AllCV=torch.hstack((yPredTest_SmoothBayes_AllCV, yPredTest_SmoothBayes))
            predLabels_AllCV=torch.hstack((predLabels_AllCV, data[:, 2]))
            probabLabels_AllCV=torch.hstack((probabLabels_AllCV, data[:, 1]))
            trueLabels_AllCV=torch.hstack((trueLabels_AllCV, data[:, 0]))
            dataSource_AllCV=torch.hstack((dataSource_AllCV, torch.ones(len(data[:, 0]))*(cv+1) ))
            bayesThr_AllCV=torch.hstack((bayesThr_AllCV, torch.ones(len(data[:, 0]))*bayesProbThresh ))

        performanceTest=calculatePerformanceWithoutSmoothing(predLabels_AllCV, yPredTest_MovAvrgStep1_AllCV, yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV,
                                             trueLabels_AllCV, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)

        #calculationg avrg for this subj over all CV
        AllSubjDiffPerf_test[patIndx, :] =torch.from_numpy(performanceTest)

        dataToSave = np.vstack((trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1_AllCV,yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV, dataSource_AllCV,bayesThr_AllCV)).transpose()  # added from which file is specific part of test set
        outputName = folderOut + '/Subj' + pat  + '_'+typeModel+'_Appended_TestPredictions.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')

        smoothenLabels_Bayes_vizualize(predLabels_AllCV, probabLabels_AllCV, seizureStableLenToTestIndx, bayesThr_AllCV, distanceBetweenSeizuresIndx, trueLabels_AllCV, folderOut,
                                       'Subj' + pat  + '_'+typeModel+'_Appended_TestPredictions')
        #plot predictions in time
        plot_PredictionsInTime_AllSmoothing(trueLabels_AllCV, predLabels_AllCV, probabLabels_AllCV, yPredTest_MovAvrgStep1_AllCV, yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV,
                                            dataSource_AllCV, folderOut, '/Subj' + pat + '_' + typeModel + '_Appended_TestPredictions.png')


    # SAVE PERFORMANCE MEASURES FOR ALL SUBJ
    outputName = folderOut + '/AllSubj_AppendedTest_' +typeModel+'_AllPerfMeasures.csv'
    saveDataToFile(AllSubjDiffPerf_test, outputName, 'gzip')

    plot_PerfPerSubject_AllSmoothing(patients, AllSubjDiffPerf_test, folderOut,'/AllSubj_AppendedTest_' +typeModel+'_AllPerformanceMeasures.png')

def plot_PerfPerSubject_AllSmoothing( patients, AllSubjDiffPerf, folderOut, fileNameOut ):
    #plot
    fig1 = plt.figure(figsize=(10, 10), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.35, hspace=0.35)
    #fig1.suptitle('All subject different performance measures ')
    xValues = torch.arange(1, len(patients)+1, 1)
    perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
                 'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
    numPerf = len(perfNames)
    for perfIndx, perf in enumerate(perfNames):
        ax1 = fig1.add_subplot(gs[perfIndx // 3, perfIndx % 3])
        ax1.plot(xValues, AllSubjDiffPerf[:, 0 +perfIndx], 'k')
        ax1.plot(xValues, AllSubjDiffPerf[:, 9 +perfIndx], 'b')
        ax1.plot(xValues, AllSubjDiffPerf[:, 18 +perfIndx], 'c')
        ax1.plot(xValues, AllSubjDiffPerf[:, 27 +perfIndx], 'm')
        ax1.legend(['NoSmooth', 'Step1', 'Step2', 'Bayes'])
        #plotting mean values
        ax1.plot(xValues, torch.ones(len(xValues))*torch.nanmean(AllSubjDiffPerf[:, 0 +perfIndx]), 'k')
        ax1.plot(xValues, torch.ones(len(xValues))*torch.nanmean(AllSubjDiffPerf[:, 9 +perfIndx]), 'b')
        ax1.plot(xValues, torch.ones(len(xValues))*torch.nanmean(AllSubjDiffPerf[:, 18 +perfIndx]), 'c')
        ax1.plot(xValues, torch.ones(len(xValues)) * torch.nanmean(AllSubjDiffPerf[:, 27 +perfIndx]), 'm')
        ax1.set_xlabel('Time steps')
        ax1.set_xticks(xValues)
        ax1.set_xticklabels(patients, fontsize=8, rotation=45)
        ax1.set_xlabel('Subjects')
        ax1.set_title(perf)
        ax1.grid()
    # if (GeneralParams.plottingON == 1):
    fig1.show()
    fig1.savefig(folderOut + fileNameOut, bbox_inches='tight')
    plt.close(fig1)



def smoothenLabels_Bayes_vizualize(prediction, probability, seizureStableLenToTestIndx, probThresh, distanceBetweenSeizuresIndx,labels, folderOut, fileName):
    ''' returns labels bayes postprocessing
    calculates cummulative probability of seizure and non seizure over the window of size seizureStableLenToTestIndx
    if log (cong_pos /cong_ned )> probThresh then seizure '''

    # convert probability to probability of pos
    probability_pos = torch.where(prediction == 0, 1 - probability, probability)

    # labels = labels.reshape(len(labels))
    smoothLabels = torch.zeros_like(prediction)
    try:
        a = int(seizureStableLenToTestIndx)
    except:
        print('error seizureStableLenToTestIndx')
        print(seizureStableLenToTestIndx)
    try:
        a = int(len(prediction))
    except:
        print('error prediction')
        print(prediction)

    # first classifying as true 1 if at laest  GeneralParams.seizureStableLenToTest in a row is 1
    p = torch.nn.ConstantPad1d((seizureStableLenToTestIndx - 1, 0), 0)
    unfolded_probability = probability_pos.unfold(0, seizureStableLenToTestIndx, 1)
    conf_pos = unfolded_probability.prod(dim=1)
    conf_neg = (1 - unfolded_probability).prod(dim=1)
    conf = ((conf_pos + 0.00000001) / (conf_neg + 0.00000001)).log()
    if (np.isscalar(probThresh)):
        out = p(torch.where(conf >= probThresh, 1, 0))
    else:
        out = p(torch.where(conf >= probThresh[len(probThresh)-len(conf):], 1, 0))

    # #conf value when seiz
    # indxS=torch.where(labels==1)[0]
    # confSeiz=torch.max(conf[indxS])
    # indxNS=torch.where(labels==0)[0]
    # indxNS = indxNS[torch.where(indxNS < len(conf))[0]]
    # confNonSeiz=torch.max(conf[indxNS])
    # print('Bayes conf S:', confSeiz,' NS: ', confNonSeiz)

    # merge close seizures
    # second part
    smoothLabelsStep2 = torch.clone(out)
    # Calculate seizure starts and stops
    events = calculateStartsAndStops(smoothLabelsStep2)
    # if  seizure started but is too close to previous one, delete second seizure by connecting them
    for idx in range(1, len(events)):
        if events[idx][0] - events[idx - 1][1] <= distanceBetweenSeizuresIndx:
            smoothLabelsStep2[events[idx - 1][1]:events[idx][0]] = 1

    # Plot predictions in time
    fig1 = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = GridSpec(3, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    xValues = torch.arange(0, len(labels), 1)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.plot(xValues, softabs(probability_pos, 20), 'm')
    ax1.plot(xValues, softabs(probability_pos,10),'b')
    ax1.plot(xValues, probability_pos, 'k')
    ax1.plot(torch.arange(0, len(labels)), labels, 'r')
    ax1.legend([ 'Softabs 20','Softabs 10', 'Raw', 'True lab'])
    ax1.set_title('Probability_pos')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 0])
    ax1.plot(torch.arange(0, len(conf_pos), 1), conf_neg, 'b')
    ax1.plot(torch.arange(0, len(conf_pos), 1), conf_pos, 'r')
    ax1.legend(['Conf ned prod','Conf pos prod'])
    ax1.set_title('Conf product ')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[2, 0])
    ax1.plot(torch.arange(0, len(conf)), conf, 'k')
    ax1.plot(torch.arange(0, len(out)),out*2, 'b')
    ax1.plot(torch.arange(0, len(smoothLabelsStep2)), smoothLabelsStep2*1.5, 'm')
    ax1.plot(torch.arange(0, len(labels)), labels, 'r')
    if (np.isscalar(probThresh)):
        ax1.plot(torch.arange(0, len(labels)), torch.ones(len(labels))*probThresh, 'darkorange')
    else:
        ax1.plot( torch.arange(0, len(labels)), probThresh, 'darkorange')
    ax1.legend(['PosNeg conf', 'PosNeg Thresholded','PosNeg Smoothed','True lab',])
    ax1.set_title('Final decisions')
    ax1.grid()

    fig1.show()
    fig1.savefig(folderOut + '/' + fileName + '_BayesProbs.png', bbox_inches='tight')
    plt.close(fig1)

    return (out, smoothLabelsStep2)

def func_plotPerformancePerTimeSegStep(allPerformances, folderOut,patient,typeModel, timeSteps):
    if (torch.is_tensor(allPerformances)):
        allPerformances=allPerformances.cpu().detach().numpy()

    numTimeSeg=allPerformances.shape[0]
    # load performance measures per subject
    fig1 = plt.figure(figsize=(10, 10), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.35, hspace=0.35)
    # fig1.suptitle('All subject different performance measures ')
    xValues = np.arange(1,numTimeSeg+1, 1)
    perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
                 'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
    numPerf = len(perfNames)
    for perfIndx, perf in enumerate(perfNames):
        ax1 = fig1.add_subplot(gs[int(np.floor(perfIndx / 3)), np.mod(perfIndx, 3)])
        ax1.plot(xValues, allPerformances[:, 0 +perfIndx], 'k--')
        ax1.plot(xValues, allPerformances[:, 9 +perfIndx], 'b--')
        ax1.plot(xValues, allPerformances[:, 18 +perfIndx], 'c--')
        ax1.plot(xValues, allPerformances[:, 27 +perfIndx], 'm--')
        ax1.legend(['NoSmooth', 'Step1', 'Step2', 'Bayes'])
        #plotting mean values
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(allPerformances[:, 0 +perfIndx]), 'k--')
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(allPerformances[:, 9 +perfIndx]), 'b--')
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(allPerformances[:, 18 +perfIndx]), 'c--')
        ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(allPerformances[:, 27 +perfIndx]), 'm--')
        ax1.set_xlabel('Time steps')
        ax1.set_xticks(xValues)
        ax1.set_xticklabels(timeSteps, fontsize=10) #, rotation=45)
        ax1.set_title(perf)
        ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/'+patient+'_'+typeModel+'_AllPerformancePerTimeSeg.png', bbox_inches='tight')
    plt.close(fig1)


#
# def func_plotPerformancePerTimeSegStep_allSubjBoxplot(allPerformances, folderOut,patient,typeModel, timeSteps):
#     if (torch.is_tensor(allPerformances)):
#         allPerformances=allPerformances.cpu().detach().numpy()
#     numSubj=allPerformances.shape[2]
#     numTimeSeg=allPerformances.shape[0]
#     # load performance measures per subject
#     fig1 = plt.figure(figsize=(10, 10), constrained_layout=False)
#     gs = GridSpec(3, 3, figure=fig1)
#     fig1.subplots_adjust(wspace=0.35, hspace=0.35)
#     # fig1.suptitle('All subject different performance measures ')
#     xValues = np.arange(1,numTimeSeg+1, 1)
#     perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
#                  'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
#     numPerf = len(perfNames)
#     for perfIndx, perf in enumerate(perfNames):
#         for subj in range(numSubj):
#             dataAppend = np.vstack( (allPerformances[:, 27 + perfIndx,subj],  timeSteps)).transpose() #np.repeat(perfNames[perfIndx], numTimeSeg),
#             if (subj==0):
#                 AllPerfAllSubj= pd.DataFrame(dataAppend, columns=['Performance', 'TimeDiff'])
#             else:
#                 AllPerfAllSubj = AllPerfAllSubj.append(pd.DataFrame(dataAppend, columns=['Performance', 'TimeDiff']))
#
#         ax1 = fig1.add_subplot(gs[int(np.floor(perfIndx / 3)), np.mod(perfIndx, 3)])
#         sns.boxplot(x='TimeDiff', y='Performance', width=0.5, data=AllPerfAllSubj, palette="Set1") #hue='Approach'
#         ax1.set_title(perf)
#         ax1.legend(loc='lower left')
#         ax1.grid(which='both')
#
#     fig1.show()
#     fig1.savefig(folderOut + '/AllSubj_'+typeModel+'_AllPerformancePerTimeSeg_BoxPlot.png', bbox_inches='tight')
#     plt.close(fig1)
#
def func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending_PerTimeSeg(folderIn, patients,PostprocessingParams, FeaturesParams , HDParams, typeModel):
    ''' goes through predictions of each file of each subject and appends them in time
    plots predictions in time
    also calculates average performance of each subject based on performance of whole appended predictions
    also calculates average of all subjects
    plots performances per subject and average of all subjects '''
    folderOut=folderIn +'/PerformanceWithAppendedTests/'
    createFolderIfNotExists(folderOut)

    AllSubjDiffPerf_test = torch.zeros((len(patients), 4* 9))
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFPBefSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFPAftSeiz / FeaturesParams.winStep)

    numTimeSeg = len(HDParams.timeStepsInSec)
    allSubjPerformances = torch.zeros((numTimeSeg, 36, len(patients)))
    for patIndx, pat in enumerate(patients):
        try:
            filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_Test_PredAndProbabPerHDSegments.csv.gz')) #_PredAndProbabPerHDSegments
        except:
            filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_Test0_PredAndProbabPerHDSegments.csv.gz'))
        if (len(filesAll) == 0):
            filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_' + typeModel + '_Test0_PredAndProbabPerHDSegments.csv.gz'))
        numFiles=len(filesAll)
        print('APPENDING ' +typeModel + '--> Subj '+ pat + ' numFiles: ', numFiles)

        if len(filesAll) == 0:
            continue

        allData=torch.zeros((0, numTimeSeg*2+1))
        for cv in range(len(filesAll)):
            fileName2 = 'Subj' + pat + '_cv' + str(cv).zfill(3)
            data= torch.from_numpy(readDataFromFile(filesAll[cv]))
            allData=torch.vstack((allData, data))
        allPerformances=torch.zeros((numTimeSeg, 36))

        # Plot predictions in time
        fig1 = plt.figure(figsize=(16, 10), constrained_layout=False)
        gs = GridSpec(numTimeSeg, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.2, hspace=0.2)
        xValues = torch.arange(0, allData.shape[0], 1)
        for s in range(numTimeSeg):
            (allPerformances[s,:], yPredTest_MovAvrgStep1_AllCV, yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV) = calculatePerformanceAfterVariousSmoothing(allData[:, s+1] ,allData[:,0], allData[:, numTimeSeg+s+1],
                                                                                      toleranceFP_bef, toleranceFP_aft,  numLabelsPerHour,  seizureStableLenToTestIndx,seizureStablePercToTest, distanceBetweenSeizuresIndx,  PostprocessingParams.bayesProbThresh)

            ax1 = fig1.add_subplot(gs[s, 0])
            ax1.plot(xValues, allData[:, s+1] , 'b')  # predicted labels
            probability_pos = torch.where(allData[:,0] == 0, 1 - allData[:, numTimeSeg+s+1], allData[:, numTimeSeg+s+1])
            ax1.plot(xValues, probability_pos , 'k')  # probability of seizure labels

            ax1.plot(xValues, allData[:,0]*0.8+1.2, 'r') # true labels
            ax1.plot(xValues, yPredTest_SmoothBayes_AllCV*0.6+1.2,'m')
            ax1.set_ylabel(str(HDParams.timeStepsInSec[s]) +'s')
            ax1.set_ylim([0,2.3])
            ax1.grid()
        ax1.set_xlabel('Time')
        fig1.show()
        fig1.savefig(folderOut + '/Subj' + pat  + '_'+typeModel+'_Appended_PredAllTimeSegments.png', bbox_inches='tight')
        plt.close(fig1)

        #save performance for this subje
        outputName = folderOut + '/Subj' + pat  + '_'+typeModel+'_Appended_PerformanceForAllTimeSeg.csv'
        saveDataToFile(allPerformances, outputName, 'gzip')
        allSubjPerformances[:,:,patIndx]=allPerformances

        #plot performanc for this subj
        func_plotPerformancePerTimeSegStep(allPerformances, folderOut, 'Subj'+pat, typeModel, HDParams.timeStepsInSec)

    # # SAVE PERFORMANCE MEASURES FOR ALL SUBJ
    allSubjPerformancesMean=torch.mean(allSubjPerformances,dim=2)
    # save performance for this subje
    outputName = folderOut + '/AllSubj_' + typeModel + '_Appended_PerformanceForAllTimeSeg.csv'
    saveDataToFile(allSubjPerformancesMean, outputName, 'gzip')

    # plot performanc for all subj
    func_plotPerformancePerTimeSegStep(allSubjPerformancesMean, folderOut, 'AllSubj' , typeModel, HDParams.timeStepsInSec)



    # outputName = folderOut + '/AllSubj_AppendedTest_' +typeModel+'_AllPerfMeasures.csv'
    # saveDataToFile(AllSubjDiffPerf_test, outputName, 'gzip')
    #
    # #plot
    # fig1 = plt.figure(figsize=(10, 10), constrained_layout=False)
    # gs = GridSpec(3, 3, figure=fig1)
    # fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    # #fig1.suptitle('All subject different performance measures ')
    # xValues = torch.arange(1, len(patients)+1, 1)
    # perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
    #              'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
    # numPerf = len(perfNames)
    # for perfIndx, perf in enumerate(perfNames):
    #     ax1 = fig1.add_subplot(gs[perfIndx // 3, perfIndx % 3])
    #     ax1.plot(xValues, AllSubjDiffPerf_test[:, 0 +perfIndx], 'k')
    #     ax1.plot(xValues, AllSubjDiffPerf_test[:, 9 +perfIndx], 'b')
    #     ax1.plot(xValues, AllSubjDiffPerf_test[:, 18 +perfIndx], 'c')
    #     ax1.plot(xValues, AllSubjDiffPerf_test[:, 27 +perfIndx], 'm')
    #     ax1.legend(['NoSmooth', 'Step1', 'Step2', 'Bayes'])
    #     #plotting mean values
    #     ax1.plot(xValues, torch.ones(len(xValues))*torch.nanmean(AllSubjDiffPerf_test[:, 0 +perfIndx]), 'k')
    #     ax1.plot(xValues, torch.ones(len(xValues))*torch.nanmean(AllSubjDiffPerf_test[:, 9 +perfIndx]), 'b')
    #     ax1.plot(xValues, torch.ones(len(xValues))*torch.nanmean(AllSubjDiffPerf_test[:, 18 +perfIndx]), 'c')
    #     ax1.plot(xValues, torch.ones(len(xValues)) * torch.nanmean(AllSubjDiffPerf_test[:, 27 +perfIndx]), 'm')
    #     ax1.set_xlabel('Subjects')
    #     ax1.set_title(perf)
    #     ax1.grid()
    # # if (GeneralParams.plottingON == 1):
    # fig1.show()
    # fig1.savefig(folderOut + '/AllSubj_AppendedTest_' +typeModel+'_AllPerformanceMeasures.png', bbox_inches='tight')
    # plt.close(fig1)

# def analysePerformancePerTime(folderIn0, patients,PostprocessingParams, FeaturesParams , HDParams, typeModel):
#
#     folderIn = folderIn0 + '/PerformanceWithAppendedTests/'
#     folderOut=folderIn +'/PerformancePerTimeDifferences/'
#     createFolderIfNotExists(folderOut)
#     numTimeSeg = len(HDParams.timeStepsInSec)
#     allSubjPerformances = np.zeros((numTimeSeg, 36, len(patients)))
#     bestTimePerSubj=np.zeros((len(patients), 3)) #per columns: perf with noTime, best time shift, perf with best time
#
#     for patIndx, pat in enumerate(patients):
#         print('Subj '+pat)
#         # save performance for this subje
#         outputName = folderIn + '/Subj' + pat + '_' + typeModel + '_Appended_PerformanceForAllTimeSeg.csv.gzip'
#         # saveDataToFile(allPerformances, outputName, 'gzip')
#         allPerformances=readDataFromFile(outputName)
#         allSubjPerformances[:, :, patIndx] = allPerformances
#
#         #find best tim Indx basd on F1score episodes with bayes
#         try:
#             bestTimePerSubj[patIndx,:]=[ allPerformances[0,27+6], HDParams.timeStepsInSec[int(np.nanargmax(allPerformances[:,27+6]))], np.nanmax(allPerformances[:,27+6])]
#         except:
#             print('something wrong with calculating best time')
#             bestTimePerSubj[patIndx, :] = [allPerformances[0, 27 + 6], 0,allPerformances[0, 27 + 6]]
#
#         # plot performanc for this subj
#         # func_plotPerformancePerTimeSegStep(allPerformances, folderOut, 'Subj' + pat, typeModel, HDParams.timeStepsInSec)
#
#     # # SAVE PERFORMANCE MEASURES FOR ALL SUBJ
#     allSubjPerformancesMean = np.mean(allSubjPerformances, axis=2)
#     # save performance for this subje
#     outputName = folderOut + '/AllSubj_' + typeModel + '_Appended_PerformanceForAllTimeSeg.csv'
#     saveDataToFile(allSubjPerformancesMean, outputName, 'gzip')
#
#     # plot performanc for all subj
#     func_plotPerformancePerTimeSegStep(allSubjPerformancesMean, folderOut, 'AllSubj', typeModel, HDParams.timeStepsInSec)
#     # boxplot for all subj
#     func_plotPerformancePerTimeSegStep_allSubjBoxplot(allSubjPerformances, folderOut, 'AllSubj', typeModel, HDParams.timeStepsInSec)
#
#     #plot best timeDiff per subj
#     fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
#     gs = GridSpec(2, 1, figure=fig1)
#     fig1.subplots_adjust(wspace=0.35, hspace=0.35)
#     # fig1.suptitle('All subject different performance measures ')
#     xValues = np.arange(1, len(patients) + 1, 1)
#     ax1 = fig1.add_subplot(gs[0,0])
#     ax1.plot(xValues, bestTimePerSubj[:, 0], 'kx')
#     ax1.plot(xValues, bestTimePerSubj[:, 2], 'bx')
#     ax1.legend(['No time', 'Best time diff'])
#     ax1.set_xlabel('Patient')
#     ax1.set_xticks(xValues)
#     ax1.set_xticklabels(patients, fontsize=10)  # , rotation=45)
#     ax1.set_title('Performance comparison')
#     ax1.grid()
#     ax1 = fig1.add_subplot(gs[1, 0])
#     ax1.plot(xValues, bestTimePerSubj[:, 1], 'rx')
#     ax1.set_xlabel('Patient')
#     ax1.set_xticks(xValues)
#     ax1.set_xticklabels(patients, fontsize=10)  # , rotation=45)
#     ax1.set_title('Optimal time difference')
#     ax1.grid()
#     fig1.show()
#     fig1.savefig(folderOut + '/AllSubj_' + typeModel + '_BestTimeDiffPerSubj.png', bbox_inches='tight')
#     plt.close(fig1)
#
#
def func_plotPredictionsOfDifferentModels(modelsList, patients, folderIn, folderOut):
    ''' loads predictions in time of different models (from appended and smoothed predictions)
    and plots predictions in time to compare how different ML models perform
    also plots true label and from which file data is
    '''
    createFolderIfNotExists(folderOut)

    for patIndx, pat in enumerate(patients):
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(3, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.4, hspace=0.4)
        fig1.suptitle('Subj '+ pat)

        for mIndx, mName in enumerate(modelsList):
            #trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1,yPredTest_MovAvrgStep2, yPredTest_SmoothBayes,dataSource_AllCV
            inName = folderIn + '/PerformanceWithAppendedTests/Subj' + pat + '_' + mName + '_Appended_TestPredictions.csv'
            inName = folderIn + '/Subj' + pat + '_' + mName + '_Appended_TestPredictions.csv'
            print('FILE NAME:', inName)
            data=readDataFromFile(inName)

            if (mIndx==0):
                timeRaw = np.arange(0, len(data[:,0]))*0.5
                ax1 = fig1.add_subplot(gs[0, 0])#plot true labels
                ax1.plot(timeRaw, data[:,0] , 'r')
                ax1.set_ylabel('True labels')
                ax1.set_xlabel('Time')
                # ax1.set_title('Raw data')
                ax1.grid()
                ax1 = fig1.add_subplot(gs[2, 0])  # plot file source labels
                ax1.plot(timeRaw, data[:, 6], 'k')
                ax1.set_ylabel('File source')
                ax1.set_xlabel('Time')
                # ax1.set_title('Raw data')
                ax1.grid()
                ax2 = fig1.add_subplot(gs[1, 0])

            ax2.plot(timeRaw, data[:,2] * 0.3 + mIndx, 'k', label='NoSmooth')
            ax2.plot(timeRaw,data[:,3] * 0.5 + mIndx, 'b', label='Avrg_Step1')
            ax2.plot(timeRaw,data[:,4] * 0.5 + mIndx, 'c', label='Avrg_step2')
            ax2.plot(timeRaw, data[:, 5]* 0.7 + mIndx, 'm', label='Bayes')
            probability_pos = np.where(data[:,2] == 0, 1 - data[:, 1], data[:, 1])
            ax2.plot(timeRaw, probability_pos * 0.7 + mIndx, 'orange', label='Probability')
            if (mIndx == 0):
                ax2.legend()

        ax2.set_yticks(np.arange(0, len(modelsList), 1))
        ax2.set_yticklabels(modelsList, fontsize=10 * 0.8)
        ax2.set_xlabel('Time')
        ax2.grid()
        fig1.savefig(folderOut + '/Subj' + pat + '_PredictionsTest_AllModelsComparison.png')
        plt.close(fig1)


def func_plotPredictionsOfDifferentModels_v2(modelsList, patients, folderIn, folderOut):
    ''' loads predictions in time of different models (from appended and smoothed predictions)
    and plots predictions in time to compare how different ML models perform
    also plots true label and from which file data is
    plots for each RF/stdHD/onlHD bigger
    '''
    createFolderIfNotExists(folderOut)

    for patIndx, pat in enumerate(patients):
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(1, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.4, hspace=0.4)
        fig1.suptitle('Subj '+ pat)
        ax2 = fig1.add_subplot(gs[0, 0])  # plot true labels
        for mIndx, mName in enumerate(modelsList):
            #trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1,yPredTest_MovAvrgStep2, yPredTest_SmoothBayes,dataSource_AllCV
            inName = folderIn + '/PerformanceWithAppendedTests/Subj' + pat + '_' + mName + '_Appended_TestPredictions.csv'
            inName = folderIn + '/Subj' + pat + '_' + mName + '_Appended_TestPredictions.csv'
            # print('FILE NAME:', inName)
            data=readDataFromFile(inName)
            timeRaw = np.arange(0, len(data[:,0]))*0.5
            ax2.plot(timeRaw, data[:,2] * 0.3 + mIndx, 'k', label='NoSmooth')
            ax2.plot(timeRaw,data[:,3] * 0.5 + mIndx, 'b', label='Avrg_Step1')
            ax2.plot(timeRaw,data[:,4] * 0.5 + mIndx, 'c', label='Avrg_step2')
            ax2.plot(timeRaw, data[:, 5]* 0.7 + mIndx, 'm', label='Bayes')
            probability_pos = np.where(data[:,2] == 0, 1 - data[:, 1], data[:, 1])
            ax2.plot(timeRaw, probability_pos * 0.7 + mIndx, 'orange', label='Probability')
            if (mIndx == 0):
                ax2.plot(timeRaw, data[:, 0]*0.5 + 3, 'r', label='True labels')
                ax2.plot(timeRaw, data[:, 6]/(2*np.max(data[:, 6])) + 4, 'k', label='Source file')
                ax2.set_xlabel('Time')
                ax2.grid()
                ax2.legend()

        ax2.set_yticks(np.arange(0, len(modelsList)+2, 1))
        ax2.set_yticklabels(modelsList+['True lab', 'Source file'], fontsize=10 * 0.8)
        ax2.set_xlabel('Time')
        ax2.grid()
        fig1.savefig(folderOut + '/Subj' + pat + '_PredictionsTest_AllModelsComparison.png')
        plt.close(fig1)


def func_plotPredictionsOfDifferentModels_persGen(modelsList, mType, patients, folderIn, folderOut):
    ''' loads predictions in time of different models (from appended and smoothed predictions)
    and plots predictions in time to compare how different ML models perform
    also plots true label and from which file data is
    '''
    # folderOut=folderIn+'/CompPersGen_PredictionInTime/'
    createFolderIfNotExists(folderOut)

    for patIndx, pat in enumerate(patients):
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(3, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.4, hspace=0.4)
        fig1.suptitle('Subj '+ pat)

        for mIndx, mName in enumerate(modelsList):
            #trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1,yPredTest_MovAvrgStep2, yPredTest_SmoothBayes,dataSource_AllCV
            inName = folderIn + '/Approach_'+mName +'/PerformanceWithAppendedTests/Subj' + pat + '_' + mType + '_Appended_TestPredictions.csv'
            print(inName)
            data=readDataFromFile(inName)

            if (mIndx==0):
                timeRaw = np.arange(0, len(data[:,0]))*0.5
                ax1 = fig1.add_subplot(gs[0, 0])#plot true labels
                ax1.plot(timeRaw, data[:,0] , 'r')
                ax1.set_ylabel('True labels')
                ax1.set_xlabel('Time')
                # ax1.set_title('Raw data')
                ax1.grid()
                ax1 = fig1.add_subplot(gs[2, 0])  # plot file source labels
                ax1.plot(timeRaw, data[:, 6], 'k')
                ax1.set_ylabel('File source')
                ax1.set_xlabel('Time')
                # ax1.set_title('Raw data')
                ax1.grid()
                ax2 = fig1.add_subplot(gs[1, 0])

            ax2.plot(timeRaw, data[:,2] * 0.3 + mIndx, 'k', label='NoSmooth')
            ax2.plot(timeRaw,data[:,3] * 0.5 + mIndx, 'b', label='Avrg_Step1')
            ax2.plot(timeRaw,data[:,4] * 0.5 + mIndx, 'c', label='Avrg_step2')
            ax2.plot(timeRaw, data[:, 5]* 0.7 + mIndx, 'm', label='Bayes')
            if (mIndx == 0):
                ax2.legend()

        ax2.set_yticks(np.arange(0, len(modelsList), 1))
        ax2.set_yticklabels(modelsList, fontsize=10 * 0.8)
        ax2.set_xlabel('Time')
        ax2.grid()
        fig1.savefig(folderOut + '/Subj' + pat + '_PredictionsTest_AllModelsComparison_'+mType+'.png')
        plt.close(fig1)

#
# def func_plotPredictionsOfDifferentFeatureSets(foldersList, nameList, mType, patients, folderIn, dataPrepType):
#     ''' loads predictions in time of different models (from appended and smoothed predictions)
#     and plots predictions in time to compare how different ML models perform
#     also plots true label and from which file data is
#     '''
#     folderOut=folderIn+'/CompFeatureSets_PredictionInTime/'
#     createFolderIfNotExists(folderOut)
#
#     for patIndx, pat in enumerate(patients):
#         fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
#         gs = GridSpec(3, 1, figure=fig1)
#         fig1.subplots_adjust(wspace=0.4, hspace=0.4)
#         fig1.suptitle('Subj '+ pat)
#
#         for mIndx, mName in enumerate(foldersList):
#             #trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1,yPredTest_MovAvrgStep2, yPredTest_SmoothBayes,dataSource_AllCV
#             inName = folderIn + '/'+ mName+'_'+dataPrepType +'/PerformanceWithAppendedTests/Subj' + pat + '_' + mType + '_Appended_TestPredictions.csv'
#             data=readDataFromFile(inName)
#
#             if (mIndx==0):
#                 timeRaw = np.arange(0, len(data[:,0]))*0.5
#                 ax1 = fig1.add_subplot(gs[0, 0])#plot true labels
#                 ax1.plot(timeRaw, data[:,0] , 'r')
#                 ax1.set_ylabel('True labels')
#                 ax1.set_xlabel('Time')
#                 # ax1.set_title('Raw data')
#                 ax1.grid()
#                 ax1 = fig1.add_subplot(gs[2, 0])  # plot file source labels
#                 ax1.plot(timeRaw, data[:, 6], 'k')
#                 ax1.set_ylabel('File source')
#                 ax1.set_xlabel('Time')
#                 # ax1.set_title('Raw data')
#                 ax1.grid()
#                 ax2 = fig1.add_subplot(gs[1, 0])
#
#             ax2.plot(timeRaw, data[:,2] * 0.3 + mIndx, 'k', label='NoSmooth')
#             ax2.plot(timeRaw,data[:,3] * 0.5 + mIndx, 'b', label='Avrg_Step1')
#             ax2.plot(timeRaw,data[:,4] * 0.5 + mIndx, 'c', label='Avrg_step2')
#             ax2.plot(timeRaw, data[:, 5]* 0.7 + mIndx, 'm', label='Bayes')
#             if (mIndx == 0):
#                 ax2.legend()
#
#         ax2.set_yticks(np.arange(0, len(nameList), 1))
#         ax2.set_yticklabels(nameList, fontsize=10 * 0.8)
#         ax2.set_xlabel('Time')
#         ax2.grid()
#         fig1.savefig(folderOut + '/Subj' + pat + '_PredictionsTest_AllModelsComparison_'+mType+'.png')
#         plt.close(fig1)
#
def func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderOut, patients, PostprocessingParams,FeaturesParams , typeModel):
    ''' goes through predictions of each file of each subject and calculates performance
    calculates average performance of eahc subject based on average of all crossvalidations
    also calculates average of all subjects
    plots performances per subject and average of all subjects '''

    AllSubjDiffPerf_test = torch.zeros((len(patients), 4* 9))
    AllSubjDiffPerf_train = torch.zeros((len(patients), 4* 9))
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFPBefSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFPAftSeiz / FeaturesParams.winStep)
    for patIndx, pat in enumerate(patients):
        try:
            filesAll = np.sort(glob.glob(folderOut + '/*Subj' + pat + '*_'+typeModel+'_TestPredictions.csv.gz'))
        except:
            filesAll = np.sort(glob.glob(folderOut + '/*Subj' + pat + '*_' + typeModel + '_Test0Predictions.csv.gz'))
        if (len(filesAll)==0):
            filesAll = np.sort(glob.glob(folderOut + '/*Subj' + pat + '*_' + typeModel + '_Test0Predictions.csv.gz'))
        numFiles=len(filesAll)
        print('AVERAGE '+ typeModel + '--> Subj '+ pat + ' numFiles: ', numFiles)
        if numFiles == 0:
            return
        trainAvailable=0
        performanceTrain = torch.zeros(( numFiles, 4*9 ))  # 3 for noSmooth, step1, step2, and 9 or 9 perf meausres
        performanceTest = torch.zeros(( numFiles, 4*9 ))  # 3 for noSmooth, step1, step2, and 9 or 9 perf meausres
        for cv in range(len(filesAll)):
            fileName2 = 'Subj' + pat + '_cv' + str(cv).zfill(3)

            data0 = torch.from_numpy(readDataFromFile(filesAll[cv]))
            label_test = data0[:, 0]
            probabLab_test = data0[:, 1]
            predLabels_test= data0[:, 2]
            # dataSource_test= data0[:, 6]

            (performanceTest[cv, :], yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2,yPredTest_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels_test, label_test,probabLab_test,
                                                                                toleranceFP_bef, toleranceFP_aft,numLabelsPerHour,seizureStableLenToTestIndx,seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
            dataToSave = np.vstack((label_test, probabLab_test, predLabels_test, yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2, yPredTest_SmoothBayes )).transpose()  # added from which file is specific part of test set
            outputName = folderOut + '/' + fileName2 + '_'+typeModel+'_TestPredictions.csv'
            saveDataToFile(dataToSave, outputName, 'gzip')

            if (os.path.exists(filesAll[cv][0:-22] + 'TrainPredictions.csv.gz')):
                trainAvailable = 1
                data0 = torch.from_numpy(readDataFromFile(filesAll[cv][0:-22] + 'TrainPredictions.csv.gz'))
                label_train=data0[:,0]
                probabLab_train=data0[:,1]
                predLabels_train=data0[:,2]

                (performanceTrain[cv, :], yPredTrain_MovAvrgStep1, yPredTrain_MovAvrgStep2,yPredTrain_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels_train, label_train, probabLab_train,
                                                                                     toleranceFP_bef, toleranceFP_aft,numLabelsPerHour, seizureStableLenToTestIndx,seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
                dataToSave = np.vstack((label_train, probabLab_train, predLabels_train, yPredTrain_MovAvrgStep1, yPredTrain_MovAvrgStep2, yPredTrain_SmoothBayes)).transpose()
                outputName = folderOut + '/' + fileName2 + '_'+typeModel+'_TrainPredictions.csv'
                saveDataToFile(dataToSave, outputName, 'gzip')



        # saving performance for all CV of this subjet
        outputName = folderOut + '/Subj' + pat + '_'+typeModel+'_TestAllPerfMeasures.csv'  # all test sets
        saveDataToFile(performanceTest, outputName, 'gzip')
        if (trainAvailable == 1):
            outputName = folderOut + '/Subj' + pat + '_'+typeModel+'_TrainAllPerfMeasures.csv'
            saveDataToFile(performanceTrain, outputName, 'gzip')

        # # calculationg avrg for this subj over all CV
        if (trainAvailable == 1):
            AllSubjDiffPerf_train[patIndx, :] = performanceTrain.nanmean(0)
        AllSubjDiffPerf_test[patIndx, :] = performanceTest.nanmean(0)

    # SAVE PERFORMANCE MEASURES FOR ALL SUBJ
    if (trainAvailable == 1):
        outputName = folderOut + '/AllSubj_'+ typeModel+'_TrainAllPerfMeasures.csv'
        saveDataToFile(AllSubjDiffPerf_train, outputName, 'gzip')
    outputName = folderOut + '/AllSubj_'+ typeModel+'_TestAllPerfMeasures.csv'
    saveDataToFile(AllSubjDiffPerf_test, outputName, 'gzip')


    # load performance measures per subject
    fig1 = plt.figure(figsize=(10, 10), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    # fig1.suptitle('All subject different performance measures ')
    xValues = np.arange(1, len(patients)+1, 1)
    perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
                 'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
    numPerf = len(perfNames)
    for perfIndx, perf in enumerate(perfNames):
        ax1 = fig1.add_subplot(gs[int(np.floor(perfIndx / 3)), np.mod(perfIndx, 3)])
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 0 +perfIndx], 'k--')
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 9 +perfIndx], 'b--')
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 18 +perfIndx], 'c--')
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 27 +perfIndx], 'm--')
        ax1.legend(['NoSmooth', 'Step1', 'Step2', 'Bayes'])
        if (trainAvailable == 1):
            ax1.plot(xValues, AllSubjDiffPerf_train[:, 0 +perfIndx], 'k')
            ax1.plot(xValues, AllSubjDiffPerf_train[:, 9 +perfIndx], 'b')
            ax1.plot(xValues, AllSubjDiffPerf_train[:, 18 +perfIndx], 'c')
            ax1.plot(xValues, AllSubjDiffPerf_train[:, 27 +perfIndx], 'm')
        #plotting mean values
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_test[:, 0 +perfIndx]), 'k--')
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_test[:, 9 +perfIndx]), 'b--')
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_test[:, 18 +perfIndx]), 'c--')
        ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(AllSubjDiffPerf_test[:, 27 +perfIndx]), 'm--')
        if (trainAvailable == 1):
            ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_train[:, 0 +perfIndx]), 'k')
            ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_train[:, 9 +perfIndx]), 'b')
            ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_train[:, 18 +perfIndx]), 'c')
            ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(AllSubjDiffPerf_train[:, 27 +perfIndx]), 'm')
        ax1.set_xlabel('Subjects')
        ax1.set_title(perf)
        ax1.grid()
    # if (plottingON == 1):
    fig1.show()

    fig1.savefig(folderOut + '/AllSubj_AveragingCV_'+typeModel+'_AllPerformanceMeasures.png', bbox_inches='tight')
    plt.close(fig1)


#
# def plot_performanceComparison_RFvsSTDHDandONLHD(folderIn, folderOut,  subjects, type):
#     '''plot avarage performance of all subjects for random forest vs HD approachs'''
#     numSubj=len(subjects)
#
#     #load perofrmances per subj for all three approaches
#     Perf_RF = readDataFromFile(folderIn + '/AllSubj_RF_'+type+'AllPerfMeasures.csv.gz')
#     Perf_stdHD = readDataFromFile(folderIn + '/AllSubj_StdHD_'+type+'AllPerfMeasures.csv.gz')
#     Perf_onlHD = readDataFromFile(folderIn + '/AllSubj_OnlineHD_'+type+'AllPerfMeasures.csv.gz')
#
#     perfNames = ['TPR', 'PPV', 'F1']
#
#     #EPISODES
#     PerfIndxs=[18,19,20]
#     for t, tIndx in enumerate(PerfIndxs):
#         dataAppend = np.vstack((Perf_RF[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('RF', numSubj))).transpose()
#         if (t==0):
#             AllPerfAllSubj_E = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#         else:
#             AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#         dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
#         AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#         dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
#         AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#
#     #DURATION
#     PerfIndxs=[21,22,23]
#     for t, tIndx in enumerate(PerfIndxs):
#         dataAppend = np.vstack((Perf_RF[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('RF', numSubj))).transpose()
#         if (t==0):
#             AllPerfAllSubj_D = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#         else:
#             AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#         dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
#         AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#         dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
#         AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#
#     #DE combined and numFP
#     perfNames2 = ['numFP']
#     PerfIndxs=[26]
#     for t, tIndx in enumerate(PerfIndxs):
#         dataAppend = np.vstack((Perf_RF[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('RF', numSubj))).transpose()
#         if (t==0):
#             AllPerfAllSubj_T = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#         else:
#             AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#         dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
#         AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#         dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
#         AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#
#     # PLOTTING
#     AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'])
#     AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'])
#     AllPerfAllSubj_T['Performance'] = pd.to_numeric(AllPerfAllSubj_T['Performance'])
#
#     fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
#     gs = GridSpec(1,3, figure=fig1)
#     major_ticks = np.arange(0, 1, 0.1)
#     fig1.subplots_adjust(wspace=0.3, hspace=0.4)
#     ax2 = fig1.add_subplot(gs[0,0])
#     #sns.set_theme(style="whitegrid")
#     # ax2.grid(which='both')
#     # # ax2.grid(which='minor', alpha=0.2)
#     # # ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_E, palette="Set1")
#     ax2.set_title('Episode level performance')
#     ax2.legend(loc='lower left')
#     ax2.grid(which='both')
#     ax2 = fig1.add_subplot(gs[0,1])
#     sns.set_theme(style="whitegrid")
#     ax2.grid(which='both')
#     ax2.grid(which='minor', alpha=0.2)
#     ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_D, palette="Set1")
#     ax2.set_title('Duration performance')
#     ax2.legend(loc='lower left')
#     fig1.show()
#     ax2 = fig1.add_subplot(gs[0,2])
#     sns.set_theme(style="whitegrid")
#     ax2.grid(which='both')
#     ax2.grid(which='minor', alpha=0.2)
#     ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_T, palette="Set1")
#     ax2.set_title('Combined measures')
#     ax2.legend(loc='lower left')
#     fig1.show()
#     fig1.savefig(folderOut + '/RFvsHD_Allsubj_Performance_'+type+'.png', bbox_inches='tight')
#     fig1.savefig(folderOut + '/RFvsHD_Allsubj_Performance_'+type+'.svg', bbox_inches='tight')
#     plt.close(fig1)
#
#
def analyseSeizureDurations(folderIn, folderOut, patients):
    ''' loads one by one patient from raw data folder and if it is seizure file
    detects where is seizure, and plots positions and duration
    20200510 UnaPale'''
    createFolderIfNotExists(folderOut)
    seizLenFolder=folderOut +'/SeizLens/'
    createFolderIfNotExists(seizLenFolder)

    avrgLenPerSubj=np.zeros((len(patients),3))
    for patIndx, pat in enumerate(patients):
        print('-- Patient:', pat)
        PATIENT = pat if len(sys.argv) < 2 else '{0:02d}'.format(int(sys.argv[1]))
        SeizFiles=sorted(glob.glob(f'{folderIn}/chb{PATIENT}*.seizures'))
        numSeiz=0
        pairedFiles=[]

        seizLensThisSubj=[]
        for fileIndx,fileName in enumerate(SeizFiles):
            fileName0 = os.path.splitext(fileName)[0]  # removing .seizures from the string
            pom, fileName1 = os.path.split(fileName0)
            fileNameShort = os.path.splitext(fileName1)[0]
            print('FILE:', fileNameShort)
            # here replaced reading .hea files with .edf reading to avoid converting !!!
            (data, samplFreq, channels) = readEdfFile(fileName0)
            (lenSig, numCh) = data.shape
            # read times of seizures
            szStart = [a for a in MIT.read_annotations(fileName) if a.code == 32]  # start marked with '[' (32)
            szStop = [a for a in MIT.read_annotations(fileName) if a.code == 33]  # start marked with ']' (33)
            # for each seizure cut it out and save (with few parameters)
            numSeizures = len(szStart)
            for i in range(numSeizures):
                seizureLen = szStop[i].time - szStart[i].time
                numSeiz=numSeiz+1
                print('SEIZ NR ', numSeiz, ': ',szStart[i].time /256, '-', szStop[i].time /256, ' dur: ', seizureLen/256)
                pairedFiles.append('SEIZ NR '+ str(numSeiz)+': '+ str(szStart[i].time /256)+ ' - '+ str(szStop[i].time /256) + ' dur: '+  str(seizureLen/256) + ' seizFile: '+fileNameShort)

                seizLensThisSubj=np.append(seizLensThisSubj, seizureLen/256)

        avrgLenPerSubj[patIndx,: ]=[np.mean(seizLensThisSubj), np.min(seizLensThisSubj), np.max(seizLensThisSubj)]

        # save paired files
        file = open(folderOut + '/Subj' +pat+ '_SeizureInformation.txt', 'w')
        for i in range(len(pairedFiles)):
            file.write(pairedFiles[i] + '\n')
        file.close()

        #save lens for this subj only
        outputName = seizLenFolder + '/Subj'+pat+'_SeizLens.csv'
        np.savetxt(outputName, seizLensThisSubj, delimiter=",")

    # save avrg Lens All Subj
    outputName = seizLenFolder + '/AllSubj_AvrgSeizLens.csv'
    np.savetxt(outputName, avrgLenPerSubj, delimiter=",")


# def func_oversampleTrainingData(data_train_ToTrain, label_train, StandardMLParams):
#     if (StandardMLParams.trainingDataResampling=='ROS'):
#         ros = imblearn.over_sampling.RandomOverSampler (sampling_strategy=StandardMLParams.traininDataResamplingRatio, random_state=42)
#     elif  (StandardMLParams.trainingDataResampling=='SMOTE'):
#         ros = imblearn.over_sampling.SMOTE(sampling_strategy=StandardMLParams.traininDataResamplingRatio, random_state=42)
#     X_res, y_res = ros.fit_resample(data_train_ToTrain, label_train)
#
#     return(X_res, y_res)
#
# def plot_variabilityBetweenRuns_Fact110StoS(folderOut,  subjects, outputName,type):
#     '''plot avarage performance of all subjects for random forest vs HD approachs'''
#     numSubj = len(subjects)
#
#     folderBase1='../01_CHBMIT/05_Predictions_1to30Hz_4_0.5_MeanAmpl-LineLength-Frequency_'
#     dataSetLists=['Fact1', 'Fact10', 'AllDataStoS']
#     folderBase2='/personalized_MeanAmpl-LineLength-Frequency/LeaveOneOut_NoResampling/'
#     versionsList=['','_ver2','_ver3','_ver4','_ver5']
#
#     rangesPerSubj= np.zeros((numSubj, 36, len(dataSetLists)))
#     avrgRanges=np.zeros((len(dataSetLists), 36, 2)) #2 for mean and std
#     # load performances
#     for dIndx, dataSetName in enumerate(dataSetLists):
#         perfAll = np.zeros((numSubj, 36, len(versionsList)))
#         for vIndx, verName in enumerate(versionsList):
#             if (type == 'Appended'):
#                 perfAll[:, :, vIndx] = readDataFromFile(folderBase1 + dataSetName + verName + folderBase2 + '/PerformanceWithAppendedTests/' + '/AllSubj_AppendedTest_RF_AllPerfMeasures.csv.gz')
#             else:  # avrg of crossvalidations
#                 perfAll[:, :, vIndx] = readDataFromFile(folderBase1 + dataSetName + verName + folderBase2 + '/AllSubj_RF_TestAllPerfMeasures.csv.gz')
#
#
#
#         #calculate min and max
#         # ranges=np.max(perfAll,2)-np.min(perfAll,2)
#         ranges = np.std(perfAll, 2)
#         avrgRanges[dIndx, :, 0]=np.mean(ranges,0)
#         avrgRanges[dIndx, :, 1] = np.std(ranges, 0)
#
#         rangesPerSubj[:,:,dIndx]=ranges
#
#     #Plotting
#     plot_condensedPerfomanceForDifferentOptions(outputName, folderOut, rangesPerSubj, dataSetLists, subjects)
#
#     # # PLOTTING
#     # fig1 = plt.figure(figsize=(8, 8), constrained_layout=False)
#     # gs = GridSpec(3, 3, figure=fig1)
#     # fig1.subplots_adjust(wspace=0.3, hspace=0.3)
#     # # fig1.suptitle('Comparing features All Subj ')
#     # xValues = np.arange(0, len(dataSetLists) , 1)
#     # perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
#     #              'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
#     # numPerf = len(perfNames)
#     # for perfIndx, perf in enumerate(perfNames):
#     #     ax1 = fig1.add_subplot(gs[int(np.floor(perfIndx / 3)), np.mod(perfIndx, 3)])
#     #     ax1.errorbar(xValues, avrgRanges[:,0 +perfIndx,0], yerr=avrgRanges[:,0 +perfIndx,0],  fmt='k')
#     #     ax1.errorbar(xValues, avrgRanges[:, 9 + perfIndx, 0], yerr=avrgRanges[:, 9 + perfIndx, 0], fmt='b')
#     #     ax1.errorbar(xValues, avrgRanges[:, 18 + perfIndx, 0], yerr=avrgRanges[:, 18 + perfIndx, 0], fmt='c')
#     #     ax1.errorbar(xValues, avrgRanges[:, 27 + perfIndx, 0], yerr=avrgRanges[:, 27 + perfIndx, 0], fmt='m')
#     #     ax1.legend(['NoSmooth', 'Step1', 'Step2', 'Bayes'])
#     #     ax1.set_xticks(np.arange(0, len(dataSetLists), 1))
#     #     ax1.set_xticklabels(dataSetLists, fontsize=10) #, rotation=45)
#     #     ax1.set_title(perf)
#     #     ax1.grid()
#     # fig1.show()
#     # fig1.savefig(folderOut + '/'+outputName+'.png', bbox_inches='tight')
#     # fig1.savefig(folderOut + '/' + outputName + '.svg', bbox_inches='tight')
#     # plt.close(fig1)
#
# def plot_performanceComparison_AppendVsAvrg(folderIn, folderOut,  subjects, outputName, GeneralParams,PostprocessingParams, FeaturesParams):
#     '''plot avarage performance of all subjects for random forest vs HD approachs'''
#     numSubj=len(subjects)
#     folderInNames=['Arvg','Appended']
#     perfAll=np.zeros((numSubj, 36, len(folderInNames)))
#
#     # ## CALCULATE PERFORMANCE BASED ON PREDICTIONS (rerun for al subjects again)
#     func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderIn, subjects,  PostprocessingParams, FeaturesParams, 'RF')
#     # # ###MEASURE PERFORMANCE WHEN APPENDING TEST DATA and plot appended predictions in time
#     func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderIn, subjects,  PostprocessingParams, FeaturesParams, 'RF')
#
#     #load performances
#     perfAll[:,:,1]= readDataFromFile(folderIn+'/PerformanceWithAppendedTests/' + '/AllSubj_AppendedTest_RF_AllPerfMeasures.csv.gz')
#     perfAll[:,:,0] = readDataFromFile( folderIn+ '/AllSubj_RF_TestAllPerfMeasures.csv.gz')
#
#     perfNames = ['TPR', 'PPV', 'F1']
#     #EPISODES
#     PerfIndxs=[18,19,20]
#     for t, tIndx in enumerate(PerfIndxs):
#         for fIndx in range(len(folderInNames)):
#             dataAppend = np.vstack((perfAll[:, tIndx, fIndx], np.repeat(perfNames[t], numSubj), np.repeat(folderInNames[fIndx], numSubj))).transpose()
#             if (fIndx==0 and t==0):
#                 AllPerfAllSubj_E = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#     #DURATION
#     PerfIndxs=[21,22,23]
#     for t, tIndx in enumerate(PerfIndxs):
#         for fIndx in range(len(folderInNames)):
#             dataAppend = np.vstack((perfAll[:, tIndx, fIndx], np.repeat(perfNames[t], numSubj), np.repeat(folderInNames[fIndx],  numSubj))).transpose()
#             if (fIndx==0 and t==0):
#                 AllPerfAllSubj_D = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#     #DE combined and numFP
#     perfNames2 = ['numFP']
#     PerfIndxs=[26]
#     for t, tIndx in enumerate(PerfIndxs):
#         for fIndx in range(len(folderInNames)):
#             dataAppend = np.vstack((perfAll[:, tIndx, fIndx], np.repeat(perfNames2[t], numSubj), np.repeat(folderInNames[fIndx],  numSubj))).transpose()
#             if (fIndx==0 and t==0):
#                 AllPerfAllSubj_T = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#
#     # PLOTTING
#     AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'], errors='coerce')
#     AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'], errors='coerce')
#     AllPerfAllSubj_T['Performance'] = pd.to_numeric(AllPerfAllSubj_T['Performance'], errors='coerce')
#     fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
#     gs = GridSpec(1,3, figure=fig1)
#     ax2 = fig1.add_subplot(gs[0,0])
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_E, palette="Set1")
#     ax2.set_title('Episode level performance')
#     ax2.legend(loc='lower left')
#     ax2.grid(which='both')
#     ax2 = fig1.add_subplot(gs[0,1])
#     sns.set_theme(style="whitegrid")
#     ax2.grid(which='both')
#     ax2.grid(which='minor', alpha=0.2)
#     ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_D, palette="Set1")
#     ax2.set_title('Duration performance')
#     ax2.legend(loc='lower left')
#     ax2 = fig1.add_subplot(gs[0,2])
#     sns.set_theme(style="whitegrid")
#     ax2.grid(which='both')
#     ax2.grid(which='minor', alpha=0.2)
#     ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_T, palette="Set1")
#     ax2.set_title('Combined measures')
#     ax2.legend(loc='lower left')
#     # fig1.show()
#     fig1.savefig(folderOut + '/'+outputName+'.png', bbox_inches='tight')
#     fig1.savefig(folderOut + '/'+outputName+'.svg', bbox_inches='tight')
#     plt.close(fig1)
#
# def plot_performanceComparison_FromDifferentFolders(folderInArray, folderInNames,  folderOut,  subjects, outputName, type, HDtype):
#     '''plot avarage performance of all subjects for random forest vs HD approachs'''
#     numSubj=len(subjects)
#     indexOfset=27 #0 for no smooth, 9 for step1, 18 for step2, 27 for bayes
#     smoothType='bayes'
#
#     perfAll=np.zeros((numSubj, 36, len(folderInArray)))
#     #load performances
#     for fIndx in range(len(folderInArray)):
#         # ## CALCULATE PERFORMANCE BASED ON PREDICTIONS (rerun for al subjects again)
#         func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderInArray[fIndx], subjects,  PostprocessingParams, FeaturesParams, 'RF')
#
#         # # ###MEASURE PERFORMANCE WHEN APPENDING TEST DATA and plot appended predictions in time
#         func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderInArray[fIndx], subjects,  PostprocessingParams, FeaturesParams, 'RF')
#
#
#         if (type=='Appended'):
#             perfAll[:,:,fIndx]= readDataFromFile(folderInArray[fIndx]+'/PerformanceWithAppendedTests/' + '/AllSubj_AppendedTest_'+HDtype+'_AllPerfMeasures.csv.gz')
#         else: #avrg of crossvalidations
#             perfAll[:,:,fIndx] = readDataFromFile( folderInArray[fIndx] + '/AllSubj_'+HDtype+'_TestAllPerfMeasures.csv.gz')
#
#     # #load perofrmances per subj for all three approaches
#     # Perf_RF = readDataFromFile(folderIn + '/AllSubj_RF_'+type+'AllPerfMeasures.csv.gz')
#     # Perf_stdHD = readDataFromFile(folderIn + '/AllSubj_StdHD_'+type+'AllPerfMeasures.csv.gz')
#     # Perf_onlHD = readDataFromFile(folderIn + '/AllSubj_OnlineHD_'+type+'AllPerfMeasures.csv.gz')
#
#     perfNames = ['TPR', 'PPV', 'F1']
#
#     #EPISODES
#     PerfIndxs=[indexOfset+0,indexOfset+1,indexOfset+2]
#     for t, tIndx in enumerate(PerfIndxs):
#         for fIndx in range(len(folderInArray)):
#             dataAppend = np.vstack((perfAll[:, tIndx, fIndx], np.repeat(perfNames[t], numSubj), np.repeat(folderInNames[fIndx], numSubj))).transpose()
#             if (fIndx==0 and t==0):
#                 AllPerfAllSubj_E = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
#             # AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
#             # AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#
#     #DURATION
#     PerfIndxs=[indexOfset+3, indexOfset+4, indexOfset+5]
#     for t, tIndx in enumerate(PerfIndxs):
#         for fIndx in range(len(folderInArray)):
#             dataAppend = np.vstack((perfAll[:, tIndx, fIndx], np.repeat(perfNames[t], numSubj), np.repeat(folderInNames[fIndx],  numSubj))).transpose()
#             if (fIndx==0 and t==0):
#                 AllPerfAllSubj_D = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
#             # AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
#             # AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#
#     #DE combined and numFP
#     perfNames2 = ['numFP']
#     PerfIndxs=[indexOfset+8]
#     for t, tIndx in enumerate(PerfIndxs):
#         for fIndx in range(len(folderInArray)):
#             dataAppend = np.vstack((perfAll[:, tIndx, fIndx], np.repeat(perfNames2[t], numSubj), np.repeat(folderInNames[fIndx],  numSubj))).transpose()
#             if (fIndx==0 and t==0):
#                 AllPerfAllSubj_T = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
#             # AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
#             # AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#
#     # PLOTTING
#     AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'], errors='coerce')
#     AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'], errors='coerce')
#     AllPerfAllSubj_T['Performance'] = pd.to_numeric(AllPerfAllSubj_T['Performance'], errors='coerce')
#
#     fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
#     gs = GridSpec(1,3, figure=fig1)
#     major_ticks = np.arange(0, 1, 0.1)
#     # fig1.subplots_adjust(wspace=0.3, hspace=0.4)
#     ax2 = fig1.add_subplot(gs[0,0])
#     #sns.set_theme(style="whitegrid")
#     # ax2.grid(which='both')
#     # # ax2.grid(which='minor', alpha=0.2)
#     # # ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_E, palette="Set1")
#     ax2.set_title('Episode level performance')
#     ax2.legend(loc='lower left')
#     ax2.grid(which='both')
#     ax2 = fig1.add_subplot(gs[0,1])
#     sns.set_theme(style="whitegrid")
#     ax2.grid(which='both')
#     ax2.grid(which='minor', alpha=0.2)
#     ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_D, palette="Set1")
#     ax2.set_title('Duration performance')
#     ax2.legend(loc='lower left')
#     ax2 = fig1.add_subplot(gs[0,2])
#     sns.set_theme(style="whitegrid")
#     ax2.grid(which='both')
#     ax2.grid(which='minor', alpha=0.2)
#     ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_T, palette="Set1")
#     ax2.set_title('Combined measures')
#     ax2.legend(loc='lower left')
#     # fig1.show()
#     fig1.savefig(folderOut + '/'+outputName+'.png', bbox_inches='tight')
#     fig1.savefig(folderOut + '/'+outputName+'.svg', bbox_inches='tight')
#     plt.close(fig1)
#
# def plot_condensedPerfomanceForDifferentOptions(outputName, folderOut, data, groupsNames, subjects):
#     numSubj = len(subjects)
#
#     perfNames = ['TPR', 'PPV', 'F1']
#
#     #EPISODES
#     PerfIndxs=[18,19,20]
#     for t, tIndx in enumerate(PerfIndxs):
#         for fIndx in range(len(groupsNames)):
#             dataAppend = np.vstack((data[:, tIndx, fIndx], np.repeat(perfNames[t], numSubj), np.repeat(groupsNames[fIndx], numSubj))).transpose()
#             if (fIndx==0 and t==0):
#                 AllPerfAllSubj_E = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
#             # AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
#             # AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#
#     #DURATION
#     PerfIndxs=[21,22,23]
#     for t, tIndx in enumerate(PerfIndxs):
#         for fIndx in range(len(groupsNames)):
#             dataAppend = np.vstack((data[:, tIndx, fIndx], np.repeat(perfNames[t], numSubj), np.repeat(groupsNames[fIndx],  numSubj))).transpose()
#             if (fIndx==0 and t==0):
#                 AllPerfAllSubj_D = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
#             # AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
#             # AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#
#     #DE combined and numFP
#     perfNames2 = ['numFP']
#     PerfIndxs=[26]
#     for t, tIndx in enumerate(PerfIndxs):
#         for fIndx in range(len(groupsNames)):
#             dataAppend = np.vstack((data[:, tIndx, fIndx], np.repeat(perfNames2[t], numSubj), np.repeat(groupsNames[fIndx],  numSubj))).transpose()
#             if (fIndx==0 and t==0):
#                 AllPerfAllSubj_T = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
#             # AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#             # dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
#             # AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#
#     # PLOTTING
#     AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'], errors='coerce')
#     AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'], errors='coerce')
#     AllPerfAllSubj_T['Performance'] = pd.to_numeric(AllPerfAllSubj_T['Performance'], errors='coerce')
#
#     fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
#     gs = GridSpec(1,3, figure=fig1)
#     major_ticks = np.arange(0, 1, 0.1)
#     # fig1.subplots_adjust(wspace=0.3, hspace=0.4)
#     ax2 = fig1.add_subplot(gs[0,0])
#     #sns.set_theme(style="whitegrid")
#     # ax2.grid(which='both')
#     # # ax2.grid(which='minor', alpha=0.2)
#     # # ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_E, palette="Set1")
#     ax2.set_title('Episode level performance')
#     ax2.legend(loc='lower left')
#     ax2.grid(which='both')
#     ax2 = fig1.add_subplot(gs[0,1])
#     sns.set_theme(style="whitegrid")
#     ax2.grid(which='both')
#     ax2.grid(which='minor', alpha=0.2)
#     ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_D, palette="Set1")
#     ax2.set_title('Duration performance')
#     ax2.legend(loc='lower left')
#     ax2 = fig1.add_subplot(gs[0,2])
#     sns.set_theme(style="whitegrid")
#     ax2.grid(which='both')
#     ax2.grid(which='minor', alpha=0.2)
#     ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_T, palette="Set1")
#     ax2.set_title('Combined measures')
#     ax2.legend(loc='lower left')
#     # fig1.show()
#     fig1.savefig(folderOut + '/'+outputName+'.png', bbox_inches='tight')
#     fig1.savefig(folderOut + '/'+outputName+'.svg', bbox_inches='tight')
#     plt.close(fig1)
#
#
# def func_keepSubselectionOfData(dataAll, labelsAll, startIndxOfFiles, FeaturesParams):
#
#     ratio=FeaturesParams.winStep/ 0.5  #0.5 is basic step
#     lenData=int(len(labelsAll))
#     indxs=np.arange(0,lenData,ratio).astype(int)
#     labelsAllKept=labelsAll[indxs]
#     dataAllKept = dataAll[indxs,:]
#     startIndxOfFilesKept=(startIndxOfFiles/ratio).astype(int)
#
#     return (dataAllKept, labelsAllKept, startIndxOfFilesKept)
#
#
def plot_RFvsSTDHDandONLHD(folderIn,  subjects,PostprocessingParams, type, postprocessType, ppNum, appendOrNot , bayesName):
    '''plot avarage performance of all subjects for ransom forest vs baselin HD approach'''
    folderOut = f'{folderIn}/PlotsForPaper_Performance_Bthr' +bayesName + '/'
    createFolderIfNotExists(folderOut)
    numSubj=len(subjects)

    #load perofrmances per subj for all three approaches
    if (appendOrNot=='Appended'):
        Perf_RF = readDataFromFile(folderIn + '/PerformanceWithAppendedTests_Bthr'+bayesName+'/AllSubj_AppendedTest_RF_AllPerfMeasures.csv.gz')[0:numSubj, :]
        Perf_stdHD = readDataFromFile(folderIn + '/PerformanceWithAppendedTests_Bthr'+bayesName+'/AllSubj_AppendedTest_ClassicHD_AllPerfMeasures.csv.gz')[0:numSubj, :]
        Perf_onlHD = readDataFromFile(folderIn + '/PerformanceWithAppendedTests_Bthr'+bayesName+'/AllSubj_AppendedTest_OnlineHD_AllPerfMeasures.csv.gz')[0:numSubj, :]
    else:
        Perf_RF = readDataFromFile(folderIn + '/AllSubj_RF_'+type+'AllPerfMeasures.csv.gz')[0:numSubj,:]
        Perf_stdHD = readDataFromFile(folderIn + '/AllSubj_ClassicHD_'+type+'AllPerfMeasures.csv.gz')[0:numSubj,:]
        Perf_onlHD = readDataFromFile(folderIn + '/AllSubj_OnlineHD_'+type+'AllPerfMeasures.csv.gz')[0:numSubj,:]

    perfNames = ['TPR', 'PPV', 'F1']

    #EPISODES
    PerfIndxs=[ppNum,ppNum+1,ppNum+2]
    for t, tIndx in enumerate(PerfIndxs):
        dataAppend = np.vstack((Perf_RF[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('RF', numSubj))).transpose()
        if (t==0):
            AllPerfAllSubj_E = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
        else:
            AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
        AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
        AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))

    #DURATION
    PerfIndxs=[ppNum+3,ppNum+4,ppNum+5]
    for t, tIndx in enumerate(PerfIndxs):
        dataAppend = np.vstack((Perf_RF[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('RF', numSubj))).transpose()
        if (t==0):
            AllPerfAllSubj_D = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
        else:
            AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
        AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
        AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))

    #DE combined and numFP
    perfNames2 = ['numFP']
    PerfIndxs=[ppNum+8]
    for t, tIndx in enumerate(PerfIndxs):
        dataAppend = np.vstack((Perf_RF[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('RF', numSubj))).transpose()
        if (t==0):
            AllPerfAllSubj_T = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
        else:
            AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
        AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
        AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))

    # PLOTTING
    AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'], errors='coerce')
    AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'], errors='coerce')
    AllPerfAllSubj_T['Performance'] = pd.to_numeric(AllPerfAllSubj_T['Performance'], errors='coerce')

    fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = GridSpec(1,3, figure=fig1)
    major_ticks = np.arange(0, 1, 0.1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.4)
    ax2 = fig1.add_subplot(gs[0,0])
    #sns.set_theme(style="whitegrid")
    # ax2.grid(which='both')
    # # ax2.grid(which='minor', alpha=0.2)
    # # ax2.grid(which='major', alpha=0.5)
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_E, palette="Set1",showmeans=True)
    ax2.set_title('Episode level performance')
    ax2.legend(loc='lower left')
    ax2.set_ylim(0,1)
    ax2.grid(which='both')
    ax2 = fig1.add_subplot(gs[0,1])
    sns.set_theme(style="whitegrid")
    ax2.grid(which='both')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_D, palette="Set1",showmeans=True)
    ax2.set_title('Duration performance')
    ax2.legend(loc='lower left')
    ax2.set_ylim(0, 1)
    fig1.show()
    ax2 = fig1.add_subplot(gs[0,2])
    sns.set_theme(style="whitegrid")
    ax2.grid(which='both')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_T, palette="Set1",showmeans=True)
    ax2.set_title('Combined measures')
    ax2.legend(loc='lower left')
    fig1.show()
    fig1.savefig(folderOut + '/RFvsHD_Allsubj_Performance_'+type+'_'+postprocessType+'_'+appendOrNot+'.png', bbox_inches='tight')
    # fig1.savefig(folderOut + '/RFvsHD_Allsubj_Performance_'+type+'_'+postprocessType+'_'+appendOrNot+'.svg', bbox_inches='tight')
    plt.close(fig1)

# def removNanValues(data):
#     mask=~np.isnan(data)
#     filtData= [d[m] for d,m in zip( data.T, mask.T)]
#     return filtData
#
# def comparePersGenPerformance_bothModels(folderIn, subjects, suffixName, baythrName, appendedType, folderOut):
#     # comparePersGenPerformance(folderIn, 'ClassicHD',subjects, suffixName, baythrName, appendedType, folderOut)
#     # comparePersGenPerformance(folderIn, 'OnlineHD', subjects, suffixName, baythrName, appendedType, folderOut)
#
#     #compare perf of pers and gen models when one is better or other
#     comparePersGenPerformance_whenOneBetter(folderIn, 'ClassicHD',subjects, suffixName, baythrName, appendedType, folderOut)
#     comparePersGenPerformance_whenOneBetter(folderIn, 'OnlineHD', subjects, suffixName, baythrName, appendedType,folderOut)
#
#     #when optimizing on onlynon processed labels
#     comparePersGenPerformance_GenPersCompromise(folderIn, 'ClassicHD',subjects, suffixName, baythrName, appendedType, folderOut)
#     comparePersGenPerformance_GenPersCompromise(folderIn, 'OnlineHD', subjects, suffixName, baythrName, appendedType, folderOut)
#
#     #if optimizing for every smoothing type individually
#     comparePersGenPerformance_GenPersCompromise_v2(folderIn, 'ClassicHD',subjects, suffixName, baythrName, appendedType, folderOut)
#     comparePersGenPerformance_GenPersCompromise_v2(folderIn, 'OnlineHD', subjects, suffixName, baythrName, appendedType, folderOut)
#
# def comparePersGenPerformance(folderIn, mType,  subjects, suffixName, baythrName, appendedType, folderOut):
#     createFolderIfNotExists(folderOut)
#     numSubj=len(subjects)
#     perfNames = ['TPR', 'PPV', 'F1']
#     perfNamesAll = ['TPR_E', 'PPV_E', 'F1_E', 'TPR_D', 'PPV_D', 'F1_D', 'mean_DE', 'gmean_DE', 'numFPD']
#
#     if (appendedType=='Appended'):
#         Perf_HDPers = readDataFromFile(folderIn + '/Approach_personalized'+suffixName+'/PerformanceWithAppendedTests_'+baythrName+'/AllSubj_AppendedTest_'+mType+'_AllPerfMeasures.csv.gz')[0:numSubj,:]
#         Perf_HDGen = readDataFromFile(folderIn + '/Approach_generalized' + suffixName + '/PerformanceWithAppendedTests_' + baythrName + '/AllSubj_AppendedTest_' + mType + '_AllPerfMeasures.csv.gz')[0:numSubj, :]
#     else:
#         Perf_HDPers = readDataFromFile(folderIn + '/Approach_personalized'+suffixName+'/AllSubj_'+mType+'_TestAllPerfMeasures.csv.gz')[0:numSubj,:]
#         Perf_HDGen = readDataFromFile(folderIn + '/Approach_generalized' + suffixName + '/AllSubj_' + mType + '_TestAllPerfMeasures.csv.gz')[0:numSubj, :]
#
#     meanPers=np.nanmean(Perf_HDPers,0)
#     meanGen = np.nanmean(Perf_HDGen, 0)
#     diff=Perf_HDPers-Perf_HDGen
#     meanDiff=np.nanmean(diff,0)
#     indxPos=np.where(diff[:,2]>0)[0]
#     indxNeg=np.where(diff[:,2]<0)[0]
#     meanPersPos=np.mean(Perf_HDPers[indxPos,:],0)
#     meanGenPos = np.mean(Perf_HDGen[indxPos, :], 0)
#     meanPersNeg = np.mean(Perf_HDPers[indxNeg, :], 0)
#     meanGenNeg = np.mean(Perf_HDGen[indxNeg, :], 0)
#
#
#     xstep=0.15
#     boxWidth=0.1
#     perfSmoothingNames=['Raw', 'MovAvrg', 'MovAvrg+Merge', 'Bayes']
#     fig1 = plt.figure(figsize=(16, 10), constrained_layout=True)
#     gs = GridSpec(4, 4, figure=fig1)
#     fig1.subplots_adjust(wspace=0.5, hspace=0.5)
#     for pIndx, perf in enumerate(perfSmoothingNames):
#         ax2 = fig1.add_subplot(gs[pIndx, 0])
#         xarr = np.arange(1, 3 + 1, 1)
#         ax2.boxplot( removNanValues(Perf_HDPers[:,pIndx*9:pIndx*9+3]), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen[:, pIndx*9:pIndx*9 + 3]), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDPers[indxPos, pIndx * 9:pIndx * 9 + 3]), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen[indxPos, pIndx * 9:pIndx * 9 + 3]), positions=xarr + xstep * 3, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDPers[indxNeg, pIndx * 9:pIndx * 9 + 3]), positions=xarr + xstep*4, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen[indxNeg, pIndx * 9:pIndx * 9 + 3]), positions=xarr + xstep * 5, widths=boxWidth)
#         ax2.set_ylabel(perf)
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['Sens','Prec','F1'], fontsize=8, rotation=45)
#         if pIndx == 0:
#             ax2.set_title('Episode performance')
#         ax2.grid()
#         ax2 = fig1.add_subplot(gs[pIndx, 1])
#         xarr = np.arange(1, 3 + 1, 1)
#         ax2.boxplot( removNanValues(Perf_HDPers[:,pIndx*9+3:pIndx*9+6]), positions=xarr, widths=boxWidth)
#         ax2.boxplot( removNanValues(Perf_HDGen[:, pIndx*9+3:pIndx*9 + 6]), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDPers[indxPos,pIndx*9+3:pIndx*9 + 6]), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen[indxPos, pIndx*9+3:pIndx*9 + 6]), positions=xarr + xstep * 3, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDPers[indxNeg,pIndx*9+3:pIndx*9 + 6]), positions=xarr + xstep*4, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen[indxNeg, pIndx*9+3:pIndx*9 + 6]), positions=xarr + xstep * 5, widths=boxWidth)
#         # ax2.set_title(perf)
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['Sens','Prec','F1'], fontsize=8, rotation=45)
#         if pIndx == 0:
#             ax2.set_title('Duration performance')
#         ax2.grid()
#         ax2 = fig1.add_subplot(gs[pIndx, 2])
#         xarr = np.arange(1, 2 + 1, 1)
#         ax2.boxplot( removNanValues(Perf_HDPers[:,pIndx*9+6:pIndx*9+8]), positions=xarr, widths=boxWidth)
#         ax2.boxplot( removNanValues(Perf_HDGen[:, pIndx*9+6:pIndx*9 + 8]), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDPers[indxPos,pIndx * 9+6:pIndx * 9 + 8]), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen[indxPos, pIndx * 9+6:pIndx * 9 + 8]), positions=xarr + xstep * 3, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDPers[indxNeg,pIndx * 9+6:pIndx * 9 + 8]), positions=xarr + xstep*4, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen[indxNeg, pIndx * 9+6:pIndx * 9 + 8]), positions=xarr + xstep * 5, widths=boxWidth)
#         # ax2.set_title(perf)
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['Mean','Gmean'], fontsize=8, rotation=45)
#         if pIndx == 0:
#             ax2.set_title('ED performance')
#         ax2.grid()
#         ax2 = fig1.add_subplot(gs[pIndx, 3])
#         xarr = np.arange(1, 1 + 1, 1)
#         ax2.boxplot(removNanValues(np.reshape(Perf_HDPers[:,pIndx*9+8], (-1,1) )), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(Perf_HDGen[:, pIndx*9+8], (-1,1) )), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(Perf_HDPers[indxPos, pIndx * 9+8], (-1,1) )), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(Perf_HDGen[indxPos,pIndx * 9+8], (-1,1) )), positions=xarr + xstep * 3, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(Perf_HDPers[indxNeg, pIndx * 9+8], (-1,1) )), positions=xarr + xstep*4, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(Perf_HDGen[indxNeg, pIndx * 9+8], (-1,1) )), positions=xarr + xstep * 5, widths=boxWidth)
#         # ax2.set_title(perf)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['numFP'], fontsize=8, rotation=45)
#         if pIndx==0:
#             ax2.set_title('Num FP per day')
#         ax2.grid()
#     fig1.show()
#     fig1.savefig(folderOut + '/MeanPersAndGenPerformance_'+appendedType+'_'+baythrName+ '_'+mType+'_boxplots.png', bbox_inches='tight')
#     plt.close(fig1)
#
#
#
#     #CLAUCLATEING OVERAL PERFORMANCE IF DEFAULT IS PERSONALIZED ND SOMETIMES GENERALIZED
#     #perf tot whr pers when prs better then Gen and gen when gen better then Pes
#     perfFinal1=np.copy(Perf_HDPers)
#     perfFinal1[indxNeg,:]=Perf_HDGen[indxNeg,:]
#
#     #histogram of pers values
#     histPersPerf = np.histogram(Perf_HDPers[:,2], bins=20, range=(0,1))
#
#     thrs=np.arange(0.05, 0.95, 0.05)
#     perfThr=np.zeros(len(thrs))
#     for thrIndx, thr in enumerate(thrs):
#         indxPersTooBad=np.where(Perf_HDPers[:,2]<thr)
#         perfFinal2=np.copy(Perf_HDPers)
#         perfFinal2[indxPersTooBad,:]=Perf_HDGen[indxPersTooBad,:]
#         perfThr[thrIndx]=(np.nanmean(perfFinal2[:,2]))
#     thrBest=thrs[np.argmax(perfThr)]
#     indxPersTooBad = np.where(Perf_HDPers[:, 2] < thrBest)
#     perfFinal2 = np.copy(Perf_HDPers)
#     perfFinal2[indxPersTooBad, :] = Perf_HDGen[indxPersTooBad, :]
#
#     xstep=0.2
#     boxWidth=0.15
#     perfSmoothingNames=['Raw', 'MovAvrg', 'MovAvrg+Merge', 'Bayes']
#     fig1 = plt.figure(figsize=(16, 10), constrained_layout=True)
#     gs = GridSpec(4, 4, figure=fig1)
#     fig1.subplots_adjust(wspace=0.5, hspace=0.5)
#     for pIndx, perf in enumerate(perfSmoothingNames):
#         ax2 = fig1.add_subplot(gs[pIndx, 0])
#         xarr = np.arange(1, 3 + 1, 1)
#         ax2.boxplot(removNanValues(Perf_HDPers[:,pIndx*9:pIndx*9+3]), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen[:, pIndx*9:pIndx*9 + 3]), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal1[:, pIndx * 9:pIndx * 9 + 3]), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal2[:, pIndx * 9:pIndx * 9 + 3]), positions=xarr + xstep * 3, widths=boxWidth)
#         ax2.set_ylabel(perf)
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['Sens','Prec','F1'], fontsize=8, rotation=45)
#         if pIndx == 0:
#             ax2.set_title('Episode performance')
#         ax2.grid()
#         ax2 = fig1.add_subplot(gs[pIndx, 1])
#         xarr = np.arange(1, 3 + 1, 1)
#         ax2.boxplot(removNanValues( Perf_HDPers[:,pIndx*9+3:pIndx*9+6]), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues( Perf_HDGen[:, pIndx*9+3:pIndx*9 + 6]), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal1[:, pIndx * 9+3:pIndx * 9 + 6]), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal2[:, pIndx * 9+3:pIndx * 9 + 6]), positions=xarr + xstep * 3, widths=boxWidth)
#         # ax2.set_title(perf)
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['Sens','Prec','F1'], fontsize=8, rotation=45)
#         if pIndx == 0:
#             ax2.set_title('Duration performance')
#         ax2.grid()
#         ax2 = fig1.add_subplot(gs[pIndx, 2])
#         xarr = np.arange(1, 2 + 1, 1)
#         ax2.boxplot(removNanValues( Perf_HDPers[:,pIndx*9+6:pIndx*9+8]), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues( Perf_HDGen[:, pIndx*9+6:pIndx*9 + 8]), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal1[:, pIndx * 9+6:pIndx * 9 + 8]), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal2[:, pIndx * 9+6:pIndx * 9 + 8]), positions=xarr + xstep * 3, widths=boxWidth)
#         # ax2.set_title(perf)
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['Mean','Gmean'], fontsize=8, rotation=45)
#         if pIndx == 0:
#             ax2.set_title('ED performance')
#         ax2.grid()
#         ax2 = fig1.add_subplot(gs[pIndx, 3])
#         xarr = np.arange(1, 1 + 1, 1)
#         ax2.boxplot(removNanValues(np.reshape(Perf_HDPers[:,pIndx*9+8], (-1,1) )), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues( np.reshape(Perf_HDGen[:, pIndx*9+8], (-1,1) )), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(perfFinal1[:, pIndx * 9+8], (-1,1) )), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(perfFinal2[:, pIndx * 9+8], (-1,1) )), positions=xarr + xstep * 3, widths=boxWidth)
#         # ax2.set_title(perf)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['numFP'], fontsize=8, rotation=45)
#         if pIndx==0:
#             ax2.set_title('Num FP per day')
#         ax2.grid()
#     fig1.show()
#     fig1.savefig(folderOut + '/MeanPersAndGenPerformance_'+appendedType+'_'+baythrName+ '_'+mType+'_PersByDefault_boxplots.png', bbox_inches='tight')
#     plt.close(fig1)
#
#
#
#     #CLAUCLATEING OVERAL PERFORMANCE IF DEFAULT IS GENERLIZED ND SOMETIMES PRSONALIZED
#     #perf tot whr pers when prs better then Gen and gen when gen better then Pes
#     perfFinal1=np.copy(Perf_HDPers)
#     perfFinal1[indxNeg,:]=Perf_HDGen[indxNeg,:]
#
#     #histogram of pers values
#     histPersPerf = np.histogram(Perf_HDGen[:,2], bins=20, range=(0,1))
#
#     thrs=np.arange(0.05, 0.95, 0.05)
#     perfThr=np.zeros((len(thrs), 36))
#     numGenSubj = np.zeros(len(thrs))
#     for thrIndx, thr in enumerate(thrs):
#         indxGenTooBad=np.where(Perf_HDGen[:,2]<thr)
#         perfFinal2=np.copy(Perf_HDGen)
#         perfFinal2[indxGenTooBad,:]=Perf_HDPers[indxGenTooBad,:]
#         perfThr[thrIndx,:]=(np.nanmean(perfFinal2,0))
#         numGenSubj[thrIndx]=len(Perf_HDGen[:,0])- len(indxGenTooBad[0])
#     thrBest=thrs[np.argmax(perfThr[:,2])]
#     thrBest=0.2 #manually setting as optimum between num gneralized subj and performnce
#     indxGenTooBad = np.where(Perf_HDGen[:, 2] < thrBest)
#     perfFinal2 = np.copy(Perf_HDGen)
#     perfFinal2[indxGenTooBad, :] = Perf_HDPers[indxGenTooBad, :]
#
#
#
#     #plot traadof between numGen subj and perfomance
#     fig1 = plt.figure(figsize=(16, 10), constrained_layout=True)
#     gs = GridSpec(1, 2, figure=fig1)
#     fig1.subplots_adjust(wspace=0.5, hspace=0.5)
#     ax2 = fig1.add_subplot(gs[0, 0])
#     ax2.plot(thrs, numGenSubj/ len(Perf_HDGen[:,0]))
#     ax2.set_title('Percentage generalized subj')
#     ax2.set_ylim(0, 1)
#     ax2.grid()
#     ax2 = fig1.add_subplot(gs[0, 1])
#     ax2.plot(thrs, perfThr[:,2], 'k')
#     ax2.plot(thrs, perfThr[:, 11], 'b')
#     ax2.plot(thrs, perfThr[:,20], 'c')
#     ax2.plot(thrs, perfThr[:, 29], 'm')
#     ax2.legend(['Raw', 'Step1', 'Step2', 'Bayes'])
#     ax2.set_title('F1E performanc')
#     ax2.set_ylim(0, 1)
#     ax2.grid()
#     fig1.show()
#     fig1.savefig(folderOut + '/MeanPersAndGenPerformance_'+appendedType+'_'+baythrName+ '_'+mType+'_GenByDefault_ChooshingOptimalThr.png', bbox_inches='tight')
#     plt.close(fig1)
#
#     xstep=0.2
#     boxWidth=0.15
#     perfSmoothingNames=['Raw', 'MovAvrg', 'MovAvrg+Merge', 'Bayes']
#     fig1 = plt.figure(figsize=(16, 10), constrained_layout=True)
#     gs = GridSpec(4, 4, figure=fig1)
#     fig1.subplots_adjust(wspace=0.5, hspace=0.5)
#     for pIndx, perf in enumerate(perfSmoothingNames):
#         ax2 = fig1.add_subplot(gs[pIndx, 0])
#         xarr = np.arange(1, 3 + 1, 1)
#         ax2.boxplot(removNanValues( Perf_HDPers[:,pIndx*9:pIndx*9+3]), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen[:, pIndx*9:pIndx*9 + 3]), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal1[:, pIndx * 9:pIndx * 9 + 3]), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal2[:, pIndx * 9:pIndx * 9 + 3]), positions=xarr + xstep * 3, widths=boxWidth)
#         ax2.set_ylabel(perf)
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['Sens','Prec','F1'], fontsize=8, rotation=45)
#         if pIndx == 0:
#             ax2.set_title('Episode performance')
#         ax2.grid()
#         ax2 = fig1.add_subplot(gs[pIndx, 1])
#         xarr = np.arange(1, 3 + 1, 1)
#         ax2.boxplot(removNanValues( Perf_HDPers[:,pIndx*9+3:pIndx*9+6]), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues( Perf_HDGen[:, pIndx*9+3:pIndx*9 + 6]), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal1[:, pIndx * 9+3:pIndx * 9 + 6]), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal2[:, pIndx * 9+3:pIndx * 9 + 6]), positions=xarr + xstep * 3, widths=boxWidth)
#         # ax2.set_title(perf)
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['Sens','Prec','F1'], fontsize=8, rotation=45)
#         if pIndx == 0:
#             ax2.set_title('Duration performance')
#         ax2.grid()
#         ax2 = fig1.add_subplot(gs[pIndx, 2])
#         xarr = np.arange(1, 2 + 1, 1)
#         ax2.boxplot(removNanValues( Perf_HDPers[:,pIndx*9+6:pIndx*9+8]), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues( Perf_HDGen[:, pIndx*9+6:pIndx*9 + 8]), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal1[:, pIndx * 9+6:pIndx * 9 + 8]), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(perfFinal2[:, pIndx * 9+6:pIndx * 9 + 8]), positions=xarr + xstep * 3, widths=boxWidth)
#         # ax2.set_title(perf)
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['Mean','Gmean'], fontsize=8, rotation=45)
#         if pIndx == 0:
#             ax2.set_title('ED performance')
#         ax2.grid()
#         ax2 = fig1.add_subplot(gs[pIndx, 3])
#         xarr = np.arange(1, 1 + 1, 1)
#         ax2.boxplot(removNanValues(np.reshape(Perf_HDPers[:,pIndx*9+8], (-1,1) )), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(Perf_HDGen[:, pIndx*9+8], (-1,1) )), positions=xarr+xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(perfFinal1[:, pIndx * 9+8], (-1,1) )), positions=xarr + xstep*2, widths=boxWidth)
#         ax2.boxplot(removNanValues(np.reshape(perfFinal2[:, pIndx * 9+8], (-1,1) )), positions=xarr + xstep * 3, widths=boxWidth)
#         # ax2.set_title(perf)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['numFP'], fontsize=8, rotation=45)
#         if pIndx==0:
#             ax2.set_title('Num FP per day')
#         ax2.grid()
#     fig1.show()
#     fig1.savefig(folderOut + '/MeanPersAndGenPerformance_'+appendedType+'_'+baythrName+ '_'+mType+'_GenByDefault_boxplots_0.2.png', bbox_inches='tight')
#     plt.close(fig1)
#
#     #simpl diff in perf
#     meanPers=np.nanmean(Perf_HDPers,0)
#     meanGen = np.nanmean(Perf_HDGen, 0)
#     meanGenPers=np.nanmean(perfFinal2, 0)
#     diffFromPers=meanPers-meanGenPers
#     diffFromGen=meanGenPers-meanGen
#
#     # fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
#     # gs = GridSpec(3,1, figure=fig1)
#     # fig1.subplots_adjust(wspace=0.3, hspace=0.4)
#     # ax2 = fig1.add_subplot(gs[0,0])
#     # xarr=np.arange(1,len(meanPers)+1,1)
#     # ax2.plot(xarr, meanPers, 'r')
#     # ax2.plot(xarr, meanGen, 'b')
#     # ax2.set_title('Mean performance for pers/gen models')
#     # ax2.legend(['Pers', 'Gen'])
#     # ax2.set_ylim(0,1)
#     # # ax2.set_xticks(xarr)
#     # # ax2.set_xticklabels(perfNamesAll, fontsize=8, rotation=45)
#     # ax2.grid()
#     # ax2 = fig1.add_subplot(gs[1,0])
#     # xarr=np.arange(1,len(meanPers)+1,1)
#     # meanDiffToPlot=meanDiff
#     # meanDiffToPlot[8]=np.nan; meanDiffToPlot[17]=np.nan; meanDiffToPlot[26]=np.nan; meanDiffToPlot[35]=np.nan;
#     # ax2.plot(xarr, meanDiff, 'k')
#     # ax2.set_title('Mean  of diff in performance for pers/gen models')
#     # # ax2.set_xticks(xarr)
#     # # ax2.set_xticklabels(perfNamesAll, fontsize=8, rotation=45)
#     # ax2.grid()
#     # ax2 = fig1.add_subplot(gs[2,0])
#     # xarr=np.arange(1,len(meanPers)+1,1)
#     # ax2.plot(xarr, meanPersPos, 'r')
#     # ax2.plot(xarr, meanPersNeg, 'r--')
#     # ax2.plot(xarr, meanGenPos, 'b')
#     # ax2.plot(xarr, meanGenNeg, 'b--')
#     # ax2.set_title('Mean performance for pers/gen models')
#     # ax2.legend(['Pers P>G', 'Pers P<G', 'Gen P>G', 'Gen P<G'])
#     # ax2.set_ylim(0,1)
#     # fig1.show()
#     # fig1.savefig(folderOut + '/MeanPersAndGenPerformance_'+mType+'.png', bbox_inches='tight')
#     # plt.close(fig1)
#
# def comparePersGenPerformance_whenOneBetter(folderIn, mType,  subjects, suffixName, baythrName, appendedType, folderOut):
#     createFolderIfNotExists(folderOut)
#     numSubj=len(subjects)
#     perfNames = ['TPR', 'PPV', 'F1']
#     perfNamesAll = ['TPR_E', 'PPV_E', 'F1_E', 'TPR_D', 'PPV_D', 'F1_D', 'mean_DE', 'gmean_DE', 'numFPD']
#
#     if (appendedType=='Appended'):
#         Perf_HDPers = readDataFromFile(folderIn + '/Approach_personalized'+suffixName+'/PerformanceWithAppendedTests_'+baythrName+'/AllSubj_AppendedTest_'+mType+'_AllPerfMeasures.csv.gz')[0:numSubj,:]
#         Perf_HDGen = readDataFromFile(folderIn + '/Approach_generalized' + suffixName + '/PerformanceWithAppendedTests_' + baythrName + '/AllSubj_AppendedTest_' + mType + '_AllPerfMeasures.csv.gz')[0:numSubj, :]
#     else:
#         Perf_HDPers = readDataFromFile(folderIn + '/Approach_personalized'+suffixName+'/AllSubj_'+mType+'_TestAllPerfMeasures.csv.gz')[0:numSubj,:]
#         Perf_HDGen = readDataFromFile(folderIn + '/Approach_generalized' + suffixName + '/AllSubj_' + mType + '_TestAllPerfMeasures.csv.gz')[0:numSubj, :]
#
#     meanPers=np.nanmean(Perf_HDPers,0)
#     meanGen = np.nanmean(Perf_HDGen, 0)
#     diff=Perf_HDPers-Perf_HDGen
#     meanDiff=np.nanmean(diff,0)
#     indxPos=np.where(diff[:,2]>=0)[0]
#     indxNeg=np.where(diff[:,2]<0)[0]
#     # meanPersPos=np.mean(Perf_HDPers[indxPos,:],0)
#     # meanGenPos = np.mean(Perf_HDGen[indxPos, :], 0)
#     # meanPersNeg = np.mean(Perf_HDPers[indxNeg, :], 0)
#     # meanGenNeg = np.mean(Perf_HDGen[indxNeg, :], 0)
#
#
#     # xstep=0.15
#     # boxWidth=0.1
#     # perfSmoothingNames=['Raw', 'MovAvrg'] #, 'MovAvrg+Merge', 'Bayes'
#     # perfSmoothingIndx=[0,18]
#     # locIndxs=np.asarray([2,5,7])
#     # fig1 = plt.figure(figsize=(24, 6), constrained_layout=True)
#     # gs = GridSpec(1, 4, figure=fig1)
#     # fig1.subplots_adjust(wspace=0.2, hspace=0.2)
#     # for pIndx, perf in enumerate(perfSmoothingIndx):
#     #     ax2 = fig1.add_subplot(gs[0,pIndx])
#     #     xarr = np.arange(1, 3 + 1, 1)
#     #     Perf_HDPers2=Perf_HDPers[:,perf+locIndxs]
#     #     Perf_HDGen2 = Perf_HDGen[:, perf + locIndxs]
#     #     ax2.boxplot( removNanValues(Perf_HDPers2), positions=xarr, widths=boxWidth)
#     #     ax2.boxplot(removNanValues(Perf_HDGen2), positions=xarr+xstep, widths=boxWidth)
#     #     ax2.boxplot(removNanValues(Perf_HDPers2[indxPos, :]), positions=xarr + xstep*2, widths=boxWidth)
#     #     ax2.boxplot(removNanValues(Perf_HDGen2[indxPos, :]), positions=xarr + xstep * 3, widths=boxWidth)
#     #     ax2.boxplot(removNanValues(Perf_HDPers2[indxNeg, :]), positions=xarr + xstep*4, widths=boxWidth)
#     #     ax2.boxplot(removNanValues(Perf_HDGen2[indxNeg, :]), positions=xarr + xstep * 5, widths=boxWidth)
#     #     ax2.set_title(perfSmoothingNames[pIndx])
#     #     ax2.set_ylabel('Performance')
#     #     ax2.set_ylim(0, 1)
#     #     ax2.set_xticks(xarr)
#     #     ax2.set_xticklabels(['F1E','F1D','F1DE'], fontsize=8, rotation=0)
#     #     # if pIndx == 0:
#     #     #     ax2.set_title('Episode performance')
#     #     ax2.grid()
#     # fig1.show()
#     # fig1.savefig(folderOut + '/PersvsGenPerformance_'+appendedType+'_'+baythrName+ '_'+mType+'_boxplots.png', bbox_inches='tight')
#     # fig1.savefig(folderOut + '/PersvsGenPerformance_' + appendedType + '_' + baythrName + '_' + mType + '_boxplots.svg', bbox_inches='tight')
#     # plt.close(fig1)
#
#     # plot for paper
#     xstep = 0.15
#     boxWidth = 0.1
#     perfSmoothingNames = ['Raw']  # , 'MovAvrg+Merge', 'Bayes'
#     perfSmoothingIndx = [0]
#     locIndxs = np.asarray([2, 5, 7])
#     fig1 = plt.figure(figsize=(10, 4), constrained_layout=True)
#     gs = GridSpec(1, 1, figure=fig1)
#     fig1.subplots_adjust(wspace=0.2, hspace=0.2)
#     for pIndx, perf in enumerate(perfSmoothingIndx):
#         ax2 = fig1.add_subplot(gs[0, pIndx])
#         xarr = np.arange(1, 3 + 1, 1)
#         Perf_HDPers2 = Perf_HDPers[:, perf + locIndxs]
#         Perf_HDGen2 = Perf_HDGen[:, perf + locIndxs]
#         ax2.boxplot(removNanValues(Perf_HDPers2), positions=xarr, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen2), positions=xarr + xstep, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDPers2[indxPos, :]), positions=xarr + xstep * 2, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen2[indxPos, :]), positions=xarr + xstep * 3, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDPers2[indxNeg, :]), positions=xarr + xstep * 4, widths=boxWidth)
#         ax2.boxplot(removNanValues(Perf_HDGen2[indxNeg, :]), positions=xarr + xstep * 5, widths=boxWidth)
#         ax2.set_title(perfSmoothingNames[pIndx])
#         ax2.set_ylabel('Performance')
#         ax2.set_ylim(0, 1)
#         ax2.set_xticks(xarr)
#         ax2.set_xticklabels(['F1E', 'F1D', 'F1DE'], fontsize=8, rotation=0)
#         # if pIndx == 0:
#         #     ax2.set_title('Episode performance')
#         ax2.grid()
#     fig1.show()
#     fig1.savefig(folderOut + '/PersvsGenPerformance_' + appendedType + '_' + baythrName + '_' + mType + '_boxplots.png',bbox_inches='tight')
#     fig1.savefig(folderOut + '/PersvsGenPerformance_' + appendedType + '_' + baythrName + '_' + mType + '_boxplots.svg',  bbox_inches='tight')
#     plt.close(fig1)
#
#
#
# def comparePersGenPerformance_GenPersCompromise(folderIn, mType, subjects, suffixName, baythrName, appendedType, folderOut):
#     createFolderIfNotExists(folderOut)
#     numSubj = len(subjects)
#     perfNames = ['TPR', 'PPV', 'F1']
#     perfNamesAll = ['TPR_E', 'PPV_E', 'F1_E', 'TPR_D', 'PPV_D', 'F1_D', 'mean_DE', 'gmean_DE', 'numFPD']
#
#     if (appendedType == 'Appended'):
#         Perf_HDPers = readDataFromFile(folderIn + '/Approach_personalized' + suffixName + '/PerformanceWithAppendedTests_' + baythrName + '/AllSubj_AppendedTest_' + mType + '_AllPerfMeasures.csv.gz')[0:numSubj, :]
#         Perf_HDGen = readDataFromFile(folderIn + '/Approach_generalized' + suffixName + '/PerformanceWithAppendedTests_' + baythrName + '/AllSubj_AppendedTest_' + mType + '_AllPerfMeasures.csv.gz')[ 0:numSubj, :]
#     else:
#         Perf_HDPers = readDataFromFile(folderIn + '/Approach_personalized' + suffixName + '/AllSubj_' + mType + '_TestAllPerfMeasures.csv.gz')[ 0:numSubj, :]
#         Perf_HDGen = readDataFromFile(folderIn + '/Approach_generalized' + suffixName + '/AllSubj_' + mType + '_TestAllPerfMeasures.csv.gz')[0:numSubj, :]
#
#
#     # CLAUCLATEING OVERAL PERFORMANCE IF DEFAULT IS GENERLIZED ND SOMETIMES PRSONALIZED
#     # perf tot whr pers when prs better then Gen and gen when gen better then Pes
#     diff = Perf_HDPers - Perf_HDGen
#     meanDiff = np.nanmean(diff, 0)
#     indxPos = np.where(diff[:, 2] > 0)[0]
#     indxNeg = np.where(diff[:, 2] < 0)[0]
#     perfOpt = np.copy(Perf_HDPers)
#     perfOpt[indxNeg, :] = Perf_HDGen[indxNeg, :]
#     perfOpt_mean= np.nanmean(perfOpt, 0)
#     perfOpt_std = np.nanstd(perfOpt, 0)
#
#     thrs = np.arange(0.0, 1, 0.05)
#     perfThr = np.zeros((len(thrs), 36))
#     perfThr_std = np.zeros((len(thrs), 36))
#     numGenSubj = np.zeros(len(thrs))
#     for thrIndx, thr in enumerate(thrs):
#         indxGenTooBad = np.where(Perf_HDGen[:, 2] <= thr) #F1E
#         perfFinal2 = np.copy(Perf_HDGen)
#         perfFinal2[indxGenTooBad, :] = Perf_HDPers[indxGenTooBad, :]#use prs where gen too bad
#         perfThr[thrIndx, :] = (np.nanmean(perfFinal2, 0))
#         perfThr_std[thrIndx, :] = (np.nanstd(perfFinal2, 0))
#         numGenSubj[thrIndx] =( len(Perf_HDGen[:, 0]) - len(indxGenTooBad[0]))*100/len(Perf_HDGen[:, 0])
#
#     # # plot tradeof between numGen subj and perfomance
#     # fig1 = plt.figure(figsize=(20, 4), constrained_layout=True)
#     # gs = GridSpec(1, 4, figure=fig1)
#     # fig1.subplots_adjust(wspace=0.2, hspace=0.2)
#     # smoothingNames=['NoSmooth', 'MovAvrg', 'Bayes']
#     # perfIndx=[0,18,27]
#     # for perfI, p in enumerate(perfIndx):
#     #     ax2 = fig1.add_subplot(gs[0, 1+perfI])
#     #     # ax2.errorbar(numGenSubj, perfThr[:,2], yerr=perfThr_std[:,2], fmt='r')
#     #     # ax2.errorbar(numGenSubj, perfThr[:, 5], yerr=perfThr_std[:, 5], fmt='b')
#     #     # ax2.errorbar(numGenSubj, perfThr[:, 7], yerr=perfThr_std[:, 7], fmt='m')
#     #     ax2.plot(numGenSubj, perfThr[:,2+p],'r')
#     #     ax2.plot(numGenSubj, perfThr[:, 5+p], 'b')
#     #     ax2.plot(numGenSubj, perfThr[:, 7+p], 'm')
#     #     ax2.legend(['F1E', 'F1D', 'F1DE'])
#     #     ax2.set_title('Perf with increasing % of gen models')
#     #     # ax2.set_ylim(0, 1)
#     #     ax2.set_xlabel('Percentage generalized models')
#     #     ax2.set_ylabel(smoothingNames[perfI])
#     #     #plot optimal perf
#     #     # ax2.errorbar(numGenSubj, np.ones(len(numGenSubj))* perfOpt_mean[2], yerr=np.ones(len(numGenSubj))* perfOpt_std[2], fmt='r--')
#     #     # ax2.errorbar(numGenSubj, np.ones(len(numGenSubj))* perfOpt_mean[5], yerr=np.ones(len(numGenSubj))* perfOpt_std[5],  fmt='b--')
#     #     # ax2.errorbar(numGenSubj,  np.ones(len(numGenSubj))* perfOpt_mean[7], yerr=np.ones(len(numGenSubj))* perfOpt_std[7],  fmt='m--')
#     #     ax2.plot(numGenSubj, np.ones(len(numGenSubj)) * perfOpt_mean[2+p], 'r--')
#     #     ax2.plot(numGenSubj, np.ones(len(numGenSubj)) * perfOpt_mean[5+p], 'b--')
#     #     ax2.plot(numGenSubj, np.ones(len(numGenSubj)) * perfOpt_mean[7+p], 'm--')
#     #     ax2.grid()
#     # ax2 = fig1.add_subplot(gs[0, 0])
#     # ax2.plot(numGenSubj, thrs*100, 'k')
#     # ax2.legend(['Raw', 'Step1', 'Step2', 'Bayes'])
#     # ax2.set_title('Perf threshold to choose pers models')
#     # ax2.set_xlabel('Percentage generalized models')
#     # ax2.set_ylim(0, 100)
#     # ax2.grid()
#     # fig1.show()
#     # fig1.savefig( folderOut + '/GenAndPersModelsCombined_' + appendedType + '_' + baythrName + '_' + mType + '_GenByDefault_ChooshingOptimalThr.png',bbox_inches='tight')
#     # plt.close(fig1)
#
#     # Plot for paper - smaller version
#     # plot tradeof between numGen subj and perfomance
#     fig1 = plt.figure(figsize=(10, 4), constrained_layout=True)
#     gs = GridSpec(1, 3, figure=fig1)
#     fig1.subplots_adjust(wspace=0.25, hspace=0.25)
#     smoothingNames=['NoSmooth', 'MovAvrg']
#     perfIndx=[0,18]
#     for perfI, p in enumerate(perfIndx):
#         ax2 = fig1.add_subplot(gs[ 0,1+perfI])
#         ax2.plot(numGenSubj, perfThr[:,2+p],'r--')
#         ax2.plot(numGenSubj, perfThr[:, 5+p], 'b--')
#         ax2.plot(numGenSubj, perfThr[:, 7+p], 'm--')
#         ax2.legend(['F1E', 'F1D', 'F1DE'])
#         ax2.set_title('Perf with increasing % of gen models')
#         # ax2.set_ylim(0, 1)
#         ax2.set_xlabel('Percentage generalized models')
#         ax2.set_ylabel(smoothingNames[perfI])
#         #plot optimal perf
#         ax2.plot(numGenSubj, np.ones(len(numGenSubj)) * perfOpt_mean[2+p], 'r')
#         ax2.plot(numGenSubj, np.ones(len(numGenSubj)) * perfOpt_mean[5+p], 'b')
#         ax2.plot(numGenSubj, np.ones(len(numGenSubj)) * perfOpt_mean[7+p], 'm')
#         ax2.grid()
#     ax2 = fig1.add_subplot(gs[0, 0])
#     ax2.plot(numGenSubj, thrs*100, 'k')
#     ax2.legend(['Raw', 'Step1', 'Step2', 'Bayes'])
#     ax2.set_title('Perf threshold to choose pers models')
#     ax2.set_xlabel('Percentage generalized models')
#     ax2.set_ylim(0, 100)
#     ax2.grid()
#     fig1.show()
#     fig1.savefig( folderOut + '/GenAndPersModelsCombined_' + appendedType + '_' + baythrName + '_' + mType + '_GenByDefault_ChooshingOptimalThr.png',bbox_inches='tight')
#     fig1.savefig( folderOut + '/GenAndPersModelsCombined_' + appendedType + '_' + baythrName + '_' + mType + '_GenByDefault_ChooshingOptimalThr.svg',bbox_inches='tight')
#     plt.close(fig1)
#
#     #
#     # # simpl diff in perf
#     # meanPers = np.nanmean(Perf_HDPers, 0)
#     # meanGen = np.nanmean(Perf_HDGen, 0)
#     # meanGenPers = np.nanmean(perfFinal2, 0)
#     # diffFromPers = meanPers - meanGenPers
#     # diffFromGen = meanGenPers - meanGen
#
# def comparePersGenPerformance_GenPersCompromise_v2(folderIn, mType, subjects, suffixName, baythrName, appendedType, folderOut):
#     ''' difference from comparePersGenPerformance_GenPersCompromise is that here for every smoothing things are optimizd'''
#     createFolderIfNotExists(folderOut)
#     numSubj = len(subjects)
#     perfNames = ['TPR', 'PPV', 'F1']
#     perfNamesAll = ['TPR_E', 'PPV_E', 'F1_E', 'TPR_D', 'PPV_D', 'F1_D', 'mean_DE', 'gmean_DE', 'numFPD']
#
#     if (appendedType == 'Appended'):
#         Perf_HDPers = readDataFromFile(folderIn + '/Approach_personalized' + suffixName + '/PerformanceWithAppendedTests_' + baythrName + '/AllSubj_AppendedTest_' + mType + '_AllPerfMeasures.csv.gz')[0:numSubj, :]
#         Perf_HDGen = readDataFromFile(folderIn + '/Approach_generalized' + suffixName + '/PerformanceWithAppendedTests_' + baythrName + '/AllSubj_AppendedTest_' + mType + '_AllPerfMeasures.csv.gz')[ 0:numSubj, :]
#     else:
#         Perf_HDPers = readDataFromFile(folderIn + '/Approach_personalized' + suffixName + '/AllSubj_' + mType + '_TestAllPerfMeasures.csv.gz')[ 0:numSubj, :]
#         Perf_HDGen = readDataFromFile(folderIn + '/Approach_generalized' + suffixName + '/AllSubj_' + mType + '_TestAllPerfMeasures.csv.gz')[0:numSubj, :]
#
#
#     # CLAUCLATEING OVERAL PERFORMANCE IF DEFAULT IS GENERLIZED ND SOMETIMES PRSONALIZED
#
#     # plot traadof between numGen subj and perfomance
#     fig1 = plt.figure(figsize=(20, 8), constrained_layout=True)
#     gs = GridSpec(2, 3, figure=fig1)
#     fig1.subplots_adjust(wspace=0.2, hspace=0.3)
#     smoothingNames=['NoSmooth', 'MovAvrg', 'Bayes']
#     perfIndx=[0,18,27]
#     for perfI, p in enumerate(perfIndx):
#
#         # perf tot whr pers when prs better then Gen and gen when gen better then Pes
#         diff = Perf_HDPers - Perf_HDGen
#         meanDiff = np.nanmean(diff, 0)
#         indxPos = np.where(diff[:, 2+p] > 0)[0]
#         indxNeg = np.where(diff[:, 2+p] < 0)[0]
#         perfOpt = np.copy(Perf_HDPers)
#         perfOpt[indxNeg, :] = Perf_HDGen[indxNeg, :]
#         perfOpt_mean = np.nanmean(perfOpt, 0)
#         perfOpt_std = np.nanstd(perfOpt, 0)
#
#         thrs = np.arange(0.0, 1, 0.05)
#         perfThr = np.zeros((len(thrs), 36))
#         perfThr_std = np.zeros((len(thrs), 36))
#         numGenSubj = np.zeros(len(thrs))
#         for thrIndx, thr in enumerate(thrs):
#             indxGenTooBad = np.where(Perf_HDGen[:, 2+p] <= thr)  # F1E
#             perfFinal2 = np.copy(Perf_HDGen)
#             perfFinal2[indxGenTooBad, :] = Perf_HDPers[indxGenTooBad, :]  # use prs where gen too bad
#             perfThr[thrIndx, :] = (np.nanmean(perfFinal2, 0))
#             perfThr_std[thrIndx, :] = (np.nanstd(perfFinal2, 0))
#             numGenSubj[thrIndx] = (len(Perf_HDGen[:, 0]) - len(indxGenTooBad[0])) * 100 / len(Perf_HDGen[:, 0])
#
#         ax2 = fig1.add_subplot(gs[0, perfI])
#         # ax2.errorbar(numGenSubj, perfThr[:,2], yerr=perfThr_std[:,2], fmt='r')
#         # ax2.errorbar(numGenSubj, perfThr[:, 5], yerr=perfThr_std[:, 5], fmt='b')
#         # ax2.errorbar(numGenSubj, perfThr[:, 7], yerr=perfThr_std[:, 7], fmt='m')
#         ax2.plot(numGenSubj, perfThr[:,2+p],'r')
#         ax2.plot(numGenSubj, perfThr[:, 5+p], 'b')
#         ax2.plot(numGenSubj, perfThr[:, 7+p], 'm')
#         ax2.legend(['F1E', 'F1D', 'F1DE'])
#         ax2.set_title('Perf with increasing % of gen models')
#         ax2.set_ylim(0.5, 1)
#         ax2.set_xlabel('Percentage generalized models')
#         ax2.set_ylabel(smoothingNames[perfI])
#         #plot optimal perf
#         # ax2.errorbar(numGenSubj, np.ones(len(numGenSubj))* perfOpt_mean[2], yerr=np.ones(len(numGenSubj))* perfOpt_std[2], fmt='r--')
#         # ax2.errorbar(numGenSubj, np.ones(len(numGenSubj))* perfOpt_mean[5], yerr=np.ones(len(numGenSubj))* perfOpt_std[5],  fmt='b--')
#         # ax2.errorbar(numGenSubj,  np.ones(len(numGenSubj))* perfOpt_mean[7], yerr=np.ones(len(numGenSubj))* perfOpt_std[7],  fmt='m--')
#         ax2.plot(numGenSubj, np.ones(len(numGenSubj)) * perfOpt_mean[2+p], 'r--')
#         ax2.plot(numGenSubj, np.ones(len(numGenSubj)) * perfOpt_mean[5+p], 'b--')
#         ax2.plot(numGenSubj, np.ones(len(numGenSubj)) * perfOpt_mean[7+p], 'm--')
#         ax2.grid()
#
#         ax2 = fig1.add_subplot(gs[1, perfI])
#         ax2.plot(numGenSubj, thrs*100, 'k')
#         ax2.legend(['Raw', 'Step1', 'Step2', 'Bayes'])
#         ax2.set_title('Perf threshold to choose pers models')
#         ax2.set_xlabel('Percentage generalized models')
#         ax2.set_ylabel('F1E perf threshold')
#         ax2.set_ylim(0, 100)
#         ax2.grid()
#         fig1.show()
#         fig1.savefig( folderOut + '/GenAndPersModelsCombined_' + appendedType + '_' + baythrName + '_' + mType + '_GenByDefault_ChooshingOptimalThr_v2.png',bbox_inches='tight')
#         plt.close(fig1)
#
#
def plotPerfComp_PersGenModels_AllSmothingTypes(folderIn, modelsList, subjects, suffixName, baythrName, appendedType, folderOut):
    plotPerfComp_PersGenModels(folderIn, 'ClassicHD', modelsList, subjects, 'NoSmooth', 0, suffixName,  baythrName, appendedType, folderOut)
    plotPerfComp_PersGenModels(folderIn, 'ClassicHD', modelsList, subjects, 'Step1', 9, suffixName, baythrName, appendedType, folderOut)
    plotPerfComp_PersGenModels(folderIn, 'OnlineHD', modelsList,subjects, 'NoSmooth', 0, suffixName, baythrName, appendedType, folderOut)
    plotPerfComp_PersGenModels(folderIn, 'OnlineHD', modelsList, subjects, 'Step1', 9, suffixName, baythrName, appendedType, folderOut)
    plotPerfComp_PersGenModels(folderIn, 'ClassicHD', modelsList, subjects, 'Step2', 18, suffixName, baythrName, appendedType, folderOut)
    plotPerfComp_PersGenModels(folderIn, 'ClassicHD', modelsList, subjects, 'Bayes', 27, suffixName, baythrName, appendedType, folderOut)
    plotPerfComp_PersGenModels(folderIn, 'OnlineHD', modelsList, subjects, 'Step2', 18,  suffixName, baythrName, appendedType, folderOut)
    plotPerfComp_PersGenModels(folderIn, 'OnlineHD', modelsList, subjects, 'Bayes', 27, suffixName, baythrName, appendedType, folderOut)

def plotPerfComp_PersGenModels(folderIn, mType, modelList, subjects, postprocessType, ppNum, suffixName, baythrName, appendedType, folderOut):
    '''plot avarage performance of all subjects for ransom forest vs baselin HD approach'''
    # folderOut=folderIn+'/CompPersGen_PerformanceAppended'
    createFolderIfNotExists(folderOut)

    numSubj=len(subjects)
    perfNames = ['TPR', 'PPV', 'F1']

    #load perofrmances per subj for all three approaches
    if (appendedType=='Appended'):
        Perf_RF = readDataFromFile(folderIn + '/Approach_personalized'+suffixName+'/PerformanceWithAppendedTests_'+baythrName+'/AllSubj_AppendedTest_RF_AllPerfMeasures.csv.gz')[0:numSubj,:]
    else:
        Perf_RF = readDataFromFile(folderIn + '/Approach_personalized'+suffixName+'/AllSubj_RF_TestAllPerfMeasures.csv.gz')[0:numSubj,:]

    for mIndx, mName in enumerate(modelList):
        if (appendedType == 'Appended'):
            Perf_HD = readDataFromFile(folderIn + '/Approach_'+mName+suffixName+'/PerformanceWithAppendedTests_'+baythrName+'/AllSubj_AppendedTest_'+mType+'_AllPerfMeasures.csv.gz')[0:numSubj,:]
        else:
            Perf_HD = readDataFromFile(folderIn + '/Approach_'+mName + suffixName + '/AllSubj_'+mType+'_TestAllPerfMeasures.csv.gz')[0:numSubj, :]

        #EPISODES
        PerfIndxs=[ppNum,ppNum+1,ppNum+2]
        for t, tIndx in enumerate(PerfIndxs):
            dataAppend = np.vstack((Perf_RF[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('RF', numSubj))).transpose()
            if (t==0 and mIndx==0):
                AllPerfAllSubj_E = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
            else:
                AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
            dataAppend = np.vstack( (Perf_HD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat(mName, numSubj))).transpose()
            AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))

        #DURATION
        PerfIndxs=[ppNum+3,ppNum+4,ppNum+5]
        for t, tIndx in enumerate(PerfIndxs):
            dataAppend = np.vstack((Perf_RF[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('RF', numSubj))).transpose()
            if (t==0 and mIndx==0):
                AllPerfAllSubj_D = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
            else:
                AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
            dataAppend = np.vstack( (Perf_HD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat(mName, numSubj))).transpose()
            AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))


        # #DE combined and numFP
        # perfNames2 = ['numFP']
        # PerfIndxs=[ppNum+8]
        # for t, tIndx in enumerate(PerfIndxs):
        #     dataAppend = np.vstack((Perf_RF[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('RF', numSubj))).transpose()
        #     if (t==0):
        #         AllPerfAllSubj_T = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
        #     else:
        #         AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
        #     dataAppend = np.vstack( (Perf_stdHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('stdHD', numSubj))).transpose()
        #     AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
        #     dataAppend = np.vstack( (Perf_onlHD[:, tIndx], np.repeat(perfNames2[t], numSubj), np.repeat('onlHD', numSubj))).transpose()
        #     AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))

    # PLOTTING
    AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'],errors='coerce')
    AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'],errors='coerce')
    # AllPerfAllSubj_T['Performance'] = pd.to_numeric(AllPerfAllSubj_T['Performance'])

    fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = GridSpec(1,2, figure=fig1)
    major_ticks = np.arange(0, 1, 0.1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.4)
    ax2 = fig1.add_subplot(gs[0,0])
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_E, palette="Set1",showmeans=True)
    ax2.set_title('Episode level performance')
    ax2.legend(loc='lower left')
    ax2.set_ylim(0,1)
    ax2.grid(which='both')
    ax2 = fig1.add_subplot(gs[0,1])
    sns.set_theme(style="whitegrid")
    ax2.grid(which='both')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_D, palette="Set1",showmeans=True)
    ax2.set_title('Duration performance')
    ax2.legend(loc='lower left')
    ax2.set_ylim(0, 1)
    fig1.show()
    fig1.savefig(folderOut + '/AllSubj_Performance_AllModelsComparison_'+appendedType+'_'+baythrName+'_'+mType+'_'+postprocessType+'.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/AllSubj_Performance_AllModelsComparison_' + appendedType + '_' + baythrName + '_' + mType + '_' + postprocessType + '.svg', bbox_inches='tight')
    plt.close(fig1)


#
# def plotPerfComp_DiffFeatures(folderIn,  mType, featFoldNames, featNames, subjects, postprocessType, ppNum, prepType, appendOrNot):
#     '''plot avarage performance of all subjects for ransom forest vs baselin HD approach'''
#     folderOut=folderIn+'/08_CompFeatureSets_Performance'+ appendOrNot+'/'
#     createFolderIfNotExists(folderOut)
#     folderOut=folderOut+prepType+'/'
#     createFolderIfNotExists(folderOut)
#
#     numSubj=len(subjects)
#     perfNames = ['TPR', 'PPV', 'F1']
#
#     #load perofrmances per subj for all three approaches
#     for mIndx, fName in enumerate(featFoldNames):
#         if (appendOrNot=='Appended'):
#             Perf_HD = readDataFromFile(folderIn + '/'+ fName+'_'+ prepType+'/PerformanceWithAppendedTests/AllSubj_AppendedTest_'+mType+'_AllPerfMeasures.csv.gz')[0:numSubj,:]
#         else:
#             Perf_HD = readDataFromFile(folderIn + '/' + fName + '_' + prepType + '/AllSubj_' + mType + '_TestAllPerfMeasures.csv.gz')[ 0:numSubj, :]
#
#         #EPISODES
#         PerfIndxs=[ppNum,ppNum+1,ppNum+2]
#         for t, tIndx in enumerate(PerfIndxs):
#             dataAppend = np.vstack( (Perf_HD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat(featNames[mIndx], numSubj))).transpose()
#             if (t==0 and mIndx==0):
#                 AllPerfAllSubj_E = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#
#
#         #DURATION
#         PerfIndxs=[ppNum+3,ppNum+4,ppNum+5]
#         for t, tIndx in enumerate(PerfIndxs):
#             dataAppend = np.vstack( (Perf_HD[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat(featNames[mIndx], numSubj))).transpose()
#             if (t==0 and mIndx==0):
#                 AllPerfAllSubj_D = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
#             else:
#                 AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
#
#
#     # PLOTTING
#     AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'], errors='coerce')
#     AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'], errors='coerce')
#
#     fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
#     gs = GridSpec(1,2, figure=fig1)
#     major_ticks = np.arange(0, 1, 0.1)
#     fig1.subplots_adjust(wspace=0.3, hspace=0.4)
#     ax2 = fig1.add_subplot(gs[0,0])
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_E, palette="Set1",showmeans=True)
#     ax2.set_title('Episode level performance')
#     ax2.legend(loc='lower left')
#     ax2.set_ylim(0,1)
#     ax2.grid(which='both')
#     ax2 = fig1.add_subplot(gs[0,1])
#     sns.set_theme(style="whitegrid")
#     ax2.grid(which='both')
#     ax2.grid(which='minor', alpha=0.2)
#     ax2.grid(which='major', alpha=0.5)
#     sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_D, palette="Set1",showmeans=True)
#     ax2.set_title('Duration performance')
#     ax2.legend(loc='lower left')
#     ax2.set_ylim(0, 1)
#     fig1.show()
#     fig1.show()
#     fig1.savefig(folderOut + '/AllSubj_Performance_AllFeatSetsComparison_'+mType+'_'+postprocessType+'.png', bbox_inches='tight')
#     plt.close(fig1)
#
# def extractSimBetweenNeighbouringVectors(matrix):
#     numSteps=int(len(matrix)/2)
#     NS,S=[],[]
#     for i in range(0,numSteps-1):
#         if (matrix[i,i+1]!=1):
#             NS.append(matrix[i,i+1])
#     for i in range(numSteps,numSteps*2-1):
#         if (matrix[i,i+1]!=1):
#             S.append(matrix[i,i+1])
#     return (NS, S)
#
#
#
# def calcHistogramValues_v2(sig, segmentedLabels, histbins):
#     '''takes one window of signal - all ch and labels, separates seiz and nonSeiz and
#     calculates histogram of values  during seizure and non seizure '''
#     numBins=int(histbins)
#     sig2 = sig[~np.isnan(sig)]
#     sig2 = sig2[np.isfinite(sig2)]
#     # maxValFeat=np.max(sig)
#     # binBorders=np.arange(0, maxValFeat+1, (maxValFeat+1)/numBins)
#
#     # sig[sig == np.inf] = np.nan
#     indxs=np.where(segmentedLabels==0)[0]
#     nonSeiz = sig[indxs]
#     nonSeiz = nonSeiz[~np.isnan(nonSeiz)]
#     try:
#         nonSeiz_hist = np.histogram(nonSeiz, bins=numBins, range=(np.min(sig2), np.max(sig2)))
#     except:
#         print('Error with hist ')
#
#     indxs = np.where(segmentedLabels == 1)[0]
#     Seiz = sig[indxs]
#     Seiz = Seiz[~np.isnan(Seiz)]
#     try:
#         Seiz_hist = np.histogram(Seiz, bins=numBins, range=(np.min(sig2), np.max(sig2)))
#     except:
#         print('Error with hist ')
#
#     # normalizing values that are in percentage of total samples - to not be dependand on number of samples
#     nonSeiz_histNorm=[]
#     nonSeiz_histNorm.append(nonSeiz_hist[0]/len(nonSeiz))
#     nonSeiz_histNorm.append(nonSeiz_hist[1])
#     Seiz_histNorm=[]
#     Seiz_histNorm.append(Seiz_hist[0]/len(Seiz))
#     Seiz_histNorm.append(Seiz_hist[1])
#     # Seiz_hist[0] = Seiz_hist[0] / len(Seiz_allCh)
#     return( Seiz_histNorm, nonSeiz_histNorm)
#
#
def calcHistogramValues(sig, histbins, minVal, maxVal):
    '''takes one window of signal - all ch and labels, separates seiz and nonSeiz and
    calculates histogram of values  during seizure and non seizure '''
    numBins=int(histbins)
    sig2 = sig[~np.isnan(sig)]
    sig2 = sig2[np.isfinite(sig2)]

    try:
        # sig_hist = np.histogram(sig2, bins=numBins, range=(np.min(sig2), np.max(sig2)))
        sig_hist = np.histogram(sig2, bins=numBins, range=(minVal, maxVal))
    except:
        print('Error with hist ')

    # normalizing values that are in percentage of total samples - to not be dependand on number of samples
    sig_histNorm=[]
    sig_histNorm.append(sig_hist[0]/len(sig2))
    sig_histNorm.append(sig_hist[1])

    return( sig_histNorm)

# def kl_divergence(p,q):
#     delta=0.000001
#     deltaArr=np.ones(len(p))*delta
#     p=p+deltaArr
#     q=q+deltaArr
#     res=sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))
#     return res
#
# def js_divergence(p,q):
#     m=0.5* (p+q)
#     res=0.5* kl_divergence(p,m) +0.5* kl_divergence(q,m)
#     return (res)
#
#
# def calculateFeatDivergenceBetweenSeizures( data, seizlabels, histbins, minVal, maxVal):
#
#     numFeat=len(data[0][0,:])
#     numSeiz=len(seizlabels)
#     #
#     # fig = plt.figure(figsize=(10,6), constrained_layout=False)
#     # gs = GridSpec(1, 1, figure=fig)
#     # ax = fig.add_subplot(gs[0, 0])
#
#     JSdiverg=np.ones((numSeiz, numSeiz, numFeat))*np.nan
#     for f in range(numFeat):
#         for s1 in range(numSeiz):
#             for s2 in range(s1+1, numSeiz):
#
#                 histS1=calcHistogramValues(data[s1][:,f], histbins, minVal, maxVal )
#                 histS2 = calcHistogramValues(data[s2][:, f], histbins, minVal, maxVal)
#                 JSdiverg[s1, s2, f] = js_divergence(histS1[0], histS2[0])
#                 JSdiverg[s2, s1, f] =JSdiverg[s1, s2, f]
#     #
#     #
#     #     histS1 = calcHistogramValues(data[s1][:, f], histbins, minVal, maxVal)
#     #
#     # ax.plot(histNS[0], 'k')
#     # ax.plot(histS[0], 'r')
#     # ax.legend([ 'NS', 'S'])
#     # ax.grid()
#     # ax.set_title('Subj'+pat +'- JSdiver: '+str(JSdiverg))
#     # plt.tight_layout()
#     # plt.savefig(folderOut+'/Subj'+pat+'_SvsNShistogram_'+addName+'.png', dpi=100)
#     # plt.close()
#
#     return (JSdiverg)
#
# def calculateFeatDivergence_BetweenSubj(folderOut,  seizHist, nonSeizHist, histbins, minVal, maxVal):
#
#     numSubj=len(seizHist[0,0,:])
#     numFeat=len(seizHist[:,0,0])
#
#     JSdiverg_SS=np.ones((numSubj, numSubj, numFeat))*np.nan
#     JSdiverg_NSNS = np.ones((numSubj, numSubj, numFeat)) * np.nan
#     JSdiverg_SNS = np.ones((numSubj, numSubj, numFeat)) * np.nan
#     for s1 in range(numSubj):
#         for s2 in range(s1+1, numSubj):
#             for f in range(numFeat):
#                 JSdiverg_SS[s1, s2, f] = js_divergence(seizHist[f,:,s1], seizHist[f,:, s2])
#                 JSdiverg_SS[s2, s1,f] =JSdiverg_SS[s1, s2, f]
#                 JSdiverg_NSNS[s1, s2, f] = js_divergence(nonSeizHist[f, :,s1], nonSeizHist[ f,:, s2])
#                 JSdiverg_NSNS[s2, s1, f] = JSdiverg_NSNS[s1, s2, f]
#                 JSdiverg_SNS[s1, s2, f] = js_divergence(seizHist[f,:,s1], nonSeizHist[f,:, s2])
#                 JSdiverg_SNS[s2, s1, f] = js_divergence(seizHist[f,:, s2], nonSeizHist[f,:,s1])
#
#     JSdiverg_SS_mean=np.nanmean(JSdiverg_SS,2)
#     JSdiverg_NSNS_mean = np.nanmean(JSdiverg_NSNS, 2)
#     JSdiverg_SNS_mean = np.nanmean(JSdiverg_SNS, 2)
#     fig = plt.figure(figsize=(10,6), constrained_layout=False)
#     gs = GridSpec(1, 3, figure=fig)
#     fig.suptitle('All Subj')
#     ax = fig.add_subplot(gs[0, 0])
#     pcm = ax.imshow(1- JSdiverg_SS_mean,  vmin=0, vmax=1)
#     ax.set_title(' 1- JS divergence for S')
#     fig.colorbar(pcm, ax=ax, orientation=("horizontal"))
#     ax = fig.add_subplot(gs[0, 1])
#     pcm = ax.imshow(1- JSdiverg_NSNS_mean,  vmin=0, vmax=1)
#     ax.set_title(' 1- JS divergence for NS')
#     fig.colorbar(pcm, ax=ax, orientation=("horizontal"))
#     ax = fig.add_subplot(gs[0, 2])
#     pcm = ax.imshow(1- JSdiverg_SNS_mean,  vmin=0, vmax=1)
#     ax.set_title(' 1- JS divergence for S-NS')
#     fig.colorbar(pcm, ax=ax, orientation=("horizontal"))
#     plt.tight_layout()
#     plt.savefig(folderOut+'/AllSubj_JSsimilarity_betweenPatients.png', dpi=100)
#     plt.close()
#
#
#
#
# def plotHistogramSeizVsNonSeiz(folderOut, pat,  addName, dataSeiz, dataNonSeiz, histbins, minVal, maxVal):
#     numFeat=len(dataSeiz[0,:])
#     histS=np.zeros((numFeat, histbins))
#     histNS = np.zeros((numFeat, histbins))
#     JSdiverg = np.zeros((numFeat))
#     for f in range(numFeat):
#         histS[f,:] = calcHistogramValues(dataSeiz[:,f], histbins, minVal, maxVal)[0]
#         histNS[f,:]  = calcHistogramValues(dataNonSeiz[:,f], histbins, minVal, maxVal)[0]
#         JSdiverg[f] = js_divergence(histS[f,:], histNS[f,:])
#
#     #plot divergence
#     fig = plt.figure(figsize=(10,6), constrained_layout=False)
#     gs = GridSpec(2, 1, figure=fig)
#     ax = fig.add_subplot(gs[0, 0])
#     for f in range (numFeat):
#         ax.plot(histNS[f,:], 'k')
#         ax.plot(histS[f,:], 'r')
#         if f==0:
#             ax.legend([ 'NS', 'S'])
#     ax.grid()
#     ax.set_title('Subj'+pat +'- JSdiver: '+str(np.mean(JSdiverg)))
#     ax = fig.add_subplot(gs[1, 0])
#     ax.plot(np.nanmean(histNS,0), 'k')
#     ax.plot(np.nanmean(histS,0), 'r')
#     ax.legend(['NS', 'S'])
#     ax.grid()
#     plt.tight_layout()
#     plt.savefig(folderOut+'/Subj'+pat+'_SvsNShistogram_'+addName+'.png', dpi=100)
#     plt.close()
#
#     return (histS, histNS)
#
#
# def func_plotJSsim_corrWith_HDvecSim(JSSimilarityPerSubj, sim_std, sim_onl, folderSimilarities, pat):
#     np.fill_diagonal(sim_std,np.nan)
#     np.fill_diagonal(sim_onl, np.nan)
#     JSsimReshaped=np.reshape(JSSimilarityPerSubj,(1,-1)).squeeze()
#     STDsimReshaped = np.reshape(sim_std, (1, -1)).squeeze()
#     ONLsimReshaped = np.reshape(sim_onl, (1, -1)).squeeze()
#     JSsimReshaped= JSsimReshaped[~np.isnan(JSsimReshaped)]
#     STDsimReshaped = STDsimReshaped[~np.isnan(STDsimReshaped)]
#     ONLsimReshaped = ONLsimReshaped[~np.isnan(ONLsimReshaped)]
#
#     #measure correlation
#     corrPears_STDHD,_=scipy.stats.pearsonr( JSsimReshaped, STDsimReshaped)
#     corrPears_ONLHD, _ = scipy.stats.pearsonr(JSsimReshaped, ONLsimReshaped)
#     corrSpear_STDHD,_=scipy.stats.spearmanr( JSsimReshaped, STDsimReshaped)
#     corrSpear_ONLHD, _ = scipy.stats.spearmanr(JSsimReshaped, ONLsimReshaped)
#
#     #plot divergence
#     fig = plt.figure(figsize=(10,8), constrained_layout=False)
#     gs = GridSpec(1, 2, figure=fig)
#     fig.suptitle('Subj'+pat)
#     ax = fig.add_subplot(gs[0, 0])
#     ax.scatter(JSsimReshaped, STDsimReshaped, marker='x')
#     ax.set_title(' Correlation STD HD ='  +str(corrPears_STDHD))
#     ax.set_xlabel('JS similarity')
#     ax.set_ylabel ('HD vectors similarity')
#     ax.grid()
#     ax = fig.add_subplot(gs[0, 1])
#     ax.scatter(JSsimReshaped, ONLsimReshaped, marker='x')
#     ax.set_title(' Correlation ONL HD ='  +str(corrPears_ONLHD))
#     ax.set_xlabel('JS similarity')
#     ax.set_ylabel ('HD vectors similarity')
#     ax.grid()
#     plt.tight_layout()
#     plt.savefig(folderSimilarities+'/Subj'+pat+'_Correlation_JSvsHDsim.png', dpi=100)
#     plt.close()
#
#     return(JSsimReshaped,STDsimReshaped ,ONLsimReshaped)
#
# def plotFinalCorrelation( JSsimReshaped,STDsimReshaped ,ONLsimReshaped, folderSimilarities):
#
#     # measure correlation
#     corrPears_STDHD, _ = scipy.stats.pearsonr(JSsimReshaped, STDsimReshaped)
#     corrPears_ONLHD, _ = scipy.stats.pearsonr(JSsimReshaped, ONLsimReshaped)
#     corrSpear_STDHD, _ = scipy.stats.spearmanr(JSsimReshaped, STDsimReshaped)
#     corrSpear_ONLHD, _ = scipy.stats.spearmanr(JSsimReshaped, ONLsimReshaped)
#
#     # plot divergence
#     fig = plt.figure(figsize=(10, 8), constrained_layout=False)
#     gs = GridSpec(1, 2, figure=fig)
#     fig.suptitle('All Subj')
#     ax = fig.add_subplot(gs[0, 0])
#     ax.scatter(JSsimReshaped, STDsimReshaped, marker='x')
#     ax.set_title(' Correlation STD HD =' + str(corrPears_STDHD))
#     ax.set_xlabel('JS similarity')
#     ax.set_ylabel('HD vectors similarity')
#     ax.grid()
#     ax = fig.add_subplot(gs[0, 1])
#     ax.scatter(JSsimReshaped, ONLsimReshaped, marker='x')
#     ax.set_title(' Correlation ONL HD =' + str(corrPears_ONLHD))
#     ax.set_xlabel('JS similarity')
#     ax.set_ylabel('HD vectors similarity')
#     ax.grid()
#     plt.tight_layout()
#     plt.savefig(folderSimilarities + '/AllSubj_Correlation_JSvsHDsim.png', dpi=100)
#     plt.close()
#
def softstep(x):
    if (torch.is_tensor(x)):
        return (torch.tanh(5 * (x - 1)) + 1) / 2 + (torch.tanh(5 * (x + 1)) - 1) / 2
    else:
        return (np.tanh(5 * (x - 1)) + 1) / 2 + (np.tanh(5 * (x + 1)) - 1) / 2

def step(x):
    if (torch.is_tensor(x)):
        return (torch.sign((x - 1)) + 1) / 2 + (torch.sign((x + 1)) - 1) / 2
    else:
        return (np.sign((x - 1)) + 1) / 2 + (np.sign((x + 1)) - 1) / 2

def sigmoid (x):
    if (torch.is_tensor(x)):
        return 1/(1+ torch.exp(-x))
    else:
        return 1/(1+ np.exp(-x))

def softabs(x, steepness=10):
    return sigmoid(steepness * (x - 0.5)) + sigmoid(steepness * (-x - 0.5))

def scaledexp(x, s=1.0):
    if (torch.is_tensor(x)):
        return torch.exp(x * s)
    else:
        return np.exp(x*s)

def softrelu(x, steepness=10):
    return sigmoid(steepness * (x - 0.5))

def TanhX(x, steepness=10):
    if (torch.is_tensor(x)):
        return torch.tanh(steepness * x)
    else:
        return np.tanh(steepness*x)

def visualizeVariousActivationFunctions(folderOut):
    x=np.arange(0,1,0.02)

    fig = plt.figure(figsize=(10, 8), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig)
    fig.suptitle('Different activation functions')
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, softstep(x))
    ax.set_title(' softstep ')
    ax.grid()
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, step(x))
    ax.set_title(' step ')
    ax.grid()
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(x, softabs(x))
    ax.set_title(' softabs ')
    ax.grid()
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, scaledexp(x))
    ax.set_title(' scaledexp ')
    ax.grid()
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, softrelu(x, 20))
    ax.set_title(' softrelu ')
    ax.grid()
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, TanhX(x))
    ax.set_title(' TanhX ')
    ax.grid()
    plt.tight_layout()
    plt.savefig(folderOut +'/VariousActivationFunctions.png', dpi=100)
    plt.close()




if __name__ == "__main__":
    #
    # pool = mp.Pool(processes =24)
    pass
