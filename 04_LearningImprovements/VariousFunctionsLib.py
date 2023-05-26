''' library including various functions for HD project but not necessarily related to HD vectors'''
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

import os
import glob
import csv
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

def createFolderIfNotExists(folderOut):
    ''' creates folder if doesnt already exist
    warns if creation failed '''
    if not os.path.exists(folderOut):
        try:
            os.mkdir(folderOut)
        except OSError:
            print("Creation of the directory %s failed" % folderOut)
        # else:
        #     print("Successfully created the directory %s " % folderOut)
#
def calculateMeanAmplitudeFeatures(X,  SegSymbParams, SigInfoParams):
    (lenSig, N_channels) = X.shape
    segLenIndx = int(SegSymbParams.segLenSec * SigInfoParams.samplFreq)  # length of EEG segments in samples
    slidWindStepIndx = int(
        SegSymbParams.slidWindStepSec * SigInfoParams.samplFreq)  # step of slidin window to extract segments in samples
    index = np.arange(0, lenSig - segLenIndx, slidWindStepIndx).astype(int)

    featureValues = np.zeros((len(index), len(SigInfoParams.chToKeep)))
    for i in range(len(index)):
        sig = X[index[i]:index[i] + segLenIndx, :]
        for ch in range(len(SigInfoParams.chToKeep)):
            featureValues[i, ch ] = np.mean(np.abs(sig[:, ch]))

    return (featureValues)

def calculateMLfeatures(X,HDParams, SegSymbParams ,SigInfoParams):
    (lenSig, N_channels) = X.shape
    segLenIndx = int(SegSymbParams.segLenSec * SigInfoParams.samplFreq)  # length of EEG segments in samples
    slidWindStepIndx = int(
        SegSymbParams.slidWindStepSec * SigInfoParams.samplFreq)  # step of slidin window to extract segments in samples
    index = np.arange(0, lenSig - segLenIndx, slidWindStepIndx).astype(int)

    featureValues = np.zeros((len(index), HDParams.numFeat * len(SigInfoParams.chToKeep)))
    for i in range(len(index)):
        sig = X[index[i]:index[i] + segLenIndx, :]
        for ch in range(len(SigInfoParams.chToKeep)):
            if (HDParams.numFeat == 45):
                featureValues[i,ch * HDParams.numFeat:(ch + 1) * HDParams.numFeat] = calculateMLfeatures_oneDataWindow(sig[:, ch],  SigInfoParams.samplFreq)
            else:  # 46 feat
                featureValues[i, (ch + 1) * HDParams.numFeat - 1] = np.mean(np.abs(sig[:, ch]))
                featureValues[i,ch * HDParams.numFeat:(ch + 1) * HDParams.numFeat - 1] = calculateMLfeatures_oneDataWindow(
                    sig[:, ch], SigInfoParams.samplFreq)

    return(featureValues)

def calculateMLfeatures_oneDataWindow(data,  samplFreq):
    ''' function that calculates various features relevant for epileptic seizure detection
    from paper: D. Sopic, A. Aminifar, and D. Atienza, e-Glass: A Wearable System for Real-Time Detection of Epileptic Seizures, 2018
    but uses only features that are (can be) normalized
    '''
    #some parameters
    DWTfilterName = 'db4'  # 'sym5'
    DWTlevel = 7
    n1 = 2  #num dimensions for sample entropy
    r1 = 0.2 # num of STD for sample entropy
    r2 = 0.35 # num of STD for sample entropy
    a = 2 # param for shannon, renyi and tsallis enropy
    q = 2 # param for shannon, renyi and tsallis enropy

    #DWT
    coeffs = pywt.wavedec(data, DWTfilterName, level=DWTlevel)
    a7, d7, d6, d5, d4, d3, d2, d1= coeffs

    #sample entropy
    samp_1_d7_1 = sampen2(n1, r1 * np.std(d7), d7)
    samp_1_d6_1 = sampen2(n1, r1 * np.std(d6), d6)
    samp_2_d7_1 = sampen2(n1, r2 * np.std(d7), d7)
    samp_2_d6_1 = sampen2(n1, r2 * np.std(d6), d6)

    #permutation entropy
    perm_d7_3 = perm_entropy(d7, order=3, delay=1, normalize=True)  # normalize=True instead of false as in paper
    perm_d7_5 = perm_entropy(d7, order=5, delay=1, normalize=True)
    perm_d7_7 = perm_entropy(d7, order=7, delay=1, normalize=True)
    perm_d6_3 = perm_entropy(d6, order=3, delay=1, normalize=True)
    perm_d6_5 = perm_entropy(d6, order=5, delay=1, normalize=True)
    perm_d6_7 = perm_entropy(d6, order=7, delay=1, normalize=True)
    perm_d5_3 = perm_entropy(d5, order=3, delay=1, normalize=True)
    perm_d5_5 = perm_entropy(d5, order=5, delay=1, normalize=True)
    perm_d5_7 = perm_entropy(d5, order=7, delay=1, normalize=True)
    perm_d4_3 = perm_entropy(d4, order=3, delay=1, normalize=True)
    perm_d4_5 = perm_entropy(d4, order=5, delay=1, normalize=True)
    perm_d4_7 = perm_entropy(d4, order=7, delay=1, normalize=True)
    perm_d3_3 = perm_entropy(d3, order=3, delay=1, normalize=True)
    perm_d3_5 = perm_entropy(d3, order=5, delay=1, normalize=True)
    perm_d3_7 = perm_entropy(d3, order=7, delay=1, normalize=True)

    #shannon renyi and tsallis entropy
    (shannon_en_sig, renyi_en_sig, tsallis_en_sig) = sh_ren_ts_entropy(data, a, q)
    (shannon_en_d7, renyi_en_d7, tsallis_en_d7)  = sh_ren_ts_entropy(d7, a, q)
    (shannon_en_d6, renyi_en_d6, tsallis_en_d6)  = sh_ren_ts_entropy(d6, a, q)
    (shannon_en_d5, renyi_en_d5, tsallis_en_d5)  = sh_ren_ts_entropy(d5, a, q)
    (shannon_en_d4, renyi_en_d4, tsallis_en_d4)  = sh_ren_ts_entropy(d4, a, q)
    (shannon_en_d3, renyi_en_d3, tsallis_en_d3)  = sh_ren_ts_entropy(d3, a, q)

    #band power
    p_tot = bandpower(data, samplFreq, 0,  45)
    p_dc = bandpower(data, samplFreq, 0, 0.5)
    p_mov = bandpower(data, samplFreq, 0.1, 0.5)
    p_delta = bandpower(data, samplFreq, 0.5, 4)
    p_theta = bandpower(data, samplFreq, 4, 8)
    p_alfa = bandpower(data, samplFreq, 8, 13)
    p_middle = bandpower(data, samplFreq, 12, 13)
    p_beta = bandpower(data, samplFreq, 13, 30)
    p_gamma = bandpower(data, samplFreq, 30, 45)
    p_dc_rel = p_dc / p_tot
    p_mov_rel = p_mov / p_tot
    p_delta_rel = p_delta / p_tot
    p_theta_rel = p_theta / p_tot
    p_alfa_rel = p_alfa / p_tot
    p_middle_rel = p_middle / p_tot
    p_beta_rel = p_beta / p_tot
    p_gamma_real = p_gamma / p_tot

    featuresAll= [samp_1_d7_1, samp_1_d6_1, samp_2_d7_1, samp_2_d6_1, perm_d7_3, perm_d7_5, perm_d7_7, perm_d6_3, perm_d6_5, perm_d6_7,   perm_d5_3, perm_d5_5, \
             perm_d5_7, perm_d4_3, perm_d4_5, perm_d4_7, perm_d3_3, perm_d3_5, perm_d3_7, shannon_en_sig, renyi_en_sig, tsallis_en_sig, shannon_en_d7, renyi_en_d7, tsallis_en_d7, \
             shannon_en_d6, renyi_en_d6, tsallis_en_d6, shannon_en_d5, renyi_en_d5, tsallis_en_d5, shannon_en_d4, renyi_en_d4, tsallis_en_d4, shannon_en_d3, renyi_en_d3, tsallis_en_d3, \
             p_dc_rel, p_mov_rel, p_delta_rel, p_theta_rel, p_alfa_rel, p_middle_rel, p_beta_rel, p_gamma_real]
    return (featuresAll)

def sh_ren_ts_entropy(x, a, q):
    ''' function that calculates three different entropy meausres from given widow:
    shannon, renyi and tsallis entropy'''
    p, bin_edges = np.histogram(x)
    p = p/ np.sum(p)
    p=p[np.where(p >0)] # to exclude log(0)
    shannon_en = - np.sum(p* np.log2(p))
    renyi_en = np.log2(np.sum(pow(p,a))) / (1 - a)
    tsallis_en = (1 - np.sum(pow(p,q))) / (q - 1)
    return (shannon_en, renyi_en, tsallis_en)

def bandpower(x, fs, fmin, fmax):
    '''function that calculates energy of specific frequency band of FFT spectrum'''
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f >fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max+1], f[ind_min: ind_max+1])

def sampen2(dim,r,data):
    ''' function that calculates sample entropy from given window of data'''
    epsilon = 0.001
    N = len(data)
    correl = np.zeros( 2)
    dataMat = np.zeros((dim + 1, N - dim))
    for i in range(dim+1):
        dataMat[i,:]= data[i: N - dim + i]

    for m in range(dim,dim + 2):
        count = np.zeros( N - dim)
        tempMat = dataMat[0:m,:]

        for i in range(N - m):
            #calculate distance, excluding self - matching case
            dist = np.max(np.abs(tempMat[:, i + 1: N - dim] - np.tile(tempMat[:, i],( (N - dim - i-1),1)).T  ), axis=0)
            D = (dist < r)
            count[i] = np.sum(D) / (N - dim - 1)

        correl[m - dim] = np.sum(count) / (N - dim)

    saen = np.log((correl[0] + epsilon) / (correl[1] + epsilon))
    return saen

def func_calculateFFT(sig, SegSymbParams, SigInfoParams,freqRange):
    ''' function that calculates FFT spectrum of given data window'''
    #calculate num freq points
    lenFreqsTot = int(SigInfoParams.samplFreq * SegSymbParams.segLenSec / 2)
    lenFreqs = int(freqRange * lenFreqsTot / (SigInfoParams.samplFreq / 2))

    #calculate fft
    fftTransf=np.fft.fft(sig)/len(sig)
    fftTransf2 = fftTransf[range(int(len(sig) / 2))]

    #abs
    fftTransf2=abs(fftTransf2)

    #normalize for max value to be 1
    fftTransf2_nm=fftTransf2/np.max(fftTransf2)
    #normalize so that sum is 1 - less sesnsitive to outliers
    fftTransf2_ns=fftTransf2/np.sum(fftTransf2)

    #keep only freq of interest
    #fftTransf2=fftTransf[range(int(len(sig)/2))]
    fftTransf2_nm = fftTransf2_nm[range(lenFreqs)]
    fftTransf2_ns = fftTransf2_ns[range(lenFreqs)]

    #frequencies
    timeLen=len(sig)/SigInfoParams.samplFreq
    freqs=np.arange(lenFreqs)/timeLen

    #replace nan values
    fftTransf2_nm[np.isnan(fftTransf2_nm)]=0
    fftTransf2_ns[np.isnan(fftTransf2_ns)] = 0
    return (fftTransf2_nm, fftTransf2_ns, freqs)


def segmentLabels(labels, SegSymbParams, SigInfoParams):
    ''' given the list of true labels in original data performes segmenation on the same window as data that is used for ML
    so that we have true  and predicted labels for the same window
    it basically segments true original labels to the same segments and then votes on label for each segment'''
    lenSig= len(labels)
    # numFreqBands = len(EEGfreqBands.stopFreq)
    segLenIndx = int(SegSymbParams.segLenSec * SigInfoParams.samplFreq)  # length of EEG segments in samples
    slidWindStepIndx = int(SegSymbParams.slidWindStepSec * SigInfoParams.samplFreq) # step of slidin window to extract segments in samples
    numSeg = math.floor((lenSig - segLenIndx - 1) / slidWindStepIndx)
    index = np.arange(0, lenSig - segLenIndx, slidWindStepIndx)

    # segmentedLabelsPerSample = np.zeros((numSeg, segLenIndx))
    # for i in range(numSeg):
    #     segmentedLabelsPerSample[i, :] = labels[i * slidWindStepIndx:i * slidWindStepIndx + segLenIndx]
    segmentedLabelsPerSample = np.zeros((len(index), segLenIndx))
    for i in range(len(index)): #-1
        segmentedLabelsPerSample[i, :] = labels[index[i]:index[i] + segLenIndx]
        # calculating one label per segment
    segmentedLabels = calculateLabelPerSegment(segmentedLabelsPerSample, SegSymbParams.labelVotingType)
    return segmentedLabels

def calculateLabelPerSegment(labelsPerSample, type):
    """ calculate one label from labels of all samples in a segment
    assumes labels are only 0  and 1
    three types of voting are possible """
    (numSeg, segLen) = labelsPerSample.shape
    labelsPerSeg = np.zeros(numSeg)
    for s in range(numSeg):
        if type == 'majority':
            labelsPerSeg[s] = np.round(np.average(labelsPerSample[s, :])+0.001)
            #labelsPerSeg[s] = math.ceil(np.average(labelsPerSample[s, :])) was wrong!! everythign that way above 0 was 1
        elif type == 'atLeastOne':
            labelsPerSeg[s] = int(1 in labelsPerSample[s, :])
        elif type == 'allOne':
            labelsPerSeg[s] = int(sum(labelsPerSample[s, :]) == segLen)
    return labelsPerSeg


def smoothenLabels(prediction,  seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx):
    ''' returns labels after two steps of postprocessing
    first moving window with voting  - if more then threshold of labels are 1 final label is 1 otherwise 0
    second merging seizures that are too close '''

    #labels = labels.reshape(len(labels))
    smoothLabelsStep1=np.zeros((len(prediction)))
    smoothLabelsStep2=np.zeros((len(prediction)))
    try:
        a=int(seizureStableLenToTestIndx)
    except:
        print('error seizureStableLenToTestIndx')
        print(seizureStableLenToTestIndx)
    try:
        a=int(len(prediction))
    except:
        print('error prediction')
        print(prediction)
    #first classifying as true 1 if at laest  GeneralParams.seizureStableLenToTest in a row is 1
    for i in range(int(seizureStableLenToTestIndx), int(len(prediction))):
        s= sum( prediction[i-seizureStableLenToTestIndx+1: i+1] )/seizureStableLenToTestIndx
        try:
            if (s>= seizureStablePercToTest):  #and prediction[i]==1
                smoothLabelsStep1[i]=1
        except:
            print('error')
    smoothLabelsStep2=np.copy(smoothLabelsStep1)

    #second part
    prevSeizureEnd=-distanceBetweenSeizuresIndx
    for i in range(1,len(prediction)):
        if (smoothLabelsStep2[i] == 1 and smoothLabelsStep2[i-1] == 0):  # new seizure started
            # find end of the seizure
            j = i
            while (smoothLabelsStep2[j] == 1 and j< len(smoothLabelsStep2)-1):
                j = j + 1
            #if current seizure distance from prev seizure is too close merge them
            if ((i - prevSeizureEnd) < distanceBetweenSeizuresIndx):  # if  seizure started but is too close to previous one
                #delete secon seizure
                #prevSeizureEnd = j
                #[i:prevSeizureEnd]=np.zeros((prevSeizureEnd-i-1)) #delete second seizure - this was before
                #concatenate seizures
                if (prevSeizureEnd<0): #if exactly first seizure
                    prevSeizureEnd=0
                smoothLabelsStep2[prevSeizureEnd:j] = np.ones((j - prevSeizureEnd ))
            prevSeizureEnd = j
            i=prevSeizureEnd

    return  (smoothLabelsStep2, smoothLabelsStep1)


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

def writeToCsvFile( data, labels,  fileName):
    outputName= fileName+'.csv'
    myFile = open(outputName, 'w',newline='')
    dataToWrite=np.column_stack((data, labels))
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(dataToWrite)

def extractEDFdataToCSV_MoreNonSeizThenSeizData(folderIn, folderOut, SigInfoParams, patients, factor):
    ''' loads one by one patient from raw data folder and if it is seizure file detects where is seizure, and cuts that out
    - picks random nonSeizure file (but that hasn't been used so far) and takes FactorxSeizLen data and puts half before seizure and half after
    - output is for each seizure cutout in form of one .csv file where first n columns are channels data and last column is label (0 non seizure, 1 seizure)
    20200311 UnaPale'''
    createFolderIfNotExists(folderOut)

    for pat in patients:
        print('-- Patient:', pat, folderIn)
        PATIENT = pat if len(sys.argv) < 2 else '{0:02d}'.format(int(sys.argv[1]))
        #number of Seiz and nonSeiz files
        SeizFiles=sorted(glob.glob(f'{folderIn}/chb{PATIENT}*.seizures'))
        EDFNonSeizFiles=sorted(glob.glob(f'{folderIn}/chb{PATIENT}*.edf'))
        fileIndxNonSeiz=0
        print('--- ', len(SeizFiles))
        print('---__ ', len(EDFNonSeizFiles))
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
        for fIndx, f in enumerate(EDFNonSeizFiles):
            justName = os.path.split(f)[1][:-4]
            if (justName not in SeizFileNames):
                if (fIndx == 0):
                    NonSeizFileNames = [justName]
                    NonSeizFileFullNames = [f]
                else:
                    NonSeizFileNames.append(justName)
                    NonSeizFileFullNames.append(f)

        #CREATE NEW FILE FOR EACH SEIZRUE SUBFILE
        for fileIndx,fileName in enumerate(SeizFiles):
            allGood=1

            #load seiz data
            fileName0 = os.path.splitext(fileName)[0]  # removing .seizures from the string
            # here replaced reading .hea files with .edf reading to avoid converting !!!
            (rec, samplFreq, channels) = readEdfFile(fileName0)
            # take only the channels we need and in correct order
            try:
                chToKeepAndInCorrectOrder=[channels.index(SigInfoParams.channels[i]) for i in range(len(SigInfoParams.channels))]
            except:
                print('Sth wrong with the channels in a file: ', fileName)
                allGood=0

            if (allGood==1):
                data = rec[1:, chToKeepAndInCorrectOrder]
                (lenSig, numCh) = data.shape
                # read times of seizures
                szStart = [a for a in MIT.read_annotations(fileName) if a.code == 32]  # start marked with '[' (32)
                szStop = [a for a in MIT.read_annotations(fileName) if a.code == 33]  # start marked with ']' (33)
                # for each seizure cut it out and save
                numSeizures = len(szStart)
                for i in range(numSeizures):
                    seizureLen = szStop[i].time - szStart[i].time
                    newLabel = np.zeros(seizureLen * (factor+1))  # both for seizure nad nonSeizure lavels
                    newData = np.zeros((seizureLen * (factor+1), numCh))

                    nonSeizLen=int(factor*seizureLen)
                    newData[int(nonSeizLen/2):int(nonSeizLen/2)+seizureLen] =  data[(szStart[i].time): (szStart[i].time + seizureLen), :]
                    newLabel[int(nonSeizLen/2):int(nonSeizLen/2)+seizureLen] = np.ones(seizureLen)

                    #load non seizure data
                    goodNonSezFile=0
                    while (goodNonSezFile == 0 and fileIndxNonSeiz<len(NonSeizFileFullNames) ):
                        (rec, samplFreq, channels) = readEdfFile(NonSeizFileFullNames[fileIndxNonSeiz])

                        # take only the channels we need and in correct order
                        try:
                            chToKeepAndInCorrectOrder = [channels.index(SigInfoParams.channels[i]) for i in  range(len(SigInfoParams.channels))]
                            # order data with proper order of channels
                            dataNonSeiz = rec[1:, chToKeepAndInCorrectOrder]
                            (lenSigNonSeiz, numCh) = dataNonSeiz.shape
                            if (lenSigNonSeiz> nonSeizLen):
                                goodNonSezFile = 1
                            else:
                                goodNonSezFile = 0
                                fileIndxNonSeiz = fileIndxNonSeiz + 1
                        except:
                            print('Sth wrong with the channels in a nonSeiz file: ', NonSeizFileFullNames[fileIndx])
                            goodNonSezFile = 0
                            fileIndxNonSeiz = fileIndxNonSeiz + 1

                    if (fileIndxNonSeiz<len(NonSeizFileFullNames)):
                        print(fileName0, '--', NonSeizFileFullNames[fileIndxNonSeiz])

                        #cut nonseizure part
                        nonSeizStart=np.random.randint(lenSigNonSeiz-nonSeizLen-1)
                        nonSeizCutout=dataNonSeiz[nonSeizStart: nonSeizStart + nonSeizLen, :]
                        newData[0:int(nonSeizLen/2)] =nonSeizCutout[0:int(nonSeizLen/2)]
                        newData[int(nonSeizLen/2)+seizureLen:] = nonSeizCutout[int(nonSeizLen / 2):]

                        # SAVING TO CSV FILE
                        pom, fileName1 = os.path.split(fileName0)
                        fileName2 = os.path.splitext(fileName1)[0]
                        fileName3 = folderOut + '/' + fileName2 + '_' + str(i) + '_s'  # 's' marks it is file with seizure
                        writeToCsvFile(newData, newLabel, fileName3)

                        fileIndxNonSeiz = fileIndxNonSeiz + 1
                    else:
                        print('No more nonSeizure files  for file ', fileName2, 'starting again with nonSeiz')
                        fileIndxNonSeiz = 0


def extractEDFdataToCSV_originalData(folderIn, folderOut, SigInfoParams, patients):
    ''' converts data from edf format to csv
    20210705 UnaPale'''
    createFolderIfNotExists(folderOut)

    for pat in patients:
        print('-- Patient:', pat)
        PATIENT = pat if len(sys.argv) < 2 else '{0:02d}'.format(int(sys.argv[1]))
        #number of Seiz and nonSeiz files
        SeizFiles=sorted(glob.glob(f'{folderIn}/chb{PATIENT}*.seizures'))
        EDFNonSeizFiles=sorted(glob.glob(f'{folderIn}/chb{PATIENT}*.edf'))

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
        for fIndx, f in enumerate(EDFNonSeizFiles):
            justName = os.path.split(f)[1][:-4]
            if (justName not in SeizFileNames):
                if (fIndx == 0):
                    NonSeizFileNames = [justName]
                    NonSeizFileFullNames = [f]
                else:
                    NonSeizFileNames.append(justName)
                    NonSeizFileFullNames.append(f)

        #EXPORT SEIZURE FILES
        for fileIndx,fileName in enumerate(SeizFiles):
            allGood=1

            fileName0 = os.path.splitext(fileName)[0]  # removing .seizures from the string
            # here replaced reading .hea files with .edf reading to avoid converting !!!
            (rec, samplFreq, channels) = readEdfFile(fileName0)
            # take only the channels we need and in correct order
            try:
                chToKeepAndInCorrectOrder=[channels.index(SigInfoParams.channels[i]) for i in range(len(SigInfoParams.channels))]
            except:
                print('Sth wrong with the channels in a file: ', fileName)
                allGood=0

            if (allGood==1):
                newData = rec[1:, chToKeepAndInCorrectOrder]
                (lenSig, numCh) = newData.shape
                newLabel = np.zeros(lenSig)
                # read times of seizures
                szStart = [a for a in MIT.read_annotations(fileName) if a.code == 32]  # start marked with '[' (32)
                szStop = [a for a in MIT.read_annotations(fileName) if a.code == 33]  # start marked with ']' (33)
                # for each seizure cut it out and save (with few parameters)
                numSeizures = len(szStart)
                for i in range(numSeizures):
                    seizureLen = szStop[i].time - szStart[i].time
                    newLabel[int(szStart[i].time):int(szStop[i].time)] = np.ones(seizureLen)

                    # saving to csv file
                    pom, fileName1 = os.path.split(fileName0)
                    fileName2 = os.path.splitext(fileName1)[0]
                    fileName3 = folderOut + '/' + fileName2 + '_' + str(i) + '_s'  # 's' marks it is file with seizure
                    print(fileName3)
                    writeToCsvFile(newData, newLabel, fileName3)


        #EXPORT NON SEIZURE FILES
        for fileIndx,fileName in enumerate(NonSeizFileFullNames):
            allGood=1

            (rec, samplFreq, channels) = readEdfFile(fileName)
            # take only the channels we need and in correct order
            try:
                chToKeepAndInCorrectOrder=[channels.index(SigInfoParams.channels[i]) for i in range(len(SigInfoParams.channels))]
            except:
                print('Sth wrong with the channels in a file: ', fileName)
                allGood=0

            if (allGood==1):
                newData = rec[1:, chToKeepAndInCorrectOrder]
                (lenSig, numCh) = newData.shape
                newLabel = np.zeros(lenSig)

                # saving to csv file
                pom, fileName1 = os.path.split(fileName)
                fileName2 = os.path.splitext(fileName1)[0]
                fileName3 = folderOut + '/' + fileName2
                print(fileName3)
                writeToCsvFile(newData, newLabel, fileName3)


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

# def concatenateDataFromFiles_v2(fileNames):
#     dataAll = []
#     for f, fileName in enumerate(fileNames):
#         reader = csv.reader(open(fileName, "r"))
#         data0 = list(reader)
#         data = np.array(data0).astype("float")
#         # separating to data and labels
#         X = data[:, 0:-1]
#         y = data[:, -1]
#         dataSource= np.ones(len(y))*f
#
#         if (dataAll == []):
#             dataAll = X
#             labelsAll = y.astype(int)
#             dataSourceAll=dataSource
#         else:
#             dataAll = np.vstack((dataAll, X))
#             labelsAll = np.hstack((labelsAll, y.astype(int)))
#             dataSourceAll=np.hstack((dataSourceAll, dataSource))
#     return (dataAll, labelsAll, dataSourceAll)
#
def normalizeAndDiscretizeTrainAndTestData(data_train, data_test, numSegLevels):
    ''' normalize train and test data using normalization values from train set
    also discretize values to specific number of levels  given by numSegLevels'''
    # normalizing data
    (numWin_train, numFeat) = data_train.shape
    data_train_Norm = np.zeros((numWin_train, numFeat))
    data_train_Discr = np.zeros((numWin_train, numFeat))
    (numWin_test, numFeat) = data_test.shape
    data_test_Norm = np.zeros((numWin_test, numFeat))
    data_test_Discr = np.zeros((numWin_test, numFeat))
    for f in range(numFeat):
        #normalize and discretize train adn test data
        data_train_Norm[:, f] = (data_train[:, f] - np.min(data_train[:, f])) / ( np.max(data_train[:, f]) - np.min(data_train[:, f]))
        data_train_Discr[:, f] = np.floor((numSegLevels - 1) * data_train_Norm[:, f])
        data_test_Norm[:, f] = (data_test[:, f] - np.min(data_train[:, f])) / ( np.max(data_train[:, f]) - np.min(data_train[:, f]))
        data_test_Discr[:, f] = np.floor((numSegLevels - 1) * data_test_Norm[:, f])
        #check for outliers
        indxs = np.where(data_test_Discr[:, f] >= numSegLevels)
        data_test_Discr[indxs, f] = numSegLevels - 1
        indxs = np.where(data_test_Discr[:, f] < 0)
        data_test_Discr[indxs, f] = 0
        indxs = np.where(np.isnan(data_test_Discr[:, f]))
        data_test_Discr[indxs, f] = 0
        indxs = np.where(data_train_Discr[:, f] >= numSegLevels)
        data_train_Discr[indxs, f] = numSegLevels - 1
        indxs = np.where(data_train_Discr[:, f] < 0)
        data_train_Discr[indxs, f] = 0
        indxs = np.where(np.isnan(data_train_Discr[:, f] ))
        data_train_Discr[indxs, f] = 0
    return(data_train_Norm, data_test_Norm, data_train_Discr, data_test_Discr)

def calculateAllPairwiseDistancesOfVectors_returnMatrix(VecMatrix1,VecMatrix2, vecType ):
    (a, b) = VecMatrix1.shape
    # rearange to row be D and columns subclases
    if (a < b):
        VecMatrix1 = VecMatrix1.transpose()
        VecMatrix2 = VecMatrix2.transpose()
    (D, numClasses1) = VecMatrix1.shape
    (D, numClasses2) = VecMatrix2.shape
    distances = []
    distMat=np.zeros((numClasses1, numClasses2))
    for i in range(numClasses1):
        for j in range( numClasses2):
            # hamming distance
            # vec_c = np.abs(VecMatrix1[:, i] - VecMatrix2[:, j])
            # distances.append(np.sum(vec_c) / float(D))
            # distMat [i,j]= np.sum(vec_c) / float(D)
            dist=ham_dist_arr( VecMatrix1[:, i],VecMatrix2[:, j], D, vecType)
            distances.append(dist)
            distMat [i,j]= dist
    return (distMat, distances)

def ham_dist_arr( vec_a, vec_b, D, vecType='bin'):
    ''' calculate relative hamming distance fur for np array'''
    if (vecType=='bin'):
        vec_c= np.abs(vec_a-vec_b)
        rel_dist = np.sum(vec_c) / float(D)
    elif (vecType=='bipol'):
        vec_c= vec_a+vec_b
        rel_dist = np.sum(vec_c==0) / float(D)
    return rel_dist

def func_plotPerformancesOfDiffApproaches_thisSubj_multiClassPaper( pat, trainTestName,  performancessAll, folderOut):
    ApproachName=[ 'StandardLearning' ,  'MultiClassLearning', 'MultiClassReduced', 'MultiClassClustered']
    AppShortNames=[ '2C','MC','MCred','MCclust']
    AppLineStyle=[ 'k','k--','m','r','r--']
    (numCV, nc, numApp)=performancessAll.shape

    # PLOT PERFORMANCE FOR ALL APPROACHES
    fontSizeNum = 20
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    fig1.suptitle('Subj ' +pat + ' ' + trainTestName)
    xValues = np.arange(0, numCV, 1)
    perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
                 'Precision duration', 'F1score duration', 'F1DEgeoMean', 'simplAcc', 'numFPperDay']
    perfIndxes =[6,7,8,9,10,11,13,2,14]
    for perfIndx, perf in enumerate(perfIndxes):
        ax1 = fig1.add_subplot(gs[int(np.floor(perfIndx / 3)), np.mod(perfIndx, 3)])
        for appIndx, appName in enumerate(ApproachName):
            ax1.plot(xValues, performancessAll[:, perf, appIndx], AppLineStyle[appIndx])
        ax1.legend(AppShortNames)
        ax1.set_xlabel('CVs')
        ax1.set_ylabel('Performance')
        ax1.set_title(perfNames[perfIndx])
        ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/Subj' +pat + 'AllPerfForDiffApproaches_'+trainTestName)
    plt.close(fig1)


def func_plotRawSignalAndPredictionsOfDiffApproaches_thisFile(justName, predictions_test, predictions_train,  approachNames, approachIndx, folderInRawData, folderOut, SigInfoParams, GeneralParams, SegSymbParams):
    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)


    ## LOAD RAW DATA
    reader = csv.reader(open(folderInRawData +'/' + justName+'.csv', "r"))
    data = np.array(list(reader)).astype("float")
    numCh=np.size(data,1)

    # PLOTTING
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(2, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    fig1.suptitle(justName)

    #plotting raw data
    timeRaw=np.arange(0,len(data[:,0]))/256
    ax1 = fig1.add_subplot(gs[0,0])
    # plot all ch raw data
    for ch in range(numCh-1):
        sig=data[:,ch]
        sigNorm=(sig-np.min(sig))/(np.max(sig)-np.min(sig))
        ax1.plot(timeRaw,sigNorm+ch, 'k')
    # plot true label
    ax1.plot(timeRaw, data[:,numCh-1] *numCh, 'r')
    ax1.set_ylabel('Channels')
    ax1.set_xlabel('Time')
    ax1.set_title('Raw data')
    ax1.grid()
    yTrueRaw=data[:,numCh-1]

    #plotting predictions
    yTrue=predictions_test[:,0]
    # approachNames=['2C', '2Citter','MC', 'MCred', 'MCredItter']
    # approachIndx=[1,2,4,6,8]
    ax2 = fig1.add_subplot(gs[1,0])
    for appIndx, app in enumerate(approachIndx):
        yPred_NoSmooth=predictions_test[:,app]
        (yPred_OurSmoothing_step2, yPred_OurSmoothing_step1) = smoothenLabels(yPred_NoSmooth, seizureStableLenToTestIndx, seizureStablePercToTest,distanceBetweenSeizuresIndx)
        yPred_OurSmoothing_step1 = yPred_OurSmoothing_step1 * 0.4 + appIndx
        yPred_OurSmoothing_step2 = yPred_OurSmoothing_step2 * 0.3 + appIndx
        yPred_NoSmooth =yPred_NoSmooth * 0.5 + appIndx
        time = np.arange(0, len(yTrue)) * 0.5
        ax2.plot(time, yPred_NoSmooth, 'k', label='NoSmooth')
        ax2.plot(time, yPred_OurSmoothing_step1, 'b', label='OurSmoothing_step1')
        ax2.plot(time, yPred_OurSmoothing_step2, 'm', label='OurSmoothing_step2')
        if (appIndx == 0):
            ax2.legend()
    #segmentedLabels = segmentLabels(yTrueRaw, SegSymbParams, SigInfoParams)
    time = np.arange(0, len(yTrue)) * 0.5
    ax2.plot(time, yTrue * len(approachNames), 'r')  # label='Performance')
    ax2.set_yticks(np.arange(0,len(approachNames),1))
    ax2.set_yticklabels(approachNames, fontsize=12 * 0.8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Different models')
    ax2.set_title('Predictions')
    ax2.grid()
    if (GeneralParams.plottingON == 1):
        fig1.show()
    fig1.savefig(folderOut + '/'+justName+'_RawDataPlot_Test.png')
    plt.close(fig1)


    #PLOTTING JUST PREDICTIONS AND LABELS FOR TRAIN, WITHOUR RAW DATA
    # PLOTTING
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    fig1.suptitle(justName)
    #plotting predictions
    yTrue=predictions_train[:,0]
    # approachNames=['2C', '2Citter','MC', 'MCred', 'MCredItter']
    # approachIndx=[1,2,4,6,8]
    ax2 = fig1.add_subplot(gs[0,0])
    for appIndx, app in enumerate(approachIndx):
        yPred_NoSmooth=predictions_train[:,app]
        (yPred_OurSmoothing_step2, yPred_OurSmoothing_step1) = smoothenLabels(yPred_NoSmooth, seizureStableLenToTestIndx, seizureStablePercToTest,distanceBetweenSeizuresIndx)
        yPred_OurSmoothing_step1 = yPred_OurSmoothing_step1 * 0.4 + appIndx
        yPred_OurSmoothing_step2 = yPred_OurSmoothing_step2 * 0.3 + appIndx
        yPred_NoSmooth =yPred_NoSmooth * 0.5 + appIndx
        time = np.arange(0, len(yTrue)) * 0.5
        ax2.plot(time, yPred_NoSmooth, 'k', label='NoSmooth')
        ax2.plot(time, yPred_OurSmoothing_step1, 'b', label='OurSmoothing_step1')
        ax2.plot(time, yPred_OurSmoothing_step2, 'm', label='OurSmoothing_step2')
        if (appIndx == 0):
            ax2.legend()
    #segmentedLabels = segmentLabels(yTrueRaw, SegSymbParams, SigInfoParams)
    time = np.arange(0, len(yTrue)) * 0.5
    ax2.plot(time, yTrue * len(approachNames), 'r')  # label='Performance')
    ax2.set_yticks(np.arange(0,len(approachNames),1))
    ax2.set_yticklabels(approachNames, fontsize=12 * 0.8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Different models')
    ax2.set_title('Predictions')
    ax2.grid()
    if (GeneralParams.plottingON == 1):
        fig1.show()
    fig1.savefig(folderOut + '/'+justName+'_RawDataPlot_Train.png')
    plt.close(fig1)

def funct_plotPerformancesForMultiClassPaper_ComparisonSeveralParamsSetup_boxplot( folderOutList, folderOutMulticlass):
    xLabNames = ['2C', 'MC']
    indexsesArray = [6, 7, 8, 9, 10,11 ]
    perfNames = ['Episode sensitivity', 'Episode predictivity ', 'Episode F1score', 'Duration sensitivity', 'Duration predictivity', 'Duration F1score']

    # plotting all perf
    fig1 = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.3)
    #fig1.suptitle('Average of all subject performances', fontsize=20)

    #fig1, axs = plt.subplots(2, 3)
    for indx, i in enumerate(indexsesArray):
        ax1 = fig1.add_subplot(gs[int(np.floor(indx / 3)), int(np.mod(indx, 3))])

        for fIndx, folderIn in enumerate(folderOutList):
            xValues = np.arange(fIndx * len(xLabNames), (fIndx + 1) * len(xLabNames), 1)

            outputName = folderIn + '/AllSubj_StandardLearning_TestRes_mean.csv'
            reader = csv.reader(open(outputName, "r"))
            AllSubj_Standard_Test = np.array(list(reader)).astype("float")
            outputName = folderIn + '/AllSubj_MultiClassLearning_TestRes_mean.csv'
            reader = csv.reader(open(outputName, "r"))
            AllSubj_MultiClass_Test = np.array(list(reader)).astype("float")

            if (fIndx==0):
                dataToPlot=np.vstack((AllSubj_Standard_Test[:,i+9]*100, AllSubj_MultiClass_Test[:,i+9]*100))
            else:
                dataToPlot = np.vstack((dataToPlot, AllSubj_Standard_Test[:, i + 9] * 100, AllSubj_MultiClass_Test[:, i + 9] * 100))


        #boxplots only for test smooth
        ax1.boxplot(dataToPlot.transpose(), medianprops=dict(color='mediumvioletred', linewidth=2),boxprops=dict(linewidth=2),capprops=dict(linewidth=2), whiskerprops=dict(linewidth=2), showfliers=False)
        if (indx==0 or indx==3):
            #ax1.legend(loc='lower right', fontsize=16)
            ax1.set_ylabel('Performance [%]', fontsize=16)
        ax1.set_xticks(np.arange(1, len(folderOutList) * 2+1, 1))
        #xTickNames = ['F1 2C', 'F1 MC', 'F5 2C', 'F5 MC', 'F10 2C', 'F10 MC']
        xTickNames = [ '2C', 'MC', '2C', 'MC', '2C', 'MC']
        ax1.set_xlim([0.5, 7])
        ax1.set_xticklabels(xTickNames, fontsize=16)
        ax1.set_ylim([0,105])
        ax1.set_title(perfNames[indx], fontsize=16)
        ax1.grid()

    fig1.show()
    fig1.savefig(folderOutMulticlass + '/2CvsMCperformance_AllSubj6PerfMeasures_ForMultiClassPaper.png', bbox_inches='tight')
    fig1.savefig(folderOutMulticlass + '/2CvsMCperformance_AllSubj6PerfMeasures_ForMultiClassPaper.svg',bbox_inches='tight')
    plt.close(fig1)

def funct_plotPerformancesForMultiClassPaper_ComparisonSeveralParamsSetup_graph2( folderOutList, folderOutMulticlass, factNames):
    xLabNames = ['2C', 'MC', 'MCred', 'MCclust']

    # plotting all perf
    fig1 = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(2, len(folderOutList), figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.25)
    #fig1.suptitle('Average of all subject performances', fontsize=20)

    for fIndx, folderIn in enumerate(folderOutList):
        xValues = np.arange(0, len(xLabNames), 1)

        outputName = folderIn + '/AllSubjAvrg_StandardLearning_TrainRes.csv'
        reader = csv.reader(open(outputName, "r"))
        TotalMean_2class_train = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubjAvrg_StandardLearning_TestRes.csv'
        reader = csv.reader(open(outputName, "r"))
        TotalMean_2class_test = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubjAvrg_MultiClassLearning_TrainRes.csv'
        reader = csv.reader(open(outputName, "r"))
        TotalMean_Multi_train = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubjAvrg_MultiClassLearning_TestRes.csv'
        reader = csv.reader(open(outputName, "r"))
        TotalMean_Multi_test = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubjAvrg_MultiClassReduced_TrainRes.csv'
        reader = csv.reader(open(outputName, "r"))
        TotalMean_MultiRed_train = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubjAvrg_MultiClassReduced_TestRes.csv'
        reader = csv.reader(open(outputName, "r"))
        TotalMean_MultiRed_test = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubjAvrg_MultiClassClustered_TrainRes.csv'
        reader = csv.reader(open(outputName, "r"))
        TotalMean_MultiClust_train = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubjAvrg_MultiClassClustered_TestRes.csv'
        reader = csv.reader(open(outputName, "r"))
        TotalMean_MultiClust_test = np.array(list(reader)).astype("float")

        dataToPlotMean_train = np.dstack((TotalMean_2class_train, TotalMean_Multi_train, TotalMean_MultiRed_train, TotalMean_MultiClust_train))
        dataToPlotMean_test = np.dstack((TotalMean_2class_test, TotalMean_Multi_test, TotalMean_MultiRed_test, TotalMean_MultiClust_test))


        #plotting performance F1DEgmean
        pIndx=13
        ax1 = fig1.add_subplot(gs[0, int(np.mod(fIndx, 3))])
        # # train
        # ax1.errorbar(xValues, dataToPlotMean_train[0, pIndx, :] * 100, yerr=dataToPlotMean_train[1, pIndx, :] * 100, fmt='k',label='Train NoSmooth')
        # ax1.errorbar(xValues, dataToPlotMean_train[0, pIndx + 9, :] * 100, yerr=dataToPlotMean_train[1, pIndx + 9, :] * 100,fmt='b', label='Train Smooth')
        # # ax1.errorbar(xValues, dataToPlotMean_train[0, pIndx+18,:]*100, yerr=dataToPlotMean_train[1, pIndx+18,:]*100, fmt='m', label='Train Step2')
        # # test
        # ax1.errorbar(xValues, dataToPlotMean_test[0, pIndx, :] * 100, yerr=dataToPlotMean_test[1, pIndx, :] * 100, fmt='k--', label='Test NoSmooth')
        # ax1.errorbar(xValues, dataToPlotMean_test[0, pIndx + 9, :] * 100, yerr=dataToPlotMean_test[1, pIndx + 9, :] * 100, fmt='b--', label='Test Smooth')
        # # ax1.errorbar(xValues, dataToPlotMean_test[0, pIndx+18,:]*100, yerr=dataToPlotMean_test[1, pIndx+18,:]*100, fmt='m--', label='Test Step2')

        # just test
        ax1.errorbar(xValues, dataToPlotMean_test[0, pIndx, :] * 100, yerr=dataToPlotMean_test[1, pIndx, :] * 100, fmt='gray', label='Test NoSmooth', linewidth=2)
        ax1.errorbar(xValues, dataToPlotMean_test[0, pIndx + 9, :] * 100, yerr=dataToPlotMean_test[1, pIndx + 9, :] * 100, fmt='black', label='Test Smooth', linewidth=2)
        # ax1.errorbar(xValues, dataToPlotMean_test[0, pIndx+18,:]*100, yerr=dataToPlotMean_test[1, pIndx+18,:]*100, fmt='m--', label='Test Step2')
        if (fIndx==0):
            ax1.set_ylabel('Performance [%]', fontsize=16) #F1DEgmean
            ax1.legend(loc='lower right', fontsize=14)
        ax1.set_title(factNames[fIndx], fontsize=16)
        ax1.set_xticks(np.arange(0, len(xLabNames), 1))
        ax1.set_xticklabels(xLabNames, fontsize=16)
        ax1.set_ylim([30,105])
        ax1.grid()

        #plotting numSubclasses
        ax1 = fig1.add_subplot(gs[1, int(np.mod(fIndx, 3))])
        # seiz
        pIndx = 0
        ax1.errorbar(xValues, dataToPlotMean_train[0, pIndx, :] , yerr=dataToPlotMean_train[1, pIndx, :] , fmt='mediumvioletred',label='Seizure', linewidth=2)
        # seiz
        pIndx = 1
        ax1.errorbar(xValues, dataToPlotMean_train[0, pIndx, :] , yerr=dataToPlotMean_train[1, pIndx, :] , fmt='royalblue',label='NonSeizure', linewidth=2)
        if (fIndx == 0):
            ax1.set_ylabel('Number sub-classes', fontsize=16)
            ax1.legend(loc='upper right', fontsize=14)
        #ax1.set_title('Number of sub-classes', fontsize=16)
        ax1.set_xticks(np.arange(0, len(xLabNames), 1))
        ax1.set_xticklabels(xLabNames, fontsize=16)
        ax1.set_ylim([0, 30])
        ax1.grid()

    fig1.show()
    fig1.savefig(folderOutMulticlass + '/2CvsMCall_PerfAndNumSubclasses_ForMultiClassPaper.png', bbox_inches='tight')
    fig1.savefig(folderOutMulticlass + '/2CvsMCall_PerfAndNumSubclasses_ForMultiClassPaper.svg', bbox_inches='tight')
    plt.close(fig1)

def funct_plotPerformancesForMultiClassPaper_ComparisonSeveralParamsSetup_graph3( folderOutList, folderOutMulticlass, factNames):

    DiffAllParams_train_mean=np.zeros((len(folderOutList), 33 ))
    DiffAllParams_train_std= np.zeros((len(folderOutList), 33))
    DiffAllParams_test_mean = np.zeros((len(folderOutList), 33))
    DiffAllParams_test_std = np.zeros((len(folderOutList), 33))
    NumSubclassesRed_mean= np.zeros((len(folderOutList), 2))
    NumSubclassesRed_std = np.zeros((len(folderOutList), 2))
    for fIndx, folderIn in enumerate(folderOutList):

        outputName = folderIn + '/AllSubj_StandardLearning_TrainRes_mean.csv'
        reader = csv.reader(open(outputName, "r"))
        AllSubj_2class_train = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubj_StandardLearning_TestRes_mean.csv'
        reader = csv.reader(open(outputName, "r"))
        AllSubj_2class_test = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubj_MultiClassReduced_TrainRes_mean.csv'
        reader = csv.reader(open(outputName, "r"))
        AllSubj_MultiRed_train = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubj_MultiClassReduced_TestRes_mean.csv'
        reader = csv.reader(open(outputName, "r"))
        AllSubj_MultiRed_test = np.array(list(reader)).astype("float")

        Diff_train= AllSubj_MultiRed_train-AllSubj_2class_train
        Diff_test = AllSubj_MultiRed_test - AllSubj_2class_test
        if (fIndx==0):
            DiffAllParams_train= Diff_train
            DiffAllParams_test = Diff_test
        else:
            DiffAllParams_train = np.dstack((DiffAllParams_train, Diff_train))
            DiffAllParams_test = np.dstack((DiffAllParams_test, Diff_test))

        DiffAllParams_train_mean[fIndx,:] =np.mean(Diff_train,0)
        DiffAllParams_train_std[fIndx,:]  = np.std(Diff_train, 0)
        DiffAllParams_test_mean[fIndx,:] =np.mean(Diff_test,0)
        DiffAllParams_test_std[fIndx,:]  = np.std(Diff_test, 0)

        NumSubclassesRed_mean[fIndx,:]=[ np.mean(AllSubj_MultiRed_train[:,0]), np.mean(AllSubj_MultiRed_train[:,1])  ] #siez, nonSeiz
        NumSubclassesRed_std[fIndx,:]=[ np.std(AllSubj_MultiRed_train[:,0]), np.std(AllSubj_MultiRed_train[:,1])  ]

    # plotting Perf improvement - of F1DEgmean
    fig1 = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = GridSpec(1,2, figure=fig1)
    fig1.subplots_adjust(wspace=0.1, hspace=0.2)
    #fig1.suptitle('Average of all subject performances', fontsize=20)

    #plotting performance F1DEgmean
    pIndx=13
    ax1 = fig1.add_subplot(gs[0, 0])
    xValues=np.arange(0,len(folderOutList),1)
    # # train
    # ax1.errorbar(xValues, DiffAllParams_train_mean[:, pIndx] * 100, yerr=DiffAllParams_train_std[:, pIndx] * 100, fmt='k',label='Train NoSmooth')
    # ax1.errorbar(xValues, DiffAllParams_train_mean[:, pIndx+9] * 100, yerr=DiffAllParams_train_std[:, pIndx+9] * 100,fmt='b', label='Train Smooth')
    # # test
    # ax1.errorbar(xValues, DiffAllParams_test_mean[:, pIndx] * 100, yerr=DiffAllParams_test_std[:, pIndx] * 100, fmt='k--', label='Test NoSmooth')
    # ax1.errorbar(xValues, DiffAllParams_test_mean[:, pIndx+9]  * 100, yerr=DiffAllParams_test_std[:, pIndx+9] * 100, fmt='b--', label='Test Smooth')

    #  just test
    ax1.errorbar(xValues, DiffAllParams_test_mean[:, pIndx] * 100, yerr=DiffAllParams_test_std[:, pIndx] * 100, fmt='k', label='Test NoSmooth')
    ax1.errorbar(xValues, DiffAllParams_test_mean[:, pIndx+9]  * 100, yerr=DiffAllParams_test_std[:, pIndx+9] * 100, fmt='b', label='Test Smooth')
    ax1.set_ylabel('Performance improvement from 2C model [%]', fontsize=16) #F1DEgmean
    ax1.set_title('F1DEgmean', fontsize=20)
    ax1.legend(loc='lower right', fontsize=16)
    ax1.set_xticks(np.arange(0, len(factNames), 1))
    ax1.set_xticklabels(factNames, fontsize=18)
    ax1.grid()

    #plotting numSubclasses
    ax1 = fig1.add_subplot(gs[0,1])
    # seiz
    ax1.errorbar(xValues, NumSubclassesRed_mean[:,0] , yerr=NumSubclassesRed_std[:,0] , fmt='m',label='Seizure')
    # nonseiz
    ax1.errorbar(xValues, NumSubclassesRed_mean[:,1] , yerr=NumSubclassesRed_std[:,1] , fmt='k',label='NonSeizure')
    ax1.set_title('Number sub-classes', fontsize=20)
    ax1.legend(loc='lower right', fontsize=16)
    ax1.set_xticks(np.arange(0, len(factNames), 1))
    ax1.set_xticklabels(factNames, fontsize=18)
    ax1.grid()

    fig1.show()
    fig1.savefig(folderOutMulticlass + '/Fact1510comparison_PerfImprovAndNumSubclasses_ForMultiClassPaper.png', bbox_inches='tight')
    plt.close(fig1)


def funct_plotPerformancesForMultiClassPaper_ComparisonSeveralParamsSetup_graph3_boxplot( folderOutList, folderOutMulticlass, factNames):

    DiffAllParams_train_mean=np.zeros((len(folderOutList), 33 ))
    DiffAllParams_train_std= np.zeros((len(folderOutList), 33))
    DiffAllParams_test_mean = np.zeros((len(folderOutList), 33))
    DiffAllParams_test_std = np.zeros((len(folderOutList), 33))
    NumSubclassesRed_mean= np.zeros((len(folderOutList), 2))
    NumSubclassesRed_std = np.zeros((len(folderOutList), 2))
    for fIndx, folderIn in enumerate(folderOutList):

        outputName = folderIn + '/AllSubj_StandardLearning_TrainRes_mean.csv'
        reader = csv.reader(open(outputName, "r"))
        AllSubj_2class_train = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubj_StandardLearning_TestRes_mean.csv'
        reader = csv.reader(open(outputName, "r"))
        AllSubj_2class_test = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubj_MultiClassReduced_TrainRes_mean.csv'
        reader = csv.reader(open(outputName, "r"))
        AllSubj_MultiRed_train = np.array(list(reader)).astype("float")
        outputName = folderIn + '/AllSubj_MultiClassReduced_TestRes_mean.csv'
        reader = csv.reader(open(outputName, "r"))
        AllSubj_MultiRed_test = np.array(list(reader)).astype("float")

        Diff_train= AllSubj_MultiRed_train-AllSubj_2class_train
        Diff_test = AllSubj_MultiRed_test - AllSubj_2class_test
        if (fIndx==0):
            DiffAllParams_train= Diff_train
            DiffAllParams_test = Diff_test
            NumSeiz=AllSubj_MultiRed_train[:,0]
            NumNonSeiz = AllSubj_MultiRed_train[:, 1]
        else:
            DiffAllParams_train = np.dstack((DiffAllParams_train, Diff_train))
            DiffAllParams_test = np.dstack((DiffAllParams_test, Diff_test))
            NumSeiz=np.vstack((NumSeiz, AllSubj_MultiRed_train[:, 0]))
            NumNonSeiz =np.vstack((NumNonSeiz, AllSubj_MultiRed_train[:, 1]))

        DiffAllParams_train_mean[fIndx,:] =np.mean(Diff_train,0)
        DiffAllParams_train_std[fIndx,:]  = np.std(Diff_train, 0)
        DiffAllParams_test_mean[fIndx,:] =np.mean(Diff_test,0)
        DiffAllParams_test_std[fIndx,:]  = np.std(Diff_test, 0)

        NumSubclassesRed_mean[fIndx,:]=[ np.mean(AllSubj_MultiRed_train[:,0]), np.mean(AllSubj_MultiRed_train[:,1])  ] #siez, nonSeiz
        NumSubclassesRed_std[fIndx,:]=[ np.std(AllSubj_MultiRed_train[:,0]), np.std(AllSubj_MultiRed_train[:,1])  ]

    # plotting Perf improvement - of F1DEgmean
    fig1 = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = GridSpec(1,3, figure=fig1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.2)
    #fig1.suptitle('Average of all subject performances', fontsize=20)

    #plotting performance F1DEgmean
    pIndx=13
    ax1 = fig1.add_subplot(gs[0, 0])
    xValues=np.arange(0,len(folderOutList),1)
    #  just test
    ax1.boxplot( DiffAllParams_test[:, pIndx] * 100, medianprops=dict(color='red', linewidth=2),boxprops=dict(linewidth=2),capprops=dict(linewidth=2), whiskerprops=dict(linewidth=2), showfliers=False)
    ax1.set_ylabel('Performance improvement from 2C model [%]', fontsize=16) #F1DEgmean
    ax1.set_title('F1DEgmean', fontsize=16)
    #ax1.legend(loc='lower right', fontsize=16)
    ax1.set_xticks(np.arange(1, len(factNames)+1, 1))
    ax1.set_xticklabels([ 'F1', 'F5', 'F10'], fontsize=16)
    ax1.set_xlim([0.5, 3.5])
    ax1.grid()

    #plotting numSubclasses seiz
    ax1 = fig1.add_subplot(gs[0,1])
    # seiz
    #ax1.errorbar(xValues, NumSubclassesRed_mean[:,0] , yerr=NumSubclassesRed_std[:,0] , fmt='m',label='Seizure')
    ax1.boxplot( NumSeiz.transpose(), medianprops = dict(color='black',linestyle=':', linewidth=2),meanprops = dict(color='mediumvioletred',linestyle='-',  linewidth=2), boxprops = dict(linewidth=2), capprops = dict(
        linewidth=2), whiskerprops = dict(linewidth=2), showfliers = False, meanline=True, showmeans=True)
    # nonseiz
    #ax1.errorbar(xValues, NumSubclassesRed_mean[:,1] , yerr=NumSubclassesRed_std[:,1] , fmt='k',label='NonSeizure')
    ax1.set_title('Nr. sub-classes', fontsize=16)
    #ax1.legend(loc='lower right', fontsize=16)
    ax1.set_xticks(np.arange(1, len(factNames)+1, 1))
    ax1.set_xticklabels([ 'F1', 'F5', 'F10'], fontsize=18)
    ax1.set_xlim([0.5, 3.5])
    ax1.grid()

    #plotting numSubclasses non siez2CvsMCall_PerfAndNumSubclasses_ForMultiClassPaper
    ax1 = fig1.add_subplot(gs[0,2])
    # seiz
    #ax1.errorbar(xValues, NumSubclassesRed_mean[:,0] , yerr=NumSubclassesRed_std[:,0] , fmt='m',label='Seizure')
    ax1.boxplot( NumNonSeiz.transpose(),medianprops = dict(color='black',linestyle=':', linewidth=2),meanprops = dict(color='royalblue',linestyle='-',  linewidth=2), boxprops = dict(linewidth=2), capprops = dict(
        linewidth=2), whiskerprops = dict(linewidth=2), showfliers = False, meanline=True, showmeans=True)
    # nonseiz
    #ax1.errorbar(xValues, NumSubclassesRed_mean[:,1] , yerr=NumSubclassesRed_std[:,1] , fmt='k',label='NonSeizure')
    ax1.set_title('Nr. sub-classes', fontsize=16)
    #ax1.legend(loc='lower right', fontsize=16)
    ax1.set_xticks(np.arange(1, len(factNames)+1, 1))
    ax1.set_xticklabels([ 'F1', 'F5', 'F10'], fontsize=18)
    ax1.set_xlim([0.5, 3.5])
    ax1.grid()

    fig1.show()
    fig1.savefig(folderOutMulticlass + '/Fact1510comparison_PerfImprovAndNumSubclasses_ForMultiClassPaper.png', bbox_inches='tight')
    fig1.savefig(folderOutMulticlass + '/Fact1510comparison_PerfImprovAndNumSubclasses_ForMultiClassPaper.svg',bbox_inches='tight')
    plt.close(fig1)

def func_plotNumDataPerSubclasses_forMultiClassPaper( folderIn, folderOut, GeneralParams):
    ''' function that plot amount of data per subclasses '''


    fig2= plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = GridSpec(2,len(GeneralParams.patients), figure=fig2)
    fig2.subplots_adjust(wspace=0.4, hspace=0.2)
    for patIndx, pat in enumerate(GeneralParams.patients):
        files = np.sort(glob.glob(folderIn + '/*chb' + pat + '*_MultiClass_AddedToEachSubClass.csv'))

        seizData_thisSubj=[]
        nonSeizData_thisSubj = []
        for fIndx, fileName in enumerate(files):
            if (fIndx==0):
                reader = csv.reader(open(fileName, "r"))
                data = np.array(list(reader)).astype("float")
                #calculating percentages
                seizData=data[~np.isnan(data[:,0]), 0]
                nonSeizData = data[~np.isnan(data[:, 1]), 1]
                seizData=seizData*100/np.nansum(seizData)
                nonSeizData= nonSeizData*100/ np.nansum(nonSeizData)
                #counting to make histogram
                seizData_thisSubj=np.concatenate((seizData_thisSubj, seizData))
                nonSeizData_thisSubj = np.concatenate((nonSeizData_thisSubj, nonSeizData))

                ax1 = fig2.add_subplot(gs[ 0, patIndx])
                matplotlib.pyplot.bar(np.arange(0,len(seizData),1) , seizData, color=(0.8, 0.4, 0.6, 0.6))
                if (patIndx==0):
                    ax1.set_ylabel('Percentage of data', fontsize=16)
                ax1.set_title('Subj '+ pat, fontsize=16)
                ax1.set_ylim(0,100)
                ax1.grid()
                ax1 = fig2.add_subplot(gs[ 1, patIndx])
                matplotlib.pyplot.bar(np.arange(0, len(nonSeizData), 1), nonSeizData, color=(0.2, 0.4, 0.6, 0.6))
                ax1.set_xlabel('Subclass', fontsize=16)
                if (patIndx == 0):
                    ax1.set_ylabel('Percentage of data', fontsize=16)
                ax1.set_ylim(0, 100)
                ax1.grid()

    fig2.show()
    fig2.savefig(folderIn + '/AllSubj_PercDataPerSubclasses_forMultiClassPaper.png', bbox_inches='tight')
    fig2.savefig(folderIn + '/AllSubj_PercDataPerSubclasses_forMultiClassPaper.svg', bbox_inches='tight')
    plt.close(fig2)

def func_calculateMean_andPlotPerSubj_RemovingClusteringSublasses(folder, numSteps, GeneralParams):
    percDataSteps = 1 - np.arange(0, numSteps, 1) / numSteps
    # allPerfMeasures_train_AllSubj = np.zeros((numSteps + 1, 9 * 3+1, len(GeneralParams.patients)))
    # allPerfMeasures_test_AllSubj = np.zeros((numSteps + 1, 9 * 3+1, len(GeneralParams.patients)))
    # variousPerfMeasures_AllSubj = np.zeros((numSteps + 1, 9, len(GeneralParams.patients)))
    allPerfMeasures_train_AllSubj = np.ones((numSteps, 9 * 3 + 1, len(GeneralParams.patients))) * np.nan
    allPerfMeasures_test_AllSubj = np.ones((numSteps, 9 * 3 + 1, len(GeneralParams.patients))) * np.nan
    variousPerfMeasures_AllSubj = np.ones((numSteps, 12, len(GeneralParams.patients))) * np.nan
    # percSubclassesKept_AllSubj=np.ones((numSteps+1, len(GeneralParams.patients)))*np.nan
    for patIndx, pat in enumerate(GeneralParams.patients):
        files = np.sort(glob.glob(folder + '/*chb' + pat + '*_PerformanceMeasuresPerItteration_Train.csv'))

        fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
        gs = GridSpec(3, 3, figure=fig1)
        fig1.subplots_adjust(wspace=0.4, hspace=0.6)
        fig1.suptitle('Removing subclasses - Subj ' + pat)
        # allPerfMeasures_train_thisSubj=np.zeros((numSteps+1, 9*3+1, len(files)))
        # allPerfMeasures_test_thisSubj=np.zeros((numSteps+1, 9*3+1, len(files)))
        # variousPerfMeasures_thisSubj=np.zeros((numSteps+1, 9, len(files)))
        allPerfMeasures_train_thisSubj = np.ones((numSteps, 9 * 3 + 1, len(files))) * np.nan
        allPerfMeasures_test_thisSubj = np.ones((numSteps, 9 * 3 + 1, len(files))) * np.nan
        variousPerfMeasures_thisSubj = np.ones((numSteps, 12, len(files))) * np.nan
        # percSubclassesKept_thisSubj=np.ones((numSteps+1, len(files)))*np.nan
        for fIndx, fileName in enumerate(files):
            fileName2 = fileName[0:-43] + '_VariousMeasuresPerItteration.csv'
            reader = csv.reader(open(fileName2, "r"))
            data = np.array(list(reader)).astype("float")
            percSubclassesKept_thisSubj = data[:, 2] / data[0, 2]
            minVal = np.min(percSubclassesKept_thisSubj)
            goodIndx = np.where(percDataSteps >= minVal)
            (nr, nc) = data.shape
            # interpolate to fixted steps of removing data
            for c in range(nc):
                f = interpolate.interp1d(percSubclassesKept_thisSubj, data[:, c])
                variousPerfMeasures_thisSubj[goodIndx, c, fIndx] = f(percDataSteps[goodIndx])

            reader = csv.reader(open(fileName, "r"))
            data = np.array(list(reader)).astype("float")
            (nr, nc) = data.shape
            # interpolate to fixted steps of removing data
            for c in range(nc):
                f = interpolate.interp1d(percSubclassesKept_thisSubj, data[:, c])
                allPerfMeasures_train_thisSubj[goodIndx, c, fIndx] = f(percDataSteps[goodIndx])

            fileName2 = fileName[0:-10] + '_Test.csv'
            reader = csv.reader(open(fileName2, "r"))
            data = np.array(list(reader)).astype("float")
            (nr, nc) = data.shape
            # interpolate to fixted steps of removing data
            for c in range(nc):
                f = interpolate.interp1d(percSubclassesKept_thisSubj, data[:, c])
                allPerfMeasures_test_thisSubj[goodIndx, c, fIndx] = f(percDataSteps[goodIndx])

        allPerfMeasures_train_thisSubj_mean = np.nanmean(allPerfMeasures_train_thisSubj, 2)
        allPerfMeasures_train_thisSubj_std = np.nanstd(allPerfMeasures_train_thisSubj, 2)
        allPerfMeasures_test_thisSubj_mean = np.nanmean(allPerfMeasures_test_thisSubj, 2)
        allPerfMeasures_test_thisSubj_std = np.nanstd(allPerfMeasures_test_thisSubj, 2)
        variousPerfMeasures_thisSubj_mean = np.nanmean(variousPerfMeasures_thisSubj, 2)
        variousPerfMeasures_thisSubj_std = np.nanstd(variousPerfMeasures_thisSubj, 2)
        # percSubclassesKept_thisSubj_mean=np.nanmean(percSubclassesKept_thisSubj,1)
        outputName = folder + '/Subj' + pat + '_PerformanceMeasuresPerItteration_Train_mean.csv'
        np.savetxt(outputName, allPerfMeasures_train_thisSubj_mean, delimiter=",")
        outputName = folder + '/Subj' + pat + '_PerformanceMeasuresPerItteration_Train_std.csv'
        np.savetxt(outputName, allPerfMeasures_train_thisSubj_std, delimiter=",")
        outputName = folder + '/Subj' + pat + '_PerformanceMeasuresPerItteration_Test_mean.csv'
        np.savetxt(outputName, allPerfMeasures_test_thisSubj_mean, delimiter=",")
        outputName = folder + '/Subj' + pat + '_PerformanceMeasuresPerItteration_Test_std.csv'
        np.savetxt(outputName, allPerfMeasures_test_thisSubj_std, delimiter=",")
        outputName = folder + '/Subj' + pat + '_VariousMeasuresPerItteration_mean.csv'
        np.savetxt(outputName, variousPerfMeasures_thisSubj_mean, delimiter=",")
        outputName = folder + '/Subj' + pat + '_VariousMeasuresPerItteration_std.csv'
        np.savetxt(outputName, variousPerfMeasures_thisSubj_std, delimiter=",")

        allPerfMeasures_train_AllSubj[:, :, patIndx] = allPerfMeasures_train_thisSubj_mean
        allPerfMeasures_test_AllSubj[:, :, patIndx] = allPerfMeasures_test_thisSubj_mean
        variousPerfMeasures_AllSubj[:, :, patIndx] = variousPerfMeasures_thisSubj_mean
        # percSubclassesKept_AllSubj[:, patIndx] = percSubclassesKept_thisSubj_mean

        # xValues=np.arange(0,1+1/numSteps,1/numSteps)
        xValues = 1 - percDataSteps
        # xValues=1-percSubclassesKept_thisSubj_mean #percentage of data removed or clustered
        # numClasses kept
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.errorbar(xValues, variousPerfMeasures_thisSubj_mean[:, 0], yerr=variousPerfMeasures_thisSubj_std[:, 0], fmt='k-.', label='NonSeiz')
        ax1.errorbar(xValues, variousPerfMeasures_thisSubj_mean[:, 1], yerr=variousPerfMeasures_thisSubj_std[:, 1], fmt='m-.', label='Seiz')
        ax1.errorbar(xValues, variousPerfMeasures_thisSubj_mean[:, 2], yerr=variousPerfMeasures_thisSubj_std[:, 2], fmt='b-.', label='Both')
        ax1.legend()
        ax1.set_xlabel('Percentage of data removed')
        # ax1.set_ylabel('Num subclasses kept')
        ax1.set_title('Num subclasses kept')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[0, 1])
        ax1.errorbar(xValues, variousPerfMeasures_thisSubj_mean[:, 3], yerr=variousPerfMeasures_thisSubj_std[:, 3],  fmt='k-.', label='NonSeiz')
        ax1.errorbar(xValues, variousPerfMeasures_thisSubj_mean[:, 4], yerr=variousPerfMeasures_thisSubj_std[:, 4], fmt='m-.', label='Seiz')
        ax1.errorbar(xValues, variousPerfMeasures_thisSubj_mean[:, 5], yerr=variousPerfMeasures_thisSubj_std[:, 5], fmt='b-.', label='Both')
        ax1.legend()
        ax1.set_xlabel('Percentage of data removed')
        # ax1.set_ylabel('Num subclasses kept')
        ax1.set_title('Percentage of data in subclasses')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[0, 2])
        ax1.errorbar(xValues, variousPerfMeasures_thisSubj_mean[:, 7], yerr=variousPerfMeasures_thisSubj_std[:, 7], fmt='k-.', label='CorrClass')
        ax1.errorbar(xValues, variousPerfMeasures_thisSubj_mean[:, 8], yerr=variousPerfMeasures_thisSubj_std[:, 8], fmt='m-.', label='WrongClass')
        ax1.errorbar(xValues, variousPerfMeasures_thisSubj_mean[:, 6], yerr=variousPerfMeasures_thisSubj_std[:, 6], fmt='b-.', label='Misclassified')
        ax1.legend()
        ax1.set_xlabel('Percentage of data removed')
        # ax1.set_ylabel('Num subclasses kept')
        ax1.set_title('Distances ')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[1, 0])
        ax1.errorbar(xValues, allPerfMeasures_train_thisSubj_mean[:, 3], yerr=allPerfMeasures_train_thisSubj_std[:, 3], fmt='k-.', label='NoSmooth')
        ax1.errorbar(xValues, allPerfMeasures_train_thisSubj_mean[:, 12], yerr=allPerfMeasures_train_thisSubj_std[:, 12], fmt='b-.', label='Step1')
        ax1.errorbar(xValues, allPerfMeasures_train_thisSubj_mean[:, 21], yerr=allPerfMeasures_train_thisSubj_std[:, 21], fmt='m-.', label='Step2')
        ax1.legend()
        ax1.set_xlabel('Percentage of data removed')
        # ax1.set_ylabel('Num subclasses kept')
        ax1.set_title('Performance F1E -Train')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[1, 1])
        ax1.errorbar(xValues, allPerfMeasures_train_thisSubj_mean[:, 6], yerr=allPerfMeasures_train_thisSubj_std[:, 6], fmt='k-.', label='NoSmooth')
        ax1.errorbar(xValues, allPerfMeasures_train_thisSubj_mean[:, 15], yerr=allPerfMeasures_train_thisSubj_std[:, 15], fmt='b-.', label='Step1')
        ax1.errorbar(xValues, allPerfMeasures_train_thisSubj_mean[:, 24], yerr=allPerfMeasures_train_thisSubj_std[:, 24], fmt='m-.', label='Step2')
        ax1.legend()
        ax1.set_xlabel('Percentage of data removed')
        # ax1.set_ylabel('Num subclasses kept')
        ax1.set_title('Performance F1D -Train ')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[1, 2])
        ax1.errorbar(xValues, allPerfMeasures_train_thisSubj_mean[:, 8], yerr=allPerfMeasures_train_thisSubj_std[:, 8], fmt='k-.', label='NoSmooth')
        ax1.errorbar(xValues, allPerfMeasures_train_thisSubj_mean[:, 17], yerr=allPerfMeasures_train_thisSubj_std[:, 17], fmt='b-.', label='Step1')
        ax1.errorbar(xValues, allPerfMeasures_train_thisSubj_mean[:, 26], yerr=allPerfMeasures_train_thisSubj_std[:, 26], fmt='m-.', label='Step2')
        ax1.legend()
        ax1.set_xlabel('Percentage of data removed')
        # ax1.set_ylabel('Num subclasses kept')
        ax1.set_title('Performance F1both -Train ')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[2, 0])
        ax1.errorbar(xValues, allPerfMeasures_test_thisSubj_mean[:, 3], yerr=allPerfMeasures_test_thisSubj_std[:, 3],   fmt='k-.', label='NoSmooth')
        ax1.errorbar(xValues, allPerfMeasures_test_thisSubj_mean[:, 12], yerr=allPerfMeasures_test_thisSubj_std[:, 12], fmt='b-.', label='Step1')
        ax1.errorbar(xValues, allPerfMeasures_test_thisSubj_mean[:, 21], yerr=allPerfMeasures_test_thisSubj_std[:, 21], fmt='m-.', label='Step2')
        ax1.legend()
        ax1.set_xlabel('Percentage of data removed')
        # ax1.set_ylabel('Num subclasses kept')
        ax1.set_title('Performance F1E -Test')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[2, 1])
        ax1.errorbar(xValues, allPerfMeasures_test_thisSubj_mean[:, 6], yerr=allPerfMeasures_test_thisSubj_std[:, 6],  fmt='k-.', label='NoSmooth')
        ax1.errorbar(xValues, allPerfMeasures_test_thisSubj_mean[:, 15], yerr=allPerfMeasures_test_thisSubj_std[:, 15],fmt='b-.', label='Step1')
        ax1.errorbar(xValues, allPerfMeasures_test_thisSubj_mean[:, 24], yerr=allPerfMeasures_test_thisSubj_std[:, 24], fmt='m-.', label='Step2')
        ax1.legend()
        ax1.set_xlabel('Percentage of data removed')
        # ax1.set_ylabel('Num subclasses kept')
        ax1.set_title('Performance F1D -Test')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[2, 2])
        ax1.errorbar(xValues, allPerfMeasures_test_thisSubj_mean[:, 8], yerr=allPerfMeasures_test_thisSubj_std[:, 8],  fmt='k-.', label='NoSmooth')
        ax1.errorbar(xValues, allPerfMeasures_test_thisSubj_mean[:, 17], yerr=allPerfMeasures_test_thisSubj_std[:, 17], fmt='b-.', label='Step1')
        ax1.errorbar(xValues, allPerfMeasures_test_thisSubj_mean[:, 26], yerr=allPerfMeasures_test_thisSubj_std[:, 26], fmt='m-.', label='Step2')
        ax1.legend()
        ax1.set_xlabel('Percentage of data removed')
        # ax1.set_ylabel('Num subclasses kept')
        ax1.set_title('Performance F1both -Test')
        ax1.grid()
        fig1.show()
        fig1.savefig(folder + '/Subj' + pat + '_ItterativeRemovinAllRes.png')
        plt.close(fig1)

    allPerfMeasures_train_AllSubj_mean = np.nanmean(allPerfMeasures_train_AllSubj, 2)
    allPerfMeasures_train_AllSubj_std = np.nanstd(allPerfMeasures_train_AllSubj, 2)
    allPerfMeasures_test_AllSubj_mean = np.nanmean(allPerfMeasures_test_AllSubj, 2)
    allPerfMeasures_test_AllSubj_std = np.nanstd(allPerfMeasures_test_AllSubj, 2)
    variousPerfMeasures_AllSubj_mean = np.nanmean(variousPerfMeasures_AllSubj, 2)
    variousPerfMeasures_AllSubj_std = np.nanstd(variousPerfMeasures_AllSubj, 2)
    # percSubclassesKept_AllSubj_mean = np.nanmean(percSubclassesKept_AllSubj, 1)

    outputName = folder + '/AllSubj_PerformanceMeasuresPerItteration_Train_mean.csv'
    np.savetxt(outputName, allPerfMeasures_train_AllSubj_mean, delimiter=",")
    outputName = folder + '/AllSubj_PerformanceMeasuresPerItteration_Train_std.csv'
    np.savetxt(outputName, allPerfMeasures_train_AllSubj_std, delimiter=",")
    outputName = folder + '/AllSubj_PerformanceMeasuresPerItteration_Test_mean.csv'
    np.savetxt(outputName, allPerfMeasures_test_AllSubj_mean, delimiter=",")
    outputName = folder + '/AllSubj_PerformanceMeasuresPerItteration_Test_std.csv'
    np.savetxt(outputName, allPerfMeasures_test_AllSubj_std, delimiter=",")
    outputName = folder + '/AllSubj_VariousMeasuresPerItteration_mean.csv'
    np.savetxt(outputName, variousPerfMeasures_AllSubj_mean, delimiter=",")
    outputName = folder + '/AllSubj_VariousMeasuresPerItteration_std.csv'
    np.savetxt(outputName, variousPerfMeasures_AllSubj_std, delimiter=",")

    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    fig1.suptitle('Removing subclasses - All Subj ')
    # xValues=np.arange(0,1+1/numSteps,1/numSteps)
    xValues = 1 - percDataSteps  # percentage of data removed or clustered
    # numClasses kept
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xValues, variousPerfMeasures_AllSubj_mean[:, 0], yerr=variousPerfMeasures_AllSubj_std[:, 0], fmt='k-.',label='NonSeiz')
    ax1.errorbar(xValues, variousPerfMeasures_AllSubj_mean[:, 1], yerr=variousPerfMeasures_AllSubj_std[:, 1], fmt='m-.', label='Seiz')
    ax1.errorbar(xValues, variousPerfMeasures_AllSubj_mean[:, 2], yerr=variousPerfMeasures_AllSubj_std[:, 2], fmt='b-.', label='Both')
    ax1.legend()
    ax1.set_xlabel('Percentage of data removed')
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Num subclasses kept')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.errorbar(xValues, variousPerfMeasures_AllSubj_mean[:, 3], yerr=variousPerfMeasures_AllSubj_std[:, 3], fmt='k-.', label='NonSeiz')
    ax1.errorbar(xValues, variousPerfMeasures_AllSubj_mean[:, 4], yerr=variousPerfMeasures_AllSubj_std[:, 4], fmt='m-.', label='Seiz')
    ax1.errorbar(xValues, variousPerfMeasures_AllSubj_mean[:, 5], yerr=variousPerfMeasures_AllSubj_std[:, 5], fmt='b-.', label='Both')
    ax1.legend()
    ax1.set_xlabel('Percentage of data removed')
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Percentage of data in subclasses')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 2])
    ax1.errorbar(xValues, variousPerfMeasures_AllSubj_mean[:, 7], yerr=variousPerfMeasures_AllSubj_std[:, 7], fmt='k-.', label='CorrClass')
    ax1.errorbar(xValues, variousPerfMeasures_AllSubj_mean[:, 8], yerr=variousPerfMeasures_AllSubj_std[:, 8], fmt='m-.',label='WrongClass')
    ax1.errorbar(xValues, variousPerfMeasures_AllSubj_mean[:, 6], yerr=variousPerfMeasures_AllSubj_std[:, 6], fmt='b-.', label='Misclassified')
    ax1.legend()
    ax1.set_xlabel('Percentage of data removed')
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Distances ')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 0])
    ax1.errorbar(xValues, allPerfMeasures_train_AllSubj_mean[:, 3], yerr=allPerfMeasures_train_AllSubj_std[:, 3], fmt='k-.', label='NoSmooth')
    ax1.errorbar(xValues, allPerfMeasures_train_AllSubj_mean[:, 12], yerr=allPerfMeasures_train_AllSubj_std[:, 12], fmt='b-.', label='Step1')
    ax1.errorbar(xValues, allPerfMeasures_train_AllSubj_mean[:, 21], yerr=allPerfMeasures_train_AllSubj_std[:, 21], fmt='m-.', label='Step2')
    ax1.legend()
    ax1.set_xlabel('Percentage of data removed')
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1E -Train ')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 1])
    ax1.errorbar(xValues, allPerfMeasures_train_AllSubj_mean[:, 6], yerr=allPerfMeasures_train_AllSubj_std[:, 6],  fmt='k-.', label='NoSmooth')
    ax1.errorbar(xValues, allPerfMeasures_train_AllSubj_mean[:, 15], yerr=allPerfMeasures_train_AllSubj_std[:, 15], fmt='b-.', label='Step1')
    ax1.errorbar(xValues, allPerfMeasures_train_AllSubj_mean[:, 24], yerr=allPerfMeasures_train_AllSubj_std[:, 24],  fmt='m-.', label='Step2')
    ax1.legend()
    ax1.set_xlabel('Percentage of data removed')
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1D -Train ')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 2])
    ax1.errorbar(xValues, allPerfMeasures_train_AllSubj_mean[:, 8], yerr=allPerfMeasures_train_AllSubj_std[:, 8], fmt='k-.', label='NoSmooth')
    ax1.errorbar(xValues, allPerfMeasures_train_AllSubj_mean[:, 17], yerr=allPerfMeasures_train_AllSubj_std[:, 17], fmt='b-.', label='Step1')
    ax1.errorbar(xValues, allPerfMeasures_train_AllSubj_mean[:, 26], yerr=allPerfMeasures_train_AllSubj_std[:, 26],fmt='m-.', label='Step2')
    ax1.legend()
    ax1.set_xlabel('Percentage of data removed')
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1both -Train ')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[2, 0])
    ax1.errorbar(xValues, allPerfMeasures_test_AllSubj_mean[:, 3], yerr=allPerfMeasures_test_AllSubj_std[:, 3], fmt='k-.', label='NoSmooth')
    ax1.errorbar(xValues, allPerfMeasures_test_AllSubj_mean[:, 12], yerr=allPerfMeasures_test_AllSubj_std[:, 12],  fmt='b-.', label='Step1')
    ax1.errorbar(xValues, allPerfMeasures_test_AllSubj_mean[:, 21], yerr=allPerfMeasures_test_AllSubj_std[:, 21], fmt='m-.', label='Step2')
    ax1.legend()
    ax1.set_xlabel('Percentage of data removed')
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1E -Test ')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[2, 1])
    ax1.errorbar(xValues, allPerfMeasures_test_AllSubj_mean[:, 6], yerr=allPerfMeasures_test_AllSubj_std[:, 6],  fmt='k-.', label='NoSmooth')
    ax1.errorbar(xValues, allPerfMeasures_test_AllSubj_mean[:, 15], yerr=allPerfMeasures_test_AllSubj_std[:, 15], fmt='b-.', label='Step1')
    ax1.errorbar(xValues, allPerfMeasures_test_AllSubj_mean[:, 24], yerr=allPerfMeasures_test_AllSubj_std[:, 24], fmt='m-.', label='Step2')
    ax1.legend()
    ax1.set_xlabel('Percentage of data removed')
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1D -Test ')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[2, 2])
    ax1.errorbar(xValues, allPerfMeasures_test_AllSubj_mean[:, 8], yerr=allPerfMeasures_test_AllSubj_std[:, 8],  fmt='k-.', label='NoSmooth')
    ax1.errorbar(xValues, allPerfMeasures_test_AllSubj_mean[:, 17], yerr=allPerfMeasures_test_AllSubj_std[:, 17],  fmt='b-.', label='Step1')
    ax1.errorbar(xValues, allPerfMeasures_test_AllSubj_mean[:, 26], yerr=allPerfMeasures_test_AllSubj_std[:, 26],fmt='m-.', label='Step2')
    ax1.legend()
    ax1.set_xlabel('Percentage of data removed')
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1both -Test ')
    ax1.grid()
    fig1.show()
    fig1.savefig(folder + '/AllSubj_ItterativeRemovinAllRes.png')
    plt.close(fig1)


def func_plotWhenItterativelyRemovingSubclasses_forMultiClassPaper(folderInRemov, folderInClust, folderOut, GeneralParams, numSteps):

    #if havent caluclate mean for all subj calculate
    if (os.path.exists(folderInRemov + '/AllSubj_PerformanceMeasuresPerItteration_Train_mean.csv')==0):
        func_calculateMean_andPlotPerSubj_RemovingClusteringSublasses(folderInRemov, numSteps, GeneralParams)
        func_calculateMean_andPlotPerSubj_RemovingClusteringSublasses(folderInClust, numSteps, GeneralParams)

    #read data from removing subclasses
    reader = csv.reader(open(folderInRemov + '/AllSubj_PerformanceMeasuresPerItteration_Train_mean.csv', "r"))
    allPerfMeasuresRemov_train_AllSubj_mean = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInRemov + '/AllSubj_PerformanceMeasuresPerItteration_Train_std.csv', "r"))
    allPerfMeasuresRemov_train_AllSubj_std = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInRemov + '/AllSubj_PerformanceMeasuresPerItteration_Test_mean.csv', "r"))
    allPerfMeasuresRemov_test_AllSubj_mean = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInRemov + '/AllSubj_PerformanceMeasuresPerItteration_Test_std.csv', "r"))
    allPerfMeasuresRemov_test_AllSubj_std = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInRemov + '/AllSubj_VariousMeasuresPerItteration_mean.csv', "r"))
    variousPerfMeasuresRemov_AllSubj_mean = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInRemov + '/AllSubj_VariousMeasuresPerItteration_std.csv', "r"))
    variousPerfMeasuresRemov_AllSubj_std = np.array(list(reader)).astype("float")
    #read data from clustering subclasses
    reader = csv.reader(open(folderInClust + '/AllSubj_PerformanceMeasuresPerItteration_Train_mean.csv', "r"))
    allPerfMeasuresClust_train_AllSubj_mean = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInClust + '/AllSubj_PerformanceMeasuresPerItteration_Train_std.csv', "r"))
    allPerfMeasuresClust_train_AllSubj_std = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInClust + '/AllSubj_PerformanceMeasuresPerItteration_Test_mean.csv', "r"))
    allPerfMeasuresClust_test_AllSubj_mean = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInClust + '/AllSubj_PerformanceMeasuresPerItteration_Test_std.csv', "r"))
    allPerfMeasuresClust_test_AllSubj_std = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInClust + '/AllSubj_VariousMeasuresPerItteration_mean.csv', "r"))
    variousPerfMeasuresClust_AllSubj_mean = np.array(list(reader)).astype("float")
    reader = csv.reader(open(folderInClust + '/AllSubj_VariousMeasuresPerItteration_std.csv', "r"))
    variousPerfMeasuresClust_AllSubj_std = np.array(list(reader)).astype("float")

    #plot number of subclasses and percentage of data how is dropping
    fig1 = plt.figure(figsize=(16,8), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig1)
    fig1.subplots_adjust(wspace=0.1, hspace=0.2)
    #fig1.suptitle('Removing subclasses - All Subj ')
    numSteps=len(allPerfMeasuresRemov_train_AllSubj_mean[:, 0])
    xValues = np.arange(0, 1 , 1 / numSteps)
    # numClasses kept
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 0], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 0], fmt='b-.', label='NonSeiz')
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 1], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 1], fmt='m-.', label='Seiz')
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 2], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 2], fmt='k-.', label='Both')
    ax1.legend(fontsize=14)
    ax1.set_xlabel('Percentage of sub-classes removed/clustered', fontsize=16)
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Num subclasses kept', fontsize=20)
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 3], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 3], fmt='b-.', label='NonSeiz')
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 4], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 4], fmt='m-.',label='Seiz')
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 5], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 5], fmt='k-.', label='Both')
    ax1.legend(fontsize=14)
    ax1.set_xlabel('Percentage of sub-classes removed/clustered', fontsize=16)
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Percentage of data kept', fontsize=20)
    ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/AllSubj_ItterativeRemovinAllRes_Numsubclasses_forMultiClassPaper.png', bbox_inches='tight')
    plt.close(fig1)

    #plotting performance for both removal and clustering
    fig1 = plt.figure(figsize=(16,12), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    #fig1.suptitle('Removing subclasses - All Subj ')
    xValues=np.arange(0,1,1/numSteps)
    # REMOVAL
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 3], yerr=allPerfMeasuresRemov_train_AllSubj_std[:,3], fmt='k', label='Train NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 12], yerr=allPerfMeasuresRemov_train_AllSubj_std[:,12], fmt='b', label='Train Step1')
    #ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 21], yerr=allPerfMeasuresRemov_train_AllSubj_std[:,21], fmt='m-.', label='Train Step2')
    ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 3], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 3], fmt='k-.', label='Test NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 12], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 13], fmt='b-.', label='Test Step1')
    #ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 21], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 21], fmt='m-.', label='Test Step2')
    ax1.legend( loc='lower left', fontsize=14)
    #ax1.set_xlabel('Percentage of data removed', fontsize=16)
    ax1.set_ylabel('Femoving sub-classes', fontsize=16)
    ax1.set_title('Performance F1E ', fontsize=20)
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 6], yerr=allPerfMeasuresRemov_train_AllSubj_std[:, 6], fmt='k', label='Train NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 15], yerr=allPerfMeasuresRemov_train_AllSubj_std[:, 15], fmt='b', label='Train Step1')
    #ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 24], yerr=allPerfMeasuresRemov_train_AllSubj_std[:, 24], fmt='m', label='Train Step2')
    ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 6], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 6], fmt='k-.', label='Test NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 15], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 15], fmt='b-.', label='Test Step1')
    #ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 24], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 24], fmt='m-.', label='Test Step2')
    #ax1.legend( loc='lower left')
    #ax1.set_xlabel('Percentage of data removed', fontsize=16)
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1D', fontsize=20)
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 2])
    ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 8], yerr=allPerfMeasuresRemov_train_AllSubj_std[:, 8], fmt='k', label='Train NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 17], yerr=allPerfMeasuresRemov_train_AllSubj_std[:, 17], fmt='b', label='Train Step1')
    #ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 26], yerr=allPerfMeasuresRemov_train_AllSubj_std[:, 26], fmt='m', label='Train Step2')
    ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 8], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 8], fmt='k-.', label='Test NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 17], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 17], fmt='b-.', label='Test Step1')
    #ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 26], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 26], fmt='m-.', label='Test Step2')
    #ax1.legend( loc='lower left')
    #ax1.set_xlabel('Percentage of data removed', fontsize=16)
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1both  ', fontsize=20)
    ax1.grid()
    # CLUSTERING
    numSteps=len(allPerfMeasuresClust_train_AllSubj_mean[:, 0])
    xValues = np.arange(0, 1 , 1 / numSteps)
    ax1 = fig1.add_subplot(gs[1, 0])
    ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 3], yerr=allPerfMeasuresClust_train_AllSubj_std[:,3], fmt='k', label='Train NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 12], yerr=allPerfMeasuresClust_train_AllSubj_std[:,12], fmt='b', label='Train Step1')
    #ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 21], yerr=allPerfMeasuresClust_train_AllSubj_std[:,21], fmt='m-.', label='Train Step2')
    ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 3], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 3], fmt='k-.', label='Test NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 12], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 13], fmt='b-.', label='Test Step1')
    #ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 21], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 21], fmt='m-.', label='Test Step2')
    ax1.legend( loc='lower left', fontsize=14)
    ax1.set_xlabel('Percentage of data clustered', fontsize=16)
    ax1.set_ylabel('Clustering sub-clusses', fontsize=16)
    ax1.set_title('Performance F1E ', fontsize=20)
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 1])
    ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 6], yerr=allPerfMeasuresClust_train_AllSubj_std[:, 6], fmt='k', label='Train NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 15], yerr=allPerfMeasuresClust_train_AllSubj_std[:, 15], fmt='b', label='Train Step1')
    #ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 24], yerr=allPerfMeasuresClust_train_AllSubj_std[:, 24], fmt='m', label='Train Step2')
    ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 6], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 6], fmt='k-.', label='Test NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 15], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 15], fmt='b-.', label='Test Step1')
    #ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 24], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 24], fmt='m-.', label='Test Step2')
    #ax1.legend( loc='lower left')
    ax1.set_xlabel('Percentage of data clustered', fontsize=16)
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1D', fontsize=20)
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 2])
    ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 8], yerr=allPerfMeasuresClust_train_AllSubj_std[:, 8], fmt='k', label='Train NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 17], yerr=allPerfMeasuresClust_train_AllSubj_std[:, 17], fmt='b', label='Train Step1')
    #ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 26], yerr=allPerfMeasuresClust_train_AllSubj_std[:, 26], fmt='m', label='Train Step2')
    ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 8], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 8], fmt='k-.', label='Test NoSmooth')
    ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 17], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 17], fmt='b-.', label='Test Step1')
    #ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 26], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 26], fmt='m-.', label='Test Step2')
    #ax1.legend( loc='lower left')
    ax1.set_xlabel('Percentage of data clustered', fontsize=16)
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Performance F1both  ', fontsize=20)
    ax1.grid()

    fig1.show()
    fig1.savefig(folderOut + '/AllSubj_ItterativeRemovinAllRes_Performances_forMultiClassPaper.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/AllSubj_ItterativeRemovinAllRes_Performances_forMultiClassPaper.svg',   bbox_inches='tight')
    plt.close(fig1)


    #plot number of subclasses and percentage of data how is dropping and just F1score performances
    fig1 = plt.figure(figsize=(12,8), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig1)
    fig1.subplots_adjust(wspace=0.1, hspace=0.4)
    #fig1.suptitle('Removing subclasses - All Subj ')
    numSteps=len(allPerfMeasuresRemov_train_AllSubj_mean[:, 0])
    xValues = np.arange(0, 1 , 1 / numSteps)
    # numClasses kept
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 0], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 0], fmt='royalblue', label='NonSeiz', linewidth=2)
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 1], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 1], fmt='mediumvioletred', label='Seiz', linewidth=2)
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 2], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 2], fmt='k', label='Both', linewidth=2)
    ax1.legend(fontsize=14)
    ax1.set_xlabel('Percentage removed/clustered', fontsize=16)
    ax1.set_ylabel('Number/percentage', fontsize=16)
    ax1.set_title('Nr. subclasses kept', fontsize=18)
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 3], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 3], fmt='royalblue', label='NonSeiz', linewidth=2)
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 4], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 4], fmt='mediumvioletred',label='Seiz', linewidth=2)
    ax1.errorbar(xValues, variousPerfMeasuresRemov_AllSubj_mean[:, 5], yerr=variousPerfMeasuresRemov_AllSubj_std[:, 5], fmt='k', label='Both', linewidth=2)
    ax1.legend(fontsize=14)
    ax1.set_xlabel('Percentage removed/clustered', fontsize=16)
    # ax1.set_ylabel('Num subclasses kept')
    ax1.set_title('Percentage of data kept', fontsize=18)
    ax1.grid()
    # ax1 = fig1.add_subplot(gs[1, 0])
    # ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 8], yerr=allPerfMeasuresRemov_train_AllSubj_std[:, 8], fmt='k', label='Train NoSmooth')
    # ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 17], yerr=allPerfMeasuresRemov_train_AllSubj_std[:, 17], fmt='b', label='Train Step1')
    # #ax1.errorbar(xValues, allPerfMeasuresRemov_train_AllSubj_mean[:, 26], yerr=allPerfMeasuresRemov_train_AllSubj_std[:, 26], fmt='m', label='Train Step2')
    # ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 8], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 8], fmt='k-.', label='Test NoSmooth')
    # ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 17], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 17], fmt='b-.', label='Test Step1')
    # #ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 26], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 26], fmt='m-.', label='Test Step2')
    # ax1.legend( loc='lower left', fontsize=14)
    # ax1.set_xlabel('Percentage of sub-classes removed', fontsize=16)
    # ax1.set_ylabel('F1DE gmean', fontsize=16)
    # ax1.set_title('Performance - Removing ', fontsize=20)
    # ax1.set_ylim([0.1, 1])
    # ax1.grid()
    # ax1 = fig1.add_subplot(gs[1, 1])
    # ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 8], yerr=allPerfMeasuresClust_train_AllSubj_std[:, 8], fmt='k', label='Train NoSmooth')
    # ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 17], yerr=allPerfMeasuresClust_train_AllSubj_std[:, 17], fmt='b', label='Train Step1')
    # #ax1.errorbar(xValues, allPerfMeasuresClust_train_AllSubj_mean[:, 26], yerr=allPerfMeasuresClust_train_AllSubj_std[:, 26], fmt='m', label='Train Step2')
    # ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 8], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 8], fmt='k-.', label='Test NoSmooth')
    # ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 17], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 17], fmt='b-.', label='Test Step1')
    # #ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 26], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 26], fmt='m-.', label='Test Step2')
    # #ax1.legend( loc='lower left', fontsize=14)
    # ax1.set_xlabel('Percentage of sub-classes clustered', fontsize=16)
    # #ax1.set_ylabel('F1DE gmean', fontsize=16)
    # ax1.set_ylim([0.1, 1])
    # ax1.set_title('Performance - Clustering', fontsize=20)
    # ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 0])
    ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 8], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 8], fmt='gray', label='Test NoSmooth', linewidth=2)
    ax1.errorbar(xValues, allPerfMeasuresRemov_test_AllSubj_mean[:, 17], yerr=allPerfMeasuresRemov_test_AllSubj_std[:, 17], fmt='k', label='Test Smooth', linewidth=2)
    ax1.legend( loc='lower left', fontsize=14)
    ax1.set_xlabel('Percentage of sub-classes removed', fontsize=16)
    ax1.set_ylabel('F1DE gmean', fontsize=16)
    ax1.set_title('Performance - Removing ', fontsize=18)
    ax1.set_ylim([0.1, 1])
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 1])
    ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 8], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 8], fmt='gray', label='Test NoSmooth', linewidth=2)
    ax1.errorbar(xValues, allPerfMeasuresClust_test_AllSubj_mean[:, 17], yerr=allPerfMeasuresClust_test_AllSubj_std[:, 17], fmt='k', label='Test Smooth', linewidth=2)
    #ax1.legend( loc='lower left', fontsize=14)
    ax1.set_xlabel('Percentage of sub-classes clustered', fontsize=16)
    #ax1.set_ylabel('F1DE gmean', fontsize=16)
    ax1.set_ylim([0.1, 1])
    ax1.set_title('Performance - Clustering', fontsize=18)
    ax1.grid()

    fig1.show()
    fig1.savefig(folderOut + '/AllSubj_ItterativeRemovinClustering_AllResSummarized_forMultiClassPaper.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/AllSubj_ItterativeRemovinClustering_AllResSummarized_forMultiClassPaper.svg',   bbox_inches='tight')
    plt.close(fig1)



def func_plotAllPerformancesForManyApproaches(dataToPlotMean_train, dataToPlotMean_test, xLabNames, folderOut):
    '''function that plots comparison in performance between differnt approaches '''
    xValues = np.arange(0,len(xLabNames),1)

    # plotting all perf - 9 of them
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    fig1.suptitle('All subj different performances - 2 vs multi class learning ')
    indexsesArray = [6, 7, 8, 9, 10, 11, 12, 13, 14]
    perfNames = ['sensE', 'precisE', 'F1E', 'sensD', 'precisD', 'F1D', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
    for indx, i in enumerate(indexsesArray):
        # plotting total mean
        ax1 = fig1.add_subplot(gs[int(np.floor(indx / 3)), int(np.mod(indx, 3))])
        #train
        ax1.errorbar(xValues, dataToPlotMean_train[0, i,:], yerr=dataToPlotMean_train[1, i,:], fmt='k', label='NoSmooth')
        ax1.errorbar(xValues, dataToPlotMean_train[0, i+9,:], yerr=dataToPlotMean_train[1, i+9,:], fmt='b', label='Step1')
        ax1.errorbar(xValues, dataToPlotMean_train[0, i+18,:], yerr=dataToPlotMean_train[1, i+18,:], fmt='m', label='Step2')
        #test
        ax1.errorbar(xValues, dataToPlotMean_test[0, i,:], yerr=dataToPlotMean_test[1, i,:], fmt='k--', label='NoSmooth')
        ax1.errorbar(xValues, dataToPlotMean_test[0, i+9,:], yerr=dataToPlotMean_test[1, i+9,:], fmt='b--', label='Step1')
        ax1.errorbar(xValues, dataToPlotMean_test[0, i+18,:], yerr=dataToPlotMean_test[1, i+18,:], fmt='m--', label='Step2')
        ax1.set_xticks(np.arange(len(xValues)))
        ax1.set_xticklabels(xLabNames)
        ax1.set_ylabel(perfNames[indx])
        ax1.legend()
        ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/AllSubj_AllPerfMeasures_9Values.png', bbox_inches='tight')
    plt.close(fig1)

    # plotting all perf - 9 of them
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    fig1.suptitle('All subj different performances - 2 vs multi class learning ')
    indexsesArray = [0, 1, 2, 13, 22, 31]
    perfNames = ['numClassSeiz', 'numClassNonSeiz', 'SimplAcc', 'F1DEnoSmooth', 'F1DEstep1', 'F1DEstep2']
    for indx, i in enumerate(indexsesArray):
        # plotting total mean
        ax1 = fig1.add_subplot(gs[int(np.floor(indx / 3)), int(np.mod(indx, 3))])
        ax1.errorbar(xValues, dataToPlotMean_train[0, i,:], yerr=dataToPlotMean_train[1, i,:], fmt='k', label='Train')
        ax1.errorbar(xValues, dataToPlotMean_test[0, i,:], yerr=dataToPlotMean_test[1, i,:], fmt='b', label='Test')
        ax1.set_xticks(np.arange(len(xValues)))
        ax1.set_xticklabels(xLabNames)
        ax1.set_ylabel(perfNames[indx])
        ax1.legend()
        ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/AllSubj_AllPerfMeasures_MainValues.png', bbox_inches='tight')
    plt.close(fig1)

def train_StandardML_moreModelsPossible(X_train, y_train,  StandardMLParams):
    ''' functions that has many possible standard ML approaches that can be used to train
     exact model and its parameters are defined in StandardMLParams
     output is trained model'''

    if (np.size(X_train,0)==0):
            print('X train size 0 is 0!!', X_train.shape, y_train.shape)
    if (np.size(X_train,1)==0):
            print('X train size 1 is 0!!', X_train.shape, y_train.shape)
    col_mean = np.nanmean(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_mean, inds[1])
    # if still somewhere nan replace with 0
    X_train[np.where(np.isnan(X_train))] = 0
    X_train=X_train

    #MLmodels.modelType = 'KNN'
    if (StandardMLParams.modelType=='KNN'):
        model = KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric)
        model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='SVM'):
        model = svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma)
        model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='DecisionTree'):
        if (StandardMLParams.DecisionTree_max_depth==0):
            model = DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter)
        else:
            model = DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter,  max_depth=StandardMLParams.DecisionTree_max_depth)
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='RandomForest'):
        if (StandardMLParams.DecisionTree_max_depth == 0):
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.RandomForest_n_estimators, criterion=StandardMLParams.DecisionTree_criterion, min_samples_leaf=10 )
        else:
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.RandomForest_n_estimators, criterion=StandardMLParams.DecisionTree_criterion,  max_depth=StandardMLParams.DecisionTree_max_depth, min_samples_leaf=10 )
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='BaggingClassifier'):
        if (StandardMLParams.Bagging_base_estimator=='SVM'):
            model = BaggingClassifier(base_estimator=svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif  (StandardMLParams.Bagging_base_estimator=='KNN'):
            model = BaggingClassifier(base_estimator= KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif (StandardMLParams.Bagging_base_estimator == 'DecisionTree'):
            model = BaggingClassifier(DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter),
                n_estimators=StandardMLParams.Bagging_n_estimators, random_state=0)
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='AdaBoost'):
        if (StandardMLParams.Bagging_base_estimator=='SVM'):
            model = AdaBoostClassifier(base_estimator=svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif  (StandardMLParams.Bagging_base_estimator=='KNN'):
            model = AdaBoostClassifier(base_estimator= KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif (StandardMLParams.Bagging_base_estimator == 'DecisionTree'):
            model = AdaBoostClassifier(DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter),
                n_estimators=StandardMLParams.Bagging_n_estimators, random_state=0)
        model = model.fit(X_train, y_train)

    return (model)


def test_StandardML_moreModelsPossible_v2(data, model):
    '''function that based on given dana and model calculates predictions and its probabilities '''

    # clean input data
    X_test0 = np.float32(data)
    X_test0[np.where(np.isinf(X_test0))] = np.nan
    if (np.size(X_test0, 0) == 0):
        print('X test size 0 is 0!!', X_test0.shape)
    if (np.size(X_test0, 1) == 0):
        print('X test size 1 is 0!!', X_test0.shape)
    col_mean = np.nanmean(X_test0, axis=0)
    inds = np.where(np.isnan(X_test0))
    X_test0[inds] = np.take(col_mean, inds[1])
    # if still somewhere nan replace with 0
    X_test0[np.where(np.isnan(X_test0))] = 0
    X_test = X_test0

    # calculate predictions
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)

    # pick only probability of predicted class
    y_probability_fin = np.zeros(len(y_pred))
    indx = np.where(y_pred == 1)
    y_probability_fin[indx] = y_probability[indx, 1]
    indx = np.where(y_pred == 0)
    y_probability_fin[indx] = y_probability[indx, 0]

    return (y_pred, y_probability_fin)


def func_plotPerformancesOfDiffApproaches_thisSubj( pat, trainTestName,  performancessAll, folderOut, ApproachName, AppShortNames, AppLineStyle):
    ''' function that plots performance for different approaches for one subject '''
    (numCV, nc, numApp)=performancessAll.shape

    # PLOT PERFORMANCE FOR ALL APPROACHES
    fontSizeNum = 20
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    fig1.suptitle('Subj ' +pat + ' ' + trainTestName)
    xValues = np.arange(0, numCV, 1)
    perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
                 'Precision duration', 'F1score duration', 'F1DEgeoMean', 'simplAcc', 'numFPperDay']
    perfIndxes =[6,7,8,9,10,11,13,2,14]
    for perfIndx, perf in enumerate(perfIndxes):
        ax1 = fig1.add_subplot(gs[int(np.floor(perfIndx / 3)), np.mod(perfIndx, 3)])
        for appIndx, appName in enumerate(ApproachName):
            ax1.plot(xValues, performancessAll[:, perf, appIndx], AppLineStyle[appIndx])
        ax1.legend(AppShortNames)
        ax1.set_xlabel('CVs')
        ax1.set_ylabel('Performance')
        ax1.set_title(perfNames[perfIndx])
        ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/Subj' +pat + 'AllPerfForDiffApproaches_'+trainTestName)
    plt.close(fig1)

def plot_perfHDvsRF_onlyTest(folderIn, folderOut,  subjects):
    '''plot avarage performance of all subjects for ransom forest vs baselin HD approach'''
    numSubj=len(subjects)

    #load perofrmances per subj for all three approaches
    fileNameIn = folderIn + '/AllSubj_RandForest_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_RF_Train = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_StandardLearning_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_2C_Train  = np.array(list(reader)).astype("float")

    fileNameIn = folderIn + '/AllSubj_RandForest_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_RF_Test= np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_StandardLearning_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_2C_Test  = np.array(list(reader)).astype("float")

    perfNames = ['TPR', 'PPV', 'F1']
    #EPISODES
    PerfIndxs=[24,25,26]
    for t, tIndx in enumerate(PerfIndxs):
        dataAppend = np.vstack((Perf_RF_Test[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('RF', numSubj))).transpose()
        if (t==0):
            AllPerfAllSubj_E = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
        else:
            AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack( (Perf_2C_Test[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('HD', numSubj))).transpose()
        AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))

    #DURATION
    PerfIndxs=[27,28,29]
    for t, tIndx in enumerate(PerfIndxs):
        dataAppend = np.vstack((Perf_RF_Test[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('RF', numSubj))).transpose()
        if (t==0):
            AllPerfAllSubj_D = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
        else:
            AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack( (Perf_2C_Test[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('HD', numSubj))).transpose()
        AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))

    # PLOTTING
    AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'])
    AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'])

    fig1 = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = GridSpec(2,1, figure=fig1)
    major_ticks = np.arange(0, 1, 0.1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.4)
    ax2 = fig1.add_subplot(gs[0,0])
    sns.boxplot(x='Measure', y='Performance', width=0.3, hue='Approach', data=AllPerfAllSubj_E, palette="Set1")
    ax2.set_title('Episode level performance')
    ax2.legend(loc='lower left')
    ax2.grid(which='both')
    ax2 = fig1.add_subplot(gs[1,0])
    sns.set_theme(style="whitegrid")
    ax2.grid(which='both')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)
    sns.boxplot(x='Measure', y='Performance', width=0.3, hue='Approach', data=AllPerfAllSubj_D, palette="Set1")
    ax2.set_title('Duration performance')
    ax2.legend(loc='lower left')
    fig1.show()
    fig1.savefig(folderOut + '/RFvsHD_Allsubj_Performance_onlyTest.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/RFvsHD_Allsubj_Performance_onlyTest.svg', bbox_inches='tight')
    plt.close(fig1)


def plotItterativeApproachGraphs_violin_onlyTest(folderIn, folderOut):
    ''' plot comparison of average performance of all subjects for itterative learning HD approach '''

    # load mean number of itterations and perc readded per subject
    fileNameIn = folderIn + '/AllSubj_Itterative' + '2C_AddAndSubtract' + '_PercWrongClassified.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PercReadded_2Cpm = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itterative' + '2C_AddOnly' + '_PercWrongClassified.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PercReadded_2Cp = np.array(list(reader)).astype("float")

    fileNameIn = folderIn + '/AllSubj_Itterative' + '2C_AddAndSubtract' + '_NumIter.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    NumItter_2Cpm = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itterative' + '2C_AddOnly' + '_NumIter.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    NumItter_2Cp = np.array(list(reader)).astype("float")

    numSubj = int(len(PercReadded_2Cpm[0, :]))
    # put to data frame
    # dataAppend=np.vstack(( np.zeros((numSubj)), np.repeat('2C',numSubj) )).transpose()
    # AllSubj_numItter = pd.DataFrame(dataAppend, columns=['Number of iterations', 'Approach'])
    dataAppend = np.vstack((NumItter_2Cp[0, :], np.repeat('2Ci+', numSubj))).transpose()
    AllSubj_numItter = pd.DataFrame(dataAppend, columns=['Number of iterations', 'Approach'])
    dataAppend = np.vstack((NumItter_2Cpm[0, :], np.repeat('2Ci+-', numSubj))).transpose()
    AllSubj_numItter = AllSubj_numItter.append(pd.DataFrame(dataAppend, columns=['Number of iterations', 'Approach']))

    # dataAppend=np.vstack(( np.zeros((numSubj)), np.repeat('2C',numSubj) )).transpose()
    # AllSubj_PercDataReaded = pd.DataFrame(dataAppend, columns=['Percentage', 'Approach'])
    dataAppend = np.vstack((PercReadded_2Cp[0, :], np.repeat('2Ci+', numSubj))).transpose()
    AllSubj_PercDataReaded = pd.DataFrame(dataAppend, columns=['Percentage', 'Approach'])
    dataAppend = np.vstack((PercReadded_2Cpm[0, :], np.repeat('2Ci+-', numSubj))).transpose()
    AllSubj_PercDataReaded = AllSubj_PercDataReaded.append(pd.DataFrame(dataAppend, columns=['Percentage', 'Approach']))

    # load perofrmances per subj for all three approaches
    fileNameIn = folderIn + '/AllSubj_StandardLearning_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTrain_2C = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itter2class_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTrain_2Cpm = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itter2classAddOnly_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTrain_2Cp = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_StandardLearning_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest_2C = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itter2class_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest_2Cpm = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itter2classAddOnly_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest_2Cp = np.array(list(reader)).astype("float")

    # put to dataframe
    perfNames = ['F1_E', 'F1_D', 'F1_DE']
    PerfIndxs = [26, 29, 30]
    for t, tIndx in enumerate(PerfIndxs):
        dataAppend = np.vstack(
            (PerfTest_2C[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('2C', numSubj))).transpose()
        if (t == 0):
            AllPerfAllSubj = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
        else:
            AllPerfAllSubj = AllPerfAllSubj.append(
                pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack(
            (PerfTest_2Cp[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('2C+', numSubj))).transpose()
        AllPerfAllSubj = AllPerfAllSubj.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack(
            (PerfTest_2Cpm[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('2C+-', numSubj))).transpose()
        AllPerfAllSubj = AllPerfAllSubj.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))

    # PLOT FOR ALL SUBJ USING VIOLIN PLOT
    AllPerfAllSubj['Performance'] = pd.to_numeric(AllPerfAllSubj['Performance'])
    AllSubj_PercDataReaded['Percentage'] = pd.to_numeric(AllSubj_PercDataReaded['Percentage'])
    AllSubj_numItter['Number of iterations'] = pd.to_numeric(AllSubj_numItter['Number of iterations'])

    fig1 = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig1)
    major_ticks = np.arange(0, 1, 0.1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.4)
    ax1 = fig1.add_subplot(gs[0, 0])
    sns.set_theme(style="whitegrid")
    ax1.grid(axis='y')
    sns.violinplot(x='Approach', y='Number of iterations', data=AllSubj_numItter, palette="Set1", scale="count",
                   inner="box", width=0.5)
    ax1.set_title('Number of iterations')
    ax1 = fig1.add_subplot(gs[0, 1])
    sns.set_theme(style="whitegrid")
    sns.violinplot(x='Approach', y='Percentage', data=AllSubj_PercDataReaded, palette="Set1", scale="count",
                   inner="box", width=0.5)
    ax1.set_title('Percentage of data readded')
    ax1.set_ylim([-0.3, 1])
    major_ticks = np.arange(0, 3, 0.1)
    minor_ticks = np.arange(0, 3, 0.1)
    ax1.grid(which='both')
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    # ax1.grid(which='both')
    ax1 = fig1.add_subplot(gs[1, :])
    sns.set_theme(style="whitegrid")
    sns.boxplot(x='Measure', y='Performance', hue='Approach', data=AllPerfAllSubj, palette="Set1", width=0.5)
    ax1.set_title('F1 score for episodes and duration')
    ax1.set_ylim([0, 1])
    ax1.grid(which='both')
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    fig1.show()
    fig1.savefig(folderOut + '/Itterative_Allsubj_Performance_ViolinPlot_onlyTest.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/Itterative_Allsubj_Performance_ViolinPlot_onlyTest.svg', bbox_inches='tight')
    plt.close(fig1)


def runStatistics_ItterativeApproach(folderIn, trainTest):
    ''' run statistical analysis (Wilcoxon paired test) on diffferent things related to itterative learning '''

    # load mean number of itterations and perc readded per subject
    fileNameIn = folderIn + '/AllSubj_Itterative' + '2C_AddAndSubtract' + '_PercWrongClassified.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PercReadded_2Cpm = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itterative' + '2C_AddOnly' + '_PercWrongClassified.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PercReadded_2Cp = np.array(list(reader)).astype("float")

    fileNameIn = folderIn + '/AllSubj_Itterative' + '2C_AddAndSubtract' + '_NumIter.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    NumItter_2Cpm = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itterative' + '2C_AddOnly' + '_NumIter.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    NumItter_2Cp = np.array(list(reader)).astype("float")

    numSubj = int(len(PercReadded_2Cpm[0, :]))
    NumItter_All = np.zeros((numSubj, 3))
    NumItter_All[:, 1] = NumItter_2Cp[0, :]
    NumItter_All[:, 2] = NumItter_2Cpm[0, :]
    PercReadded_All = np.zeros((numSubj, 3))
    PercReadded_All[:, 1] = PercReadded_2Cp[0, :]
    PercReadded_All[:, 2] = PercReadded_2Cpm[0, :]

    # load perofrmances per subj for all three approaches
    fileNameIn = folderIn + '/AllSubj_StandardLearning_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_2C = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itter2class_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_2Cpm = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itter2classAddOnly_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_2Cp = np.array(list(reader)).astype("float")
    numSubj = int(len(Perf_2C[:, 0]))
    Perf_All = np.ones((numSubj, 7)) * np.nan
    F1EIndx = 26  # for step2
    F1DIndx = 29
    F1DEIndx = 30
    Perf_All[:, 0] = Perf_2C[:, F1EIndx]
    Perf_All[:, 1] = Perf_2Cp[:, F1EIndx]
    Perf_All[:, 2] = Perf_2Cpm[:, F1EIndx]
    Perf_All[:, 4] = Perf_2C[:, F1DIndx]
    Perf_All[:, 5] = Perf_2Cp[:, F1DIndx]
    Perf_All[:, 6] = Perf_2Cpm[:, F1DIndx]

    # WILCOXON TEST
    print('--> WILCOXON:')
    st = scipy.stats.wilcoxon(Perf_2C[:, F1EIndx], Perf_2Cp[:, F1EIndx])
    print('F1E 2C vs 2Cp :', st)
    st = scipy.stats.wilcoxon(Perf_2C[:, F1EIndx], Perf_2Cpm[:, F1EIndx])
    print('F1E 2C vs 2Cpm:', st)
    st = scipy.stats.wilcoxon(Perf_2C[:, F1DIndx], Perf_2Cp[:, F1DIndx])
    print('F1D 2C vs 2Cp :', st)
    st = scipy.stats.wilcoxon(Perf_2C[:, F1DIndx], Perf_2Cpm[:, F1DIndx])
    print('F1D 2C vs 2Cpm:', st)
    st = scipy.stats.wilcoxon(Perf_2C[:, F1DEIndx], Perf_2Cp[:, F1DEIndx])
    print('F1DE 2C vs 2Cp :', st)
    st = scipy.stats.wilcoxon(Perf_2C[:, F1DEIndx], Perf_2Cpm[:, F1DEIndx])
    print('F1DE 2C vs 2Cpm:', st)
    st = scipy.stats.wilcoxon(Perf_2Cp[:, F1DEIndx], Perf_2Cpm[:, F1DEIndx])
    print('F1DE 2Cp vs 2Cpm:', st)

    # print('--> KRUSKAL:')
    # st = scipy.stats.kruskal(Perf_2C[:, F1EIndx], Perf_2Cp[:, F1EIndx], Perf_2Cpm[:, F1EIndx])
    # print('F1E 2C vs 2Cp vs 2Cpm:', st)
    # st = scipy.stats.kruskal(Perf_2C[:, F1DIndx], Perf_2Cp[:, F1DIndx], Perf_2Cpm[:, F1DIndx])
    # print('F1D 2C vs 2Cp vs 2Cpm:', st)
    # st = scipy.stats.kruskal(Perf_2C[:, F1DEIndx], Perf_2Cp[:, F1DEIndx], Perf_2Cpm[:, F1DEIndx])
    # print('F1DE 2C vs 2Cp vs 2Cpm:', st)


def plotMultiClassApproachGraphs_violin_withoutMCc(folderIn, folderOut):
    ''' plot comparison of average performance of all subjects for multiclass learning HD approach '''

    # load mean number of itterations and perc readded per subject
    fileNameIn = folderIn + '/AllSubj_MultiClass_MeanNumberSeizSubclass_perSubj.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    NumSubclasses_seiz = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClass_MeanNumberNonSeizSubclass_perSubj.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    NumSubclasses_nonSeiz = np.array(list(reader)).astype("float")

    # put to data frame
    numSubj = int(len(NumSubclasses_seiz[:, 0]))
    approaches = ['MC', 'MCr', 'MCri']
    dataAppend = np.vstack((np.ones((numSubj)), np.repeat('2C', numSubj), np.repeat('Seiz', numSubj))).transpose()
    AllSubj_numSubclasses = pd.DataFrame(dataAppend, columns=['Number', 'Approach', 'Seiz-NonSeiz'])
    dataAppend = np.vstack((np.ones((numSubj)), np.repeat('2C', numSubj), np.repeat('NonSeiz', numSubj))).transpose()
    AllSubj_numSubclasses = AllSubj_numSubclasses.append(
        pd.DataFrame(dataAppend, columns=['Number', 'Approach', 'Seiz-NonSeiz']))
    for ap in range(len(approaches)):
        if (ap >= 2):
            ap2 = ap - 1
        else:
            ap2 = ap
        dataAppend = np.vstack( (NumSubclasses_seiz[:, ap2], np.repeat(approaches[ap], numSubj), np.repeat('Seiz', numSubj))).transpose()
        AllSubj_numSubclasses = AllSubj_numSubclasses.append(pd.DataFrame(dataAppend, columns=['Number', 'Approach', 'Seiz-NonSeiz']))
        dataAppend = np.vstack((NumSubclasses_nonSeiz[:, ap2], np.repeat(approaches[ap], numSubj), np.repeat('NonSeiz', numSubj))).transpose()
        AllSubj_numSubclasses = AllSubj_numSubclasses.append(pd.DataFrame(dataAppend, columns=['Number', 'Approach', 'Seiz-NonSeiz']))

    # load perofrmances per subj for all three approaches
    fileNameIn = folderIn + '/AllSubj_StandardLearning_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    perf0 = np.array(list(reader)).astype("float")
    numSubj = len(perf0[:, 0])
    PerfTrain = np.zeros((numSubj, 33, 4))
    PerfTest = np.zeros((numSubj, 33, 4))
    PerfTrain[:, :, 0] = perf0
    fileNameIn = folderIn + '/AllSubj_MultiClassLearning_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTrain[:, :, 1] = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassReducedRemov_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTrain[:, :, 2] = np.array(list(reader)).astype("float")
    # fileNameIn = folderIn + '/AllSubj_MultiClassReducedClust_TrainRes_mean.csv'
    # reader = csv.reader(open(fileNameIn, "r"))
    # PerfTrain[:,:,3] = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassRedItterRemov_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTrain[:, :, 3] = np.array(list(reader)).astype("float")
    # fileNameIn = folderIn + '/AllSubj_MultiClassRedItterClust_TrainRes_mean.csv'
    # reader = csv.reader(open(fileNameIn, "r"))
    # PerfTrain[:,:,5]= np.array(list(reader)).astype("float")

    fileNameIn = folderIn + '/AllSubj_StandardLearning_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest[:, :, 0] = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassLearning_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest[:, :, 1] = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassReducedRemov_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest[:, :, 2] = np.array(list(reader)).astype("float")
    # fileNameIn = folderIn + '/AllSubj_MultiClassReducedClust_TestRes_mean.csv'
    # reader = csv.reader(open(fileNameIn, "r"))
    # PerfTest[:,:,3] = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassRedItterRemov_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest[:, :, 3] = np.array(list(reader)).astype("float")
    # fileNameIn = folderIn + '/AllSubj_MultiClassRedItterClust_TestRes_mean.csv'
    # reader = csv.reader(open(fileNameIn, "r"))
    # PerfTest[:,:,5] = np.array(list(reader)).astype("float")

    # put to dataframe
    numSubj = int(len(PerfTrain[:, 0, 0]))
    F1DEIndx = 30
    dataAppend = np.vstack((PerfTrain[:, F1DEIndx, 0], np.repeat('2C', numSubj), np.repeat('Train', numSubj))).transpose()
    AllPerfAllSubj = pd.DataFrame(dataAppend, columns=['Performance', 'Type', 'Train-Test'])
    dataAppend = np.vstack((PerfTest[:, F1DEIndx, 0], np.repeat('2C', numSubj), np.repeat('Test', numSubj))).transpose()
    AllPerfAllSubj = AllPerfAllSubj.append(pd.DataFrame(dataAppend, columns=['Performance', 'Type', 'Train-Test']))
    for t in range(1, 4):
        dataAppend = np.vstack( (PerfTrain[:, F1DEIndx, t], np.repeat(approaches[t - 1], numSubj), np.repeat('Train', numSubj))).transpose()
        AllPerfAllSubj = AllPerfAllSubj.append(pd.DataFrame(dataAppend, columns=['Performance', 'Type', 'Train-Test']))
        dataAppend = np.vstack((PerfTest[:, F1DEIndx, t], np.repeat(approaches[t - 1], numSubj), np.repeat('Test', numSubj))).transpose()
        AllPerfAllSubj = AllPerfAllSubj.append(pd.DataFrame(dataAppend, columns=['Performance', 'Type', 'Train-Test']))

    # PLOT FOR ALL SUBJ USING VIOLIN PLOT
    AllPerfAllSubj['Performance'] = pd.to_numeric(AllPerfAllSubj['Performance'])
    AllSubj_numSubclasses['Number'] = pd.to_numeric(AllSubj_numSubclasses['Number'])

    fig1 = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig1)
    major_ticks = np.arange(0, 1, 0.1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.4)
    ax2 = fig1.add_subplot(gs[0, 0])
    sns.set_theme(style="whitegrid")
    ax2.grid(which='both')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)
    sns.violinplot(x='Approach', y='Number', data=AllSubj_numSubclasses, hue='Seiz-NonSeiz', palette="Set1", split=True,
                   scale="count", inner="box")
    ax2.set_title('Number of subclasses')
    # ax1.grid(which='minor', alpha=0.2)
    # ax1.grid(which='major', alpha=0.5)
    ax1 = fig1.add_subplot(gs[1, 0])
    sns.set_theme(style="whitegrid")
    sns.boxplot(x='Type', y='Performance', hue='Train-Test', data=AllPerfAllSubj, palette="Set1", width=0.5)
    # sns.violinplot(x='Type', y='Performance', hue='Train-Test', data=AllPerfAllSubj, palette="Set1", split=True,  scale="count", inner="box")
    ax1.set_title('F1_DE score for episodes and duration')
    ax1.set_ylim([0, 1.1])
    ax1.grid(which='both')
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    fig1.show()
    fig1.savefig(folderOut + '/MultiClass_Allsubj_Performance_ViolinPlot.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/MultiClass_Allsubj_Performance_ViolinPlot.svg', bbox_inches='tight')
    plt.close(fig1)


def runStatistics_MultiClassApproach(folderIn, trainTest):
    ''' run statistical analysis (Wilcoxon paired test) on diffferent things related to multiclass learning '''

    # load mean number of itterations and perc readded per subject
    fileNameIn = folderIn + '/AllSubj_MultiClass_MeanNumberSeizSubclass_perSubj.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    NumSubclasses_seiz = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClass_MeanNumberNonSeizSubclass_perSubj.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    NumSubclasses_nonSeiz = np.array(list(reader)).astype("float")

    NumSubclasses_seiz_mean = np.mean(NumSubclasses_seiz, 0)
    NumSubclasses_seiz_std = np.std(NumSubclasses_seiz, 0)
    NumSubclasses_nonSeiz_mean = np.mean(NumSubclasses_nonSeiz, 0)
    NumSubclasses_nonSeiz_std = np.std(NumSubclasses_nonSeiz, 0)

    # load perofrmances per subj for all three approaches
    fileNameIn = folderIn + '/AllSubj_StandardLearning_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_2C = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassLearning_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MC = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassReducedRemov_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MCr = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassReducedClust_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MCc = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassRedItterRemov_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MCri = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassRedItterClust_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MCci = np.array(list(reader)).astype("float")

    numSubj = int(len(Perf_2C[:, 0]))
    F1EIndx = 26  # for step2
    Perf_F1E_All = np.ones((numSubj, 6)) * np.nan
    Perf_F1E_All[:, 0] = Perf_2C[:, F1EIndx]
    Perf_F1E_All[:, 1] = Perf_MC[:, F1EIndx]
    Perf_F1E_All[:, 2] = Perf_MCr[:, F1EIndx]
    Perf_F1E_All[:, 3] = Perf_MCc[:, F1EIndx]
    Perf_F1E_All[:, 4] = Perf_MCri[:, F1EIndx]
    Perf_F1E_All[:, 5] = Perf_MCci[:, F1EIndx]
    F1DIndx = 29
    Perf_F1D_All = np.ones((numSubj, 6)) * np.nan
    Perf_F1D_All[:, 0] = Perf_2C[:, F1DIndx]
    Perf_F1D_All[:, 1] = Perf_MC[:, F1DIndx]
    Perf_F1D_All[:, 2] = Perf_MCr[:, F1DIndx]
    Perf_F1D_All[:, 3] = Perf_MCc[:, F1DIndx]
    Perf_F1D_All[:, 4] = Perf_MCri[:, F1DIndx]
    Perf_F1D_All[:, 5] = Perf_MCci[:, F1DIndx]

    F1DEIndx = 30
    # WILCOXON TEST
    print('--> WILCOXON:')
    st = scipy.stats.wilcoxon(Perf_2C[:, F1DEIndx], Perf_MC[:, F1DEIndx])
    print('F1DE 2C vs MC :', st)
    st = scipy.stats.wilcoxon(Perf_2C[:, F1DEIndx], Perf_MCr[:, F1DEIndx])
    print('F1DE 2C vs MCr:', st)
    st = scipy.stats.wilcoxon(Perf_2C[:, F1DEIndx], Perf_MCc[:, F1DEIndx])
    print('F1DE 2C vs MCc :', st)
    st = scipy.stats.wilcoxon(Perf_2C[:, F1DEIndx], Perf_MCri[:, F1DEIndx])
    print('F1DE 2C vs MCri:', st)
    st = scipy.stats.wilcoxon(Perf_2C[:, F1DEIndx], Perf_MCci[:, F1DEIndx])
    print('F1DE 2C vs MCci :', st)

    # print('--> KRUSKAL:')
    # st = scipy.stats.kruskal(Perf_2C[:, F1EIndx], Perf_MC[:, F1EIndx], Perf_MCr[:, F1EIndx], Perf_MCc[:, F1EIndx], Perf_MCri[:, F1EIndx], Perf_MCci[:, F1EIndx])
    # print('F1E 2C vs 2Cp vs 2Cpm:', st)
    # st = scipy.stats.kruskal(Perf_2C[:, F1DIndx], Perf_MC[:, F1DIndx], Perf_MCr[:, F1DIndx], Perf_MCc[:, F1DIndx],Perf_MCri[:, F1DIndx], Perf_MCci[:, F1DIndx])
    # print('F1D 2C vs 2Cp vs 2Cpm:', st)
    # st = scipy.stats.kruskal(Perf_2C[:, F1DEIndx], Perf_MC[:, F1DEIndx], Perf_MCr[:, F1DEIndx], Perf_MCc[:, F1DEIndx], Perf_MCri[:, F1DEIndx], Perf_MCci[:, F1DEIndx])
    # print('F1DE 2C vs 2Cp vs 2Cpm:', st)


def plotOnlineHDApproach_weightsAndPerfomance_onlyTest(folderIn, folderOut, GeneralParams):
    ''' plot weights and average performance of all subjects for onlineHD learning approach '''

    # load perofrmances per subj for all three approaches
    fileNameIn = folderIn + '/AllSubj_StandardLearning_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTrain_2C = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_OnlineHDAdd_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTrain_ONp = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_OnlineHDAddSub_TrainRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTrain_ONpm = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_StandardLearning_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest_2C = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_OnlineHDAdd_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest_ONp = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_OnlineHDAddSub_TestRes_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    PerfTest_ONpm = np.array(list(reader)).astype("float")

    numSubj = len(PerfTrain_2C[:, 0])
    # put to dataframe
    perfNames = ['F1_E', 'F1_D', 'F1_DE']
    PerfIndxs = [26, 29, 30]
    for t, tIndx in enumerate(PerfIndxs):
        dataAppend = np.vstack(
            (PerfTest_2C[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('2C', numSubj))).transpose()
        if (t == 0):
            AllPerfAllSubj = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
        else:
            AllPerfAllSubj = AllPerfAllSubj.append(
                pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack(
            (PerfTest_ONp[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('On+', numSubj))).transpose()
        AllPerfAllSubj = AllPerfAllSubj.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))
        dataAppend = np.vstack(
            (PerfTest_ONpm[:, tIndx], np.repeat(perfNames[t], numSubj), np.repeat('On+-', numSubj))).transpose()
        AllPerfAllSubj = AllPerfAllSubj.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))

    ######################
    for patIndx, pat in enumerate(GeneralParams.patients):
        # read
        reader = csv.reader(open(folderIn + '/OnlineHD/Subj' + pat + '_AvrgWeightsAdd&AddSub.csv', "r"))
        avrgWightVals = np.array(list(reader)).astype("float")
        # put to dataframe
        numFiles = len(avrgWightVals[:, 0])
        subjectArr = np.ones((numFiles * 2)) * (patIndx + 1)
        weightsAppended = np.hstack((avrgWightVals[:, 5], avrgWightVals[:, 4]))  # nonseiz then seiz
        seizLabel = np.repeat('Seiz', 2 * numFiles)  # np.ones((2*numFiles))
        seizLabel[numFiles:] = np.repeat('NonSeiz', numFiles)  # np.zeros((numFiles))
        # dataForPlotting = np.hstack((avrgWightVals[:, 1:3], subjectArr))
        dataForPlotting = np.vstack((weightsAppended, seizLabel, subjectArr)).transpose()
        if (patIndx == 0):
            AllWeightsAllSubj = pd.DataFrame(dataForPlotting, columns=['Weights', 'Seiz-NonSeiz', 'Subjects'])
        else:
            AllWeightsAllSubj = AllWeightsAllSubj.append(
                pd.DataFrame(dataForPlotting, columns=['Weights', 'Seiz-NonSeiz', 'Subjects']))

    # PLOT FOR ALL SUBJ USING BOXPLOT PLOT
    AllPerfAllSubj['Performance'] = pd.to_numeric(AllPerfAllSubj['Performance'])
    fig1 = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.4)
    ax1 = fig1.add_subplot(gs[0, 0])
    sns.set_theme(style="whitegrid")
    AllWeightsAllSubj['Weights'] = AllWeightsAllSubj['Weights'].astype('float')
    AllWeightsAllSubj['Subjects'] = AllWeightsAllSubj['Subjects'].astype('float').astype('int')
    sns.violinplot(x="Subjects", y="Weights", hue="Seiz-NonSeiz", data=AllWeightsAllSubj, palette="Set1", split=True,
                   scale="count", inner="stick")
    ax1.set_title('Weight distribution ')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 0])
    sns.set_theme(style="whitegrid")
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj, palette="Set1")
    ax1.set_title('Performance ')
    ax1.set_ylim([0, 1])
    ax1.grid(which='both')
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    fig1.show()
    fig1.savefig(folderOut + '/OnlineHD_Allsubj_WeightsAndPerformance_onlyTest.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/OnlineHD_Allsubj_WeightsAndPerformance_onlyTest.svg', bbox_inches='tight')
    plt.close(fig1)


def readAllPerformances_forAllApproaches(folderIn, trainTest):
    ''' function to sumarize reading performances for all approaches '''

    # load perofrmances per subj for all three approaches
    fileNameIn = folderIn + '/AllSubj_RandForest_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_RF = np.array(list(reader)).astype("float")

    fileNameIn = folderIn + '/AllSubj_StandardLearning_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_2C = np.array(list(reader)).astype("float")

    fileNameIn = folderIn + '/AllSubj_Itter2class_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_2Cpm = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_Itter2classAddOnly_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_2Cp = np.array(list(reader)).astype("float")

    fileNameIn = folderIn + '/AllSubj_MultiClassLearning_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MC = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassReducedRemov_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MCr = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassReducedClust_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MCc = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassRedItterRemov_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MCri = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_MultiClassRedItterClust_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_MCci = np.array(list(reader)).astype("float")

    fileNameIn = folderIn + '/AllSubj_OnlineHDAdd_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_ONp = np.array(list(reader)).astype("float")
    fileNameIn = folderIn + '/AllSubj_OnlineHDAddSub_' + trainTest + 'Res_mean.csv'
    reader = csv.reader(open(fileNameIn, "r"))
    Perf_ONpm = np.array(list(reader)).astype("float")

    return ( Perf_RF, Perf_2C, Perf_2Cpm, Perf_2Cp, Perf_MC, Perf_MCr, Perf_MCc, Perf_MCri, Perf_MCci, Perf_ONp, Perf_ONpm)

def createDataFramePerformance_onlyTest(data, indx, typeApproach,  numSubj):
    dataAppend = np.vstack( (data[:, indx], np.repeat(typeApproach, numSubj))).transpose()
    dataFrame=pd.DataFrame(dataAppend, columns=['Performance', 'Type'])
    return (dataFrame)

def plotAllApproachGraps_onlyTest(folderIn, folderOut, GeneralParams):
    '''plot comaprison of mean performance on all subects for all different learning approaches sued, only on test data'''
    numSubj=len(GeneralParams.patients)

    #load perofrmances per subj for all approaches
    #(PerfTrain_RF,PerfTrain_2C, PerfTrain_2Cpm, PerfTrain_2Cp, PerfTrain_MC, PerfTrain_MCr,PerfTrain_MCc , PerfTrain_MCri, PerfTrain_MCci, PerfTrain_ONp, PerfTrain_ONpm ) =readAllPerformances_forAllApproaches(folderIn, 'Train')
    (PerfTest_RF, PerfTest_2C, PerfTest_2Cpm, PerfTest_2Cp, PerfTest_MC, PerfTest_MCr, PerfTest_MCc, PerfTest_MCri, PerfTest_MCci, PerfTest_ONp, PerfTest_ONpm) = readAllPerformances_forAllApproaches(folderIn,  'Test')

    # F1DE
    F1DEIndx = 30
    AllPerfAllSubj=createDataFramePerformance_onlyTest(PerfTest_RF, F1DEIndx, 'RF', numSubj)

    AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_2C, F1DEIndx, '2C',  numSubj))
    AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_2Cp, F1DEIndx, '2C+',  numSubj))
    AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_2Cpm, F1DEIndx, '2C+-',  numSubj))

    AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_MC, F1DEIndx, 'MC',  numSubj))
    AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_MCr, F1DEIndx, 'MCr',  numSubj))
    # AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_MCc, F1DEIndx, 'MCc',  numSubj))
    AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_MCri, F1DEIndx, 'MCri', numSubj))
    # AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_MCci, F1DEIndx, 'MCci',  numSubj))

    AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_ONp, F1DEIndx, 'On+',  numSubj))
    AllPerfAllSubj = AllPerfAllSubj.append( createDataFramePerformance_onlyTest(PerfTest_ONpm, F1DEIndx, 'On+-',  numSubj))

    # PLOTTING
    approachNames = ['RF','2C','On+', 'On+-', '2C+', '2C+-', 'MC', 'MCr', 'MCc', 'MCri', 'MCci']
    approachNames = ['RF','2C','On+', 'On+-', '2C+', '2C+-', 'MC', 'MCr', 'MCri']

    # PLOT FOR ALL SUBJ USING BOXPLOT PLOT
    AllPerfAllSubj['Performance'] = pd.to_numeric(AllPerfAllSubj['Performance'])
    fig1 = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.4)
    ax1 = fig1.add_subplot(gs[0, 0])
    sns.set_theme(style="whitegrid")
    sns.boxplot(x='Type', y='Performance', width=0.5,  data=AllPerfAllSubj, palette="Set1")
    ax1.set_ylim([0, 1])
    ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/AllApproaches_Allsubj_Performance_onlyTest.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/AllApproaches_Allsubj_Performance_onlyTest.svg', bbox_inches='tight')
    plt.close(fig1)

    #WILCOXON TEST
    print('--> WILCOXON:')
    st = scipy.stats.wilcoxon(PerfTest_RF[:, F1DEIndx], PerfTest_MCri[:, F1DEIndx])
    print('F1DE Test RF vs MCri:', st)
    st = scipy.stats.wilcoxon(PerfTest_RF[:, F1DEIndx], PerfTest_ONpm[:, F1DEIndx])
    print('F1DE Test RF vs ONpm:', st)
    st = scipy.stats.wilcoxon(PerfTest_MCri[:, F1DEIndx], PerfTest_ONpm[:, F1DEIndx])
    print('F1DE Test MCri vs ONpm:', st)