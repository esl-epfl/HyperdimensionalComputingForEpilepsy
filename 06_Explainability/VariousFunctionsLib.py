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
import antropy as ant
import imblearn


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
    ''' function taht aves data and labels in .csv file'''
    outputName= fileName+'.csv'
    myFile = open(outputName, 'w',newline='')
    dataToWrite=np.column_stack((data, labels))
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(dataToWrite)

def exportNonSeizFile(fileName, folderOut, pat, FileOutIndx, DatasetPreprocessParams, keepOrigNames):
    ''' load .edf file that doesn't contain seizure and export it as gzip file
    each column represents values of one channel and the last column is labels
    1 is seizure and 0 is non seizure
    also channels that are not in interest are removed '''
    allGood = 1
    (rec, samplFreq, channels) = readEdfFile(fileName)
    # take only the channels we need and in correct order
    try:
        chToKeepAndInCorrectOrder = [channels.index(DatasetPreprocessParams.channelNamesToKeep[i]) for i in
                                     range(len(DatasetPreprocessParams.channelNamesToKeep))]
    except:
        print('Sth wrong with the channels in a file: ', fileName)
        allGood = 0
    if (allGood == 1):
        newData = rec[1:, chToKeepAndInCorrectOrder]
        (lenSig, numCh) = newData.shape
        newLabel = np.zeros(lenSig)
        # saving
        if (keepOrigNames==0):
            fileNameOut = folderOut + '/Subj' + pat + '_f' + str(FileOutIndx).zfill(3)
        else:
            fName=os.path.splitext(os.path.split(fileName)[1])[0]
            fileNameOut = folderOut +'/' +fName
        print(fileNameOut)
        saveDataToFile(np.hstack((newData, np.reshape(newLabel, (-1, 1)))), fileNameOut, 'gzip')
        FileOutIndx=FileOutIndx+1
    return(FileOutIndx)

def exportSeizFile(fileName, folderOut, pat, FileOutIndx, DatasetPreprocessParams, keepOrigNames):
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
        chToKeepAndInCorrectOrder = [channels.index(DatasetPreprocessParams.channelNamesToKeep[i]) for i in   range(len(DatasetPreprocessParams.channelNamesToKeep))]
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
            if (keepOrigNames==0):
                fileNameOut = folderOut + '/Subj' + pat + '_f' + str(FileOutIndx).zfill(3) + '_s'
            else:
                fName = os.path.splitext(os.path.split(fileName)[1])[0]
                fileNameOut = folderOut + '/' + fName + '_s'
            print(fileNameOut)
            saveDataToFile(np.hstack((newData, np.reshape(newLabel, (-1, 1)))), fileNameOut, 'gzip')
            FileOutIndx=FileOutIndx+1
    return(FileOutIndx)

def extractEDFdataToCSV_originalData_gzip(folderIn, folderOut, DatasetPreprocessParams, patients, keepOriginalFileNames):
    ''' converts data from edf format to csv using gzip compression
    20210705 UnaPale'''
    createFolderIfNotExists(folderOut)

    for pat in patients:
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

        #EXPORT  FILES
        FileOutIndx=0
        for fileIndx, fileName in enumerate(AllFiles):
            justName = os.path.split(fileName)[1][:-4]
            # pom, fileName1 = os.path.split(fileName)
            # fileName2 = os.path.splitext(fileName1)[0]
            if (justName not in SeizFileNames): #nonseizure file
                # fileNameOut = folderOut + '/Subj'+ pat + '_f'+ str(FileOutIndx).zfill(3)
                FileOutIndx=exportNonSeizFile(fileName, folderOut, pat, FileOutIndx, DatasetPreprocessParams, keepOriginalFileNames)
            else: # seizure file
                # fileNameOut = folderOut + '/Subj'+ pat + '_f'+ str(FileOutIndx).zfill(3) + '_s'
                FileOutIndx= exportSeizFile(fileName, folderOut, pat, FileOutIndx, DatasetPreprocessParams, keepOriginalFileNames)


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

def ham_dist_arr( vec_a, vec_b, D, vecType='bin'):
    ''' calculate relative hamming distance fur for np array'''
    if (vecType=='bin'):
        vec_c= np.abs(vec_a-vec_b)
        rel_dist = np.sum(vec_c) / float(D)
    elif (vecType=='bipol'):
        vec_c= vec_a+vec_b
        rel_dist = np.sum(vec_c==0) / float(D)
    return rel_dist



def train_StandardML_moreModelsPossible(X_train, y_train,  StandardMLParams):
    ''' functions that has many possible standard ML approaches that can be used to train
     exact model and its parameters are defined in StandardMLParams
     output is trained model'''

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
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.RandomForest_n_estimators, criterion=StandardMLParams.DecisionTree_criterion , n_jobs=10) #, min_samples_leaf=10
        else:
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.RandomForest_n_estimators, criterion=StandardMLParams.DecisionTree_criterion,  max_depth=StandardMLParams.DecisionTree_max_depth, n_jobs=10) #, min_samples_leaf=10
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


def saveDataToFile( data,  outputName, type):
    ''' saves data to .csv or .gzip file, depending on type parameter'''
    if ('.csv' not in outputName):
        outputName= outputName+'.csv'
    df = pd.DataFrame(data=data)
    if (type=='gzip'):
        df.to_csv(outputName + '.gz', index=False, compression='gzip')
    else:
        df.to_csv(outputName, index=False)

def readDataFromFile( inputName):
    ''' read data from cs.csv od .gzip file'''
    if not os.path.isfile(inputName):
        inputName=inputName+'.gz'
    #inputName= fileName+'.csv'
    if ('.csv.gz' in inputName):
        df= pd.read_csv(inputName, compression='gzip')
    else:
        df= pd.read_csv(inputName, header=None)
    data=df.to_numpy()
    return (data)

def calculateMovingAvrgMeanWithUndersampling_v2(data, winLen, winStep):
    ''' calculates moving average over data  '''
    lenSig=len(data)
    index = np.arange(0, lenSig - winLen, winStep)

    segmData = np.zeros(len(index))
    for i in range(len(index)): #-1
        x = data[index[i]:index[i] + winLen]
        segmData[i]=np.mean(x)
    return(segmData)

def calculateFeaturesOneDataWindow_Entropy(data,  samplFreq):
    ''' function that calculates entropy  features relevant for epileptic seizure detection
    from paper: D. Sopic, A. Aminifar, and D. Atienza, e-Glass: A Wearable System for Real-Time Detection of Epileptic Seizures, 2018
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

    # #sample entropy
    # samp_1_d7_1 = sampen2(n1, r1 * np.std(d7), d7)
    # samp_1_d6_1 = sampen2(n1, r1 * np.std(d6), d6)
    # samp_2_d7_1 = sampen2(n1, r2 * np.std(d7), d7)
    # samp_2_d6_1 = sampen2(n1, r2 * np.std(d6), d6)

    #sample entropy
    samp_1_d7_1 = ant.sample_entropy(d7)
    samp_1_d6_1 = ant.sample_entropy(d7)
    samp_2_d7_1 = 0
    samp_2_d6_1 = 0

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

    featuresAll= [samp_1_d7_1, samp_1_d6_1, samp_2_d7_1, samp_2_d6_1, perm_d7_3, perm_d7_5, perm_d7_7, perm_d6_3, perm_d6_5, perm_d6_7,   perm_d5_3, perm_d5_5, \
             perm_d5_7, perm_d4_3, perm_d4_5, perm_d4_7, perm_d3_3, perm_d3_5, perm_d3_7, shannon_en_sig, renyi_en_sig, tsallis_en_sig, shannon_en_d7, renyi_en_d7, tsallis_en_d7, \
             shannon_en_d6, renyi_en_d6, tsallis_en_d6, shannon_en_d5, renyi_en_d5, tsallis_en_d5, shannon_en_d4, renyi_en_d4, tsallis_en_d4, shannon_en_d3, renyi_en_d3, tsallis_en_d3]
    return (featuresAll)

def calculateFeaturesOneDataWindow_Frequency(data,  samplFreq):
    ''' function that calculates various frequency features relevant for epileptic seizure detection
    from paper: D. Sopic, A. Aminifar, and D. Atienza, e-Glass: A Wearable System for Real-Time Detection of Epileptic Seizures, 2018
    '''

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
    p_gamma_rel = p_gamma / p_tot

    featuresAll= [p_dc_rel, p_mov_rel, p_delta_rel, p_theta_rel, p_alfa_rel, p_middle_rel, p_beta_rel, p_gamma_rel,
             p_dc, p_mov, p_delta, p_theta, p_alfa, p_middle, p_beta, p_gamma, p_tot]
    return (featuresAll)

def polygonal_approx(arr, epsilon):
    """
    Performs an optimized version of the Ramer-Douglas-Peucker algorithm assuming as an input
    an array of single values, considered consecutive points, and **taking into account only the
    vertical distances**.
    """
    def max_vdist(arr, first, last):
        """
        Obtains the distance and the index of the point in *arr* with maximum vertical distance to
        the line delimited by the first and last indices. Returns a tuple (dist, index).
        """
        if first == last:
            return (0.0, first)
        frg = arr[first:last+1]
        leng = last-first+1
        dist = np.abs(frg - np.interp(np.arange(leng),[0, leng-1], [frg[0], frg[-1]]))
        idx = np.argmax(dist)
        return (dist[idx], first+idx)

    if epsilon <= 0.0:
        raise ValueError('Epsilon must be > 0.0')
    if len(arr) < 3:
        return arr
    result = set()
    stack = [(0, len(arr) - 1)]
    while stack:
        first, last = stack.pop()
        max_dist, idx = max_vdist(arr, first, last)
        if max_dist > epsilon:
            stack.extend([(first, idx),(idx, last)])
        else:
            result.update((first, last))
    return np.array(sorted(result))

def zero_crossings(arr):
    """Returns the positions of zero-crossings in the derivative of an array, as a binary vector"""
    return np.diff(np.sign(np.diff(arr))) != 0

def calulateZCfeaturesRelative_oneCh(sigFilt, DatasetPreprocessParams, FeaturesParams, sigRange):
    ''' feature that calculates zero-cross features for signal of one channel '''
    numFeat=len(FeaturesParams.ZC_thresh_arr)+1
    actualThrValues=np.zeros((numFeat-1))

    '''Zero-crossing of the original signal, counted in 1-second continuous sliding window'''
    # zeroCrossStandard[:,ch] = np.convolve(zero_crossings(sigFilt), np.ones(ZeroCrossFeatures.samplFreq), mode='same')
    x = np.convolve(zero_crossings(sigFilt), np.ones(DatasetPreprocessParams.samplFreq), mode='same')
    # zeroCrossStandard[:,ch] =calculateMovingAvrgMeanWithUndersampling(x, ZeroCrossFeatureParams.samplFreq)
    featVals= calculateMovingAvrgMeanWithUndersampling_v2(x, int(DatasetPreprocessParams.samplFreq * FeaturesParams.winLen), int( DatasetPreprocessParams.samplFreq * FeaturesParams.winStep))
    zeroCrossFeaturesAll = np.zeros((len(featVals), numFeat ))
    zeroCrossFeaturesAll[:, 0]=featVals

    for EPSthrIndx, EPSthr in enumerate(FeaturesParams.ZC_thresh_arr):

        if (FeaturesParams.ZC_tresh_type=='abs'):
            actualThrValues[ EPSthrIndx]=EPSthr
            # Signal simplification at the given threshold, and zero crossing count in the same way
            sigApprox = polygonal_approx(sigFilt, epsilon=EPSthr)#!!!!ABSOLUTE THRESHOLDS
        else:  #relative threshold
            actualThrValues[ EPSthrIndx]=EPSthr*sigRange
            # Signal simplification at the given threshold, and zero crossing count in the same way
            sigApprox = polygonal_approx(sigFilt, epsilon=EPSthr*sigRange)#!!!! NEW TO HAVE RELATIVE THRESHOLDS

        # axs[0].plot(sigApprox, sigFilt[sigApprox], alpha=0.6)
        sigApproxInterp = np.interp(np.arange(len(sigFilt)), sigApprox, sigFilt[sigApprox])
        # zeroCrossApprox[:,ch] = np.convolve(zero_crossings(sigApproxInterp), np.ones(ZeroCrossFeatures.samplFreq), mode='same')
        x = np.convolve(zero_crossings(sigApproxInterp), np.ones(DatasetPreprocessParams.samplFreq),  mode='same')
        # zeroCrossApprox[:, ch] =  calculateMovingAvrgMeanWithUndersampling(x, ZeroCrossFeatureParams.samplFreq)
        zeroCrossFeaturesAll[:,  EPSthrIndx + 1] = calculateMovingAvrgMeanWithUndersampling_v2(x, int(DatasetPreprocessParams.samplFreq * FeaturesParams.winLen), int(DatasetPreprocessParams.samplFreq * FeaturesParams.winStep))

    return(zeroCrossFeaturesAll, actualThrValues)

def calculateMLfeatures_oneCh(X, DatasetPreprocessParams, FeaturesParams, type):
    ''' function that calculate feature of interest for specific signal
    it discretizes signal into windows and calculates feature(s) for each window'''
    segLenIndx = int(FeaturesParams.winLen * DatasetPreprocessParams.samplFreq)  # length of EEG segments in samples
    slidWindStepIndx = int( FeaturesParams.winStep * DatasetPreprocessParams.samplFreq)  # step of slidin window to extract segments in samples
    index = np.arange(0, len(X) - segLenIndx, slidWindStepIndx).astype(int)
    for i in range(len(index)):
        sig = X[index[i]:index[i] + segLenIndx]

        if (type == 'MeanAmpl'):
            featVal = np.mean(np.abs(np.copy(sig)))
            numFeat = 1
        elif (type == 'LineLength'):
            featVal = np.mean(np.abs(np.diff(np.copy(sig))))
            numFeat = 1
        elif (type == 'Entropy'):
            featVal=calculateFeaturesOneDataWindow_Entropy(np.copy(sig), DatasetPreprocessParams.samplFreq)
            numFeat = len(featVal)
        elif (type == 'Frequency'):
            featVal=calculateFeaturesOneDataWindow_Frequency(np.copy(sig), DatasetPreprocessParams.samplFreq)
            numFeat = len(featVal)

        if (i==0):
            featureValues=np.zeros((len(index), numFeat))
        featureValues[i,:]=featVal

    return (featureValues)


def calculateFeaturesPerEachFile_gzip(EPS_thresh_arr,folderIn, folderOutFeatures, GeneralParams, DatasetPreprocessParams, FeaturesParams, sigRanges):
    '''function that loads one by one file, filters data and calculates different features, saves each of them in individual files so that
    later it can be chosen and combined '''


    numFeat = len(EPS_thresh_arr) + 1
    ##butterworth filter initialization
    # sos = signal.butter(4, 20, 'low', fs=ZeroCrossFeatureParams.samplFreq, output='sos')
    #sos = signal.butter(4, [1, 30], 'bandpass', fs=ZeroCrossFeatureParams.samplFreq, output='sos')
    sos = signal.butter(4, [1, 20], 'bandpass', fs=DatasetPreprocessParams.samplFreq, output='sos')

    # go through all patients
    for patIndx, pat in enumerate(GeneralParams.patients):
        filesIn=np.sort(glob.glob(folderIn + '/*Subj' + pat + '*.csv.gz'))
        if (len(filesIn)==0):
            filesIn = np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv.gz'))
        numFiles=len(filesIn)
        print('-- Patient:', pat, 'NumSeizures:', numFiles)

        for fileIndx, fileIn in enumerate(filesIn):
            pom, fileName1 = os.path.split(fileIn)
            fileName2 = fileName1[0:-7]

            # reading data
            data=readDataFromFile(fileIn)
            X = data[:, 0:-1]
            y = data[:, -1]
            (lenData, numCh) = X.shape
            labels = y[0:lenData - 2]
            index = np.arange(0, lenData - int(DatasetPreprocessParams.samplFreq * FeaturesParams.winLen), int(DatasetPreprocessParams.samplFreq * FeaturesParams.winStep))
            labelsSegm = calculateMovingAvrgMeanWithUndersampling_v2(labels, int( DatasetPreprocessParams.samplFreq * FeaturesParams.winLen), int( DatasetPreprocessParams.samplFreq * FeaturesParams.winStep))
            labelsSegm = (labelsSegm > 0.5) * 1


            actualThrValues = np.zeros((numCh, len(EPS_thresh_arr)))
            for fIndx, fName in enumerate(FeaturesParams.featNames):
                print(fName)
                for ch in range(numCh):
                    sig = X[:, ch]
                    sigFilt = signal.sosfiltfilt(sos, sig) #filtering

                    if (fName == 'ZeroCross'):
                        (featVals, actualThrValues[ch,:])=calulateZCfeaturesRelative_oneCh(np.copy(sigFilt), DatasetPreprocessParams, FeaturesParams,sigRanges[int(pat) - 1, ch])
                    else:
                        featVals= calculateMLfeatures_oneCh(np.copy(sigFilt), DatasetPreprocessParams, FeaturesParams, fName)

                    if (ch == 0):
                        AllFeatures = featVals
                    else:
                        AllFeatures = np.hstack((AllFeatures, featVals))

                # save for this file  features and labels
                if (fName == 'ZeroCross'):
                    #save thr values
                    outputName = folderOutFeatures + '/' + fileName2 + '_ZCthreshValues_'+FeaturesParams.ZC_tresh_type+'.csv'
                    saveDataToFile(actualThrValues, outputName, 'gzip')
                    #save feature values
                    outputName = folderOutFeatures + '/' + fileName2 + '_' + fName +'_'+ FeaturesParams.ZC_tresh_type+'.csv'
                    saveDataToFile(AllFeatures, outputName, 'gzip')
                else:
                    outputName = folderOutFeatures + '/' + fileName2 + '_' + fName + '.csv'
                    saveDataToFile(AllFeatures, outputName, 'gzip')

            outputName = folderOutFeatures + '/' + fileName2 + '_Labels.csv'
            saveDataToFile(labelsSegm, outputName, 'gzip')


def calculateStatisticsOnSignalValues(folderIn, folderOut, GeneralParams):
    ''' function that analyses 5 to 95 percentile range of raw data
    but only form first 5h or data (otherwise it would not be fair)
    this values are needed if we use relative Zero-cross feature thresholds'''
    #save only file with first 5h of data
    for patIndx, pat in enumerate(GeneralParams.patients):
        allFiles = np.sort(glob.glob(folderIn + '/Subj' + pat + '*.csv.gz'))
        if (len(allFiles)==0):
            allFiles = np.sort(glob.glob(folderIn + '/chb' + pat + '*.csv.gz'))
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

def loadAndConcatenateAllFeatFilesForThisFile(fileName, FeaturesParams, numCh):
    ''' concatenates files containing different features calculated in a way that output file contains
    first all features for ch1, chen all features for ch2 etc '''
    for featIndx, featName in enumerate(FeaturesParams.featNames):
        fileIn=fileName+ featName+ '.csv.gz'
        if (featIndx==0):
            mat1 = readDataFromFile(fileIn)
        else:
            mat2= readDataFromFile(fileIn)
            mat1= mergeFeatFromTwoMatrixes(mat1, mat2, numCh)
    return (mat1)

def concatenateAllFeatures_moreNonseizureForFactor_gzip(folderIn, folderOut,GeneralParams, DatasetPreprocessParams, FeaturesParams,   factor):
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
            data=loadAndConcatenateAllFeatFilesForThisFile(fileName[0:numLettersToRemove], FeaturesParams, numCh)
            numFeat = int(len(data[0, :]))

            fileName2=fileName[0:numLettersToRemove]+'Labels.csv'
            labels = readDataFromFile(fileName2)

            #find starts and stops of seizures
            diffSig=np.diff(np.squeeze(labels))
            szStart=np.where(diffSig==1)[0]
            szStop= np.where(diffSig == -1)[0]

            # for each seizure cut it out and save
            numSeizures = len(szStart)
            for i in range(numSeizures):
                #prepare where to save new cutout
                seizureLen = int(szStop[i]- szStart[i])
                newLabel = np.zeros(seizureLen * (factor + 1))  # both for seizure nad nonSeizure lavels
                newData = np.zeros((seizureLen * (factor + 1), numFeat))
                #save seizure part
                nonSeizLen = int(factor * seizureLen)
                newData[int(nonSeizLen / 2):int(nonSeizLen / 2) + seizureLen] = data[(szStart[i]): (szStart[i] + seizureLen), :]
                newLabel[int(nonSeizLen / 2):int(nonSeizLen / 2) + seizureLen] = np.ones(seizureLen)

                #LOAD NON SEIZRUE DATA
                for fns in range(IndxNonSeizFile,len(nonSeizFiles)):
                    pom, fileName1 = os.path.split(nonSeizFiles[fns])
                    fileNameNS = fileName1[0:numLettersToRemove-1]

                    numCh = 18
                    dataNS = loadAndConcatenateAllFeatFilesForThisFile(nonSeizFiles[fns][0:numLettersToRemove], FeaturesParams, numCh)
                    numFeat = int(len(dataNS[0, :]))

                    lenSigNonSeiz= len(dataNS[:,0])
                    if (lenSigNonSeiz > nonSeizLen):
                        # cut nonseizure part
                        nonSeizStart = np.random.randint(lenSigNonSeiz - nonSeizLen - 1)
                        nonSeizCutout = dataNS[nonSeizStart: nonSeizStart + nonSeizLen, :]
                        newData[0:int(nonSeizLen / 2)] = nonSeizCutout[0:int(nonSeizLen / 2)]
                        newData[int(nonSeizLen / 2) + seizureLen:] = nonSeizCutout[int(nonSeizLen / 2):]

                        # SAVING TO CSV FILE
                        fileNameOut = os.path.splitext(fileName1)[0][0:6]
                        fileName3 = folderOut + '/' + fileNameOut  + '_f' + str(outputFileIndx).zfill(3) # 's' marks it is file with seizure
                        # writeToCsvFile(newData, newLabel, fileName3)
                        saveDataToFile(np.hstack((newData, np.reshape(newLabel, (-1,1)))), fileName3, 'gzip')

                        print('PAIRED: ', fileNameS , '- ', fileNameNS)
                        paieredFiles.append( fileNameOut  + '_cv' + str(outputFileIndx).zfill(3)  +' : ' +fileNameS + ' -- '+ fileNameNS)

                        outputFileIndx=outputFileIndx+1
                        IndxNonSeizFile = IndxNonSeizFile + 1


                        #in cases when there is more seizure files then non seizure ones, we will not save this seizures
                        # because there is no seizure file to randomly select from
                        # thus we can start from firs non seizure file again
                        # or if we want to be absolutely sure there is no overlap of non seizure files we can comment this,
                        # but we will loose some seizures  (or we need to think of smarter way to do this matching)
                        if (IndxNonSeizFile==len(nonSeizFiles)):
                            IndxNonSeizFile=0

                        break
                    else:
                        #fns = fns + 1
                        print('not enough nonSeiz data in this file')

    #save paired files
    file= open(folderOut + '/PairedFiles.txt', 'w')
    for i in range(len(paieredFiles)):
        file.write(paieredFiles[i]+'\n')
    file.close()


def plotRearangedDataLabelsInTime(folderIn,  GeneralParams,PostprocessingParams, FeaturesParams):
    ''' function that plots of all data of one subject in appended way
    this way it is possible to test if data rearanging worked and no data is lost'''
    folderOut=folderIn +'/LabelsInTime'
    createFolderIfNotExists(folderOut)

    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    for patIndx, pat in enumerate(GeneralParams.patients):
        inputFiles = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*.csv.gz'))
        numFiles = len(inputFiles)

        #concatinatin predictions so that predictions for one seizure are based on train set with all seizures before it
        for fIndx, fileName in enumerate(inputFiles):
            data=readDataFromFile(fileName)
            if fIndx==0:
                labels = np.squeeze(data[:,-1])
                testIndx=np.ones(len(data[:,-1]))*(fIndx+1)
            else:
                labels = np.hstack((labels,  np.squeeze(data[:,-1])))
                testIndx= np.hstack((testIndx, np.ones(len(data[:,-1]))*(fIndx+1)))

        (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(labels, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx)


        #Plot predictions in time
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(4, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.2, hspace=0.2)
        xValues = np.arange(0, len(labels), 1) / (60*60*2)
        ax1 = fig1.add_subplot(gs[0,0])
        ax1.plot(xValues, labels , 'r')
        ax1.set_ylabel('TrueLabel')
        ax1.set_title('Subj'+pat)
        ax1.grid()
        ax1 = fig1.add_subplot(gs[1, 0])
        ax1.plot(xValues, yPred_SmoothOurStep1, 'b')
        ax1.set_ylabel('Step1')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[2, 0])
        ax1.plot(xValues, yPred_SmoothOurStep2, 'm')
        ax1.set_ylabel('Step2')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[3, 0])
        ax1.plot(xValues, testIndx , 'k')
        ax1.set_ylabel('FileNr')
        ax1.grid()
        ax1.set_xlabel('Time [h]')
        fig1.show()
        fig1.savefig(folderOut + '/Subj' + pat + '_RawLabels.png', bbox_inches='tight')
        plt.close(fig1)


def removePreAndPostIctalAreas(data, labels, szStart, szStop, lenBeforeSeizIndx, lenAfterSeizIndx):
    ''' function that remores some amount of pre and post ictal data '''
    keepIndxs=np.ones(len(labels))
    for s in range(len(szStart)):
        #pre seizure part
        if (s == 0): #if first seizure, so no seizure before
            keepIndxs[ np.max([0,szStart[s]-lenBeforeSeizIndx ]): szStart[s]]=0
        elif (szStop[s - 1]  in range(szStart[s] - lenBeforeSeizIndx, szStart[s])): #previous seizure would we cut
            keepIndxs[szStop[s - 1]: szStart[s]] = 0
        else:  # seizure in the middle and all ok
            keepIndxs[szStart[s]-lenBeforeSeizIndx : szStart[s]] = 0
        #post seizure part
        if (s == (len(szStart) - 1)): #if last seizure, so no seizure after
            keepIndxs[szStop[s]: np.min([szStop[s]+lenAfterSeizIndx,len(labels)])] = 0
        elif (szStart[s + 1]  in range(szStop[s], szStop[s] + lenAfterSeizIndx)): #if next seizure in postictal of this, dont cut
            keepIndxs[szStop[s]: szStart[s+1]] = 0
        else: #seizure in the middle and all ok
            keepIndxs[szStop[s]: szStop[s]+lenAfterSeizIndx] = 0

    pos=np.where(keepIndxs==1)[0]
    try:
        dataNew=data[pos, :]
    except:
        print('dsadas')
    labelsNew=labels[pos]
    return(dataNew, labelsNew)

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


def concatenateDataFromFiles(fileNames):
    ''' loads and concatenates data from all files in file name list
    creates array noting lengths of each file to know when new file started in appeded data'''
    dataAll, labelsAll = [],[]
    startIndxOfFiles=np.zeros(len(fileNames))
    for f, fileName in enumerate(fileNames):
        data = readDataFromFile(fileName)
        data= np.float32(data)

        if (dataAll == []):
            dataAll = data[:, 0:-1]
            labelsAll = data[:, -1].astype(int)
            lenPrevFile=int(len(data[:, -1]))
            startIndxOfFiles[f]=lenPrevFile
        else:
            dataAll = np.vstack((dataAll, data[:, 0:-1]))
            labelsAll = np.hstack((labelsAll, data[:, -1].astype(int)))
            # startIndxOfFiles[f]=int(lenPrevFile)
            lenPrevFile = lenPrevFile+ len(data[:, -1])
            startIndxOfFiles[f] = int(lenPrevFile)
    startIndxOfFiles = startIndxOfFiles.astype((int))
    return (dataAll, labelsAll, startIndxOfFiles)

def test_StandardML_moreModelsPossible(data,trueLabels,  model):
    ''' gives predictions for standard machine learning models (not HD)
    returns predictions and probability
    calculates also simple accuracy and accuracy per class'''

    # number of clases
    (unique_labels, counts) = np.unique(trueLabels, return_counts=True)
    numLabels = len(unique_labels)
    if (numLabels==1): #in specific case when in test set all the same label
        numLabels=2

    #PREDICT LABELS
    y_pred= model.predict(data)
    y_probability = model.predict_proba(data)

    #pick only probability of predicted class
    y_probability_fin=np.zeros(len(y_pred))
    indx=np.where(y_pred==1)
    if (len(indx[0])!=0):
        y_probability_fin[indx]=y_probability[indx,1]
    else:
        a=0
        print('no seiz predicted')
    indx = np.where(y_pred == 0)
    if (len(indx[0])!=0):
        y_probability_fin[indx] = y_probability[indx,0]
    else:
        a=0
        print('no non seiz predicted')

    #calculate accuracy
    diffLab=y_pred-trueLabels
    indx=np.where(diffLab==0)
    acc= len(indx[0])/len(trueLabels)

    # calculate performance and distances per class
    accPerClass=np.zeros(numLabels)
    distFromCorr_PerClass = np.zeros(numLabels)
    distFromWrong_PerClass = np.zeros(numLabels)
    for l in range(numLabels):
        indx=np.where(trueLabels==l)
        trueLabels_part=trueLabels[indx]
        predLab_part=y_pred[indx]
        diffLab = predLab_part - trueLabels_part
        indx2 = np.where(diffLab == 0)
        if (len(indx[0])==0):
            accPerClass[l] = np.nan
        else:
            accPerClass[l] = len(indx2[0]) / len(indx[0])

    return(y_pred, y_probability_fin, acc, accPerClass)


def func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Appending(folderIn, GeneralParams,PostprocessingParams, FeaturesParams , typeModel):
    ''' goes through predictions of each file of each subject and appends them in time
    plots predictions in time
    also calculates average performance of each subject based on performance of whole appended predictions
    also calculates average of all subjects
    plots performances per subject and average of all subjects '''
    folderOut=folderIn +'/PerformanceWithAppendedTests/'
    createFolderIfNotExists(folderOut)

    AllSubjDiffPerf_test = np.zeros((len(GeneralParams.patients), 4* 9))
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFP_befSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFP_aftSeiz / FeaturesParams.winStep)
    for patIndx, pat in enumerate(GeneralParams.patients):
        filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+typeModel+'_TestPredictions.csv.gz'))
        for cv in range(len(filesAll)):
            data= readDataFromFile(filesAll[cv])
            if cv==0:
                trueLabels_AllCV = data[:, 0]
                probabLabels_AllCV=data[:,1]
                predLabels_AllCV=data[ :,2]
                dataSource_AllCV=np.ones(len(data[:, 0]))*(cv+1)
                if ('Fact' in folderIn):
                    yPredTest_MovAvrgStep1_AllCV=data[ :,3]
                    yPredTest_MovAvrgStep2_AllCV = data[:, 4]
                    yPredTest_SmoothBayes_AllCV = data[:, 5]

            else:
                trueLabels_AllCV = np.hstack((trueLabels_AllCV, data[:, 0]))
                probabLabels_AllCV = np.hstack((probabLabels_AllCV, data[: ,1]))
                predLabels_AllCV=np.hstack((predLabels_AllCV,data[:,2]))
                dataSource_AllCV= np.hstack((dataSource_AllCV, np.ones(len(data[:, 0]))*(cv+1)))
                if ('Fact' in folderIn):
                    yPredTest_MovAvrgStep1_AllCV = np.hstack((yPredTest_MovAvrgStep1_AllCV,data[:,3]))
                    yPredTest_MovAvrgStep2_AllCV = np.hstack((yPredTest_MovAvrgStep2_AllCV, data[:, 4]))
                    yPredTest_SmoothBayes_AllCV = np.hstack((yPredTest_SmoothBayes_AllCV, data[:, 5]))


        if ('Fact' not in folderIn): #smooth and calculate performance
            (performanceTest, yPredTest_MovAvrgStep1_AllCV, yPredTest_MovAvrgStep2_AllCV,yPredTest_SmoothBayes_AllCV) = calculatePerformanceAfterVariousSmoothing(predLabels_AllCV, trueLabels_AllCV,probabLabels_AllCV,
                                                                                toleranceFP_bef, toleranceFP_aft, numLabelsPerHour, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
        else:
            performanceTest = calculatePerformanceWithoutSmoothing(predLabels_AllCV, yPredTest_MovAvrgStep1_AllCV, yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV, trueLabels_AllCV, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        # calculationg avrg for this subj over all CV
        AllSubjDiffPerf_test[patIndx, :] =performanceTest

        dataToSave = np.vstack((trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1_AllCV,yPredTest_MovAvrgStep2_AllCV, yPredTest_SmoothBayes_AllCV, dataSource_AllCV)).transpose()  # added from which file is specific part of test set
        outputName = folderOut + '/Subj' + pat  + '_'+typeModel+'_Appended_TestPredictions.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')

        # Plot predictions in time
        fig1 = plt.figure(figsize=(12, 8), constrained_layout=False)
        gs = GridSpec(5, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.2, hspace=0.2)
        xValues = np.arange(0, len(trueLabels_AllCV), 1)
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.plot(xValues, predLabels_AllCV, 'k')
        ax1.set_ylabel('NoSmooth')
        ax1.set_title('Subj' + pat)
        ax1.grid()
        ax1 = fig1.add_subplot(gs[1, 0])
        ax1.plot(xValues, yPredTest_MovAvrgStep1_AllCV * 0.8, 'b')
        ax1.plot(xValues, yPredTest_MovAvrgStep2_AllCV, 'c')
        ax1.set_ylabel('Step1&2')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[2, 0])
        ax1.plot(xValues, yPredTest_SmoothBayes_AllCV, 'm')
        ax1.set_ylabel('Bayes')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[3, 0])
        ax1.plot(xValues, trueLabels_AllCV, 'r')
        ax1.set_ylabel('TrueLabel')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[4, 0])
        ax1.plot(xValues, dataSource_AllCV, 'k')
        ax1.set_ylabel('FileNr')
        ax1.grid()
        ax1.set_xlabel('Time')
        fig1.show()
        fig1.savefig(folderOut + '/Subj' + pat  + '_'+typeModel+'_Appended_TestPredictions.png', bbox_inches='tight')
        plt.close(fig1)

    # SAVE PERFORMANCE MEASURES FOR ALL SUBJ
    outputName = folderOut + '/AllSubj_AppendedTest_' +typeModel+'_AllPerfMeasures.csv'
    saveDataToFile(AllSubjDiffPerf_test, outputName, 'gzip')

    #plot
    fig1 = plt.figure(figsize=(10, 10), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    xValues = np.arange(1, len(GeneralParams.patients)+1, 1)
    perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
                 'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
    for perfIndx, perf in enumerate(perfNames):
        ax1 = fig1.add_subplot(gs[int(np.floor(perfIndx / 3)), np.mod(perfIndx, 3)])
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 0 +perfIndx], 'k')
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 9 +perfIndx], 'b')
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 18 +perfIndx], 'c')
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 27 +perfIndx], 'm')
        ax1.legend(['NoSmooth', 'Step1', 'Step2', 'Bayes'])
        #plotting mean values
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_test[:, 0 +perfIndx]), 'k')
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_test[:, 9 +perfIndx]), 'b')
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_test[:, 18 +perfIndx]), 'c')
        ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(AllSubjDiffPerf_test[:, 27 +perfIndx]), 'm')
        ax1.set_xlabel('Subjects')
        ax1.set_title(perf)
        ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/AllSubj_AppendedTest_' +typeModel+'_AllPerformanceMeasures.png', bbox_inches='tight')
    plt.close(fig1)

def func_plotPredictionsOfDifferentModels(modelsList, GeneralParams, folderIn, folderOut):
    ''' loads predictions in time of different models (from appended and smoothed predictions)
    and plots predictions in time to compare how different ML models perform
    also plots true label and from which file data is
    '''
    for patIndx, pat in enumerate(GeneralParams.patients):
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(3, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.4, hspace=0.4)
        fig1.suptitle('Subj '+ pat)

        for mIndx, mName in enumerate(modelsList):
            #trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1,yPredTest_MovAvrgStep2, yPredTest_SmoothBayes,dataSource_AllCV
            inName = folderIn + '/PerformanceWithAppendedTests/Subj' + pat + '_' + mName + '_Appended_TestPredictions.csv'
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
            ax2.plot(timeRaw, data[:, 5]* 0.5 + mIndx, 'm', label='Bayes')
            ax2.plot(timeRaw,data[:,3] * 0.7 + mIndx, 'b', label='Avrg_Step1')
            ax2.plot(timeRaw,data[:,4] * 0.7 + mIndx, 'c', label='Avrg_step2')
            if (mIndx == 0):
                ax2.legend()
        ax2.set_yticks(np.arange(0, len(modelsList), 1))
        ax2.set_yticklabels(modelsList, fontsize=10 * 0.8)
        ax2.set_xlabel('Time')
        ax2.grid()
        fig1.savefig(folderOut + '/Subj' + pat + '_PredictionsTest_AllModelsComparison.png')
        plt.close(fig1)


def func_calculatePerfmanceBasedOnPredictions_ForAllFiles_Average(folderOut, GeneralParams, PostprocessingParams,FeaturesParams , typeModel):
    ''' goes through predictions of each file of each subject and calculates performance
    calculates average performance of eahc subject based on average of all crossvalidations
    also calculates average of all subjects
    plots performances per subject and average of all subjects '''

    AllSubjDiffPerf_test = np.zeros((len(GeneralParams.patients), 4* 9))
    AllSubjDiffPerf_train = np.zeros((len(GeneralParams.patients), 4* 9))
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFP_befSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFP_aftSeiz / FeaturesParams.winStep)
    for patIndx, pat in enumerate(GeneralParams.patients):
        trainAvailable = 0
        filesAll = np.sort(glob.glob(folderOut + '/*Subj' + pat + '*_'+typeModel+'_TestPredictions.csv.gz'))
        numFiles=len(filesAll)
        performanceTrain = np.zeros(( numFiles, 4*9 ))  # 3 for noSmooth, step1, step2, and 9 or 9 perf meausres
        performanceTest = np.zeros(( numFiles, 4*9 ))  # 3 for noSmooth, step1, step2, and 9 or 9 perf meausres
        for cv in range(len(filesAll)):
            fileName2 = 'Subj' + pat + '_cv' + str(cv).zfill(3)

            data0 = readDataFromFile(filesAll[cv][0:-22]+'TestPredictions.csv.gz')
            label_test = data0[:, 0]
            probabLab_test = data0[:, 1]
            predLabels_test= data0[:, 2]
            # dataSource_test= data0[:, 6]
            (performanceTest[cv, :], yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2,yPredTest_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels_test, label_test,probabLab_test,
                                                                                toleranceFP_bef, toleranceFP_aft,numLabelsPerHour,seizureStableLenToTestIndx,seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
            dataToSave = np.vstack((label_test, probabLab_test, predLabels_test, yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2, yPredTest_SmoothBayes )).transpose()  # added from which file is specific part of test set
            outputName = folderOut + '/' + fileName2 + '_'+typeModel+'_TestPredictions.csv'
            saveDataToFile(dataToSave, outputName, 'gzip')

            if (os.path.exists(filesAll[cv][0:-22]+'TrainPredictions.csv.gz')):
                trainAvailable=1
                data0 =  readDataFromFile(filesAll[cv][0:-22]+'TrainPredictions.csv.gz')
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
        if ( trainAvailable==1):
            outputName = folderOut + '/Subj' + pat + '_'+typeModel+'_TrainAllPerfMeasures.csv'
            saveDataToFile(performanceTrain, outputName, 'gzip')

        # calculationg avrg for this subj over all CV
        if (trainAvailable == 1):
            AllSubjDiffPerf_train[patIndx, :] = np.nanmean(performanceTrain, 0)
        AllSubjDiffPerf_test[patIndx, :] = np.nanmean(performanceTest, 0)

    # SAVE PERFORMANCE MEASURES FOR ALL SUBJ
    if (trainAvailable == 1):
        outputName = folderOut + '/AllSubj_'+ typeModel+'_TrainAllPerfMeasures.csv'
        saveDataToFile(AllSubjDiffPerf_train, outputName, 'gzip')
    outputName = folderOut + '/AllSubj_'+ typeModel+'_TestAllPerfMeasures.csv'
    saveDataToFile(AllSubjDiffPerf_test, outputName, 'gzip')

    # plotting performance
    fig1 = plt.figure(figsize=(10, 10), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    xValues = np.arange(1, len(GeneralParams.patients)+1, 1)
    perfNames = ['Sensitivity episodes', 'Precision episodes', 'F1score episodes', 'Sensitivity duration',
                 'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
    for perfIndx, perf in enumerate(perfNames):
        ax1 = fig1.add_subplot(gs[int(np.floor(perfIndx / 3)), np.mod(perfIndx, 3)])
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 0 +perfIndx], 'k--')
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 9 +perfIndx], 'b--')
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 18 +perfIndx], 'c--')
        ax1.plot(xValues, AllSubjDiffPerf_test[:, 27 +perfIndx], 'm--')
        ax1.legend(['NoSmooth', 'Step1', 'Step2', 'Bayes'])
        if (trainAvailable == 1):
            ax1.plot(xValues, AllSubjDiffPerf_train[:, 0 + perfIndx], 'k')
            ax1.plot(xValues, AllSubjDiffPerf_train[:, 9 + perfIndx], 'b')
            ax1.plot(xValues, AllSubjDiffPerf_train[:, 18 + perfIndx], 'c')
            ax1.plot(xValues, AllSubjDiffPerf_train[:, 27 + perfIndx], 'm')
        #plotting mean values
        if (trainAvailable == 1):
            ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_train[:, 0 +perfIndx]), 'k')
            ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_train[:, 9 +perfIndx]), 'b')
            ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_train[:, 18 +perfIndx]), 'c')
            ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(AllSubjDiffPerf_train[:, 27 +perfIndx]), 'm')
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_test[:, 0 +perfIndx]), 'k--')
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_test[:, 9 +perfIndx]), 'b--')
        ax1.plot(xValues, np.ones(len(xValues))*np.nanmean(AllSubjDiffPerf_test[:, 18 +perfIndx]), 'c--')
        ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(AllSubjDiffPerf_test[:, 27 +perfIndx]), 'm--')
        ax1.set_xlabel('Subjects')
        ax1.set_title(perf)
        ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/AllSubj_AveragingCV_'+typeModel+'_AllPerformanceMeasures.png', bbox_inches='tight')
    plt.close(fig1)

def plot_performanceComparison_RFvsSTDHDandONLHD(folderIn, folderOut,  subjects, type):
    '''plot avarage performance of all subjects for random forest vs HD approachs'''
    numSubj=len(subjects)

    #load perofrmances per subj for all three approaches
    Perf_RF = readDataFromFile(folderIn + '/AllSubj_RF_'+type+'AllPerfMeasures.csv.gz')
    Perf_stdHD = readDataFromFile(folderIn + '/AllSubj_StdHD_'+type+'AllPerfMeasures.csv.gz')
    Perf_onlHD = readDataFromFile(folderIn + '/AllSubj_OnlineHD_'+type+'AllPerfMeasures.csv.gz')

    perfNames = ['TPR', 'PPV', 'F1']

    #EPISODES
    PerfIndxs=[18,19,20]
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
    PerfIndxs=[21,22,23]
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
    PerfIndxs=[26]
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
    AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'])
    AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'])
    AllPerfAllSubj_T['Performance'] = pd.to_numeric(AllPerfAllSubj_T['Performance'])

    fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = GridSpec(1,3, figure=fig1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.4)
    ax2 = fig1.add_subplot(gs[0,0])
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_E, palette="Set1")
    ax2.set_title('Episode level performance')
    ax2.legend(loc='lower left')
    ax2.grid(which='both')
    ax2 = fig1.add_subplot(gs[0,1])
    sns.set_theme(style="whitegrid")
    ax2.grid(which='both')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_D, palette="Set1")
    ax2.set_title('Duration performance')
    ax2.legend(loc='lower left')
    fig1.show()
    ax2 = fig1.add_subplot(gs[0,2])
    sns.set_theme(style="whitegrid")
    ax2.grid(which='both')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_T, palette="Set1")
    ax2.set_title('Combined measures')
    ax2.legend(loc='lower left')
    fig1.show()
    fig1.savefig(folderOut + '/RFvsHD_Allsubj_Performance_'+type+'.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/RFvsHD_Allsubj_Performance_'+type+'.svg', bbox_inches='tight')
    plt.close(fig1)


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


def plot_performanceComparison_FromDifferentFolders(folderInArray, folderInNames,  folderOut,  subjects, outputName, type, HDtype):
    '''plot avarage performance of all subjects for random forest vs HD approachs'''
    numSubj=len(subjects)
    indexOfset=18 #0 for no smooth, 9 for step1, 18 for step2, 27 for bayes
    smoothType='step2'

    perfAll=np.zeros((numSubj, 36, len(folderInArray)))
    #load performances
    for fIndx in range(len(folderInArray)):
        if (type=='Appended'):
            perfAll[:,:,fIndx]= readDataFromFile(folderInArray[fIndx]+'/PerformanceWithAppendedTests/' + '/AllSubj_AppendedTest_'+HDtype+'_AllPerfMeasures.csv.gz')
        else: #avrg of crossvalidations
            perfAll[:,:,fIndx] = readDataFromFile( folderInArray[fIndx] + '/AllSubj_'+HDtype+'_TestAllPerfMeasures.csv.gz')

    perfNames = ['TPR', 'PPV', 'F1']
    #EPISODES
    PerfIndxs=[indexOfset+0,indexOfset+1,indexOfset+2]
    for t, tIndx in enumerate(PerfIndxs):
        for fIndx in range(len(folderInArray)):
            dataAppend = np.vstack((perfAll[:, tIndx, fIndx], np.repeat(perfNames[t], numSubj), np.repeat(folderInNames[fIndx], numSubj))).transpose()
            if (fIndx==0 and t==0):
                AllPerfAllSubj_E = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
            else:
                AllPerfAllSubj_E = AllPerfAllSubj_E.append(pd.DataFrame(dataAppend,columns=['Performance', 'Measure', 'Approach']))

    #DURATION
    PerfIndxs=[indexOfset+3, indexOfset+4, indexOfset+5]
    for t, tIndx in enumerate(PerfIndxs):
        for fIndx in range(len(folderInArray)):
            dataAppend = np.vstack((perfAll[:, tIndx, fIndx], np.repeat(perfNames[t], numSubj), np.repeat(folderInNames[fIndx],  numSubj))).transpose()
            if (fIndx==0 and t==0):
                AllPerfAllSubj_D = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
            else:
                AllPerfAllSubj_D = AllPerfAllSubj_D.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))

    #DE combined and numFP
    perfNames2 = ['numFP']
    PerfIndxs=[indexOfset+8]
    for t, tIndx in enumerate(PerfIndxs):
        for fIndx in range(len(folderInArray)):
            dataAppend = np.vstack((perfAll[:, tIndx, fIndx], np.repeat(perfNames2[t], numSubj), np.repeat(folderInNames[fIndx],  numSubj))).transpose()
            if (fIndx==0 and t==0):
                AllPerfAllSubj_T = pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach'])
            else:
                AllPerfAllSubj_T = AllPerfAllSubj_T.append(pd.DataFrame(dataAppend, columns=['Performance', 'Measure', 'Approach']))

    # PLOTTING
    AllPerfAllSubj_E['Performance'] = pd.to_numeric(AllPerfAllSubj_E['Performance'])
    AllPerfAllSubj_D['Performance'] = pd.to_numeric(AllPerfAllSubj_D['Performance'])
    AllPerfAllSubj_T['Performance'] = pd.to_numeric(AllPerfAllSubj_T['Performance'])

    fig1 = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = GridSpec(1,2, figure=fig1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.4)
    ax2 = fig1.add_subplot(gs[0,0])
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_E, palette="Set1")
    ax2.set_title('Episode level performance')
    ax2.legend(loc='lower left')
    ax2.grid(which='both')
    ax2 = fig1.add_subplot(gs[0,1])
    sns.set_theme(style="whitegrid")
    ax2.grid(which='both')
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)
    sns.boxplot(x='Measure', y='Performance', width=0.5, hue='Approach', data=AllPerfAllSubj_D, palette="Set1")
    ax2.set_title('Duration performance')
    ax2.legend(loc='lower left')
    fig1.show()
    fig1.savefig(folderOut + '/'+outputName+'_'+HDtype+'_'+smoothType+'.png', bbox_inches='tight')
    fig1.savefig(folderOut + '/'+outputName+'_'+HDtype+'_'+smoothType +'.svg', bbox_inches='tight')
    plt.close(fig1)


def storeModel_InitializedVectors_GeneralWithChCombinations(model, folderOut, fileName):
    # 'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend'
    FeatValVectorMap=np.zeros((model.NumValues, model.HD_dim))
    FeatVectorMap = np.zeros((model.NumFeat, model.HD_dim))
    ChVectorMap = np.zeros((model.NumCh, model.HD_dim))
    for l in range(model.NumValues):
        FeatValVectorMap[l,:]=model.proj_mat_FeatVals[ :,l].cpu().numpy()
    for l in range(model.NumFeat):
        FeatVectorMap[l,:]=model.proj_mat_features[ :,l].cpu().numpy()
    for l in range(model.NumCh):
        ChVectorMap[l,:]=model.proj_mat_channels[ :,l].cpu().numpy()
    outputName = folderOut + '/' + fileName + '_InitVectMap_FeatValues.csv'
    saveDataToFile(FeatValVectorMap, outputName, 'gzip')
    # np.savetxt(outputName, FeatValVectorMap, fmt='%d', delimiter=",")
    outputName = folderOut + '/' + fileName + '_InitVectMap_Features.csv'
    saveDataToFile(FeatVectorMap, outputName, 'gzip')
    # np.savetxt(outputName, FeatVectorMap,  fmt='%d', delimiter=",")
    outputName = folderOut + '/' + fileName + '_InitVectMap_Channels.csv'
    saveDataToFile(ChVectorMap, outputName, 'gzip')
    # np.savetxt(outputName, ChVectorMap,  fmt='%d', delimiter=",")

def storeModel_InitializedVectors_GeneralAndNoCh(model, folderOut, fileName):
    FeatValVectorMap=np.zeros((model.NumValues, model.HD_dim))
    FeatVectorMap = np.zeros((model.NumFeat, model.HD_dim))
    for l in range(model.NumValues):
        FeatValVectorMap[l,:]=model.proj_mat_FeatVals[ :,l].cpu().numpy()
    for l in range(model.NumFeat):
        FeatVectorMap[l,:]=model.proj_mat_features[ :,l].cpu().numpy()
    outputName = folderOut + '/' + fileName + '_InitVectMap_FeatValues.csv'
    saveDataToFile(FeatValVectorMap, outputName, 'gzip')
    outputName = folderOut + '/' + fileName + '_InitVectMap_Features.csv'
    saveDataToFile(FeatVectorMap, outputName, 'gzip')

def calcHistogramValues_v2(sig, segmentedLabels, histbins):
    '''takes one window of signal - all ch and labels, separates seiz and nonSeiz and
    calculates histogram of values  during seizure and non seizure '''
    numBins=int(histbins)
    sig2 = sig[~np.isnan(sig)]
    # maxValFeat=np.max(sig)
    # binBorders=np.arange(0, maxValFeat+1, (maxValFeat+1)/numBins)

    sig[sig == np.inf] = np.nan
    indxs=np.where(segmentedLabels==0)[0]
    nonSeiz = sig[indxs]
    nonSeiz = nonSeiz[~np.isnan(nonSeiz)]
    try:
        nonSeiz_hist = np.histogram(nonSeiz, bins=numBins, range=(np.min(sig2), np.max(sig2)))
    except:
        print('Error with hist ')

    indxs = np.where(segmentedLabels == 1)[0]
    Seiz = sig[indxs]
    Seiz = Seiz[~np.isnan(Seiz)]
    try:
        Seiz_hist = np.histogram(Seiz, bins=numBins, range=(np.min(sig2), np.max(sig2)))
    except:
        print('Error with hist ')

    # normalizing values that are in percentage of total samples - to not be dependand on number of samples
    nonSeiz_histNorm=[]
    nonSeiz_histNorm.append(nonSeiz_hist[0]/len(nonSeiz))
    nonSeiz_histNorm.append(nonSeiz_hist[1])
    Seiz_histNorm=[]
    Seiz_histNorm.append(Seiz_hist[0]/len(Seiz))
    Seiz_histNorm.append(Seiz_hist[1])
    # Seiz_hist[0] = Seiz_hist[0] / len(Seiz_allCh)
    return( Seiz_histNorm, nonSeiz_histNorm)


def kl_divergence(p,q):
    delta=0.000001
    deltaArr=np.ones(len(p))*delta
    p=p+deltaArr
    q=q+deltaArr
    res=sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))
    return res

def js_divergence(p,q):
    m=0.5* (p+q)
    res=0.5* kl_divergence(p,m) +0.5* kl_divergence(q,m)
    return (res)

def measureConfidences(distancesS, distancesNS):
    '''if negative then it is more confident that it is NS and if it is positive then it is more condifent that it is seizure'''
    distDiff = distancesNS[:, 0:-1] - distancesS[:, 0:-1]  # how much it is closer to S then NS
    avrgDistDiff = np.mean(np.abs(distDiff), 1)
    distDiffNorm = np.divide(distDiff , avrgDistDiff[:,None])
    return(distDiffNorm)

def calculateConfOnlyWhenCorreclyVoting(confidences, trueLabels):
    predictionsPerFeature=1.0*((confidences>0))
    meanConf_TP=np.zeros((len(confidences[0,:])))
    meanConf_FP = np.zeros((len(confidences[0, :])))
    for f in range(len(confidences[0,:])):
        indxs=np.where(predictionsPerFeature[:,f]==trueLabels)[0]
        meanConf_TP[f]=np.mean(np.abs(confidences[indxs, f]))
        indxs=np.where(predictionsPerFeature[:,f]!=trueLabels)[0]
        meanConf_FP[f]=np.mean(np.abs(confidences[indxs, f]))
    featConf=(meanConf_TP-meanConf_FP)/(0.001+meanConf_FP) #how much more times it is more confident when TP then then FP (false prediction)
    return(featConf)

# def
def givePredictionsBasedOnqualityOfEachFeature_moreThresh( predictions, distancesS, distancesNS, featureQualityMeasure, VotingParam):

    if  (VotingParam.selectionStyle=='Perc'):
        numSteps=len(VotingParam.confThreshArray)
    else:
        numSteps=len(featureQualityMeasure)
    # sort them based on biggest KLdiverg values
    featIndxSorted = np.argsort(-featureQualityMeasure)
    JSdivergence_Sorted = featureQualityMeasure[featIndxSorted]

    (numWin, numFeat)=distancesS.shape
    numFeat=numFeat-1 #last one is all features
    predictionsFinal=np.zeros((numWin, numSteps))
    for s in range(numWin):
        # calculate average dist between Vect from Smodel and Vect from NSmodel
        distDiff = distancesNS[s, 0:-1] - distancesS[s, 0:-1] # how much it is closer to S then NS
        avrgDistDiff=np.mean(np.abs(distDiff))
        distDiffNorm=distDiff/avrgDistDiff #confidence of each feature, if <0 then more confident in nonSeiz and if >0 more confidend in seizure
        distDiffNormSorted=distDiffNorm[featIndxSorted]


        # #calculate distances so that + values are dist from seiz and -1 from non seiz
        # higherConfForEachFeat=np.zeros((numFeat))
        # for i in range(numFeat):
        #     if predictions[s,i]==1:
        #         higherConfForEachFeat[i]=1-distancesS[s,i]
        #     if predictions[s,i] == 0:
        #         higherConfForEachFeat[i] = -(1 - distancesNS[s,i])


        #keep only thresh percent of most confident ones
        for th in range(numSteps):
            if (VotingParam.selectionStyle == 'Perc'):
                numIndxToKeep=int(VotingParam.confThreshArray[th]*numFeat)
            else:
                numIndxToKeep=th+1
            # indxToKeep=featIndxSorted[0:numIndxToKeep]

            #sum predictions and if >0 it is seiazure
            sumConf=np.sum(distDiffNormSorted[0:numIndxToKeep])
            if (sumConf>0):
                predictionsFinal[s,th]=1
            else:
                predictionsFinal[s, th] = 0

    return (predictionsFinal)

def func_plotPredictionsForDifferentBinding(foldersList, namesList, GeneralParams, HDtype, folderOut):
    ''' loads predictions in time of different approaches
    and plots predictions in time to compare how different approaches perform
    also plots true label and from which file data is
    '''
    createFolderIfNotExists(folderOut)
    for patIndx, pat in enumerate(GeneralParams.patients):
        fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
        gs = GridSpec(3, 1, figure=fig1)
        fig1.subplots_adjust(wspace=0.4, hspace=0.4)
        fig1.suptitle('Subj '+ pat)

        for fIndx, fName in enumerate(foldersList):
            #trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1,yPredTest_MovAvrgStep2, yPredTest_SmoothBayes,dataSource_AllCV
            inName = fName + '/PerformanceWithAppendedTests/Subj' + pat + '_' + HDtype + '_Appended_TestPredictions.csv'
            #trueLabels_AllCV, probabLabels_AllCV, predLabels_AllCV, yPredTest_MovAvrgStep1,yPredTest_MovAvrgStep2, yPredTest_SmoothBayes, dataSource_AllCV
            data=readDataFromFile(inName)

            if (fIndx==0):
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

            ax2.plot(timeRaw, data[:,2] * 0.3 + fIndx, 'k', label='NoSmooth')
            ax2.plot(timeRaw, data[:, 5]* 0.5 + fIndx, 'm', label='Bayes')
            ax2.plot(timeRaw,data[:,3] * 0.7 + fIndx, 'b', label='Avrg_Step1')
            ax2.plot(timeRaw,data[:,4] * 0.7 + fIndx, 'c', label='Avrg_step2')
            if (fIndx == 0):
                ax2.legend()

        ax2.set_yticks(np.arange(0, len(foldersList), 1))
        ax2.set_yticklabels(namesList, fontsize=10 * 0.8)
        ax2.set_xlabel('Time')
        ax2.grid()
        fig1.savefig(folderOut + '/Subj' + pat + '_'+HDtype+'_PredictionsTest_AllModelsComparison.png')
        plt.close(fig1)

def calculateProbabilitiesPerFeature(distFromS, distFromNS, predLabels):
    (numSampl, numFeat)=predLabels.shape
    # probability
    dist_fromVotedOne = np.zeros((numSampl, numFeat))
    dist_oppositeVotedOne = np.zeros((numSampl, numFeat))
    probabilityLab= np.zeros((numSampl, numFeat))
    for f in range( numFeat):
        indx = np.where(predLabels[:,f] == 1)[0]
        dist_fromVotedOne[indx,f] = distFromS[indx, f]
        dist_oppositeVotedOne[indx,f] = distFromNS[indx,f]
        indx = np.where(predLabels[:,f] == 0)[0]
        dist_fromVotedOne[indx,f] = distFromNS[indx, f]
        dist_oppositeVotedOne[indx,f] = distFromS[indx, f]
        probabilityLab[:,f] = dist_oppositeVotedOne[:,f] / (dist_oppositeVotedOne[:,f] + dist_fromVotedOne[:,f]+ 0.00001)
    return(probabilityLab)

def calculate_corrPerFeat(corrAllFeatCh, numFeat):
    numCh=int(len(corrAllFeatCh[:,0])/numFeat)
    firstPassCorr=np.ones((numCh*numFeat, numFeat, numCh))*np.nan
    for ch in range(numCh):
        firstPassCorr[:,:,ch]=corrAllFeatCh[:,ch*numFeat: (ch+1)*numFeat]
    firstPassCorr=np.nanmean(firstPassCorr, 2)
    secondPassCorr=np.ones((numFeat, numFeat, numCh))*np.nan
    for ch in range(numCh):
        secondPassCorr[:,:,ch]=firstPassCorr[ch*numFeat: (ch+1)*numFeat,:]
    finalCorrPerFeat=np.nanmean(secondPassCorr,2)

    return(finalCorrPerFeat)

def calculate_corrPerCh(corrAllFeatCh, numCh):
    numFeat=int(len(corrAllFeatCh[:,0])/numCh)
    firstPassCorr=np.ones((numCh*numFeat, numCh, numFeat))*np.nan
    for f in range(numFeat):
        firstPassCorr[:,:,f]=corrAllFeatCh[:,f::numFeat]
    firstPassCorr=np.nanmean(firstPassCorr, 2)
    secondPassCorr=np.ones((numCh, numCh, numFeat))*np.nan
    for f in range(numFeat):
        secondPassCorr[:,:,f]=firstPassCorr[f::numFeat,:]
    finalCorrPerCh=np.nanmean(secondPassCorr,2)

    return(finalCorrPerCh)

def calculate_corrOfFeatThroughCh(corrAllFeatCh, numFeat):
    numCh = int(len(corrAllFeatCh[:, 0]) / numFeat)
    finalCorrFeatThroughCh=np.ones((numFeat))*np.nan
    for f in range(numFeat):
        corrFeatThroughCh = np.ones((numCh, numCh)) * np.nan
        for ch in range(numCh):
            for ch2 in range(numCh):
                corrFeatThroughCh[ch,ch2]=corrAllFeatCh[ch*numFeat+f,ch2*numFeat+f]
        finalCorrFeatThroughCh[f]=np.nanmean(np.nanmean(corrFeatThroughCh))

    return(finalCorrFeatThroughCh)

def calculate_corrOfChThroughFeat(corrAllFeatCh, numCh):
    numFeat = int(len(corrAllFeatCh[:, 0]) / numCh)
    finalCorrChThroughFeat=np.ones((numCh))*np.nan
    for ch in range(numCh):
        corrChThroughFeat = np.ones((numFeat, numFeat)) * np.nan
        for f in range(numFeat):
            for f2 in range(numFeat):
                corrChThroughFeat[f,f2]=corrAllFeatCh[ch*numFeat+f,ch*numFeat+f2]
        finalCorrChThroughFeat[ch]=np.nanmean(np.nanmean(corrChThroughFeat))

    return(finalCorrChThroughFeat)

def func_calculateFeatureCorrelation(predictions):
    numCols=len(predictions[ 0,:])
    numPreds=len(predictions[:, 0])
    corrs=np.zeros((numCols, numCols))
    for f1 in range(numCols):
        for f2 in range(numCols):
            corrs[f1,f2]=1- np.sum(np.abs(predictions[ :,f1] - predictions[ :,f2]))/ numPreds
    return corrs


def calculatePredictionOverlap(predLabels, trueLabels):
    (numSampl, numFeat) = predLabels.shape
    corr_CC=np.ones((numFeat, numFeat))*np.nan
    corr_WC = np.ones((numFeat, numFeat))*np.nan
    corr_CW = np.ones((numFeat, numFeat))*np.nan
    corr_WW = np.ones((numFeat, numFeat))*np.nan
    for f1 in range(numFeat):
        for f2 in range(numFeat):
            indx = np.where((predLabels[:, f1] == trueLabels ) & (predLabels[:, f2] == trueLabels ))[0]
            corr_CC[f1,f2]= len(indx)/len(trueLabels)
            # corr_CC[f2, f1]=corr_CC[f1,f2]
            indx = np.where((predLabels[:, f1] != trueLabels )  & (predLabels[:, f2] == trueLabels ) )[0]
            corr_WC[f1,f2]= len(indx)/len(trueLabels)
            # corr_WC[f2, f1]=corr_WC[f1,f2]
            indx = np.where((predLabels[:, f1] == trueLabels )  & (predLabels[:, f2] != trueLabels ) )[0]
            corr_CW[f1,f2]=len(indx)/len(trueLabels)
            # corr_CW[f2, f1]=corr_CW[f1,f2]
            indx = np.where((predLabels[:, f1] != trueLabels )  & (predLabels[:, f2] != trueLabels ) )[0]
            corr_WW[f1,f2]=len(indx)/len(trueLabels)
            # corr_WW[f2, f1]=corr_WW[f1,f2]
            # indx = np.where((predLabels[:, f1] == trueLabels ) & (predLabels[:, f2] == trueLabels )[0]
            # if (len(indx)!=0):
            #     corr_CC[f1,f2]=1 - np.sum(np.abs(predLabels[indx, f1] - predLabels[indx, f2])) / len(indx)
            #     corr_CC[f2, f1]=corr_CC[f1,f2]
            # indx = np.where((predLabels[:, f1] != trueLabels )  & (predLabels[:, f2] == trueLabels ) )[0]
            # if (len(indx) != 0):
            #     corr_WC[f1,f2]=1 - np.sum(np.abs(predLabels[indx, f1] - predLabels[indx, f2])) / len(indx)
            #     corr_WC[f2, f1]=corr_WC[f1,f2]
            # indx = np.where((predLabels[:, f1] == trueLabels )  & (predLabels[:, f2] != trueLabels ) )[0]
            # if (len(indx) != 0):
            #     corr_CW[f1,f2]=1 - np.sum(np.abs(predLabels[indx, f1] - predLabels[indx, f2])) / len(indx)
            #     corr_CW[f2, f1]=corr_CW[f1,f2]
            # indx = np.where((predLabels[:, f1] != trueLabels )  & (predLabels[:, f2] != trueLabels ) )[0]
            # if (len(indx) != 0):
            #     corr_WW[f1,f2]=1 - np.sum(np.abs(predLabels[indx, f1] - predLabels[indx, f2])) / len(indx)
            #     corr_WW[f2, f1]=corr_WW[f1,f2]
    # totalCorr=np.hstack((np.vstack((corr_CC, corr_CW)), np.vstack((corr_WC, corr_WW))))
    totalCorr = np.vstack((corr_CC, corr_CW, corr_WC, corr_WW))
    return (totalCorr)


def measurePerformanceWhenMoreFeat(distancesS, distancesNS, chosenFeats, trueLabels, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour):
    distDiff = np.sum(distancesNS[:, chosenFeats], 1) - np.sum(distancesS[:, chosenFeats], 1)
    predictions= np.zeros(len(trueLabels))
    predictions[distDiff > 0] = 1  # where its further from NS then S
    performancesAll = performance_all9(predictions, trueLabels, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    return (performancesAll)


def findWhichFeatToAdd(distancesS, distancesNS, trueLabels, chosenFeats, perfMetricIndx,toleranceFP_bef, toleranceFP_aft,numLabelsPerHour):
    numElems = len(distancesS[0, :])

    performancesAll = np.zeros((numElems - len(chosenFeats), 9))
    fIndxsTested = np.zeros(numElems - len(chosenFeats))
    fIndx = 0
    for f in range(numElems):
        if f not in chosenFeats:
            featsToTest = np.copy(chosenFeats)
            featsToTest = np.append(featsToTest, f)
            fIndxsTested[fIndx] = int(f)
            performancesAll[fIndx, :] = measurePerformanceWhenMoreFeat(distancesS, distancesNS, featsToTest, trueLabels,toleranceFP_bef, toleranceFP_aft,numLabelsPerHour)
            fIndx = fIndx + 1
    indxSorted = np.argsort(-performancesAll[:, perfMetricIndx])
    bestIndx = indxSorted[0]
    optFeat = int(fIndxsTested[bestIndx])

    return (optFeat, performancesAll[bestIndx, :])


def chooseOptimalFeatureOrder(predLabels, distancesS, distancesNS, trueLabels, PostprocessingParams, FeaturesParams, perfMetricIndx):
    # various postpocessing parameters
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFP_befSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFP_aftSeiz / FeaturesParams.winStep)
    numElems = len(predLabels[0, :])

    # calculate perfromance per feature to find the best first one
    performancesPerFeat = np.zeros((numElems, 9))
    for f in range(numElems):
        performancesPerFeat[f, :] = performance_all9(predLabels[:, f], trueLabels, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)

    # find best feature based on
    # perfMetricIndx = 2  # 2 for F1E, 7 for F1DE

    perfMeasuresWithMoreFeat = np.zeros((numElems, 9))
    featOrder = np.zeros((numElems))
    for f in range(numElems):
        if (f == 0):
            commonMeasure = performancesPerFeat[:, perfMetricIndx]  # if first pick the one with best performance
            indxSorted = np.argsort(-commonMeasure)
            optFeat = indxSorted[0]
            perfMeasuresWithMoreFeat[f,:] = performancesPerFeat[optFeat, :]
            chosenFeats = [optFeat]
            featOrder[optFeat] = f #numElems - f  # better the bigger number
        else:
            (optFeat, newPerf) = findWhichFeatToAdd(distancesS, distancesNS, trueLabels, chosenFeats, perfMetricIndx,toleranceFP_bef, toleranceFP_aft,numLabelsPerHour)
            perfMeasuresWithMoreFeat[f,:] = newPerf
            chosenFeats = np.append(chosenFeats, optFeat)
            featOrder[optFeat] = f# numElems - f  # better the bigger number

    return (chosenFeats, featOrder, perfMeasuresWithMoreFeat, performancesPerFeat)


def normalizeValues(sig):
    return ( sig-np.min(sig))/(np.max(sig)-np.min(sig))

def visualizeEEGtopoplot_AllSubj(data, folderOutName, subjects,  norm, vmin, vmax):
    from mne import create_info
    from mne.viz import plot_topomap

    chanNames2=['AF7', 'FT7', 'TP7', 'PO7', 'AF3','FC3', 'CP3', 'PO3','AF4', 'FC4', 'CP4', 'PO4', 'AF8', 'FT8','TP8', 'PO8','FCz', 'CPz']
    info= create_info(ch_names=chanNames2, sfreq=1000, ch_types='eeg')
    info.set_montage('standard_1020')

    # fig, axes= plt.subplots(nrows=1, ncols=1)
    # im,_=plot_topomap(normalizeValues(data[0,:]),info, axes=axes, show=False, cmap='gray') # names=chanNames, show_names=True, axes=axes[0])
    # plt.colorbar(im)
    # fig.savefig('eegmontage.png')

    fig1 = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = GridSpec( 4,6,  figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    for p, pat in enumerate(subjects):
        ax1 = fig1.add_subplot(gs[ int(np.floor(p / 6)),  np.mod(p, 6)])
        # im, _ = plot_topomap(normalizeValues(data[p, :]), info, axes=ax1, show=True,cmap='Purples')  # names=chanNames, show_names=True, axes=axes[0])
        # im, _ = plot_topomap(data[p, :], info, axes=ax1, show=True,cmap='Greens', vlim=(0,1))  # names=chanNames, show_names=True, axes=axes[0])
        if (norm == 1):
            im, _ = plot_topomap(normalizeValues(data[p, :]), info, axes=ax1, show=True, cmap='BuPu', vlim=(0, 1))
        else:
            if (vmin == -1 and vmax == -1):
                im, _ = plot_topomap(data[p, :], info, axes=ax1, show=True, cmap='BuPu', vlim=(np.min(data[p, :]), np.max(data[p, :])))
            else:
                im, _ = plot_topomap(data[p, :], info, axes=ax1, show=True, cmap='BuPu', vlim=(vmin, vmax))
        ax1.set_title('Subj ' + pat)
        plt.colorbar(im)
        # plt.clim(0,1)
        ax1.grid()
    fig1.show()
    # fig1.savefig(folderOut + '/AllSubj_JSDivergence_EEGtopo.png', bbox_inches='tight')
    fig1.savefig(folderOutName+'.png', bbox_inches='tight')
    fig1.savefig(folderOutName + '.svg', bbox_inches='tight')
    plt.close(fig1)


def visualizeEEGtopoplot(data, folderOutName, norm, vmin, vmax):
    from mne import create_info
    from mne.viz import plot_topomap

    chanNames2 = ['AF7', 'FT7', 'TP7', 'PO7', 'AF3', 'FC3', 'CP3', 'PO3', 'AF4', 'FC4', 'CP4', 'PO4', 'AF8', 'FT8',
                  'TP8', 'PO8', 'FCz', 'CPz']
    info = create_info(ch_names=chanNames2, sfreq=1000, ch_types='eeg')
    info.set_montage('standard_1020')

    fig1 = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1 = fig1.add_subplot(gs[0,0])
    # im, _ = plot_topomap(normalizeValues(data[p, :]), info, axes=ax1, show=True,cmap='Purples')  # names=chanNames, show_names=True, axes=axes[0])
    if (norm==1):
        im, _ = plot_topomap(normalizeValues(data), info, axes=ax1, show=True, cmap='BuPu', vlim=(0, 1))
    else:
        if (vmin==-1 and vmax==-1):
            im, _ = plot_topomap(data, info, axes=ax1, show=True, cmap='BuPu', vlim=(np.min(data), np.max(data)))
        else:
            im, _ = plot_topomap(data, info, axes=ax1, show=True, cmap='BuPu', vlim=(vmin, vmax))
    plt.colorbar(im)
    ax1.grid()
    fig1.show()
    fig1.savefig(folderOutName + '.png', bbox_inches='tight')
    plt.close(fig1)

def visualizeEEGtopoplot_AllPlotsForOneSubj(data, folderOutName,  norm, vmin, vmax, plotNames):
    from mne import create_info
    from mne.viz import plot_topomap

    numPlots=len(data[:,0])

    chanNames2=['AF7', 'FT7', 'TP7', 'PO7', 'AF3','FC3', 'CP3', 'PO3','AF4', 'FC4', 'CP4', 'PO4', 'AF8', 'FT8','TP8', 'PO8','FCz', 'CPz']
    info= create_info(ch_names=chanNames2, sfreq=1000, ch_types='eeg')
    info.set_montage('standard_1020')

    fig1 = plt.figure(figsize=(10, 2), constrained_layout=False)
    gs = GridSpec( 1,numPlots,  figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.4)
    for p in range(numPlots):
        ax1 = fig1.add_subplot(gs[ 0,p])
        if (norm[p] == 1):
            im, _ = plot_topomap(normalizeValues(data[p, :]), info, axes=ax1, show=True, cmap='BuPu', vlim=(0, 1))
        else:
            if (vmin[p]  == -1 and vmax[p]  == -1):
                im, _ = plot_topomap(data[p, :], info, axes=ax1, show=True, cmap='BuPu', vlim=(np.min(data[p, :]), np.max(data[p, :])))
            else:
                im, _ = plot_topomap(data[p, :], info, axes=ax1, show=True, cmap='BuPu', vlim=(vmin[p] , vmax[p] ))
        ax1.set_title(plotNames[p])
        # if (p==numPlots-1):
        plt.colorbar(im)
        # plt.clim(0,1)
        ax1.grid()
    fig1.show()
    # fig1.savefig(folderOut + '/AllSubj_JSDivergence_EEGtopo.png', bbox_inches='tight')
    fig1.savefig(folderOutName+'_AllTopoPlots.png', bbox_inches='tight')
    fig1.savefig(folderOutName + '_AllTopoPlots.svg', bbox_inches='tight')
    plt.close(fig1)

def findOptimumNumFeatures(perfTrend):

    maxPerf = np.max(perfTrend)
    # maxIndx=np.argmax(perfTrend)+1
    maxIndx = np.where(perfTrend > maxPerf - 0.005)[0][0]+1
    return (maxIndx, perfTrend[maxIndx])

def plotInterp_PredAndConf_allComb( predProbability, trueLabels, FeaturesParams, numCh, fileName, folderOutInterp, typePlot): #numCh
    numFeat = len(predProbability[0,:])
    if (typePlot=='PerFeat' or typePlot=='PerCh'):
        numFeatPerCh=int(numFeat/numCh)
    else:
        numFeatPerCh=numFeat

    colormap = np.array([[0, 134, 139], [139, 34, 82]]) / 256
    fig1 = plt.figure(figsize=(16, 8), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    xValues = np.arange(0, len(trueLabels), 1) * FeaturesParams.winStep
    ax1 = fig1.add_subplot(gs[0, 0])
    for f in range(numFeat):

        #opacity for n most confident features
        predLabels = np.zeros(len(predProbability[:,0])).astype(int)
        indx = np.where(predProbability[:,f] > 0)
        predLabels[indx] = 1

        # opacity=(abs(predProbability[:, f])>0.9)*0.9+ 0.1
        opacity = (abs(predProbability[:, f]) /0.3).astype(int) * 0.3 + 0.1
        opacity[opacity>1.0]=1.0
        opacity[opacity < 0] = 0.0
        opacity=np.reshape(opacity, (-1, 1))

        c1 = np.hstack((colormap[predLabels, :], opacity ))
        ax1.scatter(xValues, f + 0.5*predProbability[:, f], c=c1, s=4)

        if (typePlot=='PerCh'):
            if (f ==0):
                yLabelNames=['Ch'+str(0)]
            elif (f % numFeatPerCh==0):
                yLabelNames.append('Ch'+str(int(f/numFeatPerCh)) )
        elif (typePlot=='PerFeat'):
            if (f ==0):
                yLabelNames = [FeaturesParams.featNames[0]]
            elif (f % numCh==0):
                yLabelNames.append(FeaturesParams.featNames[int(f/numCh)])
        elif (typePlot=='MeanPerCh'):
            if (f == 0):
                yLabelNames = ['Ch' + str(0)]
            else:
                yLabelNames.append('Ch' + str(f))
        elif (typePlot=='MeanPerFeat'):
            if (f == 0):
                yLabelNames = [FeaturesParams.featNames[0]]
            else:
                yLabelNames.append(FeaturesParams.featNames[f])

    ax1.plot(xValues, trueLabels * (numFeat + 1) - 0.5, c='r')
    ax1.set_xlabel('Time [s]')
    if (typePlot=='MeanPerCh'):
        yticLocations=np.arange(0, numCh, 1)
    elif( typePlot == 'MeanPerFeat'):
        yticLocations=np.arange(0, numFeatPerCh, 1)
    elif (typePlot=='PerCh'):
        yticLocations =np.arange(0, numCh, 1)*numFeatPerCh
    elif (typePlot=='PerFeat'):
        yticLocations=np.arange(0, numFeatPerCh, 1)*numCh
    # ax1.set_yticklabels(yLabelNames, fontsize=10 * 0.8)
    ax1.set(yticks=yticLocations, yticklabels=yLabelNames)
    ax1.set_title(fileName)
    ax1.grid()
    fig1.show()
    fig1.savefig(folderOutInterp + '/' + fileName + '_FeatPredAndConf_'+typePlot+'.png', bbox_inches='tight')
    # print('SAVED: ', folderOutInterp + '/' + fileName + '_FeatPredAndConf_'+typePlot+'.png')
    plt.close(fig1)

def plotInterp_PredAndConf(meanFeatVal,predLabels, predProbability, trueLabels, FeaturesParams, fileName, folderOutInterp, elemNames):
    FSize=14
    numElems = len(predProbability[0,: ])
    if len(elemNames)<numElems:
        elemNames=elemNames+ list(['AllCh'])
    colormap = np.array([[0, 134, 139], [139, 34, 82]]) / 256 # blue then red
    fig1 = plt.figure(figsize=(16, 5), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    xValues = np.arange(0, len(trueLabels), 1) * FeaturesParams.winStep
    ax1 = fig1.add_subplot(gs[0, 0])
    for f in range(numElems):
        # plotting feature valus
        prob=np.reshape(predProbability[:, f], (-1, 1))
        c1 = np.hstack((colormap[predLabels[:, f], :], np.power(prob,4) ))
        if (meanFeatVal==[]):
            featValNorm=np.ones(len(predProbability[:,0]))
            densFactor=10
            for i in range(densFactor):
                ax1.scatter(xValues, f + i *featValNorm/densFactor, c=c1, s=4, marker='s')
        else:
            featValNorm = (meanFeatVal[:, f] - np.min(meanFeatVal[:, f])) / ( np.max(meanFeatVal[:, f]) - np.min(meanFeatVal[:, f]))
            ax1.scatter(xValues, f + 0.8 * featValNorm, c=c1, s=4, marker='s')
    ax1.plot(xValues, trueLabels * numElems , c='r')
    ax1.set_xlabel('Time [s]', fontsize=FSize)
    ax1.set_yticks(np.arange(0, numElems, 1))
    ax1.set_yticklabels(elemNames, fontsize=FSize * 0.8)
    ax1.set_title(fileName[0:6])
    ax1.grid()
    fig1.show()
    fig1.savefig(folderOutInterp + '/' + fileName + '_PredAndConfPerElement.png', bbox_inches='tight', dpi=300)
    fig1.savefig(folderOutInterp + '/' + fileName + '_PredAndConfPerElement.svg', bbox_inches='tight')
    plt.close(fig1)



def func_plotConfidencesPerElement_Appending(folderIn, GeneralParams,PostprocessingParams, FeaturesParams , HDtype, FeatChType, elemNames):
    ''' goes through predictions of each file of each subject and appends them in time
    plots predictions in time
    also calculates average performance of each subject based on performance of whole appended predictions
    also calculates average of all subjects
    plots performances per subject and average of all subjects '''
    folderOut=folderIn +'/ConfidencesInTime/'
    createFolderIfNotExists(folderOut)

    AllSubjDiffPerf_test = np.zeros((len(GeneralParams.patients), 4* 9))
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFP_befSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFP_aftSeiz / FeaturesParams.winStep)
    for patIndx, pat in enumerate(GeneralParams.patients):
        filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+HDtype+'_'+FeatChType+'_TestPredictions.csv.gz'))
        for cv in range(len(filesAll)):
            data= readDataFromFile(filesAll[cv])
            if cv==0:
                trueLabels_AllCV = data[:,-1]
                # probabLabels_AllCV=data[:,1]
                predLabels_AllCV=data[ :,0:-1]
                dataSource_AllCV=np.ones(len(data[:, 0]))*(cv+1)
                numElems= len(predLabels_AllCV[0,:])
            else:
                trueLabels_AllCV = np.hstack((trueLabels_AllCV, data[:, -1]))
                # probabLabels_AllCV = np.hstack((probabLabels_AllCV, data[: ,1]))
                predLabels_AllCV=np.vstack((predLabels_AllCV,data[:,0:-1]))
                dataSource_AllCV= np.hstack((dataSource_AllCV, np.ones(len(data[:, 0]))*(cv+1)))

            # read distances from S and NS to calculate probability)
            distances=np.zeros((len(data[:,-1]), numElems, 2))
            distances[:,:,1] = readDataFromFile(filesAll[cv][0:-23]+'_DistancesFromSeiz_Test.csv.gz') #[:,0:-1]
            distances[:,:,0] = readDataFromFile(filesAll[cv][0:-23] + '_DistancesFromNonSeiz_Test.csv.gz') #[:,0:-1]
            probabLabels=np.zeros((len(data[:,-1]), numElems))
            for elem in range(numElems):
                probabLabels[:,elem]=calculateProbabilityFromClassDistances(distances[:,elem,:], data[:,elem])
            if cv == 0:
                probabLabels_AllCV= probabLabels
            else:
                probabLabels_AllCV = np.vstack((probabLabels_AllCV, probabLabels))


        outputName = folderOut + '/Subj' + pat  + '_'+HDtype+'_'+FeatChType+'_Appended_TestProbabilities.csv'
        saveDataToFile(probabLabels_AllCV, outputName, 'gzip')

        #PLOT PRED AND CONFIDENCE PER ELEMENT IN TIME
        plotInterp_PredAndConf([],predLabels_AllCV.astype(int), probabLabels_AllCV, trueLabels_AllCV, FeaturesParams, 'Subj' + pat  + '_'+HDtype+'_'+FeatChType, folderOut, elemNames)



def calculateProbabilityFromClassDistances(distances, predLabels):
    ''' Calculated probability if given distcances from two classes (0 or 1)
    distances are in shape of numSampl x numClasses'''
    dist_fromVotedOne = np.zeros((len(distances[:, 0])))
    dist_oppositeVotedOne = np.zeros((len(distances[:, 0])))
    indx = np.where(predLabels == 0)[0]
    dist_fromVotedOne[indx] = distances[indx, 0]
    dist_oppositeVotedOne[indx] = distances[indx, 1]
    indx = np.where(predLabels == 1)[0]
    dist_fromVotedOne[indx] = distances[indx, 1]
    dist_oppositeVotedOne[indx] = distances[indx, 0]
    probabilityLab = dist_oppositeVotedOne / (dist_oppositeVotedOne + dist_fromVotedOne)

    return probabilityLab