'''
library with various functions related to calculating EEG features
(mean amplitude, line length, frequency features, entropy, approximate zero crossing),
KL divergence, and plotting raw data or features
'''

import scipy
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import pywt
from entropy import *
import antropy as ant


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

def calculateFreqFeatures_oneDataWindow(data,  samplFreq):
    ''' function that calculates frewuency features relevant for epileptic seizure detection
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


def calculateEntropyFeatures_OneDataWindow(data,  samplFreq):
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


def calulateZCfeaturesRelative_oneCh(sigFilt, samplFreq, winLen, winStep, ZC_thresh_arr, sigRange):
    ''' feature that calculates zero-cross features for signal of one channel '''
    numFeat=len(ZC_thresh_arr)+1
    actualThrValues=np.zeros((numFeat-1))

    '''Zero-crossing of the original signal, counted in 1-second continuous sliding window'''
    # zeroCrossStandard[:,ch] = np.convolve(zero_crossings(sigFilt), np.ones(ZeroCrossFeatures.samplFreq), mode='same')
    x = np.convolve(zero_crossings(sigFilt), np.ones(samplFreq), mode='same')
    # zeroCrossStandard[:,ch] =calculateMovingAvrgMeanWithUndersampling(x, ZeroCrossFeatureParams.samplFreq)
    featVals= calculateMovingAvrgMeanWithUndersampling(x, int(samplFreq * winLen), int( samplFreq * winStep))
    zeroCrossFeaturesAll = np.zeros((len(featVals), numFeat ))
    zeroCrossFeaturesAll[:, 0]=featVals

    for EPSthrIndx, EPSthr in enumerate(ZC_thresh_arr):
        actualThrValues[ EPSthrIndx]=EPSthr #*sigRange
        # Signal simplification at the given threshold, and zero crossing count in the same way
        sigApprox = polygonal_approx(sigFilt, epsilon=EPSthr)#!!!! NEW TO HAVE RELATIVE THRESHOLDS *sigRange
        # axs[0].plot(sigApprox, sigFilt[sigApprox], alpha=0.6)
        sigApproxInterp = np.interp(np.arange(len(sigFilt)), sigApprox, sigFilt[sigApprox])
        # zeroCrossApprox[:,ch] = np.convolve(zero_crossings(sigApproxInterp), np.ones(ZeroCrossFeatures.samplFreq), mode='same')
        x = np.convolve(zero_crossings(sigApproxInterp), np.ones(samplFreq),  mode='same')
        # zeroCrossApprox[:, ch] =  calculateMovingAvrgMeanWithUndersampling(x, ZeroCrossFeatureParams.samplFreq)
        zeroCrossFeaturesAll[:,  EPSthrIndx + 1] = calculateMovingAvrgMeanWithUndersampling(x, int(samplFreq * winLen), int(samplFreq * winStep))

    return(zeroCrossFeaturesAll, actualThrValues)

def calculateOtherMLfeatures_oneCh(X, samplFreq, winLen, winStep):
    # numFeat = 56 #54 from Sopic2018 and LL and meanAmpl
    lenSig= len(X)
    segLenIndx = int(winLen * samplFreq)  # length of EEG segments in samples
    slidWindStepIndx = int( winStep * samplFreq)  # step of slidin window to extract segments in samples
    index = np.arange(0, lenSig - segLenIndx, slidWindStepIndx).astype(int)

    # featureValues=np.zeros((len(index), numFeat))
    for i in range(len(index)):
        sig = X[index[i]:index[i] + segLenIndx]
        freqFeat = calculateFreqFeatures_oneDataWindow(sig, samplFreq)
        meanAmpl = np.mean(np.abs(sig))
        LL = np.mean(np.abs(np.diff(sig)))
        if i==0:
            featureValues= np.hstack((meanAmpl, LL, freqFeat))
        else:
            featureValues=np.vstack((featureValues,  np.hstack((meanAmpl, LL, freqFeat))))
    return (featureValues)

def calculateChosenMLfeatures_oneCh(X, samplFreq, winLen, winStep, type):
    ''' function that calculate feature of interest for specific signal
    it discretizes signal into windows and calculates feature(s) for each window'''
    segLenIndx = int(winLen * samplFreq)  # length of EEG segments in samples
    slidWindStepIndx = int( winStep * samplFreq)  # step of slidin window to extract segments in samples
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
            featVal=calculateEntropyFeatures_OneDataWindow(np.copy(sig), samplFreq)
            numFeat = len(featVal)
        elif (type == 'Frequency'):
            featVal=calculateFreqFeatures_oneDataWindow(np.copy(sig), samplFreq)
            numFeat = len(featVal)

        if (i==0):

            featureValues=np.zeros((len(index), numFeat))
        featureValues[i,:]=featVal

    return (featureValues)

def zero_crossings(arr):
    """Returns the positions of zero-crossings in the derivative of an array, as a binary vector"""
    return np.diff(np.sign(np.diff(arr))) != 0

def calculateMovingAvrgMeanWithUndersampling(data, winLen, winStep):
    lenSig=len(data)
    index = np.arange(0, lenSig - winLen, winStep)

    segmData = np.zeros(len(index))
    for i in range(len(index)): #-1
        x = data[index[i]:index[i] + winLen]
        segmData[i]=np.mean(x)
    return(segmData)



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


def calculateAllFeatures(data,  samplFreq, winLen, winStep, chNames):
    EPS_thresh_arr= [16, 32, 64, 128, 256] #thresholds for AZC calculation
    featNames=['mean_ampl', 'line_len', 'p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel', 'p_gamma_rel',
             'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot',
             'azc_0', 'azc_16', 'azc_32', 'azc_64', 'azc_128', 'azc_256']
    (_, numCh)=data.shape

    sos = scipy.signal.butter(4, [1, 20], btype='bandpass', output='sos',fs=samplFreq)
    allsigFilt = scipy.signal.sosfiltfilt(sos, data, axis=0)
    # allsigFilt= data
    del data
    for ch in range(numCh):
        sigFilt = allsigFilt[:, ch]
        # calculate classical features
        featOther = calculateOtherMLfeatures_oneCh(np.copy(sigFilt),  samplFreq, winLen, winStep)
        if (ch == 0):
            AllFeatures = featOther
            featAllNames= [chNames[ch] +'_' + s for s in featNames]
        else:
            AllFeatures = np.hstack((AllFeatures, featOther))
            featAllNames = np.hstack(( featAllNames, [chNames[ch] +'_' + s for s in featNames]))

        # Approximate zero crossing features
        # Zero-crossing of the original signal, counted in 1-second continuous sliding window
        x = np.convolve(zero_crossings(sigFilt), np.ones(samplFreq), mode='same')
        ZCstd=calculateMovingAvrgMeanWithUndersampling(x, int(samplFreq * winLen), int( samplFreq * winStep))
        AllFeatures = np.hstack((AllFeatures, ZCstd.reshape((-1,1))))

        # Signal simplification at the given threshold, and zero crossing count in the same way
        for EPSthrIndx, EPSthr in enumerate(EPS_thresh_arr):
            sigApprox = polygonal_approx(sigFilt, epsilon=EPSthr)
            sigApproxInterp = np.interp(np.arange(len(sigFilt)), sigApprox, sigFilt[sigApprox])
            x = np.convolve(zero_crossings(sigApproxInterp), np.ones(samplFreq),  mode='same')
            ZCapp= calculateMovingAvrgMeanWithUndersampling(x, int(samplFreq * winLen), int( samplFreq * winStep))
            AllFeatures = np.hstack((AllFeatures, ZCapp.reshape((-1, 1))))

    # print('SUM' , np.sum(AllFeatures,0))
    # print('NAN SUM', np.nansum(AllFeatures, 0))

    return ( AllFeatures, featAllNames, featNames)



def calcHistogramValues(sig, segmentedLabels, histbins):
    '''takes one window of signal - all ch and labels, separates seiz and nonSeiz and
    calculates histogram of values  during seizure and non seizure '''
    numBins=int(histbins)
    sig2 = sig[~np.isnan(sig)]
    sig2 = sig2[np.isfinite(sig2)]
    # maxValFeat=np.max(sig)
    # binBorders=np.arange(0, maxValFeat+1, (maxValFeat+1)/numBins)

    # sig[sig == np.inf] = np.nan
    indxs=np.where(segmentedLabels==0)[0]
    nonSeiz = sig[indxs]
    nonSeiz = nonSeiz[~np.isnan(nonSeiz)]
    try:
        nonSeiz_hist = np.histogram(nonSeiz, bins=numBins, range=(np.min(sig2), np.max(sig2)))
    except:
        print('Error with hist ')

    indxs = np.where(segmentedLabels >= 1)[0]
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
    p=p+np.ones(len(p))*delta #to avoid case when they are 0
    q=q+np.ones(len(p))*delta
    res=sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))
    return res

def js_divergence(p,q):
    m=0.5* (p+q)
    res=0.5* kl_divergence(p,m) +0.5* kl_divergence(q,m)
    return (res)


def plot_rawEEGdata(data, labels, samplFreq, winLen, winStep, chNames, folderOut, nameOut):
    (lenSig, numCh)=data.shape
    labelsNorm=labels/np.max(labels)
    # plot
    fig1 = plt.figure(figsize=(10, 10), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    xValues = np.arange(0, lenSig, 1)

    ax1 = fig1.add_subplot(gs[0, 0])
    for ch in range(numCh):
        dataNorm= (data[:,ch]-np.nanmin(data[:,ch])) /( np.nanmax(data[:,ch])- np.nanmin(data[:,ch]))
        ax1.plot(xValues*winStep, dataNorm*0.8+ ch,'k')
    ax1.plot(xValues * winStep, labelsNorm*numCh , 'r')
    ax1.set_xlabel('Time [s]')
    ax1.set_yticks(np.arange(0, numCh, 1))
    ax1.set_yticklabels(chNames, fontsize=12 * 0.8)
    ax1.set_title(nameOut)
    # ax1.set_ylabel('Kl diverg')
    ax1.grid()

    fig1.show()
    fig1.savefig(folderOut +'/'+ nameOut+ '_RawEEGData.png', bbox_inches='tight')
    plt.close(fig1)


def calculate_KLdivergence(labels, features, chNames, featNames, outName):
    # calculate histograms per seizure and non seizure
    numBins = 30
    numFeatTot = len(features[0, :])
    numCh = len(chNames)
    numFeatPerCh = int(numFeatTot / numCh)
    KLdiverg = np.zeros((numCh, numFeatPerCh))
    JSdiverg = np.zeros((numCh, numFeatPerCh))
    for f in range(numFeatTot):
        ch = int(f / numFeatPerCh)
        feat = np.mod(f, numFeatPerCh)
        if (np.nansum(features[:, f])>0): #check that not all values of features are nan
            (SeizHist, nonSeizHist) = calcHistogramValues(features[:, f], labels, numBins)
            KLdiverg[ch, feat] = kl_divergence(nonSeizHist[0], SeizHist[0])
            JSdiverg[ch, feat] = js_divergence(nonSeizHist[0], SeizHist[0])
        else:
            KLdiverg[ch, feat] =np.nan
            JSdiverg[ch, feat] =np.nan

    # plot
    plotKLDivergence(KLdiverg, JSdiverg, featNames, outName)

    return KLdiverg, JSdiverg


def plotKLDivergence(KLdiverg, JSdiverg,featNames , outName):
    (folder, fname) = os.path.split(outName)
    numFeatPerCh=len(featNames)
    # plot
    fig1 = plt.figure(figsize=(10, 6), constrained_layout=False)
    gs = GridSpec(2, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    xValues = np.arange(0, numFeatPerCh, 1)

    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xValues, np.nanmean(KLdiverg, 0), yerr=np.nanstd(KLdiverg, 0), fmt='b')
    # ax1.set_xlabel('Features')
    # ax1.set_xticks(xValues)
    # ax1.set_xticklabels(featNames, fontsize=12 * 0.8, rotation=45)
    ax1.set_ylabel('Kl diverg')
    ax1.set_title(fname)
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 0])
    ax1.errorbar(xValues, np.mean(JSdiverg, 0), yerr=np.std(JSdiverg, 0), fmt='b')
    # ax1.set_xlabel('Features')
    ax1.set_ylabel('JS diverg')
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(featNames, fontsize=12 * 0.8, rotation=45)
    ax1.grid()
    fig1.show()
    fig1.savefig(outName+ '_FeatDiverg.png', bbox_inches='tight')
    plt.close(fig1)