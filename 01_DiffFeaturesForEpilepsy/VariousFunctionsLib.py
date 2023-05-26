''' library including various functions for HD project but not necessarily related to HD vectors'''
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

import os
import glob
import csv
import math
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pywt
from entropy import *
import scipy


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

def encoding_symbolization_CWT(data, SegSymbParams, EEGfreqBands,SigInfoParams):
    ''' given one discretized window of data function calculates CWT feature value
    CWT feature value is in this case index of frequency band with the highest energy '''

    percSig1= 0.4; percSig2 = 0.6;
    energyPerBandPerc = np.zeros((SegSymbParams.CWTlevel))
    if (SegSymbParams.CWTlevel == 4):
        freqArray_CWT = EEGfreqBands.freqVal4Lev
    elif (SegSymbParams.CWTlevel == 10):
        freqArray_CWT = EEGfreqBands.freqVal10Lev
    elif (SegSymbParams.CWTlevel == 20):
        freqArray_CWT = EEGfreqBands.freqVal20Lev
    elif (SegSymbParams.CWTlevel == 30):
        freqArray_CWT = EEGfreqBands.freqVal30Lev
    centralFreq = pywt.central_frequency(SegSymbParams.CWTfilterName, precision=12)
    scaleArray = centralFreq * SigInfoParams.samplFreq/ freqArray_CWT

    #CWT decomposition and energy per band
    # normalizing amplitude values
    data = data / np.max(data)
    # replicate???
    coeffs, freqs = pywt.cwt(data, scaleArray, SegSymbParams.CWTfilterName, 1 / SigInfoParams.samplFreq)
    # extracting coeffs for each freq band  and calculating energy in that band
    for b in range(SegSymbParams.CWTlevel ):
        c = coeffs[b, :]
        s1 = floor(len(c) * percSig1)
        s2 = floor(len(c) * percSig2)
        energyPerBandPerc[b] = np.sqrt(sum(abs(c[s1:s2]) ** 2) / (s2 - s1))
    # calculating percentages per band
    energyPerBandPerc = energyPerBandPerc/ sum(energyPerBandPerc)
    #energyPerBandPerc=calculateCWT_onEachSegment(data, SegSymbParams, EEGfreqBands, SigInfoParams)

    #normalize with noise composition if needed
    if (SegSymbParams.noiseNormType=='noiseNorm'):#noNoiseNorm
        # load noise CWT decomposition values
        fullPath = '../01_CWTnoise/CWTdecompositionNoise_' + SegSymbParams.CWTfilterName + '_numFreqLev=' + str(
            SegSymbParams.CWTlevel) + '.csv'
        if os.path.isfile(fullPath):
            reader = csv.reader(open(fullPath, "r"))
            data0 = list(reader)
            energyPerBand_Noise = np.squeeze(np.array(data0).astype("float"))
        energyPerBandPerc= energyPerBandPerc/ energyPerBand_Noise #normalized with noise
        energyPerBandPerc= energyPerBandPerc/ np.sum(energyPerBandPerc) #normaize so that sum is 1

    #symbolization
    #if (SegSymbParams.symbolType == 'CWTabs'):
    en = energyPerBandPerc.tolist()
    symbol= en.index(max(en) )
    # else:  #'CWTratio' #would need cwt decomposition of all nonSeizure and normalize wih it - not implemented now
    # symbol = calculateRepresentationUsingCWT_ver2(energyPerBandPerc, segmentedLabels, 'ratio')

    return (symbol, energyPerBandPerc)

def encoding_symbolization_entropy(x, SegSymbParams):
    ''' given one discretized window of data function calculates entropy feature value
        entropy feature value is in this case discretized value (between 0 and 1) of entropy of that window
        many different entropy measures are available '''

    Xmin = 0; Xmax = 1.0;
    # segmenting based on different entropy types
    if (SegSymbParams.entropyType == 'perm_entropy'):
        res = perm_entropy(x, order=3, normalize=True)  # Permutation entropy
        Xmin=0.8; Xmax=1.0;
    elif (SegSymbParams.entropyType == 'spectral_entropy'):
        res = spectral_entropy(x, 100, method='welch', normalize=True)  # Spectral entropy
        Xmin = 0.3;Xmax = 1.0;
    elif (SegSymbParams.entropyType == 'svd_entropy'):
        res = svd_entropy(x, order=3, delay=1, normalize=True)  # Singular value decomposition entropy
        Xmin = 0.2; Xmax = 1.0;
    elif (SegSymbParams.entropyType == 'app_entropy'):
        res = app_entropy(x, order=2, metric='chebyshev')  # Approximate entropy
        Xmin = 0.2; Xmax = 1.8;
    elif (SegSymbParams.entropyType == 'sample_entropy'):
        res = sample_entropy(x, order=2, metric='chebyshev')  # Sample entropy
        Xmin = 0.2; Xmax = 2.2;
    # elif (SegSymbParams.entropyType == 'lziv_complexity'):
    #     res = lziv_complexity('01111000011001', normalize=True)  # Lempel-Ziv complexity
    elif (SegSymbParams.entropyType == 'spectral_entropy'):
        res = spectral_entropy(x, 100, method='welch', normalize=True)
    #discretizing
    delta=(Xmax-Xmin)/SegSymbParams.numSegLevels
    symbol= np.floor((res-Xmin) / delta)
    if (symbol<0):
        symbol=0
    elif (symbol>=SegSymbParams.numSegLevels):
        symbol=SegSymbParams.numSegLevels-1
    if symbol==np.nan: #in case nan have to replace with some number
        symbol=0
    return (symbol, res)

def encoding_symbolization_amplitude(x, SegSymbParams,ch):
    ''' given one discretized window of data function calculates amplitude feature value
    amplitude feature value is in this case defined as normalized and discretized mean value of signal in this window
    function also keeps track in real time min, max and mean values of signal so that it can normalize it
    (without knowing total min and max values of whole signal - because this is not implementable in real time embedded system)'''

    # mean value of this segment
    # Xmean = np.mean(x)
    Xmean= np.mean(np.abs(x))
    if (SegSymbParams.cntForAmplitude != 0):
        # updeating total mean  and min values
        try:
            SegSymbParams.meanValueSignal[ch] = (SegSymbParams.meanValueSignal[ch] * SegSymbParams.cntForAmplitude +Xmean) / (SegSymbParams.cntForAmplitude + 1)
        except:
            print('sth wrong with num ch')
        if (Xmean < SegSymbParams.minValueSignal[ch]):
            SegSymbParams.minValueSignal[ch] = Xmean

        # normalize with average value of whole signal in total
        Xnorm = (SegSymbParams.meanValueSignal[ch] - SegSymbParams.minValueSignal[ch]) / (Xmean- SegSymbParams.minValueSignal[ch])
    else:
        Xnorm = 1.0
        SegSymbParams.meanValueSignal[ch] = Xmean
        SegSymbParams.minValueSignal[ch] = Xmean

    # SegSymbParams.cntForAmplitude = SegSymbParams.cntForAmplitude + 1 * this shoould be done only after all ch are processed

    # segmenting  - equal spacing
    if (SegSymbParams.amplitudeBinsSpacing == 'equal'):
        delta = SegSymbParams.amplitudeRangeFactor / SegSymbParams.numSegLevels  # assumption that mean will not be more then 2xless them true mean of whole signal
        symbol = np.floor(Xnorm / delta)
        if (symbol < 0):
            symbol = 0
        elif (symbol >= SegSymbParams.numSegLevels):
            symbol = SegSymbParams.numSegLevels - 1
    else:
        # segmentin - non equal spacing   but only works for now if range is 2
        if (SegSymbParams.numSegLevels == 4):
            #thresholds = [0.75, 1.0, 1.25, 2.0]
            thresholds = [0.25, 0.5, 1.0, 2.0]
        elif (SegSymbParams.numSegLevels == 10):
            #thresholds = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
            thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5,  2.0]
        elif (SegSymbParams.numSegLevels == 20):
            # thresholds = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15, 1.2,
            #               1.25, 1.5, 1.75, 2.0]
            thresholds = [0.1,0.2,0.25,0.3,0.35,0.4,0.45, 0.5,0.55,0.6,0.65, 0.7,0.8,0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
        elif (SegSymbParams.numSegLevels == 30):
            #thresholds = [0.25, 0.5, 0.6, 0.7, 0.76, 0.82, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0, 1.01,1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.18, 1.24, 1.3, 1.4, 1.5, 1.75, 2.0]
            thresholds = [0.1,0.2,0.25,0.3,0.35,0.4,0.45, 0.5,0.55,0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9,0.95, 1.0, 1.05,1.1, 1.15,1.2, 1.3,1.4, 1.5, 1.6,1.7,1.8,1.9, 2.0]
        symbol = np.sum(np.asarray(thresholds) < Xnorm)-1

    return (symbol, Xmean)



def calculateAllFeatures(sigTensor,  SegSymbParams, EEGfreqBands, SigInfoParams):
    '''function that calculates CWT, entropy and amplitude features from given discretized window and returnes them all'''

    sigArr = sigTensor.cpu().numpy()
    (numCh, sigLen )=sigArr.shape
    dataEncoded_Amplitude =np.zeros((numCh))
    dataEncoded_Entropy = np.zeros((numCh))
    dataEncoded_CWT = np.zeros((numCh))
    for ch in range(numCh):
        sig=sigArr[ch,:]
        (dataEncoded_CWT[ch], p) = encoding_symbolization_CWT(sig, SegSymbParams, EEGfreqBands, SigInfoParams)
        (dataEncoded_Entropy[ch], p) = encoding_symbolization_entropy(sig, SegSymbParams)
        (dataEncoded_Amplitude[ch], p) = encoding_symbolization_amplitude(sig, SegSymbParams,ch)

    SegSymbParams.cntForAmplitude = SegSymbParams.cntForAmplitude + 1
    dataEncoded=np.vstack((dataEncoded_Amplitude, dataEncoded_Entropy, dataEncoded_CWT))
    return dataEncoded


def symbolizeSegment(sigTensor, SegSymbParams, EEGfreqBands, SigInfoParams):
    '''function reused from different project - it basically calculates one discretized feature value that represents given window of data
    it calls specific functions to calculate values of features based on feature (or features chosen)'''

    try:
        sigArr = sigTensor.cpu().numpy()  # .transpose()
    except:
        print('CPU error')
    (numCh, sigLen) = sigArr.shape
    dataEncoded = np.zeros((numCh))
    for ch in range(numCh):
        sig = sigArr[ch, :]
        if 'CWT' in SegSymbParams.symbolType:
            (dataEncoded[ch] , p) = encoding_symbolization_CWT(sig, SegSymbParams, EEGfreqBands, SigInfoParams)
        elif 'Entropy' in SegSymbParams.symbolType:
            (dataEncoded[ch] , p) = encoding_symbolization_entropy(sig, SegSymbParams)
        elif 'Amplitude' in SegSymbParams.symbolType:
            amplitude = encoding_symbolization_amplitude(sig, SegSymbParams, ch)
            (dataEncoded[ch] , p) = amplitude
    if 'Amplitude' in SegSymbParams.symbolType:
        SegSymbParams.cntForAmplitude = SegSymbParams.cntForAmplitude + 1
    return dataEncoded


def calculateMLfeatures(data,  samplFreq):
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

def func_normAmplitudeValsSamplBySampl(x,SegSymbParams):
    ''' function that calculates normalized value of given sample and continuously updates min, max and mean values
    to be able to in real time normalize values (without knowing total min and max - which is not feasible in real time applications)'''
    X=np.abs(x)
    numCh=len(X)
    Xnorm=np.zeros((numCh))
    symbol = np.zeros((numCh))

    for ch in range(numCh):
        if (SegSymbParams.cntForAmplitude != 0):
            # updeating total mean  and min values
            SegSymbParams.meanValueSignal[ch] = (SegSymbParams.meanValueSignal[ch] * SegSymbParams.cntForAmplitude +X[ch]) / (SegSymbParams.cntForAmplitude + 1)
            if (X[ch] < SegSymbParams.minValueSignal[ch]):
                SegSymbParams.minValueSignal[ch] = X[ch]
            # normalize with average value of whole signal in total
            Xnorm[ch] = (SegSymbParams.meanValueSignal[ch] - SegSymbParams.minValueSignal[ch]) / (X[ch]- SegSymbParams.minValueSignal[ch])
        else:
            Xnorm[ch] = 1.0
            SegSymbParams.meanValueSignal[ch] = X[ch]
            SegSymbParams.minValueSignal[ch] = X[ch]


        # segmenting  - equal spacing
        if (SegSymbParams.amplitudeBinsSpacing == 'equal'):
            delta = SegSymbParams.amplitudeRangeFactor / SegSymbParams.numSegLevels  # assumption that mean will not be more then 2xless them true mean of whole signal
            symbol[ch] = np.floor(Xnorm[ch] / delta)
            if (symbol[ch] < 0 or np.isnan(symbol[ch])):
                symbol[ch] = 0
            elif (symbol[ch]  >= SegSymbParams.numSegLevels):
                symbol[ch]  = SegSymbParams.numSegLevels - 1
        else:
            # segmentin - non equal spacing   but only works for now if range is 2
            if (SegSymbParams.numSegLevels == 4):
                #thresholds = [0.75, 1.0, 1.25, 2.0]
                thresholds = [0.25, 0.5, 1.0, 2.0]
            elif (SegSymbParams.numSegLevels == 10):
                #thresholds = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
                thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5,  2.0]
            elif (SegSymbParams.numSegLevels == 20):
                # thresholds = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15, 1.2,
                #               1.25, 1.5, 1.75, 2.0]
                thresholds = [0.1,0.2,0.25,0.3,0.35,0.4,0.45, 0.5,0.55,0.6,0.65, 0.7,0.8,0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
            elif (SegSymbParams.numSegLevels == 30):
                #thresholds = [0.25, 0.5, 0.6, 0.7, 0.76, 0.82, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0, 1.01,1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.18, 1.24, 1.3, 1.4, 1.5, 1.75, 2.0]
                thresholds = [0.1,0.2,0.25,0.3,0.35,0.4,0.45, 0.5,0.55,0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9,0.95, 1.0, 1.05,1.1, 1.15,1.2, 1.3,1.4, 1.5, 1.6,1.7,1.8,1.9, 2.0]
            symbol[ch]  = np.sum(np.asarray(thresholds) < Xnorm[ch])-1
    symbol=symbol.astype(int)
    SegSymbParams.cntForAmplitude = SegSymbParams.cntForAmplitude + 1
    return (symbol, Xnorm)

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
            labelsPerSeg[s] = math.ceil(np.average(labelsPerSample[s, :]))
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
    #first classifying as true 1 if at laest  GeneralParams.seizureStableLenToTest in a row is 1
    for i in range(seizureStableLenToTestIndx, int(len(prediction))):
        s= sum( prediction[i-seizureStableLenToTestIndx+1: i+1] )/seizureStableLenToTestIndx
        if (s>= seizureStablePercToTest):  #and prediction[i]==1
            smoothLabelsStep1[i]=1
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

def reportRelevantPerformanceMeasures_v3(y_true, y_pred,SegSymbParams, GeneralParams):
    ''' calculates different performance measures - slightly adapted for epileptic seizures
    - measures for episodes and duration
    - also number of FP per hour '''

    y_true=np.squeeze(y_true.astype(int))
    y_pred=np.squeeze(y_pred.astype(int))

    ##  PERFORMANCES ON THE LEVEL OF SEIZURE EPISODES
    #sensitivity - true positive rate
    (sensitivity,  numPredSeizure, numTrueSeizure)= calcualteSensitivity(y_pred, y_true,  GeneralParams, SegSymbParams)
    numFP= calcualteNumFPperHour(y_pred, y_true, GeneralParams, SegSymbParams)
    numLabelsPerHour=60*60/SegSymbParams.slidWindStepSec
    timeDurOfLabels=len(y_true)/numLabelsPerHour
    if (timeDurOfLabels!=0):
        numFPperHour=numFP/timeDurOfLabels
    else:
        numFPperHour=np.nan

    try:
        confMat=sklearn.metrics.confusion_matrix(y_true, y_pred,labels=[0,1])
    except:
        print('problem with confusion matrix')


    #sensitivity
    #sensitivity = confMat[0, 0] / (confMat[0, 0] + confMat[0, 1])
    #specificity
    #specificity0 = confMat[1,1]/(confMat[1,0]+confMat[1,1]) #not correct!!
    if (len(confMat.ravel())<4):
        print('confMat probelm:', confMat, confMat.ravel())
        print('y_true:',y_true)
        print('y_pred:',y_pred)
    tn, fp, fn, tp = confMat.ravel()
    specificity=tn/(tn+fp)

    #numFPperHourCorr is opposite - if 1 then it is no FP per hour and if 0 > thres 1 per hour
    numFalsePosPerHourCorr = func_normNumFP(numFPperHour, GeneralParams)

    #trying to combine sensitivity and numFPperHour for episodes
    # combinedMeasure = sensitivity * numFalsePosPerHourCorr   #seems like too much deflates numbers
    # combinedMeasure= (sensitivity + numFalsePosPerHourCorr) / 2 #not good because somehow inflates values
    combinedMeasure = np.sqrt(sensitivity * numFalsePosPerHourCorr) #gmean
    # if (numFPperHour*24>GeneralParams.numFPperDayThr):
    #     print('numFPperHour',numFPperHour)
    #     print('numFalsePosPerHourCorr', numFalsePosPerHourCorr)
    #     print('sensitivity', sensitivity)
    #     print('combinedMeasure', combinedMeasure)


    ## F1 scores
    ## F1 score for seizure episodes
    # F1=2*sensitivity *precision /(sensitivity+precision)
    #sensitivity=TP/(TP+FN)=true predicted seizures/total num seizures
    #precision=TP/(TP+FP)=true predicted seizures/total num of predicted seizures
    if (numPredSeizure+numFP)==0:
        precision=0
    else:
        precision=numPredSeizure/(numPredSeizure+numFP)
    if ((sensitivity+precision)==0):
        F1score_episodes=0
    else:
        F1score_episodes=2*sensitivity *precision /(sensitivity+precision)

    ## PERFORMANCE ON THE LEVEL OF DURATION
    (F1score_duration, sensitivity_duration, precision_duration)=calcualate_F1score_forSeizureDuration(y_pred, y_true)

    ## TOTAL AND BALANCED ACCURACY
    totAccuracy=sklearn.metrics.accuracy_score(y_true, y_pred)
    balancedAccuracy=sklearn.metrics.balanced_accuracy_score(y_true, y_pred)

    allPerformances=[sensitivity,precision, F1score_episodes, sensitivity_duration , precision_duration, F1score_duration,
            numFPperHour, numFalsePosPerHourCorr, combinedMeasure,  totAccuracy, balancedAccuracy , specificity]
    return allPerformances

def func_normNumFP(numFP0, GeneralParams):
    ''' instead of numFPperHours calculates value that is 1 if no numFPperHour and 0 if mor than threshold GeneralParams.numFPperDayThr'''
    numFP=numFP0*24 #numFPperDay
    if (numFP>GeneralParams.numFPperDayThr):
        numFP=0
    else:
        numFP=1-numFP/GeneralParams.numFPperDayThr

    # thresholdNumFP=GeneralParams.numFPperDayThr/24
    # if (numFP0>thresholdNumFP):
    #     numFP=0
    # else:
    #     numFP=1-numFP0/thresholdNumFP
    return numFP

def calcualteSensitivity(y_pred_smoothed, y_true, GeneralParams, SegSymbParams):
    ''' caculates sensitivity on the level of seizure episodes detection '''
    #true positive rate
    numTrueSeizure=0
    numPredSeizure=0
    for i in range(1,len(y_true)):
        if (y_true[i]==1):
            if (y_true[i-1]==0 or i==1): #if new true seizure started
                numTrueSeizure=numTrueSeizure+1
                #find end of the seizure
                j=i
                while (y_true[j]==1 and j<len(y_true)-1):
                    j=j+1
                #check in range when seizure [i,j] if y_pred_smoothed is any time 1
                if (i> GeneralParams.timeBeforeSeizureConsideredAsSeizure/SegSymbParams.slidWindStepSec+1):
                    suma=np.sum(y_pred_smoothed[i-int(GeneralParams.timeBeforeSeizureConsideredAsSeizure/SegSymbParams.slidWindStepSec):j]) #aso check for some time before seizure
                else:
                    suma = np.sum(y_pred_smoothed[0:j])
                if (suma>=1):
                    numPredSeizure=numPredSeizure+1

    if (numTrueSeizure!=0):
        sensitivity=numPredSeizure/numTrueSeizure
    else:
        sensitivity=np.nan
    return (sensitivity, numPredSeizure, numTrueSeizure)

def calcualteNumFPperHour(y_pred_smoothed, y_true, GeneralParams, SegSymbParams):
    ''' estimates what would be numFPperHour from give true and predicted labels '''
    numFP = 0
    prevSeizureEnd = -15
    for i in range(1,len(y_pred_smoothed)):
        if (y_pred_smoothed[i] == 1  and y_pred_smoothed[i-1] == 0): #start of new predicted seizure
            #if ( (i-prevSeizureEnd)>GeneralParams.distanceBetween2Seizures*SegSymbParams.slidWindStepSec):  # if  seizure started and enough far from previous one
            startSeizure=i
            # find end of the seizure
            j = i
            while (y_pred_smoothed[j] == 1  and j<len(y_pred_smoothed)-1 ):
                j = j + 1
            #check if any true label 1
            sumTrue=np.sum(y_true[startSeizure:j])
            if (sumTrue==0): #this is really not seizure
                numFP = numFP + 1
            prevSeizureEnd=j
            i=j

        #previous way of detecting
        # if (y_pred_smoothed[i] == 1   and y_true[i]==0): #false positive
        #     if (y_pred_smoothed[i-1] == 0 and (i-prevSeizureEnd)>GeneralParams.distanceBetween2Seizures*SegSymbParams.slidWindStepSec):  # if  seizure started and enough far from previous one
        #         numFP = numFP + 1
        #         # find end of the seizure
        #         j = i
        #         while (y_true[j] == 1):
        #             j = j + 1
        #         prevSeizureEnd=j
    return numFP

def calcualate_F1score_forSeizureDuration(y_pred_smoothed, y_true):
    '''calculates performance metricses on the level of seizure duration '''
    #total true seizure durations
    durationTrueSeizure=np.sum(y_true)

    #total predicted seizure duration
    durationPredictedSeizure=np.sum(y_pred_smoothed)

    #total duration of true predicted seizure
    temp=2*y_true-y_pred_smoothed #where diff is 1 here both true and apredicted label are 1
    indx=np.where(temp==1)
    durationTruePredictedSeizure=np.squeeze(np.asarray(indx)).size

    if (durationTrueSeizure==0):
        sensitivity=0
        print('No seizure in test data')
    else:
        sensitivity=durationTruePredictedSeizure/durationTrueSeizure
    if (durationPredictedSeizure==0):
        precision=0
        print('No predicted seizure in test data')
    else:
        precision=durationTruePredictedSeizure/durationPredictedSeizure
    if ((sensitivity + precision)==0):
        F1score_duration=0
        print('Sensitivity and prediction are 0 in test data')
    else:
        F1score_duration = 2 * sensitivity * precision / (sensitivity + precision)

    return(F1score_duration,sensitivity, precision )

def func_plotPredictionsEachPerson(folderIn, GeneralParams, SegSymbParams, HDparams):
    ''' function that plots predictions for all files of one person
    plots true label, raw predictions and predictions after different smoothings
    this is just for visual inspection purposes'''

    folderOut = folderIn + '/Plots'
    createFolderIfNotExists(folderOut)

    # plotting
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    if (len(GeneralParams.patients) == 10):
        gs = GridSpec(5, 2, figure=fig1)
        fact = 2
    elif (len(GeneralParams.patients) == 24):
        gs = GridSpec(6, 4, figure=fig1)
        fact = 4
    elif (len(GeneralParams.patients) == 16):
        gs = GridSpec(4, 4, figure=fig1)
        fact = 4
    fig1.subplots_adjust(wspace=0.4, hspace=0.4)
    fig1.suptitle('Prediction of seizures for all subj')

    patIndx = -1
    for pat in GeneralParams.patients:
        patIndx = patIndx + 1
        fileList = np.sort(glob.glob(folderIn + '/chb_' + pat + '*True&PredLabels.csv'))
        if (len(fileList) == 0):
            fileList = np.sort(glob.glob(folderIn + '/chb' + pat + '*True&PredLabels.csv'))

        numFiles = len(fileList)
        print('-- Patient:', pat, 'NumSeizures:', numFiles)
        ax1 = fig1.add_subplot(gs[int(patIndx / fact), np.mod(patIndx, fact)])
        fileIndx = -1
        for fileIn in fileList:  # num CV as number of seizure files - leave this one out
            fileIndx = fileIndx + 1
            reader = csv.reader(open(fileIn, "r"))
            data0 = list(reader)
            data = np.array(data0).astype("float")
            yTrue = data[:, 3] * 0.5 + fileIndx
            yPred_NoSmooth = data[:, 0] * 0.5 + fileIndx
            # yPred_OurSmoothing_step1=data[:,1]*0.4+fileIndx
            # yPred_OurSmoothing_step2=data[:,2]*0.3+fileIndx
            seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
            seizureStablePercToTest = GeneralParams.seizureStablePercToTest
            distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
            (yPred_OurSmoothing_step2, yPred_OurSmoothing_step1) = smoothenLabels(data[:, 0],seizureStableLenToTestIndx, seizureStablePercToTest,distanceBetweenSeizuresIndx)
            yPred_OurSmoothing_step1 = yPred_OurSmoothing_step1 * 0.4 + fileIndx
            yPred_OurSmoothing_step2 = yPred_OurSmoothing_step2 * 0.3 + fileIndx


            xvalues = np.arange(0, len(yTrue), 1) * SegSymbParams.slidWindStepSec
            ax1.plot(xvalues, yPred_NoSmooth, 'k', label='NoSmooth')
            ax1.plot(xvalues, yPred_OurSmoothing_step1, 'b', label='OurSmoothing_step1')
            ax1.plot(xvalues, yPred_OurSmoothing_step2, 'm', label='OurSmoothing_step2')
            ax1.plot(xvalues, yTrue, 'r')  # label='Performance')
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('CVs')
            ax1.set_title('Subj ' + str(pat))
            if (patIndx == 0 and fileIndx == 0):
                ax1.legend()
            ax1.grid()
    if (GeneralParams.plottingON == 1):
        fig1.show()
    fig1.savefig(folderOut + '/AllSubj_PredictionsForEachCV_AllSmoothingTypes.png')
    plt.close(fig1)

def func_plotTestResults_AllSubj(SigInfoParams, SegSymbParams, GeneralParams, HDParams, EEGfreqBands, folderIn, folderOut):
    ''' plots various performances for each subject for different labels smoothing types
        performances that are plotted:
        'sensitivity_episodes', 'precision_episodes', 'F1score_episodes', 'sensitivity_duration', 'precision_duration','F1score_duration',\
        'numFPperHour', 'numFalsePosPerHourCorr', 'combinedMeasure', 'totAccuracy', 'balancedAccuracy','specificity',\
        'sensitivity_episodes', 'specificity_duration', 'delay'
    '''
    folderOutPlots = folderOut + '/Plots/'
    createFolderIfNotExists(folderOutPlots)
    # #calculate our and original performance (both in one file) for all smoothing options (noSmooth, origSmooth, ourSmoothStep1, ourSmoothStep2)
    func_measureAllPerformanceForDiffSmooth(folderOut, folderOutPlots, GeneralParams, SegSymbParams, HDParams)
    # plotting
    func_plotAllPerformances_AllSubj_variableSmoothing(folderOut, folderOutPlots, 'NoSmooth', GeneralParams.plottingON)
    func_plotAllPerformances_AllSubj_variableSmoothing(folderOut, folderOutPlots, 'OurSmoothStep1', GeneralParams.plottingON)
    func_plotAllPerformances_AllSubj_variableSmoothing(folderOut, folderOutPlots, 'OurSmoothStep2', GeneralParams.plottingON)
    # plot for all smoothing on one graph
    smoothingTypesArray = ['NoSmooth', 'OurSmoothStep1', 'OurSmoothStep2']
    func_plotAllPerformances_AllSubj_AllSmoothingTypes(folderOut, folderOutPlots, smoothingTypesArray, GeneralParams.plottingON)


def func_measureAllPerformanceForDiffSmooth(folderIn, folderOut, GeneralParams, SegSymbParams, HDparams):
    ''' loads files with predictions and true labels and calculates various performance measures
    also tests different label smoothing approaches '''

    performancesAll_allSubj_noSmooth = np.zeros((len(GeneralParams.patients), 15))   # 12 diff performance measures
    performancesAll_allSubj_ourSmoothStep1 = np.zeros((len(GeneralParams.patients), 15))
    performancesAll_allSubj_ourSmoothStep2 = np.zeros((len(GeneralParams.patients), 15))

    patIndx = -1
    for pat in GeneralParams.patients:
        patIndx = patIndx + 1
        if (GeneralParams.PersGenApproach == 'generalized'):
            fileList = np.sort(glob.glob(folderIn + '/chb' + pat + '*True&PredLabels.csv'))
        else:
            fileList = np.sort(glob.glob(folderIn + '/chb_' + pat + '*True&PredLabels.csv'))
            if (len(fileList) == 0):
                fileList = np.sort(glob.glob(folderIn + '/chb' + pat + '*True&PredLabels.csv'))
        numFiles = len(fileList)
        print('-- Patient:', pat, 'NumSeizures:', numFiles)
        performancesAll_eachSubj_allCV_noSmooth = np.zeros((numFiles, 12))  # 12 diff performance measures
        performancesAll_eachSubj_allCV_origSmooth = np.zeros((numFiles, 12))
        performancesAll_eachSubj_allCV_ourSmoothStep1 = np.zeros((numFiles, 12))
        performancesAll_eachSubj_allCV_ourSmoothStep2 = np.zeros((numFiles, 12))
        fileIndx = -1
        # for fileIn in np.sort(glob.glob(folderIn + '/chb_' + pat + '*True&PredLabels.csv')):  # num CV as number of seizure files - leave this one out
        for fileIn in fileList:  # num CV as number of seizure files - leave this one out
            fileIndx = fileIndx + 1
            reader = csv.reader(open(fileIn, "r"))
            data0 = list(reader)
            data = np.array(data0).astype("float")
            yTrue = data[:, 3]
            yPred = data[:, 0]

            # yPred_SmoothOurStep1 = data[:, 1]
            # yPred_SmoothOurStep2 = data[:, 2]
            seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
            seizureStablePercToTest = GeneralParams.seizureStablePercToTest
            distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
            (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(yPred, seizureStableLenToTestIndx,seizureStablePercToTest, distanceBetweenSeizuresIndx)

            # calculate perfomences for all smoothing types
            # no smooth
            performancesAll_eachSubj_allCV_noSmooth[fileIndx, 0:12] = reportRelevantPerformanceMeasures_v3(yTrue, yPred, SegSymbParams,GeneralParams)
            # ourSmooth_step1
            performancesAll_eachSubj_allCV_ourSmoothStep1[fileIndx, 0:12] = reportRelevantPerformanceMeasures_v3(yTrue, yPred_SmoothOurStep1, SegSymbParams, GeneralParams)
            # ourSmooth_step2
            performancesAll_eachSubj_allCV_ourSmoothStep2[fileIndx, 0:12] = reportRelevantPerformanceMeasures_v3(yTrue, yPred_SmoothOurStep2, SegSymbParams, GeneralParams)

        # saving all performances for all types of smoothing - for this subject
        outputName = folderIn + '/chb_' + pat + '_AllPerformances_ForAllCV_NoSmooth.csv'
        np.savetxt(outputName, performancesAll_eachSubj_allCV_noSmooth, delimiter=",")
        outputName = folderIn + '/chb_' + pat + '_AllPerformances_ForAllCV_OurSmoothStep1.csv'
        np.savetxt(outputName, performancesAll_eachSubj_allCV_ourSmoothStep1, delimiter=",")
        outputName = folderIn + '/chb_' + pat + '_AllPerformances_ForAllCV_OurSmoothStep2.csv'
        np.savetxt(outputName, performancesAll_eachSubj_allCV_ourSmoothStep2, delimiter=",")

        # calculatin mean for all files of that person
        performancesAll_allSubj_noSmooth[patIndx, :] = np.nanmean(performancesAll_eachSubj_allCV_noSmooth, 0)
        performancesAll_allSubj_ourSmoothStep1[patIndx, :] = np.nanmean(performancesAll_eachSubj_allCV_ourSmoothStep1,0)
        performancesAll_allSubj_ourSmoothStep2[patIndx, :] = np.nanmean(performancesAll_eachSubj_allCV_ourSmoothStep2,0)

    # saving average for all subjects
    outputName = folderIn + '/AllSubj_AllPerformances_ForAllCV_NoSmooth.csv'
    np.savetxt(outputName, performancesAll_allSubj_noSmooth, delimiter=",")
    outputName = folderIn + '/AllSubj_AllPerformances_ForAllCV_OurSmoothStep1.csv'
    np.savetxt(outputName, performancesAll_allSubj_ourSmoothStep1, delimiter=",")
    outputName = folderIn + '/AllSubj_AllPerformances_ForAllCV_OurSmoothStep2.csv'
    np.savetxt(outputName, performancesAll_allSubj_ourSmoothStep2, delimiter=",")


def func_plotAllPerformances_AllSubj_variableSmoothing(folderIn, folderOut, smoothingTypeString, plottingON):
    ''' plots various performances for each subject for given smoothing type of labels
    performances that are plotted:
    'sensitivity_episodes', 'precision_episodes', 'F1score_episodes', 'sensitivity_duration', 'precision_duration','F1score_duration',\
    'numFPperHour', 'numFalsePosPerHourCorr', 'combinedMeasure', 'totAccuracy', 'balancedAccuracy','specificity'
    '''

    numSubj = len(np.sort(glob.glob(folderIn + '/chb_' + '*_AllPerformances_ForAllCV_' + smoothingTypeString + '.csv')))
    performancesAllSubj_mean = np.zeros((numSubj, 12))  # 12 performance measures
    performancesAllSubj_std = np.zeros((numSubj, 12))
    # reading all data and calculating meand ans std
    patIndx = -1
    for fileIn in np.sort(glob.glob(
            folderIn + '/chb_' + '*_AllPerformances_ForAllCV_' + smoothingTypeString + '.csv')):  # num CV as number of seizure files - leave this one out
        patIndx = patIndx + 1
        reader = csv.reader(open(fileIn, "r"))
        data0 = list(reader)
        data = np.array(data0).astype("float")
        # sensitivity_episodes, precision_episodes, F1score_episodes, sensitivity_duration, precision_duration,F1score_duration
        # numFPperHour, numFalsePosPerHourCorr, combinedMeasure, totAccuracy, balancedAccuracy,specificity
        # 'sensitivity_episodes', 'specificity_duration', 'delay'
        performancesAllSubj_mean[patIndx, :] = np.nanmean(data, 0)
        performancesAllSubj_std[patIndx, :] = np.nanstd(data, 0)

    performancesAllSubj_Totalmean = np.nanmean(performancesAllSubj_mean, 0)
    # plotting
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(5, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    fig1.suptitle('Different performance scores for each Subj')

    # performancesNames=['sensitivity_episodes', 'specificity_duration', 'delay']
    performancesNames = ['sensitivity_episodes', 'precision_episodes', 'F1score_episodes', 'sensitivity_duration',
                         'precision_duration', 'F1score_duration', \
                         'numFPperHour', 'numFalsePosPerHourCorr', 'combinedMeasure', 'totAccuracy', 'balancedAccuracy',
                         'specificity']
    xvalues = np.arange(0, numSubj, 1)
    for perf in range(len(performancesNames)):
        ax1 = fig1.add_subplot(gs[int(perf / 3), np.mod(perf, 3)])
        ax1.errorbar(xvalues, performancesAllSubj_mean[:, perf], yerr=performancesAllSubj_std[:, perf],
                     fmt='m-.')  # ,label='Performance')
        ax1.plot(xvalues, np.ones(len(xvalues)) * performancesAllSubj_Totalmean[perf], 'k')
        # ax1.set_ylabel(performancesNames[perf])
        ax1.set_xlabel('Subjects')
        ax1.set_title(performancesNames[perf])
        # ax1.legend()
        ax1.grid()
    if (plottingON == 1):
        fig1.show()
    fig1.savefig(folderOut + '/AllSubj_AllPerformances_' + smoothingTypeString + '.png')
    plt.close(fig1)


def func_plotAllPerformances_AllSubj_AllSmoothingTypes(folderIn, folderOut, smoothingTypesArray, plottingON):
    ''' plots various performances average of all subjects  for all label smoothing types
    performances that are plotted:
    'sensitivity_episodes', 'precision_episodes', 'F1score_episodes', 'sensitivity_duration', 'precision_duration','F1score_duration',\
    'numFPperHour', 'numFalsePosPerHourCorr', 'combinedMeasure', 'totAccuracy', 'balancedAccuracy','specificity'
    '''

    colorsArray = ['k-.', 'b-.', 'm-.', 'r-.', 'g-.', 'y-.', 'k--', 'b--', 'm--', 'r--', 'g--', 'y--', 'k..', 'b..','m..', 'r..', 'g..', 'y..']
    numSubj = len(np.sort(glob.glob(folderIn + '/chb_' + '*_AllPerformances_ForAllCV_' + smoothingTypesArray[0] + '.csv')))

    allPerformances_AllSubj_AllSmoothingTypes_mean = np.zeros((numSubj, 12, len(smoothingTypesArray)))
    allPerformances_AllSubj_AllSmoothingTypes_std = np.zeros((numSubj, 12, len(smoothingTypesArray)))
    allPerformances_AllSubj_AllSmoothingTypes_totalMeanAllSubj = np.zeros((len(smoothingTypesArray), 15))
    for smIndx, smooth in enumerate(smoothingTypesArray):
        (allPerformances_AllSubj_AllSmoothingTypes_mean[:, :, smIndx],allPerformances_AllSubj_AllSmoothingTypes_std[:, :, smIndx],
         allPerformances_AllSubj_AllSmoothingTypes_totalMeanAllSubj[smIndx,:]) = readAllSubjPerformanceFiles_AllPerformances(folderIn, smooth)

    # plotting
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(5, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    fig1.suptitle('Different performance scores for each Subj')

    performancesNames = ['sensitivity_episodes', 'precision_episodes', 'F1score_episodes', 'sensitivity_duration','precision_duration', 'F1score_duration', \
                         'numFPperHour', 'numFalsePosPerHourCorr', 'combinedMeasure', 'totAccuracy', 'balancedAccuracy','specificity']
    xvalues = np.arange(0, numSubj, 1)
    for perf in range(len(performancesNames)):
        ax1 = fig1.add_subplot(gs[int(perf / 3), np.mod(perf, 3)])
        for smIndx, smooth in enumerate(smoothingTypesArray):
            ax1.errorbar(xvalues, allPerformances_AllSubj_AllSmoothingTypes_mean[:, perf, smIndx],
                         yerr=allPerformances_AllSubj_AllSmoothingTypes_std[:, perf, smIndx], fmt=colorsArray[smIndx],
                         label=smooth)
            ax1.plot(xvalues,
                     np.ones(len(xvalues)) * allPerformances_AllSubj_AllSmoothingTypes_totalMeanAllSubj[smIndx, perf],
                     colorsArray[smIndx])
        # ax1.set_ylabel(performancesNames[perf])
        ax1.set_xlabel('Subjects')
        ax1.set_title(performancesNames[perf])
        if (perf == 0):
            ax1.legend()
        ax1.grid()

    if (plottingON == 1):
        fig1.show()
    fig1.savefig(folderOut + '/AllSubj_AllPerformances_AllSmoothingTypes.png')
    plt.close(fig1)

def readAllSubjPerformanceFiles_AllPerformances(folderIn, smoothingType):
    ''' reads files with performance per subject and cauclates mean and std for all subjects'''

    numSubj = len(np.sort(glob.glob(folderIn + '/chb_' + '*_AllPerformances_ForAllCV_'+smoothingType+'.csv')))
    performancesAllSubj_mean=np.zeros((numSubj,12))
    performancesAllSubj_std = np.zeros((numSubj, 12))
    #reading all data and calculating meand ans std
    patIndx = -1
    for fileIn in np.sort(glob.glob(folderIn + '/chb_' + '*_AllPerformances_ForAllCV_'+smoothingType+'.csv')):  # num CV as number of seizure files - leave this one out
        patIndx = patIndx + 1
        reader = csv.reader(open(fileIn, "r"))
        data0 = list(reader)
        data = np.array(data0).astype("float")
        # sensitivity_episodes, precision_episodes, F1score_episodes, sensitivity_duration, precision_duration,F1score_duration
        # numFPperHour, numFalsePosPerHourCorr, combinedMeasure, totAccuracy, balancedAccuracy,specificity
        performancesAllSubj_mean[patIndx,:] = np.nanmean(data, 0)
        performancesAllSubj_std[patIndx,:]  = np.nanstd(data, 0)

    performancesAllSubj_Totalmean= np.nanmean(performancesAllSubj_mean, 0)
    return (performancesAllSubj_mean, performancesAllSubj_std, performancesAllSubj_Totalmean)

def func_plotAllPerformances_AllSubj_severalParams_AllSmoothTypes_forPaper(approachTypesArray, datasetNames, namesArray, foldersInArray, folderOut, outName,smoothingTypesArray,plottingON):
    ''' function that plots performances for both dataset, different approaches and smoothing levels
    in a way that we wanted for a paper '''

    colorsArray=['k-.','b-.','m-.','r-.', 'g-.','y-.','k--','b--','m--','r--', 'g--','y--','k..','b..','m..','r..', 'g..','y..']
    colorsArray=['black','purple','crimson']
    performancesNames = ['F1score_episodes',  'F1score_duration']
    perfIndexes=[2,  5]
    # performancesNames=['sensitivity_episodes', 'precision_episodes', 'F1score_episodes', 'sensitivity_duration', 'precision_duration','F1score_duration']
    #                   # 'numFPperHour', 'numFalsePosPerHourCorr', 'combinedMeasure', 'totAccuracy', 'balancedAccuracy','specificity',\
    #                   # 'sensitivity_episodes', 'specificity_duration', 'delay']
    datasets=['EEG-CHBMIT', 'IEEG-Bern']
    smoothingTypesArray2=['RawLabels','Postprocessing Step1','Postprocessing Step2']
    # plotting
    fontSizeNum=20
    fig1 = plt.figure(figsize=(16,12), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig1)
    #fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    fig1.suptitle('Different performance scores depending on parameter', fontsize=fontSizeNum)
    fig1.tight_layout()
    xvalues = np.arange(0, len(approachTypesArray ), 1)

    performancesAllSubj_Totalmean=np.zeros((len(approachTypesArray), 12, len(smoothingTypesArray)))
    performancesAllSubj_Totalstd=np.zeros((len(approachTypesArray), 12, len(smoothingTypesArray)))
    for datasetIndx in range(len(datasets)):
        # # load to check just the numbers
        for smoothIndx, smoothingType in enumerate(smoothingTypesArray):
            fileIn = foldersInArray[datasetIndx] + '/AllSubj_AllPerformances_'+smoothingType+'_testingParameter_'+outName+'_mean.csv'
            reader = csv.reader(open(fileIn, "r"))
            performancesAllSubj_Totalmean[:, :, smoothIndx] = np.array(list(reader)).astype("float")
            fileIn = foldersInArray[datasetIndx] + '/AllSubj_AllPerformances_' + smoothingType + '_testingParameter_' + outName + '_std.csv'
            reader = csv.reader(open(fileIn, "r"))
            performancesAllSubj_Totalstd[:, :, smoothIndx] = np.array(list(reader)).astype("float")

            outputName = folderOut + '/AllPerformances_AllApproaches_'+smoothingType+'_'+datasets[datasetIndx]+'.csv'
            dts=performancesAllSubj_Totalmean[:, :, smoothIndx].transpose()*100
            np.savetxt(outputName, dts, fmt="%.2f", delimiter=";")

        # calculate improvements from noSmooth to SMoothStep2
        smoothImprovement_step1=performancesAllSubj_Totalmean[:, :, 1]-performancesAllSubj_Totalmean[:, :, 0]
        smoothImprovement_step2 = performancesAllSubj_Totalmean[:, :, 2] - performancesAllSubj_Totalmean[:, :, 0]
        smoothImprovementBestForAllApproach1=np.max(smoothImprovement_step1,0)
        smoothImprovementBestForAllApproach2 = np.max(smoothImprovement_step2, 0)

        for perf in range(len(performancesNames)):
            ax1 = fig1.add_subplot(gs[datasetIndx, np.mod(perf, 2)])
            ax1.grid()
            for smoothIndx, smoothingType in enumerate(smoothingTypesArray):
                # ax1.errorbar(xvalues, performancesAllSubj_Totalmean[:, perfIndexes[perf], smoothIndx], yerr=performancesAllSubj_Totalstd[:, perfIndexes[perf],smoothIndx],
                #             fmt=colorsArray[smoothIndx], label=smoothingTypesArray[smoothIndx])  # ,label='Performance') #
                #ax1.plot(xvalues, performancesAllSubj_Totalmean[:,perfIndexes[perf], smoothIndx],colorsArray[smoothIndx])

                barWidth=0.35
                ax1.bar(xvalues-barWidth*len(smoothingTypesArray)/2+barWidth*smoothIndx/2, performancesAllSubj_Totalmean[:,perfIndexes[perf], smoothIndx], color=colorsArray[smoothIndx], width=barWidth/2, label=smoothingTypesArray[smoothIndx])
            ax1.set_xticks(xvalues)
            ax1.set_xticklabels(namesArray, fontsize=fontSizeNum*0.8)
            #ax1.set_xlabel('Vector relations')
            ax1.set_ylabel(performancesNames[perf], fontsize=fontSizeNum)
            ax1.set_ylim(0, 1)
            ax1.set_title(datasetNames[datasetIndx], fontsize=fontSizeNum)
            #ax1.set_legend(smoothingTypesArray)

            if (perf == 1):
                ax1.legend(smoothingTypesArray2, loc='lower right',fontsize=fontSizeNum*0.8)

    if (plottingON==1):
        fig1.show()

    fig1.savefig(folderOut + '/AllSubj_AllPerformances_AllSmoothingTypes_'+outName+'.png')
    plt.close(fig1)