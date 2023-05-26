''' file with all parameters'''
import numpy as np
import pickle

Dataset='01_CHBMIT' #'01_CHBMIT', '01_iEEG_Bern'
patients =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']


class GeneralParams:
    #LABEL SMOOTHING PARAMETERS
    seizureStableLenToTest=5 #in seconds  - window for performin label voting
    seizureStablePercToTest=0.5 # 50% of 1 in last seizureStableLenToTest values that needs to be 1 to finally keep label 1
    distanceBetween2Seizures=30 #in seconds - if seizures are closer then this then they are merged
    timeBeforeSeizureConsideredAsSeizure=30 #in seconds - if seizure starts bit before true seizure to still consider ok
    numFPperDayThr=1 #for additional meausre of performance what number of FP seizures per days we consider ok

    patients=patients  #on which subjects to train and test
    plottingON=0  #determines whether some additional plots are plotted
    PersGenApproach='personalized' #'personalized', 'generalized' approaches
    #for generalized model defining iterations for CV (which subj are in test)
    CViterations_testSubj=[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]

class SigInfoParams:
    chToKeep=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])  #which channels to keep
    channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
                'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                'FZ-CZ', 'CZ-PZ']
    samplFreq=256 #sampling frequency of data

class SegSymbParams:
    # WINDOW DISCRETIZATION
    segLenSec=4 #length of discrete EEG windows on which to perform analysis
    slidWindStepSec=0.5 #step of sliding window
    labelVotingType='majority' #'majority', 'atLeastOne' or 'allOne' #defines how final label of a segment is chosen
    # HD APPROACH
    symbolType = 'LBP' #'CWT', 'Entropy', 'Amplitude', 'LBP', 'FFT','RawAmpl', 'StandardMLFeatures', 'AllFeatures'
    numSegLevels = 20 #number levels on which to normalize values
    # VARIOUS PARAMETERS OF APPROACHES
    entropyType='spectral_entropy'  #'perm_entropy', 'spectral_entropy', 'svd_entropy', 'app_entropy', 'sample_entropy'
    #wavelet transform parameters
    DWTfilterName='db4' #'sym5'
    DWTlevel=0 # 0 means none, means automatic, 4,7,10, 0
    CWTfilterName='cmor1.0-2.0' #'gaus1'
    numFreqPerBand = 10  # only needed for CWT
    CWTlevel=20 # only needed for CWT, the same as numSegLevels for others
    noiseNormType = 'noiseNorm'  #'noiseNorm', 'noNoiseNorm' defines if CWT values are normalized with CWT of noise
    #for amplitude normalization min and mean values of whole signal are needed
    amplitudeRangeFactor=2 # how much bigger and smaller values from mean we expect
    amplitudeBinsSpacing='equal' #'equal','adjusted'  #whether we make equal spacing bins between min and max values of we adjust them manually
    minValueSignal=np.zeros((len(SigInfoParams.chToKeep))) #variable to keep track in real time about min and max value
    meanValueSignal=np.zeros((len(SigInfoParams.chToKeep))) #variable to keep track in real time about min and max value
    cntForAmplitude=0 #variable to keep track in real time about min and max value

# FREQUENCY BANDS - DEFINED FOR E.G. CWT
class EEGfreqBands:
    freqVal4Lev = [2.5,6,11.5,22.5] #middle of alfa, beta, gamma, delta bands
    freqVal10Lev = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    freqVal20Lev = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 5, 6, 8, 10, 15]
    freqVal30Lev = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.4, 3.8, 4.2, 4.6,
                    5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

class HDParams:
    # GENERAL PARAMETERS NEEDED FOR HD
    CUDAdevice = 0 #number of the GPU used
    D = 10000  #dimension of hypervectors
    similarityType= 'hamming' #'hamming','cosine' #similarity measure used for comparing HD vectors
    vectorTypeCh= 'random' # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2' ... #defines how HD vectors are initialized
    vectorTypeLevel = 'random'# 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'... #defines how HD vectors are initialized
    roundingTypeForHDVectors='inSteps' #'inSteps','onlyOne','noRounding' #defines how and when HD vectors are binarized
    # LBP approach
    LBPlen = 7  # 1 + dimension l of binary pattern
    totalNumberBP = 2 ** (LBPlen - 1)
    # more features
    numFeat=3 #3 for similar to symbolization, 45  for standard Ml features
    vectorTypeFeat='random'
    bindingFeatures='ChxFeatxVal' #'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal' #defines how HD vectors encoded
    normValuesForFeatures=np.zeros((2,numFeat))  #variable to keep track in real time
    # FFT encoding
    FFTUpperBound=16 # upped border for freq of interest, in Hz (up to 128, but should be mod2 of 128)
    vectorTypeFreq='random'# 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2' #defines how HD vectors are initialized
    bindingFFT = 'FreqxVal'  # 'FreqxVal', 'ChxFreqxVal', 'PermChFreqxVal', 'FeatAppend' #defines how HD vectors encoded
    #Raw amplitude encoding
    bindingRawAmpl='ValxCh' #'ValxCh','PermValSamplxCh','PermValSampl' #defines how HD vectors encoded


#SAVING SETUP once again to update if new info
with open('../PARAMETERS.pickle', 'wb') as f:
    pickle.dump([GeneralParams, SegSymbParams, SigInfoParams, EEGfreqBands, patients], f)