''' Script that based on parameters set runs training and testing of specific HD features model
- it is possible to choose between two different databases and which patients to use
- it can perform personalized or generalized training
    - personalized is for each subject performing leave one seizure out and training on the rest of seizures
    - generalized requires to define CViterations_testSubj variable - which determines CV iterations and which subjects are in test set (all others are in training)
- outputs prediction labels for each input file with saved original predictions, and predictions after two sets of labels postprocessing (smoothing) and also true labels
    - by analysing and plotting this labes many things can be analized
- plots performances in dependence of performance measures and amount of postprocessing
'''
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"


from HDfunctionsLib import *
from parametersSetup import *

#SETUPS
GeneralParams.plottingON=0
GeneralParams.PersGenApproach='personalized' #'personalized', 'generalized'
torch.cuda.set_device(HDParams.CUDAdevice)
HDParams.D=10000

####################################
## SETUP DATASET USED
#IEEG Bern
Dataset='01_iEEG_Bern'
patients =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']

# #CHBMIT
# Dataset='01_CHBMIT'
# patients =['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

GeneralParams.patients =patients

# DEFINING INPUT/OUTPUT FOLDERS
DatasetProcessed= Dataset+'/01_datasetProcessed'
folderIn = DatasetProcessed +'/'

folderOut0=Dataset+'/02_Predictions' +'/'
createFolderIfNotExists(folderOut0)
folderOut0=folderOut0 +'/'+ str(GeneralParams.PersGenApproach)+'/'
createFolderIfNotExists(folderOut0)


###################################################################
### SETTING UP PARAMETERS FOR EACH APPROACH INDIVIDUALLY
# #TESTING LBP
# SegSymbParams.symbolType = 'LBP'
# SegSymbParams.segLenSec= 0.5
# SegSymbParams.slidWindStepSec= 0.5
# HDParams.similarityType = 'hamming'  # 'hamming','cosine'
# HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'....
# HDParams.vectorTypeLevel = 'scaleNoRand4'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'....
# HDParams.roundingTypeForHDVectors='inSteps' #'inSteps','onlyOne','noRounding'


# TESTING SINGLE FEATURE - AMPLITUDE
SegSymbParams.symbolType ='Amplitude'
SegSymbParams.amplitudeRangeFactor= 2
SegSymbParams.amplitudeBinsSpacing='adjusted' #'equal','adjusted'
SegSymbParams.numSegLevels= 20
SegSymbParams.segLenSec = 4
SegSymbParams.slidWindStepSec = 0.5
HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'....
HDParams.vectorTypeLevel = 'scaleNoRand4'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'....
HDParams.roundingTypeForHDVectors='inSteps' #'inSteps','onlyOne','noRounding'

# # TESTING SINGLE FEATURE - ENTROPY
# SegSymbParams.symbolType = 'Entropy'
# SegSymbParams.numSegLevels=20
# SegSymbParams.segLenSec=4
# SegSymbParams.slidWindStepSec=0.5
# SegSymbParams.entropyType = 'spectral_entropy'  # 'perm_entropy', 'spectral_entropy', 'svd_entropy', 'app_entropy', 'sample_entropy', 'lziv_complexity'
# HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'....
# HDParams.vectorTypeLevel = 'scaleNoRand4'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'....
# HDParams.roundingTypeForHDVectors = 'inSteps'  # 'inSteps','onlyOne','noRounding'
#
# # TESTING SINGLE FEATURE - CWT
# SegSymbParams.symbolType = 'CWT'
# SegSymbParams.CWTlevel= 20
# SegSymbParams.numSegLevels= 20
# SegSymbParams.segLenSec= 4
# SegSymbParams.slidWindStepSec=0.5
# HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'...
# HDParams.vectorTypeLevel = 'scaleNoRand4'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'...
# HDParams.roundingTypeForHDVectors = 'inSteps'  # 'inSteps','onlyOne','noRounding'

# # TESTING FREQUENCY SPECTRUM FFT
# SegSymbParams.symbolType ='FFT'
# HDParams.FFTUpperBound= 16
# SegSymbParams.numSegLevels=20
# SegSymbParams.segLenSec = 4
# SegSymbParams.slidWindStepSec = 0.5
# HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'...
# HDParams.vectorTypeLevel = 'scaleNoRand4'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'...
# HDParams.vectorTypeFreq='random'# 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
# HDParams.bindingFFT = 'FreqxVal'  # 'FreqxVal', 'ChxFreqxVal', 'PermChFreqxVal'
# HDParams.roundingTypeForHDVectors='inSteps' #'inSteps','onlyOne','noRounding'

# # TESTING RAW AMPLITUDE
# SegSymbParams.symbolType ='RawAmpl'
# SegSymbParams.amplitudeRangeFactor=2
# SegSymbParams.amplitudeBinsSpacing='adjusted' #'equal','adjusted'
# SegSymbParams.numSegLevels=20
# SegSymbParams.segLenSec = 4
# SegSymbParams.slidWindStepSec = 0.5
# HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
# HDParams.vectorTypeLevel = 'scaleNoRand4'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
# HDParams.bindingRawAmpl='ValxCh' #'ValxCh','PermValSamplxCh','PermValSampl'
# HDParams.roundingTypeForHDVectors='inSteps' #'inSteps','onlyOne','noRounding'

# # ALL FEATURES - 3 FEAT
# SegSymbParams.symbolType ='AllFeatures'
# SegSymbParams.amplitudeRangeFactor=2
# SegSymbParams.amplitudeBinsSpacing='adjusted' #'equal','adjusted'
# SegSymbParams.numSegLevels=20
# SegSymbParams.CWTlevel=20
# SegSymbParams.entropyType = 'spectral_entropy'
# SegSymbParams.segLenSec = 4
# SegSymbParams.slidWindStepSec = 0.5
# HDParams.vectorTypeLevel = 'scaleNoRand4'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
# HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
# HDParams.vectorFeatures='random'
# HDParams.bindingFeatures='FeatxVal'
# HDParams.roundingTypeForHDVectors='inSteps'

# # TESTING STANDARD ML FEATURES - 45 FEAT
# SegSymbParams.symbolType ='StandardMLFeatures'
# SegSymbParams.numSegLevels=20
# SegSymbParams.segLenSec = 4  # length of EEG sements in sec
# SegSymbParams.slidWindStepSec = 0.5  # step of slidin window to extract segments in sec
# HDParams.vectorTypeLevel = 'scaleNoRand4'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
# HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
# HDParams.vectorTypeFeat='random'
# HDParams.bindingFeatures='FeatxVal' #'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend1000'
# HDParams.numFeat=45
# HDParams.roundingTypeForHDVectors='inSteps' #'inSteps','onlyOne','noRounding'



#################################################################
## CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
if (SegSymbParams.symbolType == 'CWT'):
    folderOutName = SegSymbParams.symbolType + '_' + str(SegSymbParams.CWTlevel)
elif (SegSymbParams.symbolType == 'Entropy'):
    folderOutName = SegSymbParams.symbolType + '_' + SegSymbParams.entropyType + '_' + str(SegSymbParams.numSegLevels)
elif (SegSymbParams.symbolType == 'Amplitude'):
    folderOutName = SegSymbParams.symbolType + '_' + str(SegSymbParams.numSegLevels) + '_' + str(
        SegSymbParams.amplitudeRangeFactor) + '_' + SegSymbParams.amplitudeBinsSpacing
elif (SegSymbParams.symbolType == 'LBP'):
    folderOutName = SegSymbParams.symbolType
elif (SegSymbParams.symbolType == 'FFT'):
    folderOutName = SegSymbParams.symbolType + '_' + HDParams.bindingFFT + '_' + str(
        SegSymbParams.numSegLevels) + '_UpFreq' + str(HDParams.FFTUpperBound) + '_FREQVect' + HDParams.vectorTypeFreq
elif (SegSymbParams.symbolType == 'RawAmpl'):
    folderOutName = SegSymbParams.symbolType + '_' + HDParams.bindingRawAmpl + '_' + str(
        SegSymbParams.numSegLevels) + '_' + str(
        SegSymbParams.amplitudeRangeFactor) + '_' + SegSymbParams.amplitudeBinsSpacing
elif (SegSymbParams.symbolType == 'AllFeatures'):
    folderOutName = SegSymbParams.symbolType + '_' + str(
        SegSymbParams.CWTlevel) + '_' + SegSymbParams.entropyType + '_' + str(
        SegSymbParams.numSegLevels) + '_' + SegSymbParams.amplitudeBinsSpacing + '_' + HDParams.bindingFeatures + '_FEATvec' + HDParams.vectorTypeFeat
elif (SegSymbParams.symbolType == 'StandardMLFeatures'):
    folderOutName = SegSymbParams.symbolType + '_' + str(SegSymbParams.numSegLevels) + '_numFeat' + str(
        HDParams.numFeat) + '_' + HDParams.bindingFeatures + '_FEATvec' + HDParams.vectorTypeFeat
else:
    folderOutName = SegSymbParams.symbolType + '_' + str(
        SegSymbParams.CWTlevel) + '_' + SegSymbParams.entropyType + '_' + str(
        SegSymbParams.numSegLevels) + '_' + SegSymbParams.amplitudeBinsSpacing
    # folderOut = 'Predictions_PermEntropy_05s/'# 'Predictions/'
folderOutName = folderOutName + '_' + str(SegSymbParams.segLenSec) + '_' + str(
    SegSymbParams.slidWindStepSec) + 's' + '_' + HDParams.similarityType + '_RND' + HDParams.roundingTypeForHDVectors + '_CHVect' + HDParams.vectorTypeCh + '_LVLVect' + HDParams.vectorTypeLevel

folderOut_ML = folderOut0 + folderOutName
createFolderIfNotExists(folderOut_ML)
print('FOLDER OUT:', folderOut_ML)

# read normalization values - needed for standarMLFeatures - to know how to normalize feature values
if (SegSymbParams.symbolType == 'StandardMLFeatures'):
    HDParams.normValuesForFeatures = np.zeros((2, HDParams.numFeat))
    normFile = Dataset + '/StandardMLFeaturesNormalization_TotalMax.csv'
    reader = csv.reader(open(normFile, "r"))
    data = np.array(list(reader)).astype("float")
    HDParams.normValuesForFeatures[1, :] = np.max(data, 0)
    normFile = Dataset + '/StandardMLFeaturesNormalization_TotalMin.csv'
    reader = csv.reader(open(normFile, "r"))
    data = np.array(list(reader)).astype("float")
    HDParams.normValuesForFeatures[0, :] = np.min(data, 0)


#################################################################
## TRAINING AND TESTING HR APPROACHE - EITHER IN PERSONALIZE OR GENERALIZED APPROACH
# saves for each file predicted labels and true labels
# in file 'fileName' + '_True&PredLabels.csv'
if GeneralParams.PersGenApproach == 'personalized':
    func_trainAndTest_personalized(SigInfoParams, SegSymbParams, GeneralParams, HDParams, EEGfreqBands, folderIn, folderOut_ML)
else:
    func_trainAndTest_Generalized(SigInfoParams, SegSymbParams, GeneralParams, HDParams, EEGfreqBands, folderIn, folderOut_ML)

#plotting predictions for each file
func_plotPredictionsEachPerson(folderOut_ML, GeneralParams, SegSymbParams, HDParams)

## CALCULATING PERFORMANCE AND PLOTTING
#calculating performance for different smoothings and plotting
func_plotTestResults_AllSubj(SigInfoParams, SegSymbParams, GeneralParams, HDParams, EEGfreqBands, folderIn, folderOut_ML)


