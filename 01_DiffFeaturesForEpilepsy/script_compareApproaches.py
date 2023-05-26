''' Script that:
- meausres number of seizure per subjects for databases
- compares different approaches in terms of performance and plots comparison
- meausures time for calculation for different approaches
- plots time and memory requirements for different approaches
'''
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

from HDfunctionsLib import *
from parameters_HD_MIT_setup import *

#SETUPS
GeneralParams.plottingON=0
GeneralParams.PersGenApproach='personalized' #'personalized', 'generalized'
torch.cuda.set_device(HDParams.CUDAdevice)

##############################################
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
folderIn = DatasetProcessed+'/'

folderOut0=Dataset+'/02_Predictions/'
createFolderIfNotExists(folderOut0)
folderOut0=folderOut0 +'/'+ str(GeneralParams.PersGenApproach)+'/'
createFolderIfNotExists(folderOut0)

folderOutPaper='04_resultsForPapers'
createFolderIfNotExists(folderOutPaper)

##############################################
####Calculating average number of seizures per person for datasets
numSeizPerSubj=np.zeros((len(GeneralParams.patients),1))
for patIndx, pat in enumerate(GeneralParams.patients):
    numSeizPerSubj[patIndx,0]= len(np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv')))
print('min:', np.min(numSeizPerSubj))
print('max:', np.max(numSeizPerSubj))
print('mean:', np.mean(numSeizPerSubj))
print('std:', np.std(numSeizPerSubj))
print('sum:', np.sum(numSeizPerSubj))

#############################################
## COMPARING ON THE LEVEL OF PERFORMANCE
## DEFININING WHICH APPROACHES TO COMPARE (AND THEIR DATA FOLDERS)
datasetsArray=['01_CHBMIT','01_iEEG_Bern']
datasetNamesArray=['EEG_CHBMIT','IEEG_Bern']
algNames=['Ampl','Entr','CWT','3Feat','45Feat','FFT','RawAmp','LBP']
smoothingTypesArray = ['NoSmooth', 'OurSmoothStep1', 'OurSmoothStep2']
ThingToTest='ComparisonAllApproaches'+'_'+str(GeneralParams.PersGenApproach)
foldersInArray0=['Amplitude_20_2_adjusted_4_0.5s_hamming_RNDinSteps_CHVectrandom_LVLVectscaleNoRand4', 'Entropy_spectral_entropy_20_4_0.5s_hamming_RNDinSteps_CHVectrandom_LVLVectscaleNoRand4',
                'CWT_20_4_0.5s_hamming_RNDinSteps_CHVectrandom_LVLVectscaleNoRand4','AllFeatures_20_spectral_entropy_20_adjusted_FeatxVal_FEATvecrandom_4_0.5s_hamming_RNDinSteps_CHVectrandom_LVLVectscaleNoRand4',
                'StandardMLFeatures_20_numFeat45_FeatxVal_FEATvecrandom_4_0.5s_hamming_RNDinSteps_CHVectrandom_LVLVectscaleNoRand4',
                'FFT_FreqxVal_20_UpFreq16_FREQVectrandom_4_0.5s_hamming_RNDinSteps_CHVectrandom_LVLVectscaleNoRand4',
                'RawAmpl_ValxCh_20_2_adjusted_4_0.5s_hamming_RNDinSteps_CHVectrandom_LVLVectscaleNoRand4',
                'LBP_0.5_0.5s_hamming_RNDinSteps_CHVectrandom_LVLVectscaleNoRand4']

datasetsFoldersArray=[]
for d in range(len(datasetsArray)):
    f=datasetsArray[d]+ '/02_TestingDifferentParameters/'+'ComparisonAllApproaches_'+str(GeneralParams.PersGenApproach)+ '/'
    datasetsFoldersArray.append(f)
func_plotAllPerformances_AllSubj_severalParams_AllSmoothTypes_forPaper(foldersInArray0, datasetNamesArray, algNames, datasetsFoldersArray, folderOutPaper, ThingToTest,smoothingTypesArray,GeneralParams.plottingON)


#############################################
# #MEASURING TIME FOR CALCULATING HD VECTORS PER DISCRETE WINDOW
# patients =['01']  #possible to change which patient, but they all should be the same
# GeneralParams.patients=patients
# symbolTypeArray=['Amplitude','Entropy', 'CWT','AllFeatures', 'StandardMLFeatures','FFT','RawAmpl', 'LBP']
# folderOut0=Dataset+'/02_TestingDifferentParameters/'
# createFolderIfNotExists(folderOut0)
# folderOutPlots=folderOut0 +'/TimeForCalculation/'
# createFolderIfNotExists(folderOutPlots)
#
# timePerApproach_All_mean=np.zeros((len(symbolTypeArray),4))  #total time, Feat time, HDVec time, ratio Feat/HDvec time
# timePerApproach_All_std=np.zeros((len(symbolTypeArray), 4))
# for s in range(len(symbolTypeArray)):
#     SegSymbParams.symbolType =symbolTypeArray[s]
#     (timeTotalPerSubj_mean, timeTotalPerSubj_std,timeFeatPerSubj_mean,timeFeatPerSubj_std,timeHDVecPerSubj_mean, timeHDVecPerSubj_std, ratioFeatHDTime_mean, ratioFeatHDTime_std)\
#         = calculate_HDcalcTime_onOneWindow(folderIn, SegSymbParams, SigInfoParams,HDParams, GeneralParams, EEGfreqBands)
#     # total time, Feat time, HDVec time, ratio Feat/HDvec time
#     timePerApproach_All_mean[s,:]=[np.mean(timeTotalPerSubj_mean), np.mean(timeFeatPerSubj_mean), np.mean(timeHDVecPerSubj_mean), np.mean(ratioFeatHDTime_mean)]
#     timePerApproach_All_std[s,:] = [np.std(timeTotalPerSubj_mean), np.std(timeFeatPerSubj_mean), np.std(timeHDVecPerSubj_mean), np.std(ratioFeatHDTime_mean)]
#
#
#     outputName = folderOutPlots + '/TimeForComputation_AllApproach_mean.csv'
#     np.savetxt(outputName, timePerApproach_All_mean, delimiter=",")
#     outputName = folderOutPlots + '/TimeForComputation_AllApproach_std.csv'
#     np.savetxt(outputName, timePerApproach_All_std, delimiter=",")

# ##############################################
# plotting memory and time (num operations not plotted)
# dataset='CHBMIT'
# fileIn = '01_CHBMIT/02_TestingDifferentParameters/TimeForCalculation/TimeForComputation_AllApproach_mean.csv'
dataset='IEEGBern'
fileIn = '01_iEEG_Bern/02_TestingDifferentParameters/TimeForCalculation/TimeForComputation_AllApproach_mean.csv'
reader = csv.reader(open(fileIn, "r"))
timePerfPerApproach = np.array(list(reader)).astype("float")

fig2 = plt.figure(figsize=(16, 4), constrained_layout=False)
# plt.rcParams.update({'font.size': fontSizeNum})
fontSizeNum=20
plt.rc('xtick', labelsize=12)
gs = GridSpec(1, 2, figure=fig2)
#fig2.subplots_adjust(wspace=0.4, hspace=0.6)
fig2.tight_layout()
fig2.suptitle('Memory and computational requirements',fontsize=fontSizeNum*0.7)
algNames=['Ampl','Entr','CWT','3Feat','45Feat','FFT','RawAmp','LBP']
algNames2={'Ampl','Entr','CWT','3Feat','45Feat','FFT','RawAmp','LBP'}
memoryArray=[36,36,36,23,65,84,36,80]
# numOperArray=[32,32,32,112,1456,2064,33792,33792]
# timeArray=[1,6,288,360,360,12,550,370]
ax1 = fig2.add_subplot(gs[0, 0])
ax1.bar(algNames,memoryArray, color='indigo')
ax1.set_ylabel('Memory for HD vectors [bit/HD_D]', fontsize=fontSizeNum*0.6)
# ax1.set_xticks(np.arange(0, len(algNames ), 1))
# ax1.set_xlabel(algNames2, fontsize=fontSizeNum*0.6)
ax1.grid()
ax3 = fig2.add_subplot(gs[0, 1])
ax3.bar(algNames,timePerfPerApproach[:,0], color='indigo')
ax3.bar(algNames,timePerfPerApproach[:,1], color='plum')
ax3.set_ylabel('Time per segmentation window [ms]', fontsize=fontSizeNum*0.6)
# ax3.set_xticks(np.arange(0, len(algNames ), 1))
# ax3.set_xlabel(algNames2, fontsize=fontSizeNum*0.6)
ax3.grid()

if (GeneralParams.plottingON==1):
    fig2.show()
fig2.savefig(folderOutPaper + '/AllPerformancesComparison_'+dataset+'.png')
plt.close(fig2)

