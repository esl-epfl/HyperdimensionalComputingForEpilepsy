__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

''' script to measure KL and JS divergence of features '''

from HDfunctionsLib import *
from parametersSetup import *


###################################
# SETUP DATASET USED
# CHBMIT
Dataset = '01_CHBMIT'  # '01_CHBMIT', '01_iEEG_Bern'
GeneralParams.patients = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
# create folder for dataset if it doesnt exist
createFolderIfNotExists('../' + Dataset)

# PARAMETERS
GeneralParams.PersGenApproach = 'personalized'  # 'personalized', 'generalized'
datasetFiltering = '1to30Hz'  # 'Raw', 'MoreNonSeizure_Fact10' #depends if we want all data or some subselection
GeneralParams.CVtype = 'LeaveOneOut'  # 'LeaveOneOut', 'RollingBase'
datasetPreparationType = 'Fact10'  # 'Fact1', 'Fact10' ,'AllDataWin3600s', 'AllDataWin3600s_1File6h', 'AllDataStoS'  # !!!!!!

# features used
FeaturesParams.featNames = np.array(['MeanAmpl', 'LineLength', 'Frequency'])
FeaturesUsed = '-'.join(FeaturesParams.featNames)
AllExistingFeatures = FeaturesParams.allFeatTypesName
FeaturesParams.featNames = constructAllfeatNames(FeaturesParams)  # feat used here
totalNumFeat = len(FeaturesParams.allFeatNames)  # all features that exist in original data
FeaturesParams.featNorm = 'Norm&Discr'  # 'Norm', 'Norm&Discr'  #!!! HAS TO BE DISCRETIZED
FeaturesParams.winLen = 4  # in seconds, window length on which to calculate features
FeaturesParams.winStep = 0.5  # in seconds, step of moving window length

# HD COMPUTING PARAMS
torch.cuda.set_device(HDParams.CUDAdevice)
HDParams.vectorTypeLevel = 'scaleNoRand1'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
HDParams.vectorTypeCh = 'random'  # 'random','sandwich','scaleNoRand1','scaleNoRand2','scaleRand1', ,'scaleRand2'
HDParams.vectorTypeFeat = 'random'
HDParams.roundingTypeForHDVectors = 'inSteps'  # 'inSteps','onlyOne','noRounding'
# IMPORTANT PARAMETERS!!!!
HDParams.D = 100
HDParams.bindingFeatures = 'FeatAppend'  # 'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend1000'
HDParams.numFeat = len(FeaturesParams.featNames)

numBins=100 #number of bins for histograms for calculating KL divergence

# ##################################################################
# DEFINING INPUT/OUTPUT FOLDERS
folderInEDF = '../../../../databases/medical/chb-mit/edf/'  # location on server so that we dont have to download to each persons folder
folderInCSV = '../' + Dataset + '/01_datasetProcessed_Raw/'  # where to save filtered raw data
# folderInCSV = '../../EEGepilepsy/02_HDcode/01_CHBMIT/01_datasetProcessed_Raw/'
createFolderIfNotExists(folderInCSV)
folderInfo = '../' + Dataset + '/01_SeizureInfoOriginalData/'  # folder to save results of basic analysis about seizures
createFolderIfNotExists(folderInfo)
folderOutFeatures = '../' + Dataset + '/02_Features_' + datasetFiltering + '_' + str(FeaturesParams.winLen) + '_' + str( FeaturesParams.winStep)  # where to save calculated features for each original file
createFolderIfNotExists(folderOutFeatures)
# raranged features folder
folderOutRearangedData = '../' + Dataset + '/04_RearangedData_MergedFeatures_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str( FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType
# predictions folder
folderOutPredictions0 = '../' + Dataset + '/05_Predictions_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str( FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType
createFolderIfNotExists(folderOutPredictions0)
if (HDParams.HDvecType == 'bin'):
    folderOutPredictions0 = folderOutPredictions0 + '/' + str(  GeneralParams.PersGenApproach) + '_' + GeneralParams.CVtype
elif (HDParams.HDvecType == 'bipol'):
    folderOutPredictions0 = folderOutPredictions0 + '/' + str( GeneralParams.PersGenApproach) + '_' + GeneralParams.CVtype + '_bipolarVec/'
createFolderIfNotExists(folderOutPredictions0)
folderOutParams = FeaturesUsed + '_' + str( HDParams.numSegmentationLevels) + '_' + HDParams.bindingFeatures + '_D' + str( HDParams.D)  # not all parameters are saved here (but coudl be if needed)
folderOutPredictions = folderOutPredictions0 + '/' + folderOutParams + '_RF_StdHD_OnlHD/'
createFolderIfNotExists(folderOutPredictions)
folderOutPredictionsPlot = folderOutPredictions + '/PredictionsComparison/'
createFolderIfNotExists(folderOutPredictionsPlot)
if (HDParams.bindingFeatures == 'FeatAppend'):
    folderOutPredictionsPerFeat = folderOutPredictions + '/PerFeat/'
    createFolderIfNotExists(folderOutPredictionsPerFeat)

#FOLDER FOR DIVERGENCE
folderOutFeatDiverg = '../' + Dataset + '/05_Predictions_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str( FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType
createFolderIfNotExists(folderOutFeatDiverg)
folderOutFeatDiverg= folderOutFeatDiverg+'/FeatureDivergence/'
createFolderIfNotExists(folderOutFeatDiverg)


##################################################################
# CALCULATING FEATURE DIVERGENCE
# loads taw data and create histograms of values based on which calculated Kullback-Leibler and Jensen-Shannon divergence
numFeat=HDParams.numFeat* len(DatasetPreprocessParams.channelNamesToKeep)
JSdiverg=np.zeros((len(GeneralParams.patients), numFeat))
KLdiverg_SNS=np.zeros((len(GeneralParams.patients), numFeat))
KLdiverg_NSS = np.zeros((len(GeneralParams.patients), numFeat))
for patIndx, pat in enumerate(GeneralParams.patients):
    filesIn= np.sort(glob.glob(folderOutRearangedData + '/*Subj' + pat + '*.csv.gz'))
    numFiles=len(filesIn)
    print('-- Patient:', pat, 'NumSeizures:', numFiles)

    # load all files of this subject
    filesToTrainOn = []
    for fIndx, fileName in enumerate(filesIn):
        filesToTrainOn.append(fileName)
    # concatenating data from more files
    (FeatAll, LabelsAll,startIndxOfFiles)=concatenateDataFromFiles(filesToTrainOn)

    #calculate histograms per seizure and non seizure
    for f in range(numFeat):
        (SeizHist, nonSeizHist) = calcHistogramValues_v2(FeatAll[:,f], LabelsAll,numBins)
        KLdiverg_SNS[patIndx, f] = kl_divergence(SeizHist[0], nonSeizHist[0])
        KLdiverg_NSS[patIndx, f] = kl_divergence(nonSeizHist[0],SeizHist[0])
        JSdiverg[patIndx,f] = js_divergence(SeizHist[0], nonSeizHist[0])

        # #plot histograms for this subject
        # fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
        # gs = GridSpec(1, 1, figure=fig1)
        # fig1.subplots_adjust(wspace=0.4, hspace=0.6)
        # xValues = np.arange(0, numBins, 1)
        # ax1 = fig1.add_subplot(gs[0,0])
        # ax1.plot(xValues, SeizHist[0], 'r', label='Seiz')
        # ax1.plot(xValues, nonSeizHist[0], 'k', label='NonSeiz')
        # ax1.legend()
        # ax1.set_xlabel('Value')
        # ax1.set_ylabel('Probability')
        # ax1.set_title('Subj ' + pat +' Feat'+ str(f))
        # ax1.grid()
        # fig1.show()
        # fig1.savefig(folderOutSubplot + '/Subj'+ pat+'_Feat'+ str(f)+'_Histogram.png', bbox_inches='tight')
        # plt.close(fig1)

    # save predictions
    outputName = folderOutFeatDiverg + '/' + 'AllSubj_KLdivergence_SNS.csv'
    np.savetxt(outputName, KLdiverg_SNS, delimiter=",")
    outputName = folderOutFeatDiverg + '/' + 'AllSubj_KLdivergence_NSS.csv'
    np.savetxt(outputName, KLdiverg_NSS, delimiter=",")
    outputName = folderOutFeatDiverg + '/' + 'AllSubj_JSdivergence.csv'
    np.savetxt(outputName, JSdiverg, delimiter=",")

# # ##################################################################
# PLOTTING DIVERGENCE PER SUBJECTS
reader = csv.reader(open(folderOutFeatDiverg + '/' + 'AllSubj_KLdivergence_SNS.csv', "r"))
KLdiverg_SNS = np.array(list(reader)).astype("float")
reader = csv.reader(open(folderOutFeatDiverg + '/' + 'AllSubj_KLdivergence_NSS.csv', "r"))
KLdiverg_NSS = np.array(list(reader)).astype("float")
reader = csv.reader(open(folderOutFeatDiverg + '/' + 'AllSubj_JSdivergence.csv', "r"))
JSdiverg = np.array(list(reader)).astype("float")

#analysing std per ch
KLdiverg_SNS_reshaped=np.reshape(KLdiverg_SNS, (len(GeneralParams.patients),-1,HDParams.numFeat ))
KLdiverg_SNS_meanForCh=np.nanmean(KLdiverg_SNS_reshaped,1)
KLdiverg_SNS_stdForCh=np.nanstd(KLdiverg_SNS_reshaped,1)
KLdiverg_NSS_reshaped=np.reshape(KLdiverg_NSS, (len(GeneralParams.patients),-1,HDParams.numFeat ))
KLdiverg_NSS_meanForCh=np.nanmean(KLdiverg_NSS_reshaped,1)
KLdiverg_NSS_stdForCh=np.nanstd(KLdiverg_NSS_reshaped,1)
JSdiverg_reshaped=np.reshape(JSdiverg, (len(GeneralParams.patients),-1,HDParams.numFeat ))
JSdiverg_meanForCh=np.nanmean(JSdiverg_reshaped,1)
JSdiverg_stdForCh=np.nanstd(JSdiverg_reshaped,1)

#plotting
fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
gs = GridSpec(6, 4, figure=fig1)
fig1.subplots_adjust(wspace=0.4, hspace=0.6)
#fig1.suptitle('Feature ')
xValues = np.arange(0, HDParams.numFeat, 1)
for p, pat in enumerate(GeneralParams.patients):
    ax1 = fig1.add_subplot(gs[int(np.floor(p / 4)), np.mod(p, 4)])
    ax1.errorbar(xValues, KLdiverg_SNS_meanForCh[p, :], yerr=KLdiverg_SNS_stdForCh[p, :], fmt='k', label='KL_SNS')
    ax1.errorbar(xValues, KLdiverg_NSS_meanForCh[p, :], yerr=KLdiverg_NSS_stdForCh[p, :], fmt='b', label='KL_NSS')
    ax1.errorbar(xValues, JSdiverg_meanForCh[p, :], yerr=JSdiverg_stdForCh[p, :], fmt='k', label='JS')
    ax1.legend()
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Divergence')
    ax1.set_title('Subj ' + pat)
    ax1.grid()
fig1.show()
fig1.savefig(folderOutFeatDiverg + '/AllSubj_DifferentDivergenceMeasures_perSubj.png', bbox_inches='tight')
plt.close(fig1)

# # ##################################################################
# CALCULATING AVERAGE FOR ALL SUBJECTS
KLdiverg_SNS_meanAllSubj=np.nanmean(KLdiverg_SNS_meanForCh,0)
KLdiverg_NSS_meanAllSubj=np.nanmean(KLdiverg_NSS_meanForCh,0)
JSdiverg_meanAllSubj=np.nanmean(JSdiverg_meanForCh,0)
KLdiverg_SNS_stdAllSubj=np.nanstd(KLdiverg_SNS_meanForCh,0)
KLdiverg_NSS_stdAllSubj=np.nanstd(KLdiverg_NSS_meanForCh,0)
JSdiverg_stdAllSubj=np.nanstd(JSdiverg_meanForCh,0)
# save predictions
outputName = folderOutFeatDiverg + '/' + 'AllSubjAvrg_KLdivergence_SNS.csv'
np.savetxt(outputName, np.vstack((KLdiverg_SNS_meanAllSubj, KLdiverg_SNS_stdAllSubj)), delimiter=",")
outputName = folderOutFeatDiverg + '/' + 'AllSubjAvrg_KLdivergence_NSS.csv'
np.savetxt(outputName, np.vstack((KLdiverg_NSS_meanAllSubj, KLdiverg_NSS_stdAllSubj)), delimiter=",")
outputName = folderOutFeatDiverg + '/' + 'AllSubjAvrg_JSdivergence.csv'
np.savetxt(outputName, np.vstack((JSdiverg_meanAllSubj, JSdiverg_stdAllSubj)), delimiter=",")

#plotting
fig1 = plt.figure(figsize=(10, 6), constrained_layout=False)
gs = GridSpec(1, 1, figure=fig1)
fig1.subplots_adjust(wspace=0.4, hspace=0.6)
#fig1.suptitle('Feature ')
xValues = np.arange(0, HDParams.numFeat, 1)
ax1 = fig1.add_subplot(gs[0,0])
ax1.errorbar(xValues, KLdiverg_SNS_meanAllSubj, yerr=KLdiverg_SNS_stdAllSubj, fmt='b', label='KL_SNS')
ax1.errorbar(xValues, KLdiverg_NSS_meanAllSubj, yerr=KLdiverg_NSS_stdAllSubj, fmt='m', label='KL_NSS')
ax1.errorbar(xValues, JSdiverg_meanAllSubj, yerr=JSdiverg_stdAllSubj, fmt='k', label='JS')
ax1.legend()
ax1.set_xlabel('Feature')
ax1.set_xticks(xValues)
ax1.set_xticklabels(FeaturesParams.allFeatNames, fontsize=12 * 0.8, rotation=90)
ax1.set_ylabel('Divergence')
ax1.set_title('KL and JS divergences')
ax1.grid()
fig1.show()
fig1.savefig(folderOutFeatDiverg + '/AllSubj_DifferentDivergenceMeasures_avrgAllSubj.png', bbox_inches='tight')
plt.close(fig1)

# PLOT JS FOR PAPER
fig1 = plt.figure(figsize=(10, 2), constrained_layout=False)
gs = GridSpec(1, 1, figure=fig1)
fig1.subplots_adjust(wspace=0.3, hspace=0.3)
xValues = np.arange(1, HDParams.numFeat + 1, 1)
ax1 = fig1.add_subplot(gs[0, 0])
ax1.boxplot(JSdiverg_meanForCh, medianprops=dict(color='orangered', linewidth=1.5),
            boxprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5), showfliers=False)
ax1.set_title('Jensen-Shannon divergence')
ax1.grid()
ax1.set_xticks(xValues)
ax1.set_xticklabels(FeaturesParams.featNames, fontsize=12 * 0.8, rotation=45)
# ax1.set_xlabel('Feature')
fig1.show()
fig1.savefig(folderOutFeatDiverg + '/AllSubj_DifferentDivergenceMeasures_boxplots.png', bbox_inches='tight')
fig1.savefig(folderOutFeatDiverg + '/AllSubj_DifferentDivergenceMeasures_boxplots.svg', bbox_inches='tight')
plt.close(fig1)
