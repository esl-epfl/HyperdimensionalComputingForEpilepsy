__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

''' 
- uses raw data created by script_prepareDataset, creates histogram of valus during different labels and calculates divergence per feature-channel combinations
- it measures divergence as Kulback-Leibler and Jensen-Shannon divergence
- plots divergences per feature (avrg of all channels for that feature) or per channel (avrg of all features) for each subject (average for all channels) and also average of all subjects 
'''

from HDfunctionsLib import *
from parametersSetup import *


###################################
# SETUP DATASET USED
# CHBMIT
Dataset = '01_CHBMIT'  # '01_CHBMIT', '01_iEEG_Bern'
GeneralParams.patients = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
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
# FeaturesParams.featNames=  ['mean_ampl', 'line_length','p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot'] #to use less features
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
HDParams.D = 1000
HDParams.bindingFeatures = 'ChAppend' # 'ChAppend', 'FeatAppend'
HDParams.numFeat = len(FeaturesParams.featNames)

numBins=100 #number of bins for histograms for calculating KL divergence
#sets various names to be able to use the same script for channel and feature selection
if (HDParams.bindingFeatures == 'FeatAppend'):
    FeatChType='PerFeat'
    numElems=len(FeaturesParams.featNames)
    elemNames=FeaturesParams.featNames
else:  #'ChAppend'
    FeatChType='PerCh'
    numElems=len(DatasetPreprocessParams.channelNamesToKeep)
    elemNames = DatasetPreprocessParams.channelNamesToKeep

# ##################################################################
# DEFINING INPUT/OUTPUT FOLDERS
folderInEDF = '../../../../databases/medical/chb-mit/edf/'  # location on server so that we dont have to download to each persons folder
folderInCSV = '../' + Dataset + '/01_datasetProcessed_Raw/'  # where to save filtered raw data
createFolderIfNotExists(folderInCSV)
folderInfo = '../' + Dataset + '/01_SeizureInfoOriginalData/'  # folder to save results of basic analysis about seizures
createFolderIfNotExists(folderInfo)
folderOutFeatures = '../' + Dataset + '/02_Features_' + datasetFiltering + '_' + str(FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep)  # where to save calculated features for each original file
createFolderIfNotExists(folderOutFeatures)
folderOutRearangedData = '../' + Dataset + '/04_RearangedData_MergedFeatures_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType


# CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
folderOutPredictions0 = '../' + Dataset + '/05_Predictions_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType
createFolderIfNotExists(folderOutPredictions0)
if (HDParams.HDvecType == 'bin'):
    folderOutPredictions0 = folderOutPredictions0 + '/' + str( GeneralParams.PersGenApproach) + '_' + GeneralParams.CVtype
elif (HDParams.HDvecType == 'bipol'):
    folderOutPredictions0 = folderOutPredictions0 + '/' + str(GeneralParams.PersGenApproach) + '_' + GeneralParams.CVtype + '_bipolarVec/'
createFolderIfNotExists(folderOutPredictions0)

# FOLDER FOR DIVERGENCE
folderOutFeatDiffMeas= '../' + Dataset + '/05_Predictions_' + datasetFiltering + '_' + str( FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep) + '_' + AllExistingFeatures + '_' + datasetPreparationType
createFolderIfNotExists(folderOutFeatDiffMeas)
if (HDParams.bindingFeatures == 'FeatAppend'):
    folderOutFeatDiverg = folderOutFeatDiffMeas + '/FeatureDivergence/'
elif (HDParams.bindingFeatures == 'ChAppend'):
    folderOutFeatDiverg = folderOutFeatDiffMeas + '/ChannelDivergence/'
createFolderIfNotExists(folderOutFeatDiverg)


# ## CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
folderOutParams = FeaturesUsed + '_' + str(HDParams.numSegmentationLevels) + '_' + HDParams.bindingFeatures + '_D' + str( HDParams.D)  # not all parameters are saved here (but coudl be if needed)
folderOutPredictions = folderOutPredictions0 + '/' + folderOutParams + '_RF_StdHD_OnlHD/'
createFolderIfNotExists(folderOutPredictions)
folderOutPredictionsPlot = folderOutPredictions + '/PredictionsComparison/'
createFolderIfNotExists(folderOutPredictionsPlot)
if (HDParams.bindingFeatures == 'FeatAppend' or HDParams.bindingFeatures == 'ChAppend'):
    folderOutPredictionsPerFeat = folderOutPredictions + '/'+FeatChType+'/'
    createFolderIfNotExists(folderOutPredictionsPerFeat)
    # # OUTPUT FOR THIS VOTING TYPE
    # folderOutPredictionsVoting = folderOutPredictions + '/Voting_'+FeatChType+'_' + VotingParam.approach+ '_'+VotingParam.selectionStyle+'_'+HDtype + '/'
    # createFolderIfNotExists(folderOutPredictionsVoting)

# ##################################################################

HDParams.numCh=len(DatasetPreprocessParams.channelNamesToKeep)
numFeat=HDParams.numFeat* len(DatasetPreprocessParams.channelNamesToKeep)

# feature indexes to keep
HDParams.numFeat = len(FeaturesParams.featNames)
featIndxs = np.zeros((len(FeaturesParams.featNames)))
for f in range(len(FeaturesParams.featNames)):
    featIndxs[f] = int(np.where(np.asarray(FeaturesParams.allFeatNames) == FeaturesParams.featNames[f])[0][0])
featIndxs = featIndxs.astype(int)
featIndxsAllCh = []
for ch in range(HDParams.numCh):
    featIndxsAllCh = np.hstack((featIndxsAllCh, featIndxs + ch * totalNumFeat))
featIndxsAllCh = featIndxsAllCh.astype(int)

# ##################################################################
# # CALCULATING FEATURE/CHANNEL DIVERGENCE
# # loads raw data and creates histograms of values during seizrue and non-seizure
# # based on histogram Kullback-Leibler and Jensen-Shannon divergence is calculated
# # it is calculated for each channel-feature combination, so later we can calculate average over all features on one channel, or vice versa
# JSdiverg=np.zeros((len(GeneralParams.patients), numFeat))
# KLdiverg_SNS=np.zeros((len(GeneralParams.patients), numFeat))
# KLdiverg_NSS = np.zeros((len(GeneralParams.patients), numFeat))
# for patIndx, pat in enumerate(GeneralParams.patients):
#     filesIn= np.sort(glob.glob(folderOutRearangedData + '/*Subj' + pat + '*.csv.gz'))
#     numFiles=len(filesIn)
#     print('-- Patient:', pat, 'NumSeizures:', numFiles)
#
#     # load all files of this subject
#     filesToTrainOn = []
#     for fIndx, fileName in enumerate(filesIn):
#         filesToTrainOn.append(fileName)
#     # concatenating data from more files
#     (FeatAll, LabelsAll,startIndxOfFiles)=concatenateDataFromFiles(filesToTrainOn)
#     FeatAll = FeatAll[:, featIndxsAllCh]
#
#     #calculate histograms per seizure and non seizure
#     for f in range(numFeat):
#         (SeizHist, nonSeizHist) = calcHistogramValues_v2(FeatAll[:,f], LabelsAll,numBins)
#         KLdiverg_SNS[patIndx, f] = kl_divergence(SeizHist[0], nonSeizHist[0])
#         KLdiverg_NSS[patIndx, f] = kl_divergence(nonSeizHist[0],SeizHist[0])
#         JSdiverg[patIndx,f] = js_divergence(SeizHist[0], nonSeizHist[0])
#
#         # #plot histograms for this subject
#         # fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
#         # gs = GridSpec(1, 1, figure=fig1)
#         # fig1.subplots_adjust(wspace=0.4, hspace=0.6)
#         # xValues = np.arange(0, numBins, 1)
#         # ax1 = fig1.add_subplot(gs[0,0])
#         # ax1.plot(xValues, SeizHist[0], 'r', label='Seiz')
#         # ax1.plot(xValues, nonSeizHist[0], 'k', label='NonSeiz')
#         # ax1.legend()
#         # ax1.set_xlabel('Value')
#         # ax1.set_ylabel('Probability')
#         # ax1.set_title('Subj ' + pat +' Feat'+ str(f))
#         # ax1.grid()
#         # fig1.show()
#         # fig1.savefig(folderOutSubplot + '/Subj'+ pat+'_Feat'+ str(f)+'_Histogram.png', bbox_inches='tight')
#         # plt.close(fig1)
#
#     # save predictions in one file for all subjects
#     outputName = folderOutFeatDiverg + '/' + 'AllSubj_KLdivergence_SNS.csv'
#     np.savetxt(outputName, KLdiverg_SNS, delimiter=",")
#     outputName = folderOutFeatDiverg + '/' + 'AllSubj_KLdivergence_NSS.csv'
#     np.savetxt(outputName, KLdiverg_NSS, delimiter=",")
#     outputName = folderOutFeatDiverg + '/' + 'AllSubj_JSdivergence.csv'
#     np.savetxt(outputName, JSdiverg, delimiter=",")

# # ##################################################################
# PLOTTING DIVERGENCE PER SUBJECTS
# loading data saved from above, so that it does not have to be calculated all the time
reader = csv.reader(open(folderOutFeatDiverg + '/' + 'AllSubj_KLdivergence_SNS.csv', "r"))
KLdiverg_SNS = np.array(list(reader)).astype("float")
reader = csv.reader(open(folderOutFeatDiverg + '/' + 'AllSubj_KLdivergence_NSS.csv', "r"))
KLdiverg_NSS = np.array(list(reader)).astype("float")
reader = csv.reader(open(folderOutFeatDiverg + '/' + 'AllSubj_JSdivergence.csv', "r"))
JSdiverg = np.array(list(reader)).astype("float")

#analysing std per channel or feature
KLdiverg_SNS_reshaped0=np.reshape(KLdiverg_SNS, (len(GeneralParams.patients),-1,HDParams.numFeat ))
KLdiverg_NSS_reshaped0=np.reshape(KLdiverg_NSS, (len(GeneralParams.patients),-1,HDParams.numFeat ))
JSdiverg_reshaped0=np.reshape(JSdiverg, (len(GeneralParams.patients),-1,HDParams.numFeat ))
#reshape 3D matrix depending if we do analysis per feature or per channel
if (HDParams.bindingFeatures == 'FeatAppend'):
    KLdiverg_SNS_reshaped=KLdiverg_SNS_reshaped0
    KLdiverg_NSS_reshaped=KLdiverg_NSS_reshaped0
    JSdiverg_reshaped=JSdiverg_reshaped0
elif (HDParams.bindingFeatures == 'ChAppend'):
    KLdiverg_SNS_reshaped= np.zeros((   len(GeneralParams.patients), HDParams.numFeat, HDParams.numCh))
    KLdiverg_NSS_reshaped= np.zeros((   len(GeneralParams.patients), HDParams.numFeat, HDParams.numCh))
    JSdiverg_reshaped= np.zeros((   len(GeneralParams.patients), HDParams.numFeat, HDParams.numCh))
    for p in range(len(GeneralParams.patients)):
        KLdiverg_SNS_reshaped[p,:,:]=np.transpose(KLdiverg_SNS_reshaped0[p,:,:])
        KLdiverg_NSS_reshaped[p,:,:]=np.transpose(KLdiverg_NSS_reshaped0[p,:,:])
        JSdiverg_reshaped[p,:,:]=np.transpose(JSdiverg_reshaped0[p,:,:])

#calculate mean and std per feature/channel
KLdiverg_SNS_mean=np.nanmean(KLdiverg_SNS_reshaped,1)
KLdiverg_SNS_std=np.nanstd(KLdiverg_SNS_reshaped,1)
KLdiverg_NSS_mean=np.nanmean(KLdiverg_NSS_reshaped,1)
KLdiverg_NSS_std=np.nanstd(KLdiverg_NSS_reshaped,1)
JSdiverg_mean=np.nanmean(JSdiverg_reshaped,1)
JSdiverg_std=np.nanstd(JSdiverg_reshaped,1)


# # ##################################################################
# PLOTTING FOR ALL SUBJECTS INDIVIDUALLY BUT ON ONE PLOT
#visualize  mean JS divergence on headtopo
if (HDParams.bindingFeatures == 'ChAppend'):
    visualizeEEGtopoplot_AllSubj(JSdiverg_mean,  folderOutFeatDiverg+'/AllSubj_JSDivergence_EEGtopo', GeneralParams.patients, 0, 0,0.6)

#plotting JS as boxplots
fig1 = plt.figure(figsize=(16, 10), constrained_layout=False)
gs = GridSpec(4, 6, figure=fig1)
fig1.subplots_adjust(wspace=0.4, hspace=0.4)
#fig1.suptitle('Feature ')
xValues = np.arange(0,numElems, 1)
for p, pat in enumerate(GeneralParams.patients):
    ax1 = fig1.add_subplot(gs[int(np.floor(p / 6)), np.mod(p, 6)])
    ax1.errorbar(xValues, JSdiverg_mean[p, :], yerr=JSdiverg_std[p, :], fmt='indigo', label='JS')
    ax1.legend()
    if (np.mod(p, 6)==0 ):
        ax1.set_ylabel('Divergence')
    if (int(np.floor(p / 6))==3):
        if (HDParams.bindingFeatures == 'FeatAppend' ):
            ax1.set_xlabel('Feature')
        else:
            ax1.set_xlabel('Channel')
    ax1.set_title('Subj ' + pat)
    ax1.grid()
fig1.show()
fig1.savefig(folderOutFeatDiverg + '/AllSubj_JSivergenceMeasures_perSubj.png', bbox_inches='tight')
fig1.savefig(folderOutFeatDiverg + '/AllSubj_JSivergenceMeasures_perSubj.svg', bbox_inches='tight')
plt.close(fig1)

#plotting KL as boxplots
fig1 = plt.figure(figsize=(16, 10), constrained_layout=False)
gs = GridSpec(4, 6, figure=fig1)
fig1.subplots_adjust(wspace=0.4, hspace=0.4)
xValues = np.arange(0,numElems, 1)
for p, pat in enumerate(GeneralParams.patients):
    ax1 = fig1.add_subplot(gs[int(np.floor(p / 6)), np.mod(p, 6)])
    ax1.errorbar(xValues, KLdiverg_SNS_mean[p, :], yerr=KLdiverg_SNS_std[p, :], fmt='k', label='KL_SNS')
    ax1.errorbar(xValues, KLdiverg_NSS_mean[p, :], yerr=KLdiverg_NSS_std[p, :], fmt='indigo', label='KL_NSS')
    # ax1.errorbar(xValues, JSdiverg_mean[p, :], yerr=JSdiverg_std[p, :], fmt='k', label='JS')
    ax1.legend()
    if (int(np.floor(p / 6))==3):
        if (HDParams.bindingFeatures == 'FeatAppend'):
            ax1.set_xlabel('Feature')
        else:
            ax1.set_xlabel('Channel')
    if (np.mod(p, 6)==0 ):
        ax1.set_ylabel('Divergence')
    ax1.set_title('Subj ' + pat)
    ax1.grid()
fig1.show()
fig1.savefig(folderOutFeatDiverg + '/AllSubj_KLDivergenceMeasures_perSubj.png', bbox_inches='tight')
plt.close(fig1)


# # ##################################################################
# CALCULATING AVERAGE FOR ALL SUBJECTS
KLdiverg_SNS_meanAllSubj=np.nanmean(KLdiverg_SNS_mean,0)
KLdiverg_NSS_meanAllSubj=np.nanmean(KLdiverg_NSS_mean,0)
JSdiverg_meanAllSubj=np.nanmean(JSdiverg_mean,0)
KLdiverg_SNS_stdAllSubj=np.nanstd(KLdiverg_SNS_mean,0)
KLdiverg_NSS_stdAllSubj=np.nanstd(KLdiverg_NSS_mean,0)
JSdiverg_stdAllSubj=np.nanstd(JSdiverg_mean,0)
# save predictions
outputName = folderOutFeatDiverg + '/' + 'AllSubjAvrg_KLdivergence_SNS.csv'
np.savetxt(outputName, np.vstack((KLdiverg_SNS_meanAllSubj, KLdiverg_SNS_stdAllSubj)), delimiter=",")
outputName = folderOutFeatDiverg + '/' + 'AllSubjAvrg_KLdivergence_NSS.csv'
np.savetxt(outputName, np.vstack((KLdiverg_NSS_meanAllSubj, KLdiverg_NSS_stdAllSubj)), delimiter=",")
outputName = folderOutFeatDiverg + '/' + 'AllSubjAvrg_JSdivergence.csv'
np.savetxt(outputName, np.vstack((JSdiverg_meanAllSubj, JSdiverg_stdAllSubj)), delimiter=",")

#plotting KL in boxplots
fig1 = plt.figure(figsize=(10, 6), constrained_layout=False)
gs = GridSpec(1, 1, figure=fig1)
fig1.subplots_adjust(wspace=0.4, hspace=0.6)
xValues = np.arange(0, numElems, 1)
ax1 = fig1.add_subplot(gs[0,0])
ax1.errorbar(xValues, KLdiverg_SNS_meanAllSubj, yerr=KLdiverg_SNS_stdAllSubj, fmt='b', label='KL_SNS')
ax1.errorbar(xValues, KLdiverg_NSS_meanAllSubj, yerr=KLdiverg_NSS_stdAllSubj, fmt='m', label='KL_NSS')
ax1.errorbar(xValues, JSdiverg_meanAllSubj, yerr=JSdiverg_stdAllSubj, fmt='k', label='JS')
ax1.legend()
if (HDParams.bindingFeatures == 'FeatAppend'):
    ax1.set_xlabel('Feature')
else:
    ax1.set_xlabel('Channel')
ax1.set_xticks(xValues)
ax1.set_xticklabels(elemNames, fontsize=12 * 0.8, rotation=90)
ax1.set_ylabel('Divergence')
ax1.set_title('KL and JS divergences')
ax1.grid()
fig1.show()
fig1.savefig(folderOutFeatDiverg + '/AllSubj_DifferentDivergenceMeasures_avrgAllSubj.png', bbox_inches='tight')
plt.close(fig1)

# plot JS in Boxplots
fig1 = plt.figure(figsize=(10, 2), constrained_layout=False)
gs = GridSpec(1, 1, figure=fig1)
fig1.subplots_adjust(wspace=0.3, hspace=0.3)
xValues = np.arange(1, numElems +1, 1)
ax1 = fig1.add_subplot(gs[0, 0])
ax1.boxplot(JSdiverg_mean, medianprops=dict(color='orangered', linewidth=1.5),
            boxprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5), showfliers=False)
ax1.set_title('Jensen-Shannon divergence')
ax1.grid()
ax1.set_xticks(xValues)
ax1.set_xticklabels(elemNames, fontsize=12 * 0.8, rotation=45)
# ax1.set_xlabel('Feature')
fig1.show()
fig1.savefig(folderOutFeatDiverg + '/AllSubj_DifferentDivergenceMeasures_boxplots.png', bbox_inches='tight')
fig1.savefig(folderOutFeatDiverg + '/AllSubj_DifferentDivergenceMeasures_boxplots.svg', bbox_inches='tight')
plt.close(fig1)

#visualize  mean JS divergence on headtopo
if (HDParams.bindingFeatures == 'ChAppend'):
    visualizeEEGtopoplot(JSdiverg_meanAllSubj, folderOutFeatDiverg+'/AllSubjMean_JSDivergence_EEGtopo', 0, -1, -1)