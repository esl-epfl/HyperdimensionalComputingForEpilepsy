__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"
''' 
- script that for each subject decides for optimal number of features/channels based on adding one by one (based on chosen approach of adding them)
- visualizes
	- performance increase when adding one by one feature/channel (F1E, F1DE, numFP)
	- chosen features/channels - in case of channels visualizes in a head topoplot
'''

from HDfunctionsLib import *
from parametersSetup import *
from sklearn.inspection import permutation_importance
from scipy.special import rel_entr


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
# FeaturesParams.featNames = constructAllfeatNames(FeaturesParams)  # feat used here
FeaturesParams.featNames=  ['mean_ampl', 'line_length','p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot']
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
    folderOutFeatCorr = folderOutFeatDiffMeas + '/FeatureCorrelations/'
elif (HDParams.bindingFeatures == 'ChAppend'):
    folderOutFeatDiverg = folderOutFeatDiffMeas + '/ChannelDivergence/'
    folderOutFeatCorr = folderOutFeatDiffMeas + '/ChannelCorrelations/'
createFolderIfNotExists(folderOutFeatDiverg)
createFolderIfNotExists(folderOutFeatCorr)


# ## CREATING OUTPUT FOLDERS AND STORING PARAMETERS IN A NAME
folderOutParams = FeaturesUsed + '_' + str(HDParams.numSegmentationLevels) + '_' + HDParams.bindingFeatures + '_D' + str( HDParams.D)  # not all parameters are saved here (but coudl be if needed)
folderOutPredictions = folderOutPredictions0 + '/' + folderOutParams + '_RF_StdHD_OnlHD/'

HDParams.numCh=len(DatasetPreprocessParams.channelNamesToKeep)
numFeat=HDParams.numFeat* len(DatasetPreprocessParams.channelNamesToKeep)


###################################################################################
# FIND OPTIMAL NUMBER OF FEATURES/CHANNELS BY ADDING ONE BY ONE
HDtype='OnlineHD' #'StdHD','OnlineHD'
perfMetricIndx = 7  # 2 for F1E, 7 for F1DE
perfMetricName='F1DE'
if (HDParams.bindingFeatures == 'FeatAppend'):
    folderOutFeatOrder=folderOutFeatDiffMeas+'/FeatureOptimalOrder_'+HDtype+'_'+ perfMetricName+'/'
    nameForGraphs='Feature'
else:
    folderOutFeatOrder=folderOutFeatDiffMeas+'/ChannelOptimalOrder_'+HDtype+'_'+ perfMetricName+'/'
    nameForGraphs = 'Channels'
createFolderIfNotExists(folderOutFeatOrder)
numPat=len(GeneralParams.patients)
FeatureRankings_AllSubj = np.zeros((numPat,  numElems ))
ChosenFeatures_AllSubj= np.zeros((numPat,  numElems ))
PerformancesTrain_AllSubj = np.zeros(((numElems ), 9, numPat))
PerformancesPerFeat_AllSubj = np.zeros(((numElems ), 9, numPat))
for patIndx, pat in enumerate(GeneralParams.patients):
    filesAll = np.sort(glob.glob(folderOutPredictions + '/'+FeatChType+'/*Subj' + pat + '*_'+HDtype+'_'+FeatChType+'_TrainPredictions.csv*'))
    print('-- Patient:', pat, 'NumSeizures:', len(filesAll))
    FeatureRankings_ThisSubj = np.zeros((len(filesAll), numElems ))
    PerformancesTrain_ThisSubj = np.zeros(((numElems ), 9, len(filesAll)))
    PerformancesPerFeat_ThisSubj = np.zeros(((numElems), 9, len(filesAll)))
    for fIndx, fileName in enumerate(filesAll):
        PredictionsTrain=readDataFromFile(fileName)

        pom, fileName1 = os.path.split(fileName)
        fileName2 = fileName1[0:-30]
        distancesFromS=readDataFromFile(fileName[0:-24]+'_DistancesFromSeiz_Train.csv.gz')
        distancesFromNS = readDataFromFile(fileName[0:-24] + '_DistancesFromNonSeiz_Train.csv.gz')

        if (fIndx==0):
            PredictionsTrainAll=PredictionsTrain
            distancesFromSAll=distancesFromS
            distancesFromNSAll=distancesFromNS
        else:
            PredictionsTrainAll=np.vstack((PredictionsTrainAll, PredictionsTrain))
            distancesFromSAll = np.vstack((distancesFromSAll, distancesFromS))
            distancesFromNSAll = np.vstack((distancesFromNSAll, distancesFromNS))

        (chosenFeatsInOrder, featRanking, perfMeasuresWithMoreFeat, performancesPerFeat)=chooseOptimalFeatureOrder(PredictionsTrain[:,:-2], distancesFromS[:,:-1], distancesFromNS[:,:-1], PredictionsTrain[:,-1], PostprocessingParams, FeaturesParams,perfMetricIndx)
        dataToSave=np.vstack((chosenFeatsInOrder.reshape((1,-1)), featRanking.reshape((1,-1)), perfMeasuresWithMoreFeat.transpose()))
        outputName = folderOutFeatOrder+'/'+fileName2+'_TrainOptimalOrdering.csv'
        saveDataToFile( dataToSave, outputName, 'gzip')

        FeatureRankings_ThisSubj[fIndx, :]=featRanking
        PerformancesTrain_ThisSubj[:,:, fIndx]=perfMeasuresWithMoreFeat
        PerformancesPerFeat_ThisSubj[:,:,fIndx]=performancesPerFeat

    outputName = folderOutFeatOrder + '/Subj'+ pat +'_OptimOrder_Rankings.csv'
    saveDataToFile(FeatureRankings_ThisSubj, outputName, 'gzip')
    outputName = folderOutFeatOrder + '/Subj'+ pat +'_OptimOrder_PerformanceIncrease.csv'
    saveDataToFile(np.nanmean(PerformancesTrain_ThisSubj,2), outputName, 'gzip')

    FeatureRankings_AllSubj[patIndx,:]=np.nanmean(FeatureRankings_ThisSubj, 0)
    PerformancesTrain_AllSubj[:, :, patIndx] = np.nanmean(PerformancesTrain_ThisSubj, 2)
    PerformancesPerFeat_AllSubj[:, :, patIndx] = np.nanmean(PerformancesPerFeat_ThisSubj, 2)

    #find optimal num of features for this subj
    (optNumFeat, _)= findOptimumNumFeatures(PerformancesTrain_AllSubj[:, 7, patIndx] ) #7 for F1DE
    indxChosen=np.argsort(FeatureRankings_AllSubj[patIndx,:])[0:optNumFeat]
    ChosenFeatures_AllSubj[patIndx,indxChosen]=1

    #PAPER: Plot rankings and performance increase when adding one by one feature
    meanRankings=np.nanmean(FeatureRankings_ThisSubj, 0)
    sortedIndx=np.argsort(meanRankings).astype(int)
    fig1 = plt.figure(figsize=(10, 8), constrained_layout=False)
    gs = GridSpec(3, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.5)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.boxplot(FeatureRankings_ThisSubj, medianprops=dict(color='purple', linewidth=1.5),
                boxprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5), showfliers=False)
    xValues = np.arange(1, numElems +1, 1)
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(elemNames, fontsize=10 * 0.8 , rotation=30)
    ax1.set_title(nameForGraphs+ ' ranking')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[1, 0])
    ax1.errorbar(xValues, np.nanmean(PerformancesTrain_ThisSubj[:,2], 1), yerr= np.nanstd(PerformancesTrain_ThisSubj[:,2], 1), color='mediumvioletred')
    ax1.errorbar(xValues, np.nanmean(PerformancesTrain_ThisSubj[:, 5], 1),yerr=np.nanstd(PerformancesTrain_ThisSubj[:, 5], 1), color='orchid')
    ax1.errorbar(xValues, np.nanmean(PerformancesTrain_ThisSubj[:, 7], 1), yerr=np.nanstd(PerformancesTrain_ThisSubj[:, 7], 1), color='purple')
    xValues = np.arange(1, numElems +1, 1)
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(xValues, fontsize=12 * 0.8) #, rotation=90)
    ax1.set_xlabel('Adding one by one '+nameForGraphs)
    ax1.legend(['F1E', 'F1D','F1DE'])
    ax1.set_title(nameForGraphs +' performance')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[2, 0])
    ax1.errorbar(xValues, np.nanmean(PerformancesTrain_ThisSubj[:,8], 1), yerr= np.nanstd(PerformancesTrain_ThisSubj[:,8], 1), color='black')
    xValues = np.arange(1, numElems +1, 1)
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(xValues, fontsize=12 * 0.8) #, rotation=90)
    ax1.set_xlabel('Adding one by one '+nameForGraphs)
    ax1.set_title('Number FP per day')
    ax1.grid()
    fig1.savefig(folderOutFeatOrder + '/Subj'+ pat+'_OptimOrder.png', bbox_inches='tight')
    fig1.savefig(folderOutFeatOrder + '/Subj' + pat + '_OptimOrder.svg', bbox_inches='tight')
    plt.close(fig1)

    #PAPER: PLOTS VARIOUS TOPOPLOTS FOR THIS SUBJECT ON ONE GRAPH
    if (HDParams.bindingFeatures == 'ChAppend' ):
        plotNames=['Performance per channel','Channel rankings','Chosen channels ('+ str(int(np.sum(ChosenFeatures_AllSubj[patIndx,:])))+')' ] # 'Performance when adding channels',
        dataToPlot=np.zeros((3,numElems))
        dataToPlot[0,:]=PerformancesPerFeat_AllSubj[:, perfMetricIndx, patIndx]
        dataToPlot[1,:]=np.abs( FeatureRankings_AllSubj[patIndx,:]-numElems)
        # dataToPlot[2,:]=PerformancesTrain_AllSubj[:, perfMetricIndx, patIndx]
        dataToPlot[2,:]=ChosenFeatures_AllSubj[patIndx,:]
        norm=[0,0,0]
        vmin=[0,0,0]
        vmax=[1,numElems,1]
        visualizeEEGtopoplot_AllPlotsForOneSubj(dataToPlot, folderOutFeatOrder+'/Subj'+ pat, norm, vmin, vmax, plotNames)

outputName = folderOutFeatOrder + '/AllSubj_OptimOrder_Rankings.csv'
saveDataToFile(FeatureRankings_AllSubj, outputName, 'gzip')
outputName = folderOutFeatOrder + '/AllSubj_OptimOrder_PerformanceIncrease.csv'
saveDataToFile(np.nanmean(PerformancesTrain_AllSubj,2), outputName, 'gzip')


#PAPER: PLOTS ON TOPOPLOTS FOR INDIVIDUAL SUBJECTS (BUT ALL ON ONE GRAPH) VARIOUS THINGS
if (HDParams.bindingFeatures == 'ChAppend'):
    #channel rankings - the darker the earlier chosen is the feature
    visualizeEEGtopoplot_AllSubj(np.abs(FeatureRankings_AllSubj-18),  folderOutFeatOrder+'/AllSubj_Rankings_EEGtopo', GeneralParams.patients, 0, -1,-1)
    #chosen channels - if dark then chosen
    visualizeEEGtopoplot_AllSubj(ChosenFeatures_AllSubj,  folderOutFeatOrder+'/AllSubj_ChosenFeatures_EEGtopo', GeneralParams.patients, 0, 0,1)
    # performnce per channel  - F1E and F1DE
    visualizeEEGtopoplot_AllSubj(PerformancesPerFeat_AllSubj[:,2,:].transpose(),  folderOutFeatOrder+'/AllSubj_PerformanceF1E_EEGtopo', GeneralParams.patients, 0, 0.5,1)
    visualizeEEGtopoplot_AllSubj(PerformancesPerFeat_AllSubj[:,7,:].transpose(),  folderOutFeatOrder+'/AllSubj_PerformanceF1DE_EEGtopo', GeneralParams.patients, 0, 0.5,1)


# PAPER: AVERAGE FOR ALL SUBJECTS
# Plot rankings and performance increase when adding one by one feature/channel
meanRankings = np.nanmean(FeatureRankings_AllSubj, 0)
sortedIndx = np.argsort(meanRankings)
fig1 = plt.figure(figsize=(10, 8), constrained_layout=False)
gs = GridSpec(3, 1, figure=fig1)
fig1.subplots_adjust(wspace=0.4, hspace=0.5)
ax1 = fig1.add_subplot(gs[0, 0])
ax1.boxplot(FeatureRankings_AllSubj, medianprops=dict(color='orangered', linewidth=1.5),
            boxprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),showfliers=False)
xValues = np.arange(1, numElems + 1, 1)
ax1.set_xticks(xValues)
ax1.set_xticklabels(elemNames, fontsize=10 * 0.8 , rotation=30)
ax1.set_title(nameForGraphs+ ' ranking')
ax1.grid()
ax1 = fig1.add_subplot(gs[1, 0])
ax1.errorbar(xValues, np.nanmean(PerformancesTrain_AllSubj[:, 2], 1), yerr=np.nanstd(PerformancesTrain_AllSubj[:, 2], 1), color='red')
ax1.errorbar(xValues, np.nanmean(PerformancesTrain_AllSubj[:, 5], 1),yerr=np.nanstd(PerformancesTrain_AllSubj[:, 5], 1), color='orange')
ax1.errorbar(xValues, np.nanmean(PerformancesTrain_AllSubj[:, 7], 1),yerr=np.nanstd(PerformancesTrain_AllSubj[:, 7], 1), color='magenta')
xValues = np.arange(1, numElems + 1, 1)
ax1.set_xticks(xValues)
ax1.set_xticklabels(xValues, fontsize=12 * 0.8) #, rotation=90)
ax1.set_xlabel('Adding one by one '+nameForGraphs)
ax1.legend(['F1E', 'F1D', 'F1DE'])
ax1.set_title(nameForGraphs+' performance')
ax1.grid()
ax1 = fig1.add_subplot(gs[2, 0])
ax1.errorbar(xValues, np.nanmean(PerformancesTrain_AllSubj[:, 8], 1), yerr=np.nanstd(PerformancesTrain_AllSubj[:, 8], 1), color='black')
xValues = np.arange(1, numElems + 1, 1)
ax1.set_xticks(xValues)
# ax1.set_xticklabels(elemNames[sortedIndx], fontsize=8 * 0.8, rotation=90)
ax1.set_xticklabels(xValues, fontsize=12 * 0.8) #, rotation=90)
ax1.set_xlabel('Adding one by one '+nameForGraphs)
ax1.set_title('Number FP per day')
ax1.grid()
fig1.savefig(folderOutFeatOrder + '/AllSubj_OptimOrder.png', bbox_inches='tight')
fig1.savefig(folderOutFeatOrder + '/AllSubj_OptimOrder.svg', bbox_inches='tight')
plt.close(fig1)