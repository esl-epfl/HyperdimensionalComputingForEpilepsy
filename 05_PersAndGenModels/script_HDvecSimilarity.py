__author__ = "Una Pale"
__email__ = "una.pale@epfl.ch"

'''
Script that loads saved personalized HD model vectors for all subjects and compare them: 
- in intra subject manner - similarity of individual S and NS within the same subject 
- in inter subject manner - similarity of S and NS between different subjects 
'''

import warnings
warnings.filterwarnings('ignore')

from HDfunctionsLib import *
from PersGenVectLib import *
import baseParams
params = baseParams.getArgs(['datasetParams','generalParams'])
from generalParams import *
from datasetParams import *

##########################
# VARIOUS PARAMETERS TO SET
suffixName = '_OnlFact1&'+str(HDParams.onlineFNSfact)+'/'
stdHDName='StdHD' #'StdHD', 'ClassicHD'
onlHDName='OnlHD' #'OnlHD', 'OnlineHDAddSub'
methodLists = [stdHDName, onlHDName]
similarityType = 'hamming'  # 'hamming', 'cosine'
GeneralDatasetParams.CVtype = 'LeaveOneOut'  # 'LeaveOneOut', 'RollingBase'

##########################
#### DEFINE INPUT OUTPUT FOLDERS

## REPOMSE DATASET
Dataset='01_Repomse'
folderBase = '../' +Dataset
createFolderIfNotExists((folderBase))
folderFeaturesIn= folderBase+ '/02_Features/'

# ## CHBMIT DATASET
# Dataset='01_CHBMIT'
# GeneralDatasetParams.datasetPreparationType = 'AllDataWin3600s_1File6h'# 'AllDataWin3600s_1File6h'  # 'Fact1', 'Fact10' ,'Fact10_TheSameFile','AllDataWin3600s', 'AllDataWin3600s_1File6h', 'AllDataStoS'  # !!!!!!
# folderBase = '../' +Dataset +'_'+ GeneralDatasetParams.datasetPreparationType
# createFolderIfNotExists((folderBase))
# folderFeaturesIn= '../../10_datasets/CHBMIT/04_RearangedData_MergedFeatures_' + GeneralDatasetParams.datasetFiltering + '_' + str(FeaturesParams.winLen) + '_' + str(FeaturesParams.winStep) +\
#                   '_' + FeaturesParams.allFeatTypesName + '_' + GeneralDatasetParams.datasetPreparationType # where to save calculated features for each original file
# subjAll= [ '01', '02', '03', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24','07']


dataOutFolder = folderBase +  '/04_ResultsPERSGEN_' + IOParams.outFolderParameters+'_D'+str(HDParams.D)
if 'Time' in HDParams.bindingMethod:
    dataOutFolder=dataOutFolder+'_withTime_Random'
    # dataOutFolder=dataOutFolder+HDParams.bindingMethod
    HDParams.timeStepsWinShift=np.array(HDParams.timeStepsInSec)/FeaturesParams.winStep
if (GeneralDatasetParams.itterativeLearning==1):
    dataOutFolder=dataOutFolder+'_ITTER'
if (GeneralDatasetParams.CVType== 'LeaveOneOut'):
    dataOutFolder=dataOutFolder+'_L10'
createFolderIfNotExists(dataOutFolder)
folderGenVectors = dataOutFolder + '/HD_ModelVectors' + suffixName + '/'
createFolderIfNotExists(folderGenVectors)
path = dataOutFolder
folderSimilarities = dataOutFolder+'/Similarities_' + similarityType +suffixName+ '/'
createFolderIfNotExists(folderSimilarities)

#####################################################################################################
#####################################################################################################

#########################################
# EXTRACT NUMBER OF PATIENTS
GeneralDatasetParams.patients =[]
if (Dataset=='01_Repomse'):
    subjAll = np.sort(glob.glob(folderFeaturesIn + '/Pat*/' ))
    for patIndx, patfold in enumerate(subjAll):
        # if (patIndx < 10):
        path, pat = os.path.split(patfold[:-1])
        GeneralDatasetParams.patients.append(pat)
elif (Dataset=='01_CHBMIT'):
    for patIndx, pat in enumerate(subjAll):
        GeneralDatasetParams.patients.append('Subj'+ pat)

#############################
# ## LOAD AND CREATE VECTORS DATA STRUCTURE FROM ALL SUBJ (to use it in the rest of the file)
vectors0, vectors_norm0, numAdded_vec0 = createVectorsDataStructure(dataOutFolder+ '/Approach_personalized'+suffixName +'/',  HDParams, GeneralDatasetParams.patients)
numPat= len(numAdded_vec0)

##############################
# ## CALCULATE SIMILARITIES BETWEEN SEIZURES FOR EACH SUBJECT -  Intra-patient similarities
print('INTRA-PATIENT SIMILARITIES')
for patient in tqdm(vectors_norm0, desc = 'Intra-patient similarities', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}'):
    if (similarityType== 'hamming'):
        sim_onl = hamming_similarity_matrix( np.vstack([vectors_norm0[patient][onlHDName][0], vectors_norm0[patient][onlHDName][1]]))
        sim_std = hamming_similarity_matrix( np.vstack([vectors_norm0[patient][stdHDName][0], vectors_norm0[patient][stdHDName][1]]))
    else:
        sim_onl = cosine_similarity_matrix(np.vstack([vectors_norm0[patient][onlHDName]  [0], vectors_norm0[patient][onlHDName]  [1]]))
        sim_std = cosine_similarity_matrix(np.vstack([vectors_norm0[patient][stdHDName][0], vectors_norm0[patient][stdHDName][1]]))

    numNS=len(numAdded_vec0[patient][onlHDName][0])
    plotSimilarities_SandNSIn1Subj( f"Patient {patient} similarities, OnlineHD", f"{folderSimilarities}/IntraSubjSeiz_OnlHD_Subj{patient}", sim_onl, numNS, [0.5, 1]  )
    clusterFromSimilarity_SandNSIn1Subj(f"Patient {patient} similarities, OnlineHD", f"{folderSimilarities}/IntraSubjSeiz_OnlHD_Subj{patient}_Dendogram", sim_onl, numNS) #plot dendrogram
    numNS=len(numAdded_vec0[patient][stdHDName][0])
    plotSimilarities_SandNSIn1Subj( f"Patient {patient} similarities, StandardHD", f"{folderSimilarities}/IntraSubjSeiz_StdHD_Subj{patient}",sim_std, numNS, [0.5, 1])
    clusterFromSimilarity_SandNSIn1Subj(f"Patient {patient} similarities, StandardHD", f"{folderSimilarities}/IntraSubjSeiz_StdHD_Subj{patient}_Dendogram", sim_std, numNS) #plot dendrogram


################################
# ## CALCULATE SIMILARITIES BETWEEN SUBJECT -  Inter-patient similarities
print('INTER-PATIENT SIMILARITIES')

# Personnal vectors aggregation
(vectors_StdHD_NonSeiz, vectors_StdHD_Seiz, vectors_OnlHD_NonSeiz, vectors_OnlHD_Seiz) =createOnePersVecPePerson(vectors_norm0, stdHDName, onlHDName, GeneralDatasetParams, HDParams)

vectors_StdHD_NonSeiz_mat = np.array([vectors_StdHD_NonSeiz[i] for i in vectors_StdHD_NonSeiz.keys()])
vectors_OnlHD_NonSeiz_mat = np.array([vectors_OnlHD_NonSeiz[i] for i in vectors_OnlHD_NonSeiz.keys()])
vectors_StdHD_Seiz_mat = np.array([vectors_StdHD_Seiz[i] for i in vectors_StdHD_Seiz.keys()])
vectors_OnlHD_Seiz_mat = np.array([vectors_OnlHD_Seiz[i] for i in vectors_OnlHD_Seiz.keys()])

# print(" [....] Computing and plotting inter-patients similarities and distributions.", end = "", flush = True)
if (similarityType== 'cosine'):
    sim_std = cosine_similarity_matrix(np.vstack([vectors_StdHD_NonSeiz_mat, vectors_StdHD_Seiz_mat]))
    sim_onl = cosine_similarity_matrix(np.vstack([vectors_OnlHD_NonSeiz_mat, vectors_OnlHD_Seiz_mat]))
else:
    sim_std  = hamming_similarity_matrix(np.vstack([vectors_StdHD_NonSeiz_mat, vectors_StdHD_Seiz_mat]))
    sim_onl  = hamming_similarity_matrix(np.vstack([vectors_OnlHD_NonSeiz_mat, vectors_OnlHD_Seiz_mat]))
sim_std_nan = sim_std.copy()
sim_std_nan[sim_std_nan > 0.9999] = np.nan
sim_onl_nan = sim_onl.copy()
sim_onl_nan[sim_onl_nan > 0.9999] = np.nan
nb_pat = len(vectors_StdHD_NonSeiz_mat)

# Plotting inter-patient similarities
plotSimilarities(
    f"Similarity among patients, OnlineHD",
    f"{folderSimilarities}/InterSubj_OnlHD_AllSubj.png",
    [ "Patient (NonSeiz | Seiz)",
        f"Patient (Seiz ({np.nanmean(sim_onl_nan[nb_pat:, :nb_pat]):.3f} | {np.nanmean(sim_onl_nan[nb_pat:, nb_pat:]):.3f}) | NonSeiz ({np.nanmean(sim_onl_nan[:nb_pat, :nb_pat]):.3f} | {np.nanmean(sim_onl_nan[:nb_pat, nb_pat:]):.3f})"],
    np.tile(np.tile(GeneralDatasetParams.patients, 2)[np.newaxis,:], [2, 1]),
    sim_onl, [0, 1], cbar_horiz=False
)
plotSimilarities(
    f"Similarity among patients, OnlineHD",
    f"{folderSimilarities}/InterSubj_OnlHD_AllSubj.svg",
    [ "Patient (NonSeiz | Seiz)",
        f"Patient (Seiz ({np.nanmean(sim_onl_nan[nb_pat:, :nb_pat]):.3f} | {np.nanmean(sim_onl_nan[nb_pat:, nb_pat:]):.3f}) | NonSeiz ({np.nanmean(sim_onl_nan[:nb_pat, :nb_pat]):.3f} | {np.nanmean(sim_onl_nan[:nb_pat, nb_pat:]):.3f})"],
    np.tile(np.tile(GeneralDatasetParams.patients, 2)[np.newaxis,:], [2, 1]),
    sim_onl, [0, 1], cbar_horiz=False
)


plotSimilarityDistributions(
    "Similarity distributions, OnlineHD",
    f"{folderSimilarities}/InterSubj_OnlHD_AllSubj_Distr.png",
    ["NonSeiz vs NonSeiz", "NonSeiz vs Seiz", "Seiz vs Seiz"], 20,
    [sim_onl_nan[:nb_pat, :nb_pat].flatten(), sim_onl_nan[nb_pat:, :nb_pat].flatten(), sim_onl_nan[nb_pat:, nb_pat:].flatten()]
)

plotSimilarities(
    f"Similarity among patients, StandardHD",
    f"{folderSimilarities}/InterSubj_StdHD_AllSubj.png",
    [ "Patient (NonSeiz | Seiz)",
        f"Patient (Class Seiz ({np.nanmean(sim_std_nan[nb_pat:, :nb_pat]):.3f} | {np.nanmean(sim_std_nan[nb_pat:, nb_pat:]):.3f}) | Class NonSeiz ({np.nanmean(sim_std_nan[:nb_pat, :nb_pat]):.3f} | {np.nanmean(sim_std_nan[:nb_pat, nb_pat:]):.3f})"],
    np.tile(np.tile(GeneralDatasetParams.patients, 2)[np.newaxis,:], [2, 1]),
    sim_std, [0, 1], cbar_horiz=False
)
plotSimilarities(
    f"Similarity among patients, StandardHD",
    f"{folderSimilarities}/InterSubj_StdHD_AllSubj.svg",
    [ "Patient (NonSeiz | Seiz)",
        f"Patient (Class Seiz ({np.nanmean(sim_std_nan[nb_pat:, :nb_pat]):.3f} | {np.nanmean(sim_std_nan[nb_pat:, nb_pat:]):.3f}) | Class NonSeiz ({np.nanmean(sim_std_nan[:nb_pat, :nb_pat]):.3f} | {np.nanmean(sim_std_nan[:nb_pat, nb_pat:]):.3f})"],
    np.tile(np.tile(GeneralDatasetParams.patients, 2)[np.newaxis,:], [2, 1]),
    sim_std, [0, 1], cbar_horiz=False
)

plotSimilarityDistributions(
    "Similarity distributions, StandardHD",
    f"{folderSimilarities}/InterSubj_StdHD_AllSubj_Distr.png",
    ["NonSeiz vs NonSeiz", "NonSeiz vs Seiz", "Seiz vs Seiz"], 20,
    [ sim_std_nan[:nb_pat, :nb_pat].flatten(),  sim_std_nan[nb_pat:, :nb_pat].flatten(), sim_std_nan[nb_pat:, nb_pat:].flatten()]
)

################################
# PLOTTING IN A FORM OF BOXPLOTS
simSTD_InterSubj_SS=np.nanmean(sim_std_nan[nb_pat:, nb_pat:],0)
simSTD_InterSubj_NSNS=np.nanmean(sim_std_nan[:nb_pat, :nb_pat],0)
simSTD_InterSubj_SNS=np.nanmean(sim_std_nan[nb_pat:, :nb_pat],0)
simONL_InterSubj_SS=np.nanmean(sim_onl_nan[nb_pat:, nb_pat:],0)
simONL_InterSubj_NSNS=np.nanmean(sim_onl_nan[:nb_pat, :nb_pat],0)
simONL_InterSubj_SNS=np.nanmean(sim_onl_nan[nb_pat:, :nb_pat],0)
simSTD_InterSubj=np.vstack([simSTD_InterSubj_SS, simSTD_InterSubj_NSNS, simSTD_InterSubj_SNS])
simONL_InterSubj=np.vstack([simONL_InterSubj_SS, simONL_InterSubj_NSNS, simONL_InterSubj_SNS])

fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
gs = GridSpec(1, 2, figure=fig1)
fig1.subplots_adjust(wspace=0.25, hspace=0.25)
xValues = np.arange(1,4, 1)
ax1 = fig1.add_subplot(gs[0, 0])
ax1.boxplot(simSTD_InterSubj.transpose(), notch=False,  showfliers=False, showmeans=True)
ax1.set_ylabel('Average similarity')
ax1.set_xticks(xValues)
ax1.set_xticklabels(['S-S', 'NS-NS', 'S-NS'], fontsize=12 * 0.8, rotation=45)
ax1.set_title('Std HD')
ax1.grid()
ax1.set_ylim((0,1))
ax1 = fig1.add_subplot(gs[0, 1])
ax1.boxplot(simONL_InterSubj.transpose(), notch=False,  showfliers=False, showmeans=True)
ax1.set_ylabel('Average similarity')
ax1.set_xticks(xValues)
ax1.set_xticklabels(['S-S', 'NS-NS', 'S-NS'], fontsize=12 * 0.8, rotation=45)
ax1.set_title('Online HD')
ax1.grid()
ax1.set_ylim((0,1))
fig1.show()
fig1.savefig( f"{folderSimilarities}/AverageSim_InterSubj.png", bbox_inches='tight', dpi=100)
# fig1.savefig(  f"{folderSimilarities}/AverageSim_InterSubj.svg", bbox_inches='tight')
plt.close(fig1)

## Online HD only - for PAPER
fig1 = plt.figure(figsize=(3, 6), constrained_layout=False)
gs = GridSpec(1, 1, figure=fig1)
fig1.subplots_adjust(wspace=0.25, hspace=0.25)
xValues = np.arange(1,4, 1)
ax1 = fig1.add_subplot(gs[0, 0])
ax1.boxplot(simONL_InterSubj.transpose(), notch=False,  showfliers=False, showmeans=True)
ax1.set_ylabel('Average similarity')
ax1.set_xticks(xValues)
ax1.set_xticklabels(['S-S', 'NS-NS', 'S-NS'], fontsize=12 * 0.8) #, rotation=45)
ax1.set_title('Online HD')
ax1.grid()
ax1.set_ylim((0.5,1))
fig1.show()
fig1.savefig( f"{folderSimilarities}/ForPaper_AverageSim_InterSubj.png", bbox_inches='tight', dpi=100)
fig1.savefig( f"{folderSimilarities}/ForPaper_AverageSim_InterSubj.svg", bbox_inches='tight', dpi=100)
plt.close(fig1)

################################
## RUN STATISTICAL ANALYSIS BETWEEN BOXPLOTS  -  WILCOXON PAIRED TEST
print('--> WILCOXON:')
st = scipy.stats.wilcoxon(simONL_InterSubj_SS, simONL_InterSubj_NSNS)
print('SS vs NSNS:', st)
st = scipy.stats.wilcoxon(simONL_InterSubj_SS, simONL_InterSubj_SNS)
print('SS vs SNS:', st)
st = scipy.stats.wilcoxon(simONL_InterSubj_NSNS, simONL_InterSubj_SNS)
print('NSNS vs SNS:', st)