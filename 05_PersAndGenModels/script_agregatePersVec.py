__author__ = "Una Pale"
__email__ = "una.pale@epfl.ch"

'''
Script that focuses on generating generalized vectors from personalized ones 
- different ways of generating are tested: mean of personalize vectors, weighted adding of personalized 
vectors with subtraction from opposite class and doing this itteratively 
- evolution of generalized vectors as adding more subjects is also tested 
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

# #########################################
# # CREATE VECTORS DATA STRUCTURE FROM ALL SUBJ
# # vectors, vectors_norm, numAdded_vec  = createVectorsDataStructure_PerSubj(dataOutFolder+ '/Approach_personalized'+suffixName +'/', HDParams,GeneralDatasetParams.patients, GeneralDatasetParams.CVtype)
# vectors0, vectors_norm0, numAdded_vec0 = createVectorsDataStructure(dataOutFolder+ '/Approach_personalized'+suffixName +'/',  HDParams, GeneralDatasetParams.patients) #OLD ?
# numPat= len(numAdded_vec0)
#
# ########################################
# # PERSONAL VECTOR AGGRGATION
# # Personnal vectors aggregation
# print('PERSONALIZED VECTORS AGGREGATION')
# vectors, vectors_norm, numAdded_vec  =createOnePersVecPePerson(vectors0, vectors_norm0,numAdded_vec0, methodLists,  GeneralDatasetParams, HDParams)
#
# #saving personalized vectors
# outName = folderGenVectors + 'PersonalizedVectorsForEachSubj.pickle'
# with open(outName, 'wb') as f:
#     pickle.dump([vectors, vectors_norm, numAdded_vec ], f)

#load persnal vectors files not to have to calculate them again above
outName = folderGenVectors + 'PersonalizedVectorsForEachSubj.pickle'
with open(outName, 'rb') as file:
    # data = pickle.load(file)
    ( vectors, vectors_norm, numAdded_vec ) =pickle.load(file)

#
# # #########################################
# ## EVOLUTION OF GENERALIZED
# # study evolution of generalized when addding one by one subject (generalized created by mean of pers vectors)
# studyEvolutionOfGeneralizedVectors_fromPers_v2(vectors_norm,  similarityType, folderGenVectors, methodLists, fact=1)
# #study evolution with different ways of creating generalized models
# # studyEvolutionOfGeneralizedVectors_fromPers_diffMethods( vectors_norm, similarityType, folderGenVectors, methodLists, 'Avrg', fact=10)
# # studyEvolutionOfGeneralizedVectors_fromPers_diffMethods( vectors_norm, similarityType, folderGenVectors, methodLists, 'WAdd', fact=10)
# # studyEvolutionOfGeneralizedVectors_fromPers_diffMethods( vectors_norm, similarityType, folderGenVectors, methodLists, 'WAdd&Sub', fact=10)
# studyEvolutionOfGeneralizedVectors_fromPers_diffMethods( vectors_norm, similarityType, folderGenVectors, methodLists, 'WSub', fact=10)
#
# # #########################################
# ##  CREATING GENERALIZED VECTORS (USING MEAN OF NORM INDIV PERSONALIZED VECTORS)
# GenVectors = createGeneralizedVectors(vectors, vectors_norm, numAdded_vec, HDParams, folderGenVectors, methodLists, -1)
# # create gen vectors whn laeving one subj out
# for patIndx, patient in enumerate(tqdm(vectors_norm, desc='General-patient similarities', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}')):
#     createGeneralizedVectors_exceptOneSubj(vectors, vectors_norm, numAdded_vec, HDParams, patient, folderGenVectors, methodLists, -1)
#
#
# ########################################
# # COMAPRING DIFFERENT WAYS TO CREATE GENERALIZED
# #  CREATING GENERALIZED VECTORS USING AVERAGE OF PERS VECTOS
# GenVectorsAverage= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors,methodLists, 'Average', 1)
#
# #  CREATING GENERALIZED VECTORS USING WEIGHTED ADDING AND SUBTRACTING OF INDIV PERS VECTORS
# GenVectors= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors,methodLists, 'Weighted', 1)
# # weighted with itterations
# GenVectors= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors, methodLists,'Weighted', 10)
# GenVectors= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors, methodLists, 'Weighted', 100)
# approachesList=['Average', 'Weighted', 'Weighted_Itter10', 'Weighted_Itter100']
# compareDiffGenCreatingApproaches( approachesList, folderGenVectors, vectors_norm, methodLists)
#
#
# # combinations with adding and subtraction
# GenVectorsAverage= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors,methodLists, 'Avrg', 1)
# # GenVectors= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors,methodLists, 'WAdd', 1)
# GenVectors= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors, methodLists,'WAdd&Sub', 1)
# GenVectors= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors, methodLists, 'WSub', 1)
# approachesList=['Avrg', 'WAdd&Sub', 'WSub']
# compareDiffGenCreatingApproaches( approachesList, folderGenVectors, vectors_norm, methodLists)

# GenVectors= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors, methodLists,'WAdd&Sub', 10)
GenVectors= createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec, HDParams, folderGenVectors, methodLists, 'WSub', 10)
approachesList=['Avrg', 'WAdd&Sub', 'WSub', 'WSub_Itter10' ]
compareDiffGenCreatingApproaches( approachesList, folderGenVectors, vectors_norm, methodLists)

