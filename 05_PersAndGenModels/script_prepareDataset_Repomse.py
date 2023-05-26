__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

'''
Script that loads raw Repomse data and adapts it: 
- keeps only subjects with more than 3 seizures, with correct frequency, channels and labels 
- resaves it in format NS-S-NS  with also labels 
- calculates and saves features that we need for ML later 
- calculates and visualizes some statistics on dataset
'''
import warnings
warnings.filterwarnings('ignore')
import baseParams
params = baseParams.getArgs(['datasetParams','generalParams'])
from datasetParams import *
from REPOMSEfunctionsLib import *


######################################################################
### REPOMSE DATABASE
Dataset='01_Repomse'
# folderIn = '../../../../shares/eslfiler1/databases/medical/chuv/repomse-epilepsy-2019.12.13/Data/' #when running in pycharm ?
folderIn = '../../../../databases/medical/chuv/repomse-epilepsy-2019.12.13/Data/' #when running on server
folderOutBase = '../' +Dataset
createFolderIfNotExists(folderOutBase)

######################################################################
### KEEP ONLY GOOD FILES AND RESAVE THEM IN GZIP, ALSO CALCULATES FEATURES
saveRawDataPreprocessed=1
extractFeatures=1
prepareDataset_REPOMSE(folderIn, folderOutBase, DatasetPreprocessParams, FeaturesParams, saveRawDataPreprocessed,extractFeatures )

######################################################################
### ANALYSE DATASET
# Load file with statistics
folderOut=folderOutBase+ '/01_datasetProcessed/'
df= pd.read_csv( folderOut + '/00_numSeizPerSubj_DF.csv')

# Plot number of original seizures per subject as histogram
numSeizPerPat=df['numSeiz'].to_numpy()
fig1 = plt.figure(figsize=(12, 4), constrained_layout=False)
gs = GridSpec(1, 1, figure=fig1)
fig1.subplots_adjust(wspace=0.2, hspace=0.2)
numBins = int(np.max(numSeizPerPat))
n, bins, patches = plt.hist(numSeizPerPat, numBins, facecolor='blue', alpha=0.5)
plt.grid()
plt.xlabel('Number of seizures')
plt.ylabel('Number of patients')
plt.show()
plt.savefig(folderOut + '/NumSeizHistogram.png', bbox_inches='tight')
plt.savefig(folderOut + '/NumSeizHistogram.svg', bbox_inches='tight')
plt.close(fig1)

# Plot reasons for discarding specific subject
matrixReasons=df.to_numpy()[:,3:]
matrixReasons[np.where(matrixReasons>1)]=1
ReasonNames=['No enough S', 'File opening', 'Freq wrong', 'Ch wrong', 'Seiz len', 'Seiz label', 'All reasons']
percSubj=np.mean(matrixReasons,0)
fig = plt.figure(figsize=(12, 4), constrained_layout=False)
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
xValues=np.arange(0,len(ReasonNames),1)
# ax.bar(xValues, percSubj)
bars = ax.barh(xValues, percSubj*100)
ax.grid()
ax.bar_label(bars)
ax.set_yticks(xValues)
ax.set_yticklabels(ReasonNames, fontsize=12, rotation=0)
ax.set_xlim([0,100])
ax.set_xlabel('Percentage [%]')
ax.set_title('Reasons for rejecting subjects')
plt.show()
plt.savefig(folderOut + '/RejectingSubjects.png', bbox_inches='tight')
plt.savefig(folderOut + '/RejectingSubjects.svg', bbox_inches='tight')
plt.close()

