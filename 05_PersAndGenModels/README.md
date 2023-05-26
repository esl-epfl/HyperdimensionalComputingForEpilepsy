## Personalized and generalized models with HD computing for epilepsy detection

Here we demonstrate a few additional aspects in which HD computing, and the way its models are built and stored, can be used for further understanding, comparing, and creating more advanced machine learning models for epilepsy detection. These possibilities are not feasible with other state-of-the-art models, such as random forests or neural networks. 
We compare inter-subject similarity of models per different classes (seizure and non-seizure), then study the process of creation of generalized models from personalized ones, and in the end, how to combine personalized and generalized models to create hybrid models. This results in improved epilepsy detection performance. We also tested knowledge transfer between models created on two different datasets. Finally, all those examples could be highly interesting not only from an engineering perspective to create better models for wearables, but also from a neurological perspective to better understand individual epilepsy patterns. 


### Related publication
Paper: „Combining General and Personalized Models for Epilepsy Detection with Hyperdimensional Computing“, Una Pale, Tomas Teijeiro, David Atienza

Arxiv version: https://arxiv.org/abs/2303.14745 


## Scripts

### Scripts with all parameters 
- baseParams.py  - for loading all parameters
- datasetParams.py - parameters related to Datasets and specific epilepsy use case
- generalParams.py - parameters related to HD computing and machine learning models 


### Scripts for preparing datasets
loading and rearanging, filtering and feature calculation

script_prepareDataset_Repomse.py
- Script that loads raw Repomse data and adapts it: 
	- keeps only subjects with more than 3 seizures, with correct frequency, channels and labels 
	- resaves it in format NS-S-NS  with also labels 
	- calculates and saves features that we need for ML later 
	- calculates and visualizes some statistics on dataset

script_prepareDataset_CHBMIT.py
- Script that prepares CHBMIT databse 
- several things are done (comment out things not needed) 
	- converts edf files to csv files without any changes in data 
	- calculates features for each input file and saves them 
		- possible to choose which features (e.g. meanAmpl, lineLength, FrequencyFeatues, ZeroCross 
	- performas subselection of data or reorders is 
		- subselection x1 - contains the same amount of seizure and non seizure data (non seizure randomly selected from files with no seizure)
		- subselection x10 - the same as x1, but contains 10 times more non-seizure data then seizure 
		- all data StoS - rearanges all data so that each file contains data from end of previous seizure to beginning of current seizure 
		- all data fixed size files - rearanges all data to contain fixes amount of data (e.g. 1h or 4h) 
	- also plots labels for any rearangement - this is useful to check if rearangement was done correctly and no data was lost 


### Personalized HD computing training 
script_HDtraining.py
- script that performs HD computing (and RF) training and testing of seizures detection
	- uses files that are prepared using script_prepareDatasets_...
		- with parameter "datasetPreparationType" possible to choose which dataset to use
	- uses 3 different ML models 
		- standard Random Forest (RF) model with 100 trees 
		- standard HD computing where vectors of all samples from the same class are accumulated 
		- online HD computing which used weighted approach so that if current sample vector is alsoready similar it is multiplied with lower weight 
			(this usually helps to prevent majority class dominating model vectors) 
	- possible to perform training on 2 ways
		- leave one file out - train on all but that one (this doesn't keep time information into account - that some data was before another one) 
		- rolling base approach - uses all previous files to train for the current file and tests on current file 
	- it can perform personalized training or load already trained HD models (e.g. 'generalized', 'NSpers_Sgen', 'NSgen_Spers'
	- script saves predictions (raw and also after different predictions smoothing processes) 
	- also calculates performance and plots per subject and in average of all subjects 
	- in the end compares prediction and performance between different models
- parameter GeneralDatasetParams.persGenApproach has to be set to 'personalized' in order to train initial personalized models



### Comparing personalized models 
script_HDvecSimilarity.py
- Script that loads saved HD model vectors for all subjects and compare them: 
	- in intra subject manner - similarity of individual S and NS within the same subject 
	- in inter subject manner - similarity of S and NS between different subjects 


### Creating generalized models 
script_agregatePersVec.py
- Script that focuses on generating generalized vectors from personalized ones 
	- different ways of generating are tested: mean of personalize vectors, weighted adding of personalized  vectors with subtraction from opposite class and doing this itteratively 
	- evolution of generalized vectors as adding more subjects is also tested 


### Testing generalized models
script_HDtraining.py 
- but parameter GeneralDatasetParams.persGenApproach has to be set to 'generalzed'/'NSpers_Sgen'/'NSgen_Spers'

### Libraries with various functions
REPOMSEfeaturesLib.py
- library with functions related to reading Repomse dataset 

featuresLib.py 
- library with various functions related to calculating EEG features (mean amplitude, line length, frequency features, entropy, approximate zero crossing), KL divergence, and plotting raw data or features 

HDFunctionsLib.py 
- library with different functions on HD vectors, from learning, predictions, calculating probabilities etc. 
- it uses torch library 

PersGenVectLib.py 
- library for oparating with HD models and vectors, both personalized and generalized

PerformanceMetricsLib.py 
- library with different functions for assesing predicion performance in the use case of epilepsy'

MITAnnotation.py
- module with definition of classes to work with annotations in the MIT format