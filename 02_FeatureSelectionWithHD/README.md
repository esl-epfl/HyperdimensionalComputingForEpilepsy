## Code description
	
Code for testing different encoding approaches of the spatio-temporal data (such as EEG or EMG) to HD vectors. Using one of the possible encoding approaches (FeatAppend) we demonstrate feature selection with HD computing. 
It is tested on a use case of epileptic seizure detection from EEG data, using the publicly available CHB-MIT database.

Code compares several different approaches encoding approaches: 
- FeatxVal
- FeatxChxVal  and ChxFeatxVal 
- ChFeatCombxVal 
- FeatAppend 
In the end, using FeatAppend approach several ways to do feature selection are tested: 
- Using performance per feature for feature selection 
- Using confidence per feature 
- Using approach that takes both performance and correlation of features 

### Related to paper

Paper: „ExG Signal Feature Selection Using Hyperdimensional Computing Encoding“, Una Pale, Tomas Teijeiro, David Atienza

2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), https://ieeexplore.ieee.org/abstract/document/9995107

Arxiv version: https://arxiv.org/abs/2205.07654 


## Python files description
script_prepareDataset.py
- loads raw CHB-MIT files in .edf format and transforms them to prepared dataset 
- it can be done using differnt factor (in this case 10) that defines how much more non-seizure data we want to keep for each seizure episode
- outputs are .csv files where in each file is one seizure episode and 'factor' times more non seizure data 

script_featureDivergence.py 
- uses raw data created by script_prepareDataset, creates histogram of valus during different labels and calculates divergence per feature (and channel)
- it measures divergence as Kulback-Leibler and Jensen-Shannon divergence 
- plots divergences per feature for each subject (average for all channels) and also average of all subjects 

script_runDifferentHDBinding.py 
- script that runs HD training using different encoding approaches 
	- encoding is defined in variable HDParams.bindingFeatures 
- for every encoding it runs standard HD (StdHD) and also online HD (OnlineHD) approach 
	- it also runs random forest training as a baseline 
- it saves predictions in time for each cross-validation 
	- for FeatAppend approach it also saves predictions per feature in time (as well as distances from classes etc)
- later it is used to also analyse feature qualities 
	- separability of model vectors for seizure and non-seizure class of each feature 
	- average confidences per feature 
	- average performance per feature 
	- correlation between those values 
	
script_calculatePredictions_FeatSelMoreOptions.py
- performs different feature selection 
	- based on parameter VotingParam.approach 
	- options are: FeatPerformance, FeatConfidence and OptimalFeatOrder
- outputs predictions and performance when adding one by one feature (or certain perfcentages of features) that are sorted based on VotingParam.approach from above 
- plots:
	- performance when increasing number of features (average over all subj)
	- optimal order of features (average over all subj)
	- optimal number of features chosen per subject and performance improvement (drop) with that number of features vs when using all features 
	
script_compareApproaches.py
- compares performances of different encoding 
	- loads performances for different encoding approaches and plots them in a simple way to visually compare 
- compares memory and computational complexity of different encoding approaches 
	- calculates and plots memory and number of operations needed for each endcoding approach 
- creates table with all results on feature selection 
	- for all three approaches of feature selection 
	- reports improvements on performance and number of features chosen

HdfunctionsLib.py
- library with different functions on HD vectors, uses torch library

VariousFunctionsLib.py 
- library including various functions for HD project but not necessarily related to HD vectors

PerformanceMetricsLib.py
- Library with functions to measure performance on episode and duration level (for epilepsy application)

paramtersSetup.py
- script where all important parameters are defined and grouped into several categories 