# HD_Explainability

## Description
HD computing approach for epilepsy detection 

Code for performing channel selection and then also visualizing decisions per channel in time. 



## Python scripts 

#### script01_prepareDataset.py
Loads raw CHB-MIT files in .edf format and transforms them to prepared dataset
- it can be done using differnt factor (in this case 10) that defines how much more non-seizure data we want to keep for each seizure episode
- outputs are .csv files where in each file is one seizure episode and 'factor' times more non seizure data

#### script02_featureDivergence.py
Uses raw data created by script_prepareDataset, creates histogram of valus during different labels and calculates divergence per feature-channel combinations
- it measures divergence as Kulback-Leibler and Jensen-Shannon divergence
- plots divergences per feature (avrg of all channels for that feature) or per channel (avrg of all features) for each subject (average for all channels) and also average of all subjects 

#### script03_runDifferentHDBinding.py 
Script that runs HD training using different encoding approaches
- encoding is defined in variable HDParams.bindingFeatures
- for every encoding it runs standard HD (StdHD), online HD (OnlineHD) and random forest (RD) approach
- it saves predictions in time for each cross-validation
- for FeatAppend approach it also saves predictions per feature in time (as well as distances from classes etc)
- later it is used to also analyse feature qualities
    - separability of model vectors for seizure and non-seizure class of each feature
    - average confidences per feature
    - average performance per feature
    - correlation between those values 
	
#### script04_predictionsWithSelection.py
Performs different feature selection 
	- based on parameter VotingParam.approach 
	- options are: Performance, Confidence and OptimalOrder
- outputs predictions and performance when adding one by one feature/channel (or certain percentage of feature/channels) that are sorted based on VotingParam.approach from above 
- plots:
	- performance when increasing number of features/channels (average over all subj)
	- optimal order of features/channels  (average over all subj)
	- optimal number of features/channels  chosen per subject and performance improvement (drop) with that number of features/channels  vs when using all features/channels   
	
#### script05_optimalChoices.py
Script that for each subject decides for optimal number of features/channels based on adding one by one (based on chosen approach of adding them)
- visualizes
	- performance increase when adding one by one feature/channel (F1E, F1DE, numFP)
	- chosen features/channels - in case of channels visualizes in a head topoplot
 

## Python scripts with functions 
 
#### HdfunctionsLib.py
Library with different functions on HD vectors, uses torch library

#### VariousFunctionsLib.py
Library including various functions for HD project but not necessarily related to HD vectors

#### PerformanceMetricsLib.py
Library with functions to measure performance on episode and duration level (for epilepsy application)

#### paramtersSetup.py
Script where all important parameters are defined and grouped into several categories 