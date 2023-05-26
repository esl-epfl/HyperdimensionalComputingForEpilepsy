## Code description

Code for multi-centroid HD computing approach. Tested on a use case of epileptic seizure detection. 
Code compares standard single pass but 2 class HD approach with multi-centroid approach. 
Also tests two different approaches to reduce number of subclasses: removing less populated and clustering. 
Analysis is done using publicly available CHB-MIT database. 

Paper: „Multi-Centroid Hyperdimensional Computing Approach for Epileptic Seizure Detection“, Una Pale, Tomas Teijeiro, David Atienza

Frontiers in Neurology, 13, 1-13, 816294, https://www.frontiersin.org/articles/10.3389/fneur.2022.816294/full 

Arxiv version: https://arxiv.org/abs/2111.08463


## Python files description
script_prepareDataset.py
- loads raw CHB-MIT files in .edf format and transforms them to prepared dataset 
- it can be done using differnt factor (1,5 or 10) that defines how much more non-seizure data we want to keep for each seizure episode
- outputs are .csv files where in each file is one seizure episode and 'factor' times more non seizure data 

script_MultiClassPaper.py 
- main script that first calculated features for all files 
- then performs personalized training with leave-one-out cross-validation of each subject using several approaches 
	- standard 2 class model (2C)
	- multi-centroid model (MC)
	- MC with step of removing less commong subclasses 
	- MC with step of clustering subclasses to reduce their number 
- plots and compares performances 
- plots performances in dependence of 'factor' and amount of postprocessing

HdfunctionsLib.py
- library with different functions on HD vectors, uses torch library

VariousFunctionsLib.py 
- library including various functions for HD project but not necessarily related to HD vectors

PerformanceMetricsLib.py
- Library with functions to measure performance on episode and duration level (for epilepsy application)

paramtersSetup.py
- script where all important parameters are defined and grouped into several cathegories 