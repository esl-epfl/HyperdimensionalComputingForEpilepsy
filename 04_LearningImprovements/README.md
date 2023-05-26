## Code description

Code for different approaches for improving performance of baseline HD computing approach. Tested on a use case of epileptic seizure detection. 
Code compares several different approaches: 
- standard single-pass single-centroid HD learning
- iterative (multi-pass) HD learning
- multi-centroid HD learning
- multi-centroid and multi-pass learning combined
- onlineHD (weighted) learning
- random forest 
 
Analysis is done using publicly available CHB-MIT database. 

### Related publication 
Paper: „Exploration of Hyperdimensional Computing Strategies for Enhanced Learning on Epileptic Seizure Detection“, Una Pale, Tomas Teijeiro, David Atienza

2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), https://ieeexplore.ieee.org/document/9870919 

Arxiv version: https://arxiv.org/abs/2201.09759 


## Python files description
script_prepareDataset.py
- loads raw CHB-MIT files in .edf format and transforms them to prepared dataset 
- it can be done using differnt factor (in this case 10) that defines how much more non-seizure data we want to keep for each seizure episode
- outputs are .csv files where in each file is one seizure episode and 'factor' times more non seizure data 

script_MultiClassItterativeOnline_forPaper.py 
- main script that first calculates features for all files 
- then performs personalized training with leave-one-out cross-validation of each subject using all mentione approaches one afer another 
- plots and compares performances

HdfunctionsLib.py
- library with different functions on HD vectors, uses torch library

VariousFunctionsLib.py 
- library including various functions for HD project but not necessarily related to HD vectors

PerformanceMetricsLib.py
- Library with functions to measure performance on episode and duration level (for epilepsy application)

paramtersSetup.py
- script where all important parameters are defined and grouped into several cathegories 