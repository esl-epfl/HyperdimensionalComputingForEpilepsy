## Code description

Code necassary for testing different feature approaches used with HD computing for epileptioc seizure detection. 
It compares approaches in three aspects: 
1) Prediction performance 
2) Memory needed to store all the HD vectors
3) Computational complexity (time needed for execution)

### Related publication 
Paper: „Systematic Assessment of Hyperdimensional Computing for Epileptic Seizure Detection“, Una Pale, Tomas Teijeiro, David Atienza

2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), https://ieeexplore.ieee.org/document/9629648

Arxiv version: https://arxiv.org/abs/2105.00934 


## Python files description

Script_runAllApproaches.py 
Script that based on parameters set runs training and testing of specific HD features model
- it is possible to choose between two different databases and which patients to use
- it can perform personalized or generalized training
    - personalized is for each subject performing leave one seizure out and training on the rest of seizures
    - generalized requires to define CViterations_testSubj variable - which determines CV iterations and which subjects are in test set (all others are in training)
- outputs prediction labels for each input file with saved original predictions, and predictions after two sets of labels postprocessing (smoothing) and also true labels
    - by analysing and plotting this labes many things can be analized
- plots performances in dependence of performance measures and amount of postprocessing

Script_compareApproaches.py
Script that:
- meausres number of seizure per subjects for databases
- compares different approaches in terms of performance and plots comparison 
- meausures time for calculation for different approaches
- plots time and memory requirements for different approaches

HdfunctionsLib.py
- library with different functions on HD vectors, uses torch library

VariousFunctionsLib.py 
- library including various functions for HD project but not necessarily related to HD vectors
