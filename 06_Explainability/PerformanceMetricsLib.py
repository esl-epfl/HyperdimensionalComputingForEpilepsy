''' library with different functions for assesing predicion performance in the use case of epilepsy'''

__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

import numpy as np


def performance_duration(y_pred_smoothed, y_true):
    '''calculates performance metricses on the level of seizure duration '''
    #total true seizure durations
    durationTrueSeizure=np.sum(y_true)

    #total predicted seizure duration
    durationPredictedSeizure=np.sum(y_pred_smoothed)

    #total duration of true predicted seizure
    temp=2*y_true-y_pred_smoothed #where diff is 1 here both true and apredicted label are 1
    indx=np.where(temp==1)
    durationTruePredictedSeizure=np.squeeze(np.asarray(indx)).size


    if (durationPredictedSeizure==0):
        precision=0 #np.nan #if no seizures predicted (if there was really no seiz ti will be np.nan later)
        #print('No predicted seizure in test data')
    else:
        precision=durationTruePredictedSeizure/durationPredictedSeizure
    if (durationTrueSeizure==0):
        sensitivity=np.nan
        precision = np.nan
        #print('No seizure in test data')
    else:
        sensitivity=durationTruePredictedSeizure/durationTrueSeizure
    if ((sensitivity + precision)==0):
        F1score_duration=0 #np.nan
        #print('Sensitivity and prediction are 0 in test data')
    else:
        F1score_duration = 2 * sensitivity * precision / (sensitivity + precision)

    return(sensitivity, precision , F1score_duration)


def calculateStartsAndStops(labels):
    ''' funtion that detects starts and stop of seizure episodes (or groups of labels 1) '''
    sigLen = len(labels)
    events=[]
    for i in range(1, sigLen-1):
        # if  label seizure starts
        if (labels[i] == 1 and labels[i - 1] == 0) | (i == 1 and labels[i-1] == 1 and labels[i] == 1):
            sstart=i
            while labels[i]==1 and i<sigLen-1:
                i=i+1
            sstop=i
            events.append([sstart, sstop])

    return(events)


def calc_TPAndFP(ref, hyp, toleranceFP_bef, toleranceFP_aft):
    ''' for pair of ref and hyp event decides for fp and tp'''
    ## collect start and stop times from input arg events
    start_ref = ref[0]
    stop_ref = ref[1]
    start_hyp = hyp[0]
    stop_hyp = hyp[1]

    ##### detect if hyp and ref have some overlap
    tp = 0
    #     ref:              <--------------------->
    #     hyp:     <----------------
    if (start_hyp <= start_ref  and stop_hyp > start_ref  ): #+ tolerance
        tp = 1
    #     ref:              <--------------------->
    #     hyp:                         <----------------
    elif (start_hyp < stop_ref  and  stop_hyp >= stop_ref ): #- tolerance
        tp = 1
    #     ref:              <--------------------->
    #     hyp:                         <---->
    elif (stop_hyp <= stop_ref  and  start_hyp >= start_ref  ): #- tolerance
        tp = 1

    #### detect fp
    fp = 0
    fp_bef=0
    fp_aft = 0
    #     ref:         |     <--------------------->     |
    #     hyp:     <----------------
    if (start_hyp < start_ref - toleranceFP_bef):
        fp = fp + 1
        fp_bef = 1
    #     ref:         |     <--------------------->     |
    #     hyp:    						  ------------------>
    if (stop_hyp > stop_ref + toleranceFP_aft):
        fp = fp + 1
        fp_aft = 1
    return (tp, fp, fp_bef, fp_aft )



def performance_episodes(predLab, trueLab,  toleranceFP_bef, toleranceFP_aft):

    totalTP=0
    totalFP=0
    #transform to events
    predEvents= calculateStartsAndStops(predLab)
    trueEvents = calculateStartsAndStops(trueLab)
    #create flags for each event if it has been used
    flag_predEvents=np.zeros(len(predEvents))
    flag_trueEvents = np.zeros(len(trueEvents))
    flag_trueEventsFPAround = np.zeros(len(trueEvents))
    # goes through ref events
    if (len(trueEvents)==0):
        totalFP=len(predEvents)
    else:
        for etIndx, eTrue  in enumerate(trueEvents):
            for epIndx, ePred in enumerate(predEvents):
                (tp0, fp0, fp_bef, fp_aft)= calc_TPAndFP(eTrue, ePred,  toleranceFP_bef, toleranceFP_aft)

                #if overlap detected (tp=1) and this refSeiz hasnt been used
                #     ref:           <----->        <----->              <----->             <-------------->
                #     hyp:     <---------->          <---------->     <-------------->           <----->
                if (tp0==1 and flag_trueEvents[etIndx]==0 and flag_predEvents[epIndx]==0):
                    totalTP=totalTP+tp0
                    totalFP = totalFP + fp0
                    flag_trueEvents[etIndx] = 1
                    flag_predEvents[epIndx] = 1 #1 means match
                    if (fp0==2):
                        flag_trueEventsFPAround[etIndx]=2
                    else:
                        flag_trueEventsFPAround[etIndx] = fp_aft-fp_bef #1 if was after, or -1 if was before
                #if ref event was already matched and now we have some extra predicted seiz ones
                #     ref:           <------------------------------------------>
                #     hyp:     <---------->     <-------------->     <----->
                elif (tp0 == 1 and flag_trueEvents[etIndx] == 1 and flag_predEvents[epIndx] == 0):
                    #totalTP = totalTP + tp0 #we already counted that one
                    totalFP = totalFP + fp0 #ideally fp0 should be 0, but if at the end we might have 1 fp
                    #flag_trueEvents[etIndx] = 1 it is already 1 so not needed again
                    flag_predEvents[epIndx] = 2 #2 means overlaping but not the first match with siezure
                #if one big pred seizure covering more ref
                #     ref:         <---------->     <-------------->     <----->
                #     hyp:              <------------------------------------------>
                elif (tp0==1 and flag_trueEvents[etIndx]==0 and flag_predEvents[epIndx]==1):
                    #totalTP=totalTP+tp0 # HERE WE NEED TO DECIDE TO WE COUND THEM AS TP OR NOT  !!!!
                    totalFP = totalFP + fp0 #we treat this as 1 FP
                    if (flag_trueEventsFPAround[etIndx-1]>0 and fp_bef==1): # if there was FP after true seiyure and already counted and now we want to count it again because of before
                        totalFP=totalFP-1
                    flag_trueEvents[etIndx] = 0 #it has to stay unmatched
                    #flag_predEvents[epIndx] = 1 #already matched
                #if pred seiyure was named FP in pass with previous seizure but now matches this seizure
                elif (tp0==1 and flag_trueEvents[etIndx]==0 and flag_predEvents[epIndx]==-1):
                    totalTP=totalTP+tp0
                    totalFP = totalFP -1 + fp0 #remove fp from before
                    flag_trueEvents[etIndx] = 1 #match seizure
                    if (fp0==2):
                        flag_trueEventsFPAround[etIndx]=2
                    else:
                        flag_trueEventsFPAround[etIndx] = fp_aft-fp_bef #1 if was after, or -1 if was before
                    flag_predEvents[epIndx] = 1 #relabel this pred seizure
                #if pred seizure was named FP in pass with previous seizure , now overlaps with seizure but this seizure was already matched
                elif (tp0 == 1 and flag_trueEvents[etIndx] == 1 and flag_predEvents[epIndx] == -1):
                    #totalTP = totalTP + tp0 #we already counted that one
                    totalFP = totalFP -1 + fp0 #ideally fp0 should be 0, but if at the end we might have 1 fp, remove Fp from before
                    #flag_trueEvents[etIndx] = 1 it is already 1 so not needed again
                    flag_predEvents[epIndx] = 2 #2 means overlaping but not the first match with siezure
                #prdiction but not matched with true seizure
                elif (tp0==0 and flag_predEvents[epIndx]==0):
                    totalFP = totalFP + 1  #+fp0
                    flag_predEvents[epIndx] = -1 #-1 means used as FP
                elif (flag_predEvents[epIndx]==2): #already counted as part of previous seiyure, we dont need to count it again
                    a=0
                elif (flag_predEvents[epIndx]==-1): #already counted as FP, we dont need to count it again
                    a=0
                elif (flag_trueEvents[etIndx]==1 and flag_predEvents[epIndx]==1): #both already matched
                    a=0
                elif ( flag_trueEvents[etIndx]==0 and flag_predEvents[epIndx]==1): #pred seiz was matched, true hasnt found yet
                    a=0
                else:
                    #flag_predEvents[epIndx] = 1
                    print('ERROR: new case i havent covered')

    #calculating final performance
    numTrueSeiz=len(trueEvents)
    numPredSeiz = len(predEvents)

    numMissedSeiz = numTrueSeiz- np.sum(flag_trueEvents)

    #precision =TP/ numPredSeiz but if all is one big predicted seiz then thigs are wrong and value would be >1
    if((totalTP +totalFP) != 0):
        precision=totalTP/ (totalTP +totalFP)
    else:
        precision=0 #np.nan #if no seizures predicted (if there was really no seiz ti will be np.nan later)

    #sensitivity= TP/ numTrueSeiy
    if ((numTrueSeiz) != 0):
        sensitivity= totalTP/numTrueSeiz
    else:
        sensitivity=np.nan #IF NOT TRUE SEIZRUES SENSITIVITY=NAN
        precision = np.nan

    if ((sensitivity + precision)!=0):
        F1score= (2* sensitivity * precision)/ (sensitivity + precision)
    else:
        F1score= 0 #np.nan


    #checkups
    # if ( (totalTP +totalFP)!= numPredSeiz):
    #     print('there are long pred seiz')
    if ( (numMissedSeiz +totalTP)!= numTrueSeiz):
        print('sth wrong with counting seizures')
    if ( totalFP < len(np.where(flag_predEvents==-1)[0])):
        print('sth wrong with counting FP')
    if (totalTP != len(np.where(flag_predEvents==1)[0])):
        print('sth wrong with counting seizures 2')

    # #visualize
    # xvalues = np.arange(0, len(trueLab), 1)
    # fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    # gs = GridSpec(1, 1, figure=fig1)
    # fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    # fig1.suptitle('True and pred labels')
    # ax1 = fig1.add_subplot(gs[0,0])
    # ax1.plot(xvalues, trueLab,  'r')
    # ax1.plot(xvalues, predLab*0.9, 'k')
    # ax1.set_xlabel('Time')
    # ax1.legend(['True', 'Pred'])
    # #calcualte performance for duration to put in title
    # (sensitivity_duration, precision_duration, F1score_duration) = perfomance_duration(predLab, trueLab)
    # ax1.set_title('EPISODES: sensitivity '+ str(sensitivity)+' precision ' + str(precision)+' F1score '+ str(F1score)+' totalTP ' + str(totalTP)+ ' totalFP '+ str(totalFP) + '\n' +
    #               'DURATION: sensitivity ' + str(sensitivity_duration) + ' precision ' + str(precision_duration) + ' F1score ' + str(F1score_duration))
    # ax1.grid()
    # fig1.show()
    # fig1.savefig(folderOut + '/' +figName)
    # plt.close(fig1)

    return(sensitivity, precision, F1score, totalFP)


def performance_all9(predLab, trueLab,  toleranceFP_bef, toleranceFP_aft, numLabelsPerHour):
    ''' fucntion that returns 9 different performance measures of prediction on epilepsy
    - on the level of seizure episodes (sensitivity, precision and F1 score)
    - on the level of seizure duration, or each sample (sens, prec, F1)
    - combination of F1 scores for episodes and duration (either mean or gmean)
    - number of false positives per day '''
    (sensE, precisE, F1E, totalFP)= performance_episodes(predLab, trueLab, toleranceFP_bef, toleranceFP_aft)
    (sensD, precisD , F1D)=performance_duration(predLab, trueLab)

    #calculate combinations
    F1DEmean=(F1D+F1E)/2
    F1DEgeoMean=np.sqrt(F1D*F1E)


    #calculate numFP per day
    #numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    timeDurOfLabels = len(trueLab) / numLabelsPerHour
    if (timeDurOfLabels != 0):
        numFPperHour = totalFP / timeDurOfLabels
    else:
        numFPperHour = np.nan
    numFPperDay=numFPperHour*24

    if (sensE>1.0 or precisE>1.0 or F1E>1.0 or sensD>1.0 or precisD>1.0 or F1D>1.0 or F1DEmean>1.0 or F1DEgeoMean>1.0 ):
        print('ERROR - perf measures impossibly big!')
    # if (np.sum(trueLab)==0):
    #     print('No Seiz in file')

    return( sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay)


def smoothenLabels(prediction,  seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx):
    ''' returns labels after two steps of postprocessing
    first moving window with voting  - if more then threshold of labels are 1 final label is 1 otherwise 0
    second merging seizures that are too close '''

    #labels = labels.reshape(len(labels))
    smoothLabelsStep1=np.zeros((len(prediction)))
    smoothLabelsStep2=np.zeros((len(prediction)))
    try:
        a=int(seizureStableLenToTestIndx)
    except:
        print('error seizureStableLenToTestIndx')
        print(seizureStableLenToTestIndx)
    try:
        a=int(len(prediction))
    except:
        print('error prediction')
        print(prediction)
    #first classifying as true 1 if at laest  GeneralParams.seizureStableLenToTest in a row is 1
    for i in range(int(seizureStableLenToTestIndx), int(len(prediction))):
        s= sum( prediction[i-seizureStableLenToTestIndx+1: i+1] )/seizureStableLenToTestIndx
        try:
            if (s>= seizureStablePercToTest):  #and prediction[i]==1
                smoothLabelsStep1[i]=1
        except:
            print('error')
    smoothLabelsStep2=np.copy(smoothLabelsStep1)

    #second part
    prevSeizureEnd=-distanceBetweenSeizuresIndx
    for i in range(1,len(prediction)):
        if (smoothLabelsStep2[i] == 1 and smoothLabelsStep2[i-1] == 0):  # new seizure started
            # find end of the seizure
            j = i
            while (smoothLabelsStep2[j] == 1 and j< len(smoothLabelsStep2)-1):
                j = j + 1
            #if current seizure distance from prev seizure is too close merge them
            if ((i - prevSeizureEnd) < distanceBetweenSeizuresIndx):  # if  seizure started but is too close to previous one
                #delete secon seizure
                #prevSeizureEnd = j
                #[i:prevSeizureEnd]=np.zeros((prevSeizureEnd-i-1)) #delete second seizure - this was before
                #concatenate seizures
                if (prevSeizureEnd<0): #if exactly first seizure
                    prevSeizureEnd=0
                smoothLabelsStep2[prevSeizureEnd:j] = np.ones((j - prevSeizureEnd ))
            prevSeizureEnd = j
            i=prevSeizureEnd

    return  (smoothLabelsStep2, smoothLabelsStep1)

def smoothenLabels_Bayes(prediction,  probability, seizureStableLenToTestIndx,  probThresh):
    ''' returns labels bayes postprocessing
    calculates cummulative probability of seizure and non seizure over the window of size seizureStableLenToTestIndx
    if log (cong_pos /cong_ned )> probThresh then seizure '''

    #convert probability to probability of pos
    probability_pos=np.copy(probability)
    indxs=np.where(prediction==0)[0]
    probability_pos[indxs]=1-probability[indxs]

    #labels = labels.reshape(len(labels))
    smoothLabels=np.zeros((len(prediction)))
    try:
        a=int(seizureStableLenToTestIndx)
    except:
        print('error seizureStableLenToTestIndx')
        print(seizureStableLenToTestIndx)
    try:
        a=int(len(prediction))
    except:
        print('error prediction')
        print(prediction)

    #first classifying as true 1 if at laest  GeneralParams.seizureStableLenToTest in a row is 1
    for i in range(int(seizureStableLenToTestIndx), int(len(prediction))):
        probThisWind=probability_pos[i-seizureStableLenToTestIndx+1: i+1]
        conf_pos=np.prod(probThisWind)
        conf_neg=np.prod( 1-probThisWind)
        conf=np.log( (conf_pos+ 0.00000001) /(conf_neg + 0.00000001))
        if (conf>= probThresh):  #and prediction[i]==1
                smoothLabels[i]=1

    return  (smoothLabels)


def calculatePerformanceAfterVariousSmoothing(predLabels, label, probabilityLabels,toleranceFP_bef, toleranceFP_aft, numLabelsPerHour, seizureStableLenToTestIndx, seizureStablePercToTest,  distanceBetweenSeizuresIndx, probThresh):
    ''' function that calculates performnace for epilepsy
    it evaluates on raw predictions but also perfroms different smoothing and evaluated performance after smoothing
    - firs smoothng is just moving average with specific window size and percentage of lables that have to be 1 to give final label 1
    - then merging of too close seizure is performan in step2
    - another option for postprocessing and smoothing of labels is bayes postrpocessing
    '''
    numTypes=4 #no smooth, movignAvrg, MmovingAvrg+merging, bayes
    numPerf=9
    performancesAll=np.zeros((numPerf*numTypes))

    # calculate different performance measures - only no Smooth
    performancesAll[ : numPerf] = performance_all9(predLabels, label, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)

    #smoothing using moving average and then merging
    (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels, seizureStableLenToTestIndx, seizureStablePercToTest,  distanceBetweenSeizuresIndx)
    performancesAll[numPerf: 2*numPerf] = performance_all9(yPred_SmoothOurStep1, label, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    performancesAll[2*numPerf: 3*numPerf]= performance_all9(yPred_SmoothOurStep2, label, toleranceFP_bef, toleranceFP_aft,  numLabelsPerHour)

    #bayes smoothing
    yPred_SmoothBayes=smoothenLabels_Bayes(predLabels, probabilityLabels, seizureStableLenToTestIndx, probThresh)
    performancesAll[3 * numPerf: 4 * numPerf] = performance_all9(yPred_SmoothBayes, label, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)


    return(performancesAll,yPred_SmoothOurStep1, yPred_SmoothOurStep2, yPred_SmoothBayes )


def calculatePerformanceWithoutSmoothing(predLabels, yPred_SmoothOurStep1, yPred_SmoothOurStep2, yPred_SmoothBayes, label, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour):
    ''' function that calculates performnace for epilepsy
    '''
    numTypes=4 #no smooth, movignAvrg, MmovingAvrg+merging, bayes
    numPerf=9
    performancesAll=np.zeros((numPerf*numTypes))

    # calculate different performance measures - only no Smooth
    performancesAll[ : numPerf] = performance_all9(predLabels, label, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    performancesAll[numPerf: 2*numPerf] = performance_all9(yPred_SmoothOurStep1, label, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    performancesAll[2*numPerf: 3*numPerf]= performance_all9(yPred_SmoothOurStep2, label, toleranceFP_bef, toleranceFP_aft,  numLabelsPerHour)
    performancesAll[3 * numPerf: 4 * numPerf] = performance_all9(yPred_SmoothBayes, label, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    return(performancesAll )