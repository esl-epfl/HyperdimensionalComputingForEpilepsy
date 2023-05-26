''' library with functions related to reading Repomse dataset '''


import re
from scipy.io import loadmat
from VariousFunctionsLib import *
from featuresLib import *

def createFolderIfNotExists(folderOut):
    ''' creates folder if doesnt already exist
    warns if creation failed '''
    if not os.path.exists(folderOut):
        try:
            os.mkdir(folderOut)
        except OSError:
            print("Creation of the directory %s failed" % folderOut)

def check_read_access(fileName):
    '''
    Check reading access. Happens that after a while, the momentarely does not have access to the database.
    We try to access the file every 6 seconds for 5.06 min.
    Return True if permission to open the file is OK for reading access before Time out, False if time out.
    '''
    counter = 0
    #Try to access every 6 seconds for 5.06 minute
    while((not os.access(fileName, os.R_OK)) and (counter < 50)):
        time_1 = time.time()
        while(time.time() - time_1 < 6):
            ...
        counter+=1
    return counter<50


def get_fs(matlab_myTRC_field):
    '''    Get Sampling Frequency from a matlab file at the 'myTRC' field.
    '''
    return matlab_myTRC_field["header"][0][0]['samplingRate'][0][0][0][0]

def checkLabels(labList, thisLab):
    for lab in labList:
        if (lab in thisLab):
            return 1
    return 0

def check_file(mat_file, desired_fss_arr, seizLabelNames):
    '''
    Check a matlab file converted from TRC and weather it contains desired Fs, is well formated and contains a seizure index in m_notes
    Parameters:
        desired_fss_arr:  an array of all desired fs
        seizLabelNames: seizure label names that are acceptable
    Returns:
         the seizure index if seizure exists
         -2 if seizure label is not recognized
         -1 if no seizure or not good frequency
         0 if my_TRC varibale doesnt exist in .mat file
    '''
    if 'my_TRC' in mat_file.keys():
        data = mat_file["my_TRC"]
        Fs = get_fs(data)
        if (Fs in desired_fss_arr):
            if ("m_notes" in data.dtype.names):
                ### Get seizure index
                m_notes = data['m_notes'][0][0][0]
                seiz_idx = -1
                for note in m_notes:
                    if ((note[0][0].lower()) in seizLabelNames):   # Check if notes contain the crisis label
                        seiz_idx = note[1][0][0]
                        break
                    elif checkLabels(seizLabelNames, note[0][0].lower()):
                        seiz_idx = note[1][0][0]
                        break
                if (seiz_idx > -1):
                    return seiz_idx
                elif seiz_idx==-1:
                    print('--> No Seiz label recognized')
                    print( m_notes)
                    return -2
            else:
                print('No seiz notes in file')
        else:
            print('Freq is not good - ' + str(Fs))
            return -1
    else:
        print('No my_TCR in mat')
    return 0


def get_list_chns(data, desired_channels, desired_channels_mcn):
    '''
    Get list of channels indices from a matlab file at the 'myTRC' field corresponding to a desired list of channels.
    Returns the list of channels and their indicies in the matlab file
    '''

    chns = [i[0].upper() for i in data['m_electrodes'][0][0]['positivInputLabel'][0]]  # Channels in Matlab file
    chn_sr = pd.Series(chns)

    if (set(desired_channels).issubset(set(chns))):  # If all channels are in the matlab file, get them in a DF
        desired_sr = pd.Series(desired_channels)  # TODO change to list

    elif (set(desired_channels_mcn).issubset(set(chns))):
        desired_sr = pd.Series(desired_channels_mcn)

    else:
        return None, None

    values = list(chn_sr[chn_sr.isin(desired_sr)].values)
    indices = list(chn_sr[chn_sr.isin(desired_sr)].index)

    return values, indices


def get_records_in_df(data, indx, chns, channel_pairs):
    '''
    Store desired channel pair combinations in a df from a numpy array, the indicies of the channels TODO
    '''
    success=1
    sigs = data['m_eegAllChannels'][0][0]
    sigs_df = pd.DataFrame(np.take(sigs, indx, axis=0).T, columns=chns)
    for comb in channel_pairs:
        chn_1 = re.split('\-', comb)[0]
        chn_2 = re.split('\-', comb)[1]
        try:
            if (len(sigs_df[chn_2].shape)>1 or len(sigs_df[chn_1].shape)>1):
                success=0
            else:
                sigs_df[comb] = -(sigs_df[chn_1] - sigs_df[chn_2])
        except:
            #success=0
            print('sth wrong')
    if (success==0):
        sigs_return = pd.DataFrame(np.take(sigs, indx, axis=0).T, columns=chns)
    else:
        sigs_return=sigs_df[channel_pairs]
    return (success, sigs_return)


def get_signals(signals_df, seiz_idx, Fs, sl):
    '''
    From the signals_df, get the signals in numpy array according to defined arrays.
    seiz_idx: Index of seizure onset (SO)
    Fs: Sampling Frequency
    sl : segment length in minutes
    '''
    indices_pre = [int(Fs * 2 * sl * 60),int(Fs * sl * 60)]  # 2*sl min. -> 1*sl min before SO #TODO use iloc. df.iloc[ ,:]
    indices_ict = [int(0), int(Fs * sl * 60)]  # SO -> 1*sl min. #TODO :check -1
    # TODO: other downsampling methods
    try:
        signals_df = signals_df.reset_index(drop=True)
    except:
        print('sth wrong')

    if (Fs == 256):
        pre_ictal = signals_df.iloc[seiz_idx - indices_pre[0]:seiz_idx - indices_pre[1]].to_numpy()  # Pre ictal
        between = signals_df.iloc[seiz_idx - indices_pre[1]:seiz_idx - indices_ict[0]].to_numpy()
        ictal = signals_df.iloc[seiz_idx - indices_ict[0]:seiz_idx + indices_ict[1]].to_numpy()  # Ictal
    if (Fs == 512):
        pre_ictal = signals_df.iloc[seiz_idx - indices_pre[0]:seiz_idx - indices_pre[1]:2].to_numpy()  # Pre ictal
        between = signals_df.iloc[seiz_idx - indices_pre[1]:seiz_idx - indices_ict[0]:2].to_numpy()
        ictal = signals_df.iloc[seiz_idx - indices_ict[0]:seiz_idx + indices_ict[1]:2].to_numpy()  # Ictal
    if (Fs == 1024):
        pre_ictal = signals_df.iloc[seiz_idx - indices_pre[0]:seiz_idx - indices_pre[1]:4].to_numpy()  # Pre ictal
        between = signals_df.iloc[seiz_idx - indices_pre[1]:seiz_idx - indices_ict[0]:4].to_numpy()
        ictal = signals_df.iloc[seiz_idx - indices_ict[0]:seiz_idx + indices_ict[1]:4].to_numpy()  # Ictal
    #
    # # Ensure this is the right length, Fs is now 256Hz for all signals
    # pre_ictal_ = pre_ictal[:int(256 * sl * 60)]
    # between_ = between[:int(256 * sl * 60)]
    # ictal_ = ictal[:int(256 * sl * 60)]

    if (len(ictal)!=len(pre_ictal)):
        print('Lengths not the same!', len(ictal),len(pre_ictal) )
    return pre_ictal, ictal, between


def prepareDataset_REPOMSE(folderIn, folderOutBase, DatasetPreprocessParams, FeaturesParams, saveRawDataPreprocessed, extractFeatures):
    '''
    Function that goes through all Repomse data and
    - checks if files have correct frequency, labeled seizures, enugh of them etc
    - resaves them in gzip format with nonSeiz - Seiz - nonSeiz format (and adds labels)
    - calculates features from those data and saves it too
    '''
    # Create folders to save data
    folderOut=folderOutBase +'/01_datasetProcessed/'
    createFolderIfNotExists(folderOut)
    folderOutFeat = folderOutBase + '/02_Features/'
    createFolderIfNotExists(folderOutFeat)

    # Go through folders of all patients one by one
    goodPatIndx=1
    patFolders=sorted(glob.glob(f'{folderIn}/Pat*'))
    numSeizPerPat=np.zeros((len(patFolders),2))
    oldNames, newNames=[], []
    commentsPerSubj=np.zeros((len(patFolders),7)) #meaning of columns: not enough seiz, faild opening files, wrong freq, problem with channels, not enough seiz len,seiz label not in a list,  nt enough good seiz
    for foldIndx, fold in enumerate(patFolders):
        print(os.path.split(fold)[1])
        patStr = os.path.split(fold)[1][3:]
        patFiles = sorted(glob.glob(f'{fold}/*.mat'))
        numSeizPerPat[foldIndx,:]=[int(patStr), len(patFiles)] #count number of features per subject
        JSdivergence=np.zeros(( len( DatasetPreprocessParams.channel_pairs), 25, len(patFiles)))
        KLdivergence = np.zeros((len(DatasetPreprocessParams.channel_pairs), 25, len(patFiles)))
        goodSeizIndx=0
        listGoodFiles=[]
        # Check if enough seizures
        if len(patFiles)>=DatasetPreprocessParams.minNumSeiz:
            for fileIndx, file in enumerate(patFiles):
                # print(os.path.split(file)[1])
                # Check possible access permission denied
                if (not check_read_access(file)):
                    print("FAILED OPENING FILE %s from PATIENT %s " % (file, patStr))
                    commentsPerSubj[foldIndx,1]=commentsPerSubj[foldIndx,1]+1 #file opening issues
                else:
                    # Load matlab file
                    try:
                        mat_file = loadmat(file)
                    except:
                        print("FAILED OPENING MAT FILE %s from PATIENT %s " % (file, patStr))
                        commentsPerSubj[foldIndx, 1] = commentsPerSubj[foldIndx, 1] + 1  # file opening issues
                        break
                    #checks if seizure is in file and returns different flags if it is not or there is problem with the file
                    seiz_idx = check_file(mat_file, DatasetPreprocessParams.desired_Fs, DatasetPreprocessParams.seizLabelNames)  # Sample index
                    if (seiz_idx==-1):#problem with frequency
                        commentsPerSubj[foldIndx, 2] = commentsPerSubj[foldIndx,2]+1
                    elif (seiz_idx==-2):# seizure label not in a list
                        commentsPerSubj[foldIndx, 5] = commentsPerSubj[foldIndx, 5] + 1
                    elif (seiz_idx==0):  # file opening issues
                        commentsPerSubj[foldIndx, 1] = commentsPerSubj[foldIndx, 1] + 1

                    elif (seiz_idx>0):  # if the file contains a properly labeled seizure
                        # Get desired channels and corresponding indices
                        data = mat_file["my_TRC"]
                        chns, indx = get_list_chns(data, DatasetPreprocessParams.channels_list, DatasetPreprocessParams.channels_list_mcn)
                        if (chns != None):
                            # Collect EEG records in dataframe
                            (success,signals_df) = get_records_in_df(data, indx, chns, DatasetPreprocessParams.channel_pairs)
                            if success==1:
                                # Cut in portions of length sl minutes w.r.t crisis onset
                                Fs = get_fs(data)  # Sampling frequency
                                try:
                                    pre_ictal, ictal, between = get_signals(signals_df, seiz_idx, Fs, DatasetPreprocessParams.seizLen)
                                except:
                                    print('sth wrong')
                                    commentsPerSubj[foldIndx, 4] = 3  # problem with S/NS

                                if (len(pre_ictal)==len(ictal)): #at least 0.5 min of non seiz
                                    listGoodFiles.append(file)
                                    commentsPerSubj[foldIndx, 4] = 0  # problem with S/NS len
                                elif (len(pre_ictal)>=len(ictal)/2): #at least 0.5 min of non seiz
                                    listGoodFiles.append(file)
                                    commentsPerSubj[foldIndx, 4] = 1  # problem with S/NS len
                                elif (len(pre_ictal) < len(ictal)/2 ):  # at least 0.5 min of non seiz
                                    commentsPerSubj[foldIndx, 4] = 2  # problem with S/NS len
                                else:
                                    commentsPerSubj[foldIndx, 4] = 4  # problem with S/NS len
                            else:
                                commentsPerSubj[foldIndx,1]=commentsPerSubj[foldIndx,1]+1 #file opening issues
                        else:
                            commentsPerSubj[foldIndx, 3] =commentsPerSubj[foldIndx,3]+ 1 #problem with channels
        else:
            commentsPerSubj[foldIndx,0]=commentsPerSubj[foldIndx,0]+1 #not enough seizurs
            print(' Not enough seizures- only '+ str(len(patFiles)))

        # Enough good files with seizures kept
        if len(listGoodFiles)>=DatasetPreprocessParams.minNumSeiz:
            # Prepare output folders for this subject
            if (saveRawDataPreprocessed==1):
                folderOutPat= folderOut + '/Pat' + f"{int(patStr):03}"
                createFolderIfNotExists(folderOutPat)
            if (extractFeatures==1):
                folderOutFeatPat = folderOutFeat + '/Pat' + f"{int(patStr):03}"
                createFolderIfNotExists(folderOutFeatPat)
            # Go through one by one file and do:
            #  - last checks
            #  - rearanging and saving data, ploting raw data
            #  - calcualting features and KL divergence
            for fileIndx, file in enumerate(listGoodFiles):
                fileStr = os.path.split(file)[1]
                # Load matlab file
                mat_file = loadmat(file)
                seiz_idx = check_file(mat_file, DatasetPreprocessParams.desired_Fs, DatasetPreprocessParams.seizLabelNames)  # Sample index
                # Get desired channels and corresponding indices
                data = mat_file["my_TRC"]
                chns, indx = get_list_chns(data, DatasetPreprocessParams.channels_list, DatasetPreprocessParams.channels_list_mcn)
                # Collect EEG records in dataframe
                (success, signals_df) = get_records_in_df(data, indx, chns, DatasetPreprocessParams.channel_pairs)
                # Cut in portions of length sl minutes w.r.t crisis onset
                Fs = get_fs(data)  # Sampling frequency
                pre_ictal, ictal, between = get_signals(signals_df, seiz_idx, Fs, DatasetPreprocessParams.seizLen)

                # Rearange so that seizure is between pre_ictal
                (seizureLen, numCh) = ictal.shape
                nonSeizLen=len(pre_ictal[:,0])
                # Create new labels
                newLabel = np.zeros(seizureLen + nonSeizLen)  # both for seizure nad nonSeizure labels
                newLabel[int(nonSeizLen/2):int(nonSeizLen/2)+seizureLen] = np.ones(seizureLen)
                # Rearange new data
                newData = np.zeros((seizureLen + nonSeizLen, numCh))
                newData[int(nonSeizLen/2):int(nonSeizLen/2)+seizureLen,:] =  ictal
                newData[0:int(nonSeizLen/2),:] =pre_ictal[0:int(nonSeizLen/2),:] #add before seizure
                newData[int(nonSeizLen/2)+seizureLen:,:] = pre_ictal[int(nonSeizLen / 2):,: ]#add after seizure

                # Saving to csv file
                if (saveRawDataPreprocessed == 1):
                    fileNameOut=folderOutPat +'/Pat'+f"{int(patStr):03}"+'_' + f"{fileIndx:02}"+'_s'
                    saveDataToFile(np.hstack(( newLabel.reshape((-1,1)),newData )), fileNameOut, 'gzip')

                    # Plot raw data
                    plot_rawEEGdata(newData, newLabel, DatasetPreprocessParams.samplFreq, FeaturesParams.winLen, FeaturesParams.winStep, DatasetPreprocessParams.channel_pairs, folderOutPat,  'Pat'+f"{int(patStr):03}"+'_' + f"{fileIndx:02}"+'_s')

                ## CALCULATE FETURES
                if (extractFeatures == 1):
                    index = np.arange(0, len(newData) - int(FeaturesParams.winLen * DatasetPreprocessParams.samplFreq), int(FeaturesParams.winStep * DatasetPreprocessParams.samplFreq)).astype(int)
                    labelsPart = pd.DataFrame(newLabel[index], columns=['Labels'])
                    (features, chNamesAll, featNames) = calculateAllFeatures(newData, DatasetPreprocessParams.samplFreq, FeaturesParams.winLen, FeaturesParams.winStep,  DatasetPreprocessParams.channel_pairs)
                    featuresDf = pd.DataFrame(features, columns=chNamesAll)
                    dataToSave = pd.concat([labelsPart, featuresDf], axis=1)


                    # Save data as a parquet file
                    parquet_filename = folderOutFeatPat+ '/Pat' + f"{int(patStr):03}" + '_' + f"{fileIndx:02}" + '_s'
                    dataToSave = dataToSave.reset_index(drop=True)
                    dataToSave.to_parquet(parquet_filename, engine='fastparquet')

                    #calculate KL divergence
                    (KLdivergence[:,:,fileIndx],JSdivergence[:,:,fileIndx])=calculate_KLdivergence(newLabel[index], features, DatasetPreprocessParams.channel_pairs, featNames, folderOutFeatPat + '/Pat'+f"{int(patStr):03}"+'_' + f"{fileIndx:02}"+'_s')

                oldNames.append('Pat'+patStr+'/'+fileStr+'.csv')
                newNames.append('/Pat'+f"{int(patStr):03}"+ '_' + f"{fileIndx:02}" + '_s')
                #mapTable=np.vstack((mapTable, newEntry))

            goodPatIndx = goodPatIndx + 1

            # Plot KL divergence for this subject
            if (extractFeatures == 1):
                outName = folderOutFeat + '/Pat' + f"{int(patStr):03}"
                plotKLDivergence(np.nanmean(KLdivergence, 2), np.nanmean(JSdivergence, 2), featNames, outName)

            fileNameOut = folderOut + '/00_mappingTable'
            saveDataToFile(np.vstack((newNames,oldNames)).transpose(), fileNameOut, '.csv')
        else:
            commentsPerSubj[foldIndx,6]=commentsPerSubj[foldIndx,6]+1 #not enough good seizures keept
            print ('Not enough good seizures kept - only '+ str(len(listGoodFiles)))

        #save statsitstics per subj, how many seizures and why not kept
        fileNameOut = folderOut + '/00_numSeizPerSubj'
        saveDataToFile(np.hstack((numSeizPerPat, commentsPerSubj)), fileNameOut, 'csv')
        colNames=['Subj', 'numSeiz', 'not enough seiz', 'faild opening files', 'wrong freq', 'problem with ch', 'not enough seiz len', 'seiz label not in a list',  'not enough good seiz']
        df = pd.DataFrame(data=np.hstack((numSeizPerPat, commentsPerSubj)).astype(int), columns=colNames)
        df.to_csv(fileNameOut+'_DF.csv')

    # Plot number of original seizures per subject as histogram
    fig1 = plt.figure(figsize=(10, 4), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.2, hspace=0.2)
    numBins=int(np.max(numSeizPerPat[:,1]))
    n, bins, patches=plt.hist(numSeizPerPat[:,1], numBins, facecolor='blue', alpha=0.5)
    plt.show()
    plt.savefig(folderOut + '/NumSeizHistogram.png', bbox_inches='tight')
    plt.savefig(folderOut + '/NumSeizHistogram.svg', bbox_inches='tight')
    plt.close(fig1)
    print('numGoodPat: ', goodPatIndx)
    print('avrgNumSeizPerPers:', len(oldNames)/goodPatIndx)

    #Save one last time all information per subject
    fileNameOut = folderOut + '/00_numSeizPerSubj'
    saveDataToFile(np.hstack((numSeizPerPat, commentsPerSubj)), fileNameOut, 'csv')
    colNames = ['Subj', 'numSeiz', 'not enough seiz', 'faild opening files', 'wrong freq', 'problem with ch',
                'not enough seiz len', 'seiz label not in a list', 'not enough good seiz']
    df = pd.DataFrame(data=np.hstack((numSeizPerPat, commentsPerSubj)).astype(int), columns=colNames)
    df.to_csv(fileNameOut + '_DF.csv')


