'''
library for oparating with HD models and vectors, both personalized and generalized
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import math
import pickle
import glob
import os
import re
import torch


def unpackVector(vecArr, D, device):
    import torch
    from hdtorch import pack, unpack
    cuda0 = torch.device(device)
    if vecArr.ndim == 1:
        vecArr = np.expand_dims(vecArr, 0)
    if vecArr.ndim == 2:
        vecArr = np.expand_dims(vecArr, 0) # <<-- If the input array doesn't have a batch axis, add it
    vecTor = torch.from_numpy(vecArr).to(cuda0, dtype = torch.int32)
    # unpackedOutput = torch.full((vecTor.shape[0], vecTor.shape[1], D),-1,dtype = torch.int8, device = device) # <<-- The output array should have the same first 2 axes, but the 3rd will be expanded
    unpackedOutput = unpack(vecTor, D)
    unpackedArr=unpackedOutput.cpu().numpy().squeeze()
    return (unpackedArr)


def packVector(vecArr, D, device):
    import torch
    import math
    from hdtorch import pack, unpack
    cuda0 = torch.device(device)
    vecTor = torch.from_numpy(vecArr.reshape((vecArr.shape[0],-1))).to(cuda0, dtype = torch.int8) # <<-- Should be int8
    # packedOutput = torch.full((vecTor.shape[0],math.ceil(D/32)),-1,dtype = torch.int32, device = device)
    packedOutput = pack(vecTor)
    packedArr=packedOutput.cpu().numpy().squeeze()
    checkArr = unpackVector(packedArr, D, 'cuda')
    assert((checkArr == vecArr).all()) # <<-- Make sure it works
    return (packedOutput)

def createVectorsDataStructure(path, HDParams, patientsAll):
    '''Read and store vectors in a convinient data structure '''
    print(f" [INFO] Model vectors are being read...")

    vectors = {}
    vectors_norm = {}
    numAdded_vec = {}

    for patient in patientsAll:
        fileslist = glob.glob(f'{path}/Subj{patient}_cv???_*_ModelVecs.csv.gz')
        fileslist.sort()
        if len(fileslist):
            print(f" [INFO] {len(fileslist)} files were found for patient {patient}.")
        else:
            print(f" [FAIL] No file found for patient {patient}. Expected vector files name: 'Subj##_cv###_[Method]_ModelVecs.csv.gz'")
            exit()

        for filepath in fileslist:

            # Retrieve file metadata
            filename = os.path.basename(filepath)
            method   = re.search('_([a-zA-Z]+)_ModelVecs.csv.gz', filename).group(1)
            if (method in ['ClassicHD', 'classicHD', 'StdHD']):
                method = 'StdHD'
            elif (method in ['OnlineHD', 'OnlHD', 'OnlineHDAddSub', 'OnlineHDAddOnly']):
                method = 'OnlHD'

            # Prepare main data structure
            if patient not in vectors:
                vectors[patient] = {}
            if method not in vectors[patient]:
                vectors[patient][method] = {}
                vectors[patient][method][0] = []
                vectors[patient][method][1] = []

            if patient not in vectors_norm:
                vectors_norm[patient] = {}
            if method not in vectors_norm[patient]:
                vectors_norm[patient][method] = {}
                vectors_norm[patient][method][0] = []
                vectors_norm[patient][method][1] = []

            if patient not in numAdded_vec:
                numAdded_vec[patient] = {}
            if method not in numAdded_vec[patient]:
                numAdded_vec[patient][method] = {}
                numAdded_vec[patient][method][0] = []
                numAdded_vec[patient][method][1] = []

            #load and save nonnormalized vectors
            data = np.loadtxt(filepath, delimiter = ',', comments = '#')
            vector_zero = np.squeeze(data[1:,0])
            vector_one  = np.squeeze(data[1:,1])

            #load and save normalized vectors
            try:
                data = np.loadtxt(filepath[:-7]+'Norm.csv.gz', delimiter=',', comments='#')
                if (len(data[1:,0])!=HDParams.D): #packd so we need to unpack
                    vector_zero_norm =unpackVector(data[1:,0], HDParams.HD_dim, HDParams.device) #np.squeeze(data[1:,0])
                    vector_one_norm  =unpackVector(data[1:,1], HDParams.HD_dim, HDParams.device)  # np.squeeze(data[1:,1])
                else:
                    vector_zero_norm =  np.squeeze(data[1:,0])
                    vector_one_norm = np.squeeze(data[1:,1])
            except:
                print(' ERR: Not reading correctly file: ', filepath[:-7]+'Norm.csv.gz')
            # vector_zero_norm = (vector_zero >= 0) * 2 - 1
            # vector_one_norm  = (vector_one  >= 0) * 2 - 1

            #load and save amout of added vectors per class
            try:
                data = np.loadtxt(filepath[:-16]+'AddedToEachSubClass.csv.gz', delimiter=',', comments='#')
                # numAdded_vec[patient][method][0]= data[1]
                # numAdded_vec[patient][method][1]= data[2]
                numAdded_vec[patient][method][0].append(data[1])
                numAdded_vec[patient][method][1].append(data[2])
            except:
                print(' ERR: Not reading correctly file: ', filepath[:-7]+'AddedToEachSubClass.csv.gz')

            vectors[patient][method][0].append(vector_zero)
            vectors[patient][method][1].append(vector_one)

            vectors_norm[patient][method][0].append(vector_zero_norm)
            vectors_norm[patient][method][1].append(vector_one_norm)

    return vectors, vectors_norm, numAdded_vec


def createVectorsDataStructure_PerSubj(path, HDParams, patientsAll, CVtype):
    print(f" [INFO] Model vectors are being read...")

    vectors = {}
    vectors_norm = {}
    numAdded_vec = {}
    for patient in patientsAll:
        for method in ['StdHD','OnlHD']:
            try:
                filepath=path+'/PerFileVectors_'+method+'_Subj'+patient+'.pickle'
                #load all from pickle file
                file = open(filepath, 'rb')
                objfile = pickle.load(file) #order of arrays: ModelVectorsAll_StdHD_S, ModelVectorsAllNorm_StdHD_S, ModelVectorsAll_StdHD_NS, ModelVectorsAllNorm_StdHD_NS, NumAdded_StdHD
            except:
                print ('FAIL - NO FILE: ', filepath)

            # Prepare main data structure
            if patient not in vectors:
                vectors[patient] = {}
                vectors_norm[patient] = {}
                numAdded_vec[patient] = {}
            if method not in vectors[patient]:
                vectors[patient][method] = {}
                vectors[patient][method][0] = []
                vectors[patient][method][1] = []
                vectors_norm[patient][method] = {}
                vectors_norm[patient][method][0] = []
                vectors_norm[patient][method][1] = []
                numAdded_vec[patient][method] = {}
                numAdded_vec[patient][method][0] = []
                numAdded_vec[patient][method][1] = []


            #save S vectors only from files where there was actually seizure
            if (method=='StdHD'):
                if (CVtype != 'LeaveOneOut'): #where increas in number of seiz added
                    harr=np.hstack(([0],objfile[4][:,1]))
                    indxS=np.where(np.diff(harr)!=0)[0]
                    # indxS=np.concatenate(([1],indxS+1))
                else:  #where more some S data added
                    harr=objfile[4][:,1]
                    indxS = np.where((harr) != 0)[0]
            # load num Added
            # data = np.loadtxt(filepath, delimiter=',', comments='#')
            numAdded_vec[patient][method][0] = objfile[4][:,0].reshape((-1,1)).tolist()
            numAdded_vec[patient][method][1] = objfile[4][indxS,1].reshape((-1,1)).tolist()

            #load vectors
            vectors[patient][method][0]= objfile[2].tolist()
            vectors[patient][method][1]= objfile[0][indxS,:].tolist() #squeeze().

            #load norm vectors
            dim=len(objfile[1][0,:])
            if dim==HDParams.D:
                vectors_norm[patient][method][0]= objfile[3].tolist()
                vectors_norm[patient][method][1]= objfile[1][indxS,:].tolist() #squeeze().
            else: #if packed
                vectors_norm[patient][method][0]= unpackVector(objfile[3], HDParams.D, HDParams.device).tolist()
                vectors_norm[patient][method][1]= unpackVector(objfile[1][indxS,:], HDParams.D, HDParams.device).tolist() #.squeeze()

    return vectors, vectors_norm, numAdded_vec

def createOnePersVecPePerson(vectors, vectors_norm, numAdded,  methodList, Params, HDParams):
    # Personnal vectors aggregation
    Genvectors = {}
    Genvectors_norm = {}
    GennumAdded_vec = {}

    for patient in tqdm(vectors_norm, desc='Personnal vectors aggregation', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}'):
        Genvectors[patient] , Genvectors_norm[patient] , GennumAdded_vec[patient] = {}, {}, {}
        for method in methodList:
            Genvectors[patient][method] = {}
            Genvectors[patient][method][0] , Genvectors[patient][method][1] =  [], []
            Genvectors_norm[patient][method] = {}
            Genvectors_norm[patient][method][0] , Genvectors_norm[patient][method][1] = [], []
            GennumAdded_vec[patient][method] = {}
            GennumAdded_vec[patient][method][0] , GennumAdded_vec[patient][method][1] =  [], []

            # patient_labels.append(patient)
            if Params.CVType == 'LeaveOneOut':  # vector repreesenting that subje is average of all crossvalidations
                if (HDParams.HDvecType=='bin'):
                    # mean of normalized vectors
                    Genvectors_norm[patient][method][0]=((np.sum(vectors_norm[patient][method][0],  axis=0) >= len(vectors_norm[patient][method][0])/2) *1)
                    Genvectors_norm[patient][method][1]=((np.sum(vectors_norm[patient][method][1], axis=0) >= len(vectors_norm[patient][method][1])/2) *1)
                else: #if bipol
                    Genvectors_norm[patient][method][0]=((np.sum(vectors_norm[patient][method][0],  axis=0) >= 0) * 2 - 1)
                    Genvectors_norm[patient][method][1]=((np.sum(vectors_norm[patient][method][1], axis=0) >= 0) * 2 - 1)
                Genvectors[patient][method][0] = np.sum(vectors_norm[patient][method][0],  axis=0)
                Genvectors[patient][method][1] = np.sum(vectors_norm[patient][method][1],  axis=0)
                GennumAdded_vec[patient][method][0] = len(vectors_norm[patient][method][0])
                GennumAdded_vec[patient][method][1] = len(vectors_norm[patient][method][1])
            else:  # if rollingbase then just take last CV vetors of each person
                Genvectors_norm[patient][method][0]=(vectors_norm[patient][method][0][-1])
                Genvectors_norm[patient][method][1] = (vectors_norm[patient][method][1][-1])
                Genvectors[patient][method][0]=(vectors[patient][method][0][-1])
                Genvectors[patient][method][1] = (vectors[patient][method][1][-1])
                GennumAdded_vec[patient][method][0]=(numAdded[patient][method][0][-1])
                GennumAdded_vec[patient][method][1] = (numAdded[patient][method][1][-1])
    return (Genvectors, Genvectors_norm, GennumAdded_vec)

def plotSimilarities(title, filepath, axis_labels, axis_ticks, similarities, zrange, cbar_horiz=True):

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xticks(range(len(axis_ticks[0])))
    ax.set_xticklabels(axis_ticks[0], fontsize=4)
    ax.set_yticks(range(len(axis_ticks[1])))
    ax.set_yticklabels(axis_ticks[1], fontsize=4)
    pcm = ax.imshow(similarities, cmap='bone',  vmin=zrange[0], vmax=zrange[1])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    fig.colorbar(pcm, ax=ax, orientation=("horizontal" if cbar_horiz else "vertical"))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close()


def plotSimilarities_SandNSIn1Subj(title, filepath, similarities, numNS, zrange, cbar_horiz=True):
    fig = plt.figure(figsize=(10,8), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig)
    fig.suptitle(title)
    ax = fig.add_subplot(gs[0, 0])
    pcm = ax.imshow(similarities[0:numNS, 0:numNS], cmap='bone', vmin=zrange[0], vmax=zrange[1])
    ax.set_title('NonSeizure')
    fig.colorbar(pcm, ax=ax, orientation=("horizontal" if cbar_horiz else "vertical"))
    ax = fig.add_subplot(gs[0, 1])
    pcm = ax.imshow(similarities[numNS:, numNS:], cmap='bone', vmin=zrange[0], vmax=zrange[1])
    ax.set_title('Seizure')
    fig.colorbar(pcm, ax=ax, orientation=("horizontal" if cbar_horiz else "vertical"))
    plt.tight_layout()
    plt.savefig(filepath+'.png', dpi=100)
    plt.close()

def hamming_similarity_matrix( vectors):
    if (type(vectors)==list):
        numvec=len(vectors)
        lenvec=len(vectors[0][0])
    else:
        numvec=len(vectors[:,0])
        lenvec=len(vectors[0,:])
    mat=np.zeros((numvec,numvec))
    for i in range(numvec):
        for j in range(numvec):
            if (type(vectors) == list):
                mat[i, j] = 1.0 - np.sum(np.abs(vectors[i] - vectors[j])) / lenvec
            else:
                mat[i,j]=1.0-np.sum(np.abs(vectors[i,:]-vectors[j,:]))/lenvec
    return (mat)

def hamming_similarity_VecAndMatrix(vec, mat):
    numvec=len(mat[:,0])
    lenvec=len(mat[0,:])
    sim=np.zeros((1,numvec))
    for i in range(numvec):
        sim[0,i]=1.0-np.sum(np.abs(vec-mat[i,:]))/lenvec
    return (sim)

def cos_dist_arr( vA, vB):
    ''' calculate cosine distance of two vectors'''
    if (torch.is_tensor(vA)):
        output = torch.dot(vA, vB) / (torch.sqrt(torch.dot(vA, vA)) * torch.sqrt(torch.dot(vB, vB)))
        outTensor = torch.tensor(1.0-output) #because we use latter that if value is 0 then vectors are the same
    else:
        output = np.dot(vA, vB) / (np.sqrt(np.dot(vA, vA)) * np.sqrt(np.dot(vB, vB)))
        outTnsor=1-output
    return outTensor

def cosine_similarity_matrix( vectors):
    if (type(vectors)==list):
        numvec=len(vectors)
        # lenvec=len(vectors[0][0])
    else:
        numvec=len(vectors[:,0])
        # lenvec=len(vectors[0,:])
    mat=np.zeros((numvec,numvec))
    for i in range(numvec):
        for j in range(numvec):
            if (type(vectors) == list):
                mat[i, j] = cos_dist_arr(vectors[i], vectors[j])
            else:
                mat[i, j] = cos_dist_arr(vectors[i, :], vectors[j, :])
    return (mat)

def cosine_similarity_VecAndMatrix(vec, mat):
    numvec=len(mat[:,0])
    sim=np.zeros((1,numvec))
    for i in range(numvec):
        sim[0, i] = cos_dist_arr(vec, mat[i, :])
    return (sim)

def plotSimilarityDistributions(title, filepath, labels, nbins, similarities):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    plt.tight_layout()

    for idx, sim in enumerate(similarities):
        hist, bins = np.histogram(sim, bins=nbins, range=[.5, 1.05])
        ax.bar(x=bins[:-1] + 0.002 * idx, height=hist, width=1 / 14 / nbins, label=labels[idx])

    ax.legend()
    ax.grid()
    plt.savefig(filepath, dpi=100)
    plt.close()


def studyEvolutionOfGeneralizedVectors_fromPers( vectors_StdHD_NonSeiz, vectors_StdHD_Seiz, vectors_OnlHD_NonSeiz, vectors_OnlHD_Seiz, similarityType, path, methodLists):

    GenVectors= {}
    for method in methodLists:
        if method not in GenVectors:
            GenVectors[method] = {}
            GenVectors[method][0] = []
            GenVectors[method][1] = []

        if (method == 'ClassicHD') or (method == 'StdHD'):
            vectors_NonSeiz = np.array([vectors_StdHD_NonSeiz[i] for i in vectors_StdHD_NonSeiz.keys()])
            (StdHD_NonSeiz_SimMean, StdHD_NonSeiz_SimStd) =measureSimWhenCreatingGenVectors(vectors_NonSeiz, similarityType)
            vectors_Seiz = np.array([vectors_StdHD_Seiz[i] for i in vectors_StdHD_Seiz.keys()])
            (StdHD_Seiz_SimMean, StdHD_Seiz_SimStd) = measureSimWhenCreatingGenVectors(vectors_Seiz,similarityType)
        else :
            vectors_NonSeiz = np.array([vectors_OnlHD_NonSeiz[i] for i in vectors_OnlHD_NonSeiz.keys()])
            (OnlHD_NonSeiz_SimMean, OnlHD_NonSeiz_SimStd) =measureSimWhenCreatingGenVectors(vectors_NonSeiz, similarityType)
            vectors_Seiz = np.array([vectors_OnlHD_Seiz[i] for i in vectors_OnlHD_Seiz.keys()])
            (OnlHD_Seiz_SimMean, OnlHD_Seiz_SimStd) = measureSimWhenCreatingGenVectors(vectors_Seiz,similarityType)

    fact=4
    fig1 = plt.figure(figsize=(12, 4), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    xArr=np.arange(0, len(StdHD_NonSeiz_SimMean[::fact]),1)*fact
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xArr, StdHD_NonSeiz_SimMean[::fact], yerr=StdHD_NonSeiz_SimStd[::fact] , fmt='k')
    ax1.errorbar(xArr, StdHD_Seiz_SimMean[::fact], yerr=StdHD_Seiz_SimStd[::fact] , fmt='b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Standard HD')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with previous HD model')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.errorbar(xArr, OnlHD_NonSeiz_SimMean[::fact], yerr=OnlHD_NonSeiz_SimStd[::fact] , fmt='k')
    ax1.errorbar(xArr, OnlHD_Seiz_SimMean[::fact], yerr=OnlHD_Seiz_SimStd[::fact], fmt='b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Online HD')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with previous HD model')
    ax1.grid()
    fig1.show()
    fig1.savefig(f"{path}/EvolutionOfGeneralizedVectors.png", bbox_inches='tight', dpi=100)
    fig1.savefig(f"{path}/EvolutionOfGeneralizedVectors.svg", bbox_inches='tight', dpi=100)
    plt.close(fig1)

def studyEvolutionOfGeneralizedVectors_fromPers_v2( vectors_norm, similarityType, path, methodLists, fact=10):

    for method in methodLists:
        if (method == 'ClassicHD') or (method == 'StdHD'):
            vectors_NonSeiz=np.array(list(vectors_norm[pat][method][0] for pat in vectors_norm.keys()))
            (StdHD_NonSeiz_SimMean, StdHD_NonSeiz_SimStd) =measureSimWhenCreatingGenVectors(vectors_NonSeiz, similarityType)
            vectors_Seiz = np.array(list(vectors_norm[pat][method][1] for pat in vectors_norm.keys()))
            (StdHD_Seiz_SimMean, StdHD_Seiz_SimStd) = measureSimWhenCreatingGenVectors(vectors_Seiz,similarityType)
        else :
            vectors_NonSeiz = np.array(list(vectors_norm[pat][method][0] for pat in vectors_norm.keys()))
            (OnlHD_NonSeiz_SimMean, OnlHD_NonSeiz_SimStd) =measureSimWhenCreatingGenVectors(vectors_NonSeiz, similarityType)
            vectors_Seiz = np.array(list(vectors_norm[pat][method][1] for pat in vectors_norm.keys()))
            (OnlHD_Seiz_SimMean, OnlHD_Seiz_SimStd) = measureSimWhenCreatingGenVectors(vectors_Seiz,similarityType)


    fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    xArr=np.arange(1, len(StdHD_NonSeiz_SimMean[::fact])+1,1)*fact
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xArr, StdHD_NonSeiz_SimMean[::fact], yerr=StdHD_NonSeiz_SimStd[::fact] , fmt='k')
    ax1.errorbar(xArr, StdHD_Seiz_SimMean[::fact], yerr=StdHD_Seiz_SimStd[::fact] , fmt='b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Standard HD')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with Pers models')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.errorbar(xArr, OnlHD_NonSeiz_SimMean[::fact], yerr=OnlHD_NonSeiz_SimStd[::fact] , fmt='k')
    ax1.errorbar(xArr, OnlHD_Seiz_SimMean[::fact], yerr=OnlHD_Seiz_SimStd[::fact], fmt='b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Online HD')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with Pers models')
    ax1.grid()
    fig1.show()
    fig1.savefig(f"{path}/EvolutionOfGeneralizedVectors.png", bbox_inches='tight', dpi=100)
    plt.close(fig1)

    ## PLOT FOR PAPER
    fig1 = plt.figure(figsize=(10, 4), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    ax1 = fig1.add_subplot(gs[0, 0])
    xArr=np.arange(1, len(StdHD_NonSeiz_SimMean[::fact])+1,1)*fact
    ax1.errorbar(xArr, OnlHD_NonSeiz_SimMean[::fact], yerr=OnlHD_NonSeiz_SimStd[::fact] , fmt='k')
    ax1.errorbar(xArr, OnlHD_Seiz_SimMean[::fact], yerr=OnlHD_Seiz_SimStd[::fact], fmt='b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Online HD')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with Pers models')
    ax1.grid()
    # ax1 = fig1.add_subplot(gs[0, 1])
    # indxMax=60
    # xArr=np.arange(1, len(StdHD_NonSeiz_SimMean[:indxMax:fact])+1,1)*fact
    # ax1.errorbar(xArr, OnlHD_NonSeiz_SimMean[:indxMax:fact], yerr=OnlHD_NonSeiz_SimStd[:indxMax:fact] , fmt='k')
    # ax1.errorbar(xArr, OnlHD_Seiz_SimMean[:indxMax:fact], yerr=OnlHD_Seiz_SimStd[:indxMax:fact], fmt='b')
    # ax1.legend(['NonSeiz', 'Seiz'])
    # ax1.set_title('Online HD - Zoomed in')
    # ax1.set_xlabel('Number of patients added')
    # ax1.set_ylabel('Similarity with Pers models')
    # ax1.grid()
    fig1.show()
    fig1.savefig(f"{path}/ForPaper_EvolutionOfGeneralizedVectors_OnlineHD_part1.png", bbox_inches='tight', dpi=100)
    fig1.savefig(f"{path}/ForPaper_EvolutionOfGeneralizedVectors_OnlineHD_part1.svg", bbox_inches='tight', dpi=100)
    plt.close(fig1)

    fig1 = plt.figure(figsize=(8, 3), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    ax1 = fig1.add_subplot(gs[0, 0])
    indxMax=60
    xArr=np.arange(1, len(StdHD_NonSeiz_SimMean[:indxMax:fact])+1,1)*fact
    ax1.errorbar(xArr, OnlHD_NonSeiz_SimMean[:indxMax:fact], yerr=OnlHD_NonSeiz_SimStd[:indxMax:fact] , fmt='k')
    ax1.errorbar(xArr, OnlHD_Seiz_SimMean[:indxMax:fact], yerr=OnlHD_Seiz_SimStd[:indxMax:fact], fmt='b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Online HD - Zoomed in')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with Pers models')
    ax1.grid()
    fig1.show()
    fig1.savefig(f"{path}/ForPaper_EvolutionOfGeneralizedVectors_OnlineHD_part2.png", bbox_inches='tight', dpi=100)
    fig1.savefig(f"{path}/ForPaper_EvolutionOfGeneralizedVectors_OnlineHD_part2.svg", bbox_inches='tight', dpi=100)
    plt.close(fig1)



def studyEvolutionOfGeneralizedVectors_fromPers_diffMethods( vectors_norm, similarityType, path, methodLists, addingType, fact=10):

    for method in methodLists:
        if (method == 'ClassicHD') or (method == 'StdHD'):
            method='StdHD'
            StdHD_sim= measureSimWhenCreatingGenVectors_DiffMethods(vectors_norm, similarityType, addingType, method)
        else :
            OnlHD_sim = measureSimWhenCreatingGenVectors_DiffMethods(vectors_norm, similarityType, addingType, method)


    fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    xArr=np.arange(1, len(OnlHD_sim[0,::fact])+1,1)*fact
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.plot(xArr, StdHD_sim[0,::fact], 'k')
    ax1.plot(xArr, StdHD_sim[1, ::fact], 'b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Standard HD')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with Pers models')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.plot(xArr, OnlHD_sim[0, ::fact], 'k')
    ax1.plot(xArr, OnlHD_sim[1, ::fact], 'b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Online HD')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with Pers models')
    ax1.grid()
    fig1.show()
    fig1.savefig(f"{path}/EvolutionOfGeneralizedVectors_"+addingType+".png", bbox_inches='tight', dpi=100)
    plt.close(fig1)

    ## PLOT FOR PAPER
    fig1 = plt.figure(figsize=(10, 4), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    ax1 = fig1.add_subplot(gs[0, 0])
    xArr=np.arange(1, len(OnlHD_sim[0,::fact])+1,1)*fact
    ax1.plot(xArr, OnlHD_sim[0, ::fact], 'k')
    ax1.plot(xArr, OnlHD_sim[1, ::fact], 'b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Online HD')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with Pers models')
    ax1.grid()
    fig1.show()
    fig1.savefig(f"{path}/ForPaper_EvolutionOfGeneralizedVectors_OnlineHD_part1_"+addingType+".png", bbox_inches='tight', dpi=100)
    fig1.savefig(f"{path}/ForPaper_EvolutionOfGeneralizedVectors_OnlineHD_part1_"+addingType+".svg", bbox_inches='tight', dpi=100)
    plt.close(fig1)

    fig1 = plt.figure(figsize=(8, 3), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    ax1 = fig1.add_subplot(gs[0, 0])
    indxMax=60
    xArr=np.arange(1, len(OnlHD_sim[0,:indxMax:fact])+1,1)*fact
    ax1.plot(xArr, OnlHD_sim[0, :indxMax:fact], 'k')
    ax1.plot(xArr, OnlHD_sim[1, :indxMax:fact], 'b')
    ax1.legend(['NonSeiz', 'Seiz'])
    ax1.set_title('Online HD - Zoomed in')
    ax1.set_xlabel('Number of patients added')
    ax1.set_ylabel('Similarity with Pers models')
    ax1.grid()
    fig1.show()
    fig1.savefig(f"{path}/ForPaper_EvolutionOfGeneralizedVectors_OnlineHD_part2_"+addingType+".png", bbox_inches='tight', dpi=100)
    fig1.savefig(f"{path}/ForPaper_EvolutionOfGeneralizedVectors_OnlineHD_part2_"+addingType+".svg", bbox_inches='tight', dpi=100)
    plt.close(fig1)


def measureSimWhenCreatingGenVectors( persVectors,similarityType, nshuffles=10):
    #create matrix instead of dictionary
    # persVectors_mat0 = np.array([persVectors[i] for i in persVectors.keys()])
    # numPat = len(persVectors_mat0[:, 0])
    numPat = len(persVectors[:, 0])
    similarityMat=np.zeros((nshuffles,numPat+1))

    #shuffl order of patients
    for s in range(nshuffles):
        rowsOrder=np.random.permutation(numPat)
        persVectors_mat=persVectors[rowsOrder,:]
        prevGen=persVectors_mat[0,:]*1
        for i in range(1,numPat+1):
            GenVec= (np.mean(persVectors_mat[0:i,:], axis=0)>=0.5) *1
            # #sim with previous gen vector (with one less subj)
            # if (similarityType == 'hamming'):
            #     similarityMat[s,i]= hamming_similarity_matrix(  np.vstack([prevGen, GenVec]))[0,1]
            # else:
            #     similarityMat[s,i]= cosine_similarity_matrix( np.vstack([prevGen, GenVec]))[0,1]
            #avrg smlarity with all other subjects pers seizures
            if (similarityType == 'hamming'):
                sim=hamming_similarity_VecAndMatrix(  GenVec, persVectors)
                similarityMat[s, i] = np.mean(sim[0,1:])
            else:
                sim=[s,i]= cosine_similarity_VecAndMatrix(GenVec, persVectors)
                similarityMat[s, i] = np.mean(sim[0,1:])

    #calculate average of all shuffles
    return (np.mean(similarityMat[:,1:], 0), np.std(similarityMat[:,1:],0) )


def measureSimWhenCreatingGenVectors_DiffMethods( vectors_norm,similarityType, addingType, method):

    patients=list(vectors_norm.keys())
    numPat = len(patients)
    HD_dim=len(vectors_norm[patients[0]][method][0])
    similarityMat=np.zeros((2, numPat+1))
    GenVectors, GenVectors_Norm, numAdded = {}, {}, {}
    GenVectors[0], GenVectors[1] = np.zeros((1, HD_dim)), np.zeros((1, HD_dim))
    GenVectors_Norm[0], GenVectors_Norm[1] = np.zeros((1, HD_dim)), np.zeros( (1, HD_dim))
    numAdded[0], numAdded[1] = 0, 0

    for patIndx, pat in enumerate(patients):
        (GenVectors, GenVectors_Norm, numAdded) =updateGenModels_DiffMethods(GenVectors, GenVectors_Norm, numAdded, addingType, vectors_norm[pat][method])
        GenVec= np.array([GenVectors_Norm[i] for i in GenVectors_Norm.keys()])

        #NonSEIZURE
        persVectors = np.array([vectors_norm[i][method][0] for i in vectors_norm.keys()])
        #avrg smlarity with all other subjects pers seizures
        if (similarityType == 'hamming'):
            sim=hamming_similarity_VecAndMatrix( GenVec[0,:], persVectors)
            similarityMat[ 0,patIndx] = np.mean(sim[0,1:])
        else:
            sim=[s,i]= cosine_similarity_VecAndMatrix(GenVec, persVectors)
            similarityMat[0, patIndx] = np.mean(sim[0,1:])

        # SEIZURE
        persVectors = np.array([vectors_norm[i][method][1] for i in vectors_norm.keys()])
        # avrg smlarity with all other subjects pers seizures
        if (similarityType == 'hamming'):
            sim = hamming_similarity_VecAndMatrix(GenVec[1,:], persVectors)
            similarityMat[1, patIndx] = np.mean(sim[0, 1:])
        else:
            sim = [s, i] = cosine_similarity_VecAndMatrix(GenVec, persVectors)
            similarityMat[1, patIndx] = np.mean(sim[0, 1:])


    #calculate average of all shuffles
    # return (np.mean(similarityMat[:,1:], 0), np.std(similarityMat[:,1:],0) )
    return (similarityMat[:,1:])

def updateGenModels_DiffMethods(GenVectors, GenVectors_Norm, numAdded, addingType, vectors_norm):
    # if average
    if (addingType == 'Average' or addingType == 'Avrg'):
        numAdded[0] = numAdded[0] + 1
        numAdded[1] = numAdded[1] + 1
        GenVectors[0] = GenVectors[0] + np.array(vectors_norm[0])
        GenVectors[1] = GenVectors[1] + np.array(vectors_norm[1])

    elif (addingType == 'WAdd'):
        # just adding
        simNS = hamming_similarity_VecAndMatrix(GenVectors_Norm[0], np.reshape(np.array(vectors_norm[0]), (1, -1)))[0][0]
        simS = hamming_similarity_VecAndMatrix(GenVectors_Norm[1],  np.reshape(np.array(vectors_norm[1]), (1, -1)))[0][0]
        numAdded[0] = numAdded[0] + (1 - simNS)
        numAdded[1] = numAdded[1] + (1 - simS)
        GenVectors[0] = GenVectors[0] + (1 - simNS) * np.array(vectors_norm[0])
        GenVectors[1] = GenVectors[1] + (1 - simS) * np.array(vectors_norm[1])

    elif (addingType == 'WAdd&Sub'):
        # adding to correct class and subtractng from opposite class - both weighted
        simNS = hamming_similarity_VecAndMatrix(GenVectors_Norm[0],   np.reshape(np.array(vectors_norm[0]), (1, -1)))[0][0]
        simS = hamming_similarity_VecAndMatrix(GenVectors_Norm[1],  np.reshape(np.array(vectors_norm[1]), (1, -1)))[0][0]
        simSNS = hamming_similarity_VecAndMatrix(GenVectors_Norm[0],np.reshape(np.array(vectors_norm[1]), (1, -1)))[0][0]
        simNSS = hamming_similarity_VecAndMatrix(GenVectors_Norm[0],  np.reshape(np.array(vectors_norm[1]), (1, -1)))[0][0]
        numAdded[0] = numAdded[0] + (1 - simNS) - simSNS
        numAdded[1] = numAdded[1] + (1 - simS) - simNSS
        GenVectors[0] = GenVectors[0] + (1 - simNS) * np.array(  vectors_norm[0]) - simSNS * np.array(vectors_norm[1])
        GenVectors[1] = GenVectors[1] + (1 - simS) * np.array( vectors_norm[1]) - simNSS * np.array(vectors_norm[0])
    elif (addingType == 'WSub' or addingType == 'Weighted'):
        subtractFact = 1
        # adding to correct class and subtractng from opposite class - only subtraction wighted
        simNS = hamming_similarity_VecAndMatrix(GenVectors_Norm[0],  np.reshape(np.array(vectors_norm[0]), (1, -1)))[0][0]
        simS = hamming_similarity_VecAndMatrix(GenVectors_Norm[1],   np.reshape(np.array(vectors_norm[1]), (1, -1)))[0][0]
        simSNS = hamming_similarity_VecAndMatrix(GenVectors_Norm[0],  np.reshape(np.array(vectors_norm[1]), (1, -1)))[0][0]
        simNSS = hamming_similarity_VecAndMatrix(GenVectors_Norm[0],  np.reshape(np.array(vectors_norm[1]), (1, -1)))[0][0]
        numAdded[0] = numAdded[0] + 1 - simSNS * subtractFact
        numAdded[1] = numAdded[1] + 1 - simNSS * subtractFact
        GenVectors[0] = GenVectors[0] + np.array( vectors_norm[0]) - simSNS * subtractFact * np.array(vectors_norm[1])
        GenVectors[1] = GenVectors[1] + np.array( vectors_norm[1]) - simNSS * subtractFact * np.array(vectors_norm[0])
    # normalize
    GenVectors_Norm[0] = (GenVectors[0] > math.floor(numAdded[0] / 2)) * 1
    GenVectors_Norm[1] = (GenVectors[1] > math.floor(numAdded[1] / 2)) * 1

    return (GenVectors, GenVectors_Norm, numAdded)


def createGeneralizedVectors(vectors, vectors_norm, numAdded_vec,HDParams, path, methodLists, numSubj=-1):

    #if creating generalized from smaller subset of subjects
    if (numSubj==-1):
        numSubj=len(numAdded_vec)
    randIndx=np.random.permutation(len(numAdded_vec))[0:numSubj]
    randNames = np.array(list(vectors_norm.keys()))[randIndx]
    randNames= list(randNames)

    GenVectors= {}
    for method in methodLists:
        if method not in GenVectors:
            GenVectors[method] = {}
            GenVectors[method][0] = []
            GenVectors[method][1] = []
        vectors_sum= {}
        numAdded_sum=np.zeros(2)
        vectors_sum=np.zeros((2,HDParams.HD_dim))
        # for patIndx, patient in enumerate(tqdm(vectors_norm[randIndx,:], desc='Personnal vectors aggregation', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}')):
        for patIndx, patient in enumerate(randNames):
            if (numAdded_vec[patient][method][0]!=0): #case whn some subj smehow have 0 seiz
                numAdded_sum[0] = numAdded_sum[0] + 1
                numAdded_sum[1] = numAdded_sum[1] + 1
                vectors_sum[0] = vectors_sum[0] + vectors[patient][method][0]/ np.abs(numAdded_vec[patient][method][0]) #take numbers from last CV #[-1]
                vectors_sum[1] = vectors_sum[1] + vectors[patient][method][1] / np.abs(numAdded_vec[patient][method][1])#[-1]
        #normalize
        # if (method=='StdHD' or method=='ClassicHD'):
        GenVectors[method][0]  = (vectors_sum[0]  > math.floor(numAdded_sum[0]/ 2)) * 1
        GenVectors[method][1] = (vectors_sum[1] > math.floor(numAdded_sum[1] / 2)) * 1
        # else: #online
        #     GenVectors[method][0] = (vectors_sum[0] > 0) * 1 #for online non norm vectors are as binary (-1 and 1, not 0 and 1)
        #     GenVectors[method][1] = (vectors_sum[1] > 0) * 1

        #save gen vectors
        ModelVectors=np.zeros((2,len(GenVectors[method][0] )))
        ModelVectors[0,:]=vectors_sum[0]
        ModelVectors[1, :] =vectors_sum[1]
        ModelVectorsNorm=np.zeros((2,len(GenVectors[method][0] )))
        ModelVectorsNorm[0,:]=GenVectors[method][0]
        ModelVectorsNorm[1, :] = GenVectors[method][1]
        numAdded=np.zeros((2,1))
        numAdded[0]=numAdded_sum[0]
        numAdded[1]=numAdded_sum[1]
        if (method =='ClassicHD') or (method=='StdHD'):
            shortName='StdHD'
        else:
            shortName='OnlHD'
        outName=path+'/GenVectors_'+shortName+'_AllSubj.pickle'
        with open(outName, 'wb') as f:
            pickle.dump([ModelVectors, ModelVectorsNorm, numAdded], f)
    return GenVectors

def createGeneralizedVectors_weightedAdding(vectors_norm, numAdded_vec,HDParams, path, methodLists, addingType, numItter=1, numSubj=-1):
    #if creating from smaller random subset of patients
    if (numSubj==-1):
        numSubj=len(numAdded_vec)
    randIndx=np.random.permutation(len(numAdded_vec))[0:numSubj]
    randNames = np.array(list(vectors_norm.keys()))[randIndx]
    randNames= list(randNames)

    GenVectors, GenVectors_Norm, numAdded= {}, {}, {}
    for met in methodLists:
        if ( met in ['ClassicHD', 'classicHD', 'StdHD']):
            method='StdHD'
        elif (met in ['OnlineHD', 'OnlHD','OnlineHDAddSub', 'OnlineHDAddOnly']):
            method='OnlHD'
        if method not in GenVectors:
            GenVectors[method], GenVectors_Norm[method] , numAdded[method]= {}, {}, {}
            GenVectors[method][0] , GenVectors[method][1] =np.zeros((1,HDParams.HD_dim)), np.zeros((1,HDParams.HD_dim))
            GenVectors_Norm[method][0] , GenVectors_Norm[method][1] =np.zeros((1,HDParams.HD_dim)), np.zeros((1,HDParams.HD_dim))
            numAdded[method][0], numAdded[method][1] = 0,0

        for i in range(numItter):
            for patIndx, patient in enumerate(randNames):
                # if average
                if (addingType=='Average' or addingType=='Avrg'):
                    numAdded[method][0] = numAdded[method][0] +1
                    numAdded[method][1] = numAdded[method][1] +1
                    GenVectors[method][0] = GenVectors[method][0] + np.array(vectors_norm[patient][method][0])
                    GenVectors[method][1] = GenVectors[method][1] + np.array(vectors_norm[patient][method][1])
                    # #normalize
                    # GenVectors_Norm[method][0]  = (GenVectors[method][0]  > math.floor(numAdded[method][0]/ 2)) * 1
                    # GenVectors_Norm[method][1] = (GenVectors[method][1] > math.floor(numAdded[method][1] / 2)) * 1

                elif (addingType=='WAdd'):
                    #just adding
                    simNS= hamming_similarity_VecAndMatrix(GenVectors_Norm[method][0] ,np.reshape(np.array(vectors_norm[patient][method][0]), (1,-1)))[0][0]
                    simS = hamming_similarity_VecAndMatrix(GenVectors_Norm[method][1], np.reshape(np.array(vectors_norm[patient][method][1]), (1,-1)))[0][0]
                    numAdded[method][0] = numAdded[method][0] + (1-simNS)
                    numAdded[method][1] = numAdded[method][1] + (1-simS)
                    GenVectors[method][0] = GenVectors[method][0] + (1-simNS)* np.array(vectors_norm[patient][method][0])
                    GenVectors[method][1] = GenVectors[method][1] + (1-simS) * np.array(vectors_norm[patient][method][1])
                    # #normalize
                    # GenVectors_Norm[method][0]  = (GenVectors[method][0]  > math.floor(numAdded[method][0]/ 2)) * 1
                    # GenVectors_Norm[method][1] = (GenVectors[method][1] > math.floor(numAdded[method][1] / 2)) * 1
                elif (addingType=='WAdd&Sub'):
                    #adding to correct class and subtractng from opposite class - both weighted
                    simNS= hamming_similarity_VecAndMatrix(GenVectors_Norm[method][0] ,np.reshape(np.array(vectors_norm[patient][method][0]), (1,-1)))[0][0]
                    simS = hamming_similarity_VecAndMatrix(GenVectors_Norm[method][1], np.reshape(np.array(vectors_norm[patient][method][1]), (1,-1)))[0][0]
                    simSNS = hamming_similarity_VecAndMatrix(GenVectors_Norm[method][0], np.reshape(np.array(vectors_norm[patient][method][1]), (1, -1)))[0][0]
                    simNSS = hamming_similarity_VecAndMatrix(GenVectors_Norm[method][0],  np.reshape(np.array(vectors_norm[patient][method][1]), (1, -1)))[0][0]
                    numAdded[method][0] = numAdded[method][0] + (1-simNS) -simSNS
                    numAdded[method][1] = numAdded[method][1] + (1-simS) -simNSS
                    GenVectors[method][0] = GenVectors[method][0] + (1-simNS)* np.array(vectors_norm[patient][method][0]) -simSNS*np.array(vectors_norm[patient][method][1])
                    GenVectors[method][1] = GenVectors[method][1] + (1-simS) * np.array(vectors_norm[patient][method][1]) -simNSS*np.array(vectors_norm[patient][method][0])
                    # #normalize
                    # GenVectors_Norm[method][0]  = (GenVectors[method][0]  > math.floor(numAdded[method][0]/ 2)) * 1
                    # GenVectors_Norm[method][1] = (GenVectors[method][1] > math.floor(numAdded[method][1] / 2)) * 1
                elif (addingType=='WSub' or addingType=='Weighted'):
                    subtractFact=1
                    #adding to correct class and subtractng from opposite class - only subtraction wighted
                    simNS= hamming_similarity_VecAndMatrix(GenVectors_Norm[method][0] ,np.reshape(np.array(vectors_norm[patient][method][0]), (1,-1)))[0][0]
                    simS = hamming_similarity_VecAndMatrix(GenVectors_Norm[method][1], np.reshape(np.array(vectors_norm[patient][method][1]), (1,-1)))[0][0]
                    simSNS = hamming_similarity_VecAndMatrix(GenVectors_Norm[method][0], np.reshape(np.array(vectors_norm[patient][method][1]), (1, -1)))[0][0]
                    simNSS = hamming_similarity_VecAndMatrix(GenVectors_Norm[method][0],  np.reshape(np.array(vectors_norm[patient][method][1]), (1, -1)))[0][0]
                    numAdded[method][0] = numAdded[method][0] +1 -simSNS*subtractFact
                    numAdded[method][1] = numAdded[method][1] +1 -simNSS*subtractFact
                    GenVectors[method][0] = GenVectors[method][0] + np.array(vectors_norm[patient][method][0]) -simSNS*subtractFact*np.array(vectors_norm[patient][method][1])
                    GenVectors[method][1] = GenVectors[method][1] + np.array(vectors_norm[patient][method][1]) -simNSS*subtractFact*np.array(vectors_norm[patient][method][0])
                #normalize
                GenVectors_Norm[method][0]  = (GenVectors[method][0]  > math.floor(numAdded[method][0]/ 2)) * 1
                GenVectors_Norm[method][1] = (GenVectors[method][1] > math.floor(numAdded[method][1] / 2)) * 1


    if (numItter==1):
        outName=path+'/GenVectors_'+addingType+ '.pickle'
    else:
        outName = path + '/GenVectors_'+addingType+ '_Itter'+str(numItter)+'.pickle'
    with open(outName, 'wb') as f:
        pickle.dump([GenVectors, GenVectors_Norm, numAdded], f)
    return GenVectors_Norm


def calcAvrgSim_PersAndGen(vecsNormGen, vecsNormPers, methodLists):
    # subjNames=list(vecsNormPers.keys())
    distMat = {}
    for method in methodLists:
        distMat[method] = np.zeros((len(vecsNormPers), 4))
        persVectors_Seiz = np.array([vecsNormPers[i][method][1] for i in vecsNormPers.keys()])
        persVectors_NonSeiz = np.array([vecsNormPers[i][method][0] for i in vecsNormPers.keys()])

        # NSNS, SS, NSS, SNS
        distMat[method][:, 0] = hamming_similarity_VecAndMatrix(vecsNormGen[method][0], persVectors_NonSeiz)
        distMat[method][:, 1] = hamming_similarity_VecAndMatrix(vecsNormGen[method][1], persVectors_Seiz)
        distMat[method][:, 2] = hamming_similarity_VecAndMatrix(vecsNormGen[method][0], persVectors_Seiz)
        distMat[method][:, 3] = hamming_similarity_VecAndMatrix(vecsNormGen[method][1], persVectors_NonSeiz)
    return (distMat)

def compareDiffGenCreatingApproaches( approachesList, folder, vectorsPersNorm, methodLists):
    dataToPlot={}
    for met in methodLists:
        dataToPlot[met]= {}
        dataToPlot[met]['NSNS']=np.zeros((len(approachesList),2)) #for every approach  mean and std
        dataToPlot[met]['SS'] = np.zeros((len(approachesList), 2))  # for every approach  mean and std
        dataToPlot[met]['NSS'] = np.zeros((len(approachesList), 2))  # for every approach  mean and std
        dataToPlot[met]['SNS'] = np.zeros((len(approachesList), 2))  # for every approach  mean and std
        dataToPlot[met]['CC'] = np.zeros((len(approachesList), 2))  # for every approach  mean and std
        dataToPlot[met]['WW'] = np.zeros((len(approachesList), 2))  # for every approach  mean and std
        dataToPlot[met]['Diff'] = np.zeros((len(approachesList), 2))  # for every approach  mean and std

    for appIndx, app in enumerate(approachesList):
        #load generalized vectors
        outName = folder + '/GenVectors_'+app  + '.pickle'
        with open(outName, 'rb') as file:
            (GenVectors, GenVectors_Norm, numAdded) = pickle.load(file)
        #calculate distances from generalizd for all persnalized
        distMat= calcAvrgSim_PersAndGen(GenVectors_Norm, vectorsPersNorm, methodLists)
        for met in methodLists:
            dataToPlot[met]['NSNS'][appIndx,:]=[ np.mean(distMat[met][:,0]),  np.std(distMat[met][:,0])]
            dataToPlot[met]['SS'][appIndx, :] = [np.mean(distMat[met][:, 1]), np.std(distMat[met][:, 1])]
            dataToPlot[met]['NSS'][appIndx, :] = [np.mean(distMat[met][:, 2]), np.std(distMat[met][:, 2])]
            dataToPlot[met]['SNS'][appIndx, :] = [np.mean(distMat[met][:, 3]), np.std(distMat[met][:, 3])]
            pom=np.hstack((distMat[met][:,0], distMat[met][:,1]) )
            dataToPlot[met]['CC'][appIndx, :] = [np.mean(pom), np.std(pom)]
            pom=np.hstack((distMat[met][:,2], distMat[met][:,3]) )
            dataToPlot[met]['WW'][appIndx, :] = [np.mean(pom), np.std(pom)]
            #calculate distance between good predictions vector and wrong predictions vector
            pom=np.hstack((distMat[met][:,0]-distMat[met][:,3] , distMat[met][:,1]-distMat[met][:,2]) )
            dataToPlot[met]['Diff'][appIndx, :] = [np.mean(pom), np.std(pom)]

    #visualize differences for different gen vctors creatinon appraches
    fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.3)
    xArr=np.arange(0, len(approachesList),1)
    for metIndx, met in enumerate(methodLists):
        ax1 = fig1.add_subplot(gs[0, metIndx])
        ax1.errorbar(xArr, dataToPlot[met]['NSNS'][:,0], dataToPlot[met]['NSNS'][:,1], fmt='m--')
        ax1.errorbar(xArr, dataToPlot[met]['SS'][:, 0], dataToPlot[met]['SS'][:, 1], fmt='r--')
        ax1.errorbar(xArr, dataToPlot[met]['SNS'][:, 0], dataToPlot[met]['SNS'][:, 1], fmt='b--')
        ax1.errorbar(xArr, dataToPlot[met]['NSS'][:, 0], dataToPlot[met]['NSS'][:, 1], fmt='k--')
        ax1.errorbar(xArr, dataToPlot[met]['CC'][:,0], dataToPlot[met]['CC'][:,1], fmt='r')
        ax1.errorbar(xArr, dataToPlot[met]['WW'][:, 0], dataToPlot[met]['WW'][:, 1], fmt='k')
        ax1.legend(['NSNS', 'SS', 'SNS', 'NSS','Corr', 'Wrong'])
        ax1.set_title(met)
        ax1.set_xticks(xArr)
        ax1.set_xticklabels(approachesList, fontsize=12 * 0.8, rotation=0)
        ax1.set_ylabel('Similarity')
        ax1.grid()
        ax1 = fig1.add_subplot(gs[1, metIndx])
        ax1.errorbar(xArr, dataToPlot[met]['Diff'][:, 0], dataToPlot[met]['Diff'][:, 1], fmt='b')
        ax1.legend(['Diff'])
        ax1.set_title(met)
        ax1.set_xticks(xArr)
        ax1.set_xticklabels(approachesList, fontsize=12 * 0.8, rotation=0)
        ax1.set_ylabel('Similarity')
        ax1.grid()
    fig1.show()
    fig1.savefig(folder+'/ComprisonDiffGenVectorCreation.png', bbox_inches='tight', dpi=100)
    plt.close(fig1)

    # FOR PAPER - onyl OnlineHD
    #visualize differences for different gen vctors creatinon appraches
    fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.3)
    xArr=np.arange(0, len(approachesList),1)
    met=methodLists[1]
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xArr, dataToPlot[met]['NSNS'][:,0], dataToPlot[met]['NSNS'][:,1], fmt='m--')
    ax1.errorbar(xArr, dataToPlot[met]['SS'][:, 0], dataToPlot[met]['SS'][:, 1], fmt='r--')
    ax1.errorbar(xArr, dataToPlot[met]['SNS'][:, 0], dataToPlot[met]['SNS'][:, 1], fmt='b--')
    ax1.errorbar(xArr, dataToPlot[met]['NSS'][:, 0], dataToPlot[met]['NSS'][:, 1], fmt='k--')
    ax1.errorbar(xArr, dataToPlot[met]['CC'][:,0], dataToPlot[met]['CC'][:,1], fmt='r')
    ax1.errorbar(xArr, dataToPlot[met]['WW'][:, 0], dataToPlot[met]['WW'][:, 1], fmt='k')
    ax1.legend(['NSNS', 'SS', 'SNS', 'NSS','Corr', 'Wrong'])
    # ax1.set_title(met)
    ax1.set_xticks(xArr)
    ax1.set_xticklabels(approachesList, fontsize=12 * 0.8, rotation=0)
    ax1.set_title('Vector similarity')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.errorbar(xArr, dataToPlot[met]['Diff'][:, 0], dataToPlot[met]['Diff'][:, 1], fmt='b')
    # ax1.legend(['Diff'])
    # ax1.set_title(met)
    ax1.set_xticks(xArr)
    ax1.set_xticklabels(approachesList, fontsize=12 * 0.8, rotation=0)
    ax1.set_title('Vector separability')
    ax1.grid()
    fig1.show()
    fig1.savefig(folder+'/ForPaper_ComprisonDiffGenVectorCreation_OnlineHD.png', bbox_inches='tight', dpi=100)
    fig1.savefig(folder + '/ForPaper_ComprisonDiffGenVectorCreation_OnlineHD.svg', bbox_inches='tight', dpi=100)
    plt.close(fig1)


    # FOR PAPER - onyl OnlineHD
    #visualize differences for different gen vctors creatinon appraches
    fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.3)
    xArr=np.arange(0, len(approachesList),1)
    met=methodLists[1]
    ax1 = fig1.add_subplot(gs[0, 0])
    # ax1.bar(xArr, dataToPlot[met]['NSNS'][:,0], yerr=dataToPlot[met]['NSNS'][:,1], fmt='m--')
    # ax1.bar(xArr, dataToPlot[met]['SS'][:, 0], yerr=dataToPlot[met]['SS'][:, 1], fmt='r--')
    # ax1.bar(xArr, dataToPlot[met]['SNS'][:, 0], yerr=dataToPlot[met]['SNS'][:, 1], fmt='b--')
    # ax1.bar(xArr, dataToPlot[met]['NSS'][:, 0], yerr=dataToPlot[met]['NSS'][:, 1], fmt='k--')
    ax1.bar(xArr-0.2, dataToPlot[met]['CC'][:,0], yerr=dataToPlot[met]['CC'][:,1], color='skyblue', width=0.4)
    ax1.bar(xArr+0.2, dataToPlot[met]['WW'][:, 0], yerr=dataToPlot[met]['WW'][:, 1], color='paleturquoise', width=0.4)
    ax1.legend(['Corr', 'Wrong'])
    # ax1.set_title(met)
    ax1.set_xticks(xArr)
    ax1.set_xticklabels(approachesList, fontsize=12 * 0.8, rotation=0)
    ax1.set_title('Vector similarity')
    ax1.grid()
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.bar(xArr, dataToPlot[met]['Diff'][:, 0], yerr=dataToPlot[met]['Diff'][:, 1], color='lightblue')
    # ax1.legend(['Diff'])
    # ax1.set_title(met)
    ax1.set_xticks(xArr)
    ax1.set_xticklabels(approachesList, fontsize=12 * 0.8, rotation=0)
    ax1.set_title('Vector separability')
    ax1.grid()
    fig1.show()
    fig1.savefig(folder+'/ForPaper_ComprisonDiffGenVectorCreation_OnlineHD_bar.png', bbox_inches='tight', dpi=100)
    fig1.savefig(folder + '/ForPaper_ComprisonDiffGenVectorCreation_OnlineHD_bar.svg', bbox_inches='tight', dpi=100)
    plt.close(fig1)

def findMostSim(vector, modelVectors):
    hamMin=1
    bestIndx=0
    # if (modelVectors.ndim==1):
    #     modelVectors=modelVectors.reshape((1,-1))
    for mvi,  mv in enumerate(modelVectors):
        hamm=np.mean(abs(mv-vector))
        if (hamMin>hamm):
            hamMin=hamm
            bestIndx=mvi
    return bestIndx, hamMin

def createGeneralizedVectors_exceptOneSubj(vectors, vectors_norm, numAdded_vec, HDParams, subj, path, methodLists, numSubj):

    #if creating generalized from smaller subset of subjects
    if (numSubj==-1):
        numSubj=len(numAdded_vec)
    # numSubj=15
    randIndx=np.random.permutation(len(numAdded_vec))[0:numSubj]
    randNames = np.array(list(vectors_norm.keys()))[randIndx]
    randNames= list(randNames)

    GenVectors= {}
    # methodLists=['ClassicHD','OnlineHDAddSub']
    for method in methodLists:
        if method not in GenVectors:
            GenVectors[method] = {}
            GenVectors[method][0] = []
            GenVectors[method][1] = []
        vectors_sum= {}
        # numAdded_sum=np.zeros(2)
        numAdded_sum=np.zeros(2)
        vectors_sum=np.zeros((2,HDParams.HD_dim))
        # for patIndx, patient in enumerate(tqdm(vectors_norm, desc='Personnal vectors aggregation', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}')):
        for patIndx, patient in enumerate(randNames):
            # if patIndx==0:
            #     numAdded_sum=numAdded_vec[patient][method]
            #     vectors_sum[0]= vectors[patient][method][0][-1]
            #     vectors_sum[1] = vectors[patient][method][1][-1]
            if (subj!=patient):
                # numAdded_sum[0] = numAdded_sum[0]+numAdded_vec[patient][method][0]
                # numAdded_sum[1] = numAdded_sum[1] + numAdded_vec[patient][method][1]
                # vectors_sum[0] = vectors_sum[0]+vectors[patient][method][0][-1]
                # vectors_sum[1] = vectors_sum[1] + vectors[patient][method][1][-1]
                if (numAdded_vec[patient][method][0] != 0):  # case whn some subj smehow have 0 seiz
                    numAdded_sum[0] = numAdded_sum[0] +1
                    numAdded_sum[1] = numAdded_sum[1] +1
                    vectors_sum[0] = vectors_sum[0] + vectors[patient][method][0]/numAdded_vec[patient][method][0] #take numbers from last CV #[-1]
                    vectors_sum[1] = vectors_sum[1] + vectors[patient][method][1]/numAdded_vec[patient][method][1] #take numbers from last CV #[-1]
        # normalize
        # if (method == 'ClassicHD'):
        GenVectors[method][0] = (vectors_sum[0] > int(math.floor(numAdded_sum[0] / 2))) * 1
        GenVectors[method][1] = (vectors_sum[1] > int(math.floor(numAdded_sum[1] / 2))) * 1
        # else:  # online
        #     GenVectors[method][0] = (vectors_sum[ 0] > 0) * 1  # for online non norm vectors are as binary (-1 and 1, not 0 and 1)
        #     GenVectors[method][1] = (vectors_sum[1] > 0) * 1

        #save gen vectors
        ModelVectors=np.zeros((2,len(GenVectors[method][0] )))
        ModelVectors[0,:]=vectors_sum[0]
        ModelVectors[1, :] =vectors_sum[1]
        ModelVectorsNorm=np.zeros((2,len(GenVectors[method][0] )))
        ModelVectorsNorm[0,:]=GenVectors[method][0]
        ModelVectorsNorm[1, :] = GenVectors[method][1]
        numAdded=np.zeros((2,1))
        numAdded[0]=numAdded_sum[0]
        numAdded[1]=numAdded_sum[1]
        if (method =='ClassicHD' or method=='StdHD'):
            shortName='StdHD'
        else:
            shortName='OnlHD'
        outName=path+'/GenVectors_'+shortName+'_Subj'+subj+'.pickle'
        with open(outName, 'wb') as f:
            pickle.dump([ModelVectors, ModelVectorsNorm, numAdded], f)

    return GenVectors

def clusterFromSimilarity_SandNSIn1Subj( title, path, similarities, numNS):
     import scipy.spatial.distance as ssd
     import scipy.cluster.hierarchy as hcluster

     fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
     # gs = GridSpec(1, 1, figure=fig1)
     linkage = hcluster.linkage(similarities[0:numNS, 0:numNS])
     dendro = hcluster.dendrogram(linkage)
     # fig1.set_title(title +' - NonSeizure')
     fig1.show()
     fig1.savefig(path+'_NS.png', bbox_inches='tight', dpi=100)
     plt.close(fig1)

     fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
     # gs = GridSpec(1, 1, figure=fig1)
     linkage = hcluster.linkage(similarities[numNS:, numNS:])
     dendro = hcluster.dendrogram(linkage)
     # fig1.set_title(title +' - Seizure')
     fig1.show()
     fig1.savefig(path+'_S.png', bbox_inches='tight', dpi=100)
     plt.close(fig1)

def createSeizNames(pat, numSeiz):
    names=[]
    for s in range(numSeiz):
        names.append( 'P'+pat+'_S'+str(s+1))
    return names
