''' library with different functions on HD vectors, uses torch library '''
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

import torch
import time, sys
from VariousFunctionsLib import *


class HD_classifier_LBP:
    ''' Approach that uses LBP values and then maps them to HD vectors'''

    def __init__(self, LBP_dim,HD_dim, NumCh, LBP_val, device,  stringCH, stringLBP, roundingType,cuda = True):
        ''' Initialize an HD vectors using the torch library
        LBP_dim: number of elements inside the LBP item memory, number of possible combinations of 0100101 pattenrs, 2^(LBPlen-1)
        HD_dim: dimension of the HD vectors
        NumCh: number of channels of the channels item memory
        LBP_val: LBP_len + 1
        stringCH: type of initialization of HD vectors representing channels
        stringLBP: type of initialization of HD vectors representing LBP values
        roundingType: type of rounding of vectors to binary vectors
        device: gpu to be used
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''
        self.LBP_dim = LBP_dim
        self.NumCh = NumCh
        self.HD_dim = HD_dim
        self.LBP_val = LBP_val
        self.device = device
        self.roundingType=roundingType

        # CEREATING LBP VALUE VECTORS
        if stringLBP == 'sandwich':
            self.proj_mat_LBP=func_generateVectorsMemory_Sandwich(cuda, device, self.HD_dim, self.LBP_dim)
        elif stringLBP=='random':
            self.proj_mat_LBP=func_generateVectorsMemory_Random(cuda, device, self.HD_dim, self.LBP_dim)
        elif "scaleNoRand"in stringLBP:
            factor = int(stringLBP[11:])
            self.proj_mat_LBP = func_generateVectorsMemory_ScaleNoRand(cuda, device,  self.HD_dim, self.LBP_dim, factor)
        elif "scaleRand" in stringLBP:
            factor = int(stringLBP[9:])
            self.proj_mat_LBP = func_generateVectorsMemory_ScaleRand(cuda, device, self.HD_dim, self.LBP_dim,factor)
        else:
            self.proj_mat_LBP = func_generateVectorsMemory_Random(cuda, device, self.HD_dim, self.LBP_dim)

        # CEREATING CHANNELS VECTORS
        if stringCH == 'sandwich':
            self.proj_mat_channels = func_generateVectorsMemory_Sandwich(cuda, device, self.HD_dim, self.NumCh)
        elif stringCH == 'random':
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, device, self.HD_dim, self.NumCh)
        elif "scaleNoRand" in stringCH:
            factor = int(stringCH[11:])
            self.proj_mat_LBP = func_generateVectorsMemory_ScaleNoRand(cuda, device, self.HD_dim, self.NumCh, factor)
        elif "scaleRand" in stringCH:
            factor = int(stringCH[9:])
            self.proj_mat_LBP = func_generateVectorsMemory_ScaleRand(cuda, device, self.HD_dim, self.NumCh, factor)
        else:
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, device, self.HD_dim, self.NumCh)

    def learn_HD_proj(self,EEG):
        ''' From window of EEG data calculates feature values, maps them to vectors and returns one vector that represents whole EEG window
        '''
        N_channels,learningEnd = EEG.size()
        modelVector = torch.cuda.ShortTensor(1, 1).zero_()
        LBP_weights = torch.cuda.ShortTensor([2**0, 2**1, 2**2, 2**3, 2**4, 2**5])

        timeFeat=0
        timeHDVec=0
        for iStep in range(learningEnd-6):
            t0=time.time()
            x = EEG[:,iStep:(iStep+self.LBP_val)].short() #cutting segments of LBP_val symbols
            bp = (torch.add(-x[:,0:self.LBP_val-1], 1,x[:,1:self.LBP_val])>0).short() #calculating if it is rise or fall of values, 1 when increasing, 0 is the same or falling
            value = torch.sum(torch.mul(LBP_weights,bp), dim=1) #calculating values instead of patterns to be able to do histogram
            timeFeat=timeFeat +(time.time()-t0)
            t0 = time.time()
            bindingVector=xor(self.proj_mat_channels,self.proj_mat_LBP[:,value.long()])
            output_vector=torch.sum(bindingVector,dim=1)
            #here we broke ties summing an additional HV in case of an even number
            if N_channels%2==0:
                output_vector = torch.add(xor(self.proj_mat_LBP[:,1],self.proj_mat_LBP[:,2]),1,output_vector)
            if (self.roundingType=='inSteps'): # Rrounding for all ch
                output_vector=(output_vector>int(math.floor(self.NumCh/2))).short()
            modelVector=torch.add(modelVector,1,output_vector)
            timeHDVec=timeHDVec +(time.time()-t0)
        if (self.roundingType == 'inSteps'): # rounding for all steps
            modelVector = (modelVector> (learningEnd-6)/2).short()
        elif (self.roundingType=='onlyOne'): # rounding for all steps and ch
            modelVector = (modelVector > int(math.floor((learningEnd - 6)*self.NumCh / 2)) ).short()
        elif (self.roundingType=='noRounding'): #then doesnt convert to binary, but calculates average
            modelVector = (modelVector / int(math.floor((learningEnd - 6)*self.NumCh )) )
        numSums=(learningEnd - 6)*self.NumCh  #will be needed for noRounding
        return modelVector, timeFeat, timeHDVec #, numSums

class HD_classifier_Symbolization:
    ''' Approach that uses single feature values discretized and then maps them to HD vectors'''

    def __init__(self, NumSymbols,HD_dim, NumCh, device, stringCH, stringLEVEL, roundingType, cuda = True):
        ''' Initialize an HD vectors using the torch library
        NumSymbols: number of discrete values to which each feature is discretized
        HD_dim: dimension of the HD vectors
        NumCh: number of channels of the channels item memory
        stringCH: type of initialization of HD vectors representing channels
        stringLEVEL: type of initialization of HD vectors representing values
        roundingType: type of rounding of vectors to binary vectors
        device: gpu to be used
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''
        self.NumSymbols = NumSymbols
        self.NumCh = NumCh
        self.HD_dim = HD_dim
        self.device = device
        self.roundingType = roundingType

        # CREATING VALUE VECTORS
        if stringLEVEL == 'sandwich':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Sandwich(cuda, device, self.HD_dim, self.NumSymbols)
        elif stringLEVEL=='random':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Random(cuda, device, self.HD_dim, self.NumSymbols)
        elif "scaleNoRand" in stringLEVEL:
            factor=int(stringLEVEL[11:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleNoRand(cuda, device,  self.HD_dim, self.NumSymbols, factor)
        elif "scaleRand" in stringLEVEL:
            factor=int(stringLEVEL[9:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleRand(cuda, device, self.HD_dim, self.NumSymbols,factor)
        else:
            self.proj_mat_FeatVals = func_generateVectorsMemory_Random(cuda, device, self.HD_dim, self.NumSymbols)

        #CREATING CHANNEL VECTORS
        if stringCH == 'sandwich':
            self.proj_mat_channels = func_generateVectorsMemory_Sandwich(cuda, device, self.HD_dim, self.NumCh)
        elif stringCH == 'random':
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, device, self.HD_dim, self.NumCh)
        elif "scaleNoRand" in stringCH:
            factor = int(stringCH[11:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleNoRand(cuda, device, self.HD_dim, self.NumCh,factor)
        elif "scaleRand" in stringCH:
            factor = int(stringCH[9:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleRand(cuda, device, self.HD_dim, self.NumCh, factor)
        else:
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, device, self.HD_dim, self.NumCh)

    def learn_HD_proj(self,EEG, SegSymbParams, EEGfreqBands, SigInfoParams):
        ''' From window of EEG data calculates feature values, maps them to vectors and returns one vector that represents whole EEG window
		'''
        N_channels,learningEnd = EEG.size()
        x = EEG.short() #cutting segments of T symbols
        t0=time.time()
        valuesArr=symbolizeSegment(x,SegSymbParams, EEGfreqBands, SigInfoParams)
        valuesArr[np.isnan(valuesArr)] = 0
        timeFeat=time.time()-t0
        #values = torch.cuda.ShortTensor(valuesArr)
        #values = torch.from_numpy(valuesArr)
        t0=time.time()
        #bindingVector=xor(self.proj_mat_channels,self.proj_mat_FeatVals[:,values.long()])
        bindingVector = xor(self.proj_mat_channels, self.proj_mat_FeatVals[:, valuesArr.astype(int)])
        output_vector=torch.sum(bindingVector,dim=1)
        timeHDVec=time.time()-t0
        #here we broke ties summing an additional HV in case of an even number
        if N_channels%2==0:
            output_vector = torch.add(xor(self.proj_mat_FeatVals[:,1],self.proj_mat_FeatVals[:,2]),1,output_vector)
        if (self.roundingType == 'noRounding'):
            output_vector = (output_vector / self.NumCh )
        else:
            output_vector=(output_vector>int(math.floor(self.NumCh/2))).short()

        return output_vector, timeFeat, timeHDVec

class HD_classifier_MoreFeatures:
    ''' Approach that uses several features and then maps them all to HD vectors'''

    def __init__(self,SigInfoParams, SegSymbParams ,HDParams, cuda = True):
        ''' Initialize an HD vectors using the torch library
        SigInfoParams, SegSymbParams and HDParams contain all parameters needed
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''

        self.NumValues = SegSymbParams.numSegLevels
        self.NumCh = len(SigInfoParams.chToKeep)
        self.NumFeat =HDParams.numFeat
        self.HD_dim = HDParams.D
        self.device = HDParams.CUDAdevice
        self.CHvectTypes=HDParams.vectorTypeCh
        self.LVLvectTypes=HDParams.vectorTypeLevel
        self.FEATvectTypes=HDParams.vectorTypeFeat
        self.roundingType=HDParams.roundingTypeForHDVectors

        #CREATING VALUE LEVEL VECTORS
        if self.LVLvectTypes == 'sandwich':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumValues)
        elif self.LVLvectTypes=='random':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues)
        elif "scaleNoRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[11:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleNoRand(cuda, self.device,  self.HD_dim, self.NumValues, factor)
        elif "scaleRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[9:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumValues,factor)
        else:
            self.proj_mat_FeatVals = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues)

        #CREATING CHANNEL VECTORS
        if self.CHvectTypes == 'sandwich':
            self.proj_mat_channels = func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumCh)
        elif self.CHvectTypes == 'random':
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumCh)
        elif "scaleNoRand" in self.CHvectTypes:
            factor = int(self.CHvectTypes[11:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleNoRand(cuda, self.device, self.HD_dim, self.NumCh,factor)
        elif "scaleRand" in self.CHvectTypes:
            factor = int(self.CHvectTypes[9:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumCh, factor)
        else:
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumCh)

        #CREATING FEATURE VECTORS
        if self.FEATvectTypes == 'sandwich':
            self.proj_mat_features = func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumFeat)
        elif self.FEATvectTypes == 'random':
            self.proj_mat_features = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumFeat)
        elif "scaleNoRand" in self.FEATvectTypes:
            factor = int(self.FEATvectTypes[11:])
            self.proj_mat_features = func_generateVectorsMemory_ScaleNoRand(cuda, self.device, self.HD_dim, self.NumFeat,factor)
        elif "scaleRand" in self.FEATvectTypes:
            factor = int(self.FEATvectTypes[9:])
            self.proj_mat_features = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumFeat, factor)
        else:
            self.proj_mat_features = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumFeat)

        # CREATING COMBINED VECTORS FOR CH ANF FEATURES  - but only random
        if (HDParams.bindingFeatures =='ChFeatCombxVal'):
            self.proj_mat_featuresCh = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumFeat*self.NumCh)

    def learn_HD_proj(self,EEG, SegSymbParams, EEGfreqBands, SigInfoParams, HDParams):
        ''' From window of EEG data calculates feature values, maps them to vectors and returns one vector that represents whole EEG window
        '''
        numCh,learningEnd = EEG.size()
        x = EEG.short()

        t0=time.time()
        #calculate feature valeus for each ch- numFeat x numCh
        if (HDParams.numFeat==3): #symbolization features  # features are in order - amplitude, entropy, CWT
            valuesArr=calculateAllFeatures(x,SegSymbParams, EEGfreqBands, SigInfoParams)
            valuesArr[np.isnan(valuesArr)] = 0
        else: # 54 features from standatrd ML
            valuesArr=np.zeros((HDParams.numFeat,numCh))
            valuesArr0 = np.zeros((HDParams.numFeat, numCh))
            sigArr = x.cpu().numpy()
            for ch in range (numCh):
                valuesArr0[:,ch]=calculateMLfeatures(sigArr[ch,:], SigInfoParams.samplFreq)
                valuesArr[:,ch]=np.floor((SegSymbParams.numSegLevels-1) * (valuesArr0[:,ch]-HDParams.normValuesForFeatures[0,:]) / (HDParams.normValuesForFeatures[1,:]-HDParams.normValuesForFeatures[0,:]))
        #(numFeat, numCh)=valuesArr.shape
        #values = torch.cuda.ShortTensor(valuesArr)
        valuesArr[valuesArr>=SegSymbParams.numSegLevels]=SegSymbParams.numSegLevels-1
        valuesArr[valuesArr <0] = 0
        valuesArr[np.isnan(valuesArr)] = 0
        valuesArr=valuesArr.astype(int)
        timeFeat=time.time()-t0

        t0=time.time()
        if (HDParams.bindingFeatures == 'FeatxVal'):
            Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
            for ch in range(self.NumCh):
                bindingVector = xor(self.proj_mat_features, self.proj_mat_FeatVals[:, valuesArr[:, ch]])
                Chvect_matrix[:,ch] = torch.sum(bindingVector, dim=1)
            output_vector = torch.sum(Chvect_matrix, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                output_vector = (output_vector > int(math.floor(self.NumCh * self.NumFeat / 2))).short()
            else:
                output_vector = (output_vector / (self.NumCh * self.NumFeat))

        elif (HDParams.bindingFeatures =='ChxFeatxVal'):
            Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
            for ch in range(self.NumCh):
                #bind features and their values
                bindingVector = xor(self.proj_mat_features, self.proj_mat_FeatVals[:, valuesArr[:, ch]])
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                Chvect_matrix[:,ch]  = (bindingVectorSum > int(math.floor(self.NumFeat/ 2))).short()
            #binding with  channels vectors
            bindingVector2=xor(Chvect_matrix, self.proj_mat_channels)
            output_vector = torch.sum(bindingVector2, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                output_vector = (output_vector > int(math.floor(self.NumCh/ 2))).short()
            else:
                output_vector = (output_vector / self.NumCh)

        elif (HDParams.bindingFeatures =='FeatxChxVal'):
            Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
            for f in range(self.NumFeat):
                #bind ch and  values of current feature
                bindingVector = xor(self.proj_mat_channels, self.proj_mat_FeatVals[:, valuesArr[f, :]])
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh/ 2))).short()
            #binding with  feature vectors
            bindingVector2=xor(Featvect_matrix, self.proj_mat_features)
            output_vector = torch.sum(bindingVector2, dim=1)
            output_vector = torch.sum(bindingVector2, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                output_vector = (output_vector > int(math.floor(self.NumFeat/ 2))).short()
            else:
                output_vector = (output_vector / self.NumFeat)

        elif (HDParams.bindingFeatures =='ChFeatCombxVal'):
            Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat*self.NumCh).cuda(device=self.device)
            for f in range(self.NumFeat):
                for ch in range(self.NumCh):
                    # bind ch and  values of current feature
                    Featvect_matrix[:,f*self.NumCh+ch] = xor(self.proj_mat_featuresCh[:,f*self.NumCh+ch], self.proj_mat_FeatVals[:, int(valuesArr[f, ch])])
            output_vector = torch.sum(Featvect_matrix, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                output_vector = (output_vector > int(math.floor(self.NumFeat*self.NumCh / 2))).short()
            else:
                output_vector = (output_vector / (self.NumFeat*self.NumCh ))

        elif ('FeatAppend' in HDParams.bindingFeatures):
            Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
            output_vector=torch.zeros(self.HD_dim*self.NumFeat,1).cuda(device=self.device)
            for f in range(self.NumFeat):
                #bind ch and  values of current feature
                bindingVector = xor(self.proj_mat_channels, self.proj_mat_FeatVals[:, valuesArr[f, :]])
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh/ 2))).short()
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh / 2))).short()
                else:
                    Featvect_matrix[:,f]  = (bindingVectorSum /  self.NumCh)
                #apppending
                output_vector[f*self.HD_dim:(f+1)*self.HD_dim,0]=Featvect_matrix[:,f]
        timeHDVec=time.time()-t0

        return output_vector, timeFeat, timeHDVec

class HD_classifier_FFT:
    ''' Approach that uses normalized and discretized FFT spectrum of the window and then maps it to HD vectors'''

    def __init__(self,SigInfoParams, SegSymbParams ,HDParams, cuda = True):
        ''' Initialize an HD vectors using the torch library
        SigInfoParams, SegSymbParams and HDParams contain all parameters needed
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''

        self.NumValues = SegSymbParams.numSegLevels
        self.NumCh = len(SigInfoParams.chToKeep)
        lenFreqsTot = int(SigInfoParams.samplFreq * SegSymbParams.segLenSec / 2)
        lenFreqs = int(HDParams.FFTUpperBound * lenFreqsTot / (SigInfoParams.samplFreq / 2))
        self.NumFreq = lenFreqs
        self.HD_dim = HDParams.D
        self.device = HDParams.CUDAdevice
        self.CHvectTypes=HDParams.vectorTypeCh
        self.LVLvectTypes=HDParams.vectorTypeLevel
        self.FREQvectTypes=HDParams.vectorTypeFreq
        self.roundingType=HDParams.roundingTypeForHDVectors
        self.bindingFFT=HDParams.bindingFFT

        #CREATING VALUE LEVEL VECTORS
        if self.LVLvectTypes == 'sandwich':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumValues)
        elif self.LVLvectTypes=='random':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues)
        elif "scaleNoRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[11:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleNoRand(cuda, self.device,  self.HD_dim, self.NumValues, factor)
        elif "scaleRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[9:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumValues,factor)
        else:
            self.proj_mat_FeatVals = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues)

        #CREATING CHANNEL VECTORS
        if self.CHvectTypes == 'sandwich':
            self.proj_mat_channels = func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumCh)
        elif self.CHvectTypes == 'random':
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumCh)
        elif "scaleNoRand" in self.CHvectTypes:
            factor = int(self.CHvectTypes[11:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleNoRand(cuda, self.device, self.HD_dim, self.NumCh,factor)
        elif "scaleRand" in self.CHvectTypes:
            factor = int(self.CHvectTypes[9:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumCh, factor)
        else:
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumCh)

        #CREATING FREQ VECTORS
        if self.FREQvectTypes == 'sandwich':
            self.proj_mat_freqs = func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumFreq)
        elif self.FREQvectTypes == 'random':
            self.proj_mat_freqs = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumFreq)
        elif "scaleNoRand" in self.FREQvectTypes:
            factor = int(self.FREQvectTypes[11:])
            self.proj_mat_freqs = func_generateVectorsMemory_ScaleNoRand(cuda, self.device, self.HD_dim, self.NumFreq,factor)
        elif "scaleRand" in self.FREQvectTypes:
            factor = int(self.FREQvectTypes[9:])
            self.proj_mat_freqs = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumFreq, factor)
        else:
            self.proj_mat_freqs = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumFreq)

    def learn_HD_proj(self,EEG, SegSymbParams, EEGfreqBands, SigInfoParams, HDParams):
        ''' From window of EEG data calculates feature values, maps them to vectors and returns one vector that represents whole EEG window
        '''
        N_channels,learningEnd = EEG.size()
        xTen = EEG.short()
        x=xTen.cpu().numpy().transpose()
        timeFeat=0; timeHDVec=0;
        if (HDParams.bindingFFT == 'FreqxVal'):
            Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
            for ch in range(self.NumCh):
                t0=time.time()
                #calculateFFT
                (FFTnormMax, a,f)=func_calculateFFT(x[:,ch], SegSymbParams, SigInfoParams, HDParams.FFTUpperBound)
                #discretize them
                FFTnormMaxDisct=(FFTnormMax/(1/(SegSymbParams.numSegLevels-1))).astype(int)
                FFTnormMaxDisct[FFTnormMaxDisct>(SegSymbParams.numSegLevels-1)]=SegSymbParams.numSegLevels-1
                timeFeat=timeFeat+(time.time()-t0)
                t0=time.time()
                bindingVector = xor(self.proj_mat_freqs, self.proj_mat_FeatVals[:, FFTnormMaxDisct])
                Chvect_matrix[:,ch] = torch.sum(bindingVector, dim=1)
                if (self.roundingType == 'inSteps'):
                    Chvect_matrix[:,ch]  = (Chvect_matrix[:,ch]  > int(math.floor(self.NumFreq / 2))).short()
                timeHDVec=timeHDVec+(time.time()-t0)

            t0 = time.time()
            output_vector = torch.sum(Chvect_matrix, dim=1)
            if (self.roundingType == 'inSteps'):
                output_vector = (output_vector > int(math.floor(self.NumCh / 2))).short()
            elif (self.roundingType == 'onlyOne'):
                output_vector = (output_vector > int(math.floor(self.NumCh *self.NumFreq/ 2))).short()
            elif (self.roundingType=='noRounding'):
                output_vector = (output_vector / int(math.floor(self.NumCh * self.NumFreq )))
            timeHDVec = timeHDVec + (time.time() - t0)

        elif (HDParams.bindingFFT =='ChxFreqxVal'):
            Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
            for ch in range(self.NumCh):
                t0 = time.time()
                # calculateFFT
                (FFTnormMax, a, f) = func_calculateFFT(x[:, ch], SegSymbParams, SigInfoParams, HDParams.FFTUpperBound)
                # discretize them
                FFTnormMaxDisct=(FFTnormMax/(1/(SegSymbParams.numSegLevels-1))).astype(int)
                timeFeat = timeFeat + (time.time() - t0)
                t0 = time.time()
                #bind features and their values
                bindingVector = xor(self.proj_mat_freqs, self.proj_mat_FeatVals[:,  FFTnormMaxDisct])
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                Chvect_matrix[:,ch]  = (bindingVectorSum > int(math.floor(self.NumFreq/ 2))).short()
                timeHDVec = timeHDVec + (time.time() - t0)
            t0 = time.time()
            #binding with  channels vectors
            bindingVector2=xor(Chvect_matrix, self.proj_mat_channels)
            output_vector = torch.sum(bindingVector2, dim=1)
            if (self.roundingType=='noRounding'):
                output_vector = (output_vector / self.NumCh)
            else:
                output_vector = (output_vector > int(math.floor(self.NumCh/ 2))).short()
            timeHDVec = timeHDVec + (time.time() - t0)

        elif (HDParams.bindingFFT =='PermChFreqxVal'):
            ChFreqvect_matrix = torch.zeros(self.HD_dim, self.NumCh*self.NumFreq).cuda(device=self.device)
            i=-1
            for ch in range(self.NumCh):
                t0 = time.time()
                i=i+1
                # calculateFFT
                (FFTnormMax, a, f) = func_calculateFFT(x[:, ch], SegSymbParams, SigInfoParams, HDParams.FFTUpperBound)
                # discretize them
                FFTnormMaxDisct=(FFTnormMax/(1/(SegSymbParams.numSegLevels-1))).astype(int)
                timeFeat = timeFeat + (time.time() - t0)
                t0 = time.time()
                for fIndx in range(len(f)):
                    permutedChVec=rotateVec(self.proj_mat_channels[:,ch],fIndx)
                    ChFreqvect_matrix[:,i] = xor(permutedChVec, self.proj_mat_FeatVals[:, FFTnormMaxDisct[fIndx]])
                timeHDVec = timeHDVec + (time.time() - t0)
            t0 = time.time()
            output_vector = torch.sum(ChFreqvect_matrix, dim=1)
            if (self.roundingType=='noRounding'):
                output_vector = (output_vector / self.NumCh)
            else:
                output_vector = (output_vector > int(math.floor(self.NumCh/ 2))).short()
            timeHDVec = timeHDVec + (time.time() - t0)
        return output_vector, timeFeat, timeHDVec

class HD_classifier_RawAmpl:
    ''' Approach that uses discretized raw signal values and then maps them to HD vectors'''

    def __init__(self,SigInfoParams, SegSymbParams ,HDParams, cuda = True):
        ''' Initialize an HD vectors using the torch library
        SigInfoParams, SegSymbParams and HDParams contain all parameters needed
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''

        self.NumValues = SegSymbParams.numSegLevels
        self.NumCh = len(SigInfoParams.chToKeep)
        self.HD_dim = HDParams.D
        self.device = HDParams.CUDAdevice
        self.CHvectTypes=HDParams.vectorTypeCh
        self.LVLvectTypes=HDParams.vectorTypeLevel
        self.roundingType=HDParams.roundingTypeForHDVectors
        self.bindingRawAmpl=HDParams.bindingRawAmpl

        #CREATING VALUE LEVEL VECTORS
        if self.LVLvectTypes == 'sandwich':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumValues)
        elif self.LVLvectTypes=='random':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues)
        elif "scaleNoRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[11:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleNoRand(cuda, self.device,  self.HD_dim, self.NumValues, factor)
        elif "scaleRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[9:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumValues,factor)
        else:
            self.proj_mat_FeatVals = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues)

        #CREATING CHANNEL VECTORS
        if self.CHvectTypes == 'sandwich':
            self.proj_mat_channels = func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumCh)
        elif self.CHvectTypes == 'random':
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumCh)
        elif "scaleNoRand" in self.CHvectTypes:
            factor = int(self.CHvectTypes[11:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleNoRand(cuda, self.device, self.HD_dim, self.NumCh,factor)
        elif "scaleRand" in self.CHvectTypes:
            factor = int(self.CHvectTypes[9:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumCh, factor)
        else:
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumCh)


    def learn_HD_proj(self,EEG, SegSymbParams, EEGfreqBands, SigInfoParams, HDParams):
        ''' From window of EEG data calculates feature values, maps them to vectors and returns one vector that represents whole EEG window
        '''
        numCh,numSampl = EEG.size()
        xTen = EEG.short() #cutting segments of T symbols
        x=xTen.cpu().numpy().transpose()
        timeFeat = 0;timeHDVec = 0;
        # if  summing up for all ch  XIR(FeatVecxValVec)
        if (HDParams.bindingRawAmpl == 'ValxCh'):
            Samplvect_matrix = torch.zeros(self.HD_dim, numSampl).cuda(device=self.device)
            for s in range(numSampl):
                t0=time.time()
                (normAmplVals,v)=func_normAmplitudeValsSamplBySampl(x[s,:],SegSymbParams)
                timeFeat=timeFeat+(time.time()-t0)
                t0 = time.time()
                bindingVector = xor(self.proj_mat_channels, self.proj_mat_FeatVals[:, normAmplVals])
                Samplvect_matrix[:,s] = torch.sum(bindingVector, dim=1)
                if (self.roundingType == 'inSteps'):
                    Samplvect_matrix[:,s]  = (Samplvect_matrix[:,s]  > int(math.floor(self.NumCh / 2))).short()
                timeHDVec = timeHDVec + (time.time() - t0)
            t0 = time.time()
            output_vector = torch.sum(Samplvect_matrix, dim=1)
            if (self.roundingType == 'inSteps'):
                output_vector = (output_vector > int(math.floor(numSampl/ 2))).short()
            elif (self.roundingType == 'onlyOne'):
                output_vector = (output_vector > int(math.floor(self.NumCh *numSampl/ 2))).short()
            elif (self.roundingType=='noRounding'):
                output_vector = (output_vector / int(math.floor(self.NumCh * numSampl )))
            timeHDVec = timeHDVec + (time.time() - t0)

        elif (HDParams.bindingRawAmpl =='PermValSamplxCh'):
            Samplvect_matrix = torch.zeros(self.HD_dim, numSampl).cuda(device=self.device)
            for s in range(numSampl):
                t0 = time.time()
                (normAmplVals,v) = func_normAmplitudeValsSamplBySampl(x[s, :],SegSymbParams)
                timeFeat = timeFeat + (time.time() - t0)
                t0 = time.time()
                permutedAmplVec = rotateVec(self.proj_mat_FeatVals[:, normAmplVals], s)
                bindingVector = xor(self.proj_mat_channels, permutedAmplVec)
                Samplvect_matrix[:, s] = torch.sum(bindingVector, dim=1)
                if (self.roundingType == 'inSteps'):
                    Samplvect_matrix[:, s] = (Samplvect_matrix[:, s] > int(math.floor(self.NumCh / 2))).short()
                timeHDVec = timeHDVec + (time.time() - t0)
            t0 = time.time()
            output_vector = torch.sum(Samplvect_matrix, dim=1)
            if (self.roundingType == 'inSteps'):
                output_vector = (output_vector > int(math.floor(numSampl / 2))).short()
            elif (self.roundingType == 'onlyOne'):
                output_vector = (output_vector > int(math.floor(self.NumCh * numSampl / 2))).short()
            elif (self.roundingType=='noRounding'):
                output_vector = (output_vector / int(math.floor(self.NumCh * numSampl )))
            timeHDVec = timeHDVec + (time.time() - t0)

        elif (HDParams.bindingRawAmpl =='PermValSampl'):
            Samplvect_matrix = torch.zeros(self.HD_dim, numSampl).cuda(device=self.device)
            for s in range(numSampl):
                t0 = time.time()
                (normAmplVals,v) = func_normAmplitudeValsSamplBySampl(x[s, :],SegSymbParams)
                timeFeat = timeFeat + (time.time() - t0)
                t0 = time.time()
                permutedAmplVec = rotateVec(self.proj_mat_FeatVals[:, normAmplVals], s)
                Samplvect_matrix[:, s] = torch.sum(permutedAmplVec, dim=1)
                if (self.roundingType == 'inSteps'):
                    Samplvect_matrix[:, s] = (Samplvect_matrix[:, s] > int(math.floor(self.NumCh / 2))).short()
                timeHDVec = timeHDVec + (time.time() - t0)
            t0 = time.time()
            output_vector = torch.sum(Samplvect_matrix, dim=1)
            if (self.roundingType == 'inSteps'):
                output_vector = (output_vector > int(math.floor(numSampl / 2))).short()
            elif (self.roundingType == 'onlyOne'):
                output_vector = (output_vector > int(math.floor(self.NumCh * numSampl / 2))).short()
            elif (self.roundingType=='noRounding'):
                output_vector = (output_vector / int(math.floor(self.NumCh * numSampl )))
            timeHDVec = timeHDVec + (time.time() - t0)
        # if (HDParams.roundingTypeForHDVectors == 'noRounding'):
        # 	output_vector = (output_vector / numSampl).short()
        return output_vector, timeFeat, timeHDVec

def func_generateVectorsMemory_Random(cuda, device, HD_dim, Num_vect):
    ''' function to generate matrix of HD vectors using random method
        random - each vector is independantly randomly generated
    '''
    if cuda:
        vect_matrix = torch.randn(HD_dim, Num_vect).cuda(device=device)
    else:
        vect_matrix= torch.randn(HD_dim, Num_vect)
    vect_matrix[vect_matrix > 0] = 1
    vect_matrix[vect_matrix <= 0] = 0
    return(vect_matrix)

def func_generateVectorsMemory_Sandwich(cuda, device, HD_dim, Num_vect):
    ''' function to generate matrix of HD vectors using sandwich method
    sandwich - every two neighbouring vectors  have half of the vector the same, but the rest of the vector is random
    '''

    if cuda:
        vect_matrix = torch.zeros(HD_dim, Num_vect).cuda(device=device)
    else:
        vect_matrix = torch.zeros(HD_dim, Num_vect)
    for i in range(Num_vect):
        if i % 2 == 0:
            vect_matrix[:, i] = torch.randn(HD_dim).cuda(
                device=device)  # torch.randn(self.HD_dim, 1).cuda(device = device)
            vect_matrix[vect_matrix > 0] = 1
            vect_matrix[vect_matrix <= 0] = 0
    for i in range(Num_vect - 1):
        if i % 2 == 1:
            vect_matrix[0:int(HD_dim / 2), i] = vect_matrix[0:int(HD_dim / 2), i - 1]
            vect_matrix[int(HD_dim / 2):HD_dim, i] = vect_matrix[int(HD_dim / 2):HD_dim, i + 1]
    vect_matrix[0:int(HD_dim / 2), Num_vect - 1] = vect_matrix[0:int(HD_dim / 2), Num_vect - 2]
    if cuda:
        vect_matrix[int(HD_dim / 2):HD_dim, Num_vect - 1] = torch.randn(int(HD_dim / 2)).cuda(
            device=device)  # torch.randn(int(HD_dim/2), 1).cuda(device = device)
    else:
        vect_matrix[int(HD_dim / 2):HD_dim, Num_vect - 1] = torch.randn(
            int(HD_dim / 2))  # torch.randn(int(HD_dim/2), 1)
    vect_matrix[vect_matrix > 0] = 1
    vect_matrix[vect_matrix <= 0] = 0
    return (vect_matrix)

def func_generateVectorsMemory_ScaleRand(cuda, device, HD_dim, Num_vect, scaleFact):
    ''' function to generate matrix of HD vectors using scale method with bits of randomization
    scaleRand - every next vector is created by randomly flipping D/(numVec*scaleFact) elements - this way the further values vectors represent are, the less similar are vectors
    '''
    numValToFlip=floor(HD_dim/(scaleFact*Num_vect))

    #initialize vectors
    if cuda:
        vect_matrix = torch.zeros(HD_dim,Num_vect).cuda(device=device)
    else:
        vect_matrix = torch.zeros(HD_dim, Num_vect)
    #generate firs one as random
    vect_matrix[:, 0] = torch.randn(HD_dim).cuda(device=device)  # torch.randn(self.HD_dim, 1).cuda(device = device)
    vect_matrix[vect_matrix > 0] = 1
    vect_matrix[vect_matrix <= 0] = 0
    #iteratively the further they are flip more bits
    for i in range(1,Num_vect):
        vect_matrix[:, i]=vect_matrix[:, i-1]
        #choose random positions to flip
        posToFlip=random.sample(range(1,HD_dim),numValToFlip)
        vect_matrix[posToFlip,i] = vect_matrix[posToFlip,i]*(-1)+1

        # #test if distance is increasing
        # modelVector0 = vect_matrix[:, 0].cpu().numpy()
        # modelVector1 = vect_matrix[:,i-1].cpu().numpy()
        # modelVector2 = vect_matrix[:,i].cpu().numpy()
        # print('Vector difference for i-th:',i, ' is : ', np.sum(abs(modelVector2 - modelVector0)), ' &', np.sum(abs(modelVector2 - modelVector1)) )

    return(vect_matrix)

def func_generateVectorsMemory_ScaleNoRand(cuda, device, HD_dim, Num_vect, scaleFact):
    ''' function to generate matrix of HD vectors  using scale method with no randomization
    scaleNoRand - same idea as scaleRand, just  d=D/(numVec*scaleFact) bits are taken in order (e.g. from i-th to i+d bit) and not randomly
    '''

    numValToFlip = floor(HD_dim / (scaleFact * Num_vect))

    # initialize vectors
    if cuda:
        vect_matrix = torch.zeros(HD_dim, Num_vect).cuda(device=device)
    else:
        vect_matrix = torch.zeros(HD_dim, Num_vect)
    # generate firs one as random
    vect_matrix[:, 0] = torch.randn(HD_dim).cuda(device=device)  # torch.randn(self.HD_dim, 1).cuda(device = device)
    vect_matrix[vect_matrix > 0] = 1
    vect_matrix[vect_matrix <= 0] = 0
    # iteratively the further they are flip more bits
    for i in range(1, Num_vect):
        vect_matrix[:, i] = vect_matrix[:, 0]
        vect_matrix[0: i * numValToFlip, i] = flipVectorValues(vect_matrix[0: i * numValToFlip, 0])
    # #test if distance is increasing
    # modelVector1 = vect_matrix[:,0].cpu().numpy()
    # modelVector2 = vect_matrix[:,i].cpu().numpy()
    # print('Vector difference for i-th:',i, ' is : ', np.sum(abs(modelVector1 - modelVector2)))

    return (vect_matrix)


def flipVectorValues(vector):
    '''turns 0 into 1 and 1 into 0'''
    vectorFliped=(vector*(-1)+1)
    return(vectorFliped)

def xor( vec_a, vec_b):
	''' xor between vec_a and vec_b'''
	vec_c = (torch.add(vec_a, vec_b) == 1).short()  # xor
	return vec_c

def ham_dist( vec_a, vec_b, D):
	''' calculate relative hamming distance'''
	#vec_c = xor(vec_a, vec_b) # this was used before but only works for binary vectors
	vec_c= torch.abs(torch.sub(vec_a,vec_b))
	rel_dist = torch.sum(vec_c) / float(D)
	return rel_dist

def ham_dist_arr( vec_a, vec_b, D):
	''' calculate relative hamming distance fur for np array'''
	vec_c= np.abs(vec_a-vec_b)
	rel_dist = np.sum(vec_c) / float(D)
	return rel_dist

def cos_dist( vec_a, vec_b):
	''' calculate cosine distance of two vectors'''
	# cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
	# output = cos(vec_a.short(), vec_b.short())
	vA=np.squeeze(vec_a.cpu().numpy())
	vB=np.squeeze(vec_b.cpu().numpy())
	output=np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))
	# if (output!=1.0):
	# 	print(output)
	#outTensor=torch.from_numpy(output).float().cuda()
	outTensor=torch.tensor(1.0-output) #because we use latter that if value is 0 then vectors are the same
	return outTensor

def rotateVec(vec, numRot):
	'''shift vector for numRot bits '''
	outVec=torch.roll(vec,-numRot,0)
	return outVec


def func_trainAndTest_personalized(SigInfoParams, SegSymbParams, GeneralParams, HDParams, EEGfreqBands ,folderIn, folderOut):
    ''' function that loads data, per patients and for each patient does leave one seizure out CV
    trains HD model on all Seizure files but one and then tests on the one left out
    saves predictions and true labels for each file '''

    # creating baseline HD vectors - the same initialized vectors for all CVs
    if (SegSymbParams.symbolType == 'LBP'):
        model = HD_classifier_LBP(HDParams.totalNumberBP, HDParams.D, len(SigInfoParams.chToKeep), HDParams.LBPlen,
                                  HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                  HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'Entropy' or SegSymbParams.symbolType == 'Amplitude'):
        model = HD_classifier_Symbolization(SegSymbParams.numSegLevels, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'CWT'):
        model = HD_classifier_Symbolization(SegSymbParams.CWTlevel, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'CWT&Entropy' or SegSymbParams.symbolType == 'Amplitude&Entropy' or SegSymbParams.symbolType == 'Amplitude&CWT'):
        numDiffValues = 10 * 10  # only if num seg for each is <10
        model = HD_classifier_Symbolization(numDiffValues, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice,
                                            HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'Amplitude&Entropy&CWT'):
        numDiffValues = 10 * 10 * 10  # only if num seg for each is <10
        model = HD_classifier_Symbolization(numDiffValues, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice,
                                            HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'AllFeatures' or SegSymbParams.symbolType == 'StandardMLFeatures'):
        model = HD_classifier_MoreFeatures(SigInfoParams, SegSymbParams, HDParams)
    elif (SegSymbParams.symbolType == 'FFT'):
        model = HD_classifier_FFT(SigInfoParams, SegSymbParams, HDParams)
    elif (SegSymbParams.symbolType == 'RawAmpl'):
        model = HD_classifier_RawAmpl(SigInfoParams, SegSymbParams, HDParams)

    # PERFORMING TRAINING AND TESTING
    for patIndx, pat in enumerate(GeneralParams.patients):
        numFiles = len(np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv')))
        print('-- Patient:', pat, 'NumSeizures:', numFiles)

        for cv in range(numFiles):
            # creates list of files to train and test on
            filesToTrainOn = []
            for fIndx, fileName in enumerate(np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv'))):
                if (fIndx != cv):
                    filesToTrainOn.append(fileName)
                else:
                    filesToTestOn = list(fileName.split(" "))

            # trains model - creates seizure and nonSeizure model vectors
            (modelVector_Seiz, modelVector_NonSeiz) = train_HDmodel_v2(model, filesToTrainOn, HDParams, SigInfoParams, SegSymbParams, EEGfreqBands)

            # goes through testing file and predicts labels for it and saves them
            test_HDmodel_v2(model, filesToTestOn, folderOut, modelVector_Seiz, modelVector_NonSeiz, SigInfoParams,
                            SegSymbParams, GeneralParams, HDParams, EEGfreqBands, 'Personalized', 0)

def func_trainAndTest_Generalized(SigInfoParams, SegSymbParams, GeneralParams, HDParams, EEGfreqBands, folderIn, folderOut):
    ''' function that performs generalized training nad testing
    there is predefined sets of subjects for each CV - e.g. 4 subject to test on, and the rest (20) to train on so there is 6 CV
    trains HD model on all  files of people in train set and then tests on  all files of subjects in test set
    saves predictions and true labels for each file '''

    # creating baseline HD vectors
    if (SegSymbParams.symbolType == 'LBP'):
        model = HD_classifier_LBP(HDParams.totalNumberBP, HDParams.D, len(SigInfoParams.chToKeep), HDParams.LBPlen,
                                  HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                  HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'Entropy' or SegSymbParams.symbolType == 'Amplitude'):
        model = HD_classifier_Symbolization(SegSymbParams.numSegLevels, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'CWT'):
        model = HD_classifier_Symbolization(SegSymbParams.CWTlevel, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (
            SegSymbParams.symbolType == 'CWT&Entropy' or SegSymbParams.symbolType == 'Amplitude&Entropy' or SegSymbParams.symbolType == 'Amplitude&CWT'):
        numDiffValues = 10 * 10  # only if num seg for each is <10
        model = HD_classifier_Symbolization(numDiffValues, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'Amplitude&Entropy&CWT'):
        numDiffValues = 10 * 10 * 10  # only if num seg for each is <10
        model = HD_classifier_Symbolization(numDiffValues, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'AllFeatures' or SegSymbParams.symbolType == 'StandardMLFeatures'):
        model = HD_classifier_MoreFeatures(SigInfoParams, SegSymbParams, HDParams)
    elif (SegSymbParams.symbolType == 'FFT'):
        model = HD_classifier_FFT(SigInfoParams, SegSymbParams, HDParams)
    elif (SegSymbParams.symbolType == 'RawAmpl'):
        model = HD_classifier_RawAmpl(SigInfoParams, SegSymbParams, HDParams)

    numCVs = len(GeneralParams.CViterations_testSubj)
    performancesAll_AllCV = np.zeros((len(GeneralParams.patients), 12))
    for cv in range(numCVs):
        print('-- CV num:', cv)
        # CREATE LIST OF FILES TO TRAIN ON
        filesToTrainOn = []
        for patIndx in range(len(GeneralParams.patients)):
            if patIndx not in GeneralParams.CViterations_testSubj[cv]:
                pat = GeneralParams.patients[patIndx]
                # numFiles = len(np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv')))
                filesArray = np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv'))
                # filesToTrainOn.extend(filesArray.tolist())
                filesToTrainOn = filesToTrainOn + filesArray.tolist()
            # print('-- TRAIN Patient:', pat, 'NumSeizures:', len(filesArray))

        # CREATE LIST OF FILES TO TEST ON
        filesToTestOn = []
        for patIndx in GeneralParams.CViterations_testSubj[cv]:
            pat = GeneralParams.patients[patIndx]
            # numFiles = len(np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv')))
            filesArray = np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv'))
            # filesToTestOn.extend(filesArray.tolist())
            filesToTestOn = filesToTestOn + filesArray.tolist()
        # print('--TEST Patient:', pat, 'NumSeizures:', len(filesArray))

        # trains model - creates seizure and nonSeizure model vectors
        (modelVector_Seiz, modelVector_NonSeiz) = train_HDmodel_v2(model, filesToTrainOn, HDParams, SigInfoParams, SegSymbParams, EEGfreqBands)

        # goes through testing files and predicts and saves predictions for them
        test_HDmodel_v2(model, filesToTestOn, folderOut, modelVector_Seiz, modelVector_NonSeiz, SigInfoParams,
                        SegSymbParams, GeneralParams, HDParams, EEGfreqBands, 'Generalized', cv)


def train_HDmodel_v2(model,filesToTrainOn, HDParams,SigInfoParams, SegSymbParams, EEGfreqBands):
    ''' function that trains on all files given
    loads files and then base on label trains for seizure and after for nonseizure
    in the end returns two model vectors, one for seizure and one for nonSeizure
    there are few types of normalizations of vector that are possible
    '''

    torch.manual_seed(1)
    numFiles=len(filesToTrainOn)
    firstFile=1
    for fileIn in filesToTrainOn:
        # print(fileIn)
        pom, fileName1 = os.path.split(fileIn)
        fileName2 = os.path.splitext(fileName1)[0]
        patientNum = fileName2[3:5]

        # reading data
        reader = csv.reader(open(fileIn, "r"))
        data0 = list(reader)
        data = np.array(data0).astype("float")
        # separating to data and labels
        X = data[:, SigInfoParams.chToKeep]
        y = data[:, -1]

        # training separately for seizure and nonSeizure
        seizIndx = np.where(y == 1)
        EEGSeiz0 = np.squeeze(X[seizIndx, :])
        EEGSeiz = torch.from_numpy(np.array(EEGSeiz0)).cuda().t()
        (seizVect, timeFeatTotal, timeHDVecTotal) = learn_HDproj_wholeSignal(model,EEGSeiz,SegSymbParams, EEGfreqBands, SigInfoParams,HDParams)

        nonSeizIndx = np.where(y == 0)
        EEGNonSeiz0 = np.squeeze(X[nonSeizIndx, :])
        EEGNonSeiz = torch.from_numpy(np.array(EEGNonSeiz0)).cuda().t()
        (nonSeizVect, timeFeatTotal, timeHDVecTotal) = learn_HDproj_wholeSignal(model,EEGNonSeiz,SegSymbParams, EEGfreqBands, SigInfoParams,HDParams)

        # summing up all vectors coming fromt he same class
        if firstFile == 1:
            modelVector_Seiz = seizVect
            modelVector_NonSeiz = nonSeizVect
            firstFile=0
        else:
            modelVector_Seiz = torch.add(modelVector_Seiz, seizVect)
            modelVector_NonSeiz = torch.add(modelVector_NonSeiz, nonSeizVect)

    # make vectors binary after summation (containing only 0 or 1)
    numFiles=len( filesToTrainOn)
    if (HDParams.roundingTypeForHDVectors!='noRounding'):
        modelVector_Seiz = (modelVector_Seiz > (numFiles - 1) / 2).short()  # put finally to 1 only is more then half of files had 1 on that place
        modelVector_NonSeiz = (modelVector_NonSeiz > (numFiles - 1) / 2).short()  # put finally to 1 only is more then half of files had 1 on that place
    else:
        modelVector_Seiz = (modelVector_Seiz / numFiles)
        modelVector_NonSeiz = (modelVector_NonSeiz / numFiles)

    return(modelVector_Seiz, modelVector_NonSeiz)

def learn_HDproj_wholeSignal(model,EEG, SegSymbParams, EEGfreqBands, SigInfoParams, HDParams):
    ''' function that when given EEG data discretizes it into small windows and
    for each window calls funciton to calucate HD vector representing this window
    in the end summs up (and rounds) vectors from all windows to get model vector that represents this class
    '''

    modelVector = torch.cuda.ShortTensor(1,1).zero_()
    N_channels,lenSig = EEG.size()
    segLenIndx = int(SegSymbParams.segLenSec * SigInfoParams.samplFreq)  # length of EEG segments in samples
    slidWindStepIndx = int(SegSymbParams.slidWindStepSec * SigInfoParams.samplFreq)  # step of slidin window to extract segments in samples
    index = np.arange(0, lenSig - segLenIndx, slidWindStepIndx).astype(int)
    timeFeatTotal=0; timeHDVecTotal=0;

    for i in range(len(index)):
        # get HD vector representing this discretized window
        if (SegSymbParams.symbolType == 'LBP'):
            (temp,timeFeat, timeHDVec) = model.learn_HD_proj(EEG[:,index[i]:index[i]+segLenIndx])
            #totNumSums=totNumSums+numSums
        elif (SegSymbParams.symbolType == 'AllFeatures' or SegSymbParams.symbolType == 'StandardMLFeatures'):
            (temp,timeFeat, timeHDVec)= model.learn_HD_proj(EEG[:, index[i]:index[i] + segLenIndx], SegSymbParams, EEGfreqBands, SigInfoParams, HDParams)
        elif (SegSymbParams.symbolType == 'FFT'):
            (temp,timeFeat, timeHDVec) = model.learn_HD_proj(EEG[:, index[i]:index[i] + segLenIndx], SegSymbParams, EEGfreqBands, SigInfoParams, HDParams)
        elif (SegSymbParams.symbolType == 'RawAmpl'):
            (temp,timeFeat, timeHDVec) = model.learn_HD_proj(EEG[:, index[i]:index[i] + segLenIndx], SegSymbParams, EEGfreqBands, SigInfoParams, HDParams)
        else:
            (temp,timeFeat, timeHDVec) = model.learn_HD_proj(EEG[:, index[i]:index[i] + segLenIndx],SegSymbParams, EEGfreqBands, SigInfoParams)
        #summing up all vectors
        modelVector=torch.add(modelVector,1,temp)
        #measuring time
        timeFeatTotal=timeFeatTotal+timeFeat
        timeHDVecTotal=timeHDVecTotal+timeHDVec
    # calculating final model vetor - either binary or float values
    if (HDParams.roundingTypeForHDVectors!='noRounding'):
        modelVector = (modelVector > index.size/2).short() #where more then half of the time was 1 puts to 1
    else:
        modelVector = (modelVector / index.size)

    return modelVector, timeFeatTotal, timeHDVecTotal

def test_HDmodel_v2(model, filesToTestOn , folderOut, modelVector_Seiz, modelVector_NonSeiz, SigInfoParams, SegSymbParams, GeneralParams, HDParams, EEGfreqBands, persGenType,  CVindx):
    '''function that for each file in test set calculates predictions for all dicsrete windows
    and saves predictions together with true labels in one file for each input file
    this files can latter be used to perform different postprocessing of labels, plot predictions and also calculate performance'''

    for fileIndx, fileIn in enumerate(filesToTestOn):
        # print(fileIn)
        pom, fileName1 = os.path.split(fileIn)
        fileName2 = os.path.splitext(fileName1)[0]
        pat = str(fileName2[3:5].zfill(2))

        # reading data
        try:
            reader = csv.reader(open(fileIn, "r"))
        except:
            print('Problem with reading file:', fileIn, filesToTestOn)
        data0 = list(reader)
        data = np.array(data0).astype("float")
        # separating to data and labels
        X = data[:, SigInfoParams.chToKeep]
        y = data[:, -1]
        (lenSig, numCh) = X.shape

        # prepare to do prediction every 0.5s
        segLenIndx = int(SegSymbParams.segLenSec * SigInfoParams.samplFreq)  # length of EEG segments in samples
        slidWindStepIndx = int(SegSymbParams.slidWindStepSec * SigInfoParams.samplFreq)  # step of slidin window to extract segments in samples
        index = np.arange(0, lenSig - segLenIndx, slidWindStepIndx)
        # index = np.arange(0, lenSig -int(SigInfoParams.samplFreq/2),int(SigInfoParams.samplFreq/2))
        numSeg = len(index)
        distanceVectors_Seiz = torch.zeros(1, numSeg).cuda()
        distanceVectors_NonSeiz = torch.zeros(1, numSeg).cuda()
        predictions = torch.zeros(1, numSeg).cuda()
        # cutout data
        EEGtest = torch.from_numpy(np.array(np.squeeze(X))).cuda().t()

        # calculate prediction for every 0.5s
        for i in range(len(index)):
            if (SegSymbParams.symbolType == 'LBP'):
                (temp, timeFeat, timeHDVec) = model.learn_HD_proj(EEGtest[:, index[i]:index[i] + segLenIndx])
            elif (SegSymbParams.symbolType == 'AllFeatures' or SegSymbParams.symbolType == 'StandardMLFeatures'):
                (temp, timeFeat, timeHDVec)  = model.learn_HD_proj(EEGtest[:, index[i]:index[i] + segLenIndx], SegSymbParams, EEGfreqBands,SigInfoParams, HDParams)
            elif (SegSymbParams.symbolType == 'FFT'):
                (temp, timeFeat, timeHDVec)  = model.learn_HD_proj(EEGtest[:, index[i]:index[i] + segLenIndx], SegSymbParams, EEGfreqBands,SigInfoParams, HDParams)
            elif (SegSymbParams.symbolType == 'RawAmpl'):
                (temp, timeFeat, timeHDVec)  = model.learn_HD_proj(EEGtest[:, index[i]:index[i] + segLenIndx], SegSymbParams, EEGfreqBands,SigInfoParams, HDParams)
            else:
                (temp, timeFeat, timeHDVec)  = model.learn_HD_proj(EEGtest[:, index[i]:index[i] + segLenIndx], SegSymbParams, EEGfreqBands, SigInfoParams)
            [distanceVectors_Seiz[0, i], distanceVectors_NonSeiz[0, i], predictions[0, i]] = HDpredict(temp,modelVector_Seiz, modelVector_NonSeiz,HDParams.D,HDParams.similarityType)
            # print('Patient:' + str(pat) + 'Seizure: ' + str(fileIndx) + '; half second: ' + str(i) + '; PRED: ', predictions[0, i])

        # save prediction and true labels
        predictionsArr = predictions[0, :].cpu().numpy()
        segmentedLabels = segmentLabels(y, SegSymbParams, SigInfoParams)
        # postprocessing labels
        seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
        seizureStablePercToTest = GeneralParams.seizureStablePercToTest
        distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
        (predictionsArrSmooth_step2, predictionsArrSmooth_step1) = smoothenLabels(predictionsArr, seizureStableLenToTestIndx,seizureStablePercToTest, distanceBetweenSeizuresIndx)  # step1 is afte  movign average and step2 is after merging

        # saving predictions and labels
        if (persGenType=='Generalized'):
            outputName = folderOut + '/'+fileName2+ '_CV'+str(CVindx)+'_True&PredLabels.csv'
        else:
            outputName = folderOut + '/' + fileName2 + '_True&PredLabels.csv'
        dataToSave = np.vstack((predictionsArr, predictionsArrSmooth_step1, predictionsArrSmooth_step2, segmentedLabels)).transpose()
        np.savetxt(outputName, dataToSave, delimiter=",")

def HDpredict(testVector, Ictalprot, Interictalprot, D, simType):
    ''' function that based on given vector and two model vectors measures distance and gives prediction '''

    if (simType == 'hamming'):
        distanceVectorsS = ham_dist(testVector, Ictalprot, D)
        distanceVectorsnS = ham_dist(testVector, Interictalprot, D)
    elif (simType == 'cosine'):
        distanceVectorsS = cos_dist(testVector, Ictalprot)
        distanceVectorsnS = cos_dist(testVector, Interictalprot)

    try:
        if distanceVectorsS > distanceVectorsnS:
            prediction = 0
        else:
            prediction = 1
    except:
        print('CUDA error 2')
    return distanceVectorsS, distanceVectorsnS, prediction

def calculate_HDcalcTime_onOneWindow(folderIn, SegSymbParams, SigInfoParams,HDParams, GeneralParams, EEGfreqBands):
    '''function that performs encoding to HD vectors and measures time needed for that
    in the end calculates average times per discretized window for different files and subjects and calculates average time '''

    # #Initiallize model depending on approach
    if (SegSymbParams.symbolType == 'LBP'):
        model = HD_classifier_LBP(HDParams.totalNumberBP, HDParams.D, len(SigInfoParams.chToKeep), HDParams.LBPlen,
                                  HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                  HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'Entropy' or SegSymbParams.symbolType == 'Amplitude'):
        model = HD_classifier_Symbolization(SegSymbParams.numSegLevels, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'CWT'):
        model = HD_classifier_Symbolization(SegSymbParams.CWTlevel, HDParams.D, len(SigInfoParams.chToKeep),
                                            HDParams.CUDAdevice, HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'CWT&Entropy' or SegSymbParams.symbolType == 'Amplitude&Entropy' or SegSymbParams.symbolType == 'Amplitude&CWT'):
        numDiffValues = 10 * 10  # only if num seg for each is <10
        model = HD_classifier_Symbolization(numDiffValues, HDParams.D, len(SigInfoParams.chToKeep), HDParams.CUDAdevice,
                                            HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'Amplitude&Entropy&CWT'):
        numDiffValues = 10 * 10 * 10  # only if num seg for each is <10
        model = HD_classifier_Symbolization(numDiffValues, HDParams.D, len(SigInfoParams.chToKeep), HDParams.CUDAdevice,
                                            HDParams.vectorTypeCh, HDParams.vectorTypeLevel,
                                            HDParams.roundingTypeForHDVectors)
    elif (SegSymbParams.symbolType == 'AllFeatures' or SegSymbParams.symbolType == 'StandardMLFeatures'):
        model = HD_classifier_MoreFeatures(SigInfoParams, SegSymbParams, HDParams)
    elif (SegSymbParams.symbolType == 'FFT'):
        model = HD_classifier_FFT(SigInfoParams, SegSymbParams, HDParams)
    elif (SegSymbParams.symbolType == 'RawAmpl'):
        model = HD_classifier_RawAmpl(SigInfoParams, SegSymbParams, HDParams)

    #########################################

    numPat= len(GeneralParams.patients)
    timeTotalPerSubj_mean=np.zeros((numPat))
    timeTotalPerSubj_std=np.zeros((numPat))
    timeFeatPerSubj_mean = np.zeros((numPat))
    timeFeatPerSubj_std = np.zeros((numPat))
    timeHDVecPerSubj_mean = np.zeros((numPat))
    timeHDVecPerSubj_std = np.zeros((numPat))
    ratioFeatHDTime_mean = np.zeros((numPat))
    ratioFeatHDTime_std = np.zeros((numPat))
    #go through all patients
    patIndx = -1
    for pat in GeneralParams.patients:
        patIndx = patIndx + 1
        numFiles = len(np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv')))
        print('-- Patient:', pat, 'NumSeizures:', numFiles)

        timeTotalPerFile=np.zeros((numFiles))
        timeFeatPerFile=np.zeros((numFiles))
        timeHDVecPerFile=np.zeros((numFiles))
        ratioFeatHDTime = np.zeros((numFiles))
        for fileIndx,fileIn in enumerate(np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv'))):
            pom, fileName1 = os.path.split(fileIn)
            fileName2 = os.path.splitext(fileName1)[0]

            # reading data
            reader = csv.reader(open(fileIn, "r"))
            data0 = list(reader)
            data = np.array(data0).astype("float")
            # separating to data and labels
            X = data[:, SigInfoParams.chToKeep]
            y = data[:, -1]

            dataToUse0=np.squeeze(X).transpose()
            dataToUse=torch.from_numpy(np.array(dataToUse0)).cuda()  # .t()

            # encoding end measuring time
            start =time.time()
            (dataVec, timeFeatTotal, timeHDVecTotal) = learn_HDproj_wholeSignal(model, dataToUse, SegSymbParams, EEGfreqBands, SigInfoParams, HDParams)
            end = time.time()

            # calculating number of windows
            segLenIndx = int(SegSymbParams.segLenSec * SigInfoParams.samplFreq)  # length of EEG segments in samples
            slidWindStepIndx = int(SegSymbParams.slidWindStepSec * SigInfoParams.samplFreq)  # step of slidin window to extract segments in samples
            numWindows =int( ( len(y) - segLenIndx)/ slidWindStepIndx)

            timeTotalPerFile[fileIndx]=(end-start) /numWindows
            timeFeatPerFile[fileIndx]=timeFeatTotal/numWindows
            timeHDVecPerFile[fileIndx]=timeHDVecTotal /numWindows
            ratioFeatHDTime[fileIndx]=timeFeatTotal /timeHDVecTotal

        timeTotalPerSubj_mean[patIndx]=np.mean(timeTotalPerFile)
        timeTotalPerSubj_std[patIndx] = np.std(timeTotalPerFile)
        timeFeatPerSubj_mean[patIndx] = np.mean(timeFeatPerFile)
        timeFeatPerSubj_std[patIndx] = np.std(timeFeatPerFile)
        timeHDVecPerSubj_mean[patIndx] = np.mean(timeHDVecPerFile)
        timeHDVecPerSubj_std[patIndx] = np.std(timeHDVecPerFile)
        ratioFeatHDTime_mean[patIndx] = np.mean(ratioFeatHDTime)
        ratioFeatHDTime_std[patIndx] = np.std(ratioFeatHDTime)

    return (timeTotalPerSubj_mean, timeTotalPerSubj_std,timeFeatPerSubj_mean,timeFeatPerSubj_std,timeHDVecPerSubj_mean, timeHDVecPerSubj_std, ratioFeatHDTime_mean, ratioFeatHDTime_std)
