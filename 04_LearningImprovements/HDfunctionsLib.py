''' library with different functions on HD vectors, uses torch library '''
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

import torch
import time, sys
from VariousFunctionsLib import *

class HD_classifier_GeneralAndNoCh:
    ''' Approach that uses several features and then maps them all to HD vectors
    doesnt know and doesnt care that some features are from different channels'''

    def __init__(self,SigInfoParams, SegSymbParams ,HDParams, totNumFeat, vecType='bin', cuda = True):
        ''' Initialize an HD vectors using the torch library
        SigInfoParams, SegSymbParams and HDParams contain all parameters needed
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''

        self.NumValues = SegSymbParams.numSegLevels
        self.NumFeat =totNumFeat #HDParams.numFeat
        self.HD_dim = HDParams.D
        self.device = HDParams.CUDAdevice
        self.LVLvectTypes=HDParams.vectorTypeLevel
        self.FEATvectTypes=HDParams.vectorTypeFeat
        self.roundingType=HDParams.roundingTypeForHDVectors

        #CREATING VALUE LEVEL VECTORS
        if self.LVLvectTypes == 'sandwich':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumValues, vecType)
        elif self.LVLvectTypes=='random':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues, vecType)
        elif "scaleNoRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[11:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleNoRand(cuda, self.device,  self.HD_dim, self.NumValues, factor, vecType)
        elif "scaleRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[9:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumValues,factor, vecType)
        else:
            self.proj_mat_FeatVals = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues, vecType)

        #CREATING FEATURE VECTORS
        if self.FEATvectTypes == 'sandwich':
            self.proj_mat_features = func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumFeat, vecType)
        elif self.FEATvectTypes == 'random':
            self.proj_mat_features = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumFeat, vecType)
        elif "scaleNoRand" in self.FEATvectTypes:
            factor = int(self.FEATvectTypes[11:])
            self.proj_mat_features = func_generateVectorsMemory_ScaleNoRand(cuda, self.device, self.HD_dim, self.NumFeat,factor, vecType)
        elif "scaleRand" in self.FEATvectTypes:
            factor = int(self.FEATvectTypes[9:])
            self.proj_mat_features = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumFeat, factor, vecType)
        else:
            self.proj_mat_features = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumFeat, vecType)



    def learn_HD_proj(self,data, HDParams):
        ''' From features of one window to HDvector representing that data window
        '''
        t0=time.time()
        if (HDParams.bindingFeatures == 'FeatxVal'):
            bindingVector = xor(self.proj_mat_features, self.proj_mat_FeatVals[:, data.astype(int)] )
            output_vector = torch.sum(bindingVector, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                output_vector = (output_vector > int(math.floor(self.NumFeat / 2))).short()
            else:
                output_vector = (output_vector / self.NumFeat)

        elif (HDParams.bindingFeatures == 'FeatValRot'):
            Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
            for f in range(self.NumFeat):
                Featvect_matrix[:, f] = rotateVec(self.proj_mat_features[:, f], self.proj_mat_FeatVals[:, int(data[f])])
            output_vector = torch.sum(Featvect_matrix, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                output_vector = (output_vector > int(math.floor(self.NumFeat / 2))).short()
            else:
                output_vector = (output_vector / self.NumFeat)

        elif (HDParams.bindingFeatures  == 'FeatAppend' ):
            output_vector = torch.zeros(self.HD_dim*self.NumFeat,1).cuda(device=self.device)
            for f in range(self.NumFeat):
                #apppending
                output_vector[ f*self.HD_dim: (f+1)*self.HD_dim,0]=self.proj_mat_FeatVals[:, int(data[f])]

        timeHDVec=time.time()-t0

        return output_vector, timeHDVec

    def learn_HD_proj_bipolar(self,data, HDParams):
        ''' From features of one window to HDvector representing that data window
        '''
        t0=time.time()
        if (HDParams.bindingFeatures == 'FeatxVal'):
            bindingVector = xor_bipolar(self.proj_mat_features, self.proj_mat_FeatVals[:, data.astype(int)] ) #!!! differnt from binary
            output_vector = torch.sum(bindingVector, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                #output_vector = (output_vector > int(math.floor(self.NumFeat / 2))).short()
                output_vector = (output_vector >0 ).short()  #!!! differnt from binary
            else:
                output_vector = (output_vector / self.NumFeat)

        elif (HDParams.bindingFeatures == 'FeatValRot'):
            Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
            for f in range(self.NumFeat):
                Featvect_matrix[:, f] = rotateVec(self.proj_mat_features[:, f], self.proj_mat_FeatVals[:, int(data[f])])
            output_vector = torch.sum(Featvect_matrix, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                #output_vector = (output_vector > int(math.floor(self.NumFeat / 2))).short()
                output_vector = (output_vector > 0).short() #!!! differnt from binary
            else:
                output_vector = (output_vector / self.NumFeat)

        elif (HDParams.bindingFeatures  == 'FeatAppend' ):
            output_vector = torch.zeros(self.HD_dim*self.NumFeat,1).cuda(device=self.device)
            for f in range(self.NumFeat):
                #apppending
                output_vector[ f*self.HD_dim: (f+1)*self.HD_dim,0]=self.proj_mat_FeatVals[:, int(data[f])]

        timeHDVec=time.time()-t0
        #replace 0 with -1 after rounnn
        if (HDParams.roundingTypeForHDVectors != 'noRounding'): #!!! diffferent from binary
            output_vector[output_vector==0] =-1
        return output_vector, timeHDVec

class HD_classifier_GeneralWithChCombinations:
    ''' Approach that uses several features and then maps them all to HD vectors
    but know that some features are from different channels so there are different ways to approach it'''

    def __init__(self,SigInfoParams, SegSymbParams ,HDParams, numCh,vecType='bin', cuda = True):
        ''' Initialize an HD vectors using the torch library
        SigInfoParams, SegSymbParams and HDParams contain all parameters needed
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''

        self.NumValues = SegSymbParams.numSegLevels
        self.NumFeat =HDParams.numFeat #HDParams.numFeat
        self.HD_dim = HDParams.D
        self.NumCh = numCh
        self.device = HDParams.CUDAdevice
        self.LVLvectTypes=HDParams.vectorTypeLevel
        self.FEATvectTypes=HDParams.vectorTypeFeat
        self.CHvectTypes=HDParams.vectorTypeCh
        self.roundingType=HDParams.roundingTypeForHDVectors

        #CREATING VALUE LEVEL VECTORS
        if self.LVLvectTypes == 'sandwich':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumValues, vecType)
        elif self.LVLvectTypes=='random':
            self.proj_mat_FeatVals=func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues, vecType)
        elif "scaleNoRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[11:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleNoRand(cuda, self.device,  self.HD_dim, self.NumValues, factor, vecType)
        elif "scaleRand" in self.LVLvectTypes:
            factor=int(self.LVLvectTypes[9:])
            self.proj_mat_FeatVals = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumValues,factor, vecType)
        else:
            self.proj_mat_FeatVals = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumValues, vecType)

        #CREATING FEATURE VECTORS
        if self.FEATvectTypes == 'sandwich':
            self.proj_mat_features = func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumFeat, vecType)
        elif self.FEATvectTypes == 'random':
            self.proj_mat_features = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumFeat, vecType)
        elif "scaleNoRand" in self.FEATvectTypes:
            factor = int(self.FEATvectTypes[11:])
            self.proj_mat_features = func_generateVectorsMemory_ScaleNoRand(cuda, self.device, self.HD_dim, self.NumFeat,factor, vecType)
        elif "scaleRand" in self.FEATvectTypes:
            factor = int(self.FEATvectTypes[9:])
            self.proj_mat_features = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumFeat, factor, vecType)
        else:
            self.proj_mat_features = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumFeat, vecType)

        #CREATING CHANNEL VECTORS
        if self.CHvectTypes == 'sandwich':
            self.proj_mat_channels = func_generateVectorsMemory_Sandwich(cuda, self.device, self.HD_dim, self.NumCh, vecType)
        elif self.CHvectTypes == 'random':
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumCh, vecType)
        elif "scaleNoRand" in self.CHvectTypes:
            factor = int(self.CHvectTypes[11:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleNoRand(cuda, self.device, self.HD_dim, self.NumCh,factor, vecType)
        elif "scaleRand" in self.CHvectTypes:
            factor = int(self.CHvectTypes[9:])
            self.proj_mat_channels = func_generateVectorsMemory_ScaleRand(cuda, self.device, self.HD_dim, self.NumCh, factor, vecType)
        else:
            self.proj_mat_channels = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim, self.NumCh, vecType)

        # CREATING COMBINED VECTORS FOR CH ANF FEATURES  - but only random
        if (HDParams.bindingFeatures == 'ChFeatCombxVal'):
            self.proj_mat_featuresCh = func_generateVectorsMemory_Random(cuda, self.device, self.HD_dim,self.NumFeat * self.NumCh, vecType)

    def learn_HD_proj(self,data, HDParams):
        ''' From features of one window to HDvector representing that data window
        different options possible:'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend'
        '''
        t0=time.time()

        data.resize(self.NumCh , self.NumFeat)
        data=data.astype(int)
        if (HDParams.bindingFeatures == 'FeatxVal'):
            Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
            for ch in range(self.NumCh):
                bindingVector = xor(self.proj_mat_features, self.proj_mat_FeatVals[:,data[ch,:]])
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
                bindingVector = xor(self.proj_mat_features, self.proj_mat_FeatVals[:, data[ch,:]])
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
                bindingVector = xor(self.proj_mat_channels, self.proj_mat_FeatVals[:, data[:,f]])
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh/ 2))).short()
            if (self.NumFeat>1):
                #binding with  feature vectors
                bindingVector2=xor(Featvect_matrix, self.proj_mat_features)
                output_vector = torch.sum(bindingVector2, dim=1)
                output_vector = torch.sum(bindingVector2, dim=1)
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    output_vector = (output_vector > int(math.floor(self.NumFeat/ 2))).short()
                else:
                    output_vector = (output_vector / self.NumFeat)
            else: #special case when we have only one feature
                output_vector=Featvect_matrix[:,0]

        elif (HDParams.bindingFeatures =='ChFeatCombxVal'):
            Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat*self.NumCh).cuda(device=self.device)
            for f in range(self.NumFeat):
                for ch in range(self.NumCh):
                    # bind ch and  values of current feature
                    Featvect_matrix[:,f*self.NumCh+ch] = xor(self.proj_mat_featuresCh[:,f*self.NumCh+ch], self.proj_mat_FeatVals[:, data[ch,f]])
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
                bindingVector = xor(self.proj_mat_channels, self.proj_mat_FeatVals[:, data[:,f]])
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh / 2))).short()
                else:
                    Featvect_matrix[:,f]  = (bindingVectorSum /  self.NumCh)
                #apppending
                output_vector[f*self.HD_dim:(f+1)*self.HD_dim,0]=Featvect_matrix[:,f]


        timeHDVec=time.time()-t0

        return output_vector, timeHDVec

    def learn_HD_proj_bipolar(self,data, HDParams):
        ''' From features of one window to HDvector representing that data window
        different options possible:'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend'
        '''
        t0=time.time()

        data.resize(self.NumCh , self.NumFeat)
        data=data.astype(int)
        if (HDParams.bindingFeatures == 'FeatxVal'):
            Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
            for ch in range(self.NumCh):
                bindingVector = xor_bipolar(self.proj_mat_features, self.proj_mat_FeatVals[:,data[ch,:]]) #!!! diffferent from binary
                Chvect_matrix[:,ch] = torch.sum(bindingVector, dim=1)
            output_vector = torch.sum(Chvect_matrix, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                #output_vector = (output_vector > int(math.floor(self.NumCh * self.NumFeat / 2))).short()
                output_vector = (output_vector > 0).short() #!!! diffferent from binary
            else:
                output_vector = (output_vector / (self.NumCh * self.NumFeat))

        elif (HDParams.bindingFeatures =='ChxFeatxVal'):
            Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
            for ch in range(self.NumCh):
                #bind features and their values
                bindingVector = xor_bipolar(self.proj_mat_features, self.proj_mat_FeatVals[:, data[ch,:]]) #!!! diffferent from binary
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                #Chvect_matrix[:,ch]  = (bindingVectorSum > int(math.floor(self.NumFeat/ 2))).short()
                Chvect_matrix[:, ch] = (bindingVectorSum > 0).short() #!!! diffferent from binary
            #binding with  channels vectors
            bindingVector2=xor_bipolar(Chvect_matrix, self.proj_mat_channels) #!!! diffferent from binary
            output_vector = torch.sum(bindingVector2, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                #output_vector = (output_vector > int(math.floor(self.NumCh/ 2))).short()
                output_vector = (output_vector > 0).short()  # !!! diffferent from binary
            else:
                output_vector = (output_vector / self.NumCh)

        elif (HDParams.bindingFeatures =='FeatxChxVal'):
            Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
            for f in range(self.NumFeat):
                #bind ch and  values of current feature
                bindingVector = xor_bipolar(self.proj_mat_channels, self.proj_mat_FeatVals[:, data[:,f]]) #!!! diffferent from binary
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                #Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh/ 2))).short()
                Featvect_matrix[:,f]  = (bindingVectorSum > 0).short()# !!! diffferent from binary
            if (self.NumFeat>1):
                #binding with  feature vectors
                bindingVector2=xor_bipolar(Featvect_matrix, self.proj_mat_features) #!!! diffferent from binary
                output_vector = torch.sum(bindingVector2, dim=1)
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    #output_vector = (output_vector > int(math.floor(self.NumFeat/ 2))).short()
                    output_vector = (output_vector > 0).short()  # !!! diffferent from binary
                else:
                    output_vector = (output_vector / self.NumFeat)
            else: #special case when we have only one feature
                output_vector=Featvect_matrix[:,0]

        elif (HDParams.bindingFeatures =='ChFeatCombxVal'):
            Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat*self.NumCh).cuda(device=self.device)
            for f in range(self.NumFeat):
                for ch in range(self.NumCh):
                    # bind ch and  values of current feature
                    Featvect_matrix[:,f*self.NumCh+ch] = xor_bipolar(self.proj_mat_featuresCh[:,f*self.NumCh+ch], self.proj_mat_FeatVals[:, data[ch,f]]) #!!! diffferent from binary
            output_vector = torch.sum(Featvect_matrix, dim=1)
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                #output_vector = (output_vector > int(math.floor(self.NumFeat*self.NumCh / 2))).short()
                output_vector = (output_vector > 0).short()  # !!! diffferent from binary
            else:
                output_vector = (output_vector / (self.NumFeat*self.NumCh ))

        elif ('FeatAppend' in HDParams.bindingFeatures):
            Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
            output_vector=torch.zeros(self.HD_dim*self.NumFeat,1).cuda(device=self.device)
            for f in range(self.NumFeat):
                #bind ch and  values of current feature
                bindingVector = xor_bipolar(self.proj_mat_channels, self.proj_mat_FeatVals[:, data[:,f]]) #!!! diffferent from binary
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    #Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh / 2))).short()
                    Featvect_matrix[:, f] = (bindingVectorSum > 0).short()  # !!! diffferent from binary
                else:
                    Featvect_matrix[:,f]  = (bindingVectorSum /  self.NumCh)
                #apppending
                output_vector[f*self.HD_dim:(f+1)*self.HD_dim,0]=Featvect_matrix[:,f]

        #replace 0 with -1 after rounnn
        if (HDParams.roundingTypeForHDVectors != 'noRounding'): #!!! diffferent from binary
            output_vector[output_vector==0] =-1
        timeHDVec=time.time()-t0

        return output_vector, timeHDVec

def func_generateVectorsMemory_Random(cuda, device, HD_dim, Num_vect, vecType = 'bin'):
    ''' function to generate matrix of HD vectors using random method
        random - each vector is independantly randomly generated
    '''
    if ( vecType=='bin'):
        negValue=0
    else:
        negValue=-1

    if cuda:
        vect_matrix = torch.randn(HD_dim, Num_vect).cuda(device=device)
    else:
        vect_matrix= torch.randn(HD_dim, Num_vect)
    vect_matrix[vect_matrix > 0] = 1
    vect_matrix[vect_matrix <= 0] = negValue
    return(vect_matrix)

def func_generateVectorsMemory_Sandwich(cuda, device, HD_dim, Num_vect, vecType='bin'):
    ''' function to generate matrix of HD vectors using sandwich method
    sandwich - every two neighbouring vectors  have half of the vector the same, but the rest of the vector is random
    '''
    if ( vecType=='bin'):
        negValue=0
    else:
        negValue=-1

    if cuda:
        vect_matrix = torch.zeros(HD_dim, Num_vect).cuda(device=device)
    else:
        vect_matrix = torch.zeros(HD_dim, Num_vect)
    for i in range(Num_vect):
        if i % 2 == 0:
            vect_matrix[:, i] = torch.randn(HD_dim).cuda(
                device=device)  # torch.randn(self.HD_dim, 1).cuda(device = device)
            vect_matrix[vect_matrix > 0] = 1
            vect_matrix[vect_matrix <= 0] = negValue
    for i in range(Num_vect - 1):
        if i % 2 == 1:
            vect_matrix[0:int(HD_dim / 2), i] = vect_matrix[0:int(HD_dim / 2), i - 1]
            vect_matrix[int(HD_dim / 2):HD_dim, i] = vect_matrix[int(HD_dim / 2):HD_dim, i + 1]
    vect_matrix[0:int(HD_dim / 2), Num_vect - 1] = vect_matrix[0:int(HD_dim / 2), Num_vect - 2]
    if cuda:
        vect_matrix[int(HD_dim / 2):HD_dim, Num_vect - 1] = torch.randn(int(HD_dim / 2)).cuda( device=device)  # torch.randn(int(HD_dim/2), 1).cuda(device = device)
    else:
        vect_matrix[int(HD_dim / 2):HD_dim, Num_vect - 1] = torch.randn(int(HD_dim / 2))  # torch.randn(int(HD_dim/2), 1)
    vect_matrix[vect_matrix > 0] = 1
    vect_matrix[vect_matrix <= 0] = negValue
    return (vect_matrix)

def func_generateVectorsMemory_ScaleRand(cuda, device, HD_dim, Num_vect, scaleFact, vecType = 'bin'):
    ''' function to generate matrix of HD vectors using scale method with bits of randomization
    scaleRand - every next vector is created by randomly flipping D/(numVec*scaleFact) elements - this way the further values vectors represent are, the less similar are vectors
    '''
    if ( vecType=='bin'):
        negValue=0
    else:
        negValue=-1
    numValToFlip=floor(HD_dim/(scaleFact*Num_vect))

    #initialize vectors
    if cuda:
        vect_matrix = torch.zeros(HD_dim,Num_vect).cuda(device=device)
    else:
        vect_matrix = torch.zeros(HD_dim, Num_vect)
    #generate firs one as random
    vect_matrix[:, 0] = torch.randn(HD_dim).cuda(device=device)  # torch.randn(self.HD_dim, 1).cuda(device = device)
    vect_matrix[vect_matrix > 0] = 1
    vect_matrix[vect_matrix <= 0] =  negValue
    #iteratively the further they are flip more bits
    for i in range(1,Num_vect):
        vect_matrix[:, i]=vect_matrix[:, i-1]
        #choose random positions to flip
        posToFlip=random.sample(range(1,HD_dim),numValToFlip)
        if (vecType=='bin'):
            vect_matrix[posToFlip,i] = vect_matrix[posToFlip,i]*(-1)+1
        else:
            vect_matrix[posToFlip,i] = vect_matrix[posToFlip,i]*(-1)

        # #test if distance is increasing
        # modelVector0 = vect_matrix[:, 0].cpu().numpy()
        # modelVector1 = vect_matrix[:,i-1].cpu().numpy()
        # modelVector2 = vect_matrix[:,i].cpu().numpy()
        # print('Vector difference for i-th:',i, ' is : ', np.sum(abs(modelVector2 - modelVector0)), ' &', np.sum(abs(modelVector2 - modelVector1)) )

    return(vect_matrix)

def func_generateVectorsMemory_ScaleNoRand(cuda, device, HD_dim, Num_vect, scaleFact, vecType = 'bin'):
    ''' function to generate matrix of HD vectors  using scale method with no randomization
    scaleNoRand - same idea as scaleRand, just  d=D/(numVec*scaleFact) bits are taken in order (e.g. from i-th to i+d bit) and not randomly
    '''
    if ( vecType=='bin'):
        negValue=0
    else:
        negValue=-1
    numValToFlip = floor(HD_dim / (scaleFact * Num_vect))

    # initialize vectors
    if cuda:
        vect_matrix = torch.zeros(HD_dim, Num_vect).cuda(device=device)
    else:
        vect_matrix = torch.zeros(HD_dim, Num_vect)
    # generate firs one as random
    vect_matrix[:, 0] = torch.randn(HD_dim).cuda(device=device)  # torch.randn(self.HD_dim, 1).cuda(device = device)
    vect_matrix[vect_matrix > 0] = 1
    vect_matrix[vect_matrix <= 0] =  negValue
    # iteratively the further they are flip more bits
    for i in range(1, Num_vect):
        vect_matrix[:, i] = vect_matrix[:, 0]
        vect_matrix[0: i * numValToFlip, i] = flipVectorValues(vect_matrix[0: i * numValToFlip, 0], vecType)
    # #test if distance is increasing
    # modelVector1 = vect_matrix[:,0].cpu().numpy()
    # modelVector2 = vect_matrix[:,i].cpu().numpy()
    # print('Vector difference for i-th:',i, ' is : ', np.sum(abs(modelVector1 - modelVector2)))

    return (vect_matrix)


def flipVectorValues(vector, vecType = 'bin'):
    '''turns 0 into 1 and 1 into 0'''
    if (vecType=='bin'):
        vectorFliped=(vector*(-1)+1)
    else:
        vectorFliped = vector * (-1)
    return(vectorFliped)

def xor( vec_a, vec_b):
    ''' xor between vec_a and vec_b'''
    vec_c = (torch.add(vec_a, vec_b) == 1).short()  # xor
    return vec_c

def xor_bipolar( vec_a, vec_b):
    ''' xor between vec_a and vec_b'''
    #vec_c = (torch.sub(vec_a, vec_b) != 0).short()  # 1
    vec_c = torch.sub(vec_a, vec_b)
    vec_c [vec_c!=0]= 1
    vec_c[vec_c == 0] = -1
    return vec_c

def ham_dist( vec_a, vec_b, D):
    ''' calculate relative hamming distance'''
    #vec_c = xor(vec_a, vec_b) # this was used before but only works for binary vectors
    vec_c= torch.abs(torch.sub(vec_a,vec_b))
    rel_dist = torch.sum(vec_c) / float(D)
    return rel_dist

def ham_dist_arr( vec_a, vec_b, D, vecType='bin'):
    ''' calculate relative hamming distance fur for np array'''
    if (vecType=='bin'):
        vec_c= np.abs(vec_a-vec_b)
        rel_dist = np.sum(vec_c) / float(D)
    elif (vecType=='bipol'):
        vec_c= vec_a+vec_b
        rel_dist = np.sum(vec_c==0) / float(D)
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

def cos_dist_arr( vA, vB):
	''' calculate cosine distance of two vectors'''
	output=np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))
	outTensor=torch.tensor(1.0-output) #because we use latter that if value is 0 then vectors are the same
	return outTensor


def rotateVec(vec, numRot):
	'''shift vector for numRot bits '''
	outVec=torch.roll(vec,-numRot,0)
	return outVec


def givePrediction(temp, ModelVectors, simType, vecType='bin'):
    (numLabels,D)=ModelVectors.shape
    distances=np.zeros(numLabels)
    for l in range(numLabels):
        if (simType== 'hamming'):
            distances[l]= ham_dist_arr( ModelVectors[l,:], np.squeeze(temp), D, vecType)
        if (simType== 'cosine'):
            distances[l]= cos_dist_arr( ModelVectors[l,:], np.squeeze(temp))
    #find minimum
    mins=np.argmin(distances)
    if np.isscalar(mins):
        minVal=mins
    else:
        print('More labels possible')
        minVal=mins[0]
    return (minVal,distances)

def givePrediction_separateVectorsMulticlass(tempVec, ModelVectors_Seiz,ModelVectors_NonSeiz,  HDParams):

    (predLabels_Seiz, distances_Seiz) = givePrediction(tempVec, ModelVectors_Seiz, HDParams.similarityType)
    (predLabels_NonSeiz, distances_NonSeiz) = givePrediction(tempVec, ModelVectors_NonSeiz,
                                                                      HDParams.similarityType)
    # find the closest distance
    minSeiz_Indx = np.argmin(distances_Seiz)
    minSeiz_Dist = distances_Seiz[ minSeiz_Indx]
    minNonSeiz_Indx = np.argmin(distances_NonSeiz)
    minNonSeiz_Dist = distances_NonSeiz[minNonSeiz_Indx]
    if (minSeiz_Dist < minNonSeiz_Dist):  # sizure is final prediction
        predLabels_nonBin = predLabels_Seiz + 1
        predLabels= 1
    else:  # nonSeiz is final predictions
        predLabels_nonBin = -predLabels_NonSeiz - 1
        predLabels = 0

    return(predLabels, predLabels_nonBin, distances_Seiz, distances_NonSeiz)


def givePrediction_indivAppendFeat(temp, ModelVectors, numFeat, D, simType):
    (numLabels,totalD)=ModelVectors.shape
    distances=np.zeros((numLabels, numFeat))
    predictions=np.zeros(numFeat)
    for f in range(numFeat):
        for l in range(numLabels):
            if (simType== 'hamming'):
                distances[l,f]= ham_dist_arr( ModelVectors[l,f*D:(f+1)*D], np.squeeze(temp[f*D:(f+1)*D]), D)
            if (simType== 'cosine'):
                distances[l,f]= cos_dist_arr( ModelVectors[l,f*D:(f+1)*D], np.squeeze(temp[f*D:(f+1)*D]))
        #find minimum
        mins=np.argmin(distances[:,f])
        if np.isscalar(mins):
            predictions[f]=mins
        else:
            print('More labels possible')
            predictions[f]=mins[0]
    return (predictions,distances)

def trainModelVecOnData(data, labels, model, HDParams, vecType='bin'):
    '''learn model vectors for single pass 2 class HD learning '''

    (numWin_train, numFeat) = data.shape
    # number of clases
    (unique_labels, counts) = np.unique(labels, return_counts=True)
    numLabels = len(unique_labels)

    # initializing model vectors and then training
    if ('FeatAppend' in HDParams.bindingFeatures):
        ModelVectors = np.zeros((numLabels, HDParams.D*HDParams.numFeat))  # .cuda(device=self.device)
        ModelVectorsNorm = np.zeros((numLabels, HDParams.D*HDParams.numFeat))
    else:
        ModelVectors = np.zeros((numLabels, HDParams.D))   # .cuda(device=self.device)
        ModelVectorsNorm= np.zeros((numLabels, HDParams.D))
    numAddedVecPerClass= np.zeros(numLabels)

    #go through all data windows and add them to model vectors
    for s in range(numWin_train):
        if (vecType=='bin'): #if vectors are binary (0,1)
            (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        elif (vecType=='bipol'): #if vectors are bipolar (-1,1)
            (temp, timeHDVec) = model.learn_HD_proj_bipolar(data[s, :], HDParams)
        lab = int(labels[s])  # adding 0.5 so that in case -1 or 0 it is again 0
        tempArray=temp.cpu().numpy()
        tempArray.resize(1, len(tempArray))
        ModelVectors[lab, :] = ModelVectors[lab, :] + tempArray
        numAddedVecPerClass[lab] = numAddedVecPerClass[lab] + 1 #count number of vectors added to each subclass

    # normalize model vectors to be binary (or bipolar) again
    for l in range(numLabels):
        if (HDParams.roundingTypeForHDVectors != 'noRounding'):
            if (vecType == 'bin'):
                ModelVectorsNorm[l, :] = (ModelVectors[l, :] > int(math.floor(numAddedVecPerClass[l] / 2))) * 1
            elif (vecType == 'bipol'):
                ModelVectorsNorm[l, :] = (ModelVectors[l, :] > 0) * 1
                ModelVectorsNorm[l,ModelVectorsNorm[l, :] ==0] = -1
        else:
            ModelVectorsNorm[l, :] = (ModelVectors[l, :] / numAddedVecPerClass[l])
    return (ModelVectors, ModelVectorsNorm, numAddedVecPerClass,numLabels)

# def trainModelVecOnData_ItterMultiClass_initialOneSubclass(data, labels, model, HDParams):
#     ''' works onlz when we have just two classes - seiz and nonSeiz
#     created from fucntion trainModelVecOnData '''
#
#     (numWin_train, numFeat) = data.shape
#     # number of clases
#     (unique_labels, counts) = np.unique(labels, return_counts=True)
#     numLabels = len(unique_labels)
#
#     # initializing model and then training
#     if ('FeatAppend' in HDParams.bindingFeatures):
#         ModelVectors = np.zeros((numLabels, HDParams.D*HDParams.numFeat))  # .cuda(device=self.device)
#         ModelVectorsNorm = np.zeros((numLabels, HDParams.D*HDParams.numFeat))
#     else:
#         ModelVectors = np.zeros((numLabels, HDParams.D))   # .cuda(device=self.device)
#         ModelVectorsNorm= np.zeros((numLabels, HDParams.D))
#     numAddedVecPerClass= np.zeros(numLabels)
#
#     for s in range(numWin_train):
#         (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
#         lab = int(labels[s])  # -1
#         tempArray=temp.cpu().numpy()
#         tempArray.resize(1, len(tempArray))
#         try:
#             ModelVectors[lab, :] = ModelVectors[lab, :] + tempArray
#         except:
#             print('error')
#             (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
#         numAddedVecPerClass[lab] = numAddedVecPerClass[lab] + 1
#
#     for l in range(numLabels):
#         if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#             ModelVectorsNorm[l, :] = (ModelVectors[l, :] > int(math.floor(numAddedVecPerClass[l] / 2))) * 1
#         else:
#             ModelVectorsNorm[l, :] = (ModelVectors[l, :] / numAddedVecPerClass[l])
#     return (ModelVectors, ModelVectorsNorm, numAddedVecPerClass,numLabels)
#
def trainModelVecOnData_Multiclass(data, labels, model, HDParams, vecType='bin'):
    '''learn model vectors for single pass but multi-class learning HD approach '''

    (numWin_train, numFeat) = data.shape
    # number of clases
    (unique_labels, counts) = np.unique(labels, return_counts=True)
    numLabels = len(unique_labels)

    #initialize model vectors
    ModelVectors_Seiz = []
    ModelVectorsNorm_Seiz  = []
    ModelVectors_NonSeiz  = []
    ModelVectorsNorm_NonSeiz  = []

    #go through all data windows and add them to model vectors
    numAddedVecPerClass_Seiz = 0
    numAddedVecPerClass_NonSeiz = 0
    dist_StoSArray=[]
    dist_StoNSArray=[]
    dist_NStoSArray = []
    dist_NStoNSArray = []
    for s in range(numWin_train):
        if (vecType=='bin'):
            (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        elif (vecType == 'bipol'):
            (temp, timeHDVec) = model.learn_HD_proj_bipolar(data[s, :], HDParams)
        currVect=temp.cpu().numpy()
        lab = int(labels[s])  # -1

        if lab==1: #if siezure
            # calculate mean distances from both classes
            dist_StoS=np.mean(dist_StoSArray) #/( np.sum(numAddedVecPerClass_Seiz)-numSubClasses)
            dist_StoNS=np.mean(dist_StoNSArray) #/np.sum(numAddedVecPerClass_NonSeiz)
            # if it is too far from correct class create new subclass, else add to existing one
            (ModelVectors_Seiz, ModelVectorsNorm_Seiz, numAddedVecPerClass_Seiz, finDist, minIndx )= func_findOptClassAndAddToModelVec(ModelVectors_Seiz, ModelVectorsNorm_Seiz, numAddedVecPerClass_Seiz, currVect, HDParams.similarityType, HDParams.roundingTypeForHDVectors , dist_StoS, dist_StoNS, vecType)
            #updating mean distances from seiz and non Seiz
            if (finDist!=0):
                dist_StoSArray.append(finDist)
            if (np.sum(numAddedVecPerClass_NonSeiz) != 0):
                indx = np.where(numAddedVecPerClass_NonSeiz > 0)
                if np.isscalar(indx):
                    numSubClasses_NonSeiz = 1
                else:
                    numSubClasses_NonSeiz = len(indx[0])
                #numSubClasses_NonSeiz=len(np.where(numAddedVecPerClass_NonSeiz>0))
                distances = np.zeros(numSubClasses_NonSeiz)
                for c in range(numSubClasses_NonSeiz):
                    if (HDParams.similarityType == 'hamming'):
                        distances[c] = ham_dist_arr(ModelVectorsNorm_NonSeiz[c, :], currVect, HDParams.D, vecType)
                    if (HDParams.similarityType == 'cosine'):
                        distances[c] = cos_dist_arr(ModelVectorsNorm_NonSeiz[c, :], currVect)
                dist_StoNSArray.append(np.mean(distances))
        else: #if non seizure
            # calculate mean distances from both classes
            dist_NStoNS = np.nanmean(dist_NStoNSArray)  # /( np.sum(numAddedVecPerClass_Seiz)-numSubClasses)
            dist_NStoS = np.nanmean(dist_NStoSArray)  # /np.sum(numAddedVecPerClass_NonSeiz)
            # if it is too far from correct class create new subclass, else add to existing one
            (ModelVectors_NonSeiz, ModelVectorsNorm_NonSeiz, numAddedVecPerClass_NonSeiz, finDist, minIndx ) = func_findOptClassAndAddToModelVec( ModelVectors_NonSeiz, ModelVectorsNorm_NonSeiz, numAddedVecPerClass_NonSeiz, currVect, HDParams.similarityType, HDParams.roundingTypeForHDVectors , dist_NStoNS, dist_NStoS, vecType )
            # updating mean distances from seiz and non Seiz
            if (finDist != 0):
                dist_NStoNSArray.append(finDist)
            if (np.sum(numAddedVecPerClass_Seiz)!=0):
                indx = np.where(numAddedVecPerClass_Seiz > 0)
                if np.isscalar(indx):
                    numSubClasses_Seiz = 1
                else:
                    numSubClasses_Seiz = len(indx[0])
                #numSubClasses_Seiz = len(np.where(numAddedVecPerClass_Seiz > 0))
                distances = np.zeros(numSubClasses_Seiz)
                for c in range(numSubClasses_Seiz):
                    if (HDParams.similarityType == 'hamming'):
                        distances[c] = ham_dist_arr(ModelVectorsNorm_Seiz[c, :], currVect, HDParams.D, vecType)
                    if (HDParams.similarityType == 'cosine'):
                        distances[c] = cos_dist_arr(ModelVectorsNorm_Seiz[c, :], currVect)
                dist_NStoSArray.append(np.mean(distances))

    return (ModelVectors_Seiz, ModelVectorsNorm_Seiz, ModelVectors_NonSeiz, ModelVectorsNorm_NonSeiz, numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz)


def trainModelVecOnData_AddOneMoreSubclass(data, labels, model, HDParams, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, ModelVectors_Seiz, ModelVectors_NonSeiz,numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz):

    (numWin, numFeat) = data.shape
    (numSubclasesSeiz, D)= ModelVectors_Seiz.shape
    (numSubclasesNonSeiz, D) = ModelVectors_NonSeiz.shape
    numAddedVecPerClass_Seiz_New=0
    numAddedVecPerClass_NonSeiz_New = 0

    NeedForNewSubclassSeiz=0
    NeedForNewSubclassNonSeiz=0
    for s in range(numWin):
        (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        currVect=temp.cpu().numpy()

        (predLabel, predLabel_nonBin, distances_Seiz, distances_NonSeiz) =givePrediction_separateVectorsMulticlass(currVect, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, HDParams)
        #if would be wrongly calssified add to new subclass
        if (predLabel!=labels[s]):
            #if should be seizure
            if (labels[s]==1):
                NeedForNewSubclassSeiz=1
                if (numAddedVecPerClass_Seiz_New==0):
                    newModelVec_Seiz=currVect
                else:
                    newModelVec_Seiz = newModelVec_Seiz+ currVect
                numAddedVecPerClass_Seiz_New = numAddedVecPerClass_Seiz_New + 1
            #if should be  non seizure
            else:
                NeedForNewSubclassNonSeiz=1
                if (numAddedVecPerClass_NonSeiz_New==0):
                    newModelVec_NonSeiz=currVect
                else:
                    newModelVec_NonSeiz = newModelVec_NonSeiz+ currVect
                numAddedVecPerClass_NonSeiz_New = numAddedVecPerClass_NonSeiz_New + 1

    if (NeedForNewSubclassSeiz==1):
        #normalizing new vec
        if (HDParams.roundingTypeForHDVectors  != 'noRounding'):
            newModelVecNorm_Seiz = (newModelVec_Seiz> int(math.floor(numAddedVecPerClass_Seiz_New / 2))) * 1
        else:
            newModelVecNorm_Seiz = (newModelVec_Seiz / numAddedVecPerClass_Seiz_New)
        #adding to the model matrixes
        ModelVectors_Seiz = np.append(ModelVectors_Seiz, [newModelVec_Seiz], axis=0)
        ModelVectorsNorm_Seiz = np.append(ModelVectorsNorm_Seiz, [newModelVecNorm_Seiz], axis=0)
        numAddedVecPerClass_Seiz = np.append(numAddedVecPerClass_Seiz, numAddedVecPerClass_Seiz_New)
    if (NeedForNewSubclassNonSeiz==1):
        #normalizing new vec
        if (HDParams.roundingTypeForHDVectors  != 'noRounding'):
            newModelVecNorm_NonSeiz = (newModelVec_NonSeiz > int(math.floor(numAddedVecPerClass_NonSeiz_New / 2))) * 1
        else:
            newModelVecNorm_NonSeiz = (newModelVec_NonSeiz / numAddedVecPerClass_NonSeiz_New)

        #adding to the model matrixes
        ModelVectors_NonSeiz = np.append(ModelVectors_NonSeiz, [newModelVec_NonSeiz], axis=0)
        ModelVectorsNorm_NonSeiz = np.append(ModelVectorsNorm_NonSeiz, [newModelVecNorm_NonSeiz], axis=0)
        numAddedVecPerClass_NonSeiz = np.append(numAddedVecPerClass_NonSeiz, numAddedVecPerClass_NonSeiz_New)

    return (NeedForNewSubclassSeiz,NeedForNewSubclassNonSeiz,  ModelVectors_Seiz, ModelVectorsNorm_Seiz, ModelVectors_NonSeiz, ModelVectorsNorm_NonSeiz, numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz)


def calculateVecSeparavilityPerFeature(ModelVectors, D):
    (numClasses, TotalD)=ModelVectors.shape
    numFeat=int(TotalD /D)
    distances=np.zeros(numFeat)
    for f in range(numFeat):
        distances[f] = ham_dist_arr(ModelVectors[0, f*D:(f+1)*D], ModelVectors[1, f*D:(f+1)*D], D)
    return (distances)

def func_findOptClassAndAddToModelVec(modelVecs,modelVecsNorm, numAddedVec,  curVec, simType, rndType, dist_StoS, dist_StoNS, vecType='bin'):
    ''' compares currecnt vector with all model vectors and decides if it is too different and then creates new subclass
     or to add it to some of current subclasses '''

    indx=np.where(numAddedVec>0)
    if (np.sum(numAddedVec)==0): #if first vector adding
        D=len(curVec)
        modelVecs = np.reshape(np.array(curVec), (1,D))
        modelVecsNorm =np.zeros((1,D))
        numAddedVec= np.array([1,0]) #np.reshape(np.array(1), (1,1))
        minIndx=0
        finDist=0
        numSubClass=1
    else:
        (numSubClass, D) = modelVecs.shape
        #D=len(modelVecs)
        distances=np.zeros(numSubClass)
        for c in range(numSubClass):
            #if (numAddedVec[c]!=0):
            if (simType == 'hamming'):
                distances[c] = ham_dist_arr(modelVecsNorm[c, :], curVec, D, vecType)
            if (simType == 'cosine'):
                distances[c] = cos_dist_arr(modelVecsNorm[c, :], curVec)
        #find closest same type subclass
        minIndx=np.argmin(distances)
        finDist=distances[minIndx]
        # adding to existing subclass
        if (np.isnan(dist_StoS) or np.isnan(dist_StoNS) ): #still not enough data
            modelVecs[minIndx, :] = modelVecs[minIndx, :] + curVec
            numAddedVec[minIndx] = numAddedVec[minIndx] + 1
        else:
            # adding to existing subclass
            if ((dist_StoS- finDist)>0): #good add to existing class
                modelVecs[minIndx,:]=modelVecs[minIndx,:]+curVec
                numAddedVec[minIndx] =numAddedVec[minIndx] +1
            elif ( ((dist_StoNS- finDist)> 0) and ((dist_StoNS- finDist)> (finDist -dist_StoS)) ):#good add to existing class
                modelVecs[minIndx, :] = modelVecs[minIndx ,:] + curVec
                numAddedVec[minIndx] = numAddedVec[minIndx] + 1
            # creating new subclass
            else:
                modelVecs= np.append(modelVecs,[curVec],axis=0)
                modelVecsNorm = np.append(modelVecsNorm, [np.zeros(D)], axis=0)
                if (len(numAddedVec)>numSubClass):
                    numAddedVec[numSubClass] = 1
                else:
                    numAddedVec= np.append(numAddedVec, 1)
                minIndx=len(indx)
                finDist=0
                numSubClass=numSubClass+1

    #normalize model vectors to be again binary (or bipolar)
    for l in range(numSubClass):
        if (rndType != 'noRounding'):
            if (vecType=='bin'):
                modelVecsNorm[l, :] = (modelVecs[l, :] > int(math.floor(numAddedVec[l] / 2))) * 1
            elif (vecType=='bipol'):
                modelVecsNorm[l, :] = (modelVecs[l, :] > 0) * 1
                modelVecsNorm[l, modelVecsNorm[l, :] == 0] = -1
        else:
            modelVecsNorm[l, :] = (modelVecs[l, :] / numAddedVec[l])

    return(modelVecs, modelVecsNorm, numAddedVec, finDist, minIndx )

def testModelVecOnData(data, trueLabels, model, ModelVectors, HDParams, vecType='bin'):
    ''' test and measure performance '''
    # number of clases
    (unique_labels, counts) = np.unique(trueLabels, return_counts=True)
    numLabels = len(unique_labels)
    if (numLabels==1): #in specific case when in test set all the same label
        numLabels=2
    (numWin, numFeat) = data.shape
    #go through each data window
    predLabels = np.zeros(numWin)
    distances=np.zeros((numWin, numLabels))
    distFromCorrClass=np.zeros(numWin)
    distFromWrongClass = np.zeros(numWin)
    distWhenWrongPredict=[]
    for s in range(numWin):
        if ( vecType=='bin'):
            (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        elif (vecType=='bipol'):
            (temp, timeHDVec) = model.learn_HD_proj_bipolar(data[s, :], HDParams)
        tempVec=temp.cpu().numpy()
        (predLabels[s], distances[s,:]) = givePrediction(tempVec, ModelVectors, HDParams.similarityType, vecType)
        # calcualte distances from correct and average of wrong classes
        distFromCorrClass[s]=distances[s,trueLabels[s]]
        indx=np.where(unique_labels!=trueLabels[s])
        distFromWrongClass[s]=np.mean(distances[s,indx])
        if(predLabels[s]!=trueLabels[s]):
            distWhenWrongPredict.append(distances[s,int(predLabels[s])])

    #calculate accuracy
    diffLab=predLabels-trueLabels
    indx=np.where(diffLab==0)
    acc= len(indx[0])/len(trueLabels)
    # average distances from correct and wrong classes
    distFromCorr_AllClass=np.mean(distFromCorrClass)
    distFromWrong_AllClass = np.mean(distFromWrongClass)

    # calculate performance and distances per class
    accPerClass=np.zeros(numLabels)
    distFromCorr_PerClass = np.zeros(numLabels)
    distFromWrong_PerClass = np.zeros(numLabels)
    for l in range(numLabels):
        indx=np.where(trueLabels==l)
        trueLabels_part=trueLabels[indx]
        predLab_part=predLabels[indx]
        diffLab = predLab_part - trueLabels_part
        indx2 = np.where(diffLab == 0)
        if (len(indx[0])==0):
            accPerClass[l] = np.nan
        else:
            accPerClass[l] = len(indx2[0]) / len(indx[0])
        distFromCorr_PerClass[l]= np.mean(distFromCorrClass[indx])
        distFromWrong_PerClass[l] = np.mean(distFromWrongClass[indx])

    distWhenWrongPredict_mean=np.mean(np.array(distWhenWrongPredict))
    return(acc, accPerClass, distWhenWrongPredict_mean, distFromCorr_AllClass,distFromCorr_PerClass,  distFromWrong_AllClass, distFromWrong_PerClass, predLabels)

def testModelVecOnData_Multiclass(data, trueLabels, model, ModelVectors_Seiz, ModelVectors_NonSeiz, HDParams, vecType='bin'):
    ''' test performance for multi class HD approach'''
    (numSubClassSeiz,D)=ModelVectors_Seiz.shape
    (numSubClassNonSeiz, D) = ModelVectors_NonSeiz.shape

    #go through all data windows
    (numWin, numFeat) = data.shape
    predLabels = np.zeros(numWin)
    predLabels_nonBin = np.zeros(numWin)
    predLabels_Seiz = np.zeros(numWin)
    distances_Seiz=np.zeros((numWin, numSubClassSeiz))
    predLabels_NonSeiz = np.zeros(numWin)
    distances_NonSeiz=np.zeros((numWin, numSubClassNonSeiz))
    distFromCorrClass=np.zeros(numWin)
    distFromWrongClass = np.zeros(numWin)
    distWhenWrongPredict=[]
    for s in range(numWin):
        #calculate current HD vector representing this window
        if (vecType=='bin'):
            (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        elif (vecType=='bipol'):
            (temp, timeHDVec) = model.learn_HD_proj_bipolar(data[s, :], HDParams)
        tempVec=temp.cpu().numpy()
        #calculate distances from mocel vectors of each subclass
        (predLabels_Seiz[s], distances_Seiz[s,:]) = givePrediction(tempVec, ModelVectors_Seiz, HDParams.similarityType, vecType)
        (predLabels_NonSeiz[s], distances_NonSeiz[s, :]) = givePrediction(tempVec, ModelVectors_NonSeiz, HDParams.similarityType, vecType)
        #decide for prediction
        minSeiz_Indx=np.argmin(distances_Seiz[s,:])
        minSeiz_Dist = distances_Seiz[s,minSeiz_Indx]
        minNonSeiz_Indx = np.argmin(distances_NonSeiz[s, :])
        minNonSeiz_Dist = distances_NonSeiz[s, minNonSeiz_Indx]
        if (minSeiz_Dist< minNonSeiz_Dist): #sizure is final prediction
            predLabels_nonBin[s]=predLabels_Seiz[s]+1
            predLabels[s] =1
        else: #nonSeiz is final predictions
            predLabels_nonBin[s] = -predLabels_NonSeiz[s]-1
            predLabels[s] =0
        #calcualte distances from correct nad wrong subclasses
        if (trueLabels[s]==1):
            distFromCorrClass[s]=minSeiz_Dist
            #distFromWrongClass[s] = np.mean(distances_NonSeiz[s, :])
            distFromWrongClass[s] = minNonSeiz_Dist
        else:
            distFromCorrClass[s] = minNonSeiz_Dist
            #distFromWrongClass[s] = np.mean(distances_Seiz[s, :])
            distFromWrongClass[s] = minSeiz_Dist
        if(predLabels[s]!=trueLabels[s]):
            if (trueLabels[s] == 1):
                distWhenWrongPredict.append(minNonSeiz_Dist)
            else:
                distWhenWrongPredict.append(minSeiz_Dist)

    #calculate accuracy
    diffLab=predLabels-trueLabels
    indx=np.where(diffLab==0)
    acc= len(indx[0])/len(trueLabels)
    # average distances from correct and wrong classes
    distFromCorr_AllClass=np.mean(distFromCorrClass)
    distFromWrong_AllClass = np.mean(distFromWrongClass)
    distWhenWrongPredict_mean=np.mean(np.array(distWhenWrongPredict))

    return(acc, distWhenWrongPredict_mean, distFromCorr_AllClass,  distFromWrong_AllClass, predLabels, predLabels_nonBin)

def testModelVecOnData_ForFeatureSelection(data, trueLabels, model, ModelVectors, HDParams):
    (unique_labels, counts) = np.unique(trueLabels, return_counts=True)
    (numLabels, totalD)=ModelVectors.shape
    (numWin, x) = data.shape
    predLabels = np.zeros((numWin, HDParams.numFeat+1))
    distances=np.zeros((numLabels, HDParams.numFeat+1))
    distFromCorrClass=np.zeros((numWin, HDParams.numFeat+1))
    distFromWrongClass = np.zeros((numWin, HDParams.numFeat+1))
    distWhenWrongPredict=[]
    dist_SS= []
    dist_NS = []
    dist_SN = []
    dist_NN = []
    for s in range(numWin):
        (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        tempVec=temp.cpu().numpy()
        #total predictions using whole vectors
        (predLabels[s, -1], distances[:,-1]) = givePrediction(tempVec, ModelVectors, HDParams.similarityType)
        #predictions and distances per feature
        (predLabels[s, 0:HDParams.numFeat], distances[:, 0:HDParams.numFeat]) = givePrediction_indivAppendFeat(tempVec, ModelVectors, HDParams.numFeat, HDParams.D, HDParams.similarityType)

        # calcualte distances from correct and average of wrong classes
        distFromCorrClass[s,:]=distances[trueLabels[s],:]
        indx=np.where(unique_labels!=trueLabels[s])
        distFromWrongClass[s,:]=np.min(distances[indx,:],0)
        di=np.ones(HDParams.numFeat+1)*np.NaN
        for f in range(HDParams.numFeat+1):
            if(predLabels[s,f]!=trueLabels[s]):
                di[f]=distances[int(predLabels[s,f]),f]
        if (distWhenWrongPredict == []):
            distWhenWrongPredict = di
        else:
            distWhenWrongPredict = np.vstack((distWhenWrongPredict, di))
        #SS AND SN
        if (trueLabels[s]==1):
            if (dist_SS == []):
                dist_SS = distances[1,:]
            else:
                dist_SS = np.vstack((dist_SS, distances[1,:]))
            if (dist_SN == []):
                dist_SN = distances[0,:]
            else:
                dist_SN = np.vstack((dist_SN, distances[0,:]))
        #NS AND NN
        else:
            if (dist_NS == []):
                dist_NS = distances[1, :]
            else:
                dist_NS = np.vstack((dist_NS, distances[1, :]))
            if (dist_NN == []):
                dist_NN = distances[0, :]
            else:
                dist_NN = np.vstack((dist_NN, distances[0, :]))

    return(predLabels,distFromCorrClass, distFromWrongClass, distWhenWrongPredict, dist_SS, dist_SN, dist_NS, dist_NN )

def calcualtePerformanceMeasures_PerFeature(predictions, trueLabels,  toleranceFP_bef, toleranceFP_aft, numLabelsPerHour, seizureStableLenToTestIndx,  seizureStablePercToTest,distanceBetweenSeizuresIndx):

    (numSeg, numCol)=predictions.shape
    performancesAll=np.zeros((numCol, 9)) #9 performance measures
    performancesAll_step1 = np.zeros((numCol, 9))  # 9 performance measures
    performancesAll_step2 = np.zeros((numCol, 9))  # 9 performance measures
    for f in range(numCol):
        performancesAll[f,:]=performance_all9(predictions[:,f], trueLabels, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        # smoothen labels
        (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predictions[:,f], seizureStableLenToTestIndx,
                                                                      seizureStablePercToTest,
                                                                      distanceBetweenSeizuresIndx)
        performancesAll_step1[f, :] = performance_all9(yPred_SmoothOurStep1, trueLabels, toleranceFP_bef, toleranceFP_aft,numLabelsPerHour)
        performancesAll_step2[f, :] = performance_all9(yPred_SmoothOurStep2, trueLabels, toleranceFP_bef,
                                                      toleranceFP_aft, numLabelsPerHour)
    return(performancesAll, performancesAll_step1, performancesAll_step2)

def retrainModelVecOnData(data, trueLabels, model, ModelVectorsNorm, ModelVectors,numAddedVecPerClass, HDParams, type, factor):
    (numLabels, D)=ModelVectors.shape
    # ModelVectorsNorm=np.zeros((numLabels, D))
    (numWin, numFeat) = data.shape
    predLabels = np.zeros(numWin)
    for s in range(numWin):
        (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        tempVec=temp.cpu().numpy()
        (predLabels[s], distances) = givePrediction(tempVec, ModelVectorsNorm, HDParams.similarityType)

        #if labels is wrong
        if (predLabels[s]!=trueLabels[s]):
            # add to the correct class again
            lab = int(trueLabels[s])  # -1
            ModelVectors[lab, :] = ModelVectors[lab, :] + factor*tempVec
            numAddedVecPerClass[lab] = numAddedVecPerClass[lab] + factor*1
            # remove from wrong class
            if (type=='AddAndSubtract'):
                lab = int(predLabels[s])  # -1
                ModelVectors[lab, :] = ModelVectors[lab, :] -  factor*tempVec
                numAddedVecPerClass[lab] = numAddedVecPerClass[lab] -  factor*1

    #calcualte again normalized vectors
    for l in range(numLabels):
        if (HDParams.roundingTypeForHDVectors != 'noRounding'):
            ModelVectorsNorm[l, :] = (ModelVectors[l, :] > int(math.floor(numAddedVecPerClass[l] / 2))) * 1
        else:
            ModelVectorsNorm[l, :] = (ModelVectors[l, :] / numAddedVecPerClass[l])
    return (ModelVectors, ModelVectorsNorm, numAddedVecPerClass)

def retrainModelVecOnData_MultiClass(data, trueLabels, model, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, ModelVectors_Seiz, ModelVectors_NonSeiz,numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz , HDParams, type, factor, vecType):
    numLabels =2 #seiz and non seiz
    (numSubclassSeiz, D)=ModelVectorsNorm_Seiz.shape
    (numSubclassNonSeiz, D) = ModelVectorsNorm_NonSeiz.shape
    # ModelVectorsNorm=np.zeros((numLabels, D))
    (numWin, numFeat) = data.shape
    predLabels = np.zeros(numWin)
    predLabels_nonBin = np.zeros(numWin)
    predLabels_Seiz = np.zeros(numWin)
    predLabels_NonSeiz = np.zeros(numWin)
    distances_Seiz = np.zeros((numWin, numSubclassSeiz))
    distances_NonSeiz = np.zeros((numWin, numSubclassNonSeiz))
    for s in range(numWin):
        (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        tempVec=temp.cpu().numpy()
        (predLabels_Seiz[s], distances_Seiz[s,:]) = givePrediction(tempVec, ModelVectorsNorm_Seiz, HDParams.similarityType, vecType)
        (predLabels_NonSeiz[s], distances_NonSeiz[s, :]) = givePrediction(tempVec, ModelVectorsNorm_NonSeiz, HDParams.similarityType, vecType)
        #find the closest distance
        minSeiz_Indx=np.argmin(distances_Seiz[s,:])
        minSeiz_Dist = distances_Seiz[s,minSeiz_Indx]
        minNonSeiz_Indx = np.argmin(distances_NonSeiz[s, :])
        minNonSeiz_Dist = distances_NonSeiz[s, minNonSeiz_Indx]
        if (minSeiz_Dist< minNonSeiz_Dist): #sizure is final prediction
            predLabels_nonBin[s]=predLabels_Seiz[s]+1
            predLabels[s] =1
        else: #nonSeiz is final predictions
            predLabels_nonBin[s] = -predLabels_NonSeiz[s]-1
            predLabels[s] =0

        #if labels is wrong
        if (predLabels[s]!=trueLabels[s]):
            # add to the correct class again - to the close subclass
            lab = int(trueLabels[s])  # -1
            if (lab==1): #Seiz
                ModelVectors_Seiz[minSeiz_Indx, :] = ModelVectors_Seiz[minSeiz_Indx, :] + factor * tempVec
                numAddedVecPerClass_Seiz[minSeiz_Indx] = numAddedVecPerClass_Seiz[minSeiz_Indx] + factor * 1
            else: #NonSeiz
                ModelVectors_NonSeiz[minNonSeiz_Indx, :] = ModelVectors_NonSeiz[minNonSeiz_Indx, :] + factor * tempVec
                numAddedVecPerClass_NonSeiz[minNonSeiz_Indx] = numAddedVecPerClass_NonSeiz[minNonSeiz_Indx] + factor * 1

            # remove from wrong class
            if (type=='AddAndSubtract'):
                lab = int(predLabels[s])  # -1
                if (lab == 1):  # Seiz
                    ModelVectors_Seiz[minSeiz_Indx, :] = ModelVectors_Seiz[minSeiz_Indx, :] - factor * tempVec
                    numAddedVecPerClass_Seiz[minSeiz_Indx] = numAddedVecPerClass_Seiz[minSeiz_Indx] - factor * 1
                else:  # NonSeiz
                    ModelVectors_NonSeiz[minNonSeiz_Indx, :] = ModelVectors_NonSeiz[minNonSeiz_Indx,:] - factor * tempVec
                    numAddedVecPerClass_NonSeiz[minNonSeiz_Indx] = numAddedVecPerClass_NonSeiz[minNonSeiz_Indx] -factor * 1


    #calcualte again normalized vectors
    for l in range(numSubclassSeiz):
        if (HDParams.roundingTypeForHDVectors != 'noRounding'):
            ModelVectorsNorm_Seiz[l, :] = (ModelVectors_Seiz[l, :] > int(math.floor(numAddedVecPerClass_Seiz[l] / 2))) * 1
        else:
            ModelVectorsNorm_Seiz[l, :] = (ModelVectors_Seiz[l, :] / numAddedVecPerClass_Seiz[l])
    for l in range(numSubclassNonSeiz):
        if (HDParams.roundingTypeForHDVectors != 'noRounding'):
            ModelVectorsNorm_NonSeiz[l, :] = (ModelVectors_NonSeiz[l, :] > int(math.floor(numAddedVecPerClass_NonSeiz[l] / 2))) * 1
        else:
            ModelVectorsNorm_NonSeiz[l, :] = (ModelVectors_NonSeiz[l, :] / numAddedVecPerClass_NonSeiz[l])

    return ( ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, ModelVectors_Seiz, ModelVectors_NonSeiz,numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz)

def retrainModelVecOnData_MultiClass_v2(data, trueLabels, model, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, ModelVectors_Seiz, ModelVectors_NonSeiz,numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz , HDParams, type, factor, vecType):
    '''correcting for bin with AddAndSubtract'''
    numWrongPred=0
    numLabels =2 #seiz and non seiz
    (numSubclassSeiz, D)=ModelVectorsNorm_Seiz.shape
    (numSubclassNonSeiz, D) = ModelVectorsNorm_NonSeiz.shape
    # ModelVectorsNorm=np.zeros((numLabels, D))
    (numWin, numFeat) = data.shape
    predLabels = np.zeros(numWin)
    predLabels_nonBin = np.zeros(numWin)
    predLabels_Seiz = np.zeros(numWin)
    predLabels_NonSeiz = np.zeros(numWin)
    distances_Seiz = np.zeros((numWin, numSubclassSeiz))
    distances_NonSeiz = np.zeros((numWin, numSubclassNonSeiz))
    ModelVectors_Seiz_Adding = np.zeros((numSubclassSeiz, D))
    ModelVectors_NonSeiz_Adding = np.zeros((numSubclassNonSeiz, D))
    numAddedVecPerClass_Seiz_Adding = np.zeros((numSubclassSeiz))
    numAddedVecPerClass_NonSeiz_Adding = np.zeros((numSubclassNonSeiz))
    if (vecType=='bin' and type=='AddAndSubtract'):
        ModelVectors_Seiz_Subtracting = np.zeros((numSubclassSeiz, D))
        ModelVectors_NonSeiz_Subtracting = np.zeros((numSubclassNonSeiz, D))
        numAddedVecPerClass_Seiz_Subtracting= np.zeros((numSubclassSeiz))
        numAddedVecPerClass_NonSeiz_Subtracting = np.zeros((numSubclassNonSeiz))

    for s in range(numWin):
        if (vecType == 'bin'):
            (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        elif (vecType == 'bipol'):
            (temp, timeHDVec) = model.learn_HD_proj_bipolar(data[s, :], HDParams)
        tempVec=temp.cpu().numpy()
        (predLabels_Seiz[s], distances_Seiz[s,:]) = givePrediction(tempVec, ModelVectorsNorm_Seiz, HDParams.similarityType, vecType)
        (predLabels_NonSeiz[s], distances_NonSeiz[s, :]) = givePrediction(tempVec, ModelVectorsNorm_NonSeiz, HDParams.similarityType, vecType)
        #find the closest distance
        minSeiz_Indx=np.argmin(distances_Seiz[s,:])
        minSeiz_Dist = distances_Seiz[s,minSeiz_Indx]
        minNonSeiz_Indx = np.argmin(distances_NonSeiz[s, :])
        minNonSeiz_Dist = distances_NonSeiz[s, minNonSeiz_Indx]
        if (minSeiz_Dist< minNonSeiz_Dist): #sizure is final prediction
            predLabels_nonBin[s]=predLabels_Seiz[s]+1
            predLabels[s] =1
        else: #nonSeiz is final predictions
            predLabels_nonBin[s] = -predLabels_NonSeiz[s]-1
            predLabels[s] =0

        # if labels is wrong
        if (predLabels[s] != trueLabels[s]):
            numWrongPred=numWrongPred+1
            if (vecType == 'bipol'):
                # add to the correct class again - to the close subclass
                lab = int(trueLabels[s])  # -1
                if (lab==1): #Seiz
                    ModelVectors_Seiz[minSeiz_Indx, :] = ModelVectors_Seiz[minSeiz_Indx, :] + factor * tempVec
                    numAddedVecPerClass_Seiz[minSeiz_Indx] = numAddedVecPerClass_Seiz[minSeiz_Indx] + factor * 1
                else: #NonSeiz
                    ModelVectors_NonSeiz[minNonSeiz_Indx, :] = ModelVectors_NonSeiz[minNonSeiz_Indx, :] + factor * tempVec
                    numAddedVecPerClass_NonSeiz[minNonSeiz_Indx] = numAddedVecPerClass_NonSeiz[minNonSeiz_Indx] + factor * 1

                # remove from wrong class
                if (type=='AddAndSubtract'):
                    lab = int(predLabels[s])  # -1
                    if (lab == 1):  # Seiz
                        ModelVectors_Seiz[minSeiz_Indx, :] = ModelVectors_Seiz[minSeiz_Indx, :] - factor * tempVec
                        numAddedVecPerClass_Seiz[minSeiz_Indx] = numAddedVecPerClass_Seiz[minSeiz_Indx] + factor # - factor * 1 #for bipolar
                    else:  # NonSeiz
                        ModelVectors_NonSeiz[minNonSeiz_Indx, :] = ModelVectors_NonSeiz[minNonSeiz_Indx,:] - factor * tempVec
                        numAddedVecPerClass_NonSeiz[minNonSeiz_Indx] = numAddedVecPerClass_NonSeiz[minNonSeiz_Indx] + factor # -factor * 1 #for bipolar

            elif (vecType == 'bin'):
                # add to the correct class again - to the close subclass
                lab = int(trueLabels[s])  # -1
                if (lab == 1):  # Seiz
                    ModelVectors_Seiz_Adding[minSeiz_Indx, :] = ModelVectors_Seiz_Adding[minSeiz_Indx, :] + factor * tempVec
                    numAddedVecPerClass_Seiz_Adding[minSeiz_Indx] = numAddedVecPerClass_Seiz_Adding[minSeiz_Indx] + factor * 1
                else:  # NonSeiz
                    ModelVectors_NonSeiz_Adding[minNonSeiz_Indx, :] = ModelVectors_NonSeiz_Adding[minNonSeiz_Indx, :] + factor * tempVec
                    numAddedVecPerClass_NonSeiz_Adding[minNonSeiz_Indx] = numAddedVecPerClass_NonSeiz_Adding[  minNonSeiz_Indx] + factor * 1

                # remove from wrong class
                if (type == 'AddAndSubtract'):
                    lab = int(predLabels[s])  # -1
                    if (lab == 1):  # Seiz
                        ModelVectors_Seiz_Subtracting[minSeiz_Indx, :] = ModelVectors_Seiz_Subtracting[minSeiz_Indx, :] + factor * tempVec
                        numAddedVecPerClass_Seiz_Subtracting[minSeiz_Indx] = numAddedVecPerClass_Seiz_Subtracting[minSeiz_Indx] + factor * 1
                    else:  # NonSeiz
                        ModelVectors_NonSeiz_Subtracting[minNonSeiz_Indx, :] = ModelVectors_NonSeiz_Subtracting[minNonSeiz_Indx, :] + factor * tempVec
                        numAddedVecPerClass_NonSeiz_Subtracting[minNonSeiz_Indx] = numAddedVecPerClass_NonSeiz_Subtracting[  minNonSeiz_Indx] + factor * 1

    #calcualte again normalized vectors
    if (vecType == 'bipol'):
        for l in range(numSubclassSeiz):
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                #ModelVectorsNorm_Seiz[l, :] = (ModelVectors_Seiz[l, :] > int(math.floor(numAddedVecPerClass_Seiz[l] / 2))) * 1
                ModelVectorsNorm_Seiz[l, :] = (ModelVectors_Seiz[l, :] > 0) * 1 #for bipolar
                ModelVectorsNorm_Seiz[l, ModelVectorsNorm_Seiz[l, :] == 0] = -1
            else:
                ModelVectorsNorm_Seiz[l, :] = (ModelVectors_Seiz[l, :] / numAddedVecPerClass_Seiz[l])
        for l in range(numSubclassNonSeiz):
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                #ModelVectorsNorm_NonSeiz[l, :] = (ModelVectors_NonSeiz[l, :] > int(math.floor(numAddedVecPerClass_NonSeiz[l] / 2))) * 1
                ModelVectorsNorm_NonSeiz[l, :] = (ModelVectors_NonSeiz[l, :] > 0) * 1 #for bipolar
                ModelVectorsNorm_NonSeiz[l, ModelVectorsNorm_NonSeiz[l, :] == 0] = -1
            else:
                ModelVectorsNorm_NonSeiz[l, :] = (ModelVectors_NonSeiz[l, :] / numAddedVecPerClass_NonSeiz[l])
    if (vecType == 'bin'):
        if (type == 'AddOnly'):
            for l in range(numSubclassSeiz):
                ModelVectors_Seiz[l, :] = ModelVectors_Seiz[l, :] + ModelVectors_Seiz_Adding[l, :]
                numAddedVecPerClass_Seiz[l] = numAddedVecPerClass_Seiz[l] + numAddedVecPerClass_Seiz_Adding[l]
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    ModelVectorsNorm_Seiz[l, :] = (ModelVectors_Seiz[l, :] > int(math.floor(numAddedVecPerClass_Seiz[l] / 2))) * 1
                else:
                    ModelVectorsNorm_Seiz[l, :] = (ModelVectors_Seiz[l, :] / numAddedVecPerClass_Seiz[l])
            for l in range(numSubclassNonSeiz):
                ModelVectors_NonSeiz[l, :] = ModelVectors_NonSeiz[l, :] + ModelVectors_NonSeiz_Adding[l, :]
                numAddedVecPerClass_NonSeiz[l] = numAddedVecPerClass_NonSeiz[l] + numAddedVecPerClass_NonSeiz_Adding[l]
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    ModelVectorsNorm_NonSeiz[l, :] = (ModelVectors_NonSeiz[l, :] > int(math.floor(numAddedVecPerClass_NonSeiz[l] / 2))) * 1
                else:
                    ModelVectorsNorm_NonSeiz[l, :] = (ModelVectors_NonSeiz[l, :] / numAddedVecPerClass_NonSeiz[l])
        elif (type == 'AddAndSubtract'):
            for l in range(numSubclassSeiz):
                ModelVectors_Seiz_AddingCorr = ModelVectors_Seiz_Adding[l, :] *2 - numAddedVecPerClass_Seiz_Adding[l]
                ModelVectors_Seiz_SubtractingCorr = ModelVectors_Seiz_Subtracting[l, :] * 2 - numAddedVecPerClass_Seiz_Subtracting[l]
                ModelVectors_Seiz[l, :] = ModelVectors_Seiz[l, :] + ModelVectors_Seiz_AddingCorr -ModelVectors_Seiz_SubtractingCorr
                numAddedVecPerClass_Seiz[l] = numAddedVecPerClass_Seiz[l] + numAddedVecPerClass_Seiz_Adding[l] +numAddedVecPerClass_Seiz_Subtracting[l]
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    ModelVectorsNorm_Seiz[l, :] = (ModelVectors_Seiz[l, :] > 0) * 1
                else:
                    ModelVectorsNorm_Seiz[l, :] = (ModelVectors_Seiz[l, :] / numAddedVecPerClass_Seiz[l])
            for l in range(numSubclassNonSeiz):
                ModelVectors_NonSeiz_AddingCorr = ModelVectors_NonSeiz_Adding[l, :] *2 - numAddedVecPerClass_NonSeiz_Adding[l]
                ModelVectors_NonSeiz_SubtractingCorr = ModelVectors_NonSeiz_Subtracting[l, :] * 2 - numAddedVecPerClass_NonSeiz_Subtracting[l]
                ModelVectors_NonSeiz[l, :] = ModelVectors_NonSeiz[l, :] + ModelVectors_NonSeiz_AddingCorr -ModelVectors_NonSeiz_SubtractingCorr
                numAddedVecPerClass_NonSeiz[l] = numAddedVecPerClass_NonSeiz[l] + numAddedVecPerClass_NonSeiz_Adding[l] +numAddedVecPerClass_NonSeiz_Subtracting[l]
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    ModelVectorsNorm_NonSeiz[l, :] = (ModelVectors_NonSeiz[l, :] > 0) * 1
                else:
                    ModelVectorsNorm_NonSeiz[l, :] = (ModelVectors_NonSeiz[l, :] / numAddedVecPerClass_NonSeiz[l])

    return ( ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, ModelVectors_Seiz, ModelVectors_NonSeiz,numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz, numWrongPred/numWin)



def totalRetrainingAsLongAsNeeded(data_train_Discr, label_train, data_test_Discr, label_test, model, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz,ModelVectors_Seiz, ModelVectors_NonSeiz,numAddedVecPerClass_Seiz,
                                  numAddedVecPerClass_NonSeiz , HDParams, GeneralParams, SegSymbParams, optType, ItterType,  ItterFact,ItterImprovThresh,  savingStepData, folderOut, fileName2, vecType='bin'):
    createFolderIfNotExists(folderOut)

    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
    numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
    toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz / SegSymbParams.slidWindStepSec)

    AllRes_train=np.zeros((1,32))
    AllRes_train0 = np.zeros((1, 32))
    #MEASURE INITIAL PEFROMANCE
    (acc_train, distWhenWrongPredict_train, distFromCorr_AllClass_train, distFromWrong_AllClass_train,  predLabels_train,
     predLabels_nonBin_train) = testModelVecOnData_Multiclass(data_train_Discr, label_train, model, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, HDParams, vecType)
    AllRes_train[0,0:4] = np.hstack( (acc_train, distFromCorr_AllClass_train, distFromWrong_AllClass_train, distWhenWrongPredict_train))
    AllRes_train[0, 4:13] = performance_all9(predLabels_train, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_train, seizureStableLenToTestIndx,   seizureStablePercToTest, distanceBetweenSeizuresIndx)
    AllRes_train[0, 13:22] = performance_all9(yPred_SmoothOurStep1, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    AllRes_train[0, 22:31] = performance_all9(yPred_SmoothOurStep2, label_train, toleranceFP_bef,toleranceFP_aft, numLabelsPerHour)

    if (savingStepData==1):
        AllRes_test = np.zeros((1, 31))
        AllRes_test0 = np.zeros((1, 31))
        (acc_test, distWhenWrongPredict_test, distFromCorr_AllClass_test, distFromWrong_AllClass_test, predLabels_test,
         predLabels_nonBin_test) = testModelVecOnData_Multiclass(data_test_Discr, label_test, model,  ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz,HDParams, vecType)
        AllRes_test[0,0:4] = np.hstack( (acc_test, distFromCorr_AllClass_test, distFromWrong_AllClass_test, distWhenWrongPredict_test))
        AllRes_test[0,4:13] = performance_all9(predLabels_test, label_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_test, seizureStableLenToTestIndx,seizureStablePercToTest, distanceBetweenSeizuresIndx)
        AllRes_test[0,13:22] = performance_all9(yPred_SmoothOurStep1, label_test, toleranceFP_bef, toleranceFP_aft,  numLabelsPerHour)
        AllRes_test[0,22:31] = performance_all9(yPred_SmoothOurStep2, label_test, toleranceFP_bef, toleranceFP_aft,numLabelsPerHour)

    #if vectors are binary (0,1) transform not norm vectors as -1,1 since this is the only way to do addAndSubtract correctly
    if (vecType=='bin' and ItterType!='AddOnly'):
        for l in range(len( ModelVectors_Seiz[:, 0])):
            ModelVectors_Seiz[l, :]= ModelVectors_Seiz[l, :] * 2 - numAddedVecPerClass_Seiz[l]
        for l in range(len( ModelVectors_NonSeiz[:, 0])):
            ModelVectors_NonSeiz[l, :]= ModelVectors_NonSeiz[l, :] * 2 - numAddedVecPerClass_NonSeiz[l]

    ModelVectorsNorm_Seiz_Opt = np.copy(ModelVectorsNorm_Seiz)
    ModelVectorsNorm_NonSeiz_Opt = np.copy(ModelVectorsNorm_NonSeiz)
    ModelVectors_Seiz_Opt = np.copy(ModelVectors_Seiz)
    ModelVectors_NonSeiz_Opt = np.copy(ModelVectors_NonSeiz)
    numAddedVecPerClass_Seiz_Opt = np.copy(numAddedVecPerClass_Seiz)
    numAddedVecPerClass_NonSeiz_Opt = np.copy(numAddedVecPerClass_NonSeiz)



    if (optType == 'F1DEgmean'):
        perfIndx=11 #F1DEgmean noSmooth
    elif (optType == 'simpleAcc'):
        perfIndx=0

    accOld = AllRes_train[0,perfIndx]
    initAcc = AllRes_train[0,perfIndx]
    improv = []
    i = 0
    stopItter = 'no'

    while (i < 2 or stopItter == 'no'):
        (ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, ModelVectors_Seiz, ModelVectors_NonSeiz,numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz, percWrong0) =  retrainModelVecOnData_MultiClass_v2\
            (data_train_Discr, label_train, model, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, ModelVectors_Seiz, ModelVectors_NonSeiz,numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz, HDParams, ItterType,  ItterFact, vecType)

        (acc_train, distWhenWrongPredict_train, distFromCorr_AllClass_train, distFromWrong_AllClass_train, predLabels_train,
         predLabels_nonBin_train) = testModelVecOnData_Multiclass(data_train_Discr, label_train, model,ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz,HDParams, vecType)
        AllRes_train0[0,0:4] = np.hstack((acc_train, distFromCorr_AllClass_train, distFromWrong_AllClass_train, distWhenWrongPredict_train))
        AllRes_train0[0,4:13] = performance_all9(predLabels_train, label_train, toleranceFP_bef, toleranceFP_aft,numLabelsPerHour)
        (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_train, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx)
        AllRes_train0[0,13:22] = performance_all9(yPred_SmoothOurStep1, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        AllRes_train0[0,22:31] = performance_all9(yPred_SmoothOurStep2, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        AllRes_train0[0,31]=percWrong0
        AllRes_train = np.vstack((AllRes_train,AllRes_train0))

        improv.append(AllRes_train0[0,perfIndx] - accOld)
        accOld = AllRes_train0[0,perfIndx]
        improvFromBeggining = AllRes_train0[0,perfIndx] - initAcc
        print('Itter: ', i, ' -> acc_train: ', AllRes_train0[0,perfIndx] * 100, 'improv: ', improv[-1] * 100, 'improvFromBeg: ', improvFromBeggining * 100)
        i = i + 1

        if (savingStepData == 1):
            (acc_test, distWhenWrongPredict_test, distFromCorr_AllClass_test, distFromWrong_AllClass_test,predLabels_test,
             predLabels_nonBin_test) = testModelVecOnData_Multiclass(data_test_Discr, label_test, model, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, HDParams, vecType)
            AllRes_test0[0,0:4] = np.hstack((acc_test, distFromCorr_AllClass_test, distFromWrong_AllClass_test, distWhenWrongPredict_test))
            AllRes_test0[0,4:13] = performance_all9(predLabels_test, label_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
            (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_test, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx)
            AllRes_test0[0,13:22] = performance_all9(yPred_SmoothOurStep1, label_test, toleranceFP_bef, toleranceFP_aft,numLabelsPerHour)
            AllRes_test0[0,22:31] = performance_all9(yPred_SmoothOurStep2, label_test, toleranceFP_bef, toleranceFP_aft,numLabelsPerHour)
            AllRes_test = np.vstack((AllRes_test, AllRes_test0))

            outputName = folderOut + '/' + fileName2 + '_ItterLearning_TrainRes.csv'
            np.savetxt(outputName, AllRes_train, delimiter=",")
            outputName = folderOut + '/' + fileName2 + '_ItterLearning_TestRes.csv'
            np.savetxt(outputName, AllRes_test, delimiter=",")

        if (i > 30):
            localStopReason=-1
            break


        stopItter = 'no'
        localStopReason = 0
        # if worse then on the beginning
        if (improvFromBeggining < 0):
            stopItter = 'yes'
            localStopReason = localStopReason + 100
            print('ERROR - worse then what we started with ')

        if (i > 2):
            # if in last three itterations improvement was positive and  but saturating
            if (np.abs(improv[i - 1]) < ItterImprovThresh and np.abs(improv[i - 2]) < ItterImprovThresh and np.abs(improv[i - 3]) < ItterImprovThresh):
                stopItter = 'yes'
                localStopReason = localStopReason + 1
            # if more then 2 times in last 3 iterations were negative improvement
            if (np.sum(np.asarray(improv[i - 3:i]) <= 0) > 2):
                stopItter = 'yes'
                localStopReason = localStopReason + 10
            # if worse then on the beginning
            if (improvFromBeggining < 0):
                stopItter = 'yes'
                localStopReason = localStopReason + 100
                print('ERROR - worse then what we started with ')
        else:
            ModelVectorsNorm_Seiz_Opt=np.copy(ModelVectorsNorm_Seiz)
            ModelVectorsNorm_NonSeiz_Opt=np.copy(ModelVectorsNorm_NonSeiz)
            ModelVectors_Seiz_Opt=np.copy(ModelVectors_Seiz)
            ModelVectors_NonSeiz_Opt=np.copy(ModelVectors_NonSeiz)
            numAddedVecPerClass_Seiz_Opt=np.copy(numAddedVecPerClass_Seiz)
            numAddedVecPerClass_NonSeiz_Opt=np.copy(numAddedVecPerClass_NonSeiz)

    print(' ==> stop reason', localStopReason, 'acc_train: ',  AllRes_train0[0,perfIndx] * 100)
    return (ModelVectorsNorm_Seiz_Opt, ModelVectorsNorm_NonSeiz_Opt, ModelVectors_Seiz_Opt, ModelVectors_NonSeiz_Opt,numAddedVecPerClass_Seiz_Opt, numAddedVecPerClass_NonSeiz_Opt)

def onlineHD_ModelVecOnData(data, trueLabels, model, HDParams, type, vecType='bin'):
    # we need to use vectors with -1 and 1 instead of 0 and 1!!!

    ModelVectorsNorm = np.zeros((2, HDParams.D))
    ModelVectors = np.zeros((2, HDParams.D))
    numAddedVecPerClass = np.zeros((2))  # np.array([0, 0])

    (numLabels, D)=ModelVectors.shape
    # ModelVectorsNorm=np.zeros((numLabels, D))
    (numWin, numFeat) = data.shape
    predLabels = np.zeros(numWin)
    weights=np.zeros(numWin)
    for s in range(numWin):
        if (vecType=='bin'):
            (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
            tempVec = temp.cpu().numpy()
            tempVecBipol = np.copy(tempVec)
            indx0 = np.where(tempVec == 0)
            tempVecBipol[indx0] = -1
        elif (vecType=='bipol'):
            (temp, timeHDVec) = model.learn_HD_proj_bipolar(data[s, :], HDParams)
            tempVec = temp.cpu().numpy()
            tempVecBipol = np.copy(tempVec)
        (predLabels[s], distances) = givePrediction(tempVec, ModelVectorsNorm, HDParams.similarityType)

        # add to the correct class
        lab = int(trueLabels[s])  # -1
        weight = distances[lab] #if very similar weight it smaller
        weights[s]=weight
        if (weight<0):
            print('weight negativ?')
        ModelVectors[lab, :] = ModelVectors[lab, :] + weight*tempVecBipol
        numAddedVecPerClass[lab] = numAddedVecPerClass[lab] + weight*1

        # remove from wrong class
        if (type=='AddAndSubtract' and predLabels[s]!=trueLabels[s]):
            lab = int(predLabels[s])  # -1
            weight = 1- distances[lab] #dist from wrong class, if small distance weight it more
            ModelVectors[lab, :] = ModelVectors[lab, :] -  weight*tempVecBipol
            numAddedVecPerClass[lab] = numAddedVecPerClass[lab] +  weight*1# -  weight*1 #since values are 1 and 1 it has to be + for correct normalization

        #NORM  VECTORS HAVE TO BE UPDATE AFTER EACH STEP BECAUSE WEIGHTS CHANGE AS MORE DATA IS ADDED !!!
        #calcualte again normalized vectors
        for l in range(numLabels):
            if (HDParams.roundingTypeForHDVectors != 'noRounding'):  # rounding to 1 and 0
                ModelVectorsNorm[l, (ModelVectors[l, :] > 0)] =  1 #because of -1,1 looking for bigger then 0
                if (vecType == 'bipol'):
                    ModelVectorsNorm[l, (ModelVectors[l, :] < 0)] = -1
                elif (vecType == 'bin'):
                    ModelVectorsNorm[l, (ModelVectors[l, :] < 0)] = 0
                    #ModelVectorsNorm[l, ModelVectorsNorm[l, :] == 0] = -1
            else:  # no rounding having float values
                ModelVectorsNorm[l, :] = (ModelVectors[l, :] / numAddedVecPerClass[l])

        # if (type=='AddAndSubtract'):
        #     for l in range(numLabels):
        #         if (HDParams.roundingTypeForHDVectors != 'noRounding'):
        #             ModelVectorsNorm[l, :] = (ModelVectors[l, :] > 0) * 1 #because of -1,1 looking for bigger then 0
        #             #ModelVectorsNorm[l, :] = (ModelVectors[l, :] > math.floor(numAddedVecPerClass[l] / 2)) * 1
        #         else:
        #             print('ERRROR! Unsupported option!')
        #             #I would need to keep trakc of amoung of positivly summed and negativly and then shift for difference
        #             # and then scalw positive values with this number, negative with neg sum to get float values between -1 and 1
        #             # and then shift and scale to have them between 0 and 1
        #             #this is not needed at the moment so skiping this implementation
        #             #ModelVectorsNorm[l, :] = (ModelVectors[l, :] / numAddedVecPerClass[l])
        # else:
        #     for l in range(numLabels):
        #         if (HDParams.roundingTypeForHDVectors != 'noRounding'): #rounding to 1 and 0
        #             ModelVectorsNorm[l, :] = (ModelVectors[l, :] > math.floor(numAddedVecPerClass[l] / 2))  * 1
        #         else: #no rounding having float values
        #             ModelVectorsNorm[l, :] = (ModelVectors[l, :] / numAddedVecPerClass[l])

    return (ModelVectors, ModelVectorsNorm, numAddedVecPerClass, weights)


def func_calculateFeaturesForInputFiles(SigInfoParams, SegSymbParams, GeneralParams, HDParams, folderIn, folderOut):
    ''' for every input file segments into windows and calculates features
    and saves to the file with the same name, with the last column beeing labels '''

    for patIndx, pat in enumerate(GeneralParams.patients):
        numFiles = len(np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv')))
        print('-- Patient:', pat, 'NumSeizures:', numFiles)

        for fIndx, fileIn in enumerate(np.sort(glob.glob(folderIn + '/*chb' + pat + '*.csv'))):
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
            labels=segmentLabels(y, SegSymbParams, SigInfoParams)

            if (SegSymbParams.symbolType == 'StandardMLFeatures'):
                featureValues=calculateMLfeatures(X, HDParams, SegSymbParams, SigInfoParams)
            elif (SegSymbParams.symbolType == 'Amplitude'):
                featureValues=calculateMeanAmplitudeFeatures(X, SegSymbParams, SigInfoParams)
            elif (SegSymbParams.symbolType == '46Features'):
                featureValues_45 = calculateMLfeatures(X, HDParams, SegSymbParams, SigInfoParams)
                featureValues_Ampl=calculateMeanAmplitudeFeatures(X, SegSymbParams, SigInfoParams)
                for ch in range(len(SigInfoParams.chToKeep)):
                    if ch==0:
                        featureValues = np.hstack((featureValues_45[:,ch*HDParams.numFeat:(ch+1)*HDParams.numFeat], featureValues_Ampl[:,ch].reshape((-1,1))))
                    else:
                        featureValues = np.hstack((featureValues, featureValues_45[:,ch * 45:(ch + 1) * 45], featureValues_Ampl[:,ch].reshape((-1,1))))

            # save features
            dataToSave=np.vstack((featureValues.transpose(), labels)).transpose()
            outputName = folderOut + '/' + fileName2 + '.csv'
            np.savetxt(outputName, dataToSave, delimiter=",")


def reduceNumSubclasses_removingApproach(data_train, labelsTrue_train, data_test, labelsTrue_test,model, HDParams, ModelVectorsMulti_Seiz, ModelVectorsMulti_NonSeiz,ModelVectorsMultiNorm_Seiz, ModelVectorsMultiNorm_NonSeiz,numAddedVecPerClass_Seiz,
                                         numAddedVecPerClass_NonSeiz, numSteps, optType, perfDropThr, GeneralParams, SegSymbParams, folderOut, fileName, vecType='bin'):
    ''' function to minimize (optimize) number of subclasses by analysing performance drop after removing subclasses in steps
    measuring performance drop after each step and then deciding for the one that is last before too big performance drop
    returns reduced number of subclasses and its classes model vectors
    created based on testRemovingLessPopulatecClasses_v2 '''

    folderOut2 = folderOut + '/ItterativeRemovingSubclasses_numSteps'+ str(numSteps) +'/'
    createFolderIfNotExists(folderOut2)

    (initialNumSubclassesSeiz, D) = ModelVectorsMultiNorm_Seiz.shape
    (initialNumSubclassesNonSeiz, D) = ModelVectorsMultiNorm_NonSeiz.shape
    numAddedVecPerClass_Seiz_Perc = numAddedVecPerClass_Seiz/np.sum(numAddedVecPerClass_Seiz)
    numAddedVecPerClass_NonSeiz_Perc = numAddedVecPerClass_NonSeiz / np.sum(numAddedVecPerClass_NonSeiz)

    # calculating some parameters
    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
    numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
    toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz  / SegSymbParams.slidWindStepSec)

    numItter=numSteps+1
    numClassesKept=np.zeros((numItter,3)) # nonSeiz, seiz, sum
    perDataInClassesKept = np.zeros((numItter, 3))  # nonSeiz, seiz, sum
    distFromCorrClass_train=np.zeros(numItter)
    distFromWrongClass_train = np.zeros(numItter)
    distFromWrongMissclassifiedClass_train = np.zeros(numItter)
    distFromCorrClass_test=np.zeros(numItter)
    distFromWrongClass_test = np.zeros(numItter)
    distFromWrongMissclassifiedClass_test = np.zeros(numItter)
    simpleAccuracy_train= np.zeros((numItter,1))
    simpleAccuracy_test = np.zeros((numItter, 1))
    allPerfMeasures_train_noSmooth = np.zeros((numItter, 9))
    allPerfMeasures_train_step1 = np.zeros((numItter, 9))
    allPerfMeasures_train_step2 = np.zeros((numItter, 9))
    allPerfMeasures_test_noSmooth = np.zeros((numItter, 9))
    allPerfMeasures_test_step1 = np.zeros((numItter, 9))
    allPerfMeasures_test_step2 = np.zeros((numItter, 9))

    predLabels_train = np.zeros((len(labelsTrue_train ), numItter))
    predLabels_train_nonBin = np.zeros((len(labelsTrue_train ), numItter))
    predLabels_test= np.zeros((len(labelsTrue_test ), numItter))
    predLabels_test_nonBin = np.zeros((len(labelsTrue_test), numItter))

    # SORTING CLASSES
    # sort classes based on amount of data - in ascending order
    if (initialNumSubclassesSeiz!=1):
        sortSeizIndx=np.argsort(numAddedVecPerClass_Seiz_Perc)
        numAddedVecPerClass_Seiz_Sorted = numAddedVecPerClass_Seiz[sortSeizIndx]
        numAddedVecPerClass_Seiz_PercSorted=numAddedVecPerClass_Seiz_Perc[sortSeizIndx]
        ModelVectorsMultiNorm_SeizSorted=ModelVectorsMultiNorm_Seiz[sortSeizIndx,:]
        ModelVectorsMulti_SeizSorted = ModelVectorsMulti_Seiz[sortSeizIndx, :]
    else:
        sortSeizIndx=[0]
        numAddedVecPerClass_Seiz_Sorted = numAddedVecPerClass_Seiz
        numAddedVecPerClass_Seiz_PercSorted=numAddedVecPerClass_Seiz_Perc
        ModelVectorsMultiNorm_SeizSorted=ModelVectorsMultiNorm_Seiz
        ModelVectorsMulti_SeizSorted = ModelVectorsMulti_Seiz

    if (initialNumSubclassesNonSeiz != 1):
        sortNonSeizIndx=np.argsort(numAddedVecPerClass_NonSeiz_Perc)
        numAddedVecPerClass_NonSeiz_Sorted = numAddedVecPerClass_NonSeiz[sortNonSeizIndx]
        numAddedVecPerClass_NonSeiz_PercSorted=numAddedVecPerClass_NonSeiz_Perc[sortNonSeizIndx]
        ModelVectorsMultiNorm_NonSeizSorted=ModelVectorsMultiNorm_NonSeiz[sortNonSeizIndx,:]
        ModelVectorsMulti_NonSeizSorted = ModelVectorsMulti_NonSeiz[sortNonSeizIndx, :]
    else:
        sortNonSeizIndx = [0]
        numAddedVecPerClass_NonSeiz_Sorted = numAddedVecPerClass_NonSeiz
        numAddedVecPerClass_NonSeiz_PercSorted=numAddedVecPerClass_NonSeiz_Perc
        ModelVectorsMultiNorm_NonSeizSorted=ModelVectorsMultiNorm_NonSeiz
        ModelVectorsMulti_NonSeizSorted = ModelVectorsMulti_NonSeiz

    # IN ITTERATIONS REMOVE SUBCLASSES AND MEASURE PERFORMANCE DROP
    for i in range(numItter):
        numSeizSubclassesToRemove=int(i*initialNumSubclassesSeiz/numSteps)
        numNonSeizSubclassesToRemove =int(i * initialNumSubclassesNonSeiz / numSteps)
        if (numSeizSubclassesToRemove>=initialNumSubclassesSeiz):
            numSeizSubclassesToRemove=initialNumSubclassesSeiz-1
        if (numNonSeizSubclassesToRemove>=initialNumSubclassesNonSeiz):
            numNonSeizSubclassesToRemove=initialNumSubclassesNonSeiz-1

        #reducing amount of subclasses
        if (initialNumSubclassesSeiz - numSeizSubclassesToRemove> 1):
            ModelVectorsMultiNorm_Seiz_Red=ModelVectorsMultiNorm_SeizSorted[numSeizSubclassesToRemove:, :] #skip first numSeizSubclassesToRemove classes
            ModelVectorsMulti_Seiz_Red = ModelVectorsMulti_SeizSorted[numSeizSubclassesToRemove:, :]
            numAddedVecPerClass_Seiz_Red=numAddedVecPerClass_Seiz_Sorted[numSeizSubclassesToRemove:]
        else:
            ModelVectorsMultiNorm_Seiz_Red = ModelVectorsMultiNorm_SeizSorted[-1:]
            ModelVectorsMulti_Seiz_Red = ModelVectorsMulti_SeizSorted[-1:]
            numAddedVecPerClass_Seiz_Red = numAddedVecPerClass_Seiz_Sorted[-1:]

        if (initialNumSubclassesNonSeiz - numNonSeizSubclassesToRemove > 1):
            ModelVectorsMultiNorm_NonSeiz_Red = ModelVectorsMultiNorm_NonSeizSorted[numNonSeizSubclassesToRemove:, :]  # skip first i steps
            ModelVectorsMulti_NonSeiz_Red = ModelVectorsMulti_NonSeizSorted[numNonSeizSubclassesToRemove:, :]
            numAddedVecPerClass_NonSeiz_Red=numAddedVecPerClass_NonSeiz_Sorted[numNonSeizSubclassesToRemove:]
        else:
            ModelVectorsMultiNorm_NonSeiz_Red = ModelVectorsMultiNorm_NonSeizSorted[-1:]
            ModelVectorsMulti_NonSeiz_Red = ModelVectorsMulti_NonSeizSorted[-1:]
            numAddedVecPerClass_NonSeiz_Red = numAddedVecPerClass_NonSeiz_Sorted[-1:]

        numClassesKept[i,:]=[ len(ModelVectorsMultiNorm_NonSeiz_Red[:,0]),  len(ModelVectorsMultiNorm_Seiz_Red[:,0]),  len(ModelVectorsMultiNorm_NonSeiz_Red[:,0]) +len(ModelVectorsMultiNorm_Seiz_Red[:,0])]

        #percentage of data kept
        totalPercDataKept = (np.sum(numAddedVecPerClass_Seiz_Sorted[numSeizSubclassesToRemove:]) + np.sum(numAddedVecPerClass_NonSeiz_Sorted[numNonSeizSubclassesToRemove:])) / (np.sum(numAddedVecPerClass_NonSeiz_Sorted) + np.sum(numAddedVecPerClass_Seiz_Sorted))
        perDataInClassesKept[i, :] = [np.sum(numAddedVecPerClass_NonSeiz_PercSorted[numNonSeizSubclassesToRemove:]),np.sum(numAddedVecPerClass_Seiz_PercSorted[numSeizSubclassesToRemove:]),  totalPercDataKept]

        #train and measure performance - TRAIN
        (simpleAccuracy_train[i,0], distFromWrongMissclassifiedClass_train[i], distFromCorrClass_train[i], distFromWrongClass_train[i], predLabels_train[:,i], predLabels_nonBin0) = testModelVecOnData_Multiclass(data_train, labelsTrue_train, model, ModelVectorsMultiNorm_Seiz_Red, ModelVectorsMultiNorm_NonSeiz_Red, HDParams, vecType)
        #invert nonBin labels to mean always the same (the most persistent classes have smallest numbers)
        predLabels_nonBin1=predLabels_nonBin0
        intpos=np.where(predLabels_nonBin0>0)
        predLabels_nonBin1[intpos]=1+len(ModelVectorsMultiNorm_Seiz_Red[:,0])-predLabels_nonBin0[intpos]
        intneg=np.where(predLabels_nonBin0<0)
        predLabels_nonBin1[intneg]=-1-(len(ModelVectorsMultiNorm_NonSeiz_Red[:,0])+predLabels_nonBin0[intneg])
        predLabels_train_nonBin[:, i]=predLabels_nonBin1
        #smoothen labels
        (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_train[:,i], seizureStableLenToTestIndx, seizureStablePercToTest,  distanceBetweenSeizuresIndx)
        #caluclate all performance measures
        (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(predLabels_train[:,i], labelsTrue_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        allPerfMeasures_train_noSmooth[i,:]=[sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
        (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(yPred_SmoothOurStep1,labelsTrue_train, toleranceFP_bef,toleranceFP_aft, numLabelsPerHour)
        allPerfMeasures_train_step1[i, :] = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
        (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(yPred_SmoothOurStep2,labelsTrue_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        allPerfMeasures_train_step2[i, :] = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]

        #train and measure performance - TEST
        (simpleAccuracy_test[i,0], distFromWrongMissclassifiedClass_test[i], distFromCorrClass_test[i], distFromWrongClass_test[i], predLabels_test[:,i], predLabels_nonBin0) = testModelVecOnData_Multiclass(data_test, labelsTrue_test, model, ModelVectorsMultiNorm_Seiz_Red, ModelVectorsMultiNorm_NonSeiz_Red, HDParams, vecType)
        #invert nonBin labels to mean always the same (the most persistent classes have smallest numbers)
        predLabels_nonBin1=predLabels_nonBin0
        intpos=np.where(predLabels_nonBin0>0)
        predLabels_nonBin1[intpos]=1+len(ModelVectorsMultiNorm_Seiz_Red[:,0])-predLabels_nonBin0[intpos]
        intneg=np.where(predLabels_nonBin0<0)
        predLabels_nonBin1[intneg]=-1-(len(ModelVectorsMultiNorm_NonSeiz_Red[:,0])+predLabels_nonBin0[intneg])
        predLabels_test_nonBin[:, i]=predLabels_nonBin1
        #smoothen labels
        (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_test[:,i], seizureStableLenToTestIndx, seizureStablePercToTest,  distanceBetweenSeizuresIndx)
        #caluclate all performance measures
        (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(predLabels_test[:,i], labelsTrue_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        allPerfMeasures_test_noSmooth[i,:]=[sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
        (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(yPred_SmoothOurStep1,labelsTrue_test, toleranceFP_bef,toleranceFP_aft, numLabelsPerHour)
        allPerfMeasures_test_step1[i, :] = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
        (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(yPred_SmoothOurStep2,labelsTrue_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        allPerfMeasures_test_step2[i, :] = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]


        # CHECK IF PERFORMANCE DROPED TOO MUCH
        if (optType=='F1DEgmean'):
            #find optimal num to keep  - based on highest F1DEgmean performance to the right
            maxPerfIndx=np.argmax(allPerfMeasures_train_noSmooth[:, 7])
            limit=allPerfMeasures_train_noSmooth[maxPerfIndx, 7]-perfDropThr #tolerable 1% drop
            perfToCompare=allPerfMeasures_train_noSmooth[i, 7]
        elif (optType=='SimpleAcc'):
            # find optimal num to keep  - based on simple accuracy performance to the right
            maxPerfIndx = np.argmax(simpleAccuracy_train[:, 0])
            limit = simpleAccuracy_train[maxPerfIndx, 0] - perfDropThr # tolerable 1% drop
            perfToCompare=simpleAccuracy_train[i, 0]
        if (perfToCompare >= limit):
            optimalPerf_train = np.hstack((numClassesKept[i, :], perDataInClassesKept[i, :], simpleAccuracy_train[i, :],  allPerfMeasures_train_noSmooth[i, :], allPerfMeasures_train_step1[i, :], allPerfMeasures_train_step2[i, :]))
            optimalPerf_test = np.hstack((numClassesKept[i, :], perDataInClassesKept[i, :], simpleAccuracy_test[i, :],  allPerfMeasures_test_noSmooth[i, :],  allPerfMeasures_test_step1[i, :], allPerfMeasures_test_step2[i, :]))
            ModelVectorsMultiNorm_Seiz_Optimal = ModelVectorsMultiNorm_Seiz_Red
            ModelVectorsMultiNorm_NonSeiz_Optimal = ModelVectorsMultiNorm_NonSeiz_Red
            ModelVectorsMulti_Seiz_Optimal = ModelVectorsMulti_Seiz_Red
            ModelVectorsMulti_NonSeiz_Optimal = ModelVectorsMulti_NonSeiz_Red
            numAddedVecPerClass_Seiz_Optimal = numAddedVecPerClass_Seiz_Red
            numAddedVecPerClass_NonSeiz_Optimal = numAddedVecPerClass_NonSeiz_Red
        else:
            break

    #stack all data an save just in  case
    pom = np.vstack((distFromWrongMissclassifiedClass_train, distFromCorrClass_train, distFromWrongClass_train, distFromWrongMissclassifiedClass_test, distFromCorrClass_test, distFromWrongClass_test))
    dataToSave = np.hstack((numClassesKept, perDataInClassesKept, pom.transpose()))
    outputName = folderOut2 + '/' + fileName +'_VariousMeasuresPerItteration.csv'
    np.savetxt(outputName, dataToSave, delimiter=",")
    dataToSave = np.hstack((simpleAccuracy_train, allPerfMeasures_train_noSmooth, allPerfMeasures_train_step1, allPerfMeasures_train_step2))
    outputName = folderOut2 + '/' + fileName +'_PerformanceMeasuresPerItteration_Train.csv'
    np.savetxt(outputName, dataToSave, delimiter=",")
    dataToSave = np.hstack((simpleAccuracy_test, allPerfMeasures_test_noSmooth, allPerfMeasures_test_step1, allPerfMeasures_test_step2))
    outputName = folderOut2 + '/' + fileName +'_PerformanceMeasuresPerItteration_Test.csv'
    np.savetxt(outputName, dataToSave, delimiter=",")
    outputName = folderOut2 + '/' + fileName +'_LabelsPerItteration_Train.csv'
    np.savetxt(outputName, predLabels_train, delimiter=",")
    outputName = folderOut2 + '/' + fileName +'_LabelsPerItteration_Test.csv'
    np.savetxt(outputName, predLabels_test, delimiter=",")
    outputName = folderOut2 + '/' + fileName +'_LabelsNonBinPerItteration_Train.csv'
    np.savetxt(outputName, predLabels_train_nonBin, delimiter=",")
    outputName = folderOut2 + '/' + fileName +'_LabelsNonBinPerItteration_Test.csv'
    np.savetxt(outputName, predLabels_test_nonBin, delimiter=",")

    # #PLOTTING
    # fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    # gs = GridSpec(2, 3, figure=fig1)
    # fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    # fig1.suptitle('Vector distances vs number of subclasses')
    #
    # xValues=np.arange(0,1,1/numItter)
    # ax1 = fig1.add_subplot(gs[0, 0])
    # ax1.plot(xValues,  numClassesKept[:, 0], 'b')  # nonSeiz
    # ax1.plot(xValues, numClassesKept[:, 1], 'r')  # Seiz
    # ax1.plot(xValues, numClassesKept[:, 2], 'k')  # Total
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('Num classes kept')
    # ax1.legend(['nonSeiz', 'Seiz', 'Both'])
    # ax1.grid()
    # ax1 = fig1.add_subplot(gs[0, 1])
    # ax1.plot(xValues,  perDataInClassesKept[:, 0], 'b')  # nonSeiz
    # ax1.plot(xValues, perDataInClassesKept[:, 1], 'r')  # Seiz
    # ax1.plot(xValues, perDataInClassesKept[:, 2], 'k')  # Total
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('Perc data in kept classes')
    # ax1.legend(['nonSeiz', 'Seiz', 'Both'])
    # ax1.grid()
    # ax1 = fig1.add_subplot(gs[0, 2])
    # ax1.plot(xValues,  simpleAccuracy_train, 'k')
    # ax1.plot(xValues, simpleAccuracy_test, 'b--')
    # ax1.legend(['Train', 'Test'])
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('Simple accuracy')
    # ax1.grid()
    # #performances in more details before and after smoothing
    # ax1 = fig1.add_subplot(gs[1, 0])
    # ax1.plot(xValues,  allPerfMeasures_train_noSmooth[:, 2], 'k')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_train_step1[:, 2], 'b')  # Step1
    # ax1.plot(xValues, allPerfMeasures_train_step2[:, 2], 'm')  # Step2
    # ax1.plot(xValues,  allPerfMeasures_test_noSmooth[:, 2], 'k--')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_test_step1[:, 2], 'b--')  # Step1
    # ax1.plot(xValues, allPerfMeasures_test_step2[:, 2], 'm--')  # Step2
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('F1 Episodes')
    # ax1.legend(['noSmooth', 'Step1', 'Step2'])
    # ax1.grid()
    # ax1 = fig1.add_subplot(gs[1, 1])
    # ax1.plot(xValues,  allPerfMeasures_train_noSmooth[:, 5], 'k')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_train_step1[:, 5], 'b')  # Step1
    # ax1.plot(xValues, allPerfMeasures_train_step2[:, 5], 'm')  # Step2
    # ax1.plot(xValues,  allPerfMeasures_test_noSmooth[:, 5], 'k--')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_test_step1[:, 5], 'b--')  # Step1
    # ax1.plot(xValues, allPerfMeasures_test_step2[:, 5], 'm--')  # Step2
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('F1 Duration')
    # ax1.legend(['noSmooth', 'Step1', 'Step2'])
    # ax1.grid()
    # ax1 = fig1.add_subplot(gs[1, 2])
    # ax1.plot(xValues,  allPerfMeasures_train_noSmooth[:, 7], 'k')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_train_step1[:, 7], 'b')  # Step1
    # ax1.plot(xValues, allPerfMeasures_train_step2[:, 7], 'm')  # Step2
    # ax1.plot(xValues,  allPerfMeasures_test_noSmooth[:, 5], 'k--')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_test_step1[:, 5], 'b--')  # Step1
    # ax1.plot(xValues, allPerfMeasures_test_step2[:, 5], 'm--')  # Step2
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('F1 Gmean')
    # ax1.legend(['noSmooth', 'Step1', 'Step2'])
    # ax1.grid()
    # fig1.show()
    # fig1.savefig(folderOut2 + '/'+ fileName+ '_itterativeRemovingAllRes.png')
    # plt.close(fig1)

    print('Drop train:', (simpleAccuracy_train[i,:]-simpleAccuracy_train[0,:])*100, ' Drop test: ', (simpleAccuracy_test[i,:]-simpleAccuracy_test[0,:])*100)
    return (optimalPerf_train, optimalPerf_test, ModelVectorsMulti_Seiz_Optimal, ModelVectorsMulti_NonSeiz_Optimal, ModelVectorsMultiNorm_Seiz_Optimal, ModelVectorsMultiNorm_NonSeiz_Optimal, numAddedVecPerClass_Seiz_Optimal, numAddedVecPerClass_NonSeiz_Optimal)


def reduceNumSubclasses_clusteringApproach(data_train, labelsTrue_train, data_test, labelsTrue_test,model, HDParams,ModelVectorsMulti_Seiz, ModelVectorsMulti_NonSeiz, ModelVectorsMultiNorm_Seiz, ModelVectorsMultiNorm_NonSeiz,numAddedVecPerClass_Seiz, numAddedVecPerClass_NonSeiz, numSteps, optType,perfDropThr,  groupingThresh, GeneralParams, SegSymbParams, folderOut, fileName, vecType='bin'):
    ''' function to minimize (optimize) number of subclasses by analysing performance drop after clustering subclasses in steps
    measuring performance drop after each step and then deciding for the one that is last before too big performance drop
    returns reduced number of subclasses and its classes model vectors
    created based on testRemovingLessPopulatecClasses_clusteringApproach '''

    folderOut2 = folderOut + '/ItterativeClusteringSubclasses_numSteps'+ str(numSteps)+'_PercThr'+ str(groupingThresh) +'/'
    createFolderIfNotExists(folderOut2)

    (initialNumSubclassesSeiz, D) = ModelVectorsMultiNorm_Seiz.shape
    (initialNumSubclassesNonSeiz, D) = ModelVectorsMultiNorm_NonSeiz.shape
    totalNumSubclasses=initialNumSubclassesSeiz+initialNumSubclassesNonSeiz
    numSubclassesPerStep=int(totalNumSubclasses /numSteps)
    initialAmountOfDataAdded=(np.sum(numAddedVecPerClass_NonSeiz) + np.sum(numAddedVecPerClass_Seiz))

    numAddedVecPerClass_Seiz_Perc = numAddedVecPerClass_Seiz/np.sum(numAddedVecPerClass_Seiz)
    numAddedVecPerClass_NonSeiz_Perc = numAddedVecPerClass_NonSeiz / np.sum(numAddedVecPerClass_NonSeiz)

    #calculatng some parameters
    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
    numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
    toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz  / SegSymbParams.slidWindStepSec)


    # SORTING SUBCLASSES
    # sort classes based on amount of data - in ascenig order
    if (initialNumSubclassesSeiz!=1):
        sortSeizIndx=np.argsort(numAddedVecPerClass_Seiz_Perc)
        numAddedVecPerClass_Seiz_Sorted = numAddedVecPerClass_Seiz[sortSeizIndx]
        numAddedVecPerClass_Seiz_PercSorted=numAddedVecPerClass_Seiz_Perc[sortSeizIndx]
        ModelVectorsMultiNorm_SeizSorted=ModelVectorsMultiNorm_Seiz[sortSeizIndx,:]
        ModelVectorsMulti_SeizSorted = ModelVectorsMulti_Seiz[sortSeizIndx, :]
    else:
        sortSeizIndx=[0]
        numAddedVecPerClass_Seiz_Sorted = numAddedVecPerClass_Seiz
        numAddedVecPerClass_Seiz_PercSorted=numAddedVecPerClass_Seiz_Perc
        ModelVectorsMultiNorm_SeizSorted=ModelVectorsMultiNorm_Seiz
        ModelVectorsMulti_SeizSorted = ModelVectorsMulti_Seiz

    if (initialNumSubclassesNonSeiz != 1):
        sortNonSeizIndx=np.argsort(numAddedVecPerClass_NonSeiz_Perc)
        numAddedVecPerClass_NonSeiz_Sorted = numAddedVecPerClass_NonSeiz[sortNonSeizIndx]
        numAddedVecPerClass_NonSeiz_PercSorted=numAddedVecPerClass_NonSeiz_Perc[sortNonSeizIndx]
        ModelVectorsMultiNorm_NonSeizSorted=ModelVectorsMultiNorm_NonSeiz[sortNonSeizIndx,:]
        ModelVectorsMulti_NonSeizSorted = ModelVectorsMulti_NonSeiz[sortNonSeizIndx, :]
    else:
        sortNonSeizIndx = [0]
        numAddedVecPerClass_NonSeiz_Sorted = numAddedVecPerClass_NonSeiz
        numAddedVecPerClass_NonSeiz_PercSorted=numAddedVecPerClass_NonSeiz_Perc
        ModelVectorsMultiNorm_NonSeizSorted=ModelVectorsMultiNorm_NonSeiz
        ModelVectorsMulti_NonSeizSorted = ModelVectorsMulti_NonSeiz

    #store data before starting to reduce subclasses
    numClassesKept= [len(ModelVectorsMultiNorm_NonSeiz[:, 0]),len(ModelVectorsMultiNorm_Seiz[:, 0]),  len(ModelVectorsMultiNorm_NonSeiz[:, 0]) + len(ModelVectorsMultiNorm_Seiz[:, 0])]
    totalPercDataKept = (np.sum(numAddedVecPerClass_Seiz_Sorted) + np.sum( numAddedVecPerClass_NonSeiz_Sorted)) / initialAmountOfDataAdded
    perDataInClassesKept = [np.sum(numAddedVecPerClass_NonSeiz_PercSorted),  np.sum(numAddedVecPerClass_Seiz_PercSorted), totalPercDataKept]

    # train and measure performance - TRAIN
    (simpleAccuracy_train, distFromWrongMissclassifiedClass_train, distFromCorrClass_train, distFromWrongClass_train, predLabels_train, predLabels_nonBin0) = testModelVecOnData_Multiclass(
        data_train, labelsTrue_train, model, ModelVectorsMultiNorm_Seiz, ModelVectorsMultiNorm_NonSeiz, HDParams, vecType)
    # invert nonBin labels to mean always the same (the most persistent classes have smallest numbers)
    predLabels_nonBin1 = predLabels_nonBin0
    intpos = np.where(predLabels_nonBin0 > 0)
    predLabels_nonBin1[intpos] = 1 + len(ModelVectorsMultiNorm_Seiz[:, 0]) - predLabels_nonBin0[intpos]
    intneg = np.where(predLabels_nonBin0 < 0)
    predLabels_nonBin1[intneg] = -1 - (len(ModelVectorsMultiNorm_NonSeiz[:, 0]) + predLabels_nonBin0[intneg])
    predLabels_train_nonBin = predLabels_nonBin1
    # smoothen labels
    (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_train,  seizureStableLenToTestIndx,   seizureStablePercToTest,distanceBetweenSeizuresIndx)
    # caluclate all performance measures
    (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
        predLabels_train, labelsTrue_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    allPerfMeasures_train_noSmooth = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
    (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
        yPred_SmoothOurStep1, labelsTrue_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    allPerfMeasures_train_step1 = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
    (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
        yPred_SmoothOurStep2, labelsTrue_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    allPerfMeasures_train_step2= [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]

    # train and measure performance - TEST
    (simpleAccuracy_test, distFromWrongMissclassifiedClass_test, distFromCorrClass_test, distFromWrongClass_test,predLabels_test, predLabels_nonBin0) = testModelVecOnData_Multiclass(data_test, labelsTrue_test,
                                                                                   model, ModelVectorsMultiNorm_Seiz, ModelVectorsMultiNorm_NonSeiz,HDParams, vecType)
    # invert nonBin labels to mean always the same (the most persistent classes have smallest numbers)
    predLabels_nonBin1 = predLabels_nonBin0
    intpos = np.where(predLabels_nonBin0 > 0)
    predLabels_nonBin1[intpos] = 1 + len(ModelVectorsMultiNorm_Seiz[:, 0]) - predLabels_nonBin0[intpos]
    intneg = np.where(predLabels_nonBin0 < 0)
    predLabels_nonBin1[intneg] = -1 - (len(ModelVectorsMultiNorm_NonSeiz[:, 0]) + predLabels_nonBin0[intneg])
    predLabels_test_nonBin = predLabels_nonBin1
    # smoothen labels
    (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_test, seizureStableLenToTestIndx,   seizureStablePercToTest,  distanceBetweenSeizuresIndx)
    # caluclate all performance measures
    (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
        predLabels_test, labelsTrue_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    allPerfMeasures_test_noSmooth = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
    (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
        yPred_SmoothOurStep1, labelsTrue_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    allPerfMeasures_test_step1 = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean,  numFPperDay]
    (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
        yPred_SmoothOurStep2, labelsTrue_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    allPerfMeasures_test_step2 = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean,   numFPperDay]

    #saving to return
    optimalPerf_train = np.hstack((numClassesKept, perDataInClassesKept, simpleAccuracy_train,
                                   allPerfMeasures_train_noSmooth, allPerfMeasures_train_step1, allPerfMeasures_train_step2))
    optimalPerf_test = np.hstack((numClassesKept, perDataInClassesKept, simpleAccuracy_test,
                                  allPerfMeasures_test_noSmooth, allPerfMeasures_test_step1,  allPerfMeasures_test_step2))
    ModelVectorsMultiNorm_Seiz_Optimal = ModelVectorsMultiNorm_SeizSorted
    ModelVectorsMultiNorm_NonSeiz_Optimal = ModelVectorsMultiNorm_NonSeizSorted
    ModelVectorsMulti_Seiz_Optimal = ModelVectorsMulti_SeizSorted
    ModelVectorsMulti_NonSeiz_Optimal = ModelVectorsMulti_NonSeizSorted
    numAddedVecPerClass_Seiz_Optimal = numAddedVecPerClass_Seiz_Sorted
    numAddedVecPerClass_NonSeiz_Optimal = numAddedVecPerClass_NonSeiz_Sorted


    mergingSeizCnt=0
    mergingNonSeizCnt=0
    clusteringDone=0
    i=-1
    while (clusteringDone==0 ):
        i=i+1
        clusteringDone=1
        # caluclate pairwise distances matrixes
        (distMat_SN, distArr_SN) = calculateAllPairwiseDistancesOfVectors_returnMatrix(ModelVectorsMultiNorm_SeizSorted,  ModelVectorsMultiNorm_NonSeizSorted, vecType)

        if (len(ModelVectorsMultiNorm_SeizSorted[:, 0])!=1 ):
            (distMat_SS, distArr_SS)= calculateAllPairwiseDistancesOfVectors_returnMatrix(ModelVectorsMultiNorm_SeizSorted,ModelVectorsMultiNorm_SeizSorted, vecType )
            numSeizSubclasses= len(distMat_SS [0,:])
        else:
            numSeizSubclasses=1

        if (len(ModelVectorsMultiNorm_NonSeizSorted[:, 0]) != 1):
            (distMat_NN, distArr_NN) = calculateAllPairwiseDistancesOfVectors_returnMatrix(ModelVectorsMultiNorm_NonSeizSorted, ModelVectorsMultiNorm_NonSeizSorted, vecType)
            numNonSeizSubclasses = len(distMat_NN[0, :])
        else:
            numNonSeizSubclasses=1

        #GROUP SUBCLASSES IF POSSIBLE
        if (len(ModelVectorsMultiNorm_SeizSorted[:, 0]) != 1):
            # find the least populated one in seizure and calculate its distance to closest seizure
            try:
                minDistSeizIndx = np.argmin(distMat_SS[0, 1:])
                minDistSeiz = distMat_SS[0, 1 + minDistSeizIndx]
                averageSeizDistFromAllSubclasses = (np.nanmean(distMat_SS[0, 1:]) * (numSeizSubclasses - 1) + np.nanmean(distMat_SN[0, :]) * numNonSeizSubclasses) / (numSeizSubclasses - 1 + numNonSeizSubclasses)
            except:
                print('error')
                (distMat_SS, distArr_SS) = calculateAllPairwiseDistancesOfVectors_returnMatrix(
                    ModelVectorsMultiNorm_SeizSorted, ModelVectorsMultiNorm_SeizSorted, vecType)

            if (minDistSeiz < groupingThresh * averageSeizDistFromAllSubclasses):
                clusteringDone = 0
                mergingSeizCnt = mergingSeizCnt + 1
                newVectorSum = ModelVectorsMultiNorm_SeizSorted[minDistSeizIndx, :] * numAddedVecPerClass_Seiz_Sorted[ minDistSeizIndx] + ModelVectorsMultiNorm_SeizSorted[0, :] * numAddedVecPerClass_Seiz_Sorted[0]
                ModelVectorsMulti_SeizSorted[minDistSeizIndx, :] = ModelVectorsMulti_SeizSorted[minDistSeizIndx, :]  + ModelVectorsMulti_SeizSorted[0, :]
                numAddedVec = numAddedVecPerClass_Seiz_Sorted[minDistSeizIndx] + numAddedVecPerClass_Seiz_Sorted[0]
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    if (vecType=='bin'):
                        ModelVectorsMultiNorm_SeizSorted[minDistSeizIndx, :] = (newVectorSum > int(math.floor(numAddedVec / 2))) * 1
                    elif (vecType == 'bipol'):
                        ModelVectorsMultiNorm_SeizSorted[minDistSeizIndx, :] = (newVectorSum > 0) * 1
                        ModelVectorsMultiNorm_SeizSorted[minDistSeizIndx, ModelVectorsMultiNorm_SeizSorted[minDistSeizIndx, :] == 0] = -1
                else:
                    ModelVectorsMultiNorm_SeizSorted[minDistSeizIndx, :] = (newVectorSum / numAddedVec)
                ModelVectorsMultiNorm_SeizSorted = ModelVectorsMultiNorm_SeizSorted[1:, :]  # remove the first vector that was just merged
                ModelVectorsMulti_SeizSorted = ModelVectorsMulti_SeizSorted[1:, :]
                numAddedVecPerClass_Seiz_Sorted = numAddedVecPerClass_Seiz_Sorted[1:]
                numAddedVecPerClass_Seiz_PercSorted = numAddedVecPerClass_Seiz_PercSorted[1:]

        if (len(ModelVectorsMultiNorm_NonSeizSorted[:, 0]) != 1):
            # find the least populated one in nonSeiz and calculate its distance to closest seizure
            try:
                minDistNonSeizIndx = np.argmin(distMat_NN[0, 1:])
                minDistNonSeiz = distMat_NN[0, 1 + minDistNonSeizIndx]
                averageNonSeizDistFromAllSubclasses = (np.nanmean(distMat_NN[0, 1:]) * (numNonSeizSubclasses - 1) + np.nanmean( distMat_SN[ :,0]) * numSeizSubclasses) / (numNonSeizSubclasses - 1 + numSeizSubclasses)
            except:
                print('error')
                (distMat_NN, distArr_NN) = calculateAllPairwiseDistancesOfVectors_returnMatrix(
                    ModelVectorsMultiNorm_NonSeizSorted, ModelVectorsMultiNorm_NonSeizSorted, vecType)

            if (minDistNonSeiz < groupingThresh * averageNonSeizDistFromAllSubclasses):
                clusteringDone = 0
                mergingNonSeizCnt = mergingNonSeizCnt + 1
                newVectorSum = ModelVectorsMultiNorm_NonSeizSorted[minDistNonSeizIndx, :] * numAddedVecPerClass_NonSeiz_Sorted[ minDistNonSeizIndx] + ModelVectorsMultiNorm_NonSeizSorted[0, :] * numAddedVecPerClass_NonSeiz_Sorted[0]
                ModelVectorsMulti_NonSeizSorted[minDistNonSeizIndx, :]  = ModelVectorsMulti_NonSeizSorted[minDistNonSeizIndx, :] + ModelVectorsMulti_NonSeizSorted[0, :]
                numAddedVec = numAddedVecPerClass_NonSeiz_Sorted[minDistNonSeizIndx] + numAddedVecPerClass_NonSeiz_Sorted[0]
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    if (vecType=='bin'):
                        ModelVectorsMultiNorm_NonSeizSorted[minDistNonSeizIndx, :] = (newVectorSum > int(  math.floor(numAddedVec / 2))) * 1
                    elif (vecType == 'bipol'):
                        ModelVectorsMultiNorm_NonSeizSorted[minDistNonSeizIndx, :] = (newVectorSum > 0) * 1
                        ModelVectorsMultiNorm_NonSeizSorted[minDistNonSeizIndx, ModelVectorsMultiNorm_NonSeizSorted[minDistNonSeizIndx, :] == 0] = -1
                else:
                    ModelVectorsMultiNorm_NonSeizSorted[minDistNonSeizIndx, :] = (newVectorSum / numAddedVec)
                try:
                    ModelVectorsMultiNorm_NonSeizSorted = ModelVectorsMultiNorm_NonSeizSorted[1:, :]  # remove the first vector that was just merged
                    ModelVectorsMulti_NonSeizSorted = ModelVectorsMulti_NonSeizSorted[1:, :]
                    numAddedVecPerClass_NonSeiz_Sorted = numAddedVecPerClass_NonSeiz_Sorted[1:]
                    numAddedVecPerClass_NonSeiz_PercSorted = numAddedVecPerClass_NonSeiz_PercSorted[1:]
                except:
                    print('Error')


        #if certain amount of clusters merged evaluate performance and store
        if( (mergingNonSeizCnt+ mergingSeizCnt) >=numSubclassesPerStep):
            mergingNonSeizCnt=0; mergingSeizCnt=0;
            numClassesKept0 = [len(ModelVectorsMultiNorm_NonSeizSorted[:, 0]), len(ModelVectorsMultiNorm_SeizSorted[:, 0]),  len(ModelVectorsMultiNorm_NonSeizSorted[:, 0]) + len(ModelVectorsMultiNorm_SeizSorted[:, 0])]
            totalPercDataKept = (np.sum(numAddedVecPerClass_Seiz_Sorted) + np.sum( numAddedVecPerClass_NonSeiz_Sorted)) / initialAmountOfDataAdded
            perDataInClassesKept0 = [np.sum(numAddedVecPerClass_NonSeiz_PercSorted),  np.sum(numAddedVecPerClass_Seiz_PercSorted), totalPercDataKept]
            numClassesKept = np.vstack((numClassesKept, numClassesKept0))
            perDataInClassesKept = np.vstack((perDataInClassesKept, perDataInClassesKept0))

            # train and measure performance - TRAIN
            (simpleAccuracy_train0, distFromWrongMissclassifiedClass_train0, distFromCorrClass_train0, distFromWrongClass_train0, predLabels_train0, predLabels_nonBin0) = testModelVecOnData_Multiclass(
                data_train, labelsTrue_train, model, ModelVectorsMultiNorm_SeizSorted, ModelVectorsMultiNorm_NonSeizSorted, HDParams, vecType)
            # invert nonBin labels to mean always the same (the most persistent classes have smallest numbers)
            predLabels_nonBin1 = predLabels_nonBin0
            intpos = np.where(predLabels_nonBin0 > 0)
            predLabels_nonBin1[intpos] = 1 + len(ModelVectorsMultiNorm_Seiz[:, 0]) - predLabels_nonBin0[intpos]
            intneg = np.where(predLabels_nonBin0 < 0)
            predLabels_nonBin1[intneg] = -1 - (len(ModelVectorsMultiNorm_NonSeiz[:, 0]) + predLabels_nonBin0[intneg])
            # smoothen labels
            (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_train0, seizureStableLenToTestIndx,    seizureStablePercToTest,  distanceBetweenSeizuresIndx)
            # caluclate all performance measures
            (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
                predLabels_train0, labelsTrue_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
            allPerfMeasures_train_noSmooth0 = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean,   numFPperDay]
            (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
                yPred_SmoothOurStep1, labelsTrue_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
            allPerfMeasures_train_step10 = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
            (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
                yPred_SmoothOurStep2, labelsTrue_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
            allPerfMeasures_train_step20 = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
            #stacking
            simpleAccuracy_train = np.vstack((simpleAccuracy_train, simpleAccuracy_train0))
            distFromWrongMissclassifiedClass_train = np.vstack((distFromWrongMissclassifiedClass_train, distFromWrongMissclassifiedClass_train0))
            distFromCorrClass_train = np.vstack((distFromCorrClass_train, distFromCorrClass_train0))
            distFromWrongClass_train = np.vstack((distFromWrongClass_train, distFromWrongClass_train0))
            predLabels_train = np.hstack((predLabels_train, predLabels_train0))
            predLabels_train_nonBin = np.hstack((predLabels_train_nonBin, predLabels_nonBin1))
            allPerfMeasures_train_noSmooth = np.vstack((allPerfMeasures_train_noSmooth, allPerfMeasures_train_noSmooth0))
            allPerfMeasures_train_step1 = np.vstack((allPerfMeasures_train_step1, allPerfMeasures_train_step10))
            allPerfMeasures_train_step2 = np.vstack((allPerfMeasures_train_step2, allPerfMeasures_train_step20))

            # train and measure performance - TEST
            (simpleAccuracy_test0, distFromWrongMissclassifiedClass_test0, distFromCorrClass_test0, distFromWrongClass_test0, predLabels_test0, predLabels_nonBin0) = testModelVecOnData_Multiclass(
                data_test, labelsTrue_test, model, ModelVectorsMultiNorm_SeizSorted, ModelVectorsMultiNorm_NonSeizSorted, HDParams, vecType)
            # invert nonBin labels to mean always the same (the most persistent classes have smallest numbers)
            predLabels_nonBin1 = predLabels_nonBin0
            intpos = np.where(predLabels_nonBin0 > 0)
            predLabels_nonBin1[intpos] = 1 + len(ModelVectorsMultiNorm_Seiz[:, 0]) - predLabels_nonBin0[intpos]
            intneg = np.where(predLabels_nonBin0 < 0)
            predLabels_nonBin1[intneg] = -1 - (len(ModelVectorsMultiNorm_NonSeiz[:, 0]) + predLabels_nonBin0[intneg])
            # smoothen labels
            (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_test0, seizureStableLenToTestIndx,    seizureStablePercToTest,  distanceBetweenSeizuresIndx)
            # caluclate all performance measures
            (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
                predLabels_test0, labelsTrue_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
            allPerfMeasures_test_noSmooth0 = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean,   numFPperDay]
            (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
                yPred_SmoothOurStep1, labelsTrue_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
            allPerfMeasures_test_step10 = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
            (sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay) = performance_all9(
                yPred_SmoothOurStep2, labelsTrue_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
            allPerfMeasures_test_step20 = [sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay]
            #stacking
            simpleAccuracy_test = np.vstack((simpleAccuracy_test, simpleAccuracy_test0))
            distFromWrongMissclassifiedClass_test = np.vstack((distFromWrongMissclassifiedClass_test, distFromWrongMissclassifiedClass_test0))
            distFromCorrClass_test = np.vstack((distFromCorrClass_test, distFromCorrClass_test0))
            distFromWrongClass_test = np.vstack((distFromWrongClass_test, distFromWrongClass_test0))
            predLabels_test = np.hstack((predLabels_test, predLabels_test0))
            predLabels_test_nonBin = np.hstack((predLabels_test_nonBin, predLabels_nonBin1))
            allPerfMeasures_test_noSmooth = np.vstack((allPerfMeasures_test_noSmooth, allPerfMeasures_test_noSmooth0))
            allPerfMeasures_test_step1 = np.vstack((allPerfMeasures_test_step1, allPerfMeasures_test_step10))
            allPerfMeasures_test_step2 = np.vstack((allPerfMeasures_test_step2, allPerfMeasures_test_step20))

            # CHECK IF PERFORMANCE DROPED TOO MUCH
            if (optType == 'F1DEgmean'):
                # find optimal num to keep  - based on highest F1DEgmean performance to the right
                maxPerfIndx = np.argmax(allPerfMeasures_train_noSmooth[:, 7])
                limit = allPerfMeasures_train_noSmooth[maxPerfIndx, 7] - perfDropThr  # tolerable 1% drop
                try:
                    perfToCompare = allPerfMeasures_train_noSmooth[-1, 7]
                except:
                    print('error')
            elif (optType == 'simpleAcc'):
                # find optimal num to keep  - based on simple accuracy performance to the right
                maxPerfIndx = np.argmax(simpleAccuracy_train[:, 0])
                limit = simpleAccuracy_train[maxPerfIndx, 0] - perfDropThr # tolerable 1% drop
                perfToCompare = simpleAccuracy_train[-1, 0]
            if (perfToCompare >= limit):
                optimalPerf_train = np.hstack(
                    (numClassesKept[-1, :], perDataInClassesKept[-1, :], simpleAccuracy_train[-1, :],
                     allPerfMeasures_train_noSmooth[-1, :], allPerfMeasures_train_step1[-1, :],
                     allPerfMeasures_train_step2[-1, :]))
                optimalPerf_test = np.hstack(
                    (numClassesKept[-1, :], perDataInClassesKept[-1, :], simpleAccuracy_test[-1, :],
                     allPerfMeasures_test_noSmooth[-1, :], allPerfMeasures_test_step1[-1, :],
                     allPerfMeasures_test_step2[-1, :]))
                ModelVectorsMultiNorm_Seiz_Optimal = ModelVectorsMultiNorm_SeizSorted
                ModelVectorsMultiNorm_NonSeiz_Optimal = ModelVectorsMultiNorm_NonSeizSorted
                ModelVectorsMulti_Seiz_Optimal = ModelVectorsMulti_SeizSorted
                ModelVectorsMulti_NonSeiz_Optimal = ModelVectorsMulti_NonSeizSorted
                numAddedVecPerClass_Seiz_Optimal = numAddedVecPerClass_Seiz_Sorted
                numAddedVecPerClass_NonSeiz_Optimal = numAddedVecPerClass_NonSeiz_Sorted
            else:
                break

    #stack all data an save just for the case
    pom=np.hstack((distFromWrongMissclassifiedClass_train, distFromCorrClass_train, distFromWrongClass_train,distFromWrongMissclassifiedClass_test, distFromCorrClass_test, distFromWrongClass_test))
    dataToSave = np.hstack((numClassesKept, perDataInClassesKept, pom)) #.transpose()
    outputName = folderOut2 + '/' + fileName +'_VariousMeasuresPerItteration.csv'
    np.savetxt(outputName, dataToSave, delimiter=",")
    dataToSave = np.hstack((simpleAccuracy_train, allPerfMeasures_train_noSmooth, allPerfMeasures_train_step1, allPerfMeasures_train_step2))
    outputName = folderOut2 + '/' + fileName +'_PerformanceMeasuresPerItteration_Train.csv'
    np.savetxt(outputName, dataToSave, delimiter=",")
    dataToSave = np.hstack((simpleAccuracy_test, allPerfMeasures_test_noSmooth, allPerfMeasures_test_step1, allPerfMeasures_test_step2))
    outputName = folderOut2 + '/' + fileName +'_PerformanceMeasuresPerItteration_Test.csv'
    np.savetxt(outputName, dataToSave, delimiter=",")
    outputName = folderOut2 + '/' + fileName +'_LabelsPerItteration_Train.csv'
    np.savetxt(outputName, predLabels_train, delimiter=",")
    outputName = folderOut2 + '/' + fileName +'_LabelsPerItteration_Test.csv'
    np.savetxt(outputName, predLabels_test, delimiter=",")
    outputName = folderOut2 + '/' + fileName +'_LabelsNonBinPerItteration_Train.csv'
    np.savetxt(outputName, predLabels_train_nonBin, delimiter=",")
    outputName = folderOut2 + '/' + fileName +'_LabelsNonBinPerItteration_Test.csv'
    np.savetxt(outputName, predLabels_test_nonBin, delimiter=",")

    # #PLOTTING
    # fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    # gs = GridSpec(2, 3, figure=fig1)
    # fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    # fig1.suptitle('Vector distances vs number of subclasses')
    #
    # #numItter=len(simpleAccuracy_test)
    # #xValues=np.arange(0,numItter,1) /numItter
    # xValues=1-numClassesKept[:, 2]/numClassesKept[0, 2]
    # ax1 = fig1.add_subplot(gs[0, 0])
    # ax1.plot(xValues,  numClassesKept[:, 0], 'b')  # nonSeiz
    # ax1.plot(xValues, numClassesKept[:, 1], 'r')  # Seiz
    # ax1.plot(xValues, numClassesKept[:, 2], 'k')  # Total
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('Num classes kept')
    # ax1.legend(['nonSeiz', 'Seiz', 'Both'])
    # ax1.grid()
    # ax1 = fig1.add_subplot(gs[0, 1])
    # ax1.plot(xValues,  perDataInClassesKept[:, 0], 'b')  # nonSeiz
    # ax1.plot(xValues, perDataInClassesKept[:, 1], 'r')  # Seiz
    # ax1.plot(xValues, perDataInClassesKept[:, 2], 'k')  # Total
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('Perc data in kept classes')
    # ax1.legend(['nonSeiz', 'Seiz', 'Both'])
    # ax1.grid()
    # ax1 = fig1.add_subplot(gs[0, 2])
    # ax1.plot(xValues,  simpleAccuracy_train, 'k')
    # ax1.plot(xValues, simpleAccuracy_test, 'b--')
    # ax1.legend(['Train', 'Test'])
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('Simple accuracy')
    # ax1.grid()
    # #performances in more details before and after smoothing
    # ax1 = fig1.add_subplot(gs[1, 0])
    # ax1.plot(xValues,  allPerfMeasures_train_noSmooth[:, 2], 'k')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_train_step1[:, 2], 'b')  # Step1
    # ax1.plot(xValues, allPerfMeasures_train_step2[:, 2], 'm')  # Step2
    # ax1.plot(xValues,  allPerfMeasures_test_noSmooth[:, 2], 'k--')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_test_step1[:, 2], 'b--')  # Step1
    # ax1.plot(xValues, allPerfMeasures_test_step2[:, 2], 'm--')  # Step2
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('F1 Episodes')
    # ax1.legend(['noSmooth', 'Step1', 'Step2'])
    # ax1.grid()
    # ax1 = fig1.add_subplot(gs[1, 1])
    # ax1.plot(xValues,  allPerfMeasures_train_noSmooth[:, 5], 'k')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_train_step1[:, 5], 'b')  # Step1
    # ax1.plot(xValues, allPerfMeasures_train_step2[:, 5], 'm')  # Step2
    # ax1.plot(xValues,  allPerfMeasures_test_noSmooth[:, 5], 'k--')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_test_step1[:, 5], 'b--')  # Step1
    # ax1.plot(xValues, allPerfMeasures_test_step2[:, 5], 'm--')  # Step2
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('F1 Duration')
    # ax1.legend(['noSmooth', 'Step1', 'Step2'])
    # ax1.grid()
    # ax1 = fig1.add_subplot(gs[1, 2])
    # ax1.plot(xValues,  allPerfMeasures_train_noSmooth[:, 7], 'k')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_train_step1[:, 7], 'b')  # Step1
    # ax1.plot(xValues, allPerfMeasures_train_step2[:, 7], 'm')  # Step2
    # ax1.plot(xValues,  allPerfMeasures_test_noSmooth[:, 5], 'k--')  # noSmooth
    # ax1.plot(xValues, allPerfMeasures_test_step1[:, 5], 'b--')  # Step1
    # ax1.plot(xValues, allPerfMeasures_test_step2[:, 5], 'm--')  # Step2
    # ax1.set_xlabel('Perc classes removed')
    # ax1.set_title('F1 Gmean')
    # ax1.legend(['noSmooth', 'Step1', 'Step2'])
    # ax1.grid()
    # fig1.show()
    # fig1.savefig(folderOut2 + '/'+ fileName+ '_itterativeRemovingAllRes.png')
    # plt.close(fig1)

    print('Drop train:', (simpleAccuracy_train[-1,:]-simpleAccuracy_train[0,:])*100, ' Drop test: ', (simpleAccuracy_test[-1,:]-simpleAccuracy_test[0,:])*100)
    return (optimalPerf_train, optimalPerf_test,ModelVectorsMulti_Seiz_Optimal, ModelVectorsMulti_NonSeiz_Optimal, ModelVectorsMultiNorm_Seiz_Optimal, ModelVectorsMultiNorm_NonSeiz_Optimal, numAddedVecPerClass_Seiz_Optimal, numAddedVecPerClass_NonSeiz_Optimal)


def testModelsAndReturnAllPerformances_2class(data_train_Discr, label_train, data_test_Discr, label_test,model, ModelVectorsNorm, HDParams, GeneralParams, SegSymbParams , vecType='bin'):
    ''' test performance both on train and test dataset '''
    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
    numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
    toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz / SegSymbParams.slidWindStepSec)

    # testing on training and test dataset
    (acc_train, accPerClass_train, distWhenWrongPredict_train, distFromCorr_AllClass_train, distFromCorr_PerClass_train,
     distFromWrong_AllClass_train, distFromWrong_PerClass_train, predLabels_train) = testModelVecOnData(data_train_Discr, label_train, model, ModelVectorsNorm, HDParams, vecType)
    (acc_test, accPerClass_test, distWhenWrongPredict_test, distFromCorr_AllClass_test, distFromCorr_PerClass_test,
     distFromWrong_AllClass_test, distFromWrong_PerClass_test, predLabels_test) = testModelVecOnData(data_test_Discr,label_test, model, ModelVectorsNorm, HDParams, vecType)
    #print('acc_train: ', acc_train, 'acc_test: ', acc_test)

    #store various performance measures
    AllRes_train=np.zeros((33))
    AllRes_test = np.zeros((33))
    AllRes_train[0:6] = np.hstack( (1, 1, acc_train, distFromCorr_AllClass_train, distFromWrong_AllClass_train, distWhenWrongPredict_train))
    AllRes_test[0:6] = np.hstack( (1, 1, acc_test, distFromCorr_AllClass_test, distFromWrong_AllClass_test, distWhenWrongPredict_test))
    # calculate all perf measures
    AllRes_train[6:15] = performance_all9(predLabels_train, label_train, toleranceFP_bef, toleranceFP_aft,  numLabelsPerHour)
    AllRes_test[6:15] = performance_all9(predLabels_test, label_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    # smoothing labels and calculating perforance
    (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_train, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx)
    AllRes_train[15:24] = performance_all9(yPred_SmoothOurStep1, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    AllRes_train[24:33] = performance_all9(yPred_SmoothOurStep2, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_test, seizureStableLenToTestIndx,seizureStablePercToTest, distanceBetweenSeizuresIndx)
    AllRes_test[15:24] = performance_all9(yPred_SmoothOurStep1, label_test, toleranceFP_bef, toleranceFP_aft,   numLabelsPerHour)
    AllRes_test[24:33] = performance_all9(yPred_SmoothOurStep2, label_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)

    return(AllRes_train, AllRes_test, predLabels_train, predLabels_test)

def testModelsAndReturnAllPerformances_MoreClass(data_train_Discr, label_train, data_test_Discr, label_test,model, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, HDParams, GeneralParams,SegSymbParams, vecType='bin' ):
    ''' test performance both on train and test dataset '''

    #various parameters that are needed
    seizureStableLenToTestIndx = int(GeneralParams.seizureStableLenToTest / SegSymbParams.slidWindStepSec)
    seizureStablePercToTest = GeneralParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(GeneralParams.distanceBetween2Seizures / SegSymbParams.slidWindStepSec)
    numLabelsPerHour = 60 * 60 / SegSymbParams.slidWindStepSec
    toleranceFP_bef = int(GeneralParams.toleranceFP_befSeiz / SegSymbParams.slidWindStepSec)
    toleranceFP_aft = int(GeneralParams.toleranceFP_aftSeiz / SegSymbParams.slidWindStepSec)

    # testing on training and test dataset
    (acc_train, distWhenWrongPredict_train, distFromCorr_AllClass_train,distFromWrong_AllClass_train, predLabels_train,
     predLabels_nonBin_train) = testModelVecOnData_Multiclass(data_train_Discr, label_train, model, ModelVectorsNorm_Seiz, ModelVectorsNorm_NonSeiz, HDParams, vecType)
    (acc_test, distWhenWrongPredict_test, distFromCorr_AllClass_test, distFromWrong_AllClass_test, predLabels_test,
     predLabels_nonBin_test) = testModelVecOnData_Multiclass(data_test_Discr, label_test, model, ModelVectorsNorm_Seiz,  ModelVectorsNorm_NonSeiz, HDParams, vecType)

    #calculate various performance measures
    AllRes_train=np.zeros((33))
    AllRes_test = np.zeros((33))
    numSubClass_Seiz=len(ModelVectorsNorm_Seiz[:,0])
    numSubClass_NonSeiz=len(ModelVectorsNorm_NonSeiz[:,0])
    AllRes_train[ 0:6] = np.hstack((numSubClass_Seiz, numSubClass_NonSeiz, acc_train, distFromCorr_AllClass_train, distFromWrong_AllClass_train, distWhenWrongPredict_train))
    AllRes_test[0:6] = np.hstack((numSubClass_Seiz, numSubClass_NonSeiz, acc_test, distFromCorr_AllClass_test, distFromWrong_AllClass_test,  distWhenWrongPredict_test))
    AllRes_train[ 6:15] = performance_all9(predLabels_train, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    AllRes_test[ 6:15] = performance_all9(predLabels_test, label_test, toleranceFP_bef,  toleranceFP_aft, numLabelsPerHour)
    (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_train,  seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx)
    AllRes_train[ 15:24] = performance_all9(yPred_SmoothOurStep1, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    AllRes_train[24:33] = performance_all9(yPred_SmoothOurStep2, label_train, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predLabels_test, seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx)
    AllRes_test[ 15:24] = performance_all9(yPred_SmoothOurStep1, label_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
    AllRes_test[ 24:33] = performance_all9(yPred_SmoothOurStep2, label_test, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)

    predLabelsTrain=np.vstack((predLabels_train, predLabels_nonBin_train))
    predLabelsTest = np.vstack((predLabels_test, predLabels_nonBin_test))
    return(AllRes_train, AllRes_test, predLabelsTrain, predLabelsTest)