''' library with different functions on HD vectors, uses torch library '''
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

import torch
import time, sys
from VariousFunctionsLib import *
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

class HD_classifier_GeneralAndNoCh:
    ''' Approach that uses several features and then maps them all to HD vectors
    doesnt know and doesnt care that some features are from different channels'''

    def __init__(self, HDParams, totNumFeat, vecType='bin', cuda = True):
        ''' Initialize an HD vectors using the torch library
        SigInfoParams, SegSymbParams and HDParams contain all parameters needed
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''

        self.NumValues = HDParams.numSegmentationLevels
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
#
class HD_classifier_GeneralWithChCombinations:
    ''' Approach that uses several features and then maps them all to HD vectors
    but know that some features are from different channels so there are different ways to approach it'''

    def __init__(self,HDParams, numCh,vecType='bin', cuda = True):
        ''' Initialize an HD vectors using the torch library
        SigInfoParams, SegSymbParams and HDParams contain all parameters needed
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''

        self.NumValues = HDParams.numSegmentationLevels
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

        elif ('ChAppend' in HDParams.bindingFeatures):
            Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
            output_vector=torch.zeros(self.HD_dim*self.NumCh,1).cuda(device=self.device)
            for f in range(self.NumCh):
                #bind ch and  values of current feature
                bindingVector = xor(self.proj_mat_features, self.proj_mat_FeatVals[:, data[f,:]])
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    Chvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumFeat / 2))).short()
                else:
                    Chvect_matrix[:,f]  = (bindingVectorSum /  self.NumFeat)
                #apppending
                output_vector[f*self.HD_dim:(f+1)*self.HD_dim,0]=Chvect_matrix[:,f]
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

        elif ('ChAppend' in HDParams.bindingFeatures):
            Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
            output_vector=torch.zeros(self.HD_dim*self.NumCh,1).cuda(device=self.device)
            for f in range(self.NumCh):
                #bind ch and  values of current feature
                bindingVector = xor_bipolar(self.proj_mat_features, self.proj_mat_FeatVals[:, data[f,:]]) #!!! diffferent from binary
                bindingVectorSum=torch.sum(bindingVector, dim=1)
                #binarizing
                if (HDParams.roundingTypeForHDVectors != 'noRounding'):
                    #Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh / 2))).short()
                    Chvect_matrix[:, f] = (bindingVectorSum > 0).short()  # !!! diffferent from binary
                else:
                    Chvect_matrix[:,f]  = (bindingVectorSum /  self.NumFeat)
                #apppending
                output_vector[f*self.HD_dim:(f+1)*self.HD_dim,0]=Chvect_matrix[:,f]

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

# def ham_dist_arr( vec_a, vec_b, D, vecType='bin'):
#     ''' calculate relative hamming distance fur for np array'''
#     if (vecType=='bin'):
#         vec_c= np.abs(vec_a-vec_b)
#         rel_dist = np.sum(vec_c) / float(D)
#     elif (vecType=='bipol'):
#         vec_c= vec_a+vec_b
#         rel_dist = np.sum(vec_c==0) / float(D)
#     return rel_dist

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
    elif('ChAppend' in HDParams.bindingFeatures):
        ModelVectors = np.zeros((numLabels, HDParams.D * HDParams.numCh))  # .cuda(device=self.device)
        ModelVectorsNorm = np.zeros((numLabels, HDParams.D * HDParams.numCh))
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


def calculateVecSeparavilityPerFeature(ModelVectors0, D):
    (numClasses, TotalD)=ModelVectors0.shape
    if (numClasses>TotalD):
        ModelVectors=ModelVectors0.transpose()
    else:
        ModelVectors=ModelVectors0
    (numClasses, TotalD) = ModelVectors.shape
    numFeat=int(TotalD /D)
    distances=np.zeros(numFeat)
    for f in range(numFeat):
        distances[f] = ham_dist_arr(ModelVectors[0, f*D:(f+1)*D], ModelVectors[1, f*D:(f+1)*D], D)
    return (distances)

def testModelVecOnData_simpler(data, trueLabels, model, ModelVectors, HDParams, vecType='bin'):
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
    for s in range(numWin):
        if ( vecType=='bin'):
            (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        elif (vecType=='bipol'):
            (temp, timeHDVec) = model.learn_HD_proj_bipolar(data[s, :], HDParams)
        tempVec=temp.cpu().numpy()
        (predLabels[s], distances[s,:]) = givePrediction(tempVec, ModelVectors, HDParams.similarityType, vecType)

    #calculate accuracy
    diffLab=predLabels-trueLabels
    indx=np.where(diffLab==0)
    acc= len(indx[0])/len(trueLabels)

    # calculate performance  per class
    accPerClass=np.zeros(numLabels)
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

    #probability
    probabilityLab= calculateProbabilityFromClassDistances(distances, predLabels)
    # dist_fromVotedOne=np.zeros((len(distances[:,0])))
    # dist_oppositeVotedOne = np.zeros((len(distances[:, 0])))
    # indx=np.where(predLabels==0)[0]
    # dist_fromVotedOne[indx]=distances[indx,0]
    # dist_oppositeVotedOne[indx]=distances[indx,1]
    # indx=np.where(predLabels==1)[0]
    # dist_fromVotedOne[indx]=distances[indx,1]
    # dist_oppositeVotedOne[indx]=distances[indx,0]
    # probabilityLab =dist_oppositeVotedOne/ (dist_oppositeVotedOne + dist_fromVotedOne)

    allInfo=np.hstack((distances, trueLabels.reshape((-1,1)), predLabels.reshape((-1,1)), probabilityLab.reshape((-1,1))))
    return(acc, accPerClass, predLabels, probabilityLab, distances, allInfo)


def onlineHD_ModelVecOnData(data, trueLabels, model, HDParams, type, vecType='bin'):
    # we need to use vectors with -1 and 1 instead of 0 and 1!!!

    (numWin_train, numFeat) = data.shape
    # number of clases
    (unique_labels, counts) = np.unique(trueLabels, return_counts=True)
    numLabels = len(unique_labels)

    # initializing model vectors and then training
    if ('FeatAppend' in HDParams.bindingFeatures):
        ModelVectors = np.zeros((numLabels, HDParams.D*HDParams.numFeat))  # .cuda(device=self.device)
        ModelVectorsNorm = np.zeros((numLabels, HDParams.D*HDParams.numFeat))
    elif ('ChAppend' in HDParams.bindingFeatures):
        ModelVectors = np.zeros((numLabels, HDParams.D*HDParams.numCh))  # .cuda(device=self.device)
        ModelVectorsNorm = np.zeros((numLabels, HDParams.D*HDParams.numCh))
    else:
        ModelVectors = np.zeros((numLabels, HDParams.D))   # .cuda(device=self.device)
        ModelVectorsNorm= np.zeros((numLabels, HDParams.D))
    # ModelVectorsNorm = np.zeros((2, HDParams.D))
    # ModelVectors = np.zeros((2, HDParams.D))
    numAddedVecPerClass = np.zeros((numLabels))  # np.array([0, 0])


    (numLabels, D)=ModelVectors.shape
    # ModelVectorsNorm=np.zeros((numLabels, D))
    (numWin, numFeat) = data.shape
    predLabels = np.zeros(numWin)
    weights=np.zeros(numWin)
    allDistances=np.zeros((numWin,2))
    for s in range(numWin):
        if (vecType=='bin'):
            (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
            tempVec = temp.cpu().numpy().reshape((1,-1))
            tempVecBipol = np.copy(tempVec)
            indx0 = np.where(tempVec == 0)
            tempVecBipol[indx0] = -1
        elif (vecType=='bipol'):
            (temp, timeHDVec) = model.learn_HD_proj_bipolar(data[s, :], HDParams)
            tempVec = temp.cpu().numpy().reshape((1,-1))
            tempVecBipol = np.copy(tempVec)
        (predLabels[s], distances) = givePrediction(tempVec, ModelVectorsNorm, HDParams.similarityType)
        allDistances[s,0:2]=distances

        # add to the correct class
        lab = int(trueLabels[s])  # -1
        # weight = distances[lab] #if very similar weight it smaller
        weight = np.power(distances[lab],2)  # if very similar weight it smaller
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
    allInfo=np.hstack((allDistances, trueLabels.reshape((-1,1)), predLabels.reshape((-1,1)), weights.reshape((-1,1))))
    return (ModelVectors, ModelVectorsNorm, numAddedVecPerClass, weights, allInfo)



def testModelVecOnData_MeasureDistances(data, trueLabels, model, ModelVectors, HDParams, chFeatType):

    if (chFeatType=='PerFeat'):
        numElems=HDParams.numFeat
    else:
        numElems=HDParams.numCh
    (unique_labels, counts) = np.unique(trueLabels, return_counts=True)
    (numLabels, totalD)=ModelVectors.shape
    (numWin, x) = data.shape
    predLabels = np.zeros((numWin, numElems+1))
    distances=np.zeros((numLabels, numElems+1))
    distFromS=np.zeros((numWin, numElems+1))
    distFromNS = np.zeros((numWin, numElems+1))

    for s in range(numWin):
        (temp, timeHDVec) = model.learn_HD_proj(data[s, :], HDParams)
        tempVec=temp.cpu().numpy()
        #total predictions using whole vectors
        (predLabels[s, -1], distances[:,-1]) = givePrediction(tempVec, ModelVectors, HDParams.similarityType)
        #predictions and distances per feature
        (predLabels[s, 0:numElems], distances[:, 0:numElems]) = givePrediction_indivAppendFeat(tempVec, ModelVectors, numElems, HDParams.D, HDParams.similarityType)

        distFromS[s,:]=distances[1,:]
        distFromNS[s, :] = distances[0, :]

    return (predLabels, distFromS, distFromNS)

def testAndSavePredictionsForHD_PerAppended(folderIn, fileName, model, ModelVectorsNorm, HDParams, data_train_ToTrain, label_train,data_test_ToTrain, label_test, HDtype, FeatChType):

    (predLabels_train, distFromS_train, distFromNS_train) = testModelVecOnData_MeasureDistances(data_train_ToTrain, label_train, model,ModelVectorsNorm, HDParams, FeatChType)
    (predLabels_test, distFromS_test, distFromNS_test) = testModelVecOnData_MeasureDistances(data_test_ToTrain, label_test, model, ModelVectorsNorm, HDParams, FeatChType)
    # predictions
    dataToSave = np.hstack((predLabels_train, np.reshape(label_train, (len(label_train), -1))))
    outputName = folderIn + '/' + fileName + '_'+HDtype+'_'+FeatChType+'_TrainPredictions.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')
    dataToSave = np.hstack((predLabels_test, np.reshape(label_test, (len(label_test), -1))))
    outputName = folderIn + '/' + fileName + '_'+HDtype+'_'+FeatChType+'_TestPredictions.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')
    # distances from seiz and non seiz class
    outputName = folderIn + '/' + fileName + '_'+HDtype+'_'+FeatChType+'_DistancesFromSeiz_Train.csv'
    saveDataToFile(distFromS_train, outputName, 'gzip')
    outputName = folderIn + '/' + fileName +  '_'+HDtype+'_'+FeatChType+'_DistancesFromNonSeiz_Train.csv'
    saveDataToFile(distFromNS_train, outputName, 'gzip')
    outputName = folderIn + '/' + fileName +  '_'+HDtype+'_'+FeatChType+'_DistancesFromSeiz_Test.csv'
    saveDataToFile(distFromS_test, outputName, 'gzip')
    outputName = folderIn + '/' + fileName + '_'+HDtype+'_'+FeatChType+'_DistancesFromNonSeiz_Test.csv'
    saveDataToFile(distFromNS_test, outputName, 'gzip')



def testAndSavePredictionsForHD(folder, fileName, model, ModelVectorsNorm, HDParams, PostprocessingParams, FeaturesParams, data_train_ToTrain, label_train,data_test_ToTrain, label_test, HDtype):
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFP_befSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFP_aftSeiz / FeaturesParams.winStep)


    # all final labels
    # (acc_train, accPerClass_train, distWhenWrongPredict_train, distFromCorr_AllClass_train, distFromCorr_PerClass_train,
    #  distFromWrong_AllClass_train, distFromWrong_PerClass_train, predLabels_train,
    #  probabLabels_train) = testModelVecOnData(data_train_ToTrain, label_train, model, ModelVectorsNorm, HDParams,HDParams.HDvecType)
    # (acc_test, accPerClass_test, distWhenWrongPredict_test, distFromCorr_AllClass_test, distFromCorr_PerClass_test,
    #  distFromWrong_AllClass_test, distFromWrong_PerClass_test, predLabels_test,
    #  probabLabels_test) = testModelVecOnData(data_test_ToTrain, label_test, model, ModelVectorsNorm, HDParams, HDParams.HDvecType)
    (acc_train, accPerClass_train,predLabels_train, probabLabels_train, distances_train, allInfo_Train) = testModelVecOnData_simpler(data_train_ToTrain, label_train, model, ModelVectorsNorm, HDParams,HDParams.HDvecType)
    (acc_test, accPerClass_test, predLabels_test,probabLabels_test, distances_test, allInfo_Test) = testModelVecOnData_simpler(data_test_ToTrain, label_test, model, ModelVectorsNorm, HDParams, HDParams.HDvecType)
    print( HDtype +' acc_train: ', acc_train, 'acc_test: ', acc_test)

    #save distances
    outputName = folder + '/' + fileName + '_'+HDtype+'_TrainDistancesSNS.csv'
    saveDataToFile(distances_train, outputName, 'gzip')
    outputName = folder + '/' + fileName + '_'+HDtype+'_TestDistancesSNS.csv'
    saveDataToFile(distances_test, outputName, 'gzip')

    # perform smoothing
    (performanceTrain0, yPredTrain_MovAvrgStep1, yPredTrain_MovAvrgStep2,
     yPredTrain_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels_train, label_train,probabLabels_train,
                                                                         toleranceFP_bef, toleranceFP_aft, numLabelsPerHour,
                                                                         seizureStableLenToTestIndx,seizureStablePercToTest,distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
    (performanceTest0, yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2,
     yPredTest_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels_test, label_test, probabLabels_test,
                                                                        toleranceFP_bef, toleranceFP_aft, numLabelsPerHour,
                                                                        seizureStableLenToTestIndx, seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)

    #save predictions
    dataToSave = np.vstack((label_train, probabLabels_train, predLabels_train, yPredTrain_MovAvrgStep1, yPredTrain_MovAvrgStep2, yPredTrain_SmoothBayes)).transpose()
    outputName = folder + '/' + fileName + '_'+HDtype+'_TrainPredictions.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')
    dataToSave = np.vstack((label_test, probabLabels_test, predLabels_test, yPredTest_MovAvrgStep1, yPredTest_MovAvrgStep2, yPredTest_SmoothBayes)).transpose()  # added from which file is specific part of test set
    outputName = folder + '/' + fileName + '_'+HDtype+'_TestPredictions.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')

    return(predLabels_train, predLabels_test, allInfo_Train, allInfo_Test)

def calcualtePerformanceMeasures_PerFeature(predictions, probabLabels, trueLabels,  toleranceFP_bef, toleranceFP_aft, numLabelsPerHour, seizureStableLenToTestIndx,  seizureStablePercToTest,distanceBetweenSeizuresIndx, bayesProbThresh):

    (numSeg, numCol)=predictions.shape
    performancesAll=np.zeros((numCol, 36)) #9 performance measures
    # performancesAll_step1 = np.zeros((numCol, 9))  # 9 performance measures
    # performancesAll_step2 = np.zeros((numCol, 9))  # 9 performance measures
    for f in range(numCol):
        ( performancesAll[f,:],_,_,_) = calculatePerformanceAfterVariousSmoothing(predictions[:,f], trueLabels, probabLabels[:,f],
                                                                             toleranceFP_bef, toleranceFP_aft, numLabelsPerHour,
                                                                             seizureStableLenToTestIndx, seizureStablePercToTest,distanceBetweenSeizuresIndx,bayesProbThresh)
        #
        # performancesAll[f,:]=performance_all9(predictions[:,f], trueLabels, toleranceFP_bef, toleranceFP_aft, numLabelsPerHour)
        # # smoothen labels
        # (yPred_SmoothOurStep2, yPred_SmoothOurStep1) = smoothenLabels(predictions[:,f], seizureStableLenToTestIndx,seizureStablePercToTest, distanceBetweenSeizuresIndx)
        # performancesAll_step1[f, :] = performance_all9(yPred_SmoothOurStep1, trueLabels, toleranceFP_bef, toleranceFP_aft,numLabelsPerHour)
        # performancesAll_step2[f, :] = performance_all9(yPred_SmoothOurStep2, trueLabels, toleranceFP_bef,toleranceFP_aft, numLabelsPerHour)
    return(performancesAll) #, performancesAll_step1, performancesAll_step2)


def func_analysePerAppending(folderIn, folderOut, GeneralParams, PostprocessingParams, FeaturesParams, HDParams, DatasetPreprocessParams, HDtype, FeatChType):
    if (FeatChType=='PerFeat'):
        numElem=HDParams.numFeat
        elemNames=FeaturesParams.featNames
    else:
        numElem=HDParams.numCh
        elemNames = DatasetPreprocessParams.channelNamesToKeep

    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFP_befSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFP_aftSeiz / FeaturesParams.winStep)

    distSeparability_AllSubj = np.zeros((len(GeneralParams.patients), numElem + 1))
    confidencesPerFeat_Train_AllSubj= np.zeros((len(GeneralParams.patients), numElem ))
    confidencesPerFeat_Test_AllSubj = np.zeros((len(GeneralParams.patients), numElem ))
    performancesAll_Train_AllSubj = np.zeros((numElem + 1, 36, len(GeneralParams.patients)))
    performancesAll_Test_AllSubj = np.zeros((numElem + 1, 36, len(GeneralParams.patients)))

    for patIndx, pat in enumerate(GeneralParams.patients):
        print('Subj '+ pat)
        filesAll = np.sort(glob.glob(folderIn + '/*Subj' + pat + '*_'+HDtype+'_ModelVecsNorm.csv.gz'))
        numFiles=len(filesAll)
        distSeparability_ThisSubj = np.zeros((numFiles, numElem + 1))
        confidencesPerFeat_Train_ThisSubj = np.zeros((numFiles, numElem ))
        confidencesPerFeat_Test_ThisSubj = np.zeros((numFiles, numElem ))
        performancesAll_Train_ThisSubj = np.zeros((numElem + 1, 36, numFiles))
        performancesAll_Test_ThisSubj = np.zeros((numElem + 1, 36, numFiles))

        for fIndx, fName in enumerate(filesAll):
            pom, fileName1 = os.path.split(fName)
            fNameBase=fileName1[0:-21]
            #READ DATA
            #model vectors
            ModelVectorsNorm = readDataFromFile(fName)
            #predictions
            data0 = readDataFromFile(folderOut+ fNameBase+'_'+FeatChType+'_TrainPredictions.csv')
            predLabels_train=data0[:,0:-1]
            label_train=data0[:,-1]
            data0 = readDataFromFile(folderOut+fNameBase+'_'+FeatChType+'_TestPredictions.csv')
            predLabels_test=data0[:,0:-1]
            label_test=data0[:,-1]
            #distances
            distFromS_train = readDataFromFile(folderOut+ fNameBase+'_'+FeatChType+'_DistancesFromSeiz_Train.csv.gz')
            distFromNS_train = readDataFromFile(folderOut+ fNameBase+'_'+FeatChType+'_DistancesFromNonSeiz_Train.csv.gz')
            distFromS_test = readDataFromFile(folderOut + fNameBase + '_'+FeatChType+'_DistancesFromSeiz_Test.csv.gz')
            distFromNS_test = readDataFromFile(folderOut + fNameBase + '_'+FeatChType+'_DistancesFromNonSeiz_Test.csv.gz')

            # COMPARE FEATURES
            # based on model vector separability
            distSeparability = np.zeros(numElem + 1)
            distSeparability[0:numElem] = calculateVecSeparavilityPerFeature(ModelVectorsNorm, HDParams.D)
            distSeparability[numElem] = ham_dist_arr(ModelVectorsNorm[0, :], ModelVectorsNorm[1, :], HDParams.D * numElem)
            outputName = folderOut  + fNameBase + '_VectSeparability.csv'
            saveDataToFile(distSeparability, outputName, 'gzip')
            #feature confidences
            #negative value  means more confident in nonSeizure and positive is confidence in seizure
            confidencesPerFeat_Train=measureConfidences(distFromS_train, distFromNS_train)
            confidencesPerFeat_Test=measureConfidences(distFromS_test, distFromNS_test)
            outputName = folderOut  + fNameBase + '_'+FeatChType+'_TrainConfidence.csv'
            saveDataToFile(confidencesPerFeat_Train, outputName, 'gzip')
            outputName =folderOut  + fNameBase + '_'+FeatChType+'_TestConfidence.csv'
            saveDataToFile(confidencesPerFeat_Test, outputName, 'gzip')
            #calculate mean over all samples
            confidencesPerFeat_Train=np.mean(np.abs(confidencesPerFeat_Train),0)
            confidencesPerFeat_Test=np.mean(np.abs(confidencesPerFeat_Test),0)

            #calculate probabilities per feature
            probabilityLabels_train=calculateProbabilitiesPerFeature(distFromS_train, distFromNS_train, predLabels_train)
            probabilityLabels_test = calculateProbabilitiesPerFeature(distFromS_test, distFromNS_test, predLabels_test)

            # MEASURE PERFORMANCES
            (performancesAll_Train) = calcualtePerformanceMeasures_PerFeature(predLabels_train, probabilityLabels_train, label_train, toleranceFP_bef, toleranceFP_aft,numLabelsPerHour,
                                                                                    seizureStableLenToTestIndx, seizureStablePercToTest,distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
            (performancesAll_Test) = calcualtePerformanceMeasures_PerFeature(predLabels_test, probabilityLabels_test, label_test,toleranceFP_bef, toleranceFP_aft, numLabelsPerHour,
                                                                                   seizureStableLenToTestIndx, seizureStablePercToTest,distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)
            # print('acc_train: ', performancesAll_Train[-1, 7], 'acc_test: ', performancesAll_Test[-1, 7])
            # saving
            outputName = folderOut + '/' + fNameBase + '_'+FeatChType+'_TrainPerformance.csv'
            saveDataToFile(performancesAll_Train, outputName, 'gzip')
            outputName = folderOut + '/' + fNameBase + '_'+FeatChType+'_TestPerformance.csv'
            saveDataToFile(performancesAll_Test, outputName, 'gzip')

            # SAVING TO CALCULATE AVRG FOR THIS SUBJ
            distSeparability_ThisSubj[fIndx, :] = distSeparability
            confidencesPerFeat_Train_ThisSubj[fIndx, :] = confidencesPerFeat_Train
            confidencesPerFeat_Test_ThisSubj[fIndx, :] = confidencesPerFeat_Test
            performancesAll_Train_ThisSubj[:, :, fIndx] = performancesAll_Train
            performancesAll_Test_ThisSubj[:, :, fIndx] = performancesAll_Test

        # save for this subj
        dataToSave = np.zeros((2, numElem + 1))
        dataToSave[0, :] = np.nanmean(distSeparability_ThisSubj, 0)
        dataToSave[1, :] = np.nanstd(distSeparability_ThisSubj, 0)
        outputName = folderOut + '/Subj' + pat + '_'+ HDtype+'_VectSeparability.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')
        dataToSave = np.zeros((2, numElem ))
        dataToSave[0, :] = np.nanmean(confidencesPerFeat_Train_ThisSubj, 0)
        dataToSave[1, :] = np.nanstd(confidencesPerFeat_Train_ThisSubj, 0)
        outputName = folderOut + '/Subj' + pat + '_'+ HDtype+'_'+FeatChType+'_TrainConfidence.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')
        dataToSave = np.zeros((2, numElem ))
        dataToSave[0, :] = np.nanmean(confidencesPerFeat_Test_ThisSubj, 0)
        dataToSave[1, :] = np.nanstd(confidencesPerFeat_Test_ThisSubj, 0)
        outputName = folderOut + '/Subj' + pat + '_'+ HDtype+'_'+FeatChType+'_TestConfidence.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')
        dataToSave = np.zeros((numElem + 1, 72))
        dataToSave[:, 0:36] = np.nanmean(performancesAll_Train_ThisSubj, 2)
        dataToSave[:, 36:] = np.nanstd(performancesAll_Train_ThisSubj, 2)
        outputName = folderOut + '/Subj' + pat + '_'+ HDtype+'_'+FeatChType+'_TrainPerformance.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')
        dataToSave = np.zeros((numElem + 1, 72))
        dataToSave[:, 0:36] = np.nanmean(performancesAll_Test_ThisSubj, 2)
        dataToSave[:, 36:] = np.nanstd(performancesAll_Test_ThisSubj, 2)
        outputName = folderOut + '/Subj' + pat + '_'+ HDtype+ '_'+FeatChType+'_TestPerformance.csv'
        saveDataToFile(dataToSave, outputName, 'gzip')

        # save avrg for this subj
        distSeparability_AllSubj[patIndx, :] = np.nanmean(distSeparability_ThisSubj, 0)
        confidencesPerFeat_Train_AllSubj[patIndx, :] = np.nanmean(confidencesPerFeat_Train_ThisSubj, 0)
        confidencesPerFeat_Test_AllSubj[patIndx, :] = np.nanmean(confidencesPerFeat_Test_ThisSubj, 0)
        performancesAll_Train_AllSubj[:, :, patIndx] = np.nanmean(performancesAll_Train_ThisSubj, 2)
        performancesAll_Test_AllSubj[:, :, patIndx] = np.nanmean(performancesAll_Test_ThisSubj, 2)

        # PLOTTING
        fig1 = plt.figure(figsize=(10, 6), constrained_layout=False)
        gs = GridSpec(2, 3, figure=fig1)
        fig1.subplots_adjust(wspace=0.2, hspace=0.3)
        fig1.suptitle('Comparing features Subj ' + pat)
        xValues = np.arange(0, numElem , 1)
        # vector separability
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.errorbar(xValues, np.nanmean(distSeparability_ThisSubj[:,0:-1], 0), yerr=np.nanstd(distSeparability_ThisSubj[:,0:-1], 0), fmt='k-.')
        ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(distSeparability_ThisSubj[0:-1, -1]), 'k')
        ax1.set_xlabel('Features')
        ax1.set_title('Vector separability')
        ax1.grid()
        # vector confidence
        ax1 = fig1.add_subplot(gs[0, 1])
        ax1.errorbar(xValues, np.nanmean(confidencesPerFeat_Train_ThisSubj, 0), yerr=np.nanstd(confidencesPerFeat_Train_ThisSubj, 0), color='black', linestyle='-.')
        ax1.errorbar(xValues, np.nanmean(confidencesPerFeat_Test_ThisSubj, 0), yerr=np.nanstd(confidencesPerFeat_Test_ThisSubj, 0), color='orangered', linestyle='-.')
        ax1.legend(['Train', 'Test'])
        # ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(confidencesPerFeat_Train_ThisSubj[:, -1]), 'b')
        # ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(confidencesPerFeat_Test_ThisSubj[:, -1]), 'r')
        ax1.set_xlabel('Features')
        ax1.set_title('Vector confidence')
        ax1.grid()
        # performances
        ax1 = fig1.add_subplot(gs[0, 2])
        mv = np.nanmean(performancesAll_Train_ThisSubj, 2)
        st = np.nanstd(performancesAll_Train_ThisSubj, 2)
        ax1.errorbar(xValues, mv[0:-1, 2], yerr=st[0:-1, 2], color='black', linestyle='-.')
        ax1.errorbar(xValues, mv[0:-1, 5], yerr=st[0:-1, 5],color='orangered', linestyle='-.')
        ax1.errorbar(xValues, mv[0:-1, 7], yerr=st[0:-1, 7], color='lightsalmon', linestyle='-.')
        ax1.legend(['F1E', 'F1D', 'F1both'])
        ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 2], color='black', linestyle='--')
        ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 5], color='orangered', linestyle='--')
        ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 7], color='lightsalmon', linestyle='--')
        # mv = np.nanmean(performancesAll_step1_Train_ThisSubj, 2)
        # st = np.nanstd(performancesAll_step1_Train_ThisSubj, 2)
        # ax1.errorbar(xValues, mv[0:-1, 2], yerr=st[0:-1, 2], fmt='k')
        # ax1.errorbar(xValues, mv[0:-1, 5], yerr=st[0:-1, 5], fmt='b')
        # ax1.errorbar(xValues, mv[0:-1, 7], yerr=st[0:-1, 7], fmt='m')
        # ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 2], 'k')
        # ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 5], 'b')
        # ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 7], 'm')
        ax1.set_xlabel('Features')
        ax1.set_title('Performance train')
        ax1.grid()
        # correlation confidences adn prformance
        ax1 = fig1.add_subplot(gs[1, 0])
        perf = np.nanmean(performancesAll_Train_ThisSubj, 2)
        ax1.plot(np.nanmean(confidencesPerFeat_Train_ThisSubj, 0), perf[0:-1, 7], color='black',  marker='x', linestyle='None')
        perf = np.nanmean(performancesAll_Test_ThisSubj, 2)
        ax1.plot(np.nanmean(confidencesPerFeat_Test_ThisSubj, 0), perf[0:-1, 7], color='lightgray',  marker='x', linestyle='None')
        ax1.legend()
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Performance F1both Smooth Step1')
        ax1.set_title('Correlation='+'{:.2f}'.format(np.corrcoef(np.nanmean(confidencesPerFeat_Test_ThisSubj, 0), perf[0:-1, 7])[0,1]))
        ax1.legend(['Train', 'Test'])
        ax1.grid()
        # correlation separability adn prformance
        ax1 = fig1.add_subplot(gs[1, 1])
        perf = np.nanmean(performancesAll_Train_ThisSubj, 2)
        ax1.plot(np.nanmean(distSeparability_ThisSubj[:,0:-1], 0), perf[0:-1, 7],  color='black',  marker='x', linestyle='None')
        perf = np.nanmean(performancesAll_Test_ThisSubj, 2)
        ax1.plot(np.nanmean(distSeparability_ThisSubj[:,0:-1], 0), perf[0:-1, 7], color='lightgray',  marker='x', linestyle='None')
        ax1.legend()
        ax1.set_xlabel('Separability')
        ax1.set_ylabel('Performance F1both Smooth Step1')
        ax1.set_title('Correlation='+'{:.2f}'.format(np.corrcoef(np.nanmean(distSeparability_ThisSubj[:,0:-1], 0), perf[0:-1, 7])[0,1]))
        ax1.legend(['Train', 'Test'])
        ax1.grid()
        # correlation confidences adn prformance
        ax1 = fig1.add_subplot(gs[1, 2])
        ax1.plot(np.nanmean(distSeparability_ThisSubj[:,0:-1], 0), np.nanmean(confidencesPerFeat_Train_ThisSubj, 0), color='black',  marker='x', linestyle='None')
        ax1.plot(np.nanmean(distSeparability_ThisSubj[:,0:-1], 0), np.nanmean(confidencesPerFeat_Test_ThisSubj, 0),  color='lightgray',  marker='x', linestyle='None')
        ax1.legend()
        ax1.set_xlabel('Separability')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Correlation ='+'{:.2f}'.format(np.corrcoef(np.nanmean(distSeparability_ThisSubj[:,0:-1], 0), np.nanmean(confidencesPerFeat_Test_ThisSubj, 0))[0,1]))
        ax1.legend(['Train', 'Test'])
        ax1.grid()
        fig1.show()
        fig1.savefig(folderOut + '/Subj' + pat+ '_'+ HDtype+'_FeatureComparison.png')
        # fig1.savefig(folderOut + '/Subj' + pat + '_'+ HDtype+ '_FeatureComparison.svg')
        plt.close(fig1)

    # save avrg of all subj  subj
    dataToSave = np.zeros((2, numElem + 1))
    dataToSave[0, :] = np.nanmean(distSeparability_AllSubj, 0)
    dataToSave[1, :] = np.nanstd(distSeparability_AllSubj, 0)
    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_VectSeparability.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')
    dataToSave = np.zeros((2, numElem ))
    dataToSave[0, :] = np.nanmean(confidencesPerFeat_Train_AllSubj, 0)
    dataToSave[1, :] = np.nanstd(confidencesPerFeat_Train_AllSubj, 0)
    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'_TrainConfidence.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')
    dataToSave = np.zeros((2, numElem))
    dataToSave[0, :] = np.nanmean(confidencesPerFeat_Test_AllSubj, 0)
    dataToSave[1, :] = np.nanstd(confidencesPerFeat_Test_AllSubj, 0)
    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'_TestConfidence.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')
    dataToSave = np.zeros((numElem + 1, 72))
    dataToSave[:, 0:36] = np.nanmean(performancesAll_Train_AllSubj, 2)
    dataToSave[:, 36:] = np.nanstd(performancesAll_Train_AllSubj, 2)
    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'_TrainPerformance.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')
    dataToSave = np.zeros((numElem + 1, 72))
    dataToSave[:, 0:36] = np.nanmean(performancesAll_Test_AllSubj, 2)
    dataToSave[:, 36:] = np.nanstd(performancesAll_Test_AllSubj, 2)
    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'_TestPerformance.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')

    # PLOTTING
    fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.5)
    fig1.suptitle('Comparing features All Subj ')
    xValues = np.arange(0, numElem , 1)
    # vector separability
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xValues, np.nanmean(distSeparability_AllSubj[:,0:-1], 0), yerr=np.nanstd(distSeparability_AllSubj[:,0:-1], 0), fmt='k-.')
    ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(distSeparability_AllSubj[0:-1, -1]), 'k')
    # ax1.set_xlabel('Features')
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(elemNames, fontsize=8 * 0.8, rotation=30)
    ax1.set_title('Vector separability')
    ax1.grid()
    # vector confidence
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.errorbar(xValues, np.nanmean(confidencesPerFeat_Train_AllSubj, 0), yerr=np.nanstd(confidencesPerFeat_Train_AllSubj, 0), color='black', linestyle='-.')
    ax1.errorbar(xValues, np.nanmean(confidencesPerFeat_Test_AllSubj, 0), yerr=np.nanstd(confidencesPerFeat_Test_AllSubj, 0), color='orangered', linestyle='-.')
    ax1.legend(['Train', 'Test'])
    # ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(confidencesPerFeat_Train_AllSubj[:, -1]), 'b')
    # ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(confidencesPerFeat_Test_AllSubj[:, -1]), 'r')
    # ax1.set_xlabel('Features')
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(elemNames, fontsize=8 * 0.8, rotation=30)
    ax1.set_title('Vector confidence')
    ax1.grid()
    # performances
    ax1 = fig1.add_subplot(gs[0, 2])
    mv = np.nanmean(performancesAll_Train_AllSubj, 2)
    st = np.nanstd(performancesAll_Train_AllSubj, 2)
    ax1.errorbar(xValues, mv[0:-1, 2], yerr=st[0:-1, 2], color='black', linestyle='-.')
    ax1.errorbar(xValues, mv[0:-1, 5], yerr=st[0:-1, 5], color='orangered', linestyle='-.')
    ax1.errorbar(xValues, mv[0:-1, 7], yerr=st[0:-1, 7], color='lightsalmon', linestyle='-.')
    ax1.legend(['F1E', 'F1D', 'F1both'])
    ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 2], color='black', linestyle='--')
    ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 5],  color='orangered', linestyle='--')
    ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 7],  color='lightsalmon', linestyle='--')
    # mv = np.nanmean(performancesAll_step1_Train_AllSubj, 2)
    # st = np.nanstd(performancesAll_step1_Train_AllSubj, 2)
    # ax1.errorbar(xValues, mv[0:-1, 2], yerr=st[0:-1, 2], fmt='k')
    # ax1.errorbar(xValues, mv[0:-1, 5], yerr=st[0:-1, 5], fmt='b')
    # ax1.errorbar(xValues, mv[0:-1, 7], yerr=st[0:-1, 7], fmt='m')
    # ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 2], 'k')
    # ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 5], 'b')
    # ax1.plot(xValues, np.ones(len(xValues)) * mv[-1, 7], 'm')
    # ax1.set_xlabel('Features')
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(elemNames, fontsize=8 * 0.8, rotation=30)
    ax1.set_title('Performance train')
    ax1.grid()
    # correlation confidences adn prformance
    ax1 = fig1.add_subplot(gs[1, 0])
    perf = np.nanmean(performancesAll_Train_AllSubj, 2)
    ax1.plot(np.nanmean(confidencesPerFeat_Train_AllSubj, 0), perf[0:-1, 7], color='black',  marker='x', linestyle='None')
    perf = np.nanmean(performancesAll_Test_AllSubj, 2)
    ax1.plot(np.nanmean(confidencesPerFeat_Test_AllSubj, 0), perf[0:-1, 7],  color='lightgray',  marker='x', linestyle='None')
    ax1.legend()
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Performance F1both Smooth Step1')
    ax1.set_title('Correlation='+'{:.2f}'.format(np.corrcoef(np.nanmean(confidencesPerFeat_Test_AllSubj, 0), perf[0:-1, 7])[0,1]))
    ax1.legend(['Train', 'Test'])
    ax1.grid()
    # correlation separability adn prformance
    ax1 = fig1.add_subplot(gs[1, 1])
    perf = np.nanmean(performancesAll_Train_AllSubj, 2)
    ax1.plot(np.nanmean(distSeparability_AllSubj[:,0:-1], 0), perf[0:-1, 7],  color='black',  marker='x', linestyle='None')
    perf = np.nanmean(performancesAll_Test_AllSubj, 2)
    ax1.plot(np.nanmean(distSeparability_AllSubj[:,0:-1], 0), perf[0:-1, 7],  color='lightgray',  marker='x', linestyle='None')
    ax1.legend()
    ax1.set_xlabel('Separability')
    ax1.set_ylabel('Performance F1both Smooth Step1')
    ax1.set_title('Correlation='+'{:.2f}'.format(np.corrcoef(np.nanmean(distSeparability_AllSubj[:,0:-1], 0), perf[0:-1, 7])[0,1]))
    ax1.legend(['Train', 'Test'])
    ax1.grid()
    # correlation confidences adn prformance
    ax1 = fig1.add_subplot(gs[1, 2])
    ax1.plot(np.nanmean(distSeparability_AllSubj[:,0:-1], 0), np.nanmean(confidencesPerFeat_Train_AllSubj, 0),  color='black',  marker='x', linestyle='None')
    ax1.plot(np.nanmean(distSeparability_AllSubj[:,0:-1], 0), np.nanmean(confidencesPerFeat_Test_AllSubj, 0), color='lightgray', marker='x', linestyle='None')
    ax1.legend()
    ax1.set_xlabel('Separability')
    ax1.set_ylabel('Confidence')
    ax1.set_title('Correlation='+'{:.2f}'.format(np.corrcoef(np.nanmean(distSeparability_AllSubj[:,0:-1], 0), np.nanmean(confidencesPerFeat_Test_AllSubj, 0))[0,1]))
    ax1.legend(['Train', 'Test'])
    ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'Comparison.png')
    fig1.savefig(folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'Comparison.svg')
    plt.close(fig1)


def plotPerAppendingComparison_ForPaper(folderOut, HDParams, FeaturesParams, DatasetPreprocessParams,  HDtype, FeatChType):
    if (FeatChType=='PerFeat'):
        numElem=HDParams.numFeat
        elemNames=FeaturesParams.featNames
    else:
        numElem=HDParams.numCh
        elemNames=DatasetPreprocessParams.channelNamesToKeep

    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_VectSeparability.csv'
    distSeparability_AllSubj= readDataFromFile(outputName)
    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'_TrainConfidence.csv'
    confidencesPerFeat_Train_AllSubj= readDataFromFile(outputName)
    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'_TestConfidence.csv'
    confidencesPerFeat_Test_AllSubj=readDataFromFile(outputName)
    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'_TrainPerformance.csv'
    performancesAll_Train_AllSubj=readDataFromFile(outputName)
    outputName = folderOut + '/AllSubj'+ '_'+ HDtype+'_'+FeatChType+'_TestPerformance.csv'
    performancesAll_Test_AllSubj=readDataFromFile(outputName)

    # PLOTTING
    fig1 = plt.figure(figsize=(12, 6), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig1)
    fig1.subplots_adjust(wspace=0.3, hspace=0.5)
    fig1.suptitle('Comparing features All Subj ')
    xValues = np.arange(0, numElem, 1)
    # vector separability
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.errorbar(xValues,distSeparability_AllSubj[0, 0:-1],yerr=distSeparability_AllSubj[1, 0:-1], fmt='k-.')
    ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(distSeparability_AllSubj[0:-1, -1]), 'k')
    # ax1.set_xlabel('Features')
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(elemNames, fontsize=8 * 0.8, rotation=90)
    ax1.set_title('Vector separability')
    ax1.grid()
    # vector confidence
    ax1 = fig1.add_subplot(gs[0, 1])
    ax1.errorbar(xValues, confidencesPerFeat_Train_AllSubj[0,:],
                 yerr=confidencesPerFeat_Train_AllSubj[1,:], color='black', linestyle='-.')
    ax1.errorbar(xValues,confidencesPerFeat_Test_AllSubj[0,:],
                 yerr=confidencesPerFeat_Train_AllSubj[1,:], color='orangered', linestyle='-.')
    ax1.legend(['Train', 'Test'])
    # ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(confidencesPerFeat_Train_AllSubj[:, -1]), 'b')
    # ax1.plot(xValues, np.ones(len(xValues)) * np.nanmean(confidencesPerFeat_Test_AllSubj[:, -1]), 'r')
    # ax1.set_xlabel('Features')
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(elemNames, fontsize=8 * 0.8, rotation=90)
    ax1.set_title('Vector confidence')
    ax1.grid()
    # performances
    ax1 = fig1.add_subplot(gs[0, 2])
    ax1.errorbar(xValues, performancesAll_Train_AllSubj[0:-1, 2], yerr=performancesAll_Train_AllSubj[0:-1, 36+2], color='black', linestyle='-.')
    ax1.errorbar(xValues, performancesAll_Train_AllSubj[0:-1, 5], yerr=performancesAll_Train_AllSubj[0:-1, 36+5], color='orangered', linestyle='-.')
    ax1.errorbar(xValues, performancesAll_Train_AllSubj[0:-1, 7], yerr=performancesAll_Train_AllSubj[0:-1, 36+7], color='lightsalmon', linestyle='-.')
    ax1.legend(['F1E', 'F1D', 'F1both'])
    ax1.plot(xValues, np.ones(len(xValues)) * performancesAll_Train_AllSubj[-1, 2], color='black', linestyle='--')
    ax1.plot(xValues, np.ones(len(xValues)) * performancesAll_Train_AllSubj[-1, 5], color='orangered', linestyle='--')
    ax1.plot(xValues, np.ones(len(xValues)) * performancesAll_Train_AllSubj[-1, 7], color='lightsalmon', linestyle='--')
    ax1.set_xticks(xValues)
    ax1.set_xticklabels(elemNames, fontsize=8 * 0.8, rotation=90)
    ax1.set_title('Performance train')
    ax1.grid()
    # correlation confidences adn prformance
    ax1 = fig1.add_subplot(gs[1, 0])
    # perf = np.nanmean(performancesAll_Train_AllSubj, 2)
    ax1.plot(confidencesPerFeat_Train_AllSubj[0,:], performancesAll_Train_AllSubj[0:-1, 7], color='black', marker='x', linestyle='None')
    # perf = np.nanmean(performancesAll_Test_AllSubj, 2)
    ax1.plot(confidencesPerFeat_Test_AllSubj[0,:],  performancesAll_Test_AllSubj[0:-1, 7], color='lightgray', marker='x', linestyle='None')
    ax1.legend()
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Performance F1both Smooth Step1')
    ax1.set_title('Correlation=' + '{:.2f}'.format( np.corrcoef(confidencesPerFeat_Test_AllSubj[0,:], performancesAll_Test_AllSubj[0:-1, 7])[0, 1]))
    ax1.legend(['Train', 'Test'])
    ax1.grid()
    # correlation separability adn prformance
    ax1 = fig1.add_subplot(gs[1, 1])
    # perf = np.nanmean(performancesAll_Train_AllSubj, 2)
    ax1.plot(distSeparability_AllSubj[0, 0:-1], performancesAll_Train_AllSubj[0:-1, 7], color='black', marker='x', linestyle='None')
    # perf = np.nanmean(performancesAll_Test_AllSubj, 2)
    ax1.plot(distSeparability_AllSubj[0, 0:-1], performancesAll_Test_AllSubj[0:-1, 7], color='lightgray', marker='x', linestyle='None')
    ax1.legend()
    ax1.set_xlabel('Separability')
    ax1.set_ylabel('Performance F1both Smooth Step1')
    ax1.set_title('Correlation=' + '{:.2f}'.format( np.corrcoef(distSeparability_AllSubj[0, 0:-1], performancesAll_Test_AllSubj[0:-1, 7])[0, 1]))
    ax1.legend(['Train', 'Test'])
    ax1.grid()
    # correlation confidences adn prformance
    ax1 = fig1.add_subplot(gs[1, 2])
    ax1.plot(distSeparability_AllSubj[0, 0:-1], confidencesPerFeat_Train_AllSubj[0,:], color='black', marker='x', linestyle='None')
    ax1.plot(distSeparability_AllSubj[0, 0:-1], confidencesPerFeat_Test_AllSubj[0,:], color='lightgray', marker='x', linestyle='None')
    ax1.legend()
    ax1.set_xlabel('Separability')
    ax1.set_ylabel('Confidence')
    ax1.set_title('Correlation=' + '{:.2f}'.format( np.corrcoef(distSeparability_AllSubj[0, 0:-1], confidencesPerFeat_Test_AllSubj[0,:])[ 0, 1]))
    ax1.legend(['Train', 'Test'])
    ax1.grid()
    fig1.show()
    fig1.savefig(folderOut + '/AllSubj' + '_' + HDtype + '_'+FeatChType+'Comparison_forPaper.png')
    fig1.savefig(folderOut + '/AllSubj' + '_' + HDtype + '_'+FeatChType+'Comparison_forPaper.svg')
    plt.close(fig1)