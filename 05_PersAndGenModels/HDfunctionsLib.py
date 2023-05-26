'''
library with different functions on HD vectors, from learning, predictions, calculating probabilities etc.
 it uses torch library '''

__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

import torch
import pickle
from VariousFunctionsLib import *
# from hyperdimGenerators import *
from hdtorch import pack, unpack, vcount, hcount
import math
import numpy as np
import random


class HD_classifier_General():
    ''' Base HD classifier with generic parameters and learning and prediction
    funtions '''
    def __init__(self, HDParams):
        ''' Initialize an HD vectors using the torch library
        SigInfoParams, SegSymbParams and HDParams contain all parameters needed
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''
        self.numClasses     = HDParams.numClasses
        self.NumFeat        = HDParams.numFeat
        self.HD_dim         = HDParams.D*(self.NumFeat if HDParams.bindingMethod == 'FeatAppend' else 1)
        HDParams.HD_dim= self.HD_dim
        self.lr             = HDParams.onlineHDLearnRate
        self.onlineFNSfact  = HDParams.onlineFNSfact
        self.packed         = HDParams.packed
        self.device         = HDParams.device
        self.roundingType   = HDParams.roundingTypeForHDVectors
        self.bindingMethod  = HDParams.bindingMethod
        self.HDvecType      = HDParams.HDvecType
        self.modelVectorClasses = torch.tensor([0, 1])

        self.modelVectors = torch.zeros((self.numClasses, self.HD_dim), device=self.device)
        if self.packed:
            self.modelVectorsNorm= torch.zeros((self.numClasses, math.ceil(self.HD_dim/32)), device = self.device, dtype=torch.int32)
        else:
            self.modelVectorsNorm= torch.zeros((self.numClasses, self.HD_dim), device = self.device, dtype=torch.int8)
        self.numAddedVecPerClass = torch.zeros(self.numClasses, device=self.device)
        self.dist_func = self.ham_dist_arr if HDParams.similarityType=='hamming' else self.cos_dist_arr

        #CREATING VALUE LEVEL VECTORS
        self.proj_mat_features = generateHDVector(HDParams.vectorTypeFeat,self.NumFeat,HDParams.D,self.packed,self.device, 'HDvecs_features')
        self.proj_mat_features[self.proj_mat_features==0] = -1 if self.HDvecType=='bipol' else 0

        self.proj_mat_FeatVals = generateHDVector(HDParams.vectorTypeLevel,HDParams.numSegmentationLevels,HDParams.D,self.packed,self.device, 'HDvecs_featValues')
        self.proj_mat_FeatVals[self.proj_mat_FeatVals==0] = -1 if self.HDvecType=='bipol' else 0

        self.encoded_features = torch.bitwise_xor(self.proj_mat_FeatVals.unsqueeze(0),self.proj_mat_features.unsqueeze(1))

    def bind(self, data):
        if self.bindingMethod == 'FeatAppend':
            return self.proj_mat_FeatVals[data].view(data.shape[0],1,-1)
        elif self.bindingMethod == 'FeatValRot':
            return rotateVec(self.proj_mat_features, data)
        elif self.HDvecType == 'bin' and self.packed:
            return torch.bitwise_xor(self.proj_mat_features.unsqueeze(0), self.proj_mat_FeatVals[data])
        elif self.HDvecType == 'bin' and not self.packed:
            r = torch.arange(0,self.NumFeat,device=self.device)
            self.encoded_features[r,data]
        elif self.HDvecType == 'binand' and self.packed:
            return torch.bitwise_and(self.proj_mat_features.unsqueeze(0), self.proj_mat_FeatVals[data])
        elif self.HDvecType == 'binand' and not self.packed:
            raise TypeError('Not supported HDvecType!')
        elif self.HDvecType == 'bipol' and not self.packed:
            xor_bipolar(self.proj_mat_features.unsqueeze(0), self.proj_mat_FeatVals[data])
        elif self.HDvecType == 'bipol' and self.packed:
            raise TypeError('Model cannot be bipol and packed simultanously')
        else:
            raise TypeError(f'Unrecognized combination of bindingMethod: {self.bindingMethod}, HDvecType: {self.HDvecType}, packed: {self.packed}')

    def learn_HD_proj(self, data, label,bindingMethod):
        ''' From features of one window to HDvector representing that data window
        '''
        outputVector = torch.empty((data.shape[0],self.HD_dim),dtype = torch.int8, device = data.device)
        packedOutput = torch.full((data.shape[0],math.ceil(self.HD_dim/32)),-1,dtype = torch.int32, device = data.device) if self.packed else -1
        if self.device != 'cpu':
            t = torch.cuda.get_device_properties(data.device).total_memory
            a = torch.cuda.memory_allocated(data.device)
            b = min((t-a)//(data.shape[1]*self.proj_mat_FeatVals.shape[1]*4*12),65535)
            b=2000
        else:
            b = 10000
        for i in range(0,data.shape[0],b):
            if self.packed:
                a = vcount(self.bind(data[i:i+b]),self.HD_dim)
            else:
                a = self.bind(data[i:i+b]).sum(1,dtype=torch.int16)
                # a = self.encoded_features[self.r,data[i:i+b]].view(-1,171,2,self.HD_dim).sum(1,dtype=torch.uint8).sum(1,dtype=torch.int16)

            if self.bindingMethod != 'FeatAppend':
                if (self.roundingType != 'noRounding'):
                    if  self.HDvecType=='bin':
                        f = int(math.floor(self.NumFeat / 2))
                    elif self.HDvecType=='binand':
                        f =  int(math.floor(self.NumFeat / 4))
                    else:
                        f=0
                    torch.gt(a,f,out=outputVector[i:i+b])
                else:
                    outputVector = (outputVector[i] / self.NumFeat)
            else:
                print('ERROR - not defined properly')
                outputVector[i] = a.to(torch.int8)

        if self.HDvecType == 'bipol' and self.roundingType != 'noRounding':
            outputVector[outputVector==0] = -1

        if self.packed:
            pack(outputVector,packedOutput)
        # torch.cuda.synchronize()
        return outputVector, packedOutput, label

    def ham_dist_arr(self, vec_a, vec_b):
        ''' calculate relative hamming distance for for np array'''
        vec_c = torch.bitwise_xor(vec_a,vec_b) if ('bin' in self.HDvecType) else vec_a != vec_b
        if self.packed:
            return hcount(vec_c)/float(vec_b.shape[2]*32) #float(self.HD_dim)
        else:
            print('SOMTHING WRONG IN HAM_DIST_ARR - CHECK !!')
            vec_c = vec_c.view(vec_c.shape[0],vec_c.shape[1],250 if self.HD_dim==10000 else 128,-1).sum(2,dtype=torch.uint8)
            return vec_c.sum(2,dtype=torch.int16)/float(self.HD_dim)

    def cos_dist_arr(self, vA, vB):
        ''' calculate cosine distance of two vectors'''
        # output = np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))
        output = torch.dot(vA, vB) / (torch.sqrt(torch.dot(vA, vA)) * torch.sqrt(torch.dot(vB, vB)))
        outTensor = torch.tensor(1.0-output) #because we use latter that if value is 0 then vectors are the same
        return outTensor

    def calcDist_keepOnlyUsefulBits(self,data, modelVectorsNorm):
        modelVectorsNormUnpack=torch.zeros((2,self.HD_dim)).to(dtype=torch.int32, device=self.device)
        modelVectorsNormUnpack[0,:] = unpack(modelVectorsNorm[0,:], self.HD_dim)
        modelVectorsNormUnpack[1,:] = unpack(modelVectorsNorm[1, :], self.HD_dim)
        dataUnpack = unpack(data, self.HD_dim)
        indx=torch.where(modelVectorsNormUnpack[0,:] !=modelVectorsNormUnpack[1,:] )[0]
        if (len(indx)==0): #just to have something
            indx=[0,1]
        dataRed=dataUnpack[:,:,indx]
        modelVectorsNormRed=modelVectorsNormUnpack[:,indx]
        distances=self.dist_func(dataRed, modelVectorsNormRed.unsqueeze(0))

        Mdiff = torch.mean(abs(modelVectorsNormUnpack[0,:].to(dtype=torch.float) - modelVectorsNormUnpack[1,:].to(dtype=torch.float) )) * 100
        print('Model diff - in pred: ', Mdiff)

        return( distances)

    def givePrediction(self,data, label0, bindingMethod, encoded=False, thrTS=0.5, thrTNS=0.5 ):
        if not encoded:
            (temp, tempPacked, label) = self.learn_HD_proj(data,label0, bindingMethod)
            temp = temp if not self.packed else tempPacked
            data2 = temp.view(-1,1,temp.shape[-1])
        else:
            data2 = data.view(-1,1,data.shape[-1])


        distances0 = self.dist_func(data2, self.modelVectorsNorm.unsqueeze(0))
        distances=softabs(distances0,20)
        # the same as above, but uses only bits that are different between seiz and nonseiz (so that probabilitiey are much more diverse)
        # distances0= self.calcDist_keepOnlyUsefulBits( data2,  self.modelVectorsNorm)
        # distances = softabs(distances0, 20)

        #find minimum
        minIndx = torch.argmin(distances,-1)
        predLabel= self.modelVectorClasses[minIndx]

        # go through each class adn calculate dist from each class
        classes=list(set(self.modelVectorClasses.cpu().detach().numpy()))
        distPerClass=torch.zeros((len(distances[:,0]), len(classes))).to(device=self.device)
        for ci, c in enumerate(classes):
            indx=torch.where(self.modelVectorClasses==c)[0]
            distPerClass[:,ci]= torch.min(distances[:,indx],-1)[0]
            # distPerClass[:, ci] = torch.mean(distances[:, indx], -1)

        # torch.cuda.synchronize()
        distFromVotedClass = distPerClass.min(1)[0]
        distFromNextClass = distPerClass.kthvalue(2, dim=1)[0]
        probabLabels = distFromNextClass / (distFromNextClass + distFromVotedClass + 0.0000001)


        #if predicted seizure is not confident enough, transform it to NS
        indx=torch.where((probabLabels<thrTS) & (predLabel==1))[0]
        predLabel[indx]=0
        probabLabels[indx]=1-probabLabels[indx]

        # #if predicted nonseizure but not confidend enugh transform to S
        # indx=torch.where((probabLabels<thrTNS) & (predLabel==0))[0]
        # predLabel[indx]=1
        # probabLabels[indx]=1-probabLabels[indx]

        return (predLabel.squeeze(), distPerClass.squeeze(), probabLabels.squeeze(), label)

    def givePrediction_train(self,data,label0 ,encoded=False):
        if not encoded:
            (temp, tempPacked,label) = self.learn_HD_proj(data, label0)
            temp = temp if not self.packed else tempPacked
            data = temp.view(-1,1,temp.shape[-1])
        else:
            data = data.view(-1,1,data.shape[-1])
            label=label0

        distances0 = self.dist_func(data, self.modelVectorsNorm.unsqueeze(0))
        distances= softabs(distances0,20)
        #the same as above, but uses only bits that are different between seiz and nonseiz (so that probabilitiey are much more diverse)
        # distances= self.calcDist_keepOnlyUsefulBits( data,  self.modelVectorsNorm)

        #find minimum
        minIndx = torch.argmin(distances,-1)
        minVal= self.modelVectorClasses[minIndx]
        # torch.cuda.synchronize()
        # go through each class adn calculate dist from each class
        classes=list(set(self.modelVectorClasses.cpu().detach().numpy()))
        distPerClass=torch.zeros((len(distances[:,0]), len(classes)))
        for ci, c in enumerate(classes):
            indx=torch.where(self.modelVectorClasses==c)[0]
            distPerClass[:,ci]= torch.min(distances[:,indx],-1)[0]
            # distPerClass[:, ci] = torch.mean(distances[:, indx], -1)

        # return (minVal.squeeze(), distances.squeeze(), label)
        return (minVal.squeeze(), distPerClass.squeeze(), label)

    def trainModelVecOnData(self, data, labels):
        '''learn model vectors for single pass 2 class HD learning '''
        numLabels = self.modelVectors.shape[0]

        # Encode data
        (temp, packedTemp, labels) = self.learn_HD_proj(data, labels)
        temp = temp if not self.packed else packedTemp

        #go through all data windows and add them to model vectors
        m200 = temp.shape[0]//200
        for l in range(numLabels):
            t = temp[labels==l,:]
            if(t.numel() != 0 and self.packed):
                self.modelVectors[l, :] += vcount(temp[labels==l],self.HD_dim) #temp[labels==l].sum(0)
            elif not self.packed:
                self.modelVectors[l] += temp[labels==l].sum(0)
                # tempL = temp[labels==l]
                # m200 = tempL.shape[0]//200
                # self.modelVectors[l] += tempL[0:200*m200].view(200,-1,temp.shape[1]).sum(0,dtype=torch.uint8).sum(0,dtype=torch.int32)
                # self.modelVectors[l] += tempL[200*m200:].sum(0,dtype=torch.uint8)
            self.numAddedVecPerClass[l] +=  (labels==l).sum().cpu() #count number of vectors added to each subclass

        # normalize model vectors to be binary (or bipolar) again
        if self.roundingType != 'noRounding':
            if ('bin' and self.HDvecType)  and self.packed:
                    pack(self.modelVectors > (self.numAddedVecPerClass.unsqueeze(1)>>1),self.modelVectorsNorm)
            # elif (self.HDvecType == 'binand')   and self.packed:
            #         pack(self.modelVectors > (self.numAddedVecPerClass.unsqueeze(1)>>2),self.modelVectorsNorm)
            elif  ('bin' and self.HDvecType)  and not self.packed:
                    self.modelVectorsNorm = self.modelVectors > (self.numAddedVecPerClass.unsqueeze(1)>>1)
            # elif (self.HDvecType == 'binand')  and not self.packed:
            #         self.modelVectorsNorm = self.modelVectors > (self.numAddedVecPerClass.unsqueeze(1)>>2)
            elif self.HDvecType == 'bipol':
                self.modelVectorsNorm = torch.where(self.modelVectors>0,1,-1)
        else:
            self.modelVectorsNorm = self.modelVectors/self.numAddedVecPerClass.unsqueeze(1)

        # torch.cuda.synchronize()

    def trainModelVecOnData_withRetrainPossible(self, data, labels0, type, bindingMethod):
        '''learn model vectors for single pass 2 class HD learning '''
        numLabels = self.modelVectors.shape[0]

        #store initial model values
        ModelVectorsOrig = torch.clone(self.modelVectors)
        numAddedVecPerClassOrig = torch.clone(self.numAddedVecPerClass)

        # Encode data
        (temp, packedTemp, labels) = self.learn_HD_proj(data, labels0, bindingMethod)
        temp = temp if not self.packed else packedTemp

        # go through all data windows and add them to model vectors
        m200 = temp.shape[0] // 200
        for l in range(numLabels):
            t = temp[labels == l, :]
            if (t.numel() != 0 and self.packed):
                self.modelVectors[l, :] += vcount(temp[labels == l], self.HD_dim)  # temp[labels==l].sum(0)
            elif not self.packed:
                self.modelVectors[l] += temp[labels == l].sum(0)
                # tempL = temp[labels==l]
                # m200 = tempL.shape[0]//200
                # self.modelVectors[l] += tempL[0:200*m200].view(200,-1,temp.shape[1]).sum(0,dtype=torch.uint8).sum(0,dtype=torch.int32)
                # self.modelVectors[l] += tempL[200*m200:].sum(0,dtype=torch.uint8)
            self.numAddedVecPerClass[l] += (labels == l).sum().cpu()  # count number of vectors added to each subclass

        # if only one was supposed to be retrained set it back to orig value
        if (type == 'gen-retrain-seiz' or type == 'generalized' ):
            self.modelVectors[0, :] = ModelVectorsOrig[0, :]
            self.numAddedVecPerClass[0] = numAddedVecPerClassOrig[0]
        if (type == 'gen-retrain-nonSeiz' or type == 'generalized'):
            self.modelVectors[1, :] = ModelVectorsOrig[1, :]
            self.numAddedVecPerClass[1] = numAddedVecPerClassOrig[1]

        # normalize model vectors to be binary (or bipolar) again
        if self.roundingType != 'noRounding':
            if ('bin' in self.HDvecType) and self.packed:
                pack((self.modelVectors > (self.numAddedVecPerClass.unsqueeze(1) >> 1)).to(dtype=torch.int8), self.modelVectorsNorm)
            # elif (self.HDvecType=='binand')  and self.packed:
            #     pack((self.modelVectors > (self.numAddedVecPerClass.unsqueeze(1) >> 2)).to(dtype=torch.int8), self.modelVectorsNorm)
            elif ('bin' in self.HDvecType) and not self.packed:
                self.modelVectorsNorm = self.modelVectors > (self.numAddedVecPerClass.unsqueeze(1) >> 1)
            # elif (self.HDvecType=='binand')  and not self.packed:
            #     self.modelVectorsNorm = self.modelVectors > (self.numAddedVecPerClass.unsqueeze(1) >> 2)
            elif self.HDvecType == 'bipol':
                self.modelVectorsNorm = torch.where(self.modelVectors > 0, 1, -1)
        else:
            self.modelVectorsNorm = self.modelVectors / self.numAddedVecPerClass.unsqueeze(1)

        # torch.cuda.synchronize()

    def save(self, modelPath):
        with open(modelPath,'wb') as f:
            pickle.dump(self,f)
#
# class HD_classifier_GeneralWithChCombinations(HD_classifier_General):
#     ''' Approach that uses several features and then maps them all to HD vectors
#     but know that some features are from different channels so there are different ways to approach it'''
#
#     def __init__(self, HDParams, numCh):
#         ''' Initialize an HD vectors using the torch library
#         SigInfoParams, SegSymbParams and HDParams contain all parameters needed
#         cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
#         '''
#         HD_classifier_General.__init__(self, HDParams)
#         self.modelVectorClasses=torch.tensor([0,1])
#         #CREATING CHANNEL VECTORS
#         if self.CHvectTypes == 'sandwich':
#             self.proj_mat_channels = func_generateVectorsMemory_Sandwich(self.device, self.HD_dim, self.NumCh, self.HDvecType)
#         elif self.CHvectTypes == 'random':
#             self.proj_mat_channels = func_generateVectorsMemory_Random(self.device, self.HD_dim, self.NumCh, self.HDvecType)
#         elif "scaleNoRand" in self.CHvectTypes:
#             factor = int(self.CHvectTypes[11:])
#             self.proj_mat_channels = func_generateVectorsMemory_ScaleNoRand(self.device, self.HD_dim, self.NumCh,factor, self.HDvecType)
#         elif "scaleRand" in self.CHvectTypes:
#             factor = int(self.CHvectTypes[9:])
#             self.proj_mat_channels = func_generateVectorsMemory_ScaleRand(self.device, self.HD_dim, self.NumCh, factor, self.HDvecType)
#         elif "scaleWithRadius" in self.CHvectTypes:
#             factor = int(self.CHvectTypes[15:])
#             hdVec = func_generateVectorsMemory_ScaleWithRadius(self.device, self.HD_dim,  factor, self.HDvecType)
#         else:
#             self.proj_mat_channels = func_generateVectorsMemory_Random(self.device, self.HD_dim, self.NumCh, self.HDvecType)
#
#         # CREATING COMBINED VECTORS FOR CH ANF FEATURES  - but only random
#         if (self.bindingMethod == 'ChFeatCombxVal'):
#             self.proj_mat_featuresCh = func_generateVectorsMemory_Random(self.device, self.HD_dim,self.NumFeat * self.NumCh, self.HDvecType)
#
#     def learn_HD_proj(self,data, label, HDParams):
#         ''' From features of one window to HDvector representing that data window
#         different options possible:'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend'
#         '''
#         t0=time.time()
#
#         data.resize(self.NumCh , self.NumFeat)
#         data=data.astype(int)
#         if (self.bindingMethod == 'FeatxVal'):
#             Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
#             for ch in range(self.NumCh):
#                 bindingVector = xor(self.proj_mat_features, self.proj_mat_FeatVals[:,data[ch,:]])
#                 Chvect_matrix[:,ch] = torch.sum(bindingVector, dim=1)
#             output_vector = torch.sum(Chvect_matrix, dim=1)
#             if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                 output_vector = (output_vector > int(math.floor(self.NumCh * self.NumFeat / 2))).short()
#             else:
#                 output_vector = (output_vector / (self.NumCh * self.NumFeat))
#
#         elif (self.bindingMethod =='ChxFeatxVal'):
#             Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
#             for ch in range(self.NumCh):
#                 #bind features and their values
#                 bindingVector = xor(self.proj_mat_features, self.proj_mat_FeatVals[:, data[ch,:]])
#                 bindingVectorSum=torch.sum(bindingVector, dim=1)
#                 #binarizing
#                 Chvect_matrix[:,ch]  = (bindingVectorSum > int(math.floor(self.NumFeat/ 2))).short()
#             #binding with  channels vectors
#             bindingVector2=xor(Chvect_matrix, self.proj_mat_channels)
#             output_vector = torch.sum(bindingVector2, dim=1)
#             if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                 output_vector = (output_vector > int(math.floor(self.NumCh/ 2))).short()
#             else:
#                 output_vector = (output_vector / self.NumCh)
#
#         elif (self.bindingMethod =='FeatxChxVal'):
#             Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
#             for f in range(self.NumFeat):
#                 #bind ch and  values of current feature
#                 bindingVector = xor(self.proj_mat_channels, self.proj_mat_FeatVals[:, data[:,f]])
#                 bindingVectorSum=torch.sum(bindingVector, dim=1)
#                 #binarizing
#                 Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh/ 2))).short()
#             if (self.NumFeat>1):
#                 #binding with  feature vectors
#                 bindingVector2=xor(Featvect_matrix, self.proj_mat_features)
#                 output_vector = torch.sum(bindingVector2, dim=1)
#                 output_vector = torch.sum(bindingVector2, dim=1)
#                 if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                     output_vector = (output_vector > int(math.floor(self.NumFeat/ 2))).short()
#                 else:
#                     output_vector = (output_vector / self.NumFeat)
#             else: #special case when we have only one feature
#                 output_vector=Featvect_matrix[:,0]
#
#         elif (self.bindingMethod =='ChFeatCombxVal'):
#             Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat*self.NumCh).cuda(device=self.device)
#             for f in range(self.NumFeat):
#                 for ch in range(self.NumCh):
#                     # bind ch and  values of current feature
#                     Featvect_matrix[:,f*self.NumCh+ch] = xor(self.proj_mat_featuresCh[:,f*self.NumCh+ch], self.proj_mat_FeatVals[:, data[ch,f]])
#             output_vector = torch.sum(Featvect_matrix, dim=1)
#             if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                 output_vector = (output_vector > int(math.floor(self.NumFeat*self.NumCh / 2))).short()
#             else:
#                 output_vector = (output_vector / (self.NumFeat*self.NumCh ))
#
#         elif ('FeatAppend' in self.bindingMethod):
#             Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
#             output_vector=torch.zeros(self.HD_dim*self.NumFeat,1).cuda(device=self.device)
#             for f in range(self.NumFeat):
#                 #bind ch and  values of current feature
#                 bindingVector = xor(self.proj_mat_channels, self.proj_mat_FeatVals[:, data[:,f]])
#                 bindingVectorSum=torch.sum(bindingVector, dim=1)
#                 #binarizing
#                 if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                     Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh / 2))).short()
#                 else:
#                     Featvect_matrix[:,f]  = (bindingVectorSum /  self.NumCh)
#                 #apppending
#                 output_vector[f*self.HD_dim:(f+1)*self.HD_dim,0]=Featvect_matrix[:,f]
#
#
#         timeHDVec=time.time()-t0
#
#         return output_vector, timeHDVec, label
#
#     def learn_HD_proj_bipolar(self,data, label, HDParams):
#         ''' From features of one window to HDvector representing that data window
#         different options possible:'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend'
#         '''
#         t0=time.time()
#
#         data.resize(self.NumCh , self.NumFeat)
#         data=data.astype(int)
#         if (self.bindingMethod == 'FeatxVal'):
#             Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
#             for ch in range(self.NumCh):
#                 bindingVector = xor_bipolar(self.proj_mat_features, self.proj_mat_FeatVals[:,data[ch,:]]) #!!! diffferent from binary
#                 Chvect_matrix[:,ch] = torch.sum(bindingVector, dim=1)
#             output_vector = torch.sum(Chvect_matrix, dim=1)
#             if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                 #output_vector = (output_vector > int(math.floor(self.NumCh * self.NumFeat / 2))).short()
#                 output_vector = (output_vector > 0).short() #!!! diffferent from binary
#             else:
#                 output_vector = (output_vector / (self.NumCh * self.NumFeat))
#
#         elif (self.bindingMethod =='ChxFeatxVal'):
#             Chvect_matrix = torch.zeros(self.HD_dim, self.NumCh).cuda(device=self.device)
#             for ch in range(self.NumCh):
#                 #bind features and their values
#                 bindingVector = xor_bipolar(self.proj_mat_features, self.proj_mat_FeatVals[:, data[ch,:]]) #!!! diffferent from binary
#                 bindingVectorSum=torch.sum(bindingVector, dim=1)
#                 #binarizing
#                 #Chvect_matrix[:,ch]  = (bindingVectorSum > int(math.floor(self.NumFeat/ 2))).short()
#                 Chvect_matrix[:, ch] = (bindingVectorSum > 0).short() #!!! diffferent from binary
#             #binding with  channels vectors
#             bindingVector2=xor_bipolar(Chvect_matrix, self.proj_mat_channels) #!!! diffferent from binary
#             output_vector = torch.sum(bindingVector2, dim=1)
#             if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                 #output_vector = (output_vector > int(math.floor(self.NumCh/ 2))).short()
#                 output_vector = (output_vector > 0).short()  # !!! diffferent from binary
#             else:
#                 output_vector = (output_vector / self.NumCh)
#
#         elif (self.bindingMethod =='FeatxChxVal'):
#             Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
#             for f in range(self.NumFeat):
#                 #bind ch and  values of current feature
#                 bindingVector = xor_bipolar(self.proj_mat_channels, self.proj_mat_FeatVals[:, data[:,f]]) #!!! diffferent from binary
#                 bindingVectorSum=torch.sum(bindingVector, dim=1)
#                 #binarizing
#                 #Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh/ 2))).short()
#                 Featvect_matrix[:,f]  = (bindingVectorSum > 0).short()# !!! diffferent from binary
#             if (self.NumFeat>1):
#                 #binding with  feature vectors
#                 bindingVector2=xor_bipolar(Featvect_matrix, self.proj_mat_features) #!!! diffferent from binary
#                 output_vector = torch.sum(bindingVector2, dim=1)
#                 if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                     #output_vector = (output_vector > int(math.floor(self.NumFeat/ 2))).short()
#                     output_vector = (output_vector > 0).short()  # !!! diffferent from binary
#                 else:
#                     output_vector = (output_vector / self.NumFeat)
#             else: #special case when we have only one feature
#                 output_vector=Featvect_matrix[:,0]
#
#         elif (self.bindingMethod =='ChFeatCombxVal'):
#             Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat*self.NumCh).cuda(device=self.device)
#             for f in range(self.NumFeat):
#                 for ch in range(self.NumCh):
#                     # bind ch and  values of current feature
#                     Featvect_matrix[:,f*self.NumCh+ch] = xor_bipolar(self.proj_mat_featuresCh[:,f*self.NumCh+ch], self.proj_mat_FeatVals[:, data[ch,f]]) #!!! diffferent from binary
#             output_vector = torch.sum(Featvect_matrix, dim=1)
#             if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                 #output_vector = (output_vector > int(math.floor(self.NumFeat*self.NumCh / 2))).short()
#                 output_vector = (output_vector > 0).short()  # !!! diffferent from binary
#             else:
#                 output_vector = (output_vector / (self.NumFeat*self.NumCh ))
#
#         elif ('FeatAppend' in self.bindingMethod):
#             Featvect_matrix = torch.zeros(self.HD_dim, self.NumFeat).cuda(device=self.device)
#             output_vector=torch.zeros(self.HD_dim*self.NumFeat,1).cuda(device=self.device)
#             for f in range(self.NumFeat):
#                 #bind ch and  values of current feature
#                 bindingVector = xor_bipolar(self.proj_mat_channels, self.proj_mat_FeatVals[:, data[:,f]]) #!!! diffferent from binary
#                 bindingVectorSum=torch.sum(bindingVector, dim=1)
#                 #binarizing
#                 if (HDParams.roundingTypeForHDVectors != 'noRounding'):
#                     #Featvect_matrix[:,f]  = (bindingVectorSum > int(math.floor(self.NumCh / 2))).short()
#                     Featvect_matrix[:, f] = (bindingVectorSum > 0).short()  # !!! diffferent from binary
#                 else:
#                     Featvect_matrix[:,f]  = (bindingVectorSum /  self.NumCh)
#                 #apppending
#                 output_vector[f*self.HD_dim:(f+1)*self.HD_dim,0]=Featvect_matrix[:,f]
#
#         #replace 0 with -1 after rounnn
#         if (HDParams.roundingTypeForHDVectors != 'noRounding'): #!!! diffferent from binary
#             output_vector[output_vector==0] =-1
#         timeHDVec=time.time()-t0
#
#         return output_vector, timeHDVec, label
#
#
class HD_classifier_TimeTracking(HD_classifier_General):
    ''' Approach that uses several features and then maps them all to HD vectors
    but know that some features are from different channels so there are different ways to approach it'''

    def __init__(self, HDParams):
        ''' Initialize an HD vectors using the torch library
        SigInfoParams, SegSymbParams and HDParams contain all parameters needed
        cuda: this parameter is fixed to true by now. The code MUST be ran on GPU.
        '''
        # HD_classifier_General.__init__(self, HDParams)
        self.modelVectorClasses = torch.tensor([0, 1])
        self.numClasses = HDParams.numClasses
        self.NumFeat = HDParams.numFeat
        self.D=HDParams.D
        if 'Time' in HDParams.bindingMethod:
            numTimes=len(HDParams.timeStepsInSec)
            self.timeStepsInSec=HDParams.timeStepsInSec
            self.timeStepsWinShift=HDParams.timeStepsWinShift
            if ('App' in HDParams.bindingMethod):  #if appending
                self.HD_dim = HDParams.D * numTimes
            else:
                self.HD_dim=HDParams.D
        else:
            self.HD_dim = HDParams.D * (self.NumFeat if HDParams.bindingMethod == 'FeatAppend' else 1)
        HDParams.HD_dim=self.HD_dim
        self.numSegmentationLevels=HDParams.numSegmentationLevels
        self.lr = HDParams.onlineHDLearnRate
        self.onlineFNSfact = HDParams.onlineFNSfact
        self.packed = HDParams.packed
        self.device = HDParams.device
        self.roundingType = HDParams.roundingTypeForHDVectors
        self.bindingMethod = HDParams.bindingMethod
        self.HDvecType = HDParams.HDvecType

        self.modelVectors = torch.zeros((self.numClasses, self.HD_dim), device=self.device)
        if self.packed:
            self.modelVectorsNorm = torch.zeros((self.numClasses, math.ceil(self.HD_dim / 32)), device=self.device, dtype=torch.int32)
        else:
            self.modelVectorsNorm = torch.zeros((self.numClasses, self.HD_dim), device=self.device, dtype=torch.int8)
        self.numAddedVecPerClass = torch.zeros(self.numClasses, device=self.device)
        self.dist_func = self.ham_dist_arr if HDParams.similarityType == 'hamming' else self.cos_dist_arr

        # CREATING VALUE LEVEL VECTORS
        self.proj_mat_features = generateHDVector(HDParams.vectorTypeFeat, self.NumFeat, HDParams.D, self.packed,  self.device, 'HDvecs_features')
        self.proj_mat_features[self.proj_mat_features == 0] = -1 if self.HDvecType == 'bipol' else 0

        self.proj_mat_FeatVals = generateHDVector(HDParams.vectorTypeLevel, HDParams.numSegmentationLevels*2+1, HDParams.D,  self.packed, self.device, 'HDvecs_featValues') ##we have double of the range when calculating distances
        self.proj_mat_FeatVals[self.proj_mat_FeatVals == 0] = -1 if self.HDvecType == 'bipol' else 0

        self.encoded_features = torch.bitwise_xor(self.proj_mat_FeatVals.unsqueeze(0),  self.proj_mat_features.unsqueeze(1))

    def bind(self, data):
        if self.bindingMethod == 'FeatAppend':
            return self.proj_mat_FeatVals[data].view(data.shape[0], 1, -1)
        elif self.bindingMethod == 'FeatValRot':
            return rotateVec(self.proj_mat_features, data)
        elif self.HDvecType == 'bin' and self.packed:
            return torch.bitwise_xor(self.proj_mat_features.unsqueeze(0), self.proj_mat_FeatVals[data+self.numSegmentationLevels]) #double the range so that we can encode feature diffeence in time
        elif self.HDvecType == 'bin' and not self.packed:
            r = torch.arange(0, self.NumFeat, device=self.device)
            self.encoded_features[r, data]
        elif self.HDvecType == 'binand': #and self.packed:
            return torch.bitwise_and(self.proj_mat_features.unsqueeze(0), self.proj_mat_FeatVals[data+self.numSegmentationLevels]) #double the range so that we can encode feature diffeence in time
        # elif self.HDvecType == 'binand' and not self.packed:
        #     raise TypeError('Not supported HDvecType!')
        elif self.HDvecType == 'bipol' and not self.packed:
            xor_bipolar(self.proj_mat_features.unsqueeze(0), self.proj_mat_FeatVals[data+self.numSegmentationLevels])#double the range so that we can encode feature diffeence in time
        elif self.HDvecType == 'bipol' and self.packed:
            raise TypeError('Model cannot be bipol and packed simultanously')
        else:
            raise TypeError(
                f'Unrecognized combination of bindingMethod: {self.bindingMethod}, HDvecType: {self.HDvecType}, packed: {self.packed}')

    def learn_HD_proj_base(self, data):
        ''' From features of one window to HDvector representing that data window
        '''
        if (self.roundingType != 'noRounding'):
            outputVector = torch.empty((data.shape[0],self.D),dtype = torch.int8, device = data.device)
        else:
            outputVector = torch.empty((data.shape[0], self.D), dtype=torch.float, device=data.device)
        packedOutput = torch.full((data.shape[0],math.ceil(self.D/32)),-1,dtype = torch.int32, device = data.device) if self.packed else -1
        b=1000
        # if self.device != 'cpu':
        #     t = torch.cuda.get_device_properties(data.device).total_memory
        #     a = torch.cuda.memory_allocated(data.device)
        #     b = min((t-a)//(data.shape[1]*self.proj_mat_FeatVals.shape[1]*4*12),65535)
        # else:
        #     b = 10000
        for i in range(0,data.shape[0],b):
            if self.packed:
                a = vcount(self.bind(data[i:i+b]),self.D)
            else:
                a = self.bind(data[i:i+b]).sum(1,dtype=torch.int16)
                # a = self.encoded_features[self.r,data[i:i+b]].view(-1,171,2,self.HD_dim).sum(1,dtype=torch.uint8).sum(1,dtype=torch.int16)

            if self.bindingMethod != 'FeatAppend':
                if (self.roundingType != 'noRounding'):
                    if self.HDvecType=='bin':
                        f =  int(math.floor(self.NumFeat / 2))
                    elif self.HDvecType=='binand':
                        f = int(math.floor(self.NumFeat / 4))
                    else:
                        f=0
                    torch.gt(a,f,out=outputVector[i:i+b])
                else:
                    #outputVector = (outputSplit[i] / self.NumFeat)
                    outputVector[i:i+b] = (a/ self.NumFeat)
            else:
                outputVector[i] = a.to(torch.int8)

        if self.HDvecType == 'bipol' and self.roundingType != 'noRounding':
            outputVector[outputVector==0] = -1

        if self.packed:
            pack(outputVector,packedOutput)
        # torch.cuda.synchronize()
        return outputVector, packedOutput


    def learn_HD_proj(self, data, labels, bindingMethod):
        ''' From features of one window to HDvector representing that data window
        different options possible:'FeatxVal', 'ChxFeatxVal', 'FeatxChxVal', 'ChFeatCombxVal', 'FeatAppend'
        '''
        (numTimePoints, numFeat)=data.shape
        lenFirstPointsToRemove=int(self.timeStepsWinShift[-1])
        if (numTimePoints>lenFirstPointsToRemove):
            outputVector = torch.zeros((numTimePoints-lenFirstPointsToRemove, self.HD_dim)).cuda(device=self.device).to(torch.int8)
            outputVectorPacked = torch.zeros((numTimePoints-lenFirstPointsToRemove,int(self.HD_dim/32))).cuda(device=self.device).to(torch.int32)
            # apppending for different time steps
            for tindx, time in enumerate(self.timeStepsWinShift):
                numWindows= int(time)
                if numWindows==0:
                    dataToEncode=data[lenFirstPointsToRemove:] #remove points for which we will not have diff data
                else:
                    if ('DiffAbs' in bindingMethod):
                        # dataR = torch.roll(data, -numWindows, dims=0)
                        # dataDiff = dataR - data
                        # # rearange so that first windows where we would not have previous data is 0
                        # dataToEncode = dataDiff[0:-numWindows, :]
                        # dataToEncode = dataToEncode[(lenFirstPointsToRemove - numWindows):, :]
                        dataR = torch.roll(data, numWindows, dims=0)
                        dataDiff =  data- dataR
                        dataToEncode = dataDiff[lenFirstPointsToRemove:]
                    else: #val
                        dataDiff=torch.roll(data, numWindows, dims=0)
                        dataToEncode = dataDiff[lenFirstPointsToRemove:]
                labelsOut=labels[lenFirstPointsToRemove:]
                (vecPart, vecPartPacked) = self.learn_HD_proj_base(dataToEncode.to(torch.int64))
                if ('App' in bindingMethod):
                    outputVector[:,tindx * self.D:(tindx + 1) * self.D] = vecPart
                    outputVectorPacked[:,tindx * int(self.D/32):(tindx + 1) * int(self.D/32)] = vecPartPacked
                elif ('Sum' in bindingMethod):
                    outputVector=outputVector+vecPart
                elif ('Perm' in bindingMethod):
                    numBitsShift=int(self.HD_dim /(len(self.timeStepsWinShift)+1))
                    outputVector = outputVector +  torch.roll(vecPart, -numBitsShift*tindx, dims=1) #dimension in D
                else:
                    print ('ERROR: no good binding method chosen!')
            #normalize dpending nnumTimes added
            if ('App' not in bindingMethod):
                f = int(math.floor(len(self.timeStepsWinShift) / 2))
                torch.gt(outputVector, f, out=outputVector)
                pack(outputVector, outputVectorPacked)
        else:
            outputVector = torch.zeros((0, self.HD_dim)).cuda(device=self.device).to(torch.int8)
            outputVectorPacked = torch.zeros((0, int(self.HD_dim / 32))).cuda(  device=self.device).to(torch.int32)
            labelsOut =torch.zeros((0, 1)).cuda(device=self.device).to(torch.int8)
        return outputVector, outputVectorPacked, labelsOut

    def givePrediction_perTimeSeg(self, data, labels0,bindingMethod, encoded=False):
        if not encoded:
            (temp, tempPacked, labels) = self.learn_HD_proj(data, labels0, bindingMethod)
            temp = temp if not self.packed else tempPacked
            data = temp.view(-1, 1, temp.shape[-1])
        else:
            data = data.view(-1, 1, data.shape[-1])
        # distances = self.dist_func(data, self.modelVectorsNorm.unsqueeze(0))
        stepsD=int(self.D/32)
        distancesPredClassPerTime=torch.zeros((data.shape[0],len(self.timeStepsInSec)))
        predClassPerTime=torch.zeros((data.shape[0],len(self.timeStepsInSec)))
        probabLabelsPerTime=torch.zeros((data.shape[0],len(self.timeStepsInSec)))
        for s in range(len(self.timeStepsInSec)):
            distances=self.dist_func(data[:,:,s*stepsD:(s+1)*stepsD], self.modelVectorsNorm[:,s*stepsD:(s+1)*stepsD].unsqueeze(0))
            predClassPerTime[:,  s]  = torch.argmin(distances, -1)
            distancesPredClassPerTime[:,  s] =torch.min(distances, dim=1).values
            # calculate probability
            distFromVotedClass = distances.min(1)[0]
            distFromNextClass = distances.kthvalue(2, dim=1)[0]
            probabLabelsPerTime[:, s] = distFromNextClass / (distFromNextClass + distFromVotedClass+ 0.0000001)

        return predClassPerTime, distancesPredClassPerTime, probabLabelsPerTime, labels


def onlineHD_ModelVecOnData_withRetrainPossible(data, labels0, model, HDType, bindingMethod, type='personalized'):
    #save original model values
    ModelVectorsOrig=torch.clone(model.modelVectors)
    ModelVectorsNormOrig = torch.clone(model.modelVectorsNorm)
    numAddedVecPerClassOrig=torch.clone(model.numAddedVecPerClass)


    (temp, packedTemp, labels) = model.learn_HD_proj(data, labels0, bindingMethod)
    tempVecBipol = temp.clone()
    tempVecBipol[tempVecBipol == 0] = -1
    temp = temp if not model.packed else packedTemp

    for s, trueLab in enumerate(labels.tolist()):
        (predLabel, distances, labels) = model.givePrediction_train(temp[s], labels0,encoded=True)

        # remove from wrong class
        predLab = predLabel.item()
        if (HDType == 'WrongOnly_AddAndSubtract' and predLab != trueLab):
            # add to correct class
            model.modelVectors[trueLab, :] += tempVecBipol[s] * model.lr * (distances[trueLab])
            model.numAddedVecPerClass[trueLab] += model.lr * (distances[trueLab])
            # subtract from wrong
            model.modelVectors[predLab, :] -= tempVecBipol[s] * model.lr * (1 - distances[predLab])
            model.numAddedVecPerClass[predLab] += -model.lr * (1 - distances[predLab])
        elif (HDType == 'Always_AddAndSubtract'):
            # add always to the correct class
            # ## Factors always 1 or 10
            # model.modelVectors[trueLab, :] += tempVecBipol[s] * model.lr * (distances[trueLab])
            # model.numAddedVecPerClass[trueLab] += model.lr * (distances[trueLab])
            # # if wrong subtract from wrong
            # if (predLab != trueLab):
            #     model.modelVectors[predLab, :] -= tempVecBipol[s] * model.lr * (1 - distances[predLab])  #* 0.5
            #     model.numAddedVecPerClass[predLab] += -model.lr * (1 - distances[predLab])
            model.lr=0.5
            if (trueLab == 1):
                model.modelVectors[trueLab, :] += tempVecBipol[s] *  (distances[trueLab])*model.onlineFNSfact #model.lr *
                model.numAddedVecPerClass[trueLab] += (distances[trueLab])*model.onlineFNSfact #model.lr *
            elif (trueLab == 0):
                model.modelVectors[trueLab, :] += tempVecBipol[s] * (distances[trueLab]) #* model.lr
                model.numAddedVecPerClass[trueLab] += (distances[trueLab])  #model.lr *
            if (predLab != trueLab) & (trueLab==1): #FN - should be seizure - reduce it a lot from NS:
                model.modelVectors[predLab, :] -= tempVecBipol[s] * model.lr * (1 - distances[predLab]) *model.onlineFNSfact #* 0.5
                model.numAddedVecPerClass[predLab] += -model.lr * (1 - distances[predLab]) *model.onlineFNSfact
            elif (predLab != trueLab) & (trueLab==0): #FS - should be NS - dont push so much, because we hvent added a lot to S
                model.modelVectors[predLab, :] -= tempVecBipol[s] * model.lr * (1 - distances[predLab]) *1 #* 0.5
                model.numAddedVecPerClass[predLab] += -model.lr * (1 - distances[predLab]) *1

        # NORM  VECTORS HAVE TO BE UPDATE AFTER EACH STEP BECAUSE WEIGHTS CHANGE AS MORE DATA IS ADDED !!!
        # calcualte again normalized vectors
        if (model.roundingType != 'noRounding'):  # rounding to 1 and 0
            if 'bin'in model.HDvecType:
                if model.packed:
                    pack(torch.gt(model.modelVectors, 0, ), model.modelVectorsNorm)
                else:
                    torch.gt(model.modelVectors, 0, out=model.modelVectorsNorm)
            else:
                model.modelVectorsNorm = torch.where(model.modelVectors > 0, 1, -1)
        else:  # no rounding having float values
            model.modelVectorsNorm = (model.modelVectors / model.numAddedVecPerClass.unsqueeze(1))

        #if only one was supposed to be retrained set it back to orig value
        if (type=='gen-retrain-seiz' or type=='generalized') :
            model.modelVectors[0, :]=ModelVectorsOrig[0, :]
            model.modelVectorsNorm[0, :] = ModelVectorsNormOrig[0, :]
            model.numAddedVecPerClass[0]= numAddedVecPerClassOrig[0]
        if (type=='gen-retrain-nonSeiz' or type=='generalized') :
            model.modelVectors[1, :]=ModelVectorsOrig[1, :]
            model.modelVectorsNorm[1, :] = ModelVectorsNormOrig[1, :]
            model.numAddedVecPerClass[1]= numAddedVecPerClassOrig[1]

    return # torch.cuda.synchronize()

def onlineHD_ModelVecOnData_batches_withRetrainPossible(data, trueLabels, model, HDType, batchSize=128, type='personalized'):
    # save original model values
    ModelVectorsOrig = torch.clone(model.modelVectors)
    ModelVectorsNormOrig = torch.clone(model.modelVectorsNorm)
    numAddedVecPerClassOrig = torch.clone(model.numAddedVecPerClass)

    # we need to use vectors with -1 and 1 instead of 0 and 1!!!
    (temp, packedTemp) = model.learn_HD_proj(data)
    tempVecBipol = temp.clone()
    tempVecBipol[tempVecBipol == 0] = -1
    temp = temp if not model.packed else packedTemp

    for i in range(0, len(trueLabels), batchSize):
        labels = trueLabels[i:i + batchSize]
        (predLabel, distances) = model.givePrediction_train(temp[i:i + batchSize], encoded=True)
        if (len(labels) == 1):  # Una added because erro if only one sample for torch.gather ???
            trueDistances = distances[labels]
            predDistances = distances[predLabel]
        else:
            trueDistances = torch.gather(distances, 1, labels.unsqueeze(1)).view(-1, 1)
            predDistances = torch.gather(distances, 1, predLabel.unsqueeze(1)).view(-1, 1)
        # trueDistances=trueDistances[:-1,:]
        # predDistances = predDistances[:-1, :]
        # find incorrect predictions
        incorrect = predLabel != labels
        # add to the correct class
        if (HDType == 'Always_AddAndSubtract'):
            l = labels
            distToUse = trueDistances
            distVector = tempVecBipol[i:i + batchSize] * distToUse
        elif (HDType == 'WrongOnly_AddAndSubtract'):
            l = labels[incorrect]
            distToUse = trueDistances[incorrect]
            distVector = tempVecBipol[i:i + batchSize][incorrect] * distToUse
        model.modelVectors.index_put_([l], distVector, accumulate=True)
        model.numAddedVecPerClass.index_put_([l], distToUse.squeeze(), accumulate=True)

        # remove from wrong class
        incorrectLabels = predLabel[incorrect]
        distToUse = ((1 - predDistances[incorrect]) * model.lr).view(-1, 1)
        distVector = -tempVecBipol[i:i + batchSize][incorrect] * distToUse * 0.5
        model.modelVectors.index_put_([incorrectLabels], distVector, accumulate=True)
        model.numAddedVecPerClass.index_put_([incorrectLabels], distToUse.squeeze(), accumulate=True)

        # NORM  VECTORS HAVE TO BE UPDATE AFTER EACH STEP BECAUSE WEIGHTS CHANGE AS MORE DATA IS ADDED !!!
        # calcualte again normalized vectors
        #             if (model.roundingType != 'noRounding'):  # rounding to 1 and 0
        #                 if model.HDvecType == 'bin':

        if model.packed:
            pack(torch.gt(model.modelVectors, 0), model.modelVectorsNorm)
        else:
            torch.gt(model.modelVectors, 0, out=model.modelVectorsNorm)

    # if only one was supposed to be retrained set it back to orig value
    if (type == 'gen-retrain-seiz'):
        model.modelVectors[0, :] = ModelVectorsOrig[0, :]
        model.modelVectorsNorm[0, :] = ModelVectorsNormOrig[0, :]
        model.numAddedVecPerClass[0] = numAddedVecPerClassOrig[0]
    elif (type == 'gen-retrain-nonSeiz'):
        model.modelVectors[1, :] = ModelVectorsOrig[1, :]
        model.modelVectorsNorm[1, :] = ModelVectorsNormOrig[1, :]
        model.numAddedVecPerClass[1] = numAddedVecPerClassOrig[1]


def unpackVector(vecArr, D, device):
    import torch
    import math
    from hdtorch import pack, unpack
    cuda0 = torch.device(device)
    if vecArr.ndim == 1:
        vecArr = np.expand_dims(vecArr, 0)
    if vecArr.ndim == 2:
        vecArr = np.expand_dims(vecArr, 0) # <<-- If the input array doesn't have a batch axis, add it
    vecTor = torch.from_numpy(vecArr).to(cuda0, dtype = torch.int32)
    unpackedOutput = torch.full((vecTor.shape[0], vecTor.shape[1], D),-1,dtype = torch.int8, device = device) # <<-- The output array should have the same first 2 axes, but the 3rd will be expanded
    unpack(vecTor, unpackedOutput)
    unpackedArr=unpackedOutput.cpu().numpy().squeeze()
    return (unpackedArr)


def packVector(vecArr, D, device):
    import torch
    import math
    from hdtorch import pack, unpack
    cuda0 = torch.device(device)
    vecTor = torch.from_numpy(vecArr.reshape((vecArr.shape[0],-1))).to(cuda0, dtype = torch.int8) # <<-- Should be int8
    packedOutput = torch.full((vecTor.shape[0],math.ceil(D/32)),-1,dtype = torch.int32, device = device)
    pack(vecTor, packedOutput)
    packedArr=packedOutput.cpu().numpy().squeeze()
    checkArr = unpackVector(packedArr, D, 'cuda')
    assert((checkArr == vecArr).all()) # <<-- Make sure it works
    return (packedOutput)



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

def rotateVec(mat,shifts):
    n_rows, n_cols = mat.shape
    arange1 = torch.arange(n_cols,device=mat.device).view(( 1,n_cols)).repeat((n_rows,1))
    arange2 = (arange1.unsqueeze(0) - shifts.unsqueeze(2)) % n_cols
    return torch.gather(mat.unsqueeze(0).expand(shifts.shape[0],-1,-1),2,arange2)

def func_generateVectorsMemory_Random(numClasses, HD_dim):
    ''' function to generate matrix of HD vectors using random method
        random - each vector is independantly randomly generated
    '''
    return torch.randint(0, 2, (numClasses, HD_dim))


def func_generateVectorsMemory_Sandwich(numClasses, HD_dim):
    ''' function to generate matrix of HD vectors using sandwich method
    sandwich - every two neighbouring vectors have half of the vector the same, but the rest of the vector is random
    '''

    vect_matrix = torch.zeros(numClasses, HD_dim)
    for i in range(numClasses):
        if i % 2 == 0:
            vect_matrix[i, :] = torch.randint(0, 2, HD_dim)
    for i in range(numClasses - 1):
        if i % 2 == 1:
            vect_matrix[i, 0:int(HD_dim / 2)] = vect_matrix[i - 1, 0:int(HD_dim / 2)]
            vect_matrix[i, int(HD_dim / 2):HD_dim] = vect_matrix[i + 1, int(HD_dim / 2):HD_dim]
    vect_matrix[numClasses - 1, 0:int(HD_dim / 2)] = vect_matrix[numClasses - 2, 0:int(HD_dim / 2)]
    vect_matrix[numClasses - 1, int(HD_dim / 2):HD_dim] = torch.randn(int(HD_dim / 2))
    vect_matrix = vect_matrix > 0
    return vect_matrix


def func_generateVectorsMemory_ScaleRand(numClasses, HD_dim, scaleFact):
    ''' function to generate matrix of HD vectors using scale method with bits of randomization
    scaleRand - every next vector is created by randomly flipping D/(numVec*scaleFact) elements - this way the further values vectors represent are, the less similar are vectors
    '''

    numValToFlip = math.floor(HD_dim / (scaleFact * numClasses))

    vect_matrix = torch.zeros(numClasses, HD_dim)
    # generate first one as random
    vect_matrix[0, :] = torch.randint(0, 2, (1, HD_dim)).to(torch.int8)
    # iteratively the further they are flip more bits
    for i in range(1, numClasses):
        vect_matrix[i, :] = vect_matrix[i - 1, :]
        # choose random positions to flip
        posToFlip = random.sample(range(1, HD_dim), numValToFlip)
        vect_matrix[i, posToFlip] = 1 - vect_matrix[i, posToFlip]
    return vect_matrix


def func_generateVectorsMemory_ScaleWithRadius(numClasses, HD_dim, radius):
    ''' function to generate matrix of HD vectors using scale method with bits of randomization
    radius is after which distance vectros should not have any specific corrlation
    e.g. ir radious is 10, then every 10th vector is randomly generated (called axes)
    and the ones in between are interpolated from two neighbouring axes
    '''
    numAxes = math.floor(numClasses / radius) + 1
    numValToFlip = math.floor(HD_dim / radius)  # between two axes
    axesVecs = torch.randint(0, 2, (numAxes, HD_dim)).to(torch.int8)
    vect_matrix = torch.zeros(numClasses, HD_dim)
    for i in range(0, numClasses):
        modi = i % radius
        if (modi == 0):
            vect_matrix[i, :] = axesVecs[int(i / radius), :]
        else:
            vect_matrix[i, 0:HD_dim - modi * numValToFlip] = axesVecs[int(i / radius),
                                                             0:HD_dim - modi * numValToFlip]  # first part is from prvious axes vec
            vect_matrix[i, HD_dim - modi * numValToFlip:] = axesVecs[int(i / radius) + 1,
                                                            HD_dim - modi * numValToFlip:]  # second part is from next axes vec
    return vect_matrix


def func_generateVectorsMemory_ScaleNoRand(numClasses, HD_dim, scaleFact):
    ''' function to generate matrix of HD vectors  using scale method with no randomization
    scaleNoRand - same idea as scaleRand, just  d=D/(numVec*scaleFact) bits are taken in order (e.g. from i-th to i+d bit) and not randomly
    '''
    numValToFlip = math.floor(HD_dim / (scaleFact * numClasses))

    # initialize vectors
    vect_matrix = torch.randint(0, 2, (numClasses, HD_dim)).to(torch.int8)

    # iteratively the further they are flip more bits
    for i in range(1, numClasses):
        vect_matrix[i] = vect_matrix[0]
        vect_matrix[i, 0: i * numValToFlip] = 1 - vect_matrix[0, 0: i * numValToFlip]

    return vect_matrix


def compareSimilarityOfGeneratedVectors(hdVec):
    vecs = hdVec.cpu().detach().numpy()
    numVec = len(vecs[:, 0])
    simMat = np.zeros((numVec, numVec))
    for i in range(numVec):
        for j in range(numVec):
            simMat[i, j] = np.mean(abs(vecs[i, :] - vecs[j, :]))
    print('AVRG VEC SIM :', 100 * np.mean(np.mean(simMat)))


def generateHDVector(vecType, numVec, HDDim, packed, device, name):
    # vecPath = f'./saved_hd_vectors/{name}_{vecType}_{HDDim}x{numVec}.tensor'
    # existed=0
    # if os.path.exists(vecPath):
    #     hdVec = torch.load(vecPath, map_location=device).contiguous()
    #     existed=1
    if vecType == 'sandwich':
        hdVec = func_generateVectorsMemory_Sandwich(numVec, HDDim)
    elif vecType == 'random':
        hdVec = func_generateVectorsMemory_Random(numVec, HDDim)
    elif "scaleNoRand" in vecType:
        hdVec = func_generateVectorsMemory_ScaleNoRand(numVec, HDDim, int(vecType[11:]))
    elif "scaleRand" in vecType:
        hdVec = func_generateVectorsMemory_ScaleRand(numVec, HDDim, int(vecType[9:]))
    elif "scaleWithRadius" in vecType:
        hdVec = func_generateVectorsMemory_ScaleWithRadius(numVec, HDDim, int(vecType[15:]))
    else:
        raise TypeError(f'Unrecognized vecType {vecType}')

    # if existed==0:
    #     if not os.path.exists(vecPath): os.makedirs(f'./saved_hd_vectors/', exist_ok=True)
    #     torch.save(hdVec,vecPath)

    compareSimilarityOfGeneratedVectors(hdVec)

    if not packed:
        return hdVec.to(device).to(torch.int8)
    else:
        hdVecPacked = torch.zeros((numVec, math.ceil(HDDim / 32)), dtype=torch.int32, device=device)
        pack(hdVec.to(device).to(torch.int8), hdVecPacked)
        return hdVecPacked


def analyseOptimalConfidnce(model, data, label0, bindingMethod):

    # run testing
    (predLabels, distances, probabLabels, label) = model.givePrediction(data, label0, bindingMethod)
    # analyse confidences
    numPerThr=calc_avrgProbab(predLabels, probabLabels, label)
    # #check optiomal bayes
    # (yPred_SmoothBayes, yPred_SmoothBayes) = smoothenLabels_Bayes(predLabels, probabLabels, seizureStableLenToTestIndx, probThresh,  distanceBetweenSeizuresIndx, label)

    return numPerThr


def measurePerformance( model, PostprocessingParams, FeaturesParams,   data, label0, bindingMethod):
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFPBefSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFPAftSeiz / FeaturesParams.winStep)

    # run testing
    (predLabels, distances, probabLabels, label) = model.givePrediction(data, label0, bindingMethod)

     # perform smoothing
    (performance0, yPred_MovAvrgStep1, yPred_MovAvrgStep2,
     yPred_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels, label, probabLabels,
                                                                    toleranceFP_bef, toleranceFP_aft, numLabelsPerHour,
                                                                    seizureStableLenToTestIndx,
                                                                    seizureStablePercToTest,
                                                                    distanceBetweenSeizuresIndx,
                                                                    PostprocessingParams.bayesProbThresh)

    return (performance0[2]) #F1 for episodes for no smooth


def testAndSavePredictionsForHD(folder, fileName, model, PostprocessingParams, FeaturesParams, HDParams,
                                    data, label0, HDtype, TrainTestType, thr):
    # various postpocessing parameters
    seizureStableLenToTestIndx = int(PostprocessingParams.seizureStableLenToTest / FeaturesParams.winStep)
    seizureStablePercToTest = PostprocessingParams.seizureStablePercToTest
    distanceBetweenSeizuresIndx = int(PostprocessingParams.distanceBetween2Seizures / FeaturesParams.winStep)
    numLabelsPerHour = 60 * 60 / FeaturesParams.winStep
    toleranceFP_bef = int(PostprocessingParams.toleranceFPBefSeiz / FeaturesParams.winStep)
    toleranceFP_aft = int(PostprocessingParams.toleranceFPAftSeiz / FeaturesParams.winStep)

    #run testing
    (predLabels, distances, probabLabels, label) = model.givePrediction(data, label0, HDParams.bindingMethod,False,  HDParams.thrTS, HDParams.thrTNS)

    # #analyse dist when FP
    # if (torch.sum(label)>0):  #at lest some seizures
    #     analyseDistWhenFP(predLabels, distances, label,folder, fileName)
    #
    # #if we want to use thr
    # # thr=0.5
    # predLabels = torch.zeros((len(label)))
    # seizDistThr = (1 - thr) * distances[:, 0]
    # indxS = torch.where(distances[:, 1] < seizDistThr)[0]
    # predLabels[indxS] = 1
    # predLabels=predLabels.to(label.device).to(dtype=torch.int64)

    #Calculate accuracy
    predTrueCount = predLabels[predLabels == label].bincount(minlength=2).cpu()
    acc = predTrueCount.sum().item() / len(label)
    print(f'{HDtype} {TrainTestType}: {acc}')

    # #calculate probability
    # distFromVotedClass = distances.min(1)[0]
    # distFromNextClass= distances.kthvalue(2, dim=1)[0]
    # probabLabels= distFromNextClass /(distFromNextClass+distFromVotedClass+ 0.0000001)

    # save distances
    outputName = folder + '/' + fileName + '_' + HDtype + '_'+ TrainTestType+'DistancesSNS.csv'
    saveDataToFile(distances, outputName, 'gzip')

    # perform smoothing
    (performance0, yPred_MovAvrgStep1, yPred_MovAvrgStep2,yPred_SmoothBayes) = calculatePerformanceAfterVariousSmoothing(predLabels, label, probabLabels,
                                                                         toleranceFP_bef, toleranceFP_aft, numLabelsPerHour, seizureStableLenToTestIndx,
                                                                         seizureStablePercToTest, distanceBetweenSeizuresIndx, PostprocessingParams.bayesProbThresh)

    #save predictions in time
    dataToSave = torch.vstack((label, probabLabels.to(label.device), predLabels,yPred_MovAvrgStep1, yPred_MovAvrgStep2, yPred_SmoothBayes)).transpose(1,0)
    outputName = folder + '/' + fileName  + '_'+HDtype+'_'+ TrainTestType+'Predictions.csv'
    saveDataToFile(dataToSave, outputName, 'gzip')
    return(predLabels)

def calc_avrgProbab(predLabels, probabLabels, label):
    numBins=100
    histograms=np.zeros((4,numBins)) #TS, TNS, FS, FNS
    numSampl=np.zeros((4))
    indx=torch.where(((predLabels==label) & (label==1)))[0] #TS
    numSampl[0]=len(indx)
    if (len(indx) > 0):
        histograms[0,:]=calcHistogramValues(probabLabels[indx].cpu().detach().numpy(), numBins, 0, 1)[0]
    indx=torch.where(((predLabels==label) & (label==0)))[0] #TNS
    numSampl[1] = len(indx)
    if (len(indx) > 0):
        histograms[1, :] = calcHistogramValues(probabLabels[indx].cpu().detach().numpy(), numBins, 0, 1)[0]
    indx=torch.where(((predLabels!=label) & (label==0)))[0] #FS
    numSampl[2] = len(indx)
    if (len(indx) > 0):
        histograms[2, :] = calcHistogramValues(probabLabels[indx].cpu().detach().numpy(), numBins, 0, 1)[0]
    indx=torch.where(((predLabels!=label) & (label==1)))[0] #FNS
    numSampl[3] = len(indx)
    if (len(indx) > 0):
        histograms[3, :] = calcHistogramValues(probabLabels[indx].cpu().detach().numpy(), numBins, 0, 1)[0]

    #find min and max boxes where histograms are not 0
    thresholds=np.where( np.nansum(histograms,0)>0)[0]
    numPerThr=np.zeros((numBins, numBins, 4))
    perfTot=np.zeros((numBins, numBins))
    for thrTS in thresholds:
        # readjust TS
        numTS0 = np.sum(histograms[0, thrTS:]) * numSampl[0]
        numFNS0 =  np.sum(histograms[0, 0:thrTS]) * numSampl[0]
        # readjust FS
        numTNS0 =  np.sum(histograms[2, :thrTS]) * numSampl[2]
        numFS0 = np.sum(histograms[2, thrTS:]) * numSampl[2]

        # # readjust FNS
        # numTS = numTS +np.sum(histograms[3,  thrTS:]) * numSampl[3]
        # numFNS =numFNS + np.sum(histograms[3, 0:thrTS]) * numSampl[3]
        for thrTNS in thresholds:
            # readjust TNS
            numTNS = numTNS0+ np.sum(histograms[1,thrTNS:]) * numSampl[1]
            numFS = numFS0+  np.sum(histograms[1,:thrTNS ]) * numSampl[1]
            # readjust FNS
            numTS = numTS0 +np.sum(histograms[3,  0:thrTNS]) * numSampl[3]
            numFNS =numFNS0 + np.sum(histograms[3, thrTNS:]) * numSampl[3]

            # #readjust FS
            # numTNS= numTNS +np.sum(histograms[2,thrTNS:])*numSampl[2]
            # numFS =numFS + np.sum( histograms[2,0:thrTNS])*numSampl[2]

            sens=numTS /(numTS+ numFNS)
            prec=numTNS /(numTNS+ numFS)
            balAcc= (sens+ prec) /2
            perfTot[thrTS, thrTNS]=balAcc
            numPerThr[thrTS, thrTNS,:]=np.hstack( (numTS, numTNS, numFS, numFNS))

    return numPerThr


def measureSeparabilityOfModelVecs(model, HDParams):

    vector_zero_norm = unpackVector(model.modelVectorsNorm[0,:].cpu().detach().numpy(), HDParams.HD_dim, HDParams.device)  # np.squeeze(data[1:,0])
    vector_one_norm = unpackVector(model.modelVectorsNorm[1,:].cpu().detach().numpy(), HDParams.HD_dim, HDParams.device)  # np.squeeze(data[1:,1])

    Mdiff=np.mean(abs(vector_one_norm-vector_zero_norm))*100
    print('Model diff: ' ,Mdiff)

    #measure per timediff
    if ('Time' in HDParams.bindingMethod) and ('App' in HDParams.bindingMethod):
        numParts=len(HDParams.timeStepsInSec)
        Dstep=int(len(vector_zero_norm) /numParts)
        Mdiff=np.zeros(numParts)
        for p in range(numParts):
            Mdiff[p] = np.mean(abs(vector_one_norm[p*Dstep: (p+1)*Dstep] - vector_zero_norm[p*Dstep: (p+1)*Dstep])) * 100
        print('Model diff per segments ', str(HDParams.timeStepsInSec), 'is : ', Mdiff)

