#--------------------------------------------------------------------------------
#   make dataset
#   read to BGR
#   class=29
#   data=train+val=[3475,512,1024,3] or [3475,256,512,3]
#   label=[3475,512,1024,29] or [3475,256,512,29]
#--------------------------------------------------------------------------------

import matplotlib.pyplot as mp
import cv2
import numpy as np
import glob
import h5py
from SemanticSegmentation_AICAR import labels

# print(len(labels.labels))
# for i in range(len(labels.labels)):
#     print(labels.labels[i].color)


def CompareLabel(color):
    # print(color)
    isIn=False
    for i in range(len(labels.labels)):
        if (color[2]==labels.labels[i].color[0])and(color[1]==labels.labels[i].color[1])and(color[0]==labels.labels[i].color[2]):
            return i
    print('error_in_compare',color)



FineValList=glob.glob('../../OriginData/AI_CAR/Cityscapes/gtFine/val/*/*color.png')
FineTrainList=glob.glob('../../OriginData/AI_CAR/Cityscapes/gtFine/train/*/*color.png')
# FineTestList=glob.glob('../../OriginData/AI_CAR/Cityscapes/gtFine_trainvaltest/gtFine/test/*/*color.png')
# CoarseValList=glob.glob('../../OriginData/AI_CAR/Cityscapes/gtCoarse/gtCoarse/val/*/*color.png')
# CoarseTrainList=glob.glob('../../OriginData/AI_CAR/Cityscapes/gtCoarse/gtCoarse/train/*/*color.png')
# CoarseTrainExtraList=glob.glob('../../OriginData/AI_CAR/Cityscapes/gtCoarse/gtCoarse/train_extra/*/*color.png')
# print(len(FineValList))
# print(len(FineTrainList))
# print(len(FineTestList))
# print(len(CoarseValList))
# print(len(CoarseTrainList))
# print(len(CoarseTrainExtraList))

FineTrainList=FineTrainList+FineValList
# CoarseTrainList=CoarseTrainList+CoarseTrainExtraList+CoarseValList
# print(len(FineTrainList))
# print(len(CoarseTrainList))
#

DataTrainList=glob.glob('../../OriginData/AI_CAR/Cityscapes/leftImg8bit/train/*/*.png')
DataValList=glob.glob('../../OriginData/AI_CAR/Cityscapes/leftImg8bit/val/*/*.png')
# DataTestList=glob.glob('../../OriginData/AI_CAR/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/*/*.png')
# print(len(DataTrainList))
# print(len(DataValList))
# print(len(DataTestList))
DataTrainList=DataTrainList+DataValList
DataTrainList=sorted(DataTrainList)
FineTrainList=sorted(FineTrainList)
print(DataTrainList[3001])
print(FineTrainList[3001])
# data=cv2.imread(DataTrainList[10])
# data=cv2.pyrDown(data,dstsize=(1024,512))
# dataflip=cv2.flip(data,1)
# cv2.imshow('1',data)
# cv2.imshow('2',dataflip)
# cv2.waitKey(0)

parcelnum=0
pdW=1024
pdH=512
DataTrain=[]
LabelTrain=[]
for i in range(len(DataTrainList)):
    print('iter',i)

    data=cv2.imread(DataTrainList[i])
    # cv2.imshow('d1',data)
    # cv2.waitKey(0)
    data =cv2.resize(data,dsize=(pdW,pdH),interpolation=cv2.INTER_NEAREST)
    data = np.concatenate((cv2.equalizeHist(data[:, :, 0])[:, :, np.newaxis],
                           cv2.equalizeHist(data[:, :, 1])[:, :, np.newaxis],
                           cv2.equalizeHist(data[:, :, 2])[:, :, np.newaxis]), axis=2)
    # cv2.imshow('d1',data)
    # cv2.waitKey(100)
    dataflip=cv2.flip(data,1)
    data=np.transpose(data,axes=(2,0,1))
    dataflip=np.transpose(dataflip,axes=(2,0,1))
    # print(data.size())
    DataTrain.append(data)
    DataTrain.append(dataflip)

    label = cv2.imread(FineTrainList[i])
    label = cv2.resize(label, dsize=(pdW, pdH),interpolation=cv2.INTER_NEAREST)
    labelProg=np.zeros((pdH,pdW)).astype(np.uint8)
    for m in range(pdH):
        for n in range(pdW):
            labelProg[m,n]=CompareLabel(label[m,n])
            # labelProgflip[m,n,CompareLabel(labelflip[m,n])]=1
    labelProgflip = cv2.flip(labelProg, 1)
    # print(labelProgflip.dtype)
    # print(labelProg[128,256])
    # print(labelProgflip[128,256])
    LabelTrain.append(labelProg)
    LabelTrain.append(labelProgflip)

    if ((i+1)%500==0) or ((i+1)==len(DataTrainList)):
        print('-----------------making data')
        DataParcel=np.array(DataTrain)
        DataTrain.clear()
        LabelParcel=np.array(LabelTrain)
        LabelTrain.clear()
        file = h5py.File('../dataset/Cityscapes/BGRHist1024/Cityscapes1024_Hist_0'+str(parcelnum)+'_torch.h5', 'w')
        file.create_dataset('Data', data=DataParcel)
        file.create_dataset('Label', data=LabelParcel)
        parcelnum+=1
        file.close()
        print('data ok------------------------')

