#--------------------------------------------------------------------------------
#   Based on Efficient ConvNet for Real-time Semantic Segmentation
#   deconv->nnupsample+conv
#   input 2:1 BGR
#   output 2:1 pixel-softmax
#   2018.1.1   10.26
#   Memory Using:   5905MB
#   Speed:  1 batch of 12 per second
#   Value:  useful
#--------------------------------------------------------------------------------

from SemanticSegmentation_AICAR import labels
import cv2
import numpy as np
import tensorflow as tf
import h5py
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


def getFilePic(picname):
    Inputimg = cv2.imread('./ImageSource/' + picname )
    Inputimg=cv2.resize(Inputimg,dsize=(512,256))
    Labelimg = np.zeros([0,Inputimg.shape[0],Inputimg.shape[1],29],dtype=np.uint8)
    return Inputimg[np.newaxis,:,:,:],Labelimg

def getBatchPic(h5filename,startnum,endnum,getlabel=False):
    file = h5py.File(h5filename, 'r')
    Data = file['Data'][:]
    Dataimg=Data[startnum:endnum,:,:,:]
    Labelimg = np.zeros([Dataimg.shape[0], Dataimg.shape[1], Dataimg.shape[2], 29], dtype=np.uint8)
    if getlabel:
        Label = file['Label'][:]
        Labelimg = Label[startnum:endnum, :, :, :]
        file.close()
        return Dataimg, Labelimg
    file.close()
    return Dataimg,Labelimg


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def bn(inputs, is_training,is_conv_out=True,decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)

def Conv(Inputtensor,Wshape,Bshape,stride):
    W= weight_variable(Wshape)
    B= bias_variable(Bshape)
    return (tf.nn.conv2d(Inputtensor, W, strides=stride, padding='SAME')+B)

def ConvR(Inputtensor,Wshape,Bshape,stride):
    W= weight_variable(Wshape)
    B= bias_variable(Bshape)
    return tf.nn.relu(tf.nn.conv2d(Inputtensor, W, strides=stride, padding='SAME') + B)

def ConvBn(Inputtensor,Wshape,Bshape,stride,istrain):
    W= weight_variable(Wshape)
    B= bias_variable(Bshape)
    return bn(tf.nn.conv2d(Inputtensor, W, strides=stride, padding='SAME') + B, is_training=istrain)

def ConvBnR(Inputtensor,Wshape,Bshape,stride,istrain):
    W= weight_variable(Wshape)
    B= bias_variable(Bshape)
    return tf.nn.relu(bn(tf.nn.conv2d(Inputtensor, W, strides=stride, padding='SAME') + B, is_training=istrain))

def dilatedConv(Inputtensor,Wshape,Bshape,rate):
    W = weight_variable(Wshape)
    B = bias_variable(Bshape)
    return (tf.nn.atrous_conv2d(Inputtensor, W, rate,  padding='SAME')+ B)

def dilatedConvR(Inputtensor,Wshape,Bshape,rate):
    W = weight_variable(Wshape)
    B = bias_variable(Bshape)
    return tf.nn.relu(tf.nn.atrous_conv2d(Inputtensor, W, rate,  padding='SAME')+ B)

def dilatedConvBn(Inputtensor,Wshape,Bshape,rate,istrain):
    W = weight_variable(Wshape)
    B = bias_variable(Bshape)
    return bn(tf.nn.atrous_conv2d(Inputtensor, W, rate,  padding='SAME') + B, is_training=istrain)

def dilatedConvBnR(Inputtensor,Wshape,Bshape,rate,istrain):
    W = weight_variable(Wshape)
    B = bias_variable(Bshape)
    return tf.nn.relu(bn(tf.nn.atrous_conv2d(Inputtensor, W, rate,  padding='SAME') + B, is_training=istrain))

def maxPool(Inputtensor):
    return tf.nn.max_pool(Inputtensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def Downsample_block(Inputtensor,Inputchannel,Outputchannel):
    L1=maxPool(Inputtensor)
    L2=Conv(Inputtensor,[3,3,Inputchannel,Outputchannel-Inputchannel],[Outputchannel-Inputchannel],[1,2,2,1])
    return tf.nn.relu(tf.concat([L1,L2],axis=3))

def Res_block(Inputtensor,channel,dilatedRate,residualNum,istrain,dropoutKeepProb):
    for i in range(residualNum):
        L1=ConvR(Inputtensor,[3,1,channel,channel],[channel],stride=[1,1,1,1])
        L2=ConvBnR(L1,[1,3,channel,channel],[channel],stride=[1,1,1,1],istrain=istrain)
        L3=dilatedConvR(L2,[3,1,channel,channel],[channel],dilatedRate)
        L4=dilatedConvBn(L3,[1,3,channel,channel],[channel],dilatedRate,istrain=istrain)
        Inputtensor=tf.nn.relu(Inputtensor+tf.nn.dropout(L4,dropoutKeepProb))
    return Inputtensor

def Upsample_block(Inputtensor,Inputchannel,Outputchannel,newsize):
    L1=tf.image.resize_nearest_neighbor(Inputtensor, size=newsize)
    return ConvR(L1,[3,3,Inputchannel,Outputchannel],[Outputchannel],stride=[1,1,1,1])

def LossFunction(Inputtensor):
    return tf.nn.softmax(Inputtensor)

def GradientDescent(Inputtensor,Label,LearningRate):
    loss = tf.reduce_mean(-tf.reduce_sum(Label * tf.log(tf.clip_by_value(Inputtensor, 1e-10, 1.0)), reduction_indices=[1]))
    TrainStep = tf.train.AdamOptimizer(LearningRate).minimize(loss)
    return TrainStep,loss

DrawOriginPic=True
DrawLabelPic=True
BNtrain=False
Data=tf.placeholder("float", shape=[None,None,None ,3])
Label=tf.placeholder("float", shape=[None,None,None ,29])
keep_prob = tf.placeholder(tf.float32)

L1=Downsample_block(Inputtensor=Data,Inputchannel=3,Outputchannel=16)
L2=Downsample_block(Inputtensor=L1,Inputchannel=16,Outputchannel=64)
L2_1=Res_block(Inputtensor=L2,channel=64,dilatedRate=1,residualNum=5,istrain=BNtrain,dropoutKeepProb=keep_prob)
L3=Downsample_block(Inputtensor=L2_1,Inputchannel=64,Outputchannel=128)
L3_1=Res_block(Inputtensor=L3,channel=128,dilatedRate=2,residualNum=1,istrain=BNtrain,dropoutKeepProb=keep_prob)
L3_2=Res_block(Inputtensor=L3_1,channel=128,dilatedRate=4,residualNum=1,istrain=BNtrain,dropoutKeepProb=keep_prob)
L3_3=Res_block(Inputtensor=L3_2,channel=128,dilatedRate=6,residualNum=1,istrain=BNtrain,dropoutKeepProb=keep_prob)
L3_4=Res_block(Inputtensor=L3_3,channel=128,dilatedRate=8,residualNum=1,istrain=BNtrain,dropoutKeepProb=keep_prob)
L3_5=Res_block(Inputtensor=L3_4,channel=128,dilatedRate=2,residualNum=1,istrain=BNtrain,dropoutKeepProb=keep_prob)
L3_6=Res_block(Inputtensor=L3_5,channel=128,dilatedRate=4,residualNum=1,istrain=BNtrain,dropoutKeepProb=keep_prob)
L3_7=Res_block(Inputtensor=L3_6,channel=128,dilatedRate=6,residualNum=1,istrain=BNtrain,dropoutKeepProb=keep_prob)
L3_8=Res_block(Inputtensor=L3_7,channel=128,dilatedRate=8,residualNum=1,istrain=BNtrain,dropoutKeepProb=keep_prob)
L4=Upsample_block(Inputtensor=L3_8,Inputchannel=128,Outputchannel=64,newsize=[64,128])
L4_1=Res_block(Inputtensor=L4,channel=64,dilatedRate=1,residualNum=2,istrain=BNtrain,dropoutKeepProb=keep_prob)
L5=Upsample_block(Inputtensor=L4_1,Inputchannel=64,Outputchannel=16,newsize=[128,256])
L5_1=Res_block(Inputtensor=L5,channel=16,dilatedRate=1,residualNum=2,istrain=BNtrain,dropoutKeepProb=keep_prob)
L6=Upsample_block(Inputtensor=L5_1,Inputchannel=16,Outputchannel=29,newsize=[256,512])
Predict=LossFunction(L6)
TrainStep,loss=GradientDescent(Predict,Label=Label,LearningRate=0.0005)




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,'./Model/normal_1.ckpt')
# saver.restore(sess,'ModelSave\\normal_1.ckpt')

Data_batch,Label_batch=getBatchPic('../dataset/Cityscapes/BGR512/Cityscapes512_03.h5',800,810,getlabel=DrawLabelPic)
# Data_batch,Label_batch=getFilePic('img6.jpg')
PredictVector=sess.run(Predict,feed_dict={Data:Data_batch,Label:Label_batch,keep_prob:1.0})
PredictVectorArgmax=np.argmax(PredictVector,axis=3)
Label_batchArgmax=np.argmax(Label_batch,axis=3)
for picnum in range(PredictVector.shape[0]):
    PredictPic = np.zeros([PredictVector.shape[1], PredictVector.shape[2], 3], dtype=np.uint8)
    for i in range(PredictVector.shape[1]):
        for j in range(PredictVector.shape[2]):
            PredictPic[i,j,0]=labels.labels[PredictVectorArgmax[picnum,i,j]].color[2]
            PredictPic[i,j,1]=labels.labels[PredictVectorArgmax[picnum,i,j]].color[1]
            PredictPic[i,j,2]=labels.labels[PredictVectorArgmax[picnum,i,j]].color[0]
    cv2.imwrite('./Predict/' + str(picnum) + '_SemanticSeg.jpg',PredictPic)

    if DrawLabelPic:
        LabelPic = np.zeros([PredictVector.shape[1], PredictVector.shape[2], 3], dtype=np.uint8)
        for i in range(PredictVector.shape[1]):
            for j in range(PredictVector.shape[2]):
                LabelPic[i,j,0]=labels.labels[Label_batchArgmax[picnum,i,j]].color[2]
                LabelPic[i,j,1]=labels.labels[Label_batchArgmax[picnum,i,j]].color[1]
                LabelPic[i,j,2]=labels.labels[Label_batchArgmax[picnum,i,j]].color[0]
        cv2.imwrite('./Predict/' + str(picnum) + '_Label.jpg',LabelPic)

    if DrawOriginPic:
        cv2.imwrite('./Predict/' + str(picnum) + '_Origin.jpg',Data_batch[picnum])

    # cv2.imshow('Predict',PredictPic)
    # cv2.waitKey(0)



