#--------------------------------------------------------------------------------
#   Based on Efficient ConvNet for Real-time Semantic Segmentation
#   upsample using dconv
#   input 2:1 BGR
#   output 2:1 pixel-softmax
#   2018.1.1   11.21
#   Memory Using:   10105MB
#   Speed:  1 batch of 12 per second
#   Value:  useless
#--------------------------------------------------------------------------------

import cv2
import numpy as np
import tensorflow as tf
import h5py
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def nextParcels(parcelnum):
    file = h5py.File('E:\\cuiyuhao\\python\\dataset\\Cityscapes\\BGR512\\Cityscapes512_0'+str(parcelnum)+'.h5', 'r')
    Data =  file['Data'][:]
    Label=  file['Label'][:]
    file.close()
    return Data,Label

def getBatch(Data,Label,batchsize):
    Databatch=[]
    Labelbatch=[]
    for i in range(batchsize):
        index=np.random.randint(0,Data.shape[0])
        Labelbatch.append(Label[index,:,:,:])
        Databatch.append(Data[index,:,:,:])
    return np.array(Databatch),np.array(Labelbatch)

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

def Dconv_block(Inputtensor,Inputchannel,Outputchannel,output_shape):
    W = weight_variable([3, 3, Outputchannel, Inputchannel])
    B = bias_variable([Outputchannel])
    return tf.nn.relu(tf.nn.conv2d_transpose(Inputtensor, W, output_shape=output_shape,
                                             strides=[1, 2, 2, 1], padding='SAME') + B)

def LossFunction(Inputtensor):
    return tf.nn.softmax(Inputtensor)

def GradientDescent(Inputtensor,Label,LearningRate):
    loss = tf.reduce_mean(-tf.reduce_sum(Label * tf.log(tf.clip_by_value(Inputtensor, 1e-10, 1.0)), reduction_indices=[1]))
    TrainStep = tf.train.AdamOptimizer(LearningRate).minimize(loss)
    return TrainStep,loss

maxparcelnum=7
batchsize=12
BNtrain=True
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
L4=Dconv_block(Inputtensor=L3_8,Inputchannel=128,Outputchannel=64,output_shape=[batchsize,64,128,64])
L4_1=Res_block(Inputtensor=L4,channel=64,dilatedRate=1,residualNum=2,istrain=BNtrain,dropoutKeepProb=keep_prob)
L5=Dconv_block(Inputtensor=L4_1,Inputchannel=64,Outputchannel=16,output_shape=[batchsize,128,256,16])
L5_1=Res_block(Inputtensor=L5,channel=16,dilatedRate=1,residualNum=2,istrain=BNtrain,dropoutKeepProb=keep_prob)
L6=Dconv_block(Inputtensor=L5_1,Inputchannel=16,Outputchannel=29,output_shape=[batchsize,256,512,29])
Predict=LossFunction(L6)
TrainStep,loss=GradientDescent(Predict,Label=Label,LearningRate=0.0001)




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
# saver.restore(sess,'Model\\normal_2.ckpt')

parcelnum=0
Dataparcel,Labelparcel=nextParcels(parcelnum=parcelnum)
for iter in range(500000):
    print('\n',iter)
    Data_batch,Label_batch=getBatch(Dataparcel,Labelparcel,batchsize=batchsize)
    error,result=sess.run([loss,TrainStep],feed_dict={Data:Data_batch,Label:Label_batch,keep_prob:0.3})
    print(error)


    if (iter+1)%1000==0:
        path = saver.save(sess,'Model\\normal_2.ckpt')
        print(path)
        parcelnum+=1
        if parcelnum==maxparcelnum:
            parcelnum=0
        Dataparcel, Labelparcel = nextParcels(parcelnum=parcelnum)




