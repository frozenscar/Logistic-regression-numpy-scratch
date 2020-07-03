#!/usr/bin/env python
# coding: utf-8

# In[116]:


from matplotlib.image import imread
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys,os


dataSize=20

Y=np.ones((1,dataSize))
Y = np.c_[Y,np.zeros((1,dataSize))]

#print(Y1)

#Y = np.array([[1,1,1,1,0,0,0,0]])




path=os.getcwd()+"\\Data\\a1.jpg"
img = Image.open(path).convert("L")
imgF= np.asarray(img)

imgF =imgF.reshape(imgF.size)
imgF = imgF.reshape(1,100)
X = imgF






for i in range(2,dataSize+1):
    path=os.getcwd()+"\\Data\\a"+str(i)+".jpg"
    img = Image.open(path).convert("L")
    imgF= np.asarray(img)

    imgF =imgF.reshape(imgF.size)
    imgF = imgF.reshape(1,100)
    X = np.r_[X,imgF]
for i in range(1,dataSize+1):
    path=os.getcwd()+"\\Data\\b"+str(i)+".jpg"
    img = Image.open(path).convert("L")
    imgF= np.asarray(img)

    imgF =imgF.reshape(imgF.size)
    imgF = imgF.reshape(1,100)
    X = np.r_[X,imgF]


X = X/2550







# In[117]:

Y = np.transpose(Y)

#shuffle order

p = np.random.permutation(X.shape[0])
X = np.take(X,p,axis=0,out=X)
Y = np.take(Y,p,axis=0,out=Y)

# we now have shuffled training dataset









