#logistic regression from scratch


#imports
import random
import numpy as np
import math
import imgProcessing as data



def sigmoid(x):
    return 1/(1+np.exp(-x))


#logistic regression for Binary classification
#it is named regression,weird.
#
#  
#X is inputs 
#Y_ is predicted output
#Y is ground truth

#initialise X
#X=np.array([[1,1,1],[0,0,0],[1,0,1]])

X = data.X[0:15]


#transpose X
X = np.transpose(X)
#add a row of ones for x0 or bias
X = np.r_[np.ones((1,X.shape[1])),X]

print(X.shape)

#initialise Y
#Y=np.array([[1,0,1]],dtype='float')
Y = data.Y[0:15]
Y = np.transpose(Y)


#random initialise weights
W = np.random.random([X.shape[0],1])

print(W.shape)

#learning rate
lr=0.2
#epochs
epochs = 1000


for i in range(epochs):
    #Compute Z = W'* X

    Z = np.matmul(np.transpose(W),X)

    #Compute Y_

    Y_ = sigmoid(Z)
    


    #loss 

    d1 = -((Y/Y_)-(1-Y)/(1-Y_))*(sigmoid(Z)*(1-sigmoid(Z)))
    dw = np.matmul(d1,np.transpose(X))
    dw = np.transpose(dw)

    #cost

    cost = -np.sum((Y*np.log(Y_))+((1-Y)*np.log((1-Y_))))
    print("epoch ", i, "cost",cost)

    #Gradients of W

    W = W - lr*dw
preX=np.transpose(data.X[15:19])

print(preX.shape)
preX = np.r_[np.ones((1,preX.shape[1])),preX]
preX = sigmoid(np.matmul(np.transpose(W),preX))
preX = preX.astype('float32')
print("predicted", preX,"\n correct",data.Y[15:19])

#print(sigmoid(np.matmul(np.transpose(W),preX)))

'''
Z = np.matmul(np.transpose(W),X)
Y_ = sigmoid(Z)
d1 = ((Y/Y_)+(1-Y)/(1-Y_))*(sigmoid(Z)*(1-sigmoid(Z)))
dw = np.matmul(d1,np.transpose(X))
dw = np.transpose(dw)

#cost

cost = -np.sum((Y*np.log(Y_))+((1-Y)*np.log((1-Y_))))
print("cost",cost)
'''



