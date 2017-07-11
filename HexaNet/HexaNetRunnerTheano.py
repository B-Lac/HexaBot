import HexaNetImageProcessor
from HexaNetTheano import *
import cv2 as cv
import numpy as np
from sys import exit

#plotHistory(1456867148253,0);plotHistory(1457075589158);plotHistory(1457156924524);plotHistory(1457158352073);plotHistory(1457424731216);plotHistory(1457469026342, useLog=False)
#plotHistory(1456867148253,useLog=False)
X,y = HexaNetImageProcessor.loadDataset()
for i in range(X.shape[0]):
	X[i] = 255-X[i]

n=500
nval=20
data = {}
Xtrain = X[nval:n]
ytrain = y[nval:n]
Xval = X[:nval]
yval= y[:nval]

net = HexaNetTheano(dropout=0.25, regL1=1.0)
net.loadParams(1457908704481)
#print net.confusion(Xval,yval)
#net.visualizeW1(wait=1)
#badImg = cv.imread("badImages/1456429573991-98-p.png")
#badImg = cv.resize(badImg, (256,256)).swapaxes(0,2)

#net.visualizeW1(wait=1, savefile="asdf.png")
#X=X[0:10];y=y[0:10];net.train(X,y,X,y, visualizeSample=X[0].copy(), printEvery=10, epochs=1000, batch=10)
net.train(Xtrain, ytrain, Xval, yval, visualizeSample=X[1].copy(), printEvery=10, epochs=1000, batch=10)
#net.train(Xtrain, ytrain, Xval, yval, visualizeSample=badImg, printEvery=10, epochs=1000, batch=10)
#net.train(Xtrain, ytrain, Xval, yval, printEvery=20, epochs=2000, batch=5)
#net.saveParams()
#cv.waitKey(0)
#plotHistory(1456416389945)

#net.badImageTest()

if True:
	vval,pval=0,0
	print net.accuracy(Xval, yval.flatten())
	for i in range(20):
		v,p= net.accuracy(np.array([Xval[i]]), yval[i].flatten())
		print v,p
		#vval+=v
		#pval+=p
		#net.visualize(X[i], wait=True)
		#net.visualize(X[i], wait=True, smoothWalls=True)
		#HexaNetImageProcessor.computeImageClassGrid(i, save=True)
		#net.visualize(X[i], wait=False, smoothWalls=False, saveFile="%d-pred-nosmooth.png"%i)

	print vval/20, pval/20
