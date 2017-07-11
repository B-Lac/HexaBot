if __name__=="__main__":
    execfile("HexaNetRunnerTheano.py")
    #execfile("Hexabot.py")
import warnings
warnings.simplefilter("ignore")
import numpy as np
import cv2 as cv
from time import time, sleep
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter
from lasagne.layers import batch_norm
from collections import OrderedDict

"""
conv - relu - conv - reul - pool - conv - svm
Input:    256x256x3 image
Layer1:    CONV7-F1, pad 0, stride 3
Layer2:    CONV5-F2, pad 0, stride 1
Layer3: POOL2
Layer4: CONV1-C, pad 0, stride 1
"""
class HexaNetTheano(object):
    def __init__(self, input_dim=(3,256,256), F1=16, F2=16, num_classes=3, weight_scale=1, reg=1e-2, regL1 = 1, dropout=0):
        
        self.Wnorm = lambda :lasagne.init.HeNormal(weight_scale)
        self.xvar = T.tensor4("x")
        self.yvar = T.vector("y", dtype="int64")
        self.num_classes = num_classes
        regL1 -= reg
        self.reg, self.regL1, self.dropout = reg, regL1, dropout
        self.net = self.initNetwork2(input_dim, F1, F2, num_classes)
        #self.getNetworkCellRange()

        probs = lasagne.layers.get_output(self.net, deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(probs, self.yvar).mean()
        penalty = lasagne.regularization.regularize_network_params(self.net, lasagne.regularization.l2)*reg
        loss = loss + penalty + self.penaltyL1
        params = lasagne.layers.get_all_params(self.net, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=1e-5)# 5e-4, 1e-5
        trainFunc = theano.function([self.xvar,self.yvar], loss, updates=updates, allow_input_downcast=True)
        self.trainFunc = trainFunc
        self.timestamp = str(int(time()*1000))
        self.history = {name:[] for name in ["val", "batch", "loss", "player"]}
        self.outFunc = theano.function([self.xvar], lasagne.layers.get_output(self.net, deterministic=True), allow_input_downcast=True)

        self.smoothW1 = False
        self.bestParams = self.getParams()
        self.bestAcc = 0
        self.bestParamsIdx = 0
        #W = lasagne.layers.get_all_param_values(self.net)[0];print np.var(W), np.std(W);


    def initNetwork2(self, input_dim, F1, F2, num_classes):
        dropout = self.dropout
        C,H,W = input_dim
        batch_norm_dropout = lambda x:lasagne.layers.DropoutLayer(batch_norm(x), p=dropout)        
        net = batch_norm(lasagne.layers.InputLayer(shape=(None, C, H, W), input_var=self.xvar))
        self.penaltyL1 = lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)*0
        if True:
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F1, filter_size=7, stride=3, pad=0, W=self.Wnorm()))
            self.penaltyL1 = lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)*self.regL1
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F2, filter_size=5, stride=1, pad=0, W=self.Wnorm()))
        elif False:
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F1, filter_size=7, stride=3, pad=0, W=self.Wnorm()))
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F2, filter_size=3, stride=1, pad=0, W=self.Wnorm()))
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F2, filter_size=3, stride=1, pad=0, W=self.Wnorm()))
        elif False:
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F1, filter_size=3, stride=3, pad=1, W=self.Wnorm()))
            for _ in xrange(3):
                net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F2, filter_size=3, stride=1, pad=0, W=self.Wnorm()))
        else:
            #net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F1, filter_size=3, stride=1, pad=0, W=self.Wnorm()))
            #net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F2, filter_size=3, stride=1, pad=0, W=self.Wnorm()))
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F1, filter_size=5, stride=1, pad=0, W=self.Wnorm()))
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F2, filter_size=3, stride=3, pad=0, W=self.Wnorm()))
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F2, filter_size=3, stride=1, pad=0, W=self.Wnorm()))
            net = batch_norm_dropout(lasagne.layers.Conv2DLayer(net, num_filters=F2, filter_size=3, stride=1, pad=0, W=self.Wnorm()))

        net = lasagne.layers.MaxPool2DLayer(net, pool_size=2, stride=2, pad=0)
        net = lasagne.layers.Conv2DLayer(net, num_filters=num_classes, filter_size=1, stride=1, pad=0, nonlinearity=None, W=self.Wnorm())
        net = lasagne.layers.DimshuffleLayer(net, (0,2,3,1))
        net = lasagne.layers.ReshapeLayer(net, (-1, num_classes))
        net = lasagne.layers.NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)

        return net
    

    def initNetwork1(self, input_dim, F1, F2, num_classes):
        C,H,W = input_dim
        net = lasagne.layers.InputLayer(shape=(None, C, H, W), input_var=self.xvar)
        net = lasagne.layers.Conv2DLayer(net, num_filters=F1, filter_size=7, stride=3, pad=0, W=self.Wnorm())
        net = lasagne.layers.Conv2DLayer(net, num_filters=F2, filter_size=5, stride=1, pad=0, W=self.Wnorm())
        net = lasagne.layers.MaxPool2DLayer(net, pool_size=2, stride=2, pad=0)
        net = lasagne.layers.Conv2DLayer(net, num_filters=num_classes, filter_size=1, stride=1, pad=0, nonlinearity=None, W=self.Wnorm())
        net = lasagne.layers.DimshuffleLayer(net, (0,2,3,1))
        net = lasagne.layers.ReshapeLayer(net, (-1, num_classes))
        net = lasagne.layers.NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)
        return net

    # influence of input on single output cell
    def getNetworkCellRange(self):
        global batch_norm
        batch_norm = lambda x:x
        self.Wnorm = lambda :lasagne.init.HeNormal(1e6)
        X = np.zeros((1,3,256,256), dtype="float64")
        y = np.zeros((1600), dtype="int64")
        i,j = 15,15
        yvar = T.matrix()
        
        net = self.initNetwork2((3,256,256), 1, 1, 3)
        inputLayer = lasagne.layers.get_all_layers(net)[0]
        output, activation = lasagne.layers.get_output([net, inputLayer], deterministic=True)
        loss = lasagne.objectives.categorical_crossentropy(output, self.yvar).mean()
        grad = T.grad(loss, activation)
        gradFunc = theano.function([self.xvar, self.yvar], grad, updates=None, allow_input_downcast=True)
        
        g0 = np.abs(gradFunc(X,y))
        y[i*40+j] = 1
        g1 = np.abs(gradFunc(X,y))
        g = np.abs(g0-g1)
        #print g0.sum(), g1.sum(),g.sum(), (g>1e-2).sum()
        idxX, idxY = np.where(g>1e-2)[2:]
        xRange, yRange = max(idxX)-min(idxX),max(idxX)-min(idxX)
        print "X RANGE: ", max(idxX)-min(idxX)
        print "Y RANGE: ", max(idxX)-min(idxX)
        return xRange, yRange


    def visualize(self, x, window="", wait=False, saveFile=None, smoothWalls=False):
        im = x.swapaxes(0,2).copy()
        scores = self.loss(np.array([x,], dtype="float64")).reshape((40,40,3))
        #scores[15:,:,2]=0###
        classes = scores.argmax(axis=2)

        if smoothWalls:
            A,O = np.logical_and, np.logical_or
            XandYandZ = lambda x,y,z:A(x,A(y,z))
            XandYorZ = lambda x,y,z:A(x,O(y,z))
            walls = classes==1
            walls = np.pad(walls, 1, "constant")
            walls[:,1:-1] = XandYandZ(walls[:,1:-1], walls[:,:-2], walls[:,2:])
            walls[1:-1] = XandYorZ(walls[1:-1], walls[:-2], walls[2:])
            walls = A(np.logical_not(walls[1:-1,1:-1]), classes==1)
            classes[walls] = 0

        for i in xrange(40):
            for j in xrange(40):
                c = classes[i,j]
                if c!=0:
                    cv.rectangle(im, (i*6,j*6), (i*6+21,j*6+21), (255 if c==2 else 0,255 if c==1 else 0,0))
                # small indicator for player class
                #p = int(255*scores[i,j,2])
                #cv.rectangle(im, (i*6+10,j*6+10), (i*6+11,j*6+11), (p,p,p))


        # red around best player box
        if 0:
            bestP = scores[:,:,2].argmax()
            #print "\t",scores[:,:,2].max()
            i,j = bestP/40, bestP%40
            cv.rectangle(im, (i*6,j*6), (i*6+21,j*6+21), (0,0,255))

        if saveFile:
            cv.imwrite("output/"+saveFile, im)

        cv.imshow(window,im)
        cv.waitKey(0 if wait else 1)


    def train(self, X, y, Xval, yval, epochs=10, batch=10, printEvery=10, visualizeSample=None):
        numSamples = X.shape[0]
        Xval = Xval.astype("float64")
        yval = yval.flatten()
        yflat = y.flatten()
        itersPerBatch = numSamples/batch
        t = time()
        if visualizeSample!=None:
            windows = ["W1","W1diff",""]
            for w in range(len(windows)):
                cv.namedWindow(windows[w])
                cv.moveWindow(windows[w],50-2400,50+250*w)
            self.visualize(visualizeSample)
            self.visualizeW1(orig=True)

        for e in xrange(epochs+1):
            print "\nEPOCH %d/%d, iters=%d, t=%d"%(e, epochs, itersPerBatch, int(time()-t))
            if Xval!=None:
                batch_mask = np.random.choice(numSamples, batch)
                X_batch = X[batch_mask]
                y_batch = y[batch_mask]
                vacc, vaccPlayer = self.accuracy(Xval, yval)
                bacc, _ = self.accuracy(X_batch, y_batch.flatten())
                self.history["val"].append(vacc)
                self.history["batch"].append(bacc)
                self.history["player"].append(vaccPlayer)
                print "\t\tValidation Accuracy: %.4f"%(vacc)
                print "\t\tVal Player Accuracy: %.4f"%(vaccPlayer)
                print "\t\tBatch Accuracy:      %.4f"%(bacc)
                acc = vacc + vaccPlayer*.5
                if self.bestAcc < acc:
                    self.bestAcc = acc
                    self.bestParams = self.getParams()
                    self.bestParamsIdx = e
            if e==epochs:
                break
            self.bestAcc = 0# reset best accuracy for new epoch batch

            # smooth W1 convolutions
            if self.smoothW1 and e>0:
                sigma = .6#1.-1./e**2
                params = self.getParams()
                params[4] = gaussian_filter(params[4],(0,sigma*.8,sigma,sigma))
                lasagne.layers.set_all_param_values(self.net, params)

            t = time()
            for b in xrange(itersPerBatch):
                batch_mask = np.random.choice(numSamples, batch)
                X_batch = X[batch_mask]
                y_batch = y[batch_mask]
                loss = self.loss(X_batch, y_batch.flatten())
                self.history["loss"].append(loss)
                if b%printEvery==0:
                    print "%03d  "%b,loss
                    if visualizeSample!=None:
                        self.visualize(visualizeSample)
                        self.visualizeW1(orig=False)
            if visualizeSample!=None:
                self.visualizeW1(orig=True)

            if True and e%50==0:
                self.dumpHistory()
                self.saveParams(e)


    def accuracy(self, X, y):
        if X.dtype!="float64":
            X = X.astype("float64")
        scores = self.loss(X)
        classes = scores.argmax(axis=1)
        playerAcc = np.logical_and(classes==2,y==2).sum() / float((y==2).sum())
        return (classes==y).sum()/float(classes.shape[0]), playerAcc

    def confusion(self, X, y, doPrint=True):
        y = y.flatten()
        N = y.shape[0]
        C = self.num_classes
        scores = self.loss(X)
        classes = scores.argmax(axis=1)

        yBin = np.zeros((C, N))
        cBin = np.zeros((C, N))
        for i in range(C):
            yBin[i] = y==i
            cBin[i] = classes==i

        confusion = np.zeros((C, C), dtype="int")
        for i in range(C):
            for j in range(C):
                confusion[i,j] = np.logical_and(yBin[i], cBin[j]).sum()
                if doPrint:
                    print confusion[i,j],
            if doPrint:
                print

        return confusion


    def loss(self, X, y=None, skipScoreFlatten=False):
        if y==None:
            return self.outFunc(X)
        else:
            return self.trainFunc(X,y)


    def saveParams(self, currEpoch=None):
        for _ in range(2):
            filename = "HexaNetTheanoParams/%s"%(self.timestamp,)
            if currEpoch != None:
                filename += "-" + str(currEpoch)
            filename += ".npz"
            params = self.bestParams
            np.savez(filename, params)
            if currEpoch==None:
                break
            currEpoch=None
        print "Saved with timestamp: %s, params at epoch %s"%(self.timestamp, str(self.bestParamsIdx))

    def loadParams(self, timestamp, epoch=None):
        if epoch==None:
            params = np.load("HexaNetTheanoParams/%s.npz"%(timestamp,))
        else:
            params = np.load("HexaNetTheanoParams/%s-%s.npz"%(timestamp,epoch))
        params = params[params.files[0]]
        lasagne.layers.set_all_param_values(self.net, params)

    def dumpHistory(self):
        filePart = "HexaNetTheanoHistory/%s-"%(self.timestamp,)
        for name in self.history:
            filename = filePart+name+".txt"
            f = open(filename, "w")
            for value in self.history[name]:
                f.write("%f\n"%value)
            f.close()

    def getParams(self):
        return lasagne.layers.get_all_param_values(self.net)

    # prints framerate stats
    def fpsTest(self, x, tests=10, samplesPerTest=30):
        if len(x.shape)==3:
            x = np.array([x],dtype="float64")
        for _ in xrange(tests):
            t = time()
            for _ in xrange(samplesPerTest):
                self.loss(x)
            print samplesPerTest/(time()-t)

    # visualizes images that were previously difficult to classify
    def badImageTest(self):        
        for name in filter(lambda n:n.endswith("-p.png"),os.listdir("badImages")):
            img = cv.imread("badImages/"+name)
            cv.imshow("orig",img)
            img = cv.resize(img, (256,256)).swapaxes(0,2)
            self.visualize(np.array(img), wait=1)

    def visualizeW1(self, wait=False, orig=True, savefile=None):
        params = np.abs(self.getParams())
        W = params[4]
        F = W.shape[0]
        for i in range(F):
            W[i] -= W[i].min()
            W[i] /= W[i].max()

        l = 8# filters per line
        s = 80# size of filter
        b = 5# border margin
        W = W.swapaxes(1,3).swapaxes(1,2)
        X = np.ones(((s+b)*(F/l),(s+b)*l,3))
        for f in range(F):
            i = f%l
            j = f/l
            X[(s+b)*j:(s+b)*j+s,(s+b)*i:(s+b)*i+s] = cv.resize(W[f], (s,s), interpolation=cv.INTER_NEAREST)

        cv.imshow("W1",X)
        if savefile is not None:
            cv.imwrite(savefile, X*256)
        if orig:
            self.W1orig = X.copy()
            X = np.zeros(X.shape)
        else:
            X = np.abs(X-self.W1orig)
            X = (1-(1-X)**2)
        cv.imshow("W1diff",X)
        cv.waitKey(0 if wait else 1)

        
        if False:
            W = params[2]
            W = W.sum(axis=(0,2,3))
            W /= W.sum()/100.
            print W.astype("int"),
            for i in range(1,len(params),2):
                print np.std(params[i]),
            print
        

def plotHistory(timestamp, show=True, useLog=False):
    filePart = "HexaNetTheanoHistory/%s-"%(timestamp)
    i=1
    epsilon = 2e-2
    f = plt.figure(facecolor="white")
    f.canvas.set_window_title(str(timestamp))
    titleMap = {"val":"validation error",
        "batch":"batch error", 
        "player":"validation error (player class)",
        "loss":"training loss"}
    #names = ["val", "batch",  "player", "loss"]
    names = ["val", "player",  "loss"]
    for name in names:
        filename = filePart+name+".txt"
        f = open(filename)
        vals = map(lambda i:float(i[:-1]), f.readlines())
        plt.subplot(len(names),1,i)
        if name != "loss":
            vals = map(lambda v:1-v,vals)
            #plt.ylim([-epsilon,1+epsilon])
            if useLog:
                plt.yscale("log")
                plt.ylim([1e-2,1])
            else:
                plt.ylim([0,1])
            plt.grid()
        plt.xlim([-1, len(vals)])
        plt.plot(range(len(vals)), vals)
        plt.title(titleMap[name])
        i+=1
    plt.tight_layout()
    if show:
        plt.show()
