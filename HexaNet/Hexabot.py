import cv2 as cv
import numpy as np
from sys import exit
from HexabotUtils import *
from HexaNetTheano import *
import time

# Agent that plays Super Hexagon
class HexagonAgent:

    # create initial values
    def __init__(self, readImages=False, showImages=False, failDump=False, saveTrials=False):
        # if True, will keep restarting game until program stopped manually
        self.readImages = readImages
        self.showImages = showImages
        self.failDump = failDump
        self.saveTrials = saveTrials
        self.imgIdx = 0

        # image stuff
        self.box = (0,22,768,501)
        self.goodPixel = (17,26)
        self.badPixel = (384,239)
        self.goodColorThreshold = 64
        self.currSecond = int(time.time())

        # keys
        self.L = 0x25# LEFT
        self.R = 0x27# RIGHT
        self.E = 0x0D# ENTER

        # HexaNet convolutional neural network
        self.hexanet = HexaNetTheano()
        #self.hexanet.loadParams(1456867148253)# LV 1
        #self.hexanet.loadParams(1457077384966)# LV 1,4
        #self.hexanet.loadParams(1457158352073)#LV 2
        #self.hexanet.loadParams(1457156924524)# LV 2 (better?)
        #self.hexanet.loadParams(1457417280166)# LV 2 (Heanet v2)
        #self.hexanet.loadParams(1457424731216)# LV 3
        #self.hexanet.loadParams(1457469026342)# LV 3
        #self.hexanet.loadParams(1457472321901)# LV 3
        #self.hexanet.loadParams(1457908704481)# LV 5
        self.hexanet.loadParams(1457909354951)# LV 5 (better?)
        
        
        

        self.convCell = lambda i,j:((i*6,j*6), (i*6+21,j*6+21))
        self.classColors = [(255,255,255),(0,255,0),(255,0,0)]
        self.prevPixWeights = {label:np.ones((256,256)) for label in ["wall","player"]}
        self.pixWeightAlpha = .75
        self.pyPrev,self.pxPrev=10,5

        self.prevAction = "N"
        self.smoothWalls=True

        if self.failDump:
            self.numFailFrames = 15
            self.failFrames = np.zeros((self.numFailFrames,256,256,3), dtype="float")
            self.currFailFrameIdx = 0

        if self.failDump or self.saveTrials:
            self.lastGameOver = time.time()

    """
    AGENT RUN METHODS
    """
    # run the agent
    def run(self):
        if self.showImages:
            cv.namedWindow("")
            cv.moveWindow("",0,550)
            cv.namedWindow("cells")
            cv.moveWindow("cells",400,550)

        self.iter = 0
        self.t = time.time()
        while True:
            if self.readImages:
                self.nextImage()
            else:
                self.mainLoop()

    # shows output of next image in imgs folder
    def nextImage(self):
        self.imgIdx+=1
        name = "imgs/1455548776-%04d.png"%self.imgIdx
        self.im = cv.imread(name)
        self.im2 = self.convertImageToInput(self.im)
        self.processImage(np.swapaxes(self.im2,0,2))

        cv.imshow("orig",self.im)
        cv.imshow("input",self.im2)
        cv.waitKey(0)

    # computations to perform for the current frame
    def mainLoop(self):
        # grab frame and pass through net
        #t = time.time()
        #print 1/max((time.time()-t),1e-6)
        self.im = fastImageGrab()# fps 33
        self.im2 = self.convertImageToInput(self.im)# fps huge
        self.rotateImageToPlayer()# fps 100
        self.processImage(np.swapaxes(self.im2,0,2))# fps 100
        #self.printFPS()

    # analyzes image to determine action
    def processImage(self, x):
        if self.showImages:
            self.hexanet.visualize(x, smoothWalls=self.smoothWalls)

        cellScores = self.hexanet.loss(np.array([x], dtype="float64")).swapaxes(0,1).reshape(3,40,40)            
        cellScores[2,15:,:]=0###
        cellClasses = cellScores.argmax(axis=0)
        
        if self.smoothWalls:
            A,O = np.logical_and, np.logical_or
            XandYandZ = lambda x,y,z:A(x,A(y,z))
            XandYorZ = lambda x,y,z:A(x,O(y,z))
            walls = cellClasses==1
            walls = np.pad(walls, 1, "constant")
            walls[:,1:-1] = XandYorZ(walls[:,1:-1], walls[:,:-2], walls[:,2:])
            walls[1:-1] = XandYorZ(walls[1:-1], walls[:-2], walls[2:])
            walls = A(np.logical_not(walls[1:-1,1:-1]), cellClasses==1)
            cellClasses[walls] = 0

        
        action, space = self.computePlayerAction(cellClasses)
        print "\t",action
        
        if not self.readImages:
            self.executeAction(action, space)

            if self.im[0,0].sum() != 0:
                if not self.saveTrials:
                    hitKey(self.E)
                print "GAME OVER"

                # dumps last few frames to FailDump folder
                if self.failDump:
                    if (time.time()-self.lastGameOver > 3):
                        for i in range(self.numFailFrames):
                            j = (i+self.currFailFrameIdx)%self.numFailFrames
                            cv.imwrite("FailDump/%d.png"%i, self.failFrames[j]*256)
                        print "SAVED FAIL DUMP"
                        releaseKey(self.L)
                        releaseKey(self.R)
                        releaseKey(self.E)
                        exit()
                    self.lastGameOver = time.time()

                # saves game over screen to Trials folder
                if self.saveTrials and self.im[0,0].sum() != 255*3 and self.im[160:290,0].sum() == 0:
                    if (time.time()-self.lastGameOver > 3):
                        cv.imwrite("Trials/%d.png"%(int(time.time()*1000)), self.im)
                    self.lastGameOver = time.time()
                    hitKey(self.E)

        self.visualizeCells(action)

    # determines most likely player cell
    def computePlayerPos(self, playerCells, update=False):
        py = playerCells.sum(axis=0).argmax()
        px = playerCells[5:-5,py].argmax()+5
        if playerCells[px,py]==0:
            px = playerCells[:,py].argmax()

        if playerCells[px,py]==0:
            if update:
                py,px=self.pyPrev,self.pxPrev
            else:
                py,px=20,self.pxPrev

        if update:
            self.pyPrev,self.pxPrev = py,px

        return py, px

    # draws image of 40x40 cell grid
    def visualizeCells(self, action):
        if self.showImages or self.failDump:
            cellImg = np.zeros((40,40,3), dtype="float")
            cellImg[:,:,1][self.wallCells] = 1.
            cellImg[:,:,0][self.playerCells] = 1.
            if action=="L":
                cellImg[:5,:20,2] = 1.
            elif action=="R":
                cellImg[:5,20:,0:2] = 1.
            cellImg = cv.resize(cellImg.swapaxes(0,1), (256,256), interpolation=cv.INTER_NEAREST)
            if self.showImages:
                cv.imshow("cells", cellImg)
                cv.waitKey(1)
            if self.failDump:
                self.failFrames[self.currFailFrameIdx] = cellImg
                self.currFailFrameIdx = (self.currFailFrameIdx+1)%self.numFailFrames

    # remove wall cells that player cannot die to
    def removeSafeWalls(self, wallCells, x0):
        toRemove = np.zeros((40,40), dtype="bool")
        for y in range(1,39):
            if wallCells[x0,y]:
                x = x0
                while x<30 and wallCells[x,y] and (wallCells[x,y+1] or wallCells[x,y-1]):
                    x+=1
                if x<30:
                    toRemove[x0:x,y] = True
        wallCells[toRemove] = False
        return wallCells

    # Given cell grid, compute best action
    def computePlayerAction(self, cellClasses):
        playerCells = cellClasses==2
        wallCells = cellClasses==1

        # player position
        py, px = self.computePlayerPos(playerCells, update=False)

        # compute distance to a wall at each y location
        self.wallCells = wallCells.copy()
        self.playerCells = playerCells
        wallCells = self.removeSafeWalls(wallCells, px+2)
        wallCells[:px+3] = False
        wallDistances = wallCells.argmax(axis=0)        
        wallDistances[wallDistances==0] = 40
        maxDist = wallDistances.max()
        
        # helper functions for safety metrics
        my = 1
        mlr = 1
        safety = lambda pos,m: np.min(wallDistances[max(pos-m,0):min(pos+m+1,40)])
        isSafe = lambda pos,m: safety(pos,m) >= maxDist
        minSafety = lambda p: px + 3 + abs(py-p)
        actionN = ("N",0)

        # no action if safe to stay
        if isSafe(py,my):
            return actionN

        # compute closest safe space on either side
        pl, pr = py-1, py+1
        while pl>0 and not isSafe(pl,mlr):
            pl -= 1
            if pl>0 and safety(pl,mlr) <= minSafety(pl):
                pl += 1
                break
            if pl>0 and safety(pl+1,mlr)-safety(pl,mlr) > 10:
                pl += 1
                break
        while pr<39 and not isSafe(pr,mlr):
            pr += 1
            if pr<39 and safety(pr,mlr) <= minSafety(pr):
                pr -= 1
                break
            if pr<39 and safety(pr-1,mlr)-safety(pr,mlr) > 10:
                pr -= 1
                break

        # if left and right both dangerous, use softer metric on current position
        sl,sy,sr = safety(pl,mlr), safety(py,my), safety(pr,mlr)
        safeL, safeR = sl>sy, sr>sy
        if not safeL and not safeR and sy > 10+px+3:
            print "\t\tSAFE ENOUGH",sl,sy,sr,pl,py,pr
            return actionN

        # additional metrics for safety on left and right
        safeL, safeR = safeL and isSafe(pl,mlr), safeR and isSafe(pr,mlr)
        spaceL,spaceR = py-pl, pr-py
        actionL = ("L", spaceL)
        actionR = ("R", spaceR)

        # Left safe, right not safe
        if safeL and not safeR:
            print "\t\t",sl,sy,sr,pl,py,pr
            return actionL

        # Right safe, left not safe
        elif safeR and not safeL:
            print "\t\t",sl,sy,sr,pl,py,pr
            return actionR

        # Both safe, pick better one
        elif safeL and safeR:
            print "\t\t!",sl,sy,sr,pl,py,pr
            if self.prevAction!="N" and abs(spaceL-spaceR) < 5:
                return actionL if self.prevAction=="L" else actionR
            return actionL if spaceL < spaceR else actionR

        # Neither safe, pick safer
        print "\t\t!!!!",sl,sy,sr,pl,py,pr
        if pl==py and pr==py:
            return actionN
        if pl==py:
            return actionR
        if pr==py:
            return actionL
        if self.prevAction!="N" and abs(sl-sr) < 5:
                return actionL if self.prevAction=="L" else actionR
        return actionL if sl > sr else actionR

    # executes given action
    def executeAction(self, action, space=0):
        timePerSpace = 0.01
        self.prevAction = action
        # release unwanted buttons
        if action!="L":
            releaseKey(self.L)
        if action!="R":
            releaseKey(self.R)
        # press left if L
        if action=="L":
            if space<=2:
                print "\t\ttap *",space
                hitKey(self.L, space*timePerSpace)
            else:
                pressKey(self.L)
        # press right if R
        if action=="R":
            if space<=2:
                print "\t\ttap *",space
                hitKey(self.R, space*timePerSpace)
            else:
                pressKey(self.R)



    """
    MISC METHODS
    """
    # prints estimate framerate
    def printFPS(self):
        self.iter+=1
        try:
            print int(self.iter/(time.time()-self.t))
            if self.iter==30:
                self.t = time.time()
                self.iter=0
        except:
            pass
    
    # computes polar coordinate representation of screenshot
    def convertImageToInput(self, im, theta=0):
        x,y = im.shape[1]/2,im.shape[0]/2
        im = cv.linearPolar(im, (x,y), y, cv.WARP_FILL_OUTLIERS)
        im = cv.resize(im, (256,160))

        im2 = np.zeros((256,256,3), dtype="uint8")
        im2[:160] = im
        im2[160:] = im[:256-160]
        return im2

    # rotates polar image so that player is in center
    def rotateImageToPlayer(self):
        x = np.swapaxes(self.im2,0,2)
        cellScores = self.hexanet.loss(np.array([x], dtype="float64")).swapaxes(0,1).reshape(3,40,40)            
        cellScores[2,15:,:]=0###
        cellClasses = cellScores.argmax(axis=0)
        py,px = self.computePlayerPos(cellClasses==2, update=True)
        py = int(py*6.4)
        newImg2 = self.rotatePolarImage(self.im2, py)
        self.im2 = newImg2

    # rotates polar image to given center
    def rotatePolarImage(self, im, center):
        if center >= 160:
            center -= 160
        amount = center-128 
        imPart = im[:160]

        im2 = np.concatenate((imPart[amount:], imPart[:amount]))
        im3 = np.zeros((256,256,3), dtype="uint8")

        im3[:160] = im2
        im3[160:] = im2[:256-160]
        return im3

    # saves an image under a file name
    def save(self, img, name="E:/Desktop/img.png"):
        img.save(name, "PNG")

agent = HexagonAgent(failDump=False, saveTrials=False, showImages=False)
agent.run()