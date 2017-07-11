import cv2 as cv
import numpy as np
#from scipy import stats

def computeImageClassGrid(idx, show=True, save=False):
	img = cv.imread("dataset/%05d.png"%idx)
	img = cv.resize(img, (256,256))
	if show:
		imgShow = img.copy()

	# read bounding boxes from file
	csv = open("dataset/%05d.csv"%idx, "r")
	numBoxW = int(csv.readline())
	numBoxP = int(csv.readline())
	getBoxes = lambda n: [map(lambda i:round(int(i)/3.), csv.readline()[:-1].split(",")) for _ in xrange(n)]
	boxesW = getBoxes(numBoxW)
	boxesP = getBoxes(numBoxP)
	csv.close()

	# assign classes to each pixel
	imageClass = np.zeros((256,256), dtype="int")
	for x0,y0,x1,y1 in boxesW:
		imageClass[x0:x1,y0:y1] = 1
	for x0,y0,x1,y1 in boxesP:
		imageClass[x0:x1,y0:y1] = 2

	# assign classes to each grid cell
	outClass = np.zeros((40,40), dtype="int")
	m = 9# margin
	for i in xrange(40):
		for j in xrange(40):
			#c = stats.mode(imageClass[i*6:i*6+21,j*6:j*6+21].flatten())[0]
			#c = stats.mode(imageClass[i*6+m:i*6+21-m,j*6+m:j*6+21-m].flatten())[0]
			c = imageClass[i*6+m:i*6+21-m,j*6+m:j*6+21-m].max()
			#c = imageClass[i*6+11, j*6+11]
			outClass[i,j] = c
			if show and c!=0:
				cv.rectangle(imgShow, (i*6,j*6), (i*6+21,j*6+21), (255 if c==2 else 0,255 if c==1 else 0,0))

	if show:
		cv.imshow("",imgShow)
		cv.waitKey(0)
		if save:
			cv.imwrite("output/%d.png"%idx, imgShow)

	return img, outClass.flatten()

# Processes each image in dataset and saves X,y
def processDataset(N=100):
	X = np.zeros((N,256,256,3), dtype="uint8")
	y = np.zeros((N,1600), dtype="uint8")
	for i in range(N):
		if i%10==0:
			print i,
		if i%100==0:
			print
		Xi, yi = computeImageClassGrid(i, False)
		X[i] = Xi
		y[i] = yi
	X = np.swapaxes(X,1,3)
	print "\n\nClass Percentages"
	for i in range(3):
		print "%d: %f"%(i, (y==i).sum()/(N*1600.))
	np.save("X.npy",X)
	np.save("y.npy",y)

# Loads X,y of dataset
def loadDataset():
	X = np.load("X.npy")
	y = np.load("y.npy")
	return X, y

#for i in range(20):
	#computeImageClassGrid(i, show=True, save=True)
processDataset(N=500)