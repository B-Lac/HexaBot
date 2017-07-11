# HexaBot

This is my code for my CS231n project at Stanford for a convolutional neural network (CNN) that can play the game Super Hexagon. This includes:
- a Processing sketch that generates imitation screenshots of Super Hexagon
- the Python code that trains the CNN using Theano/Lasagne
- the Python code that plays Super Hexagon using the trained CNN.

The project is described in detail in this report and demonstrated in these videos playing levels 1 and 2:
- Report - http://cs231n.stanford.edu/reports/2016/pdfs/115_Report.pdf
- Level 1 - https://vimeo.com/158433051
- Level 2 - https://vimeo.com/158855274

I am releasing this code as is, more than a year after I last worked on it. It's not as cleaned up as I would like and I am making no guarantees that it is in a working state. 

## Super Hexagon Image Generator

Located in the "HexagonGenerator" directory is my Processing sketch that generates images that look like screenshots from Super Hexagon, with objects on screen labeled with bounding boxes. 

"HexagonGenerator.pde" is the main file. I gave myself a few inputs for debugging purposes:
- Left Mouse - toggle between real images (from "img" directory) and generated image
- Space - toggle on/off polar image mode
- b - toggle on/off generated image bounding boxes in polar mode
- r - randomize params for generated image
- s - save frame in "save" directory

To actually save a full dataset, you'll need to open "SaveImages.pde" and update line 1 to set saveImages to true (and maybe numImages to set your dataset size). This will generate one image and csv of bounding boxes per frame and save it to "generated/[timestamp]".

If the variable multiplayerMode=true, the generator will cram a lot of player triangles into the image instead of walls. When saving a dataset, this flag gets toggled after every image. This was to account for the disparity of walls to players in the training set.

## HexaNet Trainer and Player

The code to train the CNN and the player code are both in the "HexaNet" directory.

I don't expect anyone to run this, but if you do, you'll need a few Python libraries:
- Theano - a deep learning library for Python http://deeplearning.net/software/theano/
- Lasagne - a high level wrapper for Theano https://lasagne.readthedocs.io/en/latest/
- OpenCV (cv2) - general purpose computer vision library http://opencv.org/
- Python Imaging Library (PIL) - image library, used here for taking in-game screenshots
- win32gui, win32ui, win32con, win32api - low level Windows libraries, used to send inputs to Super Hexagon game window
- NumPy, SciPy - popular general purpose libraries for high dimensional and scientific calculations

"HexaNetRunnerTheano.py" I believe runs the training. However, I couldn't get it this to work again. I do know the trainer needs a dataset of images stored in the "dataset" directory which it generates an "X.npy" and "y.npy" used to store X (the image dataset matrix) and y (the output label matrix). HexaNetImageProcessor.processDataset will do this (uncomment line 75). The trainer periodically generates a ".npz" file to the "HexaNetTheanoParams" directory containing the network values.

I've put a sample dataset, a few network files, and an X.npy and y.npy file GeneratedFiles.zip