# # Importing the necessary packages
from __future__ import print_function
from cv2 import cv2
import imutils
from imutils import paths
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser() #creating an ArgumentParser object
ap.add_argument("-i","--images", required=True, help="path to images directory")
args = vars(ap.parse_args()) #parse_args() inspects the command line and converts each argument into the appropriate type and invokes the required action 

# Initialize the HOG(Histogram of Oriented Gradients) descriptor/person detector
hog = cv2.HOGDescriptor() #creation of the HOG descriptor
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) # we set up the Support Vector Machine to be pre-trained for pedestrian detection

# Loop through the images in images directory
for imagePath in paths.list_images(args["images"]):
    # load the image and resize it to reduce detection time & improve accuracy
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy() 

    # detect persons in the image
    rects, weights = hog.detectMultiScale(image,winStride=(4,4),padding=(8,8),scale=1.05)
    # rects contains the bounding box co-ordinates of each image
    # weights contains the confidence value for each detection by the SVM

    # draw the original bounding boxes
    for (x,y,w,h) in rects:
        cv2.rectangle(orig, (x,y), (x+w,y+h), (0,0,255),2)

    # applying non-maxima suppression to the bounding boxes 
    # we make use of fairly large overlap threshold to maintain overlapping boxes
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects,probs=None,overlapThresh=0.65)

    # draw the final bounding box
    for (xA,yA,xB,yB) in pick:
        cv2.rectangle(image, (xA,yA), (xB,yB), (0,255,0), 2)
    
    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[Info] {}: {} original boxes, {} after suppression".format(filename,len(rects),len(pick)))

    # show the image
    cv2.imshow("Original Picture", orig)
    cv2.imshow("Picture containing boxes for each person", image)
    cv2.waitKey(0)


