#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 07:42:55 2021

@author: shalu
"""

import pytesseract
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import math
from scipy import ndimage

#from skimage.filters import threshold_local
#print(np.__version__)

def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
    #plt.figure()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()
    
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = np.sum(pts,axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    return warped

#----------------------   Reading Image -------------------------------

configuration = "--psm 1"
image = cv2.imread('/home/shalu/catkin_ws/src/character_recognition/images/frame0000.jpg')

plt_imshow("Original Image: ", image)
#print(image)
#print(image.shape)
#
#print(image.dtype)

threshold = 80

# Converting into Binary(0 to 1)
image_B = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
image_B = cv2.GaussianBlur(image_B,(5,5), 0)
plt_imshow("Binary", image_B)

edged = auto_canny(image_B)
#cv2.imshow("edged", edged)
plt_imshow("AutoCanny", edged)

# ----------------- Contour----------------------------

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
    
#print(screenCnt)
contour_image = cv2.drawContours(image, [screenCnt], -1, (255, 0, 0), 4)
plt_imshow("Contour Identified", contour_image)
#print(screenCnt)
#-----------------   Crop  ----------------------------

#cropped_image = edged[]
#
#top_left = [min(screenCnt[2][0][0], screenCnt[3][0][0]), min(screenCnt[2][0][1], screenCnt[3][0][1])]
#top_right = [max(screenCnt[2][0][0], screenCnt[3][0][0]), max(screenCnt[2][0][1], screenCnt[3][0][1])]
#
#bot_right = [min(screenCnt[0][0][0], screenCnt[1][0][0]), max(screenCnt[0][0][1], screenCnt[1][0][1])]
#bot_left= [min(screenCnt[0][0][0], screenCnt[1][0][0]), min(screenCnt[0][0][1], screenCnt[1][0][1])]
##print("top_left: ", top_left)
#
##print("top_rit: ", top_right)
##print("bot left: ", bot_left)
##print("bot_rigt: ", bot_right)
#
#cropped_img = edged[bot_left[1] - 10: bot_right[1] + 10 , top_left[1] - 10:top_right[0] + 10]
##plt_imshow("Crop", cropped_img)
#
#
#plt_imshow("Cropped: ", cropped_img)


#------------  Perspective Transform -------------

points = [(screenCnt[0][0][0],screenCnt[0][0][1]), (screenCnt[1][0][0],screenCnt[1][0][1]), (screenCnt[2][0][0],screenCnt[2][0][1]), (screenCnt[3][0][0],screenCnt[3][0][1])]
#points= [(556,147), (567,25), (670,16), (661,138)]
#
warped_image = four_point_transform(edged, points)

plt_imshow("Warped Image", warped_image)


#----------   OCR   ----------

#
#text = pytesseract.image_to_string(cropped_img)
#print(text)
#


