#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:04:10 2021

@author: shalu
"""

import numpy as np
import cv2
import math
from scipy import ndimage
import matplotlib.pyplot as plt

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

img_before =  cv2.imread("/home/shalu/catkin_ws/src/character_recognition/images/tilted.jpg")
plt_imshow("Original", img_before)    


img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)

plt_imshow("Edged",img_edges)

lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
#print(lines)
angles = []

for [[x1, y1, x2, y2]] in lines:
    cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)
    

plt_imshow("Detected lines", img_before)    

median_angle = np.median(angles)
img_rotated = ndimage.rotate(img_before, median_angle)
plt_imshow("Rotated {} degrees".format(round(median_angle,1)), img_rotated)
print("Angle is: ", median_angle)
##cv2.imwrite('rotated.jpg', img_rotated)  
