#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 19:05:26 2021

@author: shalu
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Instantiate CvBridge
bridge = CvBridge()

class Get_image:
    
    
    def __init__(self):
        
        self.image = None
        self.bridge = CvBridge()
        self.rate = rospy.Rate(0.2)
        
        self.pub = rospy.Publisher("/image_received", Image, queue_size = 2)
        rospy.Subscriber("/camera/left/image_raw", Image, self.image_callback)

    def image_callback(self, msg):
        
        #rospy.loginfo("Got the Image")
        self.image = self.bridge.imgmsg_to_cv2(msg)
        


    def start(self):
        
        while not rospy.is_shutdown():
            
            self.rate.sleep()
            rospy.loginfo("Image Published")
            if self.image is not None:
                self.pub.publish(bridge.cv2_to_imgmsg(self.image))
                Process_Image.detect_orientation(self.image)
            
            
        
        
class Process_Image:
    
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
 
    def detect_orientation(image):
        
        rospy.loginfo("Image received for processing")
        original_Image = image 
        img_gray = cv2.cvtColor(original_Image, cv2.COLOR_BGR2GRAY)
        #Process_Image.plt_imshow("GrayScale", img_gray)
        img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        #img_edges = auto_canny(img_gray)
               
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
        #print(lines)
        angles = []
        
        for [[x1, y1, x2, y2]] in lines:
            #cv2.line(original_Image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
            
        angles.sort()
        dummy_angles = []
        #print(angles)
        
        for i in angles:
            if i != 0:
                dummy_angles.append(i)
        
            
        #print("after popping out zeros: ", dummy_angles)
        median_angle = abs(np.median(dummy_angles))
        #print(median_angle)
        median_angle = 90 - median_angle
        # - ve angle gives clockwise rotation
        #img_rotated = ndimage.rotate(img_before, -median_angle)
        #plt_imshow("Rotated {} degrees".format(round(median_angle,1)), img_rotated)
        rospy.loginfo("Angle is: %d", median_angle)
   


if __name__ == '__main__':
    
    rospy.init_node('image_listener', anonymous = True)
    
    my_image = Get_image()
    my_image.start()
    

    