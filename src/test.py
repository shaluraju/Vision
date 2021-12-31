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
        Process_Image.detect_orientation(self.image)


    def start(self):
        
        while not rospy.is_shutdown():
            
            rospy.loginfo("Image Published")
            if self.image is not None:
                self.pub.publish(bridge.cv2_to_imgmsg(self.image))
            self.rate.sleep()
            
        
        
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
        Process_Image.plt_imshow("GrayScale", img_gray)

   


if __name__ == '__main__':
    
    rospy.init_node('image_listener', anonymous = True)
    
    my_image = Get_image()
    my_image.start()
    

    