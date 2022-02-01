#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 19:05:26 2021

@author: shalu
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy import ndimage
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
        
        self.pub_i = rospy.Publisher("/image_received", Image, queue_size = 2)
        self.pub_e = rospy.Publisher("/edge_detection", Image, queue_size = 1)
        self.pub_l = rospy.Publisher("/lined_image", Image, queue_size = 1)
        self.pub_r = rospy.Publisher("/rotated_image", Image, queue_size = 1)
        #rospy.Subscriber("/camera/left/image_raw", Image, self.image_callback)
        
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

    def image_callback(self, msg):
        
        #rospy.loginfo("Got the Image")
        self.image = self.bridge.imgmsg_to_cv2(msg)
        


    def start(self):
        
        while not rospy.is_shutdown():
            
            self.rate.sleep()
            rospy.loginfo("Image Published")
            if self.image is not None:
                self.pub_i.publish(bridge.cv2_to_imgmsg(self.image))
                edged, lined, rotated = Process_Image.detect_orientation(self.image)
                self.pub_e.publish(bridge.cv2_to_imgmsg(edged))
                self.pub_l.publish(bridge.cv2_to_imgmsg(lined))
                self.pub_r.publish(bridge.cv2_to_imgmsg(rotated))
            
            
        
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
        
        img_edges = cv2.Canny(img_gray, 100, 80, apertureSize=3)
        #img_edges = Process_Image.auto_canny(img_gray, sigma = 0.6)
        #Get_image.self.pub_e.publish(img_edges)
        crop_array_start = [0.3,0.1,0]
        crop_array_end = [0.6,0.9,1]
        
        size = img_edges.shape
        i = 0
        
        while True:
            r = [round(size[0]*crop_array_start[i]), round(size[0]*crop_array_end[i])]
            c = [round(size[1]*crop_array_start[i]), round(size[1]*crop_array_end[i])]
            cropped_image = img_edges[r[0]:r[1], c[0]:c[1]]

            lines = cv2.HoughLinesP(cropped_image, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)            
            #print(lines)
            angles = []
            if np.any(lines != None):
                for [[x1, y1, x2, y2]] in lines:
                    cv2.line(original_Image[r[0]:r[1], c[0]:c[1]], (x1, y1), (x2, y2), (255, 0, 0), 3)
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    angles.append(angle)
                break
            if i > 1:
                break
            i+=1
    
        #angles.sort()
        dummy_angles = [0]
        #plt_imshow("Detected lines", img_before)    
        #print(angles)        
        for i in angles:
            if abs(i) != 90 and abs(i) != 0:
                if abs(i) > 80 and abs(i) < 90: # if vertical line
                    i = 90 - abs(i)
                    dummy_angles.append(i)
                dummy_angles.append(i)
        #        for i in range(len(angles)):
        #            rospy.loginfo("Angles Detected: %f", angles[i])
        
                    
#        for i in range(len(dummy_angles)):
#            rospy.loginfo("after popping out zeros:  %d", dummy_angles[i])
                
        median_angle = abs(np.median(dummy_angles))
        #print(median_angle)
        #median_angle = 90 - median_angle
        # - ve angle gives clockwise rotation
        img_rotated = ndimage.rotate(original_Image, median_angle)

        #plt_imshow("Rotated {} degrees".format(round(median_angle,1)), img_rotated)
        rospy.loginfo("Angle is: %d", -median_angle)
        return cropped_image, original_Image, img_rotated
   


if __name__ == '__main__':
    
    rospy.init_node('image_listener', anonymous = True)
    
    my_image = Get_image()
    my_image.start()
    

    