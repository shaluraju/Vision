#! /usr/bin/python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg):
    #print("Received an image!")
	rospy.loginfo("Got the Image")


def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/camera/left/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
