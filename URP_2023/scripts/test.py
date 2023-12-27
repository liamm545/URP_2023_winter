#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from math import *
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, CompressedImage
from morai_msgs.msg import CtrlCmd

PIXEL_WIDTH = 40
PIXEL_HEIGHT = 20

SLOPE = 90

# Create an empty black image
empty_image = np.zeros((PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)

# Calculate endpoint coordinates for the line
x1, y1 = 0, int(SLOPE * 0)  # Assuming x1 is 0 for the starting point
x2, y2 = PIXEL_WIDTH - 1, int(SLOPE * (PIXEL_WIDTH - 1))

# Draw a white line on the black image
cv2.line(empty_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

# Display the image (optional, for visualization purposes)
cv2.imshow('Line Image', empty_image)
cv2.waitKey(0)
cv2.destroyAllWindows()