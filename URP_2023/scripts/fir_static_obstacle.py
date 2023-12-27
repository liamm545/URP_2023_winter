#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from math import *
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, CompressedImage
from morai_msgs.msg import CtrlCmd

PIXEL_WIDTH = 20
PIXEL_HEIGHT = 20

# Don't change
MAX_DISTANCE=1200
OBSTACLE_RADIUS=4
ACT_RAD=300
# Arc angle for detection
MIN_ARC_ANGLE=60
MAX_ARC_ANGLE=120
ANGLE_INCREMENT=4
# DETECTION_RANGE
DETECT_RANGE=180

# Default speed (km/h)
DEFAULT_VEL = 10


class LiDARPlot:
    def __init__(self):
        rospy.init_node("parking_node")

        self.image = np.empty(shape=[0])
        self.bridge = CvBridge()
        self.warp_img_w = 600
        self.warp_img_h = 300

        # self.warp_img_w = 540
        # self.warp_img_h = 120

        self.nwindows = 10
        self.margin = 30
        self.minpix = 5
        self.lane_bin_th = 180

        self.lane_img = None

        Width, Height = 600, 300
        self.warp_src = np.array([
            [220, 210],
            [420, 210],
            [50,  260],
            [630, 260]
        ], dtype=np.float32)

        #marker
        
        self.warp_dist = np.array([
            [100, 0],
            [Width, 0], #0
            [75, Height],
            [Width, Height]
        ], dtype=np.float32)
        # rospy.Subscriber("/scan", LaserScan, self.callback)
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.img_callback)
        rospy.Subscriber("/scan", LaserScan, self.callback)
        self.msg_pub = rospy.Publisher("/ctrl_cmd", CtrlCmd, queue_size=1)
    
    def img_callback(self, data):
        image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        warp_img, _, _ = self.warp_image(image, self.warp_src, self.warp_dist, (self.warp_img_w, self.warp_img_h))
        self.lane_img = self.warp_process_image(warp_img)

    def warp_image(self, img, src, dst, size):
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
        # print("dsfjlkasdjflkdsaj")
        # cv2.imshow("bird_img", warp_img)
        # cv2.waitKey(1)

        return warp_img, M, Minv

    def color_filter(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        lower = np.array([100, 100, 100])
        upper = np.array([255, 255, 255])
        
        # lower = np.array([40, 185, 10])
        # upper = np.array([255, 255, 255])

        # yellow_lower = np.array([0, 85, 81])
        # yellow_upper = np.array([190, 255, 255])

        # yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(image, lower, upper)
        # mask = cv2.bitwise_or(yellow_mask, white_mask)
        masked = cv2.bitwise_and(image, image, mask = white_mask)
        
        return masked

    def warp_process_image(self, img):
        blurred_img = cv2.GaussianBlur(img, (0, 0), 2)
        w_f_img = self.color_filter(blurred_img)
        grayscale = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
        ret, lane = cv2.threshold(grayscale, 200, 255, cv2.THRESH_BINARY) #170, 255
        # cv2.imshow("hi",lane)
        # cv2.waitKey(1)
        canny_img = cv2.Canny(lane, 60, 80)
        # cv2.imshow("hi",canny_img)
        # cv2.waitKey(1)
        lines = cv2.HoughLines(canny_img, 1, np.pi/180, 80, None, 0, 0)
        hough_img = canny_img.copy()
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int (x0 + 1000*(-b))
                y1 = int ((y0) + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                
                slope = 90 - degrees(atan(b / a))
            
                if abs(slope) < 20:
                    cv2.line(hough_img, (x1, y1), (x2, y2), 0, 30)
                
                else:    
                    cv2.line(hough_img, (x1, y1), (x2, y2), 255, 8)
        # cv2.imshow('Hough', hough_img)
        # cv2.waitKey(1)
        
        return hough_img
    
    def callback(self, data):
        # Use front lidar data & make a local map
        # print("ppppppppppp")
        self.lidar_raw_data = np.array(data.ranges[1:361]) * 40
        self.main()
    
    def combine_lidar_and_image(self, lidar_data, image_data):
        combined_data = lidar_data + image_data
        return combined_data
    
    def main(self):
        # print("dddddddddddddddddd")
        # Filter 'inf' value
        self.lidar_raw_data[self.lidar_raw_data>=MAX_DISTANCE] = -1
        current_frame = np.zeros((ACT_RAD, ACT_RAD * 2, 3), dtype=np.uint8)
        measured_points = []
        available_angles = []
        lane_image = cv2.cvtColor(self.lane_img, cv2.COLOR_GRAY2BGR)

    
        for i in range(len(self.lidar_raw_data)):
            # Skip empty points
            if self.lidar_raw_data[i] < 0: continue
            # Calculate xy points
            # xy = [self.lidar_raw_data[i] * np.cos(np.deg2rad((i-180)/2)), self.lidar_raw_data[i] * np.sin(np.deg2rad((i-180)/2))]
            xy = [self.lidar_raw_data[i] * np.cos(np.deg2rad((i-180))), self.lidar_raw_data[i] * np.sin(np.deg2rad((i-180)))]
            measured_points.append(xy)

        # Mark points on the map
        for point in measured_points:
            cv2.circle(current_frame, np.int32([ACT_RAD - np.round(point[1]), ACT_RAD - np.round(point[0])]), OBSTACLE_RADIUS, (255, 255, 255), -1)

        # Draw a line to obstacles
        for theta in range(MIN_ARC_ANGLE-90, MAX_ARC_ANGLE-90, ANGLE_INCREMENT):
            # Find maximum length of line
            r = 1
            while r < DETECT_RANGE:
                if current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta))))][0] == 255 and\
                   current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta))))][1] == 255 and\
                   current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta))))][2] == 255: break
                r += 1

            if r != DETECT_RANGE:
                # draw a red line (detected)
                cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1),(int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))), (0, 0, 255), 1)
            else:
                # draw a gray line (not detected)
                cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1), (int(ACT_RAD-1 - np.round(DETECT_RANGE * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(DETECT_RANGE * np.cos(np.deg2rad(theta))))), (0, 255, 0), 1)
                available_angles.append(theta)
        
        # control angle
        if len(available_angles) == 0:
            middle_angle = 0
        else:
            middle_angle = np.median(np.array(available_angles), axis=0)
            print(middle_angle)
        cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1), (int(ACT_RAD-1 - np.round(DETECT_RANGE * np.sin(np.deg2rad(middle_angle)))), int(ACT_RAD-1 - np.round(DETECT_RANGE * np.cos(np.deg2rad(middle_angle))))), (255, 255, 255), 1)

        # cv2.imshow("result", current_frame)
        # cv2.waitKey(1)

        combined_data = self.combine_lidar_and_image(current_frame, lane_image)

        cv2.imshow("result", combined_data)
        cv2.waitKey(1)
        
        # # Make control message
        # m = CtrlCmd()

        # m.longlCmdType = 2  # velocity, steering
        # m.velocity = DEFAULT_VEL
        # m.steering = math.radians(middle_angle)
        # m.accel = 0         # fake
        # m.brake = 0         # fake
        # m.acceleration = 0  # fake

        # self.msg_pub.publish(m)
        

if __name__ == "__main__":
    p = LiDARPlot()
    rospy.spin()