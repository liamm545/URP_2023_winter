#!/usr/bin/env python3
import time
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from math import *
from sensor_msgs.msg import Imu
from morai_msgs.msg import CtrlCmd

# Don't change
MAX_DISTANCE=1200
OBSTACLE_RADIUS=3
ACT_RAD=400
# Arc angle for detection
MIN_ARC_ANGLE=70
MAX_ARC_ANGLE=110
ANGLE_INCREMENT=4
# DETECTION_RANGE
DETECT_RANGE=200

# Default speed (km/h)
DEFAULT_VEL = 1.25
MIN_ARC_ANGLE_LEFT=60
MAX_ARC_ANGLE_LEFT=110

MIN_ARC_ANGLE_RIGHT = 80
MAX_ARC_ANGLE_RIGHT = 130

Y_OFFSET = 0

class Obstacle():
    def __init__(self):
        rospy.init_node('obstacle', anonymous=True)
        rospy.Subscriber("/scan", LaserScan, self._lidar_callback)
        self.msg_pub = rospy.Publisher("/ctrl_cmd",CtrlCmd,queue_size=1)
        
        rospy.loginfo("Obstacle Node on. Waiting for topics...")
        self.obstacle_end = False

    # Functions
    def _lidar_callback(self, data=LaserScan):
        lidar_raw_data = np.array(data.ranges) * 30

        # Filter 'inf' value
        lidar_raw_data[lidar_raw_data>=MAX_DISTANCE] = -1
        current_frame = np.zeros((ACT_RAD, ACT_RAD * 2, 3), dtype=np.uint8)
        measured_points = []
        available_angles = []

        for i in range(len(lidar_raw_data)):
            # Skip empty points
            if lidar_raw_data[i] < 0: continue
            # Calculate xy points
            xy = [lidar_raw_data[i] * np.cos(np.deg2rad((i-180))), lidar_raw_data[i] * np.sin(np.deg2rad((i-180)))]
            measured_points.append(xy)

        # Mark points on the map
        for point in measured_points:
            print(point)
            center_point = tuple(np.int32([ACT_RAD - np.round(point[1]), ACT_RAD - np.round(point[0])]))
            cv2.circle(current_frame, center_point, OBSTACLE_RADIUS, (255, 255, 255), -1)
        cv2.imshow('Obstacle Detection', current_frame)
        cv2.waitKey(1)

        # # Draw a line to obstacles
        for theta in range(MIN_ARC_ANGLE-90, MAX_ARC_ANGLE-90, ANGLE_INCREMENT):
            # Find maximum length of line
            # print(current_frame)
            r = 1
            while r < DETECT_RANGE:
                if current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta))))][0] == 255 and\
                   current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta))))][1] == 255 and\
                   current_frame[int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))][int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta))))][2] == 255: break
                r += 1

            if r != DETECT_RANGE:
                # draw a red line (detected)
                
                cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1),(int(ACT_RAD-1 - np.round(r * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(r * np.cos(np.deg2rad(theta))))), (0, 0, 255), 1)
                
                cv2.waitKey(1)
            else:
                available_angles.append(theta)
                # draw a gray line (not detected)
                cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1), (int(ACT_RAD-1 - np.round(DETECT_RANGE * np.sin(np.deg2rad(theta)))), int(ACT_RAD-1 - np.round(DETECT_RANGE * np.cos(np.deg2rad(theta))))), (0, 255, 0), 1)
                cv2.waitKey(1)

        cv2.imshow('redline',current_frame)
        cv2.waitKey(1)
                
        self.msg_pub.publish(self.control(current_frame, available_angles))
        print("av__",available_angles)

    def control(self, current_frame, available_angles):
        if len(available_angles) == 0:
                middle_angle = self.prev_angle

        else:
            middle_angle = np.median(np.array(available_angles), axis=0)

        self.prev_angle = middle_angle
        print("m_________________-",middle_angle)
        cv2.line(current_frame, (ACT_RAD-1, ACT_RAD-1-Y_OFFSET), (int(ACT_RAD-1 - np.round(DETECT_RANGE * np.sin(np.deg2rad(middle_angle)))), int(ACT_RAD-1 - np.round(DETECT_RANGE * np.cos(np.deg2rad(middle_angle))))-Y_OFFSET), (255, 255, 255), 1)

        cv2.imshow("result", current_frame)
        cv2.waitKey(1)

        if abs(middle_angle) <= 10:
            K = 0.6
        else:
            K = 0.6
        
        # Make control message
        cmd = CtrlCmd()

        cmd.steering = radians(self.determ_angle(middle_angle * K)) # 현재 erp42_msg 는 CCW (왼쪽이 +) 그래서 뒤집음.
        cmd.velocity = 5
        if abs(cmd.steering) > 10:
            cmd.velocity -= 2                       # 각도 너무 클 경우, 속도 제어

        cmd.longlCmdType = 2
        cmd.accel = 0
        cmd.brake = 0
        cmd.acceleration = 0
        # self.msg_pub.publish(cmd)

        # if abs(degrees(drive_m.angular.z)) > 15:
        #     drive_m.linear.x = DEFAULT_VEL * 0.7 

        return cmd
    
    def determ_angle(self, before_ang):
        if before_ang >= 28:
            return 28

        elif before_ang <= -28:
            return -28

        else:
            return before_ang


if __name__ == "__main__":
    try:
        obstacle = Obstacle()
        print("____")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass