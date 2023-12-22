#!/usr/bin/env python3
import cv2
import time
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, LaserScan
import matplotlib.pyplot as plt
from math import *
from morai_msgs.msg import CtrlCmd

# colors
red, green, blue, yellow, black = (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255,255,255)

class LaneTracker():
    def __init__(self):
        rospy.init_node('lane_drive', anonymous=True)
        
        self._lane_track_end = False

        self.warp_img_w, self.warp_img_h, self.warp_img_mid = 650, 120, 60

        self.lower_wlane = np.array([0, 0, 200])
        self.upper_wlane = np.array([180, 255, 255])

        self.lower_ylane = np.array([10, 100, 100])
        self.upper_ylane = np.array([40, 255, 255])

        self.warp_src  = np.array([[190, 170], [450, 170], 
                                [50,  220], [590, 220]], dtype=np.float32)     
        # self.warp_src  = np.array([[100, 100], [535, 100], 
        #                         [30,  150], [608, 150]], dtype=np.float32)

        self.warp_dist = np.array([[100, 0], [649-100, 0],
                                [100, 119], [649-100, 119]], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.warp_src, self.warp_dist)

        # canny params
        self.canny_low, self.canny_high = 100, 120

        # HoughLineP params
        self.hough_threshold, self.min_length, self.min_gap = 10, 50, 10

        self.angle = 0.0
        self.prev_angle = []

        self.default_lane = [78.0, 241.0, 409.0, 575.0]
        self.lane = np.array([78.0, 241.0, 409.0, 575.0])

        # filtering params:
        self.angle_tolerance = np.radians(35)
        self.cluster_threshold = 25

        #lidar
        self.avoid_trig = 'mid'
        # self.pid_controller = PidController()
        self.msg_pub = rospy.Publisher("/ctrl_cmd",CtrlCmd,queue_size=1)
        
        self.cam = None
        self.bridge = CvBridge()
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback_cam)
        rospy.Subscriber("/scan_origin", LaserScan, self.callback_lidar)
        rospy.loginfo("Lane Tracker Node on. Waiting for topics...")
    
    # Main Function : Make control msg.
    # Return : CtrlCmd
    def main(self):
        
        # fil_img = self.color_filter(self.cam, show=True)
        canny = self.to_canny(self.cam, show=True)
        bev = self.to_perspective(canny, show=True)
        lines = self.hough(bev, show=True)
        positions = self.filter(lines, show=True)
        lane_candidates = self.get_cluster(positions)
        predicted_lane = self.predict_lane()
        self.update_lane(lane_candidates, predicted_lane)
        self.mark_lane(bev)
        

        gray_img = cv2.cvtColor(self.cam, cv2.COLOR_BGR2GRAY)
        fil_bev = self.to_perspective(gray_img, show=True)
        ret, thresh1 = cv2.threshold(fil_bev, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow('fil_bev', thresh1)
        self.trig_obs, fil_hist = self.plothistogram(thresh1) #static obstacle trigger

        self.obs = self.obstacle_trig(self.lidar, self.angle_increment)
        
        return self.obs
    
    def callback_cam(self, msg):
        self.cam = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.main()
        cv2.imshow('img_raw', self.cam)
        cv2.waitKey(1)
    
    def callback_lidar(self,msg):
        self.lidar = msg.ranges
        self.angle_increment = msg.angle_increment * 360 / pi
        self.minAngle = msg.angle_min

    def pid(self):
        self.cmd.steering, self.cmd.velocity = self.pid_controller(self.angle, self.target)
        self.cmd.longlCmdType = 2
        self.cmd.accel = 0
        self.cmd.brake = 0
        self.cmd.acceleration = 0
        self.msg_pub.publish(self.cmd)

    def obstacle_trig(self, range, angle_inc):
        self.avoid_trig
        ranges = np.array(range)
        # print("len : ", ranges)
        
        ranges[:int(362/4)] = 0.0
        ranges[int(362*3/4):] = 0.0
        # print(len(ranges))
        # print(ranges)
        # print(angle_inc)
        deg = np.arange(362) * angle_inc - 181 * angle_inc
        # flip_deg = np.flip(deg)
        # deg = np.concatenate((flip_deg[181:362],deg[:181]))
        # print(deg)
        # print(len(deg))
        #처음 시작이 앞
        #sin이 x값, cos이 y값
        mask = (np.abs(ranges * np.sin(deg)) < 0.8) & (ranges * np.cos(deg) < 0.4)
        # print("rangessin",ranges * np.sin(deg))
        # print("rangescos",ranges * np.cos(deg))
        # print(mask)
        filtered = np.where(mask, ranges, 0.0)
        nz = np.nonzero(filtered)[0]

        # print(len(nz))
        # print(len(nz))
        if len(nz) > 5:
            # print("ooooooobsssssssstaaaaaaaaaaaacle")
            # if np.median(nz) < 362/4:
            #     self.avoid_direction = 'left'
            #     print("leftdetected")
            # else:
            #     self.avoid_direction = 'right'
            #     print("rightdetected")
            self.avoid_trig = 'left'
            print(('avoid is : ', self.avoid_trig))
        else:
            self.avoid_trig = 'mid'

        return self.avoid_trig
    

    def plothistogram(self, image):
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        midpoint = np.int32(histogram.shape[0]/2)
        leftbase = np.argmax(histogram[30:midpoint])
        rightbase = np.argmax(histogram[midpoint:]) + midpoint
        # print(leftbase, rightbase, np.argmax(histogram[midpoint:]))
        if abs(leftbase - 311.5) >= (rightbase - 311.5):
            trig_obs = 'left'
        else:
            trig_obs = 'right'
        

        return trig_obs, histogram

    def color_filter(self, img, show=True):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_wlane = cv2.inRange(img_hsv, self.lower_wlane, self.upper_wlane)
        img_ylane = cv2.inRange(img_hsv, self.lower_ylane, self.upper_ylane)
        self.img_lane = cv2.bitwise_or(img_wlane, img_ylane)
        if show:
            cv2.imshow('filtered_img', self.img_lane)
        return self.img_lane

    def to_perspective(self, img, show=True):
        img = cv2.warpPerspective(img, self.M, (self.warp_img_w, self.warp_img_h), flags=cv2.INTER_LINEAR)
        if show:
            cv2.imshow('bird-eye-view', img)
        return img
    
    def to_canny(self, img, show=True):
        if img is not None:
            cv2.imshow('img_canny', img)
            img = cv2.GaussianBlur(img, (7,7), 0)
            img = cv2.Canny(img, self.canny_low, self.canny_high)
            if show:
                cv2.imshow('canny', img)
                cv2.waitKey(1)
            return img

    def hough(self, img, show=False):
        lines = cv2.HoughLinesP(img, 1, np.pi/180, self.hough_threshold, self.min_gap, self.min_length)
        if show:
            hough_img = np.zeros((img.shape[0], img.shape[1], 3))
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    cv2.line(hough_img, (x1, y1), (x2, y2), red, 2)
            cv2.imshow('hough', hough_img)
        return lines

    def filter(self, lines, show=True):
        '''
        filter lines that are close to previous angle and calculate its positions
        '''
        # if lines is not None:
        #     print("lines : ", len(lines), lines)
        thetas, positions = [], []
        if show:
            filter_img = np.zeros((self.warp_img_h, self.warp_img_w, 3))

        if lines is not None:
            if len(lines) == 1:
                print("________one lines_______")
                positions = self.default_lane
                thetas.append(0)
            else:
                for x1, y1, x2, y2 in lines[:, 0]:
                    if y1 == y2:
                        continue
                    flag = 1 if y1-y2 > 0 else -1
                    theta = np.arctan2(flag * (x2-x1), flag * 0.9* (y1-y2))
                    if abs(theta - self.angle) < self.angle_tolerance:
                        position = float((x2-x1)*(self.warp_img_mid-y1))/(y2-y1) + x1
                        thetas.append(theta)
                        positions.append(position) 
                        # print("position : ", positions)
                        # print("theta : ", thetas)

                        if show:
                            cv2.line(filter_img, (x1, y1), (x2, y2), red, 2)
        else:
            print("________no lines_______")
            positions = self.default_lane
            thetas.append(0)
            

        self.prev_angle.append(self.angle)
        if thetas:
            self.angle = np.mean(thetas)
        if show:
            cv2.imshow('filtered lines', filter_img)
        return positions

    def get_cluster(self, positions):
        '''
        group positions that are close to each other
        '''
        clusters = []
        for position in positions:
            if 0 <= position < 640:
                for cluster in clusters:
                    if abs(cluster[0] - position) < self.cluster_threshold:
                        cluster.append(position)
                        break
                else:
                    clusters.append([position])
        lane_candidates = [np.mean(cluster) for cluster in clusters]
        # print("lane_candidates : ", lane_candidates)
        return lane_candidates

    def predict_lane(self):
        '''
        predicts lane positions from previous lane positions and angles
        '''
        predicted_lane = (self.lane[1] + self.lane[2])/2 + [-250/max(np.cos(self.angle), 0.75), -84/max(np.cos(self.angle), 0.75), 84/max(np.cos(self.angle), 0.75), 250/max(np.cos(self.angle), 0.75)]
        predicted_lane = predicted_lane + (self.angle - np.mean(self.prev_angle))*70
        # print("predicted_lane : ", predicted_lane)
        return predicted_lane

    def update_lane(self, lane_candidates, predicted_lane):
        '''
        calculate lane position using best fitted lane and predicted lane
        '''

        if not lane_candidates:
            self.lane = predicted_lane
            for i in range(4):
                if abs(self.lane[i] - self.default_lane[i]) > 40:
                    self.lane = np.array(self.default_lane)
            # print(":::::::::::::::::::::::::::::")
            return

        possibles = []

        for lc in lane_candidates:

            idx = np.argmin(abs(self.lane-lc))

            if idx == 0:
                estimated_lane = [lc, lc + 166/max(np.cos(self.angle), 0.75), lc + 334/max(np.cos(self.angle), 0.75), lc + 500/max(np.cos(self.angle), 0.75)]
                lc2_candidate, lc3_candidate, lc4_candidate = [], [], []
                for lc2 in lane_candidates:
                    if abs(lc2-estimated_lane[1]) < 50 :
                        lc2_candidate.append(lc2)
                for lc3 in lane_candidates:
                    if abs(lc3-estimated_lane[2]) < 50 :
                        lc3_candidate.append(lc3)
                for lc4 in lane_candidates:
                    if abs(lc4-estimated_lane[3]) < 50 :
                        lc4_candidate.append(lc4)
                if not lc2_candidate:
                    lc2_candidate.append(lc + 166/max(np.cos(self.angle), 0.75))
                if not lc3_candidate:
                    lc3_candidate.append(lc + 334/max(np.cos(self.angle), 0.75))
                if not lc4_candidate:
                    lc4_candidate.append(lc + 500/max(np.cos(self.angle), 0.75))
                for lc2 in lc2_candidate:
                    for lc3 in lc3_candidate:
                        for lc4 in lc4_candidate:
                            possibles.append([lc, lc2, lc3, lc4])
                # print("0 : ",lc2_candidate, lc3_candidate, lc4_candidate)

            elif idx == 1:
                estimated_lane = [lc - 166/max(np.cos(self.angle), 0.75), lc, lc + 168/max(np.cos(self.angle), 0.75), lc + 334/max(np.cos(self.angle), 0.75)]
                lc1_candidate, lc3_candidate, lc4_candidate = [], [], []
                for lc1 in lane_candidates:
                    if abs(lc1-estimated_lane[0]) < 50 :
                        lc1_candidate.append(lc1)
                for lc3 in lane_candidates:
                    if abs(lc3-estimated_lane[2]) < 50 :
                        lc3_candidate.append(lc3)
                for lc4 in lane_candidates:
                    if abs(lc4-estimated_lane[3]) < 50 :
                        lc4_candidate.append(lc4)
                if not lc1_candidate:
                    lc1_candidate.append(lc - 166/max(np.cos(self.angle), 0.75))
                if not lc3_candidate:
                    lc3_candidate.append(lc + 168/max(np.cos(self.angle), 0.75))
                if not lc4_candidate:
                    lc4_candidate.append(lc + 334/max(np.cos(self.angle), 0.75))
                for lc1 in lc1_candidate:
                    for lc3 in lc3_candidate:
                        for lc4 in lc4_candidate:
                            possibles.append([lc1, lc, lc3, lc4])
                # print("1 : ",lc1_candidate, lc3_candidate, lc4_candidate)

            elif idx == 2:
                estimated_lane = [lc - 334/max(np.cos(self.angle), 0.75), lc - 168/max(np.cos(self.angle), 0.75), lc, lc + 166/max(np.cos(self.angle), 0.75)]
                lc1_candidate, lc2_candidate, lc4_candidate = [], [], []
                for lc1 in lane_candidates:
                    if abs(lc1-estimated_lane[0]) < 50 :
                        lc1_candidate.append(lc1)
                for lc2 in lane_candidates:
                    if abs(lc2-estimated_lane[1]) < 50 :
                        lc2_candidate.append(lc2)
                for lc4 in lane_candidates:
                    if abs(lc4-estimated_lane[3]) < 50 :
                        lc4_candidate.append(lc4)
                if not lc1_candidate:
                    lc1_candidate.append(lc - 334/max(np.cos(self.angle), 0.75))
                if not lc2_candidate:
                    lc2_candidate.append(lc - 168/max(np.cos(self.angle), 0.75))
                if not lc4_candidate:
                    lc4_candidate.append(lc + 166/max(np.cos(self.angle), 0.75))
                for lc1 in lc1_candidate:
                    for lc2 in lc2_candidate:
                        for lc4 in lc4_candidate:
                            possibles.append([lc1, lc2, lc, lc4])
                # print("2 : ",lc1_candidate, lc2_candidate, lc4_candidate)

            else :
                estimated_lane = [lc - 500/max(np.cos(self.angle), 0.75), lc - 334/max(np.cos(self.angle), 0.75), lc - 166/max(np.cos(self.angle), 0.75), lc]
                lc1_candidate, lc2_candidate, lc3_candidate = [], [], []
                for lc1 in lane_candidates:
                    if abs(lc1-estimated_lane[0]) < 50 :
                        lc1_candidate.append(lc1)
                for lc2 in lane_candidates:
                    if abs(lc2-estimated_lane[1]) < 50 :
                        lc2_candidate.append(lc2)
                for lc3 in lane_candidates:
                    if abs(lc3-estimated_lane[2]) < 50 :
                        lc3_candidate.append(lc3)
                if not lc1_candidate:
                    lc1_candidate.append(lc - 500/max(np.cos(self.angle), 0.75))
                if not lc2_candidate:
                    lc2_candidate.append(lc - 334/max(np.cos(self.angle), 0.75))
                if not lc3_candidate:
                    lc3_candidate.append(lc - 166/max(np.cos(self.angle), 0.75))
                for lc1 in lc1_candidate:
                    for lc2 in lc2_candidate:
                        for lc3 in lc3_candidate:
                            possibles.append([lc1, lc2, lc3, lc])
                # print("3 : ",lc1_candidate, lc2_candidate, lc3_candidate)
        # print("estimated_lane : ", estimated_lane)
        
        possibles = np.array(possibles)
        # print("possibles : ", possibles)
        error = np.sum((possibles-predicted_lane)**2, axis=1)
        best = possibles[np.argmin(error)]
        self.lane = 0.6 * best + 0.4 * predicted_lane
        
        for i in range(4):
            if abs(self.lane[i] - self.default_lane[i]) > 40:
                self.lane = np.array(self.default_lane)
        # for i in range(3):
        #     self.lane[i] = self.moving_filter(self.lane[i], self.lane_queues[i])
        self.mid = np.mean(self.lane)

    def mark_lane(self, img, lane=None):
        '''
        mark calculated lane position to an image 
        '''
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if lane is None:
            lane = self.lane
            self.mid = (self.lane[1] + self.lane[2])/2
            
        l1, l2, l3, l4 = self.lane
        cv2.circle(img, (int(l1), self.warp_img_mid), 3, red, 5, cv2.FILLED)
        cv2.circle(img, (int(l2), self.warp_img_mid), 3, green, 5, cv2.FILLED)
        cv2.circle(img, (int(l3), self.warp_img_mid), 3, blue, 5, cv2.FILLED)
        cv2.circle(img, (int(l4), self.warp_img_mid), 3, black, 5, cv2.FILLED)
        cv2.circle(img, (int(self.mid), 60), 3, yellow, 5, cv2.FILLED)
        cv2.imshow('marked', img)

if __name__ == "__main__":
    try:
        lane_tracker = LaneTracker()
        # gps_tracker.main()
        # while not lane_tracker._lane_track_end:  # You can define your own termination condition
        # lane_tracker.main()
        # gps_tracker.plot_gps_track()
        cv2.waitKey(1)
            # time.sleep(0.1)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass