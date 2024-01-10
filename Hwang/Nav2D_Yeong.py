# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:33:29 2019

@author: Or
"""

import torch
import cv2
import numpy as np
import matplotlib as plt
import random
import math
from matplotlib.pyplot import imshow
from copy import deepcopy as dc
from collections import deque

LEFT_POINT = (9, 9)
RIGHT_POINT = (29, 9)

class Navigate2D:
    def __init__(self,Nobs,Dobs,Rmin):
        self.W = 40
        self.H = 20
        self.Nobs = Nobs
        self.Dobs = Dobs
        self.Rmin = Rmin
        self.state_dim = [self.W,self.H,3]
        self.action_dim = 5
        self.scale = 10.0
        self.consecutive_steps = 0
        self.prev_positions = deque(maxlen=4)
        
    def get_dims(self):
        return self.state_dim, self.action_dim
        
    def reset(self):
        # map 밑배경 생성
        grid = np.zeros((self.H,self.W,3))

        self.slope = math.tan(math.pi * random.randint(45,135)/180)
        # 차선 생성
        left_start_x = int(LEFT_POINT[0] - LEFT_POINT[0] / self.slope)
        left_start_y = 19

        left_end_x = int(LEFT_POINT[0] + LEFT_POINT[0] / self.slope)
        left_end_y = 0
        
        right_start_x = left_start_x + 20
        right_start_y = 19

        right_end_x = left_end_x + 20
        right_end_y = 0
        
        self.consecutive_steps = 0
        self.prev_pos = None
        
        # 차선 draw
        cv2.line(grid, (left_end_x,left_end_y),(left_start_x,left_start_y),
                 (2, 0, 0), 1)
        cv2.line(grid, (right_end_x,right_end_y),(right_start_x, right_start_y), 
                 (2, 0, 0), 1)

        curr_Nobs = self.Nobs

        # 라바콘 생성
        for _ in range(curr_Nobs):
            # 장애물의 y 중심 생성
            center_y = random.randint(7,13)
            # 선택한 y값 기준 직선 안쪽에 있는 경계들 중에서 랜덤으로 x값 생성
            y_1 , y_2 = min(np.argwhere(grid[center_y,:,0] == 2)+2), max(np.argwhere(grid[center_y,:,0] == 2)-2)
            center_x = random.randint(min(y_1,y_2), max(y_1,y_2))
            minX = center_x - 1
            minY = center_y - 1
            maxX = center_x + 1
            maxY = center_y + 1
            grid[minY:maxY,minX:maxX,0] = 1.0
            
            # labacorn surplus 생성
            grid[minY-1:minY,minX-1:maxX+1,0] = 255.0
            grid[minY:maxY,minX-1:minX,0] = 255.0
            grid[minY:maxY,maxX:maxX+1,0] = 255.0
            grid[maxY:maxY+1,minX-1:maxX+1,0] = 255.0

        # lane surplus 생성
        cv2.line(grid, (left_end_x-1,left_end_y),(left_start_x-1,left_start_y),
                    (255, 0, 0), 1)
        cv2.line(grid, (left_end_x+1,left_end_y),(left_start_x+1,left_start_y),
                    (255, 0, 0), 1)
        cv2.line(grid, (right_end_x-1,right_end_y),(right_start_x-1, right_start_y), 
                    (255, 0, 0), 1)
        cv2.line(grid, (right_end_x+1,right_end_y),(right_start_x+1, right_start_y), 
                    (255, 0, 0), 1)
        
        # 출발점 생성. 차선 안쪽에서 생성하도록 
        start = (19,int((left_start_x+right_start_x)/2))
        finish = (0,int((left_end_x+right_end_x)/2))

        grid[start[0],start[1],1] = self.scale*1.0
        grid[finish[0],finish[1],2] = self.scale*1.0
        done = False
        # imshow(grid)
        # plt.pyplot.show()
        return grid, done
    
    def make_car_boound(self, grid, yaw, pos) :
        # 먼저 차량의 앞, 뒤 코 설정
        car_front = (int(pos[1] + 8*math.cos(yaw)), int(pos[0] - 8*math.sin(yaw)))
        car_rear = (int(pos[1] + 3*math.cos(yaw)), int(pos[0] + 3*math.sin(yaw)))
        # 차량 앞, 뒤 boudary 생성
        if yaw == math.pi * 90/180 :
            front_left = (car_front[0]-4,car_front[1])
            front_right = (car_front[0]+4,car_front[1])
            rear_left = (car_rear[0]-4, car_rear[1])
            rear_right = (car_rear[0]+4, car_rear[1])
        else :        
            front_right = (car_front[0]+4*math.sin(yaw),car_front[1]+4*math.cos(yaw))
            front_left = (car_front[0]-4*math.sin(yaw),car_front[1]-4*math.cos(yaw))
            rear_right = (car_rear[0]+4*math.sin(yaw), car_rear[1]+4*math.cos(yaw))
            rear_left = (car_rear[0]-4*math.sin(yaw), car_rear[1]-4*math.cos(yaw))
        points = np.array([front_left, front_right, rear_right, rear_left], np.int32)
        points = points.reshape(-1,1,2)
        car_grid = cv2.fillPoly(grid, [points],[0,255,255])
        return car_grid
    
    def det_yaw(self, action) :
        if (action == np.array([-1,0])).all() :
            return math.pi * 90/180
        elif (action == np.array([0,1])).all() :
            return math.pi * 30/180
        elif (action == np.array([-1,1])).all() :
            return math.pi * 45/180
        elif (action == np.array([0,-1])).all() :
            return math.pi * 150/180
        elif (action == np.array([-1,-1])).all() :
            return math.pi * 135/180  
    
    ##########
    def step(self,grid,action):
        # max_norm = self.N
        new_grid = dc(grid)
        car_grid = dc(grid)
        
        done = False
        crack = False
        over_lane = False
        
        reward = -0.5
        # act = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        # act = np.array([[0,1],[-1,0],[0,-1]])
        act = np.array([[-1,0],[0,1],[-1,1],[0,-1],[-1,-1]])
        
        pos = np.argwhere(grid[:,:,1] == self.scale**1.0)[0]
        target = np.argwhere(grid[:,:,2] == self.scale*1.0)[0]
        new_pos = pos + act[action]
        
        # dist1 = np.linalg.norm(pos - target)
        # dist2 = np.linalg.norm(new_pos - target)
        dist = math.sqrt((new_pos[0]-target[0])**2+(new_pos[1]-target[1])**2)
        dist_out = np.linalg.norm(new_pos - target)
        
        yaw = self.det_yaw(act[action])
        car_grid = self.make_car_boound(car_grid,yaw,new_pos) # 현재 차량을 그린 car_grid 가져옴
        car_pos = np.where((car_grid[:,:,1]==255) & (car_grid[:,:,2]==255)) # car_grid로부터 차가 차지하는 좌표들 가져옴
        car_pos = list(zip(car_pos[0],car_pos[1]))
        
        self.prev_positions.append(pos)
        
        if len(self.prev_positions) == 4 and pos[0] == self.prev_positions[3][0]:
            reward += -1.5
        
        if (np.any(new_pos < 0.0) or new_pos[1] > (39.0)):
            #dist = np.linalg.norm(pos - target)
            #reward = (dist1 - dist2)
            reward += -5.0
            return grid, reward, done, dist_out, car_grid, crack
        
        # if (grid[new_pos[0],new_pos[1],0] == 1.0):
        #     return grid, reward, done, dist2
        
        # surplus
        # if (grid[new_pos[0],new_pos[1],0] == 255.0):
        #     reward += -0.7
        
        # obs
        # elif (grid[new_pos[0],new_pos[1],0] == 1.0):
        #     reward += -2.0
        #     return grid, reward, done, dist_out, car_grid, crack
        
        for car in car_pos :
            # 차량이 있는 위치에 이미 장애물이 있는 경우에는 이동 X
            if grid[new_pos[0],new_pos[1],0] == 255.0:
                reward += -0.8
            elif new_grid[car[0],car[1],0] == 2.0 : # 차선 밟으면 감점하고 이동
                reward += -1.5
                over_lane = True
            elif new_grid[car[0],car[1],0] == 1.0:
                crack = True
                reward += -2.0
                return grid, reward, done, dist_out, car_grid, crack
            
        
        new_grid[pos[0],pos[1],1] = 0.0
        new_grid[new_pos[0],new_pos[1],1] = self.scale*1.0
        
        if ((new_pos[0] == target[0]) and (new_pos[1] == target[1])):
            # print("good")
            reward += 100.0
            done = True
        #dist = np.linalg.norm(new_pos - target)
        #reward = (dist1 - dist2)
        return new_grid, reward, done, dist_out, car_grid, crack
    
    def get_tensor(self,grid):
        S = torch.Tensor(grid).transpose(2,1).transpose(1,0).unsqueeze(0)
        return S
    
    def render(self,grid):
        # imshow(grid)
        plot = imshow(grid)
        return plot