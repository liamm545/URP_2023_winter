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

from matplotlib.pyplot import imshow
from copy import deepcopy as dc

LEFT_POINT = (9, 9)
RIGHT_POINT = (29, 9)

class Navigate2D:
    def __init__(self,Nobs,Dobs,slope):
        self.W = 40
        self.H = 20
        self.Nobs = Nobs
        self.Dobs = Dobs
        self.state_dim = [self.W,self.H,3]
        self.action_dim = 4
        self.scale = 10.0
        self.slope = slope
        
    def get_dims(self):
        return self.state_dim, self.action_dim
        
    def reset(self):
        # map 밑배경 생성
        grid = np.zeros((self.H,self.W,3))
        # 차선 생성
        left_end_x = int(LEFT_POINT[0] + self.H/2 / self.slope)
        left_end_y1 = int(LEFT_POINT[1] - self.H/2)
        left_end_y2 = int(LEFT_POINT[1] + self.H/2)
        
        right_end_x = int(RIGHT_POINT[0] + self.H/2 / self.slope)
        right_end_y1 = int(RIGHT_POINT[1] - self.H/2)
        right_end_y2 = int(RIGHT_POINT[1] + self.H/2)

        start_point1 = (LEFT_POINT[0], LEFT_POINT[1] - self.H/2)
        end_point1 = (left_end_x, left_end_y2)

        start_point2 = (RIGHT_POINT[0], RIGHT_POINT[1] - self.H/2)
        end_point2 = (right_end_x, right_end_y2)
        cv2.line(grid, start_point1, end_point1, (1, 0, 0), 1)
        cv2.line(grid, start_point2, end_point2, (1, 0, 0), 1)

        # 라바콘 생성
        for i in range(self.Nobs):
            # 일단 장애물의 y 중심 먼저 생성
            center_y = random.randint(3,15)
            # 선택한 y값 기준 직선 안쪽에 있는 경계들 중에서 랜덤으로 x값 생성
            y_1 , y_2 = min(np.argwhere[:,center_y,0] == 1), max(np.argwhere[:,center_y,0] == 1)
            center_x = random.randint(y_1, y_2)
            center = (center_y,center_x)
            minX = np.maximum(center[0,0] - self.Dobs,1)
            minY = np.maximum(center[0,1] - self.Dobs,1)
            maxX = np.minimum(center[0,0] + self.Dobs,self.N-1)
            maxY = np.minimum(center[0,1] + self.Dobs,self.N-1)
            grid[minX:maxX,minY:maxY,0] = 1.0

        # 출발점 생성. 차선 안쪽에서 생성하도록 
        start = (19,(start_point1[1]+start_point2[2])/2)
        finish = (0,(left_end_y2+right_end_y2)/2)
        grid[start[0],start[1],1] = self.scale*1.0
        grid[finish[0],finish[1],2] = self.scale*1.0
        done = False
        return grid, done
    
    def step(self,grid,action):
        # max_norm = self.N
        new_grid = dc(grid)
        done = False
        reward = -1.0
        act = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        pos = np.argwhere(grid[:,:,1] == self.scale**1.0)[0]
        target = np.argwhere(grid[:,:,2] == self.scale*1.0)[0]
        new_pos = pos + act[action]
        
        dist1 = np.linalg.norm(pos - target)
        dist2 = np.linalg.norm(new_pos - target)
        #reward = (dist1 - dist2)*(max_norm - dist2)
        #reward = -dist2
        reward = -1
        if (np.any(new_pos < 0.0) or np.any(new_pos > (self.N - 1)) or (grid[new_pos[0],new_pos[1],0] == 1.0)):
            #dist = np.linalg.norm(pos - target)
            #reward = (dist1 - dist2)
            return grid, reward, done, dist2
        new_grid[pos[0],pos[1],1] = 0.0
        new_grid[new_pos[0],new_pos[1],1] = self.scale*1.0
        if ((new_pos[0] == target[0]) and (new_pos[1] == target[1])):
            reward = 0.0
            done = True
        #dist = np.linalg.norm(new_pos - target)
        #reward = (dist1 - dist2)
        return new_grid, reward, done, dist2
    
    def get_tensor(self,grid):
        S = torch.Tensor(grid).transpose(2,1).transpose(1,0).unsqueeze(0)
        return S
    
    def render(self,grid):
        # imshow(grid)
        plot = imshow(grid)
        return plot