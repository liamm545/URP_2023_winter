# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 07:57:53 2019

@author: orrivlin
"""

import torch 
import numpy as np
import random
import matplotlib.pyplot as plt
from Models import ConvNet
from Nav2D import Navigate2D
from copy import deepcopy as dc

TRY_TIMES = 50
success_num = 0
epochs = 100
Nobs = 4
Dobs = 2
Slope = random.randint(30,90)
env = Navigate2D(Nobs,Dobs,Slope)
[Sdim,Adim] = env.get_dims()
# first model file
model_1 = ConvNet(Sdim[0],Sdim[1],3,Adim).cuda()
model_1.load_state_dict(torch.load('model_1.pt'))
image_mean_1 = torch.load('norm_1.pt').cuda()
# second model file
model_2 = ConvNet(Sdim[0],Sdim[1],3,Adim).cuda()
model_2.load_state_dict(torch.load('model_2.pt'))
image_mean_2 = torch.load('norm_2.pt').cuda()

def visualize_episode(trajectory_1, trajectory_2):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 첫번째 모델 Visualization
    img_1 = np.zeros((20, 40, 3), dtype=np.uint8)
    img_1[trajectory_1[0][:, :, 0] == 1.0] = [255, 0, 0]  # 장애물

    for obs in trajectory_1:
        pos = np.argwhere(obs[:, :, 1] == 10.0)[0]
        img_1[pos[0], pos[1]] = [255, 255, 0]  # 이동 경로

    initial = np.argwhere(trajectory_1[0][:, :, 1] == env.scale)[0]
    img_1[initial[0], initial[1]] = [0, 255, 0]  # 시작 위치

    target = np.argwhere(trajectory_1[0][:, :, 2] == env.scale)[0]
    img_1[target[0], target[1]] = [0, 0, 255]  # 목표 위치

    ax1.imshow(img_1)
    ax1.set_title('Model 1')

    # 두번째 모델 Visualization
    img_2 = np.zeros((20, 40, 3), dtype=np.uint8)
    img_2[trajectory_2[0][:, :, 0] == 1.0] = [255, 0, 0]  # 장애물

    for obs in trajectory_2:
        pos = np.argwhere(obs[:, :, 1] == 10.0)[0]
        img_2[pos[0], pos[1]] = [255, 255, 0]  # 이동 경로

    initial = np.argwhere(trajectory_2[0][:, :, 1] == env.scale)[0]
    img_2[initial[0], initial[1]] = [0, 255, 0]  # 시작 위치

    target = np.argwhere(trajectory_2[0][:, :, 2] == env.scale)[0]
    img_2[target[0], target[1]] = [0, 0, 255]  # 목표 위치

    ax2.imshow(img_2)
    ax2.set_title('Model 2')

    plt.show()
    plt.pause(5)
    plt.close()
    
for i in range(epochs) :
    start_obs, done = env.reset()

    cum_obs_1 = dc(start_obs)
    obs_1 = dc(start_obs)
    trajectory_1 = [obs_1]
    done_1 = False
    state_1 = env.get_tensor(obs_1)
    sum_r_1 = 0

    cum_obs_2 = dc(start_obs)
    obs_2 = dc(start_obs)
    trajectory_2 = [obs_2]
    done_2 = False
    state_2 = env.get_tensor(obs_2)
    sum_r_2 = 0

    for t_1 in range(TRY_TIMES): 
        Q_1 = model_1(state_1.cuda() - image_mean_1)
        action_1 = torch.argmax(Q_1,dim=1)
        new_obs_1, reward_1, done_1, dist = env.step(obs_1,action_1.item())
        trajectory_1.append(new_obs_1)
        new_state_1 = env.get_tensor(new_obs_1)
        sum_r_1 += reward_1
        state_1 = dc(new_state_1)
        obs_1 = dc(new_obs_1)
        cum_obs_1[:,:,1] += obs_1[:,:,1]    
        # print("now pose : ", np.argwhere(new_obs_1[:,:,1] == 10.0)[0], end=' ')
        # print("goal pose : ",np.argwhere(new_obs_1[:,:,2] == 10.0)[0])
        if done_1:
            break

    for t_2 in range(TRY_TIMES) :
        Q_2 = model_2(state_2.cuda() - image_mean_2)
        action_2 = torch.argmax(Q_2,dim=1)
        new_obs_2, reward_2, done_2, dist = env.step(obs_2,action_2.item())
        trajectory_2.append(new_obs_2)
        new_state_2 = env.get_tensor(new_obs_2)
        sum_r_2 += reward_2
        state_2 = dc(new_state_2)
        obs_2 = dc(new_obs_2)
        cum_obs_2[:,:,1] += obs_2[:,:,1]
        # print("now pose : ", np.argwhere(new_obs_2[:,:,1] == 10.0)[0], end=' ')
        # print("goal pose : ",np.argwhere(new_obs_2[:,:,2] == 10.0)[0])
        if done_2:
            break

    plt.ion()
    visualize_episode(trajectory_1, trajectory_2)
    plt.ioff()
                    
    # env.render(cum_obs)
    print("for model 1 ")
    print('time: {}'.format(t_1))
    print('return: {}'.format(sum_r_1))

    print("for model 2 ")
    print('time: {}'.format(t_2))
    print('return: {}'.format(sum_r_2))