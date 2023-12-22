# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 07:57:53 2019

@author: orrivlin
"""

import torch 
import numpy as np
from Models import ConvNet
from Nav2D import Navigate2D
from copy import deepcopy as dc

TRY_TIMES = 100
epochs = 500000
success_num = 0

N = 20
Nobs = 15
Dobs = 2
Rmin = 10
env = Navigate2D(N,Nobs,Dobs,Rmin)
[Sdim,Adim] = env.get_dims()
model = ConvNet(Sdim[0],Sdim[0],3,Adim).cuda()
model.load_state_dict(torch.load('model.pt'))
image_mean = torch.load('norm.pt').cuda()
for i in range(epochs) :
    start_obs, done = env.reset()
    cum_obs = dc(start_obs)
    obs = dc(start_obs)
    done = False
    state = env.get_tensor(obs)
    sum_r = 0
    epsilon = 0.0 

    for t in range(TRY_TIMES): 
        Q = model(state.cuda() - image_mean)
        num = np.random.rand()
        if (num < epsilon):
            action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
        else:
            action = torch.argmax(Q,dim=1)
        new_obs, reward, done, dist = env.step(obs,action.item())
            # 현재 포즈와 골 포즈를 프린트 하는 아래 코드 두 줄 추가
        # print("now pose : ", np.argwhere(new_obs[:,:,1] == 10.0)[0], end=' ')
        # print("goal pose : ",np.argwhere(new_obs[:,:,2] == 10.0)[0])

        new_state = env.get_tensor(new_obs)
        sum_r = sum_r + reward
        state = dc(new_state)
        obs = dc(new_obs)
        cum_obs[:,:,1] += obs[:,:,1]
        if done:
            print('success to find')
            success_num += 1
            break
        elif t == TRY_TIMES-1 :
            print('fail to find')
                
# env.render(cum_obs)
# print('time: {}'.format(t))
# print('return: {}'.format(sum_r))

rate = success_num/epochs
print("Test Finished===========Success rate : %f"%rate)