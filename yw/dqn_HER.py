
"""
@author: orrivlin
"""

import torch 
import sys
import numpy as np
import copy
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import collections
from collections import deque
from Models import ConvNet, ConvNet_noPool,ConvNet_withLSTM
from log_utils import logger, mean_val
from HER import HER
from copy import deepcopy as dc



class DQN_HER:
    def __init__(self, env, gamma, buffer_size, ddqn,random_update=True):
        self.env = env
        [Sdim,Adim] = env.get_dims()

        self.her = HER()
        self.gamma = gamma
        self.random_update = random_update
        self.memory = collections.deque(maxlen=buffer_size)

        #############################################################
        self.model = ConvNet_withLSTM(Sdim[0],Sdim[1],3,Adim,lstm_hidden_size=128, lstm_num_layers=1).cuda()
        self.target_model = ConvNet_withLSTM(Sdim[0],Sdim[1],3,Adim,lstm_hidden_size=128, lstm_num_layers=1).cuda()
        self.target_model.load_state_dict(self.model.state_dict())

        # Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        #############################################################

        # Other parameters
        self.epsilon = 0.1
        self.learning_rate = 0.0001
        self.batch_size = 16
        self.lookup_step = 1  # You may adjust this based on your requirements
        self.min_epi_num = 100
        self.target_update_period = 10  # You may adjust this based on your requirements
        self.tau = 0.001  # Soft update parameter

        ##############################################################   

        self.batch_size = 16
        self.epsilon = 0.1
        self.buffer_size = buffer_size
        self.step_counter = 0

        self.epsi_high = 0.9
        self.epsi_low = 0.1

        self.steps = 0
        self.count = 0
        self.decay = 2000
        self.eps = self.epsi_high
        self.update_target_step = 3000
        self.log = logger()
        self.log.add_log('tot_return')
        self.log.add_log('avg_loss')
        self.log.add_log('final_dist')
        self.log.add_log('buffer')
        self.image_mean = 0
        self.image_std = 0
        self.ddqn = ddqn
        self.previous_action = 2
        self.replay_buffer = deque(maxlen=buffer_size)

    def sample_buffer(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update:  # Random update
            print(self.memory)
            sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = True  # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))  # get the minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode) - min_step + 1)  # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])  # buffers, sequence_length

    def train_drqn(self, sampled_buffer):
        # Extract data from the sampled buffer
        observations = sampled_buffer[0]['obs']
        actions = sampled_buffer[0]['acts']
        rewards = sampled_buffer[0]['rews']
        next_observations = sampled_buffer[0]['next_obs']
        dones = sampled_buffer[0]['done']

        # Convert data to PyTorch tensors
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_observations = torch.FloatTensor(next_observations).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Initialize hidden states for the DRQN
        h_target, c_target = self.model.init_hidden_state(batch_size=self.batch_size, training=True)

        # Forward pass through the target DRQN for next observations
        q_target, _, _ = self.model(next_observations, h_target.to(self.device), c_target.to(self.device))

        # Get the maximum Q-value for each next observation
        q_target_max = q_target.max(2)[0].view(self.batch_size, -1).detach()

        # Calculate target Q-values using the Bellman equation
        targets = rewards + self.gamma * q_target_max * (1 - dones)

        # Forward pass through the DRQN for current observations
        h, c = self.model.init_hidden_state(batch_size=self.batch_size, training=True)
        q_out, _, _ = self.model(observations, h.to(self.device), c.to(self.device))

        # Extract Q-values for the chosen actions
        q_a = q_out.gather(2, actions)

        # Calculate the loss using the smooth L1 loss
        loss = F.smooth_l1_loss(q_a, targets)

        # Update the DRQN parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        
    
        
    def run_episode(self, i):
        self.her.reset()
        obs, done = self.env.reset()
        done = False

        state = self.env.get_tensor(obs)
        sum_r = 0
        mean_loss = mean_val()
        min_dist = 100000
        max_t = 50 
        trajectory = [obs]
        previous_action = self.previous_action

        ############################################
        trajectory = [obs]
        trajectory2 = [obs]
        ############################################
        
        for t in range(max_t):
            self.steps += 1
            self.eps = self.epsi_low + (self.epsi_high - self.epsi_low) * (np.exp(-1.0 * self.steps / self.decay))
            Q = self.model(self.norm(state.cuda()))
            num = np.random.rand()

            if (num < self.eps):
                # action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
                possible_actions = []
                if previous_action == 0:
                    possible_actions = [0, 1]
                elif previous_action == 1:
                    possible_actions = [0, 1, 2]
                elif previous_action == 2:
                    possible_actions = [1, 2, 3]
                elif previous_action == 3:
                    possible_actions = [2, 3, 4]
                elif previous_action == 4:
                    possible_actions = [3, 4]

                action = np.random.choice(possible_actions)
                action = torch.LongTensor([action])

            else:
                action = torch.argmax(Q,dim=1)

            new_obs, reward, done, dist, car_grid, crack = self.env.step(obs,action.item(),previous_action)
            previous_action = action.item()

            if not crack:
                Print = True
                trajectory2.append(car_grid)
            new_state = self.env.get_tensor(new_obs)
            sum_r = sum_r + reward
            if dist < min_dist:
                min_dist = dist
            if (t+1) == max_t:
                done = True

            self.replay_buffer.append([dc(state.squeeze(0).numpy()),dc(action),dc(reward),dc(new_state.squeeze(0).numpy()),dc(done)])
            self.her.keep([state.squeeze(0).numpy(),action,reward,new_state.squeeze(0).numpy(),done])
            loss = self.update_model()
            mean_loss.append(loss)
            state = dc(new_state)
            obs = dc(new_obs)
            
            ############################################
            trajectory.append(new_obs)
            ############################################
            
            self.step_counter = self.step_counter + 1
            if (self.step_counter > self.update_target_step):
                self.target_model.load_state_dict(self.model.state_dict())
                self.step_counter = 0
                print('updated target model')
            if done : 
                break
#####################################################################
        if i % 20 == 0:
            self.visualize_episode(trajectory, trajectory2)

        her_list = self.her.backward()

        for item in her_list:
            self.replay_buffer.append(item)

        self.log.add_item('tot_return', sum_r)
        self.log.add_item('avg_loss', mean_loss.get())
        self.log.add_item('final_dist', min_dist)
        
        # Train DRQN
        if self.memory and len(self.replay_buffer) >= self.batch_size:
            sampled_buffer, seq_len = self.sample_buffer()
            self.train_drqn(sampled_buffer)

        return self.log
        
    def gather_data(self):
        self.her.reset()
        obs, done = self.env.reset()
        done = False
        state = self.env.get_tensor(obs)
        sum_r = 0
        min_dist = 100000
        max_t = 80
        previous_action = self.previous_action

        for t in range(max_t):
            self.eps = 1.0
            Q = self.model(state.cuda())
            num = np.random.rand()
            if (num < self.eps):
                # action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
                possible_actions = []
                if previous_action == 0:
                    possible_actions = [0, 1]
                elif previous_action == 1:
                    possible_actions = [0, 1, 2]
                elif previous_action == 2:
                    possible_actions = [1, 2, 3]
                elif previous_action == 3:
                    possible_actions = [2, 3, 4]
                elif previous_action == 4:
                    possible_actions = [3, 4]

                action = np.random.choice(possible_actions)
                action = torch.LongTensor([action])
            else:
                action = torch.argmax(Q,dim=1)
            new_obs, reward, done, dist, _, _ = self.env.step(obs,action.item(),previous_action)
            previous_action = action.item()

            new_state = self.env.get_tensor(new_obs)
            sum_r = sum_r + reward
            if dist < min_dist:
                min_dist = dist
            if (t+1) == max_t:
                done = True
            
            self.replay_buffer.append([dc(state.squeeze(0).numpy()),dc(action),dc(reward),dc(new_state.squeeze(0).numpy()),dc(done)])
            state = dc(new_state)
            obs = dc(new_obs)
        return min_dist

    def calc_norm(self):
        S0, A0, R1, S1, D1 = zip(*self.replay_buffer)
        S0 = torch.tensor( S0, dtype=torch.float)
        self.image_mean = S0.mean(dim=0).cuda()
        self.image_std = S0.std(dim=0).cuda()
        
    def norm(self,state):
        return state - self.image_mean
        
    def update_model(self):
        self.optimizer.zero_grad()
        num = len(self.replay_buffer)
        K = np.min([num,self.batch_size])
        samples = random.sample(self.replay_buffer, K)
        
        S0, A0, R1, S1, D1 = zip(*samples)
        S0 = torch.tensor( S0, dtype=torch.float)
        A0 = torch.tensor( A0, dtype=torch.long).view(K, -1)
        R1 = torch.tensor( R1, dtype=torch.float).view(K, -1)
        S1 = torch.tensor( S1, dtype=torch.float)
        D1 = torch.tensor( D1, dtype=torch.float)
        
        S0 = self.norm(S0.cuda())
        S1 = self.norm(S1.cuda())
        if self.ddqn == True:
            model_next_acts = self.model(S1).detach().max(dim=1)[1]
            target_q = R1.squeeze().cuda() + self.gamma*self.target_model(S1).gather(1,model_next_acts.unsqueeze(1)).squeeze()*(1 - D1.cuda())
        else:
            target_q = R1.squeeze().cuda() + self.gamma*self.target_model(S1).max(dim=1)[0].detach()*(1 - D1.cuda())
        policy_q = self.model(S0).gather(1,A0.cuda())
        L = F.smooth_l1_loss(policy_q.squeeze(),target_q.squeeze())
        L.backward()
        self.optimizer.step()
        return L.detach().item()
    
    def run_epoch(self, i):
        self.run_episode(i)
        self.log.add_item('buffer',len(self.replay_buffer))
        return self.log


    ##################################
    plt.ion()
    def visualize_episode(self,trajectory_1, trajectory_2):
        # _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # 첫번째 모델 Visualization
        img_1 = np.zeros((40, 40, 3), dtype=np.uint8)
        img_1[trajectory_1[0][:, :, 0] == 1.0] = [255, 0, 0]  # 장애물 red 
        img_1[trajectory_1[0][:, :, 0] == 2.0] = [255, 255, 255]  # 차선 white
        img_1[trajectory_1[0][:, :, 0] == 255.0] = [255, 0, 255]  #surplus pink

        for obs in trajectory_1:
            pos = np.argwhere(obs[:, :, 1] == 10.0)[0]
            img_1[pos[0], pos[1]] = [255, 255, 0]  # 이동 경로
        img_1[pos[0], pos[1]] = [0, 255, 255]

        initial = np.argwhere(trajectory_1[0][:, :, 1] == self.env.scale)[0]
        img_1[initial[0], initial[1]] = [0, 255, 0]  # 시작 위치

        target = np.argwhere(trajectory_1[0][:, :, 2] == self.env.scale)[0]
        img_1[target[0], target[1]] = [0, 0, 255]  # 목표 위치

        plt.subplot(1, 2, 1)
        plt.imshow(img_1)
        plt.title('Model 1')

        # 차가 지나간 자리 visualization
        img_2 = np.zeros((40, 40, 3), dtype=np.uint8)
        img_2[trajectory_2[0][:, :, 0] == 1.0] = [255, 0, 0]  # 장애물 red
        img_2[trajectory_2[0][:, :, 0] == 2.0] = [255, 255, 255] # 차선 white
        img_2[trajectory_2[0][:, :, 0] == 255.0] = [255, 0, 255]  #surplus pink
        
        for cars in trajectory_2 :
            car_pos = np.where((cars[:,:,1]==255) & (cars[:,:,2]==255))
            car_pos = list(zip(car_pos[0],car_pos[1]))

            for car in car_pos :
                img_2[car[0], car[1]] = [0,255,255]
            img_2[pos[0], pos[1]] = [255, 0, 255] # 마지막위치
            
        initial = np.argwhere(trajectory_2[0][:, :, 1] == self.env.scale)[0]
        img_2[initial[0], initial[1]] = [0, 255, 0]  # 시작 위치
            
        target = np.argwhere(trajectory_2[0][:, :, 2] == self.env.scale)[0]
        img_2[target[0], target[1]] = [0, 0, 255]  # 목표 위치
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_2)
        plt.title('Model 2')

        plt.draw()
        plt.pause(0.1)
    plt.ioff()