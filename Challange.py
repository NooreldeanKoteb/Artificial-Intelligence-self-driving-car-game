# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:21:57 2020

@author: Nooreldean Koteb
"""

#AI for self driving car

#Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


#Creating the architecture of the Neural Network
class Network(nn.Module):
    
    #Initializing class
    def __init__(self, input_size, nb_action):
        #Inherit nn.Module
        super(Network, self).__init__()
        
        #Number of input neurons
        self.input_Size = input_size
        #Number of output neurons
        self.nb_action = nb_action
        
        #Full connections
        #First full connection
        #(second input is how many neurons in hidden layer), this can be tuned
        self.fc1 = nn.Linear(input_size, 60)
        #Second full connection
        self.fc2 = nn.Linear(60, 30)

        #Third
        #Hidden layer to output layer
        self.fc3 = nn.Linear(30, nb_action)
        
    #Forward propagation
    def forward(self, state):
        #Forward propagation
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values

        
#Implementing experiance replay
class ReplayMemory(object):

    #Initializing class
    #How many last transitions to use for replay
    def __init__(self, capacity):
        self.capacity = capacity
        #List of memory with capcity number of memories
        self.memory = []
    
    #Appends capacity amount of memories to memory
    #event = event to be appended
    def push(self, event):
        self.memory.append(event)
        
        #If surpassed capacity, will clear the first element in memory
        if len(self.memory) > self.capacity:
            #Deletes first memory element
            del self.memory[0]
    
    #Takes random samples from memory
    def sample(self, batch_size):
        #Taking the random samples and reshaping the list
        samples = zip(*random.sample(self.memory, batch_size))
        
        #Lambda is a function that will transform samples into Variable format
        #and concatenate it with the first dimension
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


#Implementing Deep Q-Learning Model
class Dqn():
    
    #Initializing class
    def __init__(self, input_size, nb_action, gamma):
       self.gamma = gamma
       #Mean of last 100 rewards
       self.reward_window = []
       #Neural Netowrk class we made
       self.model = Network(input_size, nb_action)
       #Memory class we made (100,000 transitions the model will use to learn), can be changed
       self.memory = ReplayMemory(100000)
       #Optimizer, can be tuned
       self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
       #Last state
       self.last_state = torch.Tensor(input_size).unsqueeze(0)
       #Last action
       self.last_action = 0
       #Last reward
       self.last_reward = 0
       
    #Select Action method
    def select_action(self, state):
        #Using softmax function to calculate probabilities
        #Convert state from torch tensor to torch variable without the gradient (this saves memory)
        #Higher temperature means higher probability the winning q-value will be choosen
        probs = F.softmax(self.model(Variable(state, volatile = True))*90) #T = 80 (temperature)
        
        #Random draw from distribution to get final action
        action = probs.multinomial(num_samples = 1) #Had to put 1 for it to work (maybe should be capacity)
        
        return action.data[0, 0]
    
    #Training method
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #Get best action to play for each batch
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        #Maximum of the q-values of the next state
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        #Target
        target = self.gamma*next_outputs + batch_reward
        
        #Temporal difference
        #(predictions, target)
        td_loss = F.smooth_l1_loss(outputs, target)
        
        #Backpropagation
        #Reinitialize in every loop
        self.optimizer.zero_grad()
        
        #backpropagate through network
        #retain_variables improves memory usage
        td_loss.backward(retain_graph = True)#retain_variables = True)
        
        #Update weights
        self.optimizer.step()
        
    #Update variables when in a new state method
    def update(self, reward, new_signal):
        #Convert new_signal into a torch tensor
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        
        #Appending to memory
        self.memory.push((self.last_state, new_state, 
                          torch.LongTensor([int(self.last_action)]), #Convert simple Int number to torch Tensor
                          torch.Tensor([self.last_reward])))     #Convert simple float number to torch Tensor
        
        #Play the action
        action = self.select_action(new_state)
        
        #If we have more than 100 memories we can learn, 100 can change
        if len(self.memory.memory) > 100:
            #Get the batches from memory (100) this can change
            batch_state, batch_next_state, batch_action, batch_reward  = self.memory.sample(100)
            
            #Use learn method
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        #Update last action
        self.last_action = action
        #Update last state
        self.last_state = new_state
        #Update last reward
        self.last_reward = reward
        
        #Update reward window
        self.reward_window.append(reward)
        #Makes sure reward window is fixed to 1000, can be changed
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        return action
   
    #Score method
    def score(self):
        #Mean of the sum of rewards in the reward window (+1 to avoid crash if 0)
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    #Save method
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),    #Saves model
                    'optimizer': self.optimizer.state_dict(), #Saves optimizer
                    }, 'last_brain.pth')                      #Saves to this file
        print('Done!')
        
    #Load method
    def load(self):
        #Getting file with saved values
        if os.path.isfile('last_brain.pth'):
            print('=> Loading Checkpoint...')
            #Loading data from file
            checkpoint = torch.load('last_brain.pth')            
            #Loading model parameters
            self.model.load_state_dict(checkpoint['state_dict'])
            #Loading optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Done!')
        else:
            print('No Checkpoint Found...')