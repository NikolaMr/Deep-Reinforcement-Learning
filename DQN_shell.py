import tensorflow as tf
import keras
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from keras.optimizers import Adam
import json
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import skimage
from skimage import color, exposure, transform

env = gym.make('PongDeterministic-v4')

ACTIONS = env.action_space.n # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 500. # timesteps to observe before training
EXPLORE = 500000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 0.5 # starting value of epsilon
REPLAY_MEMORY = 20000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
LEARNING_RATE = 1e-5
SGD_LEARNING_RATE = 1e-3

img_rows , img_cols = 84, 84
#Convert image into Black and white
img_channels = 3 #We stack 3 frames

max_epLength = 1000

NUM_EPISODES = 500

SAVE_DIR = 'dqn_pong_sh/'

import os
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def build_model():
    
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    #model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS, activation='linear'))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    #sgd = SGD(lr=SGD_LEARNING_RATE)
    #model.compile(loss='mse',optimizer=sgd)
    print("We finish building the model")
    return model
    
from skimage import data, io
from matplotlib import pyplot as plt
    
def show_as_img(arr):
    io.imshow(arr.reshape(img_rows, img_cols, 3))
    plt.show()
    
class Memory():
    def __init__(self, buff_sz):
        self.buff_sz = buff_sz
        self.M = deque()
    def append(self, tup):
        self.M.append(tup)
        if (len(self.M) > self.buff_sz):
            dump = self.M.popleft()
            if dump[2] > 0.0:
                if random.random() < 0.65:
                    self.append(dump)
            elif dump[2] < 0.0:
                if random.random() < 0.45:
                    self.append(dump)
    def sample(self, num_samples):
        minibatch = random.sample(self.M, num_samples)
        return minibatch
        #indices_random = random.randrange(0, len(self.M) - num_samples)
        #return list(self.M)[indices_random:indices_random + num_samples]

def save_model(model, path):
    model.save(path)
    
def process_frame(x_t):
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(img_cols,img_rows), mode='constant')
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    x_t = x_t.reshape((1, img_cols, img_rows, 1))
    x_t /= 255.0
    return x_t

import time

TIMESTEP = 0

def train_model(model, env):
    
    M = Memory(REPLAY_MEMORY)
 
    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

    t = 0
    
    rewards = []
    
    for idxEpisode in range(NUM_EPISODES):
        #Reset environment and get first new observation
        x_t = env.reset()
        x_t = process_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t), axis=3)
        s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[3])
        d = False
        rAll = 0
        j = 0
        loss = 0.0
        ct_non_zero_reward = 0
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            a_t = None
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < epsilon or t < OBSERVE:
                a_t = random.randrange(ACTIONS)
            else:
                q = model.predict(s_t)
                policy_max_Q = np.argmax(q)
                a_t = policy_max_Q
            x_t1,r_t,done,_ = env.step(a_t)
            x_t1 = process_frame(x_t1)
            s_t1 = np.append(x_t1, s_t[:, :, :, :2], axis=3)
            
            t += 1
            TIMESTEP = t
            M.append((s_t, a_t, r_t, s_t1, done))
            
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                minibatch = M.sample(BATCH)
                inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
                targets = np.zeros((BATCH, ACTIONS))
                for i in range(0, BATCH):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    done_t = minibatch[i][4]

                    inputs[i] = state_t
                    targets[i] = model.predict(state_t)
                    Q_sa = model.predict(state_t1)
                    if done_t:
                        #print('targets before done', targets)
                        targets[i, action_t] = reward_t
                        #print('targets after done', targets)
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa[0])#[action_t]
                    if reward_t != 0.0:
                        ct_non_zero_reward += 1
                        #print('got reward', reward_t, 'for action', action_t)
                        #print('bef', model.predict(state_t))
                        #print('pred_s1', model.predict(state_t1))
                        #print('aft', targets)
                loss += model.train_on_batch(inputs, targets)
            rAll += r_t
            s_t = s_t1
            
            if done == True:
                break
            
        rewards.append(rAll)
        
        print('episode', idxEpisode, 'length', j, 'reward', rAll, 'epsilon', epsilon, 'loss sum', loss, 'non zero rewards', ct_non_zero_reward)
        
        if idxEpisode % 25 == 0:
            path = SAVE_DIR + 'model_episode_' + str(idxEpisode) + '.h5'
            save_model(model, path)
    return rewards
    
#model = build_model()
model = keras.models.load_model('/home/nikola/Faks/Diplomski/TreciSemestar/Projekt/atari_player/dqn_pong/model_episode_50.h5')

rewards = train_model(model, env)

from matplotlib import pyplot as plt
plt.plot(range(len(rewards)), rewards)
plt.show()
