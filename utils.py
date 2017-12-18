import tensorflow as tf
import keras
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import json
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import skimage
from skimage import color, exposure, transform

ACTIONS = 6 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 100000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 20000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
LEARNING_RATE = 1e-4

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def build_model():
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels), kernel_initializer='random_normal', bias_initializer='zeros'))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same', kernel_initializer='random_normal', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', kernel_initializer='random_normal', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='random_normal', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def test_model(model, env):
    print ("Now we load weight")
    model.load_weights("model_atari.h5")
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print ("Weight load successfully")
    env.reset()
    
    next_state, reward, done, _ = env.step(0)
    
    x_t = skimage.color.rgb2gray(next_state)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    
    while(not done):
        q = model.predict(s_t)
        policy_max_Q = np.argmax(q)
        a_t = policy_max_Q
        next_state, r_t, done, _ = env.step(a_t)
        x_t1 = skimage.color.rgb2gray(next_state)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1,out_range=(0,255))
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        s_t = s_t1
        
        next_state, reward, done, _ = env.step(a_t)
        env.render()
    env.close()
