#run the script by giving it path to model and game to play
import sys

model_path = sys.argv[1]
game_name = sys.argv[2]

import os
if not os.path.exists(model_path):
    print('no model present')
    exit()

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
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Conv2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import skimage
from skimage import color, exposure, transform

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model = keras.models.load_model(model_path)
input_shape = model.get_input_shape_at(0)
img_width = input_shape[1]
img_height = input_shape[2]
num_consecutive_frames = input_shape[3]
env = gym.make(game_name)
ACTIONS = env.action_space.n # number of valid actions

def process_frame(x_t):
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(img_width,img_height), mode='constant')
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    x_t = x_t.reshape((1, img_width, img_height, 1))
    x_t /= 255.0
    return x_t

def play_game():
    EPSILON = 0.01
    x_t = env.reset()
    x_t = process_frame(x_t)
    s_t = np.stack((x_t, x_t, x_t), axis=3)
    s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[3])
    rAll = 0
    i = 0
    for _ in range(7000):
        i +=1
        env.render()
        q = model.predict(s_t)
        #print(q)
        policy_max_Q = np.argmax(q)
        a_t = policy_max_Q
        if np.random.rand(1) < EPSILON:
                a_t = random.randrange(ACTIONS)
        x_t1,r_t,done,_ = env.step(a_t)
        x_t1 = process_frame(x_t1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :num_consecutive_frames-1], axis=3)
        s_t = s_t1
        rAll += r_t
        if done:
            break
    env.close()
    print('steps', i)
    return rAll
rAll = play_game()
print('Final reward is', rAll)
