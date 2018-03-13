import tensorflow as tf
import keras
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model
from keras.engine.topology import Input
from keras.layers import Dense
from keras import initializers
from keras.optimizers import Adam
import json
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import skimage
from skimage import color, exposure, transform

import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def copy_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model')
    return keras.models.load_model('tmp_model')

def process_frame(x_t, img_rows, img_cols):
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(img_rows, img_cols), mode='constant')
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    x_t = x_t.reshape((1, img_rows, img_cols, 1))
    x_t /= 255.0
    return x_t

def save_model(model, path):
    model.save(path)

def play_game(env, model, img_dims, img_channels):
    x_t = env.reset()
    x_t = process_frame(x_t, img_dims[0], img_dims[1])
    s_t = np.stack((x_t, x_t, x_t), axis=3)
    s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[3])
    rAll = 0
    for i in range(7000):
        q = model.predict(s_t)
        policy_max_Q = np.argmax(q)
        a_t = policy_max_Q
        if np.random.rand(1) < 0.02:
            a_t = random.randrange(env.action_space.n)
        x_t1,r_t,done,_ = env.step(a_t)
        x_t1 = process_frame(x_t1, img_dims[0], img_dims[1])
        s_t1 = np.append(x_t1, s_t[:, :, :, :img_channels-1], axis=3)
        s_t = s_t1
        rAll += r_t
        if done:
            break
    return rAll

class Agent:
    def __init__(self, setup_dict=None):
        """
        setup dict contains info like this:
            'start_eps': 0.6, 
            'end_eps': 0.1, 
            'observing_frames': 30000, 
            'exploring_frames': 500000,
            'replay_memory_size': 30000,
            'replay_batch_size': 32,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'log_filename': 'filename.txt',
            'log_freq': 25,
            'saving_freq': 100,
            'img_dims': (84, 84), // rows x cols
            'num_consecutive_frames': 3, // num frames that are stored as state
            'max_ep_length': 700, // maximum episode length
            'saving_dir': 'dirname', // where will everything be saved
            'game_name': 'BreakoutDeterministic-v4',
            'update_freq': 1
        """
        if setup_dict == None:
            setup_dict = {}
            
        if not 'start_eps' in setup_dict:
            setup_dict['start_eps'] = 0.6
        if not 'end_eps' in setup_dict:
            setup_dict['end_eps'] = 0.1
        if not 'observing_frames' in setup_dict:
            setup_dict['observing_frames'] = 30000
        if not 'exploring_frames' in setup_dict:
            setup_dict['exploring_frames'] = 500000
        if not 'replay_memory_size' in setup_dict:
            setup_dict['replay_memory_size'] = 30000
        if not 'replay_batch_size' in setup_dict:
            setup_dict['replay_batch_size'] = 32
        if not 'learning_rate' in setup_dict:
            setup_dict['learning_rate'] = 1e-4
        if not 'log_freq' in setup_dict:
            setup_dict['log_freq'] = 25
        if not 'saving_freq' in setup_dict:
            setup_dict['saving_freq'] = 100
        if not 'saving_dir' in setup_dict:
            setup_dict['saving_dir'] = "AgentResults"
        if not 'img_dims' in setup_dict:
            setup_dict['img_dims'] = (84, 84)
        if not 'num_consecutive_frames' in setup_dict:
            setup_dict['num_consecutive_frames'] = 3
        if not 'max_ep_length' in setup_dict:
            setup_dict['max_ep_length'] = 700
        if not 'game_name' in setup_dict:
            setup_dict['game_name'] = 'BreakoutDeterministic-v4'
        if not 'gamma' in setup_dict:
            setup_dict['gamma'] = 0.99
        if not 'update_freq' in setup_dict:
            setup_dict['update_freq'] = 1
        if not 'log_filename' in setup_dict:
            setup_dict['log_filename'] = 'log.txt'

        self.start_eps = setup_dict['start_eps']
        self.end_eps = setup_dict['end_eps']
        self.observing_frames = setup_dict['observing_frames']
        self.exploring_frames = setup_dict['exploring_frames']
        self.replay_memory_size = setup_dict['replay_memory_size']
        self.replay_batch_size = setup_dict['replay_batch_size']
        self.learning_rate = setup_dict['learning_rate']
        self.log_freq = setup_dict['log_freq']
        self.saving_freq = setup_dict['saving_freq']
        self.saving_dir = os.path.join(setup_dict['saving_dir'], '')
        self.img_dims = setup_dict['img_dims']
        self.num_consecutive_frames = setup_dict['num_consecutive_frames']
        self.max_ep_length = setup_dict['max_ep_length']
        self.game_name = setup_dict['game_name']
        self.gamma = setup_dict['gamma']
        self.update_freq = setup_dict['update_freq']

        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        self.log_file = open(os.path.join(self.saving_dir, setup_dict['log_filename']), 'w', 1)
        self.env = gym.make(self.game_name)
        self.num_actions = self.env.action_space.n

    def build_model(self):
        raise NotImplementedError

    def get_target(self, state_t, action_t, reward_t, state_t1, done_t):
        raise NotImplementedError

    def get_batch(self):
        raise NotImplementedError

    def post_train(self):
        raise NotImplementedError

    def replay(self):
        loss = 0.0
        if self.t % (self.update_freq) == 0:
            
            minibatch = self.get_batch()
            inputs = np.zeros((self.replay_batch_size, self.img_dims[0], self.img_dims[1], self.num_consecutive_frames))
            targets = np.zeros((self.replay_batch_size, self.num_actions))
            
            for i in range(0, self.replay_batch_size):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                done_t = minibatch[i][4]

                inputs[i] = state_t
                targets[i] = self.get_target(state_t, action_t, reward_t, state_t1, done_t)
                
            loss += self.model.train_on_batch(inputs, targets)
            
            self.post_train()
        return loss

    def initialize(self):
        raise NotImplementedError

    def choose_action(self, s_t, epsilon):
        #Choose an action by greedily (with e chance of random action) from the Q-network
        if np.random.rand(1) < epsilon or self.t < self.observing_frames:
            return random.randrange(self.num_actions)
        else:
            q = self.model.predict(s_t)
            policy_max_Q = np.argmax(q)
            return policy_max_Q

    def save_transition(self, transition):
        raise NotImplementedError
        
    def train(self):
        rewards = []
        self.initialize()
        epsilon = self.start_eps

        self.t = 0
        
        idxEpisode = 0
        
        while True:
            idxEpisode += 1
            if self.t >= self.exploring_frames + self.observing_frames:
                break
            
            if idxEpisode % self.log_freq == 0 and self.log_file != None:
                rAll = play_game(self.env, self.model, self.img_dims, self.num_consecutive_frames)
                self.log_file.write(str(idxEpisode) + ' ' + str(rAll) + '\n')
                print('tested at episode', idxEpisode, 'reward is', rAll)
            
            self.cnt_rewarded = 0
            #Reset environment and get first new observation
            x_t = self.env.reset()
            x_t = process_frame(x_t, self.img_dims[0], self.img_dims[1])
            s_t = np.stack((x_t, x_t, x_t), axis=3)
            s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[3])
            rAll = 0
            j = 0
            loss = 0.0
            #The Q-Network
            while j < self.max_ep_length:
                j+=1
                a_t = self.choose_action(s_t, epsilon)

                x_t1,r_t,done,_ = self.env.step(a_t)
                
                r_t_clipped = r_t
                # reward clipping
                if r_t_clipped > 0.0:
                    r_t_clipped = 1.0
                elif r_t_clipped < 0.0:
                    r_t_clipped = -1.0
                
                x_t1 = process_frame(x_t1, self.img_dims[0], self.img_dims[1])
                s_t1 = np.append(x_t1, s_t[:, :, :, :self.num_consecutive_frames-1], axis=3)
                
                self.t += 1
                
                self.save_transition((s_t, a_t, r_t_clipped, s_t1, done)) #M.append((s_t, a_t, r_t_clipped, s_t1, done))
                
                if epsilon > self.end_eps and self.t > self.observing_frames:
                    epsilon -= (self.start_eps - self.end_eps) / self.exploring_frames
                    
                    loss += self.replay()
                rAll += r_t
                s_t = s_t1
                
                if done == True:
                    break
            rewards.append(rAll)
            print('episode', idxEpisode, 'length', j, 'reward', rAll, 'epsilon', epsilon, 'avg batch loss', (loss / j))
            print ('rewarded count', self.cnt_rewarded, '/', j * self.replay_batch_size)
            if idxEpisode % self.saving_freq == 0:
                path = self.saving_dir + 'model_episode_' + str(idxEpisode) + '.h5'
                save_model(self.model, path)
        return rewards