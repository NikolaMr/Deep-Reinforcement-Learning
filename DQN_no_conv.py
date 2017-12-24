import tensorflow as tf
import keras
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Reshape
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

env = gym.make('PongDeterministic-v4')

ACTIONS = env.action_space.n # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 30000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
INITIAL_EPSILON = 0.4 # starting value of epsilon
REPLAY_MEMORY = 15000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
LEARNING_RATE = 1e-4

img_rows , img_cols = 84, 84
#Convert image into Black and white
img_channels = 3 #We stack 3 frames

max_epLength = 3000

update_freq = 4

NUM_EPISODES = 5000

SAVE_DIR = 'dqn_breakout_no_conv_sh/'

import os
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0 / 3.0
set_session(tf.Session(config=config))

def build_model():
    
    initializer_1 = initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None)
    initializer_2 = initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None)
    
    model = Sequential()
    model.add(Reshape((img_rows*img_cols*img_channels, ), input_shape=(img_rows, img_cols, img_channels)))
    model.add(Dense(512, bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(1024, bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(512, bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS, kernel_initializer=initializer_2, bias_initializer='zeros'))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model
    
class Memory():
    def __init__(self, buff_sz):
        self.buff_sz = buff_sz
        self.M = deque()
    def append(self, tup):
        self.M.append(tup)
        if (len(self.M) > self.buff_sz):
            self.M.popleft()
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
    #exit()
    return x_t

def train_model(model, env):
    
    #init replay memory
    #M = deque()
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
        
        inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
        targets = np.zeros((BATCH, ACTIONS))
        
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
            M.append((s_t, a_t, r_t, s_t1, done))
            #if (len(M) > REPLAY_MEMORY):
            #    M.popleft()
            
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                #if t % (update_freq) == 0:
                #minibatch = random.sample(M, BATCH-1)
                #minibatch.append(M[-1])
                minibatch = M.sample(BATCH)
                #inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
                #targets = np.zeros((BATCH, ACTIONS))
                inputs.fill(0)# = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
                targets.fill(0)# = np.zeros((BATCH, ACTIONS))
                #Below we perform the Double-DQN update to the target Q-values
                #Q1 = model.predict(s_t1)
                #Q2 = target_model.predict(s_t1)
                #end_multiplier = -(trainBatch[:,4] - 1)
                #doubleQ = Q2[range(batch_size),Q1]
                #targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                #Update the network with our target values.
                #_ = sess.run(mainQN.updateModel, \
                #    feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})

                # experience replay
                for i in range(0, BATCH):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    done_t = minibatch[i][4]

                    inputs[i] = state_t
                    #print (inputs[i])
                    #print (state_t)
                    targets[i] = model.predict(state_t)
                    # DDQN formula
                    # Q-Target = r + γQ(s’,argmax(Q(s’,a,ϴ),ϴ’))
                    #print(targets[i].shape)
                    Q_sa = model.predict(state_t1)
                    if reward_t != 0.0:
                        ct_non_zero_reward += 1
                    if done_t:
                        targets[i, action_t] = reward_t
                    else:
                    #    print('targets before', targets[i])
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa[0])#[action_t]
                    #    print('targets', targets[i])
                    #exit()

                loss += model.train_on_batch(inputs, targets)
                #if t % 5 == 0:
                #    print('%s function took %0.3f ms at iteration %d' % ('updateTarget', (time2-time1)*1000.0, t))
                #updateTarget(targetOps,sess) #Update the target network toward the primary network.
            rAll += r_t
            s_t = s_t1
            
            if done == True:
                break
            
        rewards.append(rAll)
            
            #if t % 10000:
            #    clear_session(model, target_model)
        print('episode', idxEpisode, 'length', j, 'agv reward in last 10', np.mean(rewards[-10:]), 'epsilon', epsilon, 'avg loss', (loss / j), 'non zero rewards', ct_non_zero_reward)
        
        if idxEpisode % 10 == 0:
            print('Saving model')
            path = SAVE_DIR + 'model_episode_' + str(idxEpisode) + '.h5'
            save_model(model, path)
        
        if idxEpisode % 30 == 0:
            x_t = env.reset()
            x_t = process_frame(x_t)
            s_t = np.stack((x_t, x_t, x_t), axis=3)
            s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[3])
            for i in range(500):
                env.render()
                q = model.predict(s_t)
                policy_max_Q = np.argmax(q)
                a_t = policy_max_Q
                x_t1,r_t,done,_ = env.step(a_t)
                x_t1 = process_frame(x_t1)
                s_t1 = np.append(x_t1, s_t[:, :, :, :2], axis=3)
                s_t = s_t1
                if done:
                    break
            env.close()
            env = gym.make('PongDeterministic-v4')
            
model = build_model()
train_model(model, env)
