from SumTree import SumTree

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

GAME_NAME = 'BreakoutDeterministic-v4'

env = gym.make(GAME_NAME)

ACTIONS = env.action_space.n # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 20000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 0.6 # starting value of epsilon
REPLAY_MEMORY = 20000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
LEARNING_RATE = 1e-4
SAVING_FREQ = 100 # save model every 100 iterations

LOGGING_FREQ = 25

img_rows , img_cols = 84, 84
#Convert image into Black and white
img_channels = 3 #We stack 3 frames

max_epLength = 700

update_freq = 1

#NUM_EPISODES = 1700+1

SAVE_DIR = 'dueling_ddqn_PER_breakout_logged_5e5_iters_sh/'

import os
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

tau = 0.01

EPS = 0.1
ALFA = 0.8

def build_model():
    
    
    inputs = Input(shape=(img_rows,img_cols,img_channels))
    model = Conv2D(32, (8, 8), strides=(4, 4), padding='valid')(inputs)
    model = Activation('relu')(model)
    model = Conv2D(64, (4, 4), strides=(2, 2), padding='valid')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(model)
    model = Activation('relu')(model)
    
    stream = Flatten()(model)
    
    advantage = Dense(ACTIONS)(stream)
#    print(advantage)
    value = Dense(1)(stream)
    
    def mean(x):
        import keras.backend
        res = keras.backend.mean(x, axis=1, keepdims=True)
        return res
    
#    print(mean(advantage))
    meanRes = Lambda(function=mean)(advantage)
    
    from keras.layers import Concatenate
    concatenations = []
    for i in range(ACTIONS):
        concatenations.append(meanRes)
    meanRes = Concatenate()(concatenations)
#    print(meanRes)
    
    advantage = keras.layers.subtract([advantage, meanRes])
#    print(dir(advantage))
    
    qOut = keras.layers.add([value, advantage])

    model = Model(inputs=inputs, outputs=qOut)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    #sgd = SGD(lr=SGD_LEARNING_RATE)
    #model.compile(loss='mse',optimizer=sgd)
    print("We finish building the model")
    return model
    
def copy_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model')
    return keras.models.load_model('tmp_model')
    
def assign_linear_comb(m, tm, tau):
    import tensorflow as tf
    from keras import backend as K
    '''Sets the value of a tensor variable,
    from a Numpy array.
    '''
    #tf.assign(x, np.asarray(value)).op.run(session=get_session())
    assign_op = tm.assign(m.value() * tau + (1-tau) * tm.value())
    #K.get_session().run(assign_op, feed_dict={assign_placeholder: value})
    return assign_op

def update_target_graph(target_model, model, tau):
    var_assign_ops = []
    for idxLayer in range(len(model.layers)):
        model_layer = model.layers[idxLayer]
        target_model_layer = target_model.layers[idxLayer]
        for idxWeight in range(len(model_layer.weights)):
            var_assign_ops.append(
                assign_linear_comb(model_layer.weights[idxWeight], target_model_layer.weights[idxWeight], tau)
            )
    return var_assign_ops
    
def update_target(var_assign_ops):
    from keras import backend as K
    for var_assign_op in var_assign_ops:
        K.get_session().run(var_assign_op)

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

class Memory():
    def __init__(self, buff_sz, eps, alfa):
        self.M = SumTree(buff_sz)
        self.eps = eps
        self.alfa = alfa
    def append(self, tup, error):
        priority = (error + self.eps) ** self.alfa
        self.M.add(priority, tup)
    def sample(self, num_samples):
        total = self.M.total()
        s_arr = np.random.uniform(0.0, total, num_samples)
        samples = []
        for i in range(len(s_arr)):
            samples.append(self.M.get(s_arr[i]))
        return samples

def play_game():
    x_t = env.reset()
    x_t = process_frame(x_t)
    s_t = np.stack((x_t, x_t, x_t), axis=3)
    s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[3])
    rAll = 0
    for i in range(7000):
        #env.render()
        q = model.predict(s_t)
        #print(q)
        policy_max_Q = np.argmax(q)
        a_t = policy_max_Q
        if np.random.rand(1) < 0.02:
            a_t = random.randrange(ACTIONS)
        x_t1,r_t,done,_ = env.step(a_t)
        x_t1 = process_frame(x_t1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :img_channels-1], axis=3)
        s_t = s_t1
        rAll += r_t
        if done:
            break
    return rAll

def train_model(model, env, log_file=None):
    
    target_model = copy_model(model)
    var_assign_ops = update_target_graph(target_model, model, tau)
    
    #init replay memory
    M = Memory(REPLAY_MEMORY, EPS, ALFA)
 
    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

    t = 0
    
    rewards = []
    
    idxEpisode = 0
    
    while True:
        idxEpisode += 1
        if t >= EXPLORE + OBSERVE:
            break
        
        if idxEpisode % LOGGING_FREQ == 0 and log_file != None:
            rAll = play_game()
            log_file.write(str(idxEpisode) + ' ' + str(rAll) + '\n')
            print('tested at episode', idxEpisode, 'reward is', rAll)
        
        #Reset environment and get first new observation
        x_t = env.reset()
        x_t = process_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t), axis=3)
        s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[3])
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
            
            # reward clipping
            if r_t > 0.0:
                r_t = 1.0
            elif r_t < 0.0:
                r_t = -1.0
            x_t1 = process_frame(x_t1)
            s_t1 = np.append(x_t1, s_t[:, :, :, :img_channels-1], axis=3)
            
            t += 1
            TIMESTEP = t
            error = 0.0
            if t >= OBSERVE:
                q = model.predict(s_t)
                policy_max_Q = np.argmax(q)
                a_t = policy_max_Q
                q_t1 = r_t + np.max(model.predict(s_t1)[0])
                error = abs(q[0][a_t] - q_t1)
            else:
                error = r_t

            M.append((s_t, a_t, r_t, s_t1, done), error)
            
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                
                if t % (update_freq) == 0:
                    minibatch = M.sample(BATCH)
                    inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
                    targets = np.zeros((BATCH, ACTIONS))
                    
                    #print(minibatch)
                    # experience replay
                    for i in range(0, BATCH):
                        state_t = minibatch[i][0]
                        action_t = minibatch[i][1]
                        reward_t = minibatch[i][2]
                        state_t1 = minibatch[i][3]
                        done_t = minibatch[i][4]

                        inputs[i] = state_t
                        targets[i] = model.predict(state_t)
                        # DDQN formula
                        # Q-Target = r + γQ(s’,argmax(Q(s’,a,ϴ),ϴ’))
                        Q_sa = target_model.predict(state_t1)
                        if done_t:
                            targets[i, action_t] = reward_t
                        else:
                            targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa[0])#[action_t]
                        if reward_t != 0.0:
                            ct_non_zero_reward += 1

                    loss += model.train_on_batch(inputs, targets)
                    update_target(var_assign_ops)
            rAll += r_t
            s_t = s_t1
            
            if done == True:
                break
        rewards.append(rAll)
        
        print('episode', idxEpisode, 'length', j, 'reward', rAll, 'epsilon', epsilon, 'loss sum', loss, 'non zero rewards', ct_non_zero_reward)
        
        if idxEpisode % SAVING_FREQ == 0:
            path = SAVE_DIR + 'model_episode_' + str(idxEpisode) + '.h5'
            save_model(model, path)
    return rewards
        
log_file = open(GAME_NAME +'_logged.txt', 'w', 1)
model = build_model()
#model = keras.models.load_model('/home/nikola/Faks/Diplomski/TreciSemestar/Projekt/atari_player/breakout_best.h5')
rewards = train_model(model, env, log_file)
log_file.close()

from matplotlib import pyplot as plt
plt.plot(range(len(rewards)), rewards)
plt.show()
