import tensorflow as tf
import keras
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from keras.optimizers import Adam, RMSprop
import json
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import skimage
from skimage import color, exposure, transform

env = gym.make('PongDeterministic-v4')
env.reset()

ACTIONS = env.action_space.n # number of valid actions
GAMMA = 0.9 # decay rate of past observations
OBSERVATION = 15000. # timesteps to observe before training
EXPLORE = 750000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 0.7 # starting value of epsilon
REPLAY_MEMORY = 17000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
LEARNING_RATE = 1e-3

SAVE_DIR = './backup_pong_lr-4/'

img_rows , img_cols = 80, 80
#Convert image into Black and white
#img_channels = 4 #We stack 4 frames

MODEL_NAME = "pdv4_ddqn_lr-4_tmr_100_after_500000.h5"

def build_model():
    
    initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols, 1), kernel_initializer=initializer, bias_initializer='zeros'))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same', kernel_initializer=initializer, bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', kernel_initializer=initializer, bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer=initializer, bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS, kernel_initializer=initializer, bias_initializer='zeros'))

    #adam = Adam(lr=LEARNING_RATE)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    
    #model.compile(loss='mse',optimizer=adam)
    model.compile(loss='mse',optimizer=rms)
    print("We finish building the model")
    return model
    
class Wrapped_Game:
    def __init__(self, game):
        self.game = game
        self.game.reset()
    def step(self, action):
        ns, r, d, _ = self.game.step(action)
        if d:
            self.game.reset()
            #r = - 1.0
        return ns, r, d, _
    def reset(self):
        self.game.reset()
    def render(self):
        self.game.render()
    def close(self):
        self.game.close()
        
def copy_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model')
    return keras.models.load_model('tmp_model')
    
def clear_session(model, target_model):
    from keras import backend as K
    model_path = 'tmp_model_name_ddqn'
    model.save(model_path)
    del model
    target_model_path = 'tmp_target_model_name_ddqn'
    target_model.save(target_model_path)
    del target_model
    K.clear_session()
    model = keras.models.load_model(model_path)
    target_model = keras.models.load_model(target_model_path)
    return model, target_model
    
def train_model(model, env):
    
    target_model = copy_model(model)
    
    #init replay memory
    M = deque()
 
    env.reset()
    next_state, reward, done, _ = env.step(0)

    x_t = skimage.color.rgb2gray(next_state)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    x_t = x_t.reshape((1, 80, 80, 1))
    s_t = x_t - x_t
    #s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

    prev_done = False
    
    t = 0
    while(True):
        loss = 0
        Q_sa = 0
        r_t = 0
        a_t = 0
        
        # explore
        if random.random() <= epsilon or t < OBSERVE:
            a_t = random.randrange(ACTIONS)
        # exploit
        else:
            q = model.predict(s_t)
            policy_max_Q = np.argmax(q)
            a_t = policy_max_Q
            #print(a_t)
        # move toward more exploitation
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                
        # conduct new state
        next_state, r_t, done, _ = env.step(a_t)
        #env.render()
        
        prev_done = done
        
        #env.render()
        x_t1 = skimage.color.rgb2gray(next_state)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1,out_range=(0,255))
        x_t1 = x_t1.reshape((1, 80, 80, 1))

        s_t1 = x_t1 - x_t
        
        x_t = x_t1
        
        # save in replay memory
        M.append((s_t, a_t, r_t, s_t1, done))
        if (len(M) > REPLAY_MEMORY):
            M.popleft()

        if t > OBSERVE:
            minibatch = random.sample(M, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((BATCH, ACTIONS))

            # experience replay
            for i in range(0, BATCH):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                done_t = minibatch[i][4]

                inputs[i] = state_t
                targets[i] = target_model.predict(state_t)
                # DDQN formula
                # Q-Target = r + γQ(s’,argmax(Q(s’,a,ϴ),ϴ’))
                Q_sa = target_model.predict(state_t1)
                if done_t:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * Q_sa[0][action_t]

            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t += 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            model.save_weights(SAVE_DIR + MODEL_NAME, overwrite=True)
        if t % 50000 == 0:
            print("checkpoint creation")
            model.save_weights(SAVE_DIR + str(t) + '_iters_' + MODEL_NAME, overwrite=True)
        
        if t % 1500 == 0:
            model, target_model = clear_session(model, target_model)
        if t % 200 == 0:
            target_model = copy_model(model)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        if t % 1000 == 0 or r_t > 0.0:
            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", a_t, "/ REWARD", r_t, \
                "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

def test_model(model, env):
    print ("Now we test model")
    #model.load_weights(MODEL_NAME)
    #adam = Adam(lr=LEARNING_RATE)
    #model.compile(loss='mse',optimizer=adam)
    #print ("Weight load successfully")
    env.reset()
    
    next_state, reward, done, _ = env.step(0)
    
    x_t = skimage.color.rgb2gray(next_state)
    x_t = skimage.transform.resize(x_t,(1,80,80,1))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    x_t = x_t.reshape((1, 80, 80, 1))
    
    s_t = x_t - x_t
    #s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    
    for i in range(500):
        q = model.predict(s_t)
        policy_max_Q = np.argmax(q)
        a_t = policy_max_Q
        #print(a_t)
        next_state, r_t, done, _ = env.step(a_t)
        x_t1 = skimage.color.rgb2gray(next_state)
        x_t1 = skimage.transform.resize(x_t1,(1,80,80,1))
        x_t1 = skimage.exposure.rescale_intensity(x_t1,out_range=(0,255))
        #x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        #s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        s_t1 = x_t1 - x_t

        s_t = s_t1
        x_t = x_t1
        
        next_state, reward, done, _ = env.step(a_t)
        env.render()
    env.close()
