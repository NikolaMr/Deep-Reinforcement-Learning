import Agent

import numpy as np
import keras
from keras.models import Model
from keras.engine.topology import Input
from keras.layers import Dense
from keras import initializers
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import Utilities

class DQN_Agent(Agent.Agent):
    def __init__(self, setup_dict=None):
        super(DQN_Agent, self).__init__(setup_dict)

    def post_train(self):
        super(DQN_Agent, self).post_train()

        for i in range(len(self.samples)):
            sample = self.samples[i]

            state_t = sample[0]
            action_t = sample[1]
            reward_t = sample[2]
            state_t1 = sample[3]
            done_t = sample[4]

            if reward_t != 0.0:
                self.cnt_rewarded += 1

    def get_target(self, state_t, action_t, reward_t, state_t1, done_t):
        targets = self.model.predict(state_t)
        Q_sa = self.model.predict(state_t1)
        if done_t:
            targets[0, action_t] = reward_t
        else:
            targets[0, action_t] = reward_t + self.gamma * np.max(Q_sa[0])
        return targets

    def build_model(self):
        inputs = Input(shape=(self.img_dims[0],self.img_dims[1],self.num_consecutive_frames))
        model = Conv2D(32, (8, 8), strides=(4, 4), padding='valid')(inputs)
        model = Activation('relu')(model)
        model = Conv2D(64, (4, 4), strides=(2, 2), padding='valid')(model)
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(model)
        model = Activation('relu')(model)
        model = Flatten()(model)
        model = Dense(512)(model)
        model = Activation('relu')(model)
        model = Dense(self.num_actions)(model)

        model = Model(inputs=inputs, outputs=model)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)

        print("We finish building the model")
        self.model = model
