import Agent

from SumTree import SumTree

import numpy as np
import keras
from keras.models import Model
from keras.engine.topology import Input
from keras.layers import Dense
from keras import initializers
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam

class MemoryPrioritizedForgetting():
    def __init__(self, buff_sz):
        self.buff_sz = buff_sz
        self.M = np.zeros(buff_sz, dtype=object)
        self.write = 0
    def append(self, tup):
        self.M[self.write] = tup
        self.write += 1
        self.write %= self.buff_sz
        if (len(self.M) > self.buff_sz):
            dump = self.M.popleft()
            if dump[2] > 0.0:
                if random.random() < 0.7:
                    self.append(dump)
            elif dump[2] < 0.0:
                if random.random() < 0.4:
                    self.append(dump)
    def sample(self, num_samples):
        #minibatch = random.sample(self.M, num_samples)
        indices = np.random.randint(0,self.M.shape[0],num_samples)
        return self.M[indices]

def assign_linear_comb(m, tm, tau):
    import tensorflow as tf
    from keras import backend as K
    '''Sets the value of a tensor variable,
    from a Numpy array.
    '''
    assign_op = tm.assign(m.value() * tau + (1-tau) * tm.value())
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

class Dueling_DDQN_Agent(Agent.Agent):
    def __init__(self, setup_dict=None):
        super(Dueling_DDQN_Agent, self).__init__(setup_dict)
        if setup_dict == None:
            setup_dict = {}
        if not 'tau' in setup_dict:
            setup_dict['tau'] = 0.01
        self.tau = setup_dict['tau']

    def get_batch(self):
        return self.memory.sample(self.replay_batch_size)

    def post_train(self):
        update_target(self.var_assign_ops)

    def save_transition(self, transition):
        self.memory.append(transition)

    def initialize(self):
        self.memory = MemoryPrioritizedForgetting(self.replay_memory_size)
        self.build_model()
        self.var_assign_ops = update_target_graph(self.target_model, self.model, self.tau)

    def get_target(self, state_t, action_t, reward_t, state_t1, done_t):
        targets = self.model.predict(state_t)
        # DDQN formula
        # Q-Target = r + γQ(s’,argmax(Q(s’,a,ϴ),ϴ’))
        Q_sa = self.target_model.predict(state_t1)
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
        
        stream = Flatten()(model)
        
        advantage = Dense(self.num_actions)(stream)
        value = Dense(1)(stream)
        
        def mean(x):
            import keras.backend
            res = keras.backend.mean(x, axis=1, keepdims=True)
            return res
        
        meanRes = Lambda(function=mean)(advantage)
        
        from keras.layers import Concatenate
        concatenations = []
        for i in range(self.num_actions):
            concatenations.append(meanRes)
        meanRes = Concatenate()(concatenations)

        advantage = keras.layers.subtract([advantage, meanRes])
        qOut = keras.layers.add([value, advantage])

        model = Model(inputs=inputs, outputs=qOut)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        #sgd = SGD(lr=SGD_LEARNING_RATE)
        #model.compile(loss='mse',optimizer=sgd)
        print("We finish building the model")
        self.model = model
        self.target_model = Agent.copy_model(self.model)