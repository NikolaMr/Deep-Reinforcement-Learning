import tensorflow as tf
import random
import gym
import numpy as np
import skimage
from skimage import color, exposure, transform
import threading
import sys

model_path = sys.argv[1]
should_render = sys.argv[2].strip()

if should_render == 'y' or should_render == 'Y':
    should_render = True
else:
    should_render = False

IMG_WIDTH = 105
IMG_HEIGHT = 80
GLOBAL_SCOPE = 'global'
ENV_NAME = 'BreakoutDeterministic-v4'
LEARNING_RATE = 1e-4
NUM_GAMES = 5

def process_frame(x_t, img_rows, img_cols):
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(img_rows, img_cols), mode='constant')
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    x_t = x_t.reshape((1, img_rows, img_cols, 1))
    x_t /= 255.0
    return x_t

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

class EnvWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
    def reset(self):
        s = self.env.reset()
        s = process_frame(s, IMG_WIDTH, IMG_HEIGHT)
        return s
    def step(self, a):
        s1, r, d, _ = self.env.step(a)
        s1 = process_frame(s1, IMG_WIDTH, IMG_HEIGHT)
        return s1, r, d, _

class Agent:
    def __init__(self, env, scope_name, optimizer):
        self.env = env
        self.scope_name = scope_name
        self.action_size = self.env.action_space.n
        self.optimizer = optimizer

        self.__build_model()
    def __build_model(self):
        print('building model')
        with tf.variable_scope(self.scope_name):
            weights_initializer = tf.contrib.layers.xavier_initializer_conv2d()
            bias_initializer = tf.zeros_initializer()
            self.X = tf.placeholder(shape=[None, IMG_WIDTH, IMG_HEIGHT, 1], dtype=tf.float32, name='input')
            conv1 = tf.contrib.layers.conv2d(self.X, 32, 8, stride=4, activation_fn=tf.nn.relu, padding='VALID', \
                                            weights_initializer=weights_initializer, biases_initializer = bias_initializer,\
                                            scope='first_conv')
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, stride=2, activation_fn=tf.nn.relu, padding='VALID', \
                                            weights_initializer=weights_initializer, biases_initializer = bias_initializer,\
                                            scope='second_conv')
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, stride=1, activation_fn=tf.nn.relu, padding='VALID', \
                                             weights_initializer=weights_initializer, biases_initializer = bias_initializer,\
                                            scope='third_conv')
            flattened = tf.contrib.layers.flatten(conv3, scope='flatten')
            embedding = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), biases_initializer=bias_initializer,\
                                                         scope='fc_embed')

            step_size = tf.shape(self.X)[:1]

            rnn_in = tf.expand_dims(embedding, axis=0)

            LSTM_CELL_CNT = 256

            self.lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_CNT)
            self.c_in = tf.placeholder(tf.float32, [1, self.lstm.state_size.c],
            "c_in")
            self.h_in = tf.placeholder(tf.float32, [1, self.lstm.state_size.h],
            "h_in")

            self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.c_in, \
                                          self.h_in)

            output, final_state = tf.nn.dynamic_rnn(self.lstm, rnn_in,sequence_length=step_size, initial_state=self.initial_lstm_state, dtype=tf.float32)
            self.final_lstm_state = final_state
            output = tf.reshape(output, (-1, LSTM_CELL_CNT))

            self.policy = tf.contrib.layers.fully_connected(output, self.action_size, activation_fn=tf.nn.softmax, weights_initializer=tf.random_normal_initializer(stddev=0.5), biases_initializer=None,\
                                                           scope='fc_policy')
            self.value = tf.contrib.layers.fully_connected(\
                                                           output, \
                                                           1, \
                                                           activation_fn=None, \
                                                           weights_initializer=tf.random_normal_initializer(stddev=.25), \
                                                           biases_initializer=None,\
                                                          scope='fc_value')

    def predict(self, sess, state, initial_lstm_state):
        policy, final_lstm_state = sess.run((self.policy, self.final_lstm_state), \
                                            feed_dict={\
                                                       self.c_in:initial_lstm_state[0], \
                                                       self.h_in:initial_lstm_state[1], \
                                                       self.X:state\
                                                      }\
                                           )
        policy = policy.flatten()
        prediction = np.random.choice(self.action_size, p=policy)
        return prediction, final_lstm_state

    def act(self, sess, state, initial_lstm_state):
        prediction, final_lstm_state = self.predict(sess, state, initial_lstm_state)
        a = prediction
        next_state,r,d,_ = self.env.step(a)
        return state, a, r, d, next_state, final_lstm_state

    def get_value(self, sess, state, initial_lstm_state):
        return sess.run(\
                        self.value, \
                        feed_dict={ \
                                   self.c_in:initial_lstm_state[0], \
                                   self.h_in:initial_lstm_state[1], \
                                   self.X:state\
                                  } \
                       )

    def train(self, sess, states, actions, target_values, advantages, initial_lstm_state):
        gu, value_loss, policy_loss, entropy_loss, final_lstm_state = \
            sess.run((self.global_update, self.value_loss, self.policy_loss, self.entropy_loss, self.final_lstm_state), \
                     feed_dict={
                         self.X: states,
                         self.actions: actions,
                         self.target_values: target_values,
                         self.advantages: advantages,
                         self.c_in:initial_lstm_state[0], \
                         self.h_in:initial_lstm_state[1]

                     })
        return value_loss / len(states), policy_loss / len(states), entropy_loss / len(states), final_lstm_state

    def update_to_global(self, sess):
        if self.scope_name != GLOBAL_SCOPE:
            sess.run(self.update_ops)

import time

worker_threads = []

env_global = EnvWrapper(ENV_NAME)
#global_agent = Agent(env_global, GLOBAL_SCOPE, tf.train.AdamOptimizer())
global_agent = Agent(env_global, GLOBAL_SCOPE, tf.train.GradientDescentOptimizer(LEARNING_RATE))

config = tf.ConfigProto()#device_count = {'GPU': 0})
config.gpu_options.allow_growth=True

sess = tf.Session(config=config)

def test_agent_fun(test_agent):
    test_env = EnvWrapper(ENV_NAME)
    #test_agent = Agent(test_env, 'tester', optimizer)
    test_agent.update_to_global(sess)

    state_c = np.zeros((1, test_agent.lstm.state_size.c), np.float32)
    state_h = np.zeros((1, test_agent.lstm.state_size.h), np.float32)
    initial_lstm_state = [state_c, state_h]

    done = False
    state = test_env.reset()

    reward = 0

    while not done:
        policy, final_lstm_state = sess.run((test_agent.policy, test_agent.final_lstm_state), \
                                                feed_dict={\
                                                           test_agent.c_in:initial_lstm_state[0], \
                                                           test_agent.h_in:initial_lstm_state[1], \
                                                           test_agent.X:state\
                                                          }\
                                               )
        initial_lstm_state = final_lstm_state
        policy = policy.flatten()
        prediction = np.random.choice(test_env.action_space.n, p=policy)

        ns, r, d, _ = test_env.step(prediction)
        if should_render:
            test_env.env.render()
        state = ns
        reward += (r)
        done = d
    test_env.env.close()
    print('final reward is', reward)

test_env = EnvWrapper(ENV_NAME)
tester_agent = global_agent

saver = tf.train.Saver()
saver.restore(sess, model_path)

for i in range(NUM_GAMES):
    test_agent_fun(tester_agent)
