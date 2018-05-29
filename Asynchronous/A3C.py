import tensorflow as tf
import random
import gym
import numpy as np
import skimage
from skimage import color, exposure, transform
import threading

IMG_WIDTH = 105
IMG_HEIGHT = 80
CNT_FRAMES = 1
GLOBAL_SCOPE = 'global'
VALUE_MODIFIER = 0.5
POLICY_MODIFIER = 1
ENTROPY_MODIFIER = 5*1e-2
MAX_STEPS = 30
DISCOUNT = 0.99
ENV_NAME = 'BreakoutDeterministic-v4'
#ENV_NAME = 'PongDeterministic-v4'
#MAX_ITERATIONS = 100000000
MAX_EP_LENGTH = 1000
#MAX_LEARNING_TIME = 7 * 60 * 60 # 7 hours
LEARNING_RATE = 1e-4
CLIP_VALUE = 10.0

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
            self.X = tf.placeholder(shape=[None, IMG_WIDTH, IMG_HEIGHT, CNT_FRAMES], dtype=tf.float32, name='input')
            conv1 = tf.contrib.layers.conv2d(self.X, 32, 3, stride=2, activation_fn=tf.nn.relu, padding='SAME', \
                                            weights_initializer=weights_initializer, biases_initializer = bias_initializer,\
                                            scope='first_conv')
            mp1 = tf.contrib.layers.max_pool2d(conv1, 2, scope='first_mp')
            conv2 = tf.contrib.layers.conv2d(mp1, 32, 3, stride=2, activation_fn=tf.nn.relu, padding='SAME', \
                                            weights_initializer=weights_initializer, biases_initializer = bias_initializer,\
                                            scope='second_conv')
            mp2 = tf.contrib.layers.max_pool2d(conv2, 2, scope='second_mp')
            conv3 = tf.contrib.layers.conv2d(mp2, 64, 3, stride=2, activation_fn=tf.nn.relu, padding='SAME', \
                                             weights_initializer=weights_initializer, biases_initializer = bias_initializer,\
                                            scope='third_conv')
            flattened = tf.contrib.layers.flatten(conv3, scope='flatten')
            embedding = tf.contrib.layers.fully_connected(flattened, 256, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), biases_initializer=bias_initializer,\
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

            self.policy = tf.contrib.layers.fully_connected(embedding, self.action_size, activation_fn=tf.nn.softmax, weights_initializer=tf.random_normal_initializer(stddev=0.5), biases_initializer=None,\
                                                           scope='fc_policy')
            self.value = tf.contrib.layers.fully_connected(\
                                                           embedding, \
                                                           1, \
                                                           activation_fn=None, \
                                                           weights_initializer=tf.random_normal_initializer(stddev=.25), \
                                                           biases_initializer=None,\
                                                          scope='fc_value')

            if self.scope_name != GLOBAL_SCOPE:
                print('building agent:', self.scope_name)
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
                self.actions_oh = tf.one_hot(self.actions, depth=self.action_size, dtype=tf.float32, name='actions_oh')
                self.target_values = tf.placeholder(shape=[None], dtype=tf.float32, name='target_vals')
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')
                #print('adv shape', self.advantages.shape)
                #self.advantages = tf.subtract(tf.stop_gradient(self.value), self.target_values, name='advantage')

                MIN_POLICY = 1e-8
                MAX_POLICY = 1.0 - MIN_POLICY

                self.log_policy = tf.log(tf.clip_by_value(self.policy, MIN_POLICY, MAX_POLICY), name='log_policy')

                self.log_policy_for_action = tf.reduce_sum(tf.multiply(self.log_policy, self.actions_oh), axis=1, name='log_policy_for_action')
                self.value_loss = tf.reduce_mean(tf.square(self.value - self.target_values), name='value_loss')
                self.value_loss = self.value_loss * VALUE_MODIFIER
                #self.value_loss = self.value_loss - self.value_loss
                self.policy_loss = -tf.reduce_mean(tf.multiply(self.log_policy_for_action, self.advantages), name='policy_loss')
                self.policy_loss = self.policy_loss * POLICY_MODIFIER
                #entropija je E[-log(X)] = sum(p(x) * log(x))
                self.entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(ENTROPY_MODIFIER), trainable=False)
                self.entropy_loss = -tf.reduce_mean(self.policy * -self.log_policy, name='entropy_loss')
                self.entropy_loss = self.entropy_loss * self.entropy_beta
                #self.entropy_loss = self.entropy_loss - self.entropy_loss
                self.loss = self.value_loss + \
                            self.policy_loss + \
                            self.entropy_loss
                #get locals
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name)
                #update locals
                grads = tf.gradients(self.loss, local_vars)
                #grads = [tf.clip_by_average_norm(grad, CLIP_VALUE) for grad in grads]
                grads, grad_norms = tf.clip_by_global_norm(grads, CLIP_VALUE)
                self.update_ops = update_target_graph(GLOBAL_SCOPE, self.scope_name)
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_SCOPE)
                capped_gvs = [(grad, var) for grad, var in zip(grads, global_vars)]
                self.global_update = self.optimizer.apply_gradients(capped_gvs)

    def predict(self, sess, state, initial_lstm_state):
        policy, final_lstm_state = sess.run((self.policy, self.final_lstm_state), \
                                            feed_dict={\
                                                       self.c_in:initial_lstm_state[0], \
                                                       self.h_in:initial_lstm_state[1], \
                                                       self.X:state\
                                                      }\
                                           )
        policy = policy.flatten()
        #print('cur policy', policy)
        prediction = np.random.choice(self.action_size, p=policy)
        #prediction = np.argmax(policy)
        #print('prediction', prediction)
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

global_counter = 0

start_time = time.time()

class Worker:
    def __init__(self, agent):
        self.agent = agent
        self.summary_writer = tf.summary.FileWriter(self.agent.scope_name)
    def work(self, sess, optimizer, thread_lock):

        global global_counter
        global start_time

        print('worker starting agent:', self.agent.scope_name)
        done = True
        s = None
        episode_reward = 0
        timestep = 0
        episode_counter = 0
        value_losses = []
        policy_losses = []
        entropy_losses = []
        last_rewards = []
        last_frames = []
        last_values = []
        last_advantages = []

        last_final_lstm_state = None

        elapsed_time = time.time() - start_time

        with sess.as_default(), sess.graph.as_default():
            while True:#global_counter <= MAX_ITERATIONS and elapsed_time <= MAX_LEARNING_TIME:
                self.agent.update_to_global(sess)
                if done or timestep > MAX_EP_LENGTH:
                    last_rewards.append(episode_reward)
                    last_frames.append(timestep)
                    if episode_counter > 0 and episode_counter % 5 == 0:
                        #print('for agent:', self.agent.scope_name)
                        #print('at episode', episode_counter, 'episode reward is', episode_reward)
                        if len(value_losses) > 0:
                            summary = tf.Summary()

                            summary.value.add(tag='Performance/Reward', simple_value=float(sum(last_rewards) / len(last_rewards)))
                            summary.value.add(tag='Performance/Length', simple_value=float(sum(last_frames) / len(last_frames)))
                            summary.value.add(tag='Performance/Values mean', simple_value=float(sum(last_values) / len(last_values)))
                            summary.value.add(tag='Performance/Advantage mean', simple_value=float(sum(last_advantages) / len(last_advantages)))
                            summary.value.add(tag='Losses/Value Loss', simple_value=float(sum(value_losses) / len(value_losses)))
                            summary.value.add(tag='Losses/Policy Loss', simple_value=float(sum(policy_losses) / len(policy_losses)))
                            summary.value.add(tag='Losses/Entropy', simple_value=float(sum(entropy_losses) / len(entropy_losses)))

                            self.summary_writer.add_summary(summary, episode_counter)

                            self.summary_writer.flush()

                            last_rewards = []
                            last_frames = []
                            value_losses = []
                            policy_losses = []
                            entropy_losses = []
                            last_values = []
                            last_advantages = []
                    s = self.agent.env.reset()
                    done = False
                    episode_reward = 0
                    timestep = 0
                    episode_counter += 1

                states = []
                actions = []
                rewards = []
                values = []
                advantages = []
                target_values = []

                initial_lstm_state = last_final_lstm_state

                if last_final_lstm_state == None:
                        state_c = np.zeros((1, self.agent.lstm.state_size.c), np.float32)
                        state_h = np.zeros((1, self.agent.lstm.state_size.h), np.float32)
                        initial_lstm_state = [state_c, state_h]

                while len(states) < MAX_STEPS and not done:
                    val = self.agent.get_value(sess, s, initial_lstm_state)
                    s, a, r, d, ns, initial_lstm_state = self.agent.act(sess, s, initial_lstm_state)
                    states.append(s)
                    actions.append(a)
                    rewards.append(r)
                    done = d
                    last_values.append(val)
                    values.append(val)
                    with thread_lock:
                        global_counter += 1
                    episode_reward += r
                    timestep += 1
                    s = ns

                R = 0
                if not done:
                    R = last_values[-1]#self.agent.get_value(sess, s, initial_lstm_state)

                advantages = [0 for i in range(len(values))]

                for i in range(len(rewards)):
                    idx = len(rewards) - 1 - i
                    reward = rewards[idx]
                    R += DISCOUNT * reward
                    advantage = (R - values[idx])
                    advantages[idx] = advantage
                    last_advantages.append(advantage)

                target_value = 0

                if not done:
                    target_value = last_values[-1]

                for reward in reversed(rewards):
                    target_value = reward + DISCOUNT * target_value
                    target_values.append(target_value)
                #for i in range(len(rewards)-1):
                #    idx = len(rewards) - i - 1
                #    target_values[idx-1] = rewards[idx-1] + DISCOUNT * target_values[idx]
                states = np.vstack(states)
                actions = np.vstack(actions).ravel()
                target_values = np.vstack(target_values).ravel()
                advantages = np.vstack(advantages).ravel()

                if last_final_lstm_state == None:
                    state_c = np.zeros((1, self.agent.lstm.state_size.c), np.float32)
                    state_h = np.zeros((1, self.agent.lstm.state_size.h), np.float32)
                    last_final_lstm_state = [state_c, state_h]

                value_loss, policy_loss, entropy_loss, last_final_lstm_state = \
                    self.agent.train(sess, states, actions, target_values, advantages, last_final_lstm_state)

                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
                entropy_losses.append(entropy_loss)

                if done:
                    last_final_lstm_state = None

                elapsed_time = time.time() - start_time

import time

worker_threads = []

env_global = EnvWrapper(ENV_NAME)
#global_agent = Agent(env_global, GLOBAL_SCOPE, tf.train.AdamOptimizer())
global_agent = Agent(env_global, GLOBAL_SCOPE, tf.train.GradientDescentOptimizer(LEARNING_RATE))

config = tf.ConfigProto()#device_count = {'GPU': 0})
config.gpu_options.allow_growth=True

sess = tf.Session(config=config)

def global_saving_thread(agent, sess):

    global global_counter

    MAX_MODELS = 3
    cnt_model = 0

    with sess.as_default(), sess.graph.as_default():

        saver = tf.train.Saver()

        elapsed_time = time.time() - start_time

        #save model every 15 minutes
        while True:#global_counter <= MAX_ITERATIONS and elapsed_time <= MAX_LEARNING_TIME:
            print("Current model save name:", 'model_' + str(cnt_model % MAX_MODELS))
            save_path = saver.save(sess, "models/model_" + str(cnt_model % MAX_MODELS) + ".ckpt")
            print("Current global iteration", global_counter)
            cnt_model += 1
            time.sleep(3 * 60 * 60)
        print("Learning time was", int(elapsed_time/60/60), "hours", int((elapsed_time - int(elapsed_time/60/60)*60*60)/60), "minutes")

cnt_threads = 24
thread_lock = threading.Lock()

def worker_fun(worker, sess, optimizer, thread_lock):
    worker.work(sess, optimizer, thread_lock)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

for i in range(cnt_threads):
    env = EnvWrapper(ENV_NAME)
    worker = Worker(Agent(env, 'local' + str(i), optimizer))
    t = threading.Thread(target=worker_fun, args=(worker, sess, optimizer, thread_lock))
    worker_threads.append(t)
    time.sleep(0.05)

sess.run(tf.global_variables_initializer())
for t in worker_threads:
    t.start()
    time.sleep(0.05)

global_t = threading.Thread(target=global_saving_thread, args=(global_agent, sess))

worker_threads.append(global_t)
global_t.start()

for t in worker_threads:
    t.join()
