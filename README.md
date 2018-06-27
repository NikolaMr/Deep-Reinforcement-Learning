# Deep reinforcement learning

This is a deep reinforcement learning project that was made as a part of master thesis.
It implements following deep reinforcement learning algorithms: Deep Q Network (DQN), Double Deep Q Network (DDQN), Dueling Double Deep Q Network (DDDQN) and Asynchronous advantage actor-critic (A3C).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

In order to start this project the following requirements need to be installed.
- Python 3.4
- Keras
- Tensorflow-GPU (CPU version can be installed and it will work but it will be too slow to experience well enough results in this lifetime)
- scikit-image
- Numpy
- OpenAI Gym (with support for Atari games)

### Installing

In order to install GPU version of Tensorflow run:
```
sudo pip3 install tensorflow-gpu
```
For more information about the installation procedure read the official Tensorflow documentation available at https://www.tensorflow.org/install/.

In order to install Keras run:
```
sudo pip3 install keras
```

In order to install scikit-image run:
```
sudo pip3 install scikit-image
```

In order to install OpenAI Gym with Atari games run:
```
git clone https://github.com/openai/gym.git
cd gym
sudo pip3 install -e .
sudo pip3 install -e '.[atari]' # make sure you have cmake installed
```
For more information read the official documentation available at https://github.com/openai/gym.

## Running the system

Next two sections will explain how to run value based and policy based algorithms.

### Value based algorithms

In this section will be explained how to setup the value based algorithms (DQN, DDQN, DDDQN) to work.

Value based algorithms can be trained very easily. Training can be done in one of two ways. First way is to use 'config.json' file which stores all the configuration data that will be explained later. Second way to do it is by programmatically setting up configuration data inside AgentRunner.py script.

Configuration data that can be setup is:

- __start_eps__: starting epsilon value for epsilon greedy exploration strategy (original papers recommend 1.0)
- __end_eps__: final epsilon value for epsilon greedy exploration strategy (original papers recommend 0.1)
- __observing_frames__: number of frames to observe without any learning done
- __exploring_frames__: number of frames to perform learning
- __replay_memory_size__: size of the experience buffer (must be <= observing_frames) (30000 should be enough but the bigger the better)
- __replay_batch_size__: number of experiences to consider in one train batch (original papers recommend 32)
- __learning_rate__: learning rate of the AdamOptimizer (recommended value is 1e-4)
- __log_freq__: frequency of testing the agent (log_freq=10 means that agent will be tested every 10 learning epsiodes - testing is done by letting the agent play 5 consecutive games and the average reward and episode length are recorded)
- __saving_freq__: frequency of saving model parameters
- __saving_dir__: directory in which should logs and models be stored
- __img_width__: width of input image (original papers recommend 84)
- __img_height__: height of input image (original papers recommend 84)
- __num_consecutive_frames__: number of consecutive frames to stack in order to form one input to the neural network (num_consecutive_frames=3 means to use last 3 frames as a state representation => then the input to neural network is WxHx3) (original papers recommend 4)
- __max_ep_length__: maximum episode length
- __game_name__: name of the game to learn (only games that give image as a state representation are supported)
- __gamma__: reward decay factor
- __update_freq__: frequency at which to update target network (used in DDQN and DDDQN algorithms)
- __log_filename__: where to save logging file
- __MemoryType__: which memory to use (supported values are ExperienceReplayMemory, MemoryPrioritizedForgetting and PrioritizedExperienceReplayMemory)
- __PEREps__: epsilon parameter in prioritized experience replay memory
- __PERAlfa__: alfa parameter in prioritized experience replay memory
- __ExplorationStrategy__: which explorations strategy to use (supported values are EpsilonGreedyExplorationStrategy and BoltzmannExplorationStrategy)
- __tau__: parameter that describes how fast will the target network update it's values to the primary network (parameters of target network $\theta_{t}$ are updated to the parameters of primary network $\theta_{p}$ like this $\theta_{t}=\tau*\theta_{p} + (1-\tau)*\theta_{t})$

After setting things either by script AgentRunner.py or by configuration file 'config.json' training can be done by running the AgentRunner.py script like this:
```
python3 AgentRunner.py
```
Warning!: If you have setup parameters both inside 'config.json' and inside AgentRunner.py script, setup entries that are defined in the script will be used.

In order to see how the agent plays the game just start the TestAgent.py script and give it path to the model and the game you want it to play. For instance, if you want to load model model_episode2300.h5 with game BreakoutDeterministic-v4 you can start it like this:
```
python3 TestAgent.py model_episode2300.h5 BreakoutDeterministic-v4
```

Models will be saved into the __saving_dir__. Also, there will be a Tensorboard record in the Tensorboard folder that will keep record of the value of loss, average episode length and average episode reward.

In order to start Tensorboard all you need to do is to run:
```
cd <__saving_dir__>
cd Tensorboard
tensorboard --logdir='Tensorboard':Tensorboard
```

### Policy based algorithm

In this section will be explained how to setup the policy based algorithm (A3C) to work.

Since this version uses pure Tensorflow instead of Keras and is asynchoronous it is not integrated into the framework that was made for value based algorithms.

There are two versions of the A3C algorithm implemented in this repository. The one with LSTM and the one without it.

A3C algorithm is located in the Asynchronous folder. In order to configure parameters of A3C algorithm one must configure it inside the A3C.py script (LSTM version) or inside the A3C_no_lstm.py script (version without LSTM).

Configurable parameters are:

- __IMG_WIDTH__: width of input image
- __IMG_HEIGHT__: height of input image
- __CNT_FRAMES__: number of consecutive frames to form the state of the environment (this parameter is not available in the LSTM version)
- __GLOBAL_SCOPE__: name of the global scope
- __VALUE_MODIFIER__: value of scale for value loss
- __POLICY_MODIFIER__: value of scale for policy loss
- __ENTROPY_MODIFIER__: value of scale for entropy loss
- __MAX_STEPS__: how many steps to take into the account before making an update
- __DISCOUNT__: reward decay factor
- __ENV_NAME__: name of the game to learn
- __MAX_EP_LENGTH__: maximum length of episode (feel free to set it to some big number)
- __LEARNING_RATE__: learning rate of the Adam optimizer
- __CLIP_VALUE__: gradient clipping value (since this algorithm uses n-step return there is a greater posibility of exploding gradients)
- __SAVE_DIR__: directory in which should logs and models be stored

### LSTM version of A3C algorithm

In order to start training of the LSTM version of the A3C algorithm you just need to run:
```
python3 A3C.py
```

In order to test LSTM version of the A3C algorithm you just need to run:
```
python3 A3C_test.py <model_path> <should_render> #should render is y/n character that indicates will the rendering be done or not
```
Testing is performed by playing the game __NUM_GAMES__ times. __NUM_GAMES__ can be changed in A3C_test.py. Also, __IMG_WIDTH__, __IMG_HEIGHT__, __ENV_NAME__ and __CNT_FRAMES__ can be configured too.
Make sure to use the same __IMG_WIDTH__, __IMG_HEIGHT__ and __CNT_FRAMES__ as when training in order to avoid errors when loading model.

In order to check Tensorboard output you can start __start_tensorboard.sh__ script once inside the Tensorboard directory by running:
```
. start_tensorboard.sh
```
__start_tensorboard.sh__ script needs to be copied to the Tensorboard directory in order to make it work.

### A3C algorithm version without the LSTM layer

In order to start training of the A3C algorithm version without the LSTM layer you just need to run:
```
python3 A3C_no_lstm.py
```

In order to test version of the A3C algorithm without the LSTM layer you just need to run:
```
python3 A3C_no_lstm_test.py <model_path> <should_render> #should render is y/n character that indicates will the rendering be done or not
```
Testing is performed by playing the game __NUM_GAMES__ times. __NUM_GAMES__ can be changed in A3C_no_lstm_test.py. Also, __IMG_WIDTH__, __IMG_HEIGHT__, __ENV_NAME__ and __CNT_FRAMES__ can be configured too.
Make sure to use the same __IMG_WIDTH__, __IMG_HEIGHT__ and __CNT_FRAMES__ as when training in order to avoid errors when loading model.

In order to check Tensorboard output you can start __start_tensorboard.sh__ script once inside the Tensorboard directory by running:
```
. start_tensorboard.sh
```
__start_tensorboard.sh__ script needs to be copied to the Tensorboard directory in order to make it work.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Big thanks to Arthur Juliani for great series of posts about reinforcement learning named Simple Reinforcement Learning with Tensorflow series available at https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0.
* Big thanks to Chris aka cgnicholls whose post helped me gain more insight into the maths behind the A3C algorithm. His post can be found at https://cgnicholls.github.io/reinforcement-learning/2016/08/20/reinforcement-learning.html.
* Big thanks to Jaromír Janisch for a great tutorial related to Prioritized Experience Replay available at https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/.

## References
In this section will be mentioned the most important papers used for implementing algorithms used in this repository.
* Playing Atari with Deep Reinforcement Learning, https://arxiv.org/abs/1312.5602
* Prioritized Experience Replay, https://arxiv.org/abs/1511.05952
* Deep Reinforcement Learning with Double Q-learning, https://arxiv.org/abs/1509.06461
* Dueling Network Architectures for Deep Reinforcement Learning, https://arxiv.org/abs/1511.06581
