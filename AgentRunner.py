#import DuelingDDQNAgentPER
import DQNAgent
import DuelingDDQNAgent
import os
import os.path
import json

setup_dict = {}

if os.path.isfile('config.json'):
    setup_dict = json.loads(open('config.json').read())
else:
    setup_dict['observing_frames'] = 300
    setup_dict['replay_memory_size'] = 300
    setup_dict['learning_rate'] = 1e-5
    setup_dict['start_eps'] = 0.7
    setup_dict['exploring_frames'] = 2000000
    setup_dict['saving_dir'] = "PERDuelingDDQNEpsGreedy"
    setup_dict['log_freq'] = 5
    #setup_dict['MemoryType'] = 'MemoryPrioritizedForgetting'
    setup_dict['MemoryType'] = 'PrioritizedExperienceReplayMemory'
    setup_dict['ExplorationStrategy'] = 'BoltzmannExplorationStrategy'
    setup_dict['Agent'] = 'DQN_Agent'
#agent = DuelingDDQNAgentPER.Dueling_DDQN_PER_Agent(setup_dict)
agent = None
if setup_dict['Agent'] == 'DQN_Agent':
    agent = DQNAgent.DQN_Agent(setup_dict)
if setup_dict['Agent'] == 'Dueling_DDQN_Agent':
    agent = DuelingDDQNAgent.Dueling_DDQN_Agent(setup_dict)
agent.train()
