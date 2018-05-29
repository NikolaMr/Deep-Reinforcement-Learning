#import DuelingDDQNAgentPER
import DuelingDDQNAgent

PLOT_NAME = "ResultsDuelingDDQNAgent.png"

setup_dict = {}
setup_dict['observing_frames'] = 25000
setup_dict['replay_memory_size'] = 25000
setup_dict['learning_rate'] = 1e-4
setup_dict['start_eps'] = 0.7
setup_dict['exploring_frames'] = 2000000
setup_dict['saving_dir'] = "DuelingDDQNAgentEpsilonGreedyExplorationExpReplay"
setup_dict['log_freq'] = 5

#agent = DuelingDDQNAgentPER.Dueling_DDQN_PER_Agent(setup_dict)
agent = DuelingDDQNAgent.Dueling_DDQN_Agent(setup_dict)
rewards = agent.train()
from matplotlib import pyplot as plt
plt.plot(range(len(rewards)), rewards)
plt.savefig(PLOT_NAME)
