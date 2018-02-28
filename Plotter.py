from matplotlib import pyplot as plt

import os

def get_info(f):
    episodes, rewards = [], []
    for line in list(f):
        ep, r = line.split()
        episodes.append(int(ep))
        rewards.append(float(r))
    return episodes, rewards

for file in os.listdir("."):
    if file.endswith(".txt"):
        f = open(file, 'r')
        x, y = get_info(f)
        fig = plt.figure(0)
        plt.plot(x, y)
        filename = os.path.splitext(f.name)[0]
        fig.canvas.set_window_title(filename)
        plt.show()
        #plt.plot()
