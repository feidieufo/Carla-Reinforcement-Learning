import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas
import os

f, ax = plt.subplots(1, 1)

log_fname = os.path.join('data/dqn_carla_test/dqn_carla_test_s0', 'progress.txt')
csv = pandas.read_table(log_fname)

x = range(len(csv["Averagereward"]))
y = csv["Averagereward"]
ax.plot(x, y, color='r', label="dqn")

ax.fill_between(x, y - csv['Stdreward'], y + csv['Stdreward'], color='r', alpha=0.2)

plt.show()