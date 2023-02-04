from src.models.LIF import LIF_Network
from src.graphingUtils import plot_structure, figure_size

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6)
LIF = LIF_Network(n_neurons=150, dimensions= [[0,100],[0,1],[0,1]])
LIF.p_conn = .1
d = LIF.structured_conn(LIF)
plot_structure(LIF, conn = True, conn_target = [0, 75])


data = d.flatten()
data = data[~np.isnan(data)]

print("The observed connectivity probability is: " + str(len(data)/(LIF.n_neurons**2)))

fig, ax = plt.subplots(figsize =(10, 7))
binwidth = 5
ax.hist(d.flatten(), bins=range(int(min(data)), int(max(data) + binwidth), binwidth), density= True)
plt.xlabel('connection distance')
plt.ylabel('probability of connections')
plt.title('example structured connectivity histogram')
plt.gcf().set_size_inches(figure_size[0], figure_size[1])
plt.show()