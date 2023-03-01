## Testing script

from src.model import lifNetwork as lif
from src.plotting import plotStructure as lifplot
import matplotlib.pyplot as plt
import numpy as np
import os


LIF = lif.LIF_Network(n_neurons=300)
lifplot.plot_structure(LIF)

holder_mean_network_w = []

for idx in range(10000):
  LIF.simulate(sim_duration=100)
  mean_network_w = np.mean(LIF.network_W)
  holder_mean_network_w.append(mean_network_w)

  if idx%10 == 0:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(holder_mean_network_w)
    ax.set_title("<w(t)>")
    ax.set_xlabel("i-th simulate() run")
    ax.set_ylabel("mean network weight")
    
    os.makedirs("export", exist_ok=True)
    fig.savefig(f"export/iteration{idx}.png", facecolor="white")
            
