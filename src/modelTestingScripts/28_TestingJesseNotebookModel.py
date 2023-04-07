## Testing script

import src.model.copied_from_ipynb_discussion_with_jesse as lif
from src.plotting import plotStructure as lifplot
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import numpy.typing as npt




################################################################################
### PLOTTING FUNCTIONS, DON'T TOUCH!!!
################################################################################

def plot_helper(ax:plt.Axes, y:list, label:str,
                n_sim:int, sim_duration:float, n_neurons:int, proba_conn:float,):
  """Plot the plot."""
  ax.plot(y, label=label, marker='+', linestyle='-')
  ax.set_title(f"<w(t)>\n\
                  | n_iterations: {n_sim} | sim_duration: {sim_duration} ms |\n\
                  | n_neurons: {n_neurons} | proba_of_connection: {proba_conn} |")
  ax.set_xlabel(f"n-th iteration | each iteration = {sim_duration} ms")
  ax.set_ylabel("mean network weight")
  ax.legend()

  # ax.vlines(x=iteration*sim_duration, ymin=0, ymax=1, 
  #           label="Simulation Iteration",
  #           color="red", alpha=0.3)
  # ax.set_ylim(bottom=0, top=1)
  # ax.set_xlim(left=0, right=n_sim*sim_duration)
  # plt.legend()

  return ax

def save_helper(fig:matplotlib.figure.Figure, iteration:int, export_directory:str) -> None: 
  """Save the plot in current working directory."""
  # Export plot
  os.makedirs(export_directory, exist_ok=True)
  fig.savefig(f"{export_directory}/iteration_{iteration}.png", 
              facecolor="white")
  return

def plot_network_mean_weight_over_time(n_neurons:int=300,
                                       n_sim:int=10000,
                                       sim_duration:float=100,
                                       proba_conn:float=0.08,
                                       mean_w:list=[0.1, 0.2, 0.3, 0.5],
                                       export_iteration_skip:int=10,
                                       plot_dim:tuple=(25, 10)):
  """Plot the mean network weight over time."""

  # Set variables
  export_directory = "export_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  holder_mean_network_w = {}
  holder_LIF = {}
  
  # Instantiate a model for each mean_w item
  for each_mean_w in mean_w:
    name = f"mean_w={each_mean_w}"
    holder_LIF[name] = lif.LIF_Network(n_neurons=n_neurons)
    holder_LIF[name].p_conn = proba_conn
    holder_LIF[name].mean_w = each_mean_w
    holder_LIF[name].random_conn()
    # lifplot.plot_structure(holder_LIF[name])
    holder_mean_network_w[name] = []

  # Run each of the n_sim
  for iteration in tqdm(np.arange(n_sim), 
                        desc=f"Simulation progress - total of {n_sim} simulations"):
    # Simulate for each model
    for nn_name, nn in holder_LIF.items():
      nn.simulate(timesteps=sim_duration)
      mean_network_w = np.mean(nn.network_W[nn.network_conn.astype(bool)])
      holder_mean_network_w[nn_name].append(mean_network_w)

    # Plot every `export_iteration_skip`-th simulation iteration
    if (iteration % export_iteration_skip == 0):
      fig, ax = plt.subplots(figsize=plot_dim)

      # Plot for each of the mean_w
      for nn_name, mean_network_w in holder_mean_network_w.items():
        plot_helper(ax=ax, y=mean_network_w, label=nn_name,
                    n_sim=n_sim, sim_duration=sim_duration, n_neurons=n_neurons, 
                    proba_conn=proba_conn)
      
      # Save figure
      save_helper(fig=fig, iteration=iteration, export_directory=export_directory)
      
      # Display inline (JupyterNotebook)
      # display(fig)
            
if __name__ == "__main__":
   plot_network_mean_weight_over_time()