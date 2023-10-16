## Testing script

from src.model import lifNetwork as lif
from src.plotting import plotStructure as lifplot
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime


################################################################################
### CHANGE VARIABLES WITH THIS DICTIONARY
# - kappa_noise is not used when update_g_noise method uses "Ali"'s method
#   because he doesn't use this variable.
################################################################################


choose_model_setup = {"update_g_noise_method": "Ali",
                      "update_g_syn_method": "Ali",
                      "update_v_method": "Matteo",
                      "update_v_capacitance_method": "Ali",
                      "update_thr_method": "Ali"
                      }


################################################################################
### PLOTTING FUNCTIONS, DON'T TOUCH!!!
################################################################################
def plot_network_mean_weight_over_time(n_neurons=300,
                                       n_sim = 10000,
                                       sim_duration=100,
                                       kappa=400,  # Paper states 8, Ali's code states 400
                                       kappa_noise=0.026,  # Ali's code does have this, he just used g_poisson=1.3
                                       proba_conn = 0.8,
                                       mean_w = 0.5,
                                       export_iteration_skip = 10,
                                       plot_width = 25,
                                       plot_height=10,
                                       choose_model_setup=choose_model_setup,
                                       ):
  """Plot the mean network weight over time.

  Args:
      n_neurons (int, optional): _description_. Defaults to 300.
      proba_conn (float, optional): _description_. Defaults to 0.8.
      sim_duration (float, optional): _description_. Defaults to 100.
      n_sim (int, optional): _description_. Defaults to 10000.
      plot_width (float, optional): _description_. Defaults to 20.
      plot_height (float, optional): _description_. Defaults to 10.
      export_iteration_skip (int, optional): _description_. Defaults to 10.
  """
  # Set variables
  now = datetime.now()
  now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
  export_directory = "export_"+now_str
  x_iter = []
  holder_mean_network_w = []

  # Instantiate LIF network
  LIF = lif.LIF_Network(n_neurons=n_neurons)
  LIF.random_conn(proba_conn=proba_conn, mean_w=mean_w)
  # lifplot.plot_structure(LIF)


  for iteration in tqdm(range(n_sim)):
    LIF.simulate(sim_duration=sim_duration, 
                 kappa=kappa, 
                 kappa_noise=kappa_noise,
                 temp_param=choose_model_setup)
    
    mean_network_w = np.mean(LIF.network_W[LIF.network_conn])
    
    x_iter.append(iteration)
    holder_mean_network_w.append(mean_network_w)

    if (iteration % export_iteration_skip == 0):
      fig, ax = plt.subplots(figsize=(plot_width, plot_height))
      ax.scatter(x_iter, holder_mean_network_w)
      ax.set_title(f"<w(t)>\n\
                   | n_iterations: {n_sim} | sim_duration: {sim_duration} ms |\n\
                   | n_neurons: {n_neurons} | proba_of_connection: {proba_conn} |")
      ax.set_xlabel(f"n-th iteration | each iteration = {sim_duration} ms")
      ax.set_ylabel("mean network weight")
      # ax.vlines(x=iteration*sim_duration, ymin=0, ymax=1, 
      #           label="Simulation Iteration",
      #           color="red", alpha=0.3)
      # ax.set_ylim(bottom=0, top=1)
      # ax.set_xlim(left=0, right=n_sim*sim_duration)
      # plt.legend()
      
      # Export plot
      os.makedirs(export_directory, exist_ok=True)
      fig.savefig(f"{export_directory}/iteration_{iteration}.png", 
                  facecolor="white")
      
      # Display inline (JupyterNotebook)
      # display(fig)
            
if __name__ == "__main__":
   plot_network_mean_weight_over_time()