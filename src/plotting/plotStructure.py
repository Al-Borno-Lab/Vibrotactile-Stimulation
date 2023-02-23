from matplotlib import pyplot as plt    # MatPlotLib is a plotting package. 
import matplotlib.figure
import numpy as np                 # NumPy is a numerical types package.
from scipy import stats            # ScPy is a scientific computing package. We just want the stats, because Ca2+ imaging is always calculated in z-score.
from scipy.stats import circmean
import math
from posixpath import join
from ..model import lifNetwork as lif
from typing import Type, List
import numpy.typing as npt
from IPython.display import display




def plot_structure(LIF: lif.LIF_Network, 
                   conn: bool = False, 
                   conn_target: List = None,
                   figure_size: List = [10, 8]) -> matplotlib.figure.Figure:
  """Return 3D structure of the network with options to show connectivities.

  Args:
      LIF (lif.LIF_Network): An instance of the LIF_Network object.
      conn (bool, optional): Whether to plot connections. Defaults to False.
      conn_target (list, optional): List of neurons' index to plot connectivity
        with indicated neuron as presynaptic neuron. Defaults to None.
      figure_size (List, optional): Indicates the plot dimensions in inches.

  Returns:
      (matplotlib.figure.Figure): Returns the matplotlib.Figure object of which 
        the plot is on.

  Additional notes: 
  - `conn_target` is  a list of neurons' index that we desire to plot the
    the connectivity with these neurons as the presynaptic neurons. For example, 
    if `2` is in the list, all connections that have neuron index 2 as 
    presynaptic connection will be plotted in the 3D plot.
  """

  if conn == True and conn_target is None:
    conn_target = range(LIF.n_neurons)
  
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(LIF.x, LIF.y, LIF.z)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('Location of cells')
  fig.set_size_inches(*figure_size)
  cm = plt.get_cmap('viridis', LIF.n_neurons)

  if conn:
    C = LIF.network_conn
    for i in range(LIF.n_neurons):
      for j in range(LIF.n_neurons):
        if C[i][j] > 0 and i in conn_target:
          ax.plot([LIF.x[i], LIF.x[j]], 
                  [LIF.y[i], LIF.y[j]], 
                  [LIF.z[i], LIF.z[j]], 
                  color=cm(i),
                  linewidth=.75)
  # plt.show()
  plt.close()  # In case fig is not garbage collected
  return fig



def plot_connectivity(LIF: lif.LIF_Network) -> matplotlib.figure.Figure:
  """Return the LIF instance connectivity matrix.

  Args:
      LIF (lif.LIF_Network): Instance of the LIF_Network object.

  Returns:
      matplotlib.figure.Figure: _description_
  """

  fig = plt.figure()
  ax = fig.add_subplot()
  ax.matshow(LIF.network_conn)
  ax.set_ylabel("Neuron Index")
  ax.set_xlabel("Connectivity to other neurons (index)")
  ax.set_title("Connectivity matrix")
  
  # plt.show()
  plt.close()  # In case fig is not garbage collected
  return fig



def plot_voltage(sim: List[npt.NDArray], 
                 n: int = 5, 
                 figure_size: List = [10, 8]) -> matplotlib.figure.Figure:
  """Return voltage plot using values returned by `LIF_Network.simulate()`.

  Args:
      sim (List[npt.NDArray]): List of results returned by the simulate method
        of the LIF_Network object.
      n (int, optional): Plot the first n-th indices of neurons. Defaults to 5.
      figure_size (List, optional): Figure size in inches. Defaults to [10, 8].

  Returns:
      matplotlib.figure.Figure: _description_
  """
  [v, g, p, t, inp, dw] = sim
  fig = plt.figure()
  ax = fig.add_subplot()
  for i in range(n):
    ax.plot(t[:-1],v[:-1,i]-i*100)
  ax.set_xlabel("time [ms]")
  ax.set_ylabel('neural voltage, by index x 100')
  ax.set_title('example voltages')
  fig.set_size_inches(*figure_size)
  
  # plt.show()
  plt.close()  # In case fig isn't collected by garbage colleciton
  return fig


def plot_connection_weight(LIF: lif.LIF_Network) -> matplotlib.figure.Figure:
  """Return network connection weights.

  Args:
      LIF (lif.LIF_Network): LIF_Network instance to be plotted.

  Returns:
      matplotlib.figure.Figure: _description_
  """
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.matshow(LIF.network_W)
  ax.set_ylabel("Neuron Index")
  ax.set_xlabel("Weight to other neurons (index)")
  ax.set_title("Connectivity matrix")

  # plt.show()
  plt.close()  # In case fig is not garbage collected
  return fig




def plot_sorted_weights(LIF: lif.LIF_Network) -> matplotlib.figure.Figure:
  """Return plot of sorted connection-weights. 

  Args:
      LIF (lif.LIF_Network): _description_

  Returns:
      matplotlib.figure.Figure: _description_
  """

  Wf = np.reshape(LIF.network_W, (1, -1))
  Wf = np.sort(Wf)
  Wf = Wf[Wf>0]  # Negative weight does not make sense, floating point error

  fig = plt.figure()
  ax = fig.add_subplot()
  ax.plot(Wf)
  ax.set_xlabel("Order of Connection-Weights")
  ax.set_ylabel("Connection weight")
  ax.set_title("Sorted Connection Weights")

  # plt.plot(np.sort(Wf[Wf>0]))
  # plt.xlabel('order of weights')
  # plt.ylabel('weight')
  # plt.title('Sorted weights pre-training')
  # plt.show()
  # print(np.mean(W[W > 0].flatten()))

  # plt.show()
  plt.close()
  return fig



def plot_weight_change(weight_diff: npt.NDArray) -> matplotlib.figure.Figure:
  fig = plt.figure()
  ax = fig.add_subplot()

  cax = ax.matshow(weight_diff)
  ax.set_xlabel('Weight to other neurons')
  ax.set_ylabel('Neuron index')
  ax.set_title('Change in Weights after-training')
  fig.colorbar(cax)

  # plt.show()
  plt.close()
  return fig

def plot_neural_currents(sim: List[npt.NDArray], 
                         n: int = 5, 
                         figure_size: List = [10, 8]) -> matplotlib.figure.Figure: 
  """Return current plot using values returned by `LIF_Network.simulate()`.

  Args:
      sim (List[npt.NDArray]): List of results returned by the simulate method
        of the LIF_Network object.
      n (int, optional): Plot the first n-th indices of neurons. Defaults to 5.
      figure_size (List, optional): Figure size in inches. Defaults to [10, 8].

  Returns:
      matplotlib.figure.Figure: _description_
  """
  [v, g, p, t, inp, dw] = sim
  fig = plt.figure()
  ax = fig.add_subplot()
  for i in range(n):
    ax.plot(t, g[:,i]-i*50)
  ax.set_xlabel('time [ms]')
  ax.set_ylabel('syaptic current (offset by index)')
  ax.set_title('example neural currents')

  # plt.show()
  plt.close()
  return fig


def plot_external_inputs(sim: List[npt.NDArray], 
                         n: int = 5, 
                         figure_size: List = [10, 8]) -> matplotlib.figure.Figure: 
  """Return external-input plot using values returned by `LIF_Network.simulate()`.

  Args:
      sim (List[npt.NDArray]): List of results returned by the simulate method
        of the LIF_Network object.
      n (int, optional): Plot the first n-th indices of neurons. Defaults to 5.
      figure_size (List, optional): Figure size in inches. Defaults to [10, 8].

  Returns:
      matplotlib.figure.Figure: _description_
  """
  [v, g, p, t, inp, dw] = sim
  fig = plt.figure()
  ax = fig.add_subplot()
  for i in range(n):
    ax.plot(t, p[:,i]-i*1)
  ax.set_xlabel('time [ms]')
  ax.set_ylabel('poisson spikes (offset by index)')
  ax.set_title('example external inputs')
  # plt.show()
  plt.close()
  return fig


def plot_internal_inputs(sim: List[npt.NDArray], 
                         n: int = 5, 
                         figure_size: List = [10, 8]) -> matplotlib.figure.Figure: 
  """Return internal-input plot using values returned by `LIF_Network.simulate()`.

  Args:
      sim (List[npt.NDArray]): List of results returned by the simulate method
        of the LIF_Network object.
      n (int, optional): Plot the first n-th indices of neurons. Defaults to 5.
      figure_size (List, optional): Figure size in inches. Defaults to [10, 8].

  Returns:
      matplotlib.figure.Figure: _description_
  """
  [v, g, p, t, inp, dw] = sim
  fig = plt.figure()
  ax = fig.add_subplot()
  for i in range(n):
    ax.plot(t, inp[:,i]-i*1)
  ## NOTE (TONY) - Original logic, kept to double check later.
  # for i in range(2):
  #   plt.plot(t,inp[:,i])
  ax.set_xlabel('time [ms]')
  ax.set_ylabel('poisson spikes (offset by index)')
  ax.set_title('example internal inputs')
  # plt.show()
  plt.close()
  return fig

def plot_delta_weight(sim: List[npt.NDArray], 
                      n: int = 5, 
                      figure_size: List = [10, 8]) -> matplotlib.figure.Figure: 
  """Return weight-change plot using values returned by `LIF_Network.simulate()`.

  Args:
      sim (List[npt.NDArray]): List of results returned by the simulate method
        of the LIF_Network object.
      n (int, optional): Plot the first n-th indices of neurons. Defaults to 5.
      figure_size (List, optional): Figure size in inches. Defaults to [10, 8].

  Returns:
      matplotlib.figure.Figure: _description_
  """
  [v, g, p, t, inp, dw] = sim
  fig = plt.figure()
  ax = fig.add_subplot()
  
  ax.plot(t, dw)
  ax.set_xlabel('time [ms]')
  ax.set_ylabel('weight changes')
  ax.set_title('dW')
  # plt.show()
  plt.close()
  return fig
























def plotter(LIF: lif.LIF_Network, 
            time: int, 
            pN: int = 5) -> None:
  """Function to quickly run simluations and plot results.

  This function is just a convenience function to run simulation and plot
  multiple plots at the same time.

  Args:
      LIF (lif.LIF_Network): LIF_Network object instance.
      time (int): Simulation duration (ms) for `LIF_Network.simulate()` method.
      pN (int, optional): First pN-th index of neurons to plot voltage,
        current, internal/external inputs. Defaults to 5.
  """
  # Plot structure
  structure_plot = plot_structure(LIF)
  display(structure_plot)

  # Plot connectivity matrix
  connectivity_matrix_plot = plot_connectivity(LIF)
  display(connectivity_matrix_plot)

  # Plot weights pre-training
  pre_train_weight_plot = plot_connection_weight(LIF)
  display(pre_train_weight_plot)

  # Plot sorted weights pre-training
  pre_train_sorted_weight = plot_sorted_weights(LIF)
  display(pre_train_sorted_weight)

  # Run Simulation
  W = np.copy(LIF.network_W)
  simulation_results = LIF.simulate(sim_duration = time)
  [v, g, p, t, inp, dw] = simulation_results
  W2 = LIF.network_W
  
  # Plot weights after training
  post_train_weight_plot = plot_connection_weight(LIF)
  display(post_train_weight_plot)

  # Plot change in weights after-training
  post_train_weight_change_plot = plot_weight_change((W2-W))
  display(post_train_weight_change_plot)

  # # Plot ordered weights (after training)
  post_train_sorted_weight = plot_sorted_weights(LIF)
  display(post_train_sorted_weight)

  # # Plot example neural activity = voltage
  post_train_neuronal_voltage = plot_voltage(simulation_results, pN)
  display(post_train_neuronal_voltage)

  # # Plot neural current 
  post_train_neuronal_current = plot_neural_currents(simulation_results, pN)
  display(post_train_neuronal_current)

  # # Plot external inputs (poisson spikes)
  post_train_external_inputs = plot_external_inputs(simulation_results, pN)
  display(post_train_external_inputs)

  # # Plot internal inputs
  post_train_internal_inputs = plot_internal_inputs(simulation_results, pN)
  display(post_train_internal_inputs)

  # # Plot weight changes
  post_train_weight_changes = plot_delta_weight(simulation_results, pN)
  display(post_train_weight_changes)







def plotter2(LIF: lif.LIF_Network, 
             sim_duration, 
             I: npt.NDArray, 
             pN = 5):
  """

  Args: 
    LIF (LIF_Network): 
    sim_duration: 
    I (ndarray): 
    pN (int): Plot up to index (pN-1)-th neuron.

  NOTES (TONY): 
  - This function has been reduced to be used just for the simulation part.
  """

  h = LIF.simulate(sim_duration = sim_duration, 
                   epoch_current_input = I)
  # [v,g,p,t,inp,dw] = h
  W2 = LIF.network_W

  fig = plt.figure()
 
  # for i in range(pN):
  #   plt.plot(t, g[:,i]-i*50)
  # plt.xlabel('time [ms]')
  # plt.ylabel('syaptic current (offset by index)')
  # plt.title('example neural currents')
  # plt.show()
