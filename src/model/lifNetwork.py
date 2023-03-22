# import matplotlib.pyplot as plt    # MatPlotLib is a plotting package. 
import matplotlib.pyplot as plt
import numpy as np                 # NumPy is a numerical types package.
from scipy import stats            # ScPy is a scientific computing package. We just want the stats, because Ca2+ imaging is always calculated in z-score.
from scipy.stats import circmean
import math
from posixpath import join
import numpy.typing as npt
import time
from src.utilities import timer
import tensorflow as tf





def stdp_dw(time_diff:float, scale_factor:float=0.02, 
            stdp_beta:float=1.4, tau_r:float=4,
            tau_plus:float=10, tau_neg:float=None) -> float:
  """Calculate and return connection weight change according to STDP scheme.

  Scaling factor (eta) scales the weight update per spike.
  The default values are set forth by the paper Kromer et. al. (DOI 10.1063/5.0015196)
  and used for the purpose slow STDP and coexistence of desynchronized and 
  oversynchronized states.

  Args: 
    time_diff: [ms] t_{pre} - t_{post}
    scale_factor (float): Scaling factor of the STDP scheme; 0.02 is considered 
      slow STDP. Defaults to 0.02.
    stdp_beta (float): The ratio of overall depression area under curve to 
      potentiation area under curve. Defaults to 1.4
    tau_r (float): Ratio of tau_neg to tau_plus.
    tau_plus (float): [ms] STDP decay timescale for LTP.
    tau_neg (float): [ms] STDP decay timescale for LTD.

  Returns: 
    dw (float): Connection weight change.
  """
  dw = 0

  if (tau_r is None) & (tau_plus is None) & (tau_neg is None):
    raise Exception("tau_r, tau_plus, tau_neg: two of the three have to be provided.")
  elif (tau_r is None) & (tau_plus is None):
    raise Exception("Either tau_r or tau_plus is needed.")
  elif (tau_plus is None) & (tau_neg is None):
    raise Exception("Either tau_plus or tau_neg is needed.")
  elif (tau_neg is None) & (tau_r is None):
    raise Exception("Either tau_neg or tau_r is needed.")
  
  if tau_neg is None:
    tau_neg = tau_r * tau_plus
  elif tau_plus is None: 
    tau_plus = tau_neg / tau_r
  elif tau_r is None: 
    tau_r = tau_neg / tau_plus

  ## Case: LTP (Long-term potentiation)
  if np.less_equal(time_diff, 0):
    dw = (scale_factor 
          * np.exp( time_diff / tau_plus))
  
  ## Case: LTD (Long-term depression)
  if np.greater(time_diff, 0):
    dw = (scale_factor 
          * -(stdp_beta/tau_r) 
          * np.exp( -time_diff / tau_neg ))

  return dw


def visualize_stdp_scheme_assay():
  """Plot the STDP scheme assay.

  Y-axis being the connection weight update (delta w).
  X-axis being the time diff of presynaptic spike timestamp less postsynaptic
  spike timestamp. The definition of time_diff is opposite of time_lag (termed
  in the original paper).
  """
  fig, ax = plt.subplots()
  x = np.arange(-100, 100, 1)
  
  for i in x:
    ax.scatter(x=i, y=stdp_dw(i), c="black", s=3)

  ax.set_title("STDP Scheme Curve")
  ax.set_xlabel("Time offset (Pre-Post)")
  ax.set_ylabel("dw / dt")

  return fig


class LIF_Network:
  """Leaky Intergrate-and-Fire (LIF) Neuron network model.

  Args:
    n_neurons (int): number of neurons; default to 1000 neurons
    dimensions: Spatial dimensions for plotting; this plots the neurons in a 
      100x100x100 3D space by default.
    auto_random_connect (bool): Whether to randomly connec the neurons.
  """
  def __init__(self, n_neurons = 1000, 
               dimensions = [[0,100],[0,100],[0,100]],
               auto_random_connect: bool = True) -> None:

    # Neuron count
    self.n_neurons = n_neurons

    # Values that Ali uses in his codebase
    self.v_spike = 20     # Not in paper, but shows up in Ali's codebase                    # [mV] The voltage that spikes rise to ((AP Peak))
    self.g_leak_ali = 10  # The value Ali used in his code
    self.v_ali = np.random.uniform(low=-10, high=10, size=(self.n_neurons,)) - 45           # [mV] Current membrane potential - Initialized with Unif(-55, -35)
    self.tau_spike = 1                                                                      # [ms] Time for an AP spike, also the length of the absolute refractory period

    
    # Spatial organization
    self.x = np.random.uniform(*dimensions[0], size=self.n_neurons)
    self.y = np.random.uniform(*dimensions[1], size=self.n_neurons)
    self.z = np.random.uniform(*dimensions[2], size=self.n_neurons)

    # Internal time trackers
    self.t_current = 0                                                                              # [ms] Current time 
    self.dt = 0.1                                                                           # [ms] Timestep length
    self.euler_step_idx = 0                                                                 # [idx] Euler step index
    self.relax_time = 10000                                                                 # [ms] Length of relaxation phase 

    # Electrophysiology values
    self.v = np.random.uniform(low=-38, high=-40, size=(self.n_neurons,))
    self.v_rest = -38                                                                       # [mV] Resting voltage, equilibrium voltage. Default: -38 mV
    self.v_thr = np.zeros(self.n_neurons) - 40                                              # [mV] Reversal potential for spiking; equation (3)
    self.v_thr_rest = np.zeros(self.n_neurons) - 40                                         # [mV] Reversal potential for spiking during refractory period. Defaults to -40mV; equation (3).
    self.v_syn = 0                                                                          # [mV] Reversal potential; equation (2) in paper
    self.tau_syn = 1                                                                        # [ms] Synaptic time-constant; equation (4) in paper
    self.tau_rf_thr = 5                                                                     # [ms] Timescale tau between refractory and normal thresholds relaxation period
    self.g_leak = 0.02                                                                      # [mS/cm^2] Conductivity of the leak channels; equation (2) in paper
    self.g_syn_initial_value = 0                                                            # [mS/cm^2] Initial value of synaptic conductivity; equation (2) in paper
    # Connectivity parameters (connection weight = conductivity)
    self.synaptic_delay = 3                                                                 # [ms] Synaptic transmission delay from soma to soma (default: 3 ms), equation (4) in paper
    # Threshold and Membrane Potential are set to following values after spiking ((Equation 3 in paper))
    self.v_reset = -67                                                                      # [mV] Membrane potential right after spikeing ((Hyperpolarization)); equation (3) in paper
    self.v_rf_spike = 0                                                                     # [mV] Threshold during relative refractory period; V_th_spike of equation (3) in paper
    # (Paper definition of) Specific Membrane Capacitance
    capa_rv_norm_mean = 3
    capa_rv_norm_stdev = 0.05 * capa_rv_norm_mean                                           # Defined by paper
    self.capacitance = np.random.normal(loc=capa_rv_norm_mean,                              # [microF/cm^2] Equation (2) in paper
                                        scale=capa_rv_norm_stdev, 
                                        size=self.n_neurons)

    # Input noise of Poisson distribution (input is generated with I = G*V)
    self.g_poisson = 1.3                                                                    # [mS/cm^2] conductivity of the extrinsic poisson inputs
    self.poisson_noise_spike_flag = np.zeros(self.n_neurons)

    # Internal Trackers
    self.spike_flag = np.zeros(self.n_neurons)                                              # Tracker of whether a neuron spiked
    self.t_minus_1_spike = np.zeros(self.n_neurons) - 10000                                 # Tracker of spike timestamp; Default to -10000 for mathematical computation convenience; keep track of the previous timestamps of spiked neurons.
    self.t_minus_0_spike = np.zeros(self.n_neurons) - 10000                                 # Tracker of spike timestamp; Default to -10000 for mathematical computation convenience; keep track of the current timestamps of spiked neurons.
    self.spike_record = np.empty(shape=(1, 2))                                              # Tracker of Spike record recorded as a list: [neuron_number, spike_time]
    self.g_syn = np.zeros(self.n_neurons) + self.g_syn_initial_value                        # Tracker of dynamic synaptic conductivity. Initial value of 0.; equation (2)
    self.g_noise = np.zeros(self.n_neurons)                                                 # Tracker of dynamic noise conductivity
    self.w_update_flag = np.zeros(self.n_neurons)                                           # Tracker of connetion weight updates. When a neuron spikes, it is flagged as needing update on its connection weight.
    self.spiked_input_w_sums = np.zeros(self.n_neurons)                                     # Tracker of the connected presynaptic weight sum for each neuron (Eq 4: weight * Dirac Delta Distribution)
    self.dW = 0                                                                             # Tracker of change of weight used by `run_stdp_on_all_connected_pairs()`
    self.network_conn = np.zeros((self.n_neurons, self.n_neurons))                          # Tracker of Neuron connection matrix: from row-th neuron to column-th neuron
    self.network_W = 0                                                                      # Tracker of Neuron connection weight matrix: from row-th neuron to column-th neuron

    # STDP paramters
    self.stdp_beta = 1.4                                                                    # Balance factor (ratio of LTD to LTP); equation (7)
    self.stdp_tau_r = 4                                                                     # When desynchronized and synchronized states coexist; equation (7)
    self.stdp_tau_plus = 10                                                                 # For the postive half of STDP; equation (7)
    self.stdp_tau_neg = self.stdp_tau_r * self.stdp_tau_plus                                # For the negative half of STDP; equation (7)
    self.eta = 0.02                                                                         # Scales the weight update per spike; 0.02 for "slow STDP"; equation (7)

    # ??? Speculate to be capacitance random variable drawn from a normal distribution ???
    # NOTE (Tony): Odd that this has an expected value of 150. Most reserach
    #   seems to agree that the universal specific membrane capacitance is 
    #   approx 1 microFarad/cm^2.
    tau_c1 = np.sqrt(-2 * np.log(np.random.random(size=(self.n_neurons,))))                 # ??? membrane time-constant component 1 ??? - Jesse and Tony are unsure what this is 
    tau_c2 = np.cos(2 * np.pi * np.random.random(size=(self.n_neurons,)))                   # ??? membrane time-constant component 2 ??? - Jesse and Tony are unsure what this is  
    self.tau_m = 7.5 * tau_c1 * tau_c2 + 150                                                # ??? Unsure what this is, but through usage seems like the capacitance random-variable in equation (2) (However, values are off...)
    
    # Generate neuron connection matrix
    if auto_random_connect:
      self.__random_conn()  # Create random neuron connections

  

  # def structured_conn(self, LIF, mean_w: float=0.5):
  #   """To connect the neurons according to a seemingly exponential distribution.

  #   NOTE (Tony): 
  #   I am not entirely clear what the intent is behind this kind of connectivity.
  #   However, the calculation here is redundant and inefficient, and is O(N^2).

  #   TODO (Tony): 
  #     - Object method implemented incorrectly and is asking for the lifNetwork object.
  #     - Fix this method.

  #   Args:
  #       LIF (_type_): _description_
  #       mean_w (float, optional): The mean connection weight to normalize each
  #       connection in the network to. Defaults 0.5.

  #   Returns:
  #       _type_: _description_
  #   """

  #   self.network_conn = np.zeros([self.n_neurons,self.n_neurons])  
  #   dist=np.empty([self.n_neurons,self.n_neurons])
  #   dist[:] = np.nan
  #   dist1=[]
  #   c=[]
  #   for f in range(LIF.n_neurons-1):
  #     i=f
  #     for j in range(f+1,LIF.n_neurons,1):
  #       a=(LIF.x[i]-LIF.x[j])*(LIF.x[i]-LIF.x[j])+(LIF.y[i]-LIF.y[j])*(LIF.y[i]-LIF.y[j])+(LIF.z[i]-LIF.z[j])*(LIF.z[i]-LIF.z[j])
  #       b=np.sqrt(a)
  #       c.append(b)
  #     #print('distance between neurons', i+1, 'and', j+1, ': ', b)
  #   d=sum(c)/len(c)
  #   print('The average distance between neurons in this network is:', d)
  #   print('The base of the exponent is:', LIF.p_conn**(1/d))
  #   bb=[]
  #   cc=[]
  #   for p in range(LIF.n_neurons):
  #     for p2 in range(LIF.n_neurons):
  #       if(p!=p2):
  #         a=(LIF.x[p]-LIF.x[p2])*(LIF.x[p]-LIF.x[p2])+(LIF.y[p]-LIF.y[p2])*(LIF.y[p]-LIF.y[p2])+(LIF.z[p]-LIF.z[p2])*(LIF.z[p]-LIF.z[p2])
  #         b=np.sqrt(a)
  #         dist1.append(b)
  #   for p in range(LIF.n_neurons):
  #     for p2 in range(LIF.n_neurons):
  #       if(p!=p2):
  #         a=(LIF.x[p]-LIF.x[p2])*(LIF.x[p]-LIF.x[p2])+(LIF.y[p]-LIF.y[p2])*(LIF.y[p]-LIF.y[p2])+(LIF.z[p]-LIF.z[p2])*(LIF.z[p]-LIF.z[p2])
  #         b=np.sqrt(a)
  #         #aa=((LIF.p_conn**(1/d)) ** b)
  #         aa=2.71828**(-b/(LIF.p_conn*max(dist1)))
  #         pc = np.random.random(size=(1,))
  #         if(pc<aa):
  #           self.network_conn[p][p2] = 1
  #           dist[p][p2]=b

  #   ## TODO (Tony): Verify that this block below is indeed not needed.
  #   # self.network_W = np.random.random(size=(self.n_neurons,self.n_neurons))
  #   # self.network_W[self.network_conn == 0] = 0
  #   # self.network_W = self.network_W/np.mean(self.network_W[self.network_W > 0]) * mean_w
  #   # self.network_W[self.network_W > 1] = 1
  #   # self.network_W[self.network_W < 0] = 0
  #   return dist

  def __random_conn(self, mean_w: float=0.5, proba_conn: float=0.07) -> None:
    """Randomly create connections between neurons basing on the proba_conn.

    Using LIF neuron objects intrinsic probability of presynaptic connections
    from other neurons (`proba_conn`), and then mean conductivity of synapses 
    (mean_w) to normalize the randomly generated value.

    Update `network_W` matrix to binary values indicating the connections
    between neurons.

    Args:
        mean_w (float, optional): The mean connection weight to normalize each  
          connection in the network to. Defaults 0.5
        proba_conn (float, optional): Probability of connection between neurons.
          Defaults 0.07.
    """
    # Ensure mean_w is in (0, 1) as any other value does not make sense.
    assert (np.greater(mean_w, 0) & np.less(mean_w, 1)), \
      f"mean_w has be within (0, 1) (exclusive). Current mean_w: {mean_w}"

    # Generate connectivity matrix - Because none is connected at initialization
    self.network_conn = np.random.binomial(n=1, p=proba_conn, 
                                           size=(self.n_neurons, self.n_neurons))
    # Generate weight matrix
    self.network_W = np.random.random(size=(self.n_neurons, self.n_neurons))
    # Mark non-connected pairs' weight as zero
    self.network_W = np.multiply(self.network_conn, self.network_W)
    # Normalized to mean conductivity (i.e., `mean_w`)
    self.network_W = np.multiply(self.network_W, 
                                 mean_w / np.mean(self.network_W))
    
    # Hard bound weight to [0, 1]
    # self.network_W = np.clip(self.network_W, 
    #                          a_min=0, a_max=1)
    
  def __update_w_matrix(self, dw:float, pre_idx:int, post_idx:int) -> None:
    """Updates object's connection weight matrix in-place.

    This is a "private" method, and is done so for the reason that it modifies 
    the object attribute in-place instead of returning a copy of the updated
    variable.
    """
    # Update connection weight
    self.network_conn[(pre_idx, post_idx)] += dw
    
    # Hard bound to [1, 0]  
    np.clip(self.network_conn[(pre_idx, post_idx)], a_min=0, a_max=1)

  def simulate_poisson(self, 
                       poisson_noise_lambda_hz: int = 20) -> None:
    """Calculate and update Poisson spike flags for noise input calculation.
    
    The returned NDArray has dimension of 1xn_neurons, with each element
    corresponding to all possible presynaptic neurons of a postsynaptic neuron.
    The returned binary flag indiates whether the presynaptic neuron has spiked
    according to a Binomial distribution.

    The Poisson spike train is here approximated by a binomial distribution. 
    lambda = n * p

    Instead of simulating across time, we are simulating at each euler-step 
    snapshot and such is valid because each firing is independent of another.

    Args: 
      poisson_noise_lambda_hz (int, optional): [Hz] The rate of Poisson 
        distributed noise spiking. Defaults 20.
    """
    # Convert from Poisson to Binomial
    n_per_second = 1 / 1e-3 /self.dt # Number of euler-steps per second
    p_approx = poisson_noise_lambda_hz / n_per_second

    if ((n_per_second > 100) & (p_approx < 0.01)):
      raise Exception("""time-step is too large causing the Poisson noise binomial estimation to be inaccurate. 
      See `simulate_poisson` method definition for more details.""")

    # Generate Poisson noise spike flags
    self.poisson_noise_spike_flag = np.random.binomial(n=1, p=p_approx,
                                                       size=(self.n_neurons,))


  def spikeTrain(self,lookBack=None, nNeurons = 5, purge=False):
    """Plot spiketrain plot of specified neuron counts and lookBack range.

    Args: 
      lookback: length of time [ms] to backtrack for plotting the spikeTrain; 
        default None results in the entire time duration.
      nNeurons: number of neurons to plot spike train; should be <= n_neurons 
        in the LIF_Network. Defaults 5.
      purge (boolean): Clears the spike record in the LIF_Network object

    Returns:
      SR (np.ndarray): n-by-2 ndarray recording the neuron and its spike time. 

    NOTE (Tony): 
      - Spike record is subsetted to be loopBack onwards.
      - Interesting way of utilizing argmax to find the first instance of 
        timestamp matching loopBack.
      - Spike record: first-column: The i-th neuron,
                      second-column: Time that spiked.
      - argmax returns the first instance of matched condition.
      - np.where returns the indices of items with satisfied conditions if second
        and third arguments are missing for the function.
      - The beginning attempt to subset SR with the first instance of condition
        match can be optimized as this only subsets partially and thus still
        requires the if-statement in the plotting calls. (Just not optimized.)
      - lookBack variable name can be better named as the author used the same 
        variable for two purposes. One to specify the length to look back in time,
        two to specify the starting point of spiketrain plotting timestamp.
      - `result` is a list of indices in the spike_record that matches that of 
        the specified neuron.
      - The method starts by creating a blank canvas using `plt.plot()` with 
        arguments such as the number of neurons and then fill in the spiketrain
        information in subsequent code.

    QUESTION (Tony): 
      - ??? Why does it delete the first row of spike record?
        - Answer: The first row was the placeholder placed by object __init__.
          Since the `simulation` method only appends spike records to the 
          spike_record variable, the original placeholder is still occupying the 
          first entry of the variable.
      - ??? Why is loopBack logic written in such convoluted way?
        - Answer: This may just be a preference, but the code is unpythonic.
      - ??? Why is reshaping needed if the spike_record is already in the format?
    """

    ## lookBack == 0 if None
    if lookBack is None:
      lookBack = self.t_current
    lookBack = self.t_current - lookBack  # Spiketrain plot starting timestamp
    # ## More pythonic way to accomplish the same thing
    # if lookBack is None:
    #   strt_timestamp = 0
    # else: 
    #   strt_timestamp = self.t_current - lookBack 
    # lookBack = strt_timestamp
    
    ## Subsetting spike record since lookBack onwards
    SR = np.reshape(self.spike_record,newshape = [-1,2])
    SR = np.delete(SR, 0, 0)               # Delete first row - the placeholder
    SRix = np.argmax(SR[:,1] >= lookBack)  # idx of first SR since timestamp == loopBack
    SR = SR[SRix:,:]                       # Subset using index
    
    ## Plotting spike records one neuron at a time
    # %matplotlib inline
    fig = plt.figure()
    plt.plot([lookBack, self.t_current],[0,nNeurons],'white')
    for i in range(nNeurons):
      result = np.array(np.where(SR[:,0] == i)).flatten()  # indices of i-th neuron
      for q in range(len(result)):
        loc = result[q]
        if (SR[loc,1]) >= lookBack:  # SR's second column is the spiking time
          plt.plot([SR[loc,1], SR[loc,1]],[i,i+.9],'k',linewidth=.5)
    
    plt.xlabel('time (ms)')
    plt.ylabel('neuron #')
    plt.title("Spike Train")
    fig.set_size_inches(5, 4)
    plt.show()

    if purge:
      self.spike_record = np.empty(shape=[1,2])
    
    return SR

  def vect_kuramato(self,
                    period: float = None,
                    lookback: float = None, 
                    r_cutoff = 0.3):
    """Return the phase mean of all neurons in the spike record.

    Kuramato vectors and trigonometry are used to calculate the phase mean,
    which has potential for rounding errors due to floating points. Thus, 
    r_cuttoff is used to eliminate the potential errors. The larger the
    potential rounding error, the larger the r_cutoff is needed.

    Args: 
      period: [ms] Time length of a single period we define the mapped-to polar
        coordinates.
      lookback: [ms] The length of time we are looking back to analyze.
        If set to the default `None`, we are considering the entire span of the
        spike record.
      r_cutoff: [ms] minimum Kuramato vector length threshold; r is for radius.
        If the vector length is less than the r_cutoff, that specific neuron's 
        phase influence is insignificant to the average Kuramato vector because
        we consider the neuron's phase's center of gravity symmetric.
        The r_cutoff is used to adjust for the floating point error created
        when calculating the phase angle [radian] using arctan.

    Returns: 
      r: [radian] Magnitude of the mean Kuramato vectors of all neurons.

    NOTE (Tony):
      - The variable `lb` should be renamed as the starting point.
      - 1000 periods if time step (dt) is 0.1 (1000 / 0.1 = 1e4)
      - Converting the
      - The held_neuron is a reference neuron that we are referring all other 
        neurons to. The reference neuron is the reference frame that we compare 
        other neurons to.
      - The held_neuron is reset by the starting timestamp so that the held_neuron
        is viewed in the proper reference with respect to the interval we are
        looking back to view or plot.
      - The held_neuron is just the neuron that we are calculating. It is set as
        the held_neuron when a new neuron is first encountered when iterating 
        through the sorted spike record.
      - When held_neuron is reset to the starting timestamp, we are just changing
        the reference frame to that of the lookBack time interval.
      - The period of a periodic function is the value at which the function
        repeat itself, thus we are stating that at value (100 / dt) = 1000 is when 
        the function repeats.
      - The case when the ix neuron IS an held_neuron, it is limiting the frame
        to that specified by the lookBack and then projecting the entire spike
        acitivity of that neuron onto a single period. Each of the spike would be
        an arm on the polar coordinate thus has a radian, and this radian is 
        recorded into the phase_entries array to be later processed.
      - `myarm` is named because each of the spike can be projected onto the polar
        coordinate as an arm.
      - held_neuron is a variable that is outside of the for-loop and is reset 
        when finished processing all the phases for the previous neuron by setting
        to the current neuron number (i.e., ix). The phase_entries array is also
        cleared to make room for the next neuron number.
      - The phase medians are like each neuron's center of gravity (COG) in the 
        polar coordinate, however, I am not sure why we are not taking the length
        from center to the COG into the calculation.
      - Summing the Euler's equation is equivalent to summing each of the complex
        numbers representing the Kuramato vector denoting the mean phase of each
        neuron. Summing each neuron's Kuramato vector and then dividing by the 
        number of neuron becomes finding the phase mean of all the neurons.
      - The absolute mean phase in radian is returned because at the midpoint of 
        the period, the periodic function is just opposite of that of t=0. We do
        not care whether the mean phase is lagging or ahead of the the periodic
        function at t=0, thus we only care about the magnitude of the phase, hence
        taking the absolute value of the mean phase of all neurons.
      - The cutoff value is tuned to adjust for floating point errors when 
        calculating the the phase angle using trigonometry. Seen below when
        finding the phase angle [radian] using scipy's circmean, we are no longer
        using trigonometry to find the phase angle, thus floating point errors are
        eliminated.

    QUESTION (Tony):
      - SR is already in n-rows of length-2 arrays, why do we need to reshape it?
      - Why is the time length for calculating the period variable fixed at 100ms?
      - Why add 1 to the priod during linespace creation?
        - Answer: Because the evenly spaced samples includes the start and 
          endpoints. For example, if we would like 4 sections, we would need 5 
          marking points (3x points at the center and then the two ends).
      - What is held_neuron (smallest neuron in the spike record?)
      - What unit is the `period` variable? I think it should be ms, but it is not
        clear if there is even a unit.
      - Why is the radius/hypotenus or the distance from origin to the COG only
        filtered at the cutoff of 0.3? What is the significance of 0.3? Why not 
        consider this distance in calculating the Kuramato vector? 
      - ??? The `wraps` variable doesn't really make sense as it is dividing
        lookBack of unit [ms] by period of unit [count], in addition the variable 
        is not used nor returned, what is the point of this variable?
    """

    # Convert period and lookback from ms to euler-steps
    if period is None:
      period=100  # 100ms
    steps_in_period = period / self.dt  # Number of Euler steps in a period
    if lookback is None:
      lookback = self.t_current
    steps_to_lookback = lookback / self.dt

    lb = self.t_current - steps_to_lookback  # Analysis starting-point timestamp [ms]

    # Spike record
    SR = np.reshape(self.spike_record,newshape = [-1,2])  
    SR = np.delete(SR, 0, 0)         # Delete first row - The placeholder
    SRix = np.argmax(SR[:,1] >= lb)  # Index of the starting point
    SR = SR[SRix:,:]
    SR = sorted(SR,key=lambda x: x[0])  # Sort based on 1st column - neuron
    wraps = steps_to_lookback/steps_in_period

    N = self.n_neurons
  
    theta = np.random.normal(size=[N,])
    phasespace = np.linspace(0,2*np.pi,int(steps_in_period+1))  # includes both ends

    held_neuron = np.min(SR[:][0])  # held_neuron == "Neuron being analyzed"
    phase_entries = []  # Placeholder for the phases of spikes for each neuron
    phase_medians = np.zeros(shape=[N,])  # Radian of each neuron's COG

    ## Convert phase to polar coorindate system (rho = radius; phi = radian)
    for i in range(len(SR)):
      ix = SR[i][0]  # Neuron number being analyzed
      if ix != held_neuron:  # held_neuron is the reference neuron
        x = np.cos(phase_entries)
        y = np.sin(phase_entries)
        mx = np.mean(x)
        my = np.mean(y)
        rho = np.sqrt(mx**2 + my**2)  # List of hypotenuses or radius
        phi = np.arctan2(my, mx)      # Angle in radian between vector (1, 0)

        ## Record phi (radian) if rho (radius) meets the cutoff
        if rho >= r_cutoff:
          phase_medians[int(ix)] = phi
        else:
          phase_medians[int(ix)] = np.NaN
        
        ## Reset for next neuron
        held_neuron = ix
        phase_entries = []
      else:
        myarm = SR[i][1]-lb    # Set the frame specified by the start timestamp
        while myarm > steps_in_period:  # Project onto one period
          myarm = myarm - steps_in_period
        myarm = phasespace[int(np.round(myarm))]  # Phase of the spikes
        phase_entries.append(myarm)

    ## Filter out the NaN
    phase_medians = phase_medians[~np.isnan(phase_medians)]

    # e^{xi} = cos(x) + (i)*sin(x) -- Euler's Equation
    z = 1/N * np.sum(np.exp(1.0j * phase_medians))  # Phase mean of all neurons
    r = np.abs(z)  # Only care about the magnitude
    return r  # Mean phase of all neurons in [radian]

  
  def kuramato(self,
               period:float=None,
               lookback:float=None):
    """Return the phase mean of all neurons in the spike record.

    circmean() instead of trigonometry is used to find the mean phase of all 
    the spikes of a neuron, thus, there is higher precision, hence, eliminating
    the need of a r_cutoff argument to trim off the phase noise created by
    rounding floating points.

    Essentially, this is a more accurate version of the `vect_kuramato` method.

    Simulation duration has to >= period.

    Args: 
      period: [ms] Time length of a single period we define the mapped-to polar
        coordinates.
      lookback: [ms] The length of time we are looking back to analyze.
        If set to the default `None`, we are considering the entire span of the
        spike record.

    Returns: 
      r: [radian] Magnitude of the mean Kuramato vectors of all neurons.

    NOTE (Tony): 
      - The higher the period value, the higher the resolution in the calculation
      especially when converting to radians. Perhaps period of 100ms is adequate
      as it would divide 2pi into 1000 sections.
    """
    # Convert period and lookback from ms to euler-steps
    if period is None:
      period=100  # 100ms
    steps_in_period = period / self.dt  # Number of Euler steps in a period
    if lookback is None:
      lookback = self.t_current
    steps_to_lookback = lookback / self.dt

    lb = self.t_current - steps_to_lookback  # Analysis starting-point timestamp [ms]

    # Spike record
    SR = np.reshape(self.spike_record,newshape = [-1,2])
    SR = np.delete(SR, 0, 0)         # Placeholder value in first row
    SRix = np.argmax(SR[:,1] >= lb)  # Index of first instance
    SR = SR[SRix:,:]
    SR = sorted(SR,key=lambda x: x[0])
    wraps = steps_to_lookback/steps_in_period          # ??? What is this for? 

    N = self.n_neurons
  
    theta = np.random.normal(size=[N,])
    phasespace = np.linspace(0,2*np.pi,int(steps_in_period+1))  # Includes both ends

    held_neuron = np.min(SR[:][0])  # SR is sorted
    phase_entries = []
    phase_medians = np.zeros(shape=[N,])

    ## Analyze SR
    for i in range(len(SR)):
      ix = SR[i][0]
      if ix != held_neuron:
        phase_medians[int(ix)] = circmean(phase_entries)  # More acc than trig
        ## Reset for next neuron
        held_neuron = ix
        phase_entries = []
      else:
        myarm = SR[i][1]-lb    # Reframe to lookback
        while myarm > steps_in_period:  # Project onto the first period
          myarm = myarm - steps_in_period
        myarm = phasespace[int(np.round(myarm))]
        phase_entries.append(myarm)

    phase_medians = phase_medians[~np.isnan(phase_medians)]

    z = 1/N * np.sum(np.exp(1.0j * phase_medians))
    r = np.abs(z)  # Only care about the magnitude
    return r


  def update_g_noise(self, 
                     kappa_noise: float, 
                     method:str = "Ali") -> None:
    """Run the Dynamic noise conductivity update function.

    Args:
        kappa_noise (float): _description_
        method (str, optional): _description_. Defaults to "Ali".
    """
    # Generate Poisson noise
    self.simulate_poisson()
    poisson_noise_spiked_input_count = np.matmul(self.poisson_noise_spike_flag, self.network_conn)

    # Update conductivity (denoted g) - Integrate inputs from noise and synapses (Equation 6 from paper)
    if (method=="Tony"):
      ########## TONY ##########
      del_g_noise = (-self.g_noise 
                     + kappa_noise * self.tau_syn * poisson_noise_spiked_input_count) * np.exp(-self.dt/self.tau_syn)
      self.g_noise = (self.g_noise + del_g_noise)
    elif (method=="Ali"):
      ########## ALI ##########
      self.g_noise = (self.g_noise * np.exp(-self.dt/self.tau_syn) 
                      + self.g_poisson * self.poisson_noise_spike_flag)  # Poisson conductivity * poisson_input_flag makes sense because poisson_input_flag is binary outcome.
      

  def update_g_syn(self, 
                   step: int,
                   kappa: float, 
                   external_spiked_input_w_sums, 
                   method:str = "Ali") -> None: 
    """Run the Dynamic synaptic conductivity update function.

    Args:
        step (int): _description_
        kappa (float): _description_
        external_spiked_input_w_sums (_type_): _description_
        method (str, optional): _description_. Defaults to "Ali".
    """
    
    if (method=="Tony"):
      # Set variables
      per_neuron_coup_strength = kappa / self.n_neurons # [mS/cm^2] Neuron coupling strength (Network-coupling-strength / number-of-neurons)
      external_stim_coup_strength = kappa / 5  # Coupling strength of input external inputs (i.e., vibrotactile stimuli); value 5 is arbitrary for a strong coupling strength.
      del_g_syn = (-self.g_syn 
                   + per_neuron_coup_strength * self.tau_syn * self.spiked_input_w_sums 
                   + external_stim_coup_strength * external_spiked_input_w_sums[step, :]) * np.exp(-self.dt/self.tau_syn) 
      self.g_syn = (self.g_syn + del_g_syn)
    elif (method=="Ali"):
      self.g_syn = (self.g_syn * np.exp(-self.dt/self.tau_syn)
              + kappa/self.n_neurons * self.spiked_input_w_sums)
      

  def update_v(self, 
               step: int, 
               euler_steps: int,
               I_stim: npt.NDArray,
               method:str="Ali",
               capacitance_method:str="Ali") -> None: 
    """Run the dynamic membrane potential update function.

    Args:
        step (int): _description_
        euler_steps (int): _description_
        I_stim (npt.NDArray): _description_
        method (str, optional): _description_. Defaults to "Ali".
        capacitance_method (str, optional): _description_. Defaults to "Ali".
    """

    # External stimulation current input matrix
    if (type(I_stim) == np.ndarray):
      if (I_stim.any() != None): 
        ...
    elif I_stim == None:
      I_stim = np.zeros(shape=(euler_steps, self.n_neurons))

    assert type(I_stim) == np.ndarray, "The I_stim matrix has to be a numpy ndarray."

    # Variable value to use depending on Ali or the Paper's implementation
    if (capacitance_method == "Ali"):
      # NOTE: This is how Ali have his code. The tau_m variable doesn't really make sense.
      #   Discussed with Jesse and can't quite figure it out.
      capa_rv = self.tau_m        # What Ali used, much fewer network weight updates thus runs much faster  >>>> Jesse recommends Tony to look into STN firing frequency to validate which one to use for PD's STN.
      # g_leak = 0.02  # MISTAKE!!! ALI DECLARED THIS AS 10!!! NEED TO RESIM!
      g_leak = self.g_leak_ali  # 10
    elif capacitance_method == "Paper":
      # NOTE: This is how the variable is defined in the paper (equation 2)
      capa_rv = self.capacitance  # What the paper stated, many more network updates and runs VERY SLOW!
      # g_leak = 10  # MISTAKE!!! ONE OF FOUR OF THE SIMULATION RAN USED THIS VALUE AND TOOK FOREVER. I THINK THIS IS THE REASON!!!
      g_leak = self.g_leak


    # Different ways to update membrane potential
    if method == "Tony":
      I_noise = self.g_noise * (self.v_syn - self.v)
      del_v = (g_leak * (self.v_rest - self.v)
               + self.g_syn * (self.v_syn - self.v)
               + I_stim[step, :] 
               + I_noise) * (self.dt / capa_rv)
      self.v = (self.v + del_v)
    elif method == "Matteo":
      # NOTE: This code is erroneous, may be the reason why Matteo's result is not reproducing Ali's.
      self.v = self.v + (self.dt/capa_rv) * ((self.v_rest - self.v)
                              - (self.g_noise + self.g_syn) * self.v)
    elif method == "Ali":
      # NOTE: Confusing, but this fits the paper equation, just without I_stim and I_noise
      # NOTE: Ali eliminated v_rest=0mV and just mvoed the negative sign up the product.
      self.v = self.v + (self.dt/capa_rv) * (g_leak * (self.v_rest - self.v)
                              - (self.g_noise + self.g_syn) * self.v)
    elif method == "Ali_Na_rev_potential":
      # NOTE: This method was suggested by Jesse when on call discussing about the validity of the function.
      na_rev_potential = 20  # 20 mV
      self.v = (self.v + (self.dt/capa_rv) * (g_leak * (self.v_rest - self.v) 
                                                + (self.g_noise + self.g_syn) * (na_rev_potential - self.v)))


  def update_thr(self, 
                 method:str="Ali") -> None:
    """Run the dynamic spike threshold computation.

    Args:
        method (str, optional): _description_. Defaults to "Ali".
    """
    # Determine method of dyanmic threshold update implementation
    if method=="Tony":
      del_v_thr = (self.v_thr_rest - self.v_thr) * np.exp(-self.dt/self.tau_rf_thr)
      self.v_thr = (self.v_thr + del_v_thr)
    elif method=="Ali":
      self.v_thr = (self.v_thr + self.dt * (self.v_thr_rest - self.v_thr) / self.tau_rf_thr)


  def check_if_spike(self) -> None:
    """Check if neurons spike, and mark them as needing to update connection weight.
    """
    spike = ((self.v >= self.v_thr) *   # Met dynamic spiking threshold
              (self.spike_flag == 0))   # Not in abs_refractory period because not recently spiked
    
    self.spike_flag[spike] = 1                   # Mark them as SPIKED!
    self.w_update_flag[spike] = 1                # Mark them as "Needing to update weight"
    
    ## Keep track of spike times
    self.t_minus_1_spike[spike] = self.t_minus_0_spike[spike]  # Moves the t_minus_0_spike array into t_minus_1_spike for placeholding
    self.t_minus_0_spike[spike] = self.t_current              # t_minus_0_spike keeps track of each neuron's most recent spike's timestamp


  def spiking(self) -> None:
    """Simulate the spiking stage with a rectangular spike shape of v_spike for tau_spike ms.

    The paper assumes and uses a rectangular spike shape, however, this only 
    happens if we are voltage gating instead of spiking due to current input.

    ??? When a postsynaptic neuron spike, is the input from the presynaptic neuron
    more like a voltage clamp or more like a current? The former would make sense
    for neurons live in a high resistance extracellular space, however, the latter
    is more analogous to neurotransmitter transmitting signal from neuron to
    neuron.

    TODO (Tony):
      - Check literature if STN neurons are more similar to voltage or current 
        input (neurotransmitters).
    """

    # Depolarization phase
    spiked = (self.spike_flag == 1)
    self.v[spiked] = self.v_spike         # Rectangle spike shape by setting voltage to V_spike for duration of tau_spike (equation 3)
    self.v_thr[spiked] = self.v_rf_spike  # Threshold is reset to V_th_spike=0mV right after spiking (equation 3)

    # Hyperpolarization phase
    in_abs_rf_period = (self.t_minus_1_spike + self.tau_spike) > self.t_current
    self.v[(~in_abs_rf_period) * spiked] = self.v_reset  # Rectangular spike end resets potential to -67mV
    
    # Reset spike flag tracker
    self.spike_flag[(~in_abs_rf_period) * spiked] = 0

  def check_presynaptic_spike_arrival(self) -> None:
    """Checking and updating if spike from presynaptic neurons have arrived.

    This aligns with the Dirac Delta Distribution in equation 4 of the paper.
    The intent to to check if the last spiking time plus synaptic delay (time
    it takes a signal to propagate from the presynaptic neuron's soma to the
    postsynaptic neuron's soma) has arrived found by finding the the difference
    between the current time and previous step's time plus delay.

    If so, we then sum up the pre-to-post weights of all the CONNECTED and 
    SPIKES THAT HAVE ARRIVED [at postsynaptic soma] for each postsynaptic neuron
    and this value is used to update the synaptic conductivity.
    """
    ## Dirac Delta Distribution (equation 4 in paper)
    t_diff = self.t_minus_0_spike - (self.t_minus_1_spike + self.synaptic_delay)  # [n, ] array
    s_flag = 1.0 * (abs(t_diff) < 0.01)  # 0.01 for floating point errors

    # Presynaptic neurons' weight sum for each neuron
    start = time.time()
    # element_wise = self.network_W * self.network_conn
    element_wise = np.multiply(self.network_W, self.network_conn)
    print(f"{' '*15}element-wise calc time: {(time.time()-start)*1000} ms")

    start = time.time()
    self.spiked_input_w_sums = np.matmul(s_flag, element_wise)
    print(f"{' '*15}matmul calc time: {(time.time() - start)*1000} ms")

  def run_stdp_on_all_connected_pairs(self, )-> None:
    """Checks all connected pairs and update weights based on STDP scheme.


    NOTE (Tony): 
    - Performance check reveals that this function call takes the longest time
      among all the functions called by `simulate()`, thus a performance
      bottlen
    - FOUND ERROR: THE LOOP IS EVALUATING ALL CONNECTION TWICE BECAUSE OF THE BACKPROPAGATION.
    """
    ## STDP (Spike-timing-dependent plasticity)
    ## Note: Iterates over all pairs of connections using double-nested loops
    if self.w_update_flag.any():
      for pre_idx in range(self.n_neurons):
        
        if (self.w_update_flag[pre_idx] == 1):
          # Add spike record --> MOVE TO SPIKING
          self.spike_record = np.append(self.spike_record,
                                        np.array([pre_idx, self.t_current]))

          for post_idx in range(self.n_neurons):

            # Check last spike of pre-synaptic partners (forward propagation):
            if self.network_conn[pre_idx][post_idx] == 1:
              temporal_diff = (self.t_minus_0_spike[pre_idx] + self.synaptic_delay 
                                - self.t_minus_1_spike[post_idx])
              dw = stdp_dw(temporal_diff)
              self.dW += dw
              update_w_matrix(self.network_W, dw, pre_idx, post_idx)

            # # Inform post-synaptic partners about spike (backpropagation):
            # elif self.network_conn[post_idx][pre_idx] == 1:  
            #   temporal_diff =  (self.t_minus_0_spike[post_idx] + self.synaptic_delay 
            #                     - self.t_minus_1_spike[pre_idx])
            #   self.dW = self.dW + self.stdp_weight_update(temporal_diff, post_idx, pre_idx)
            #   # self.stdp_weight_update(temporal_diff, post_idx, pre_idx)

  def simulate(self, 
               sim_duration: float = 1, 
               I_stim: npt.NDArray = None,
               external_spiked_input_w_sums: npt.NDArray = None,
               kappa: float = 8,
               kappa_noise: float = 0.026,
               temp_param:dict=None):
    """Run simulation

    Args: 
      sim_duration: [ms] Duration of time in milliseconds.
      I_stim (ndarray): 
        2D ndarray matrix denoting the current input for each neuron per each
        epoch (Euler-step); current in mA.
      external_spiked_input_w_sums (ndarray): 
        [mS/cm^2] 2D ndarray matrix denoting the weight sum of spiked and  
        connected presynaptic neurons for each neuron per epoch. Rows are for 
        each epoch, and columns for each postsynaptic neuron.
      kappa (float, optional): Max coupling strength of the network [mS / cm^2].
        Defaults to 8.
      kappa_noise (float, optional): Scales noise intensity [mS / cm^2]. 
        Defaults to 0.026.

    Returns: 
      holder_v (ndarray): 
        2D matrix of the membrane potential of each neuron at 
        each epoch (Euler-step).
      holder_g_syn (ndarray): 
        2D matrix of the conductivity of each neuron at each epoch (Euler-step).
      holder_poi_noise_flags (ndarray): 
        2D matrix of binary outcomes (spike vs not-spike) of each neuron 
        at each epoch (Euler-step).
      holder_epoch_timestamps (ndarray): Timestamps of each epoch (Euler-step).
      holder_spiked_input_w_sums (ndarray): 
        2D matrix of the weight sum of spiked and connected presynaptic neuron
        for each neuron at each epoch (Euler-step). 
        Rows are for each epoch, and columns are for each post-synaptic neuron.
      holder_dw (ndarray): 
        1D array of the net connection weight change of the entire network 
        at each epoch (Euler-step).
    """
    
    euler_steps = int(sim_duration/self.dt)   # Number of Euler-method steps
    euler_step_idx_start = self.t_current / self.dt  # Euler-step starting index


    # Weight sums of external spiked and connected input (conductivity)
    if external_spiked_input_w_sums == None: 
      external_spiked_input_w_sums = np.zeros(shape=(euler_steps, self.n_neurons))

    # Output variable placeholders
    holder_epoch_timestamps = np.zeros((euler_steps, ))
    holder_v = np.zeros((euler_steps, self.n_neurons))
    holder_g_syn = np.zeros((euler_steps, self.n_neurons))
    holder_poi_noise_flags = np.zeros((euler_steps, self.n_neurons))
    holder_spiked_input_w_sums = np.zeros((euler_steps, self.n_neurons))
    holder_dw = np.zeros((euler_steps, ))

    # Euler-step Loop
    for step in range(euler_steps):  # Step-loop: because (time_duration/dt = steps OR sections)
      print(step)
      # Dynamic Function Update Poisson noise's conductivity
      self.update_g_noise(kappa_noise=kappa_noise, method=temp_param["update_g_noise_method"])  
      # Dynamic Function Update synaptic conductivity
      self.update_g_syn(step=step, 
                        kappa=kappa, 
                        external_spiked_input_w_sums=external_spiked_input_w_sums, 
                        method=temp_param["update_g_syn_method"])
      # Reset variables
      self.spiked_input_w_sums = np.zeros(self.n_neurons)     # Weight sum of all spiked-connected presynaptic neurons
      self.w_update_flag = np.zeros(self.n_neurons)           # Connection weight update tracker
      self.dW = 0                                                  # Net connection weight change per epoch
      # Dynamic Function Update membrane potential
      timer.time_perf(self.update_v)(step=step, 
                    euler_steps=euler_steps,
                    I_stim = I_stim, 
                    method=temp_param["update_v_method"], 
                    capacitance_method=temp_param["update_v_capacitance_method"])
      # Dynamic Function Update spike threshold
      timer.time_perf(self.update_thr)(method=temp_param["update_thr_method"])
      
      # Check if the neurons spike and mark them as needing to update conn weight
      timer.time_perf(self.check_if_spike)()

      # Depolarization and Hyperpolarization (rectangular spike shape)
      timer.time_perf(self.spiking)()
      
      # Update the variable needed for next step's g_syn calculation
      timer.time_perf(self.check_presynaptic_spike_arrival)()

      # Updates the network_W and dW
      timer.time_perf(self.run_stdp_on_all_connected_pairs)()
      # timer.time_perf(stdpScheme.conn_update_STDP)(self.network_W, self.network_conn,
      #                             self.w_update_flag, self.synaptic_delay,
      #                             self.t_minus_1_spike, self.t_minus_0_spike)

     

      # End of Epoch:
      # NOTE: Used so that multiple simulation runs have continuity.
      tix = int(self.euler_step_idx - euler_step_idx_start)
      holder_epoch_timestamps[tix] = self.t_current
      holder_v[tix] = self.v   
      holder_g_syn[tix] = self.g_syn
      holder_poi_noise_flags[tix] = self.poisson_noise_spike_flag
      holder_spiked_input_w_sums[tix] = self.spiked_input_w_sums
      holder_dw[tix] = self.dW

      # Increment Euler-step index
      self.euler_step_idx += 1

      ## Increment time tracker
      self.t_current += self.dt
    
    return (holder_v, 
            holder_g_syn, 
            holder_poi_noise_flags, 
            holder_epoch_timestamps, 
            holder_spiked_input_w_sums, 
            holder_dw)