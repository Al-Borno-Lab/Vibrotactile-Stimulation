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
from src.model import stdpScheme

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
    self.dW = 0                                                                             # Tracker of change of weight used by `__run_stdp_on_all_connected_pairs()`
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

## COMMENT OUT TO BE ABLE TO COLLAPSE -- NEED SOME FIXING
  def __structured_conn(self, LIF, mean_w: float=0.5):
    # """To connect the neurons according to a seemingly exponential distribution.

    # NOTE (Tony): 
    # I am not entirely clear what the intent is behind this kind of connectivity.
    # However, the calculation here is redundant and inefficient, and is O(N^2).

    # TODO (Tony): 
    #   - Object method implemented incorrectly and is asking for the lifNetwork object.
    #   - Fix this method.

    # Args:
    #     LIF (_type_): _description_
    #     mean_w (float, optional): The mean connection weight to normalize each
    #     connection in the network to. Defaults 0.5.

    # Returns:
    #     _type_: _description_
    # """

    # self.network_conn = np.zeros([self.n_neurons,self.n_neurons])  
    # dist=np.empty([self.n_neurons,self.n_neurons])
    # dist[:] = np.nan
    # dist1=[]
    # c=[]
    # for f in range(LIF.n_neurons-1):
    #   i=f
    #   for j in range(f+1,LIF.n_neurons,1):
    #     a=(LIF.x[i]-LIF.x[j])*(LIF.x[i]-LIF.x[j])+(LIF.y[i]-LIF.y[j])*(LIF.y[i]-LIF.y[j])+(LIF.z[i]-LIF.z[j])*(LIF.z[i]-LIF.z[j])
    #     b=np.sqrt(a)
    #     c.append(b)
    #   #print('distance between neurons', i+1, 'and', j+1, ': ', b)
    # d=sum(c)/len(c)
    # print('The average distance between neurons in this network is:', d)
    # print('The base of the exponent is:', LIF.p_conn**(1/d))
    # bb=[]
    # cc=[]
    # for p in range(LIF.n_neurons):
    #   for p2 in range(LIF.n_neurons):
    #     if(p!=p2):
    #       a=(LIF.x[p]-LIF.x[p2])*(LIF.x[p]-LIF.x[p2])+(LIF.y[p]-LIF.y[p2])*(LIF.y[p]-LIF.y[p2])+(LIF.z[p]-LIF.z[p2])*(LIF.z[p]-LIF.z[p2])
    #       b=np.sqrt(a)
    #       dist1.append(b)
    # for p in range(LIF.n_neurons):
    #   for p2 in range(LIF.n_neurons):
    #     if(p!=p2):
    #       a=(LIF.x[p]-LIF.x[p2])*(LIF.x[p]-LIF.x[p2])+(LIF.y[p]-LIF.y[p2])*(LIF.y[p]-LIF.y[p2])+(LIF.z[p]-LIF.z[p2])*(LIF.z[p]-LIF.z[p2])
    #       b=np.sqrt(a)
    #       #aa=((LIF.p_conn**(1/d)) ** b)
    #       aa=2.71828**(-b/(LIF.p_conn*max(dist1)))
    #       pc = np.random.random(size=(1,))
    #       if(pc<aa):
    #         self.network_conn[p][p2] = 1
    #         dist[p][p2]=b

    # ## TODO (Tony): Verify that this block below is indeed not needed.
    # # self.network_W = np.random.random(size=(self.n_neurons,self.n_neurons))
    # # self.network_W[self.network_conn == 0] = 0
    # # self.network_W = self.network_W/np.mean(self.network_W[self.network_W > 0]) * mean_w
    # # self.network_W[self.network_W > 1] = 1
    # # self.network_W[self.network_W < 0] = 0
    # return dist
    ...

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

  def __simulate_poisson(self, 
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

    if not ((n_per_second > 100) & (p_approx < 0.01)):
      raise Exception("""time-step is too large causing the Poisson noise binomial estimation to be inaccurate. 
      See `__simulate_poisson` method definition for more details.""")

    # Generate Poisson noise spike flags
    self.poisson_noise_spike_flag = np.random.binomial(n=1, p=p_approx,
                                                       size=(self.n_neurons,))

  def __check_if_spiked(self) -> None:
    """Check if neurons spike, and mark them as needing to update connection weight.
    """
    spike = ((self.v >= self.v_thr)     # Met dynamic spiking threshold
             * (self.spike_flag == 0))  # Not in abs_refractory period because not recently spiked
    
    self.spike_flag[spike] = 1                   # Mark them as SPIKED!
    self.w_update_flag[spike] = 1                # Mark them as "Needing to update weight"
    
    ## Keep track of spike times
    self.t_minus_1_spike[spike] = self.t_minus_0_spike[spike]  # Moves the t_minus_0_spike array into t_minus_1_spike for placeholding
    self.t_minus_0_spike[spike] = self.t_current              # t_minus_0_spike keeps track of each neuron's most recent spike's timestamp

  def __rectangular_spiking(self) -> None:
    """Simulate a rectangular spike of v_spike mV for tau_spike ms.
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

  def __calc_spiked_input_w_sums(self) -> None:
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
    # start = time.time()  # DEBUG
    element_wise = np.multiply(self.network_W, self.network_conn)
    # print(f"{' '*15}element-wise calc time: {(time.time()-start)*1000} ms")  # DEBUG

    # start = time.time()  # DEBUG
    self.spiked_input_w_sums = np.matmul(s_flag, element_wise)
    # print(f"{' '*15}matmul calc time: {(time.time() - start)*1000} ms")  # DEBUG

  def __run_stdp_on_all_connected_pairs(self, )-> None:
    """Checks all connected pairs and update weights based on STDP scheme.

    This double-nested for-loop checks through all pairs, and the four cases 
    (A, B, C, D) allows for checking of a spiked neuron's pre- and post-synaptic
    partners. 

    Case B is specifically for when synaptic delay = 0, which usually would not
    happen in our current model as all neurons in this current model are 
    homogenous.
    """
    if not self.w_update_flag.any():
      return

    for i in range(self.n_neurons):
      if (self.w_update_flag[i] == 1):  ### SPOTLIGHT ###
        self.spike_record = np.append(self.spike_record,np.array([i, self.t_current]))

        for j in range(self.n_neurons):
          
          # NOTE:
          # There is a little bit of asymmetry here, that in the case A and B, 
          # the temporal diff is always greater equal than 0 (the case for equal
          # to zero is B and it is not supposed to happen because it assumes
          # synaptic delay == 0).
          #
          # For C and D, the temporal difference straddles 0 and thus makes sense
          # to use <0 and >=0 in the if-else to differentiate whether the presynaptic
          # partner of the neuron in spotlit (i) is being transmitted.
          # However, the if-else cannot tell the difference between whether 
          # spike from the presynaptic partner is still in transit or it never spiked.

          ### Spotlight is on i ### SPIKED neuron connecting to others
          # if i is pre-synaptic to j, update W(i,j)
          if self.network_conn[i][j] == 1:

            # Check if j has a spike in transit, and if so, use the spike before last:
            # Smallest value is syn_delay; range: [syn_delay, t+syn_delay]
            temporal_diff = self.t_minus_0_spike[i] - self.t_spike2[j]   + self.synaptic_delay
            # <<<<<<<<<< IGNORE - DEBUG USE
            # print(f"{' '*10} {self.t_spike2[i]} - {self.t_spike2[j]} + {self.synaptic_delay}")
            # >>>>>>>>>> IGNORE - DEBUG USE
            
            # Case A
            if temporal_diff > 0:  # ??? temporal_diff >= 0 ??? - Why not triage like Case C and D? 
              # - i has spike in transit (both spiked at the same time or j spiked no more than delay-time ago)
              # - LTD always, regardless of when j spiked
              dW = dW + self.Delta_W_tau(temporal_diff,i,j)
              # <<<<<<<<<< IGNORE - DEBUG USE
              # print(f"A: STDP on {i} -> {j} at eulerstep {t} of time {self.t} ms")
              # print(f"{' '*10} {self.t_spike2[i]} - {self.t_spike2[j]} + {self.synaptic_delay}")
              # print(f"{' '*10} t_spike2[{i}]-t_spike2[{j}]: temporal_diff: {temporal_diff} ms")
              # >>>>>>>>>> IGNORE - DEBUG USE
            
            # Case B
            else:
              # - This can only happen is synaptic delay = 0 
              # - And this is always LTD
              temporal_diff = self.t_minus_0_spike[i] - self.t_minus_1_spike[j] + self.synaptic_delay
              dW = dW + self.Delta_W_tau(temporal_diff,i,j)
              # <<<<<<<<<< IGNORE - DEBUG USE
              # print(f"B: STDP on {i} -> {j} at eulerstep {t} of time {self.t} ms")
              # print(f"{' '*10} {self.t_minus_0_spike[i]} - {self.t_minus_1_spike[j]} + {self.synaptic_delay}")
              # print(f"{' '*10} t_minus_0_spike[{i}]-t_minus_1_spike[{j}]: temporal_diff: {temporal_diff} ms")
              # >>>>>>>>>> IGNORE - DEBUG USE

          ### Spotlight is on i ### SPIKED neuron receiving connection
          # if j is pre-synaptic to i, update W(j,i)
          if self.network_conn[j][i] == 1: 
            
            # check if j has a spike in transit, and if so, use the spike before last:
            # Largest value is syn_delay; range: [syn_delay-t-10000, syn_delay]
            temporal_diff =  self.t_minus_0_spike[j] - self.t_minus_0_spike[i] + self.synaptic_delay
              # <<<<<<<<<< IGNORE - DEBUG USE
            # print(f"{' '*10} {self.t_minus_0_spike[j]} - {self.t_minus_0_spike[i]} + {self.synaptic_delay}")
              # >>>>>>>>>> IGNORE - DEBUG USE
            
            # Case C
            if temporal_diff < 0: 
              # - j's spike arrived at i before i spiked, thus LTP
              dW = dW + self.Delta_W_tau(temporal_diff,j,i)
              # <<<<<<<<<< IGNORE - DEBUG USE
              # print(f"C: STDP on {j} -> {i} at eulerstep {t} of time {self.t} ms")
              # print(f"{' '*10} t_minus_0_spike[{j}]-t_minus_0_spike[{i}]: temporal_diff: {temporal_diff} ms")
              # print(f"{' '*10} {self.t_minus_0_spike[j]} - {self.t_minus_0_spike[i]} + {self.synaptic_delay}")
              # >>>>>>>>>> IGNORE - DEBUG USE
            
            # Case D
            else: 
              # - j has spike in transit (both spiked at the same time or j spiked no more than delay-time ago)
              # - Can be LTP or LTD, really depends when the last time j spiked.
              #   - LTD if j's previous spike is less than delay-time ago from i's current spike.
              #   - LTP if j's previous spike is more than delay-time ago from i's current spike.
              temporal_diff = self.t_minus_1_spike[j] - self.t_minus_0_spike[i] + self.synaptic_delay
              dW = dW + self.Delta_W_tau(temporal_diff,j,i)
              # <<<<<<<<<< IGNORE - DEBUG USE
              # print(f"D: STDP on {j} -> {i} at eulerstep {t} of time {self.t} ms")
              # print(f"{' '*10} {self.t_minus_1_spike[j]} - {self.t_minus_0_spike[i]} + {self.synaptic_delay}")
              # print(f"{' '*10} t_minus_1_spike[{j}]-t_minus_0_spike[{i}]: temporal_diff: {temporal_diff} ms")
              # >>>>>>>>>> IGNORE - DEBUG USE

  def __update_g_noise(self, kappa_noise:float) -> None:
    """Calculate and update noise conductivity using method by Ali's code.

    This is the method used by Ali in his code, which diverges from what was 
    laid out in his paper. In his paper, kappa_noise (conductivity) is set as 
    0.026, hence would have made the noise conductivity update much slower 
    thus requiring longer simulation duration to see the same effect.

    Although this method updates the noise conducitivity much faster, it is
    making additional assumptions in that the noise conductivity is closer to 1
    as opposed to the 0.026 laid out in his paper.

    NOTE (Tony): This method is calculating the dynamic noise conductivity for the
     current euler-step, whereas the `_update_g_noise_tony` calculates the next
     step's conductivity. The difference is minor in the grand scheme of things
     when we allow some time for the neural network to stablize.

    Args:
        kappa_noise (float): [mS/cm^2] Noise conductivity.
    """
    # Generate Poisson noise
    self.__simulate_poisson()
    poisson_noise_spiked_input_count = np.matmul(self.poisson_noise_spike_flag, 
                                                 self.network_conn)
    # Update conductivity (denoted g) - 
    # Integrate inputs from noise and synapses (Equation 6 from paper)
    self.g_noise = (self.g_noise * np.exp(-self.dt/self.tau_syn) 
                    + self.g_poisson * self.poisson_noise_spike_flag)  # Poisson conductivity * poisson_input_flag makes sense because poisson_input_flag is binary outcome.
  
  def __update_g_noise_tony(self, kappa_noise:float) -> None:
    """Calcualte and update noise conductivity using method laid out in Ali's paper.

    This method is derived from equation 6 of Ali's paper with modification to 
    incorporate exponential decay. As according to the paper's eqaution, the 
    noise conductivity of 0.026 mS/cm^2 is incorporated into this update 
    function.

    The consequence of including the noise conducitivty value set forth in Ali's 
    paper is that the noise conductivity updates very slowly (factor of 38x 
    slower), and thus results in the neural network simulation taking much 
    longer time.

    NOTE (Tony): This method is calculating the dynamic noise conductivity for the
     next euler-step, whereas the `_update_g_noise` calculates the current
     step's conductivity. The difference is minor in the grand scheme of things
     when we allow some time for the neural network to stablize.

    Args:
        kappa_noise (float): [mS/cm^2] Noise conductivity.
    """
    # Generate Poisson noise
    self.__simulate_poisson()
    poisson_noise_spiked_input_count = np.matmul(self.poisson_noise_spike_flag, 
                                                 self.network_conn)
    # Update conductivity (denoted g) - 
    # Integrate inputs from noise and synapses (Equation 6 from paper)
    del_g_noise = (np.exp(-self.dt/self.tau_syn)
                   * (- self.g_noise 
                      + kappa_noise 
                        * self.tau_syn * poisson_noise_spiked_input_count))
    self.g_noise = (self.g_noise + del_g_noise)

  def __update_g_syn(self, kappa:float=400, 
                     external_stim_coup_strength:float=None, step:int=None, 
                     external_spiked_input_w_sums:npt.NDArray=None) -> None: 
    """Calculate and update synaptic conductivity using methods from Ali's code.

    This method is translated from Ali's code, of which kappa was set at 400.
    kappa=400 is a 50x that of the value set forth in the paper (8), and the 
    result of this larger kappa value is a stronger coupling and thus faster
    synaptic conductivity update.

    However, concern with using this method is that it deviates from what is 
    set forth by Ali's paper and makes the assumption of neuron max total 
    coupling strength to be 400 mS/cm^2 instead of 8 mS/cm^2 and hence may 
    significantly deviate from actual physiological results when the neural
    network is simulated on a larger scale than few thousand neurons.

    NOTE (Tony): This method is calculating the dynamic conductivity for the
     current euler-step, whereas the `_update_g_syn_tony` calculates the next
     step's conductivity. The difference is minor in the grand scheme of things
     when we allow some time for the neural network to stablize.

    Args:
        kappa (float, optional): 
         [mS/cm^2] Max coupling strength. Defaults to 400.
        external_stim_coup_strength (float, optional): 
         Coupling strength of external stimulations. When the value is None,
         the value is converted to kappa/5 to simulate a strong coupling.
         Defaults to None.
        step (int, optional): 
         The current euler-step that calls this method; only needed
         when `external_spiked_input_w_sums` is provided.
         Defaults to None. 
        external_spiked_input_w_sums (npt.NDArray, optional): 
         The sum of spiked presynaptic connection weights of each neuron. 
         When using the default value None, the value is converted to 0 for each
         neuron equivalent to none of the presynaptic neurons for each neuron 
         spiked. Defaults to None.
    """
    # Set variables
    if external_spiked_input_w_sums is None: 
      external_spiked_input_w_sums_step = np.zeros(shape=(self.n_neurons, ))
    else:
      if step is None: 
        raise AssertionError("`step` arg is needed if `external_spiked_input_w_sums` is provided.")
      external_spiked_input_w_sums_step = external_spiked_input_w_sums[step, :]
    if external_stim_coup_strength is None: 
      # Coupling strength of input external inputs (i.e., vibrotactile stimuli); 
      # value 5 is arbitrary for a strong coupling strength.
      external_stim_coup_strength = kappa / 5
    
    # Calculations
    self.g_syn = (self.g_syn * np.exp(-self.dt/self.tau_syn)
                  + kappa/self.n_neurons * self.spiked_input_w_sums
                  + external_stim_coup_strength * external_spiked_input_w_sums_step)
       
  def __update_g_syn_tony(self, kappa:float=8, 
                          external_stim_coup_strength:float=None, step:int=None, 
                          external_spiked_input_w_sums:npt.NDArray=None) -> None: 
    """Calculate and update synaptic conductivity using methods from Ali's paper.

    This method is adapted from equation 4 of Ali's paper and utilizes the 
    max coupling strength (kappa) of 8 mS/cm^2.

    When using this method compared to that of `__update_g_syn`, the synaptic
    conductance updates much slowly because the other method uses a kappa value
    of 400.

    Additionally, this method has expanded its functionality to allow for
    external stimulation inputs.

    NOTE (Tony): This method is calculating the dynamic conductivity for the
     next euler-step, whereas the `_update_g_syn` calculates the current
     step's conductivity. The difference is minor in the grand scheme of things
     when we allow some time for the neural network to stablize.

    Args:
        kappa (float, optional): 
         [mS/cm^2] Max coupling strength. Defaults to 8.
        external_stim_coup_strength (float, optional): 
         Coupling strength of external stimulations. When the value is None,
         the value is converted to kappa/5 to simulate a strong coupling.
         Defaults to None.
        step (int, optional): 
         The current euler-step that calls this method; only needed
         when `external_spiked_input_w_sums` is provided.
         Defaults to None. 
        external_spiked_input_w_sums (npt.NDArray, optional): 
         The sum of spiked presynaptic connection weights of each neuron. 
         When using the default value None, the value is converted to 0 for each
         neuron equivalent to none of the presynaptic neurons for each neuron 
         spiked. Defaults to None.
    """
    # Set variables
    if external_spiked_input_w_sums is None: 
      external_spiked_input_w_sums_step = np.zeros(shape=(self.n_neurons, ))
    else:
      if step is None: 
        raise AssertionError("`step` arg is needed if `external_spiked_input_w_sums` is provided.")
      external_spiked_input_w_sums_step = external_spiked_input_w_sums[step, :]
    if external_stim_coup_strength is None: 
      # Coupling strength of input external inputs (i.e., vibrotactile stimuli); 
      # value 5 is arbitrary for a strong coupling strength.
      external_stim_coup_strength = kappa / 5
    # [mS/cm^2] Neuron coupling strength (Network-coupling-strength / number-of-neurons)
    per_neuron_coup_strength = kappa / self.n_neurons
    
    # Update membrane conductivity
    del_g_syn = (np.exp(-self.dt/self.tau_syn)
                 *(-self.g_syn 
                   + per_neuron_coup_strength * self.tau_syn * self.spiked_input_w_sums 
                   + external_stim_coup_strength * external_spiked_input_w_sums_step))
    self.g_syn = (self.g_syn + del_g_syn)
    
  def __update_v(self, 
                 capacitance:npt.NDArray=None,
                 g_leak:float=10) -> None:
    """Calculates and updates the membrane potential with methods from Ali's code.

    This method updates the membrane potential using the logic from Ali's code
    and differs from the methods laid out in his methods in the following ways:
      - Capacitance was drawn from a normal distribution centered at 150 instead 
        of 3 as laid out in his paper.
      - g_leak has a value of 10 mS/cm^2 instead of 0.02 as laid out in his paper.

    The impact of scaling the values up from what was laid out in the paper is 
    faster changes and thus shorter simulation time, however, the concern is 
    that by scaling the values up, we are making the assumptions that capacitances
    are around 150 microFarad/cm^2 and g_leak is around 10 (instead of 3 and 0.02).

    Args:
        capacitance (npt.NDArray, optional): [microFarad/cm^2]
          An array of n_neuron length indicating the capacitance of each neuron.
          When at default `None`, the tau_m array is extracted from the object 
          attribute. Defaults to None.
        g_leak (float, optional): [mS/cm^2]
          The leaky conductivity. Defaults to 10.
    """
    # Set variables
    if capacitance is None: 
      capacitance = self.tau_m

    # Calculate membrane potential
    # NOTE: Confusing, but this fits the paper equation, just without I_stim and I_noise
    # NOTE: Ali eliminated v_rest=0mV and just mvoed the negative sign up the product.
    self.v = self.v + (self.dt/capacitance) * (g_leak * (self.v_rest - self.v)
                            - (self.g_noise + self.g_syn) * self.v)
    
  def __update_v_tony(self, 
                      step:int=None, 
                      external_current_stim: npt.NDArray=None,
                      capacitance:npt.NDArray=None,
                      g_leak:float=0.02,) -> None:
    """Calculates and updates the membrane potentials

    This method calculates and updates the membrane potential and assumes that 
    the capacitance for each neuron is a random variable drawn from a normal 
    distribution (mu=3, stdev=0.15) as specified by Ali's paper.

    Additionally, the leaky conductivity is default to 0.02 mS/cm^2.

    Comparing with the values set in Ali's code (capacitance~150, g_leak=10),
    the update steps when using this function is much smaller and thus takes 
    much longer to simulate.

    However, it is unclear why these values were chosen in Ali's code.

    Args:
        step (int, optional): 
          The n-th euler-step the calculation is on. This value is only needed 
          when the `external_current_stim` argument is provided. 
          Defaults to None.
        external_current_stim (npt.NDArray, optional): [microAmpere]
          A euler-steps by n_neurons matrix indicating the external stimulation
          current injected at specified time to specified neuron. 
          When at default `None`, it assumes no external stimulation current.
          Defaults to None.
        capacitance (npt.NDArray, optional): [microFarad/cm^2]
          An array of n_neuron length indicating the capacitance of each neuron.
          When at default `None`, the value is taken from the object attribute, 
          which draws from a normal distribution (mu=3, stdev=0.15) for each
          neuron. Defaults to None.
        g_leak (float, optional): [mS/cm^2]
          The leaky conductivity. Defaults to 0.02.
    """    
    # Set variables
    if external_current_stim == None:
      external_current_stim_step = np.zeros(shape=(self.n_neurons, ))
    else: 
      if step is None: 
        raise AssertionError("`step` arg is needed if `external_current_stim` is provided.")
      external_current_stim_step = external_current_stim[step, :]
    # Capacitance is drawn from a normal distribution
    if capacitance is None: 
      capacitance = self.capacitance
      
    # Update membrane potential
    I_noise = self.g_noise * (self.v_syn - self.v)
    del_v = (g_leak * (self.v_rest - self.v)
              + self.g_syn * (self.v_syn - self.v)
              + external_current_stim_step
              + I_noise) * (self.dt / capacitance)
    self.v = (self.v + del_v)

  def __update_thr(self, 
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


  def spikeTrain(self, lookBack:float=None, first_n_neurons:int=5, purge:bool=False):
    """Plot spiketrain plot of specified neuron counts and lookBack range.

    Args: 
      lookback: length of time [ms] to backtrack for plotting the spikeTrain; 
        default None results in the entire time duration.
      first_n_neurons: number of neurons to plot spike train; should be <= n_neurons 
        in the LIF_Network. Defaults 5.
      purge (boolean): Clears the spike record in the LIF_Network object

    Returns:
      SR (np.ndarray): n-by-2 ndarray recording the neuron and its spike time. 
    """

    if lookBack is None:
      strt_timestamp = 0
    else: 
      strt_timestamp = self.t_current - lookBack 
    
    # Subsetting spike record since strt_timestamp onwards
    spike_record = np.reshape(self.spike_record,newshape = [-1,2])
    spike_record = np.delete(spike_record, obj=0, axis=0)               # Delete first row - the placeholder
    idx_spike_record = np.argmax(spike_record[:, 1] >= strt_timestamp)  # idx of first spike_record since timestamp == loopBack
    spike_record = spike_record[idx_spike_record:, :]                   # Lookback subset (NDarray View)
    spike_record = spike_record[spike_record[:, 0] <= first_n_neurons]  # first_n_neuron subset
    
    # Plot 
    fig, ax = plt.subplots()
    ax.plot([strt_timestamp, self.t_current], [0, first_n_neurons], "white")
    for entry in spike_record:
      x_min, x_max = entry[1], entry[1]
      y_min, y_max = entry[0], entry[0] + 0.9
      ax.plot((x_min, x_max), (y_min, y_max), 'k', linewidth=0.5)
    
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron #")
    ax.set_title("Spike Train")
    fig.set_size_inches(5, 4)
    plt.show()

    # Purge spike record by releasing it for garbage collection
    if purge:
      self.spike_record = np.empty(shape=[1,2])
    
    return spike_record

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

    NOTE (Tony): 
    - g_syn dynamic function using a different kappa value (50x larger):
      - Ali's paper has a kappa_syn value of 8, but his code is using the 
        value 400 and denotes this value as capa.
      - The outcome is that his kappa/N value would be much larger (50x) larger than
        that if we were to follow the value set forth in his paper. The dynamic membrane
        conductivity will thus update much faster when the value is scaled to be 50x 
        larger.
    - g_noise dynamic function missing kappa_noise = 0.026:
      - Ali's paper states kappa_noise = 0.026, however, his code is missing this
        variable in the g_noise calculation function.
      - The outcome is that g_noisie update much faster in Ali's code than when
        using the value provided in Ali's paper.
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
      self.__update_g_noise(kappa_noise=kappa_noise, method=temp_param["update_g_noise_method"])  
      # Dynamic Function Update synaptic conductivity
      self.__update_g_syn(step=step, 
                        kappa=kappa, 
                        external_spiked_input_w_sums=external_spiked_input_w_sums, 
                        method=temp_param["update_g_syn_method"])
      # Reset variables
      self.spiked_input_w_sums = np.zeros(self.n_neurons)     # Weight sum of all spiked-connected presynaptic neurons
      self.w_update_flag = np.zeros(self.n_neurons)           # Connection weight update tracker
      self.dW = 0                                                  # Net connection weight change per epoch
      # Dynamic Function Update membrane potential
      timer.time_perf(self.__update_v)(step=step, 
                    euler_steps=euler_steps,
                    I_stim = I_stim, 
                    method=temp_param["update_v_method"], 
                    capacitance_method=temp_param["update_v_capacitance_method"])
      # Dynamic Function Update spike threshold
      timer.time_perf(self.__update_thr)(method=temp_param["update_thr_method"])
      
      # Check if the neurons spike and mark them as needing to update conn weight
      timer.time_perf(self.__check_if_spiked)()

      # Depolarization and Hyperpolarization (rectangular spike shape)
      timer.time_perf(self.__rectangular_spiking)()
      
      # Update the variable needed for next step's g_syn calculation
      timer.time_perf(self.__calc_spiked_input_w_sums)()

      # Updates the network_W and dW
      timer.time_perf(self.__run_stdp_on_all_connected_pairs)()
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