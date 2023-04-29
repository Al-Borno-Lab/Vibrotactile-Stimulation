# Crucial dependencies
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy import sparse

## NOTES (Tony) - 2023-04-27:
# - [x] Convert `network_conn`` from bool to int
# - [x] Look into using mask for `network_weight` to save time on calculation
# - [x] Network weight initialization also filters out non-connected
#   - Assuming independent connection probability, we can copy the conn sparse array
#     and create connection weight via the sparse array data.
# - [x] Sparsify - outline where to sparsify the code
# - [x] Skip over the network_conn implementation in structured_conn
# - [x] Mask out the non-connected pairs' weight in `network_weight` --> Not a good idea
# - [x] Can you select elements in a sparse matrix via index?
#   because numpy maskedarray is not compiled and has a lot of computation overhead. It works
#   best in cases when the sparsity is low, thus take note to implement this using 
#   sparse array instead.
# - Consider how to parallelize `run_stdp_on_all_pairs` with CuPy or Numba - May not matter much if sparsity is high.
# - `network_weight` non-connected pairs marked as zero in sparse matrix (this 
#   makes sense because if neurons are not connected, they shouldn't spontaneously 
#   start connecting with each other.)
# - `network_weight` normalizing to mean can be done with sparse as well
# - `calc_nn_mean_w` needs to be fixed as well
# - How does masking work and does it safe calculation time? 
# - `calc_spiked_input_w_sum` needs to be fixed as well, perhaps use masking for `network_weight`
# - `__run_stdp_on_all_connected_pairs` needs to be fixed, the if-statement checks (2x fixes).
# - `poisson_noise_spike_flag` can be turned to sparse as well (check, may not be worth it if 1D)
# - `generate_poisson_spike()` has a matmul between spike_flag and network_conn
# - `simulate()` also has a matmul with network_conn that can turn to sparse
# - `simulate()` has the two if-statement as that of the `run_stdp...`
# - May need a get method for extracting network_conn as dense matrix
# - Other variables may also need get method for extracting if they are going to be sparse.

# Calculation dependencies
from scipy import stats    # We just want the stats, because Ca2+ imaging is always calculated in z-score.
from scipy.stats import circmean

# Debug dependencies
from src.utilities import timer
from tqdm import tqdm

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

    # Spatial organization
    self.x = np.random.uniform(*dimensions[0], size=self.n_neurons)
    self.y = np.random.uniform(*dimensions[1], size=self.n_neurons)
    self.z = np.random.uniform(*dimensions[2], size=self.n_neurons)

    # Internal time trackers
    self.t_current = 0            # [ms] Current time 
    self.dt = 0.1                 # [ms] Timestep length
    self.euler_step_idx = 0       # [idx] Euler step index
    self.relax_time = 10000       # [ms] Length of relaxation phase 

    # Electrophysiology values
    self.v_rest = -38                                   # [mV] Resting voltage, equilibrium voltage. Default: -38 mV
    self.v_thr = np.zeros(self.n_neurons) - 40          # [mV] Reversal potential for spiking; equation (3)
    self.v_thr_rest = np.zeros(self.n_neurons) - 40     # [mV] Reversal potential for spiking during refractory period. Defaults to -40mV; equation (3).
    self.v_syn = 0                                      # [mV] Reversal potential; equation (2) in paper
    self.tau_syn = 1                                    # [ms] Synaptic time-constant; equation (4) in paper
    self.tau_rf_thr = 5                                 # [ms] Timescale tau between refractory and normal thresholds relaxation period
    self.g_syn_initial_value = 0                        # [mS/cm^2] Initial value of synaptic conductivity; equation (2) in paper
    self.synaptic_delay = 3                             # [ms] Synaptic transmission delay from soma to soma (default: 3 ms), equation (4) in paper
    self.v_reset = -67                                  # [mV] Membrane potential right after spiking ((Hyperpolarization)); equation (3) in paper
    self.v_thr_spike = 0                                # [mV] Threshold during refractory period; V_th_spike of equation (3) in paper
    
    # (Paper definition of) Specific Membrane Capacitance
    capa_rv_norm_mean = 3
    capa_rv_norm_stdev = 0.05 * capa_rv_norm_mean                 # Defined in paper
    self.capacitance = np.random.normal(loc=capa_rv_norm_mean,    # [microF/cm^2] Equation (2) in paper
                                        scale=capa_rv_norm_stdev, 
                                        size=self.n_neurons)

    # Input noise of Poisson distribution (input is generated with I = G*V)
    self.poisson_noise_spike_flag = np.zeros(self.n_neurons)

    # Internal Trackers
    self.flag_spike          = np.zeros((self.n_neurons, ), dtype=int)                                 # Tracker of whether a neuron spiked
    self.t_prevSpike         = np.zeros((self.n_neurons, ), dtype=float) - 10000                       # Tracker of spike timestamp; Default to -10000 for mathematical computation convenience; keep track of the previous timestamps of spiked neurons.
    self.t_currentSpike      = np.zeros((self.n_neurons, ), dtype=float) - 10000                       # Tracker of spike timestamp; Default to -10000 for mathematical computation convenience; keep track of the current timestamps of spiked neurons.
    self.spikeRecord         = np.zeros((1, 2))                                                        # Tracker of Spike record recorded as a list: [neuron_number, spike_time]
    self.g_syn               = np.zeros((self.n_neurons, ), dtype=float) + self.g_syn_initial_value    # Tracker of dynamic synaptic conductivity. Initial value of 0.; equation (2)
    self.g_noise             = np.zeros((self.n_neurons, ), dtype=float)                               # Tracker of dynamic noise conductivity
    self.flag_wUpdate        = np.zeros((self.n_neurons, ), dtype=int)                                 # Tracker of connetion weight updates. When a neuron spikes, it is flagged as needing update on its connection weight.
    self.spiked_input_w_sums = np.zeros((self.n_neurons, ), dtype=float)                               # Tracker of the connected presynaptic weight sum for each neuron (Eq 4: weight * Dirac Delta Distribution)
    self.network_conn        = None                                                                    # Tracker of Neuron connection matrix: from row-th neuron to column-th neuron
    self.network_weight      = None                                                                    # Tracker of Neuron connection weight matrix: from row-th neuron to column-th neuron

    # STDP paramters
    self.stdp_beta = 1.4                                       # Balance factor (ratio of LTD to LTP); equation (7)
    self.stdp_tau_r = 4                                        # When desynchronized and synchronized states coexist; equation (7)
    self.stdp_tau_plus = 10                                    # For the postive half of STDP; equation (7)
    self.stdp_tau_neg = self.stdp_tau_r * self.stdp_tau_plus   # For the negative half of STDP; equation (7)
    self.eta = 0.02                                            # Scales the weight update per spike; 0.02 for "slow STDP"; equation (7)

    ### Below are Variables that Differs between Ali's Code vs Ali's Paper
    # >>> Ali's Paper >>>
    self.kappa = 8
    self.kappa_noise = 0.026
    self.g_leak = 0.02    # [mS/cm^2] Conductivity of the leak channels; equation (2) in paper
    self.v = np.random.uniform(low=-38, high=-40, size=(self.n_neurons,))

    # >>> Ali's Code >>>
    self.kappa_ali_code = 400
    self.kappa_noise_ali_code = 1  # Not specified, which equals to setting the variable as 1
    self.v_spike = 20              # Not in paper, but shows up in Ali's codebase           # [mV] The voltage that spikes rise to ((AP Peak))
    self.g_leak_ali_code = 10      # The value Ali used in his code, 500x the value stated in the paper (0.02)
    self.g_poisson = 1.3                                                                    # [mS/cm^2] conductivity of the extrinsic poisson inputs
    self.v_ali = np.random.uniform(low=-10, high=10, size=(self.n_neurons,)) - 45           # [mV] Current membrane potential - Initialized with Unif(-55, -35)
    self.tau_spike = 1                                                                      # [ms] Time for an AP spike, also the length of the absolute refractory period
    
    # ??? Speculate to be capacitance random variable drawn from a normal distribution ???
    # NOTE (Tony): Odd that this has an expected value of 150. Most reserach
    #   seems to agree that the universal specific membrane capacitance is 
    #   approx 1 microFarad/cm^2.
    tau_c1 = np.sqrt(-2 * np.log(np.random.random(size=(self.n_neurons,))))    # ??? membrane time-constant component 1 ??? - Jesse and Tony are unsure what this is 
    tau_c2 = np.cos(2 * np.pi * np.random.random(size=(self.n_neurons,)))      # ??? membrane time-constant component 2 ??? - Jesse and Tony are unsure what this is  
    self.tau_m = 7.5 * tau_c1 * tau_c2 + 150                                   # ??? Unsure what this is, but through usage seems like the capacitance random-variable in equation (2) (However, values are off...)
    
    # Generate neuron connection matrix
    if auto_random_connect:
      self.random_conn()  # Create random neuron connections

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
    # # self.network_weight = np.random.random(size=(self.n_neurons,self.n_neurons))
    # # self.network_weight[self.network_conn == 0] = 0
    # # self.network_weight = self.network_weight/np.mean(self.network_weight[self.network_weight > 0]) * mean_w
    # # self.network_weight[self.network_weight > 1] = 1
    # # self.network_weight[self.network_weight < 0] = 0
    # return dist
    ...

  def random_conn(self, mean_w:float=0.5, proba_conn:float=0.07) -> None:
    """Randomly create connections between neurons basing on the proba_conn.

    Args:
        mean_w (float, optional): The mean connection weight to normalize each  
          connection in the network to. Defaults 0.5
        proba_conn (float, optional): Probability of connection between neurons.
          Defaults 0.07.
    """
    # Ensure mean_w is in (0, 1) as any other value does not make sense.
    # assert (np.greater(mean_w, 0) & np.less(mean_w, 1)), \
    #   f"mean_w has be within (0, 1) (exclusive). Current mean_w: {mean_w}"

    # Initialize network connectivity matrix
    self.network_conn = np.random.choice(a=[1, 0], 
                                         p=[proba_conn, (1-proba_conn)], 
                                         size=(self.n_neurons, self.n_neurons))
    self.network_conn = sparse.csr_array(self.network_conn)

    # Initialize weight matrix
    self.network_weight = self.network_conn.copy()  # Deep copy
    self.network_weight.data = np.random.random(size=self.network_weight.nnz)
    
    # Normalized to mean conductivity (i.e., `mean_w`)
    current_nn_mean = self.network_weight.mean(axis=None, dtype=float)
    normalization_scale = mean_w / current_nn_mean
    self.network_weight *= normalization_scale

    # COMMENTED OUT: >>> Hard bounding is enforced during weight update
    # Hard bound weight to [0, 1]
    # self.network_weight = np.clip(self.network_weight, a_min=0, a_max=1)
    
  def __stdp_w_update(self, time_diff:float, pre_idx:int, post_idx:int) -> float:
    """Calculate the weight change and update network weight in-place"""
    dw = 0 
    
    # Case: LTP (Long-term potentiation)
    if np.less_equal(time_diff, 0):
      dw = (self.eta * np.exp( time_diff / self.stdp_tau_plus))
      if dw < 0:
        raise AssertionError(f"During LTP, dw should be >= 0, it is {dw}")
    
    # Case: LTD (Long-term depression)
    elif np.greater(time_diff, 0):
      dw = (self.eta * -(self.stdp_beta / self.stdp_tau_r) * np.exp( -time_diff / self.stdp_tau_neg ))
      if dw > 0:
        raise AssertionError(f"During LTD, dw should be <= 0, it is {dw}")

    # Update connection weight in-place
    self.network_weight[(pre_idx, post_idx)] += dw
    
    # Hard bound to [0, 1]  
    self.network_weight[(pre_idx, post_idx)] = np.clip(a=self.network_weight[(pre_idx, post_idx)], 
                                                       a_min=0, a_max=1)

    # # Debug check
    # if self.network_weight[(pre_idx, post_idx)] > 1:
    #   raise AssertionError(f"Connection weight at ({pre_idx}, {post_idx}) is {self.network_weight[(pre_idx, post_idx)]} and it should NOT be greater than 1.")

    return dw
  
  def plot_stdp_scheme_assay(self):
    """Plot the STDP scheme assay

    Y-axis being the connection weight update (delta w).
    X-axis being the time diff of presynaptic spike timestamp less postsynaptic
    spike timestamp. The definition of time_diff is opposite of time_lag (termed
    in the original paper).
    """
    fig, ax = plt.subplots()
    x = np.arange(-100, 100, 1)
    
    for i in x:
      ax.scatter(x=i, y=self.__stdp_w_update(i), c="black", s=3)

    ax.set_title("STDP Scheme Assay")
    ax.set_xlabel("Time offset (PreSyn - PostSyn)")
    ax.set_ylabel("dw / dt")

    return fig

  def calc_nn_mean_w(self) -> float:
    """Calculate and return neural network mean connection weight at time of call."""
    
    mean_network_w = self.network_weight.mean(axis=None, dtype=float)
    return mean_network_w

  def __generate_poisson_noise(self, poisson_noise_lambda_hz:int=20) -> None:
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
      raise AssertionError("time-step is too large causing the Poisson noise's binomial estimation to be inaccurate.")

    # Generate Poisson noise spike flags
    self.poisson_noise_spike_flag = np.random.binomial(n=1, 
                                                       p=p_approx,
                                                       size=(self.n_neurons,))

  def __check_if_spikes(self) -> None:
    """Check if neurons spike, and mark them as needing to update connection weight.
    """
    # Check if spiked from passing threshold
    spike = ((self.v >= self.v_thr)     # Met dynamic spiking threshold
             * (self.flag_spike == 0))  # Not in abs_refractory period because not recently spiked

    # Update tracking flags
    self.flag_spike[spike]   = 1  # Mark them as SPIKED!
    self.flag_wUpdate[spike] = 1  # Mark them as "Needing to update weight"

    # Keep track of spike times
    self.t_prevSpike[spike] = self.t_currentSpike[spike]  # Moves the t_currentSpike array into t_prevSpike for placeholding
    self.t_currentSpike[spike] = self.t_current           # t_currentSpike keeps track of each neuron's most recent spike's timestamp

  def __rectangular_spiking(self) -> None:
    """Simulate a rectangular spike of v_spike mV for tau_spike ms.
    """
    # Mask for those needing to spike
    spiking = (self.flag_spike == 1)
    
    # Depolarization phase
    self.v[spiking] = self.v_spike         # Rectangle spike shape by setting voltage to V_spike for duration of tau_spike (equation 3)

    # Hyperpolarization phase
    in_abs_rf_period = ((self.t_currentSpike + self.tau_spike) > self.t_current)
    mask_update = (~in_abs_rf_period) * spiking
    self.v[mask_update] = self.v_reset  # Rectangular spike end resets potential to -67mV
    
    # Reset spike flag tracker (have to have passed absolute ref period)
    self.flag_spike[mask_update] = 0
    self.v_thr[mask_update] = self.v_thr_spike  # Threshold is reset to V_th_spike=0mV right after spiking (equation 3)

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
    t_diff = self.t_current - (self.t_currentSpike + self.synaptic_delay)  # [n, ] array
    # s_flag = 1.0 * (abs(t_diff) < 0.01)  # (Tony) Testing the following implementation
    s_flag = np.isclose(t_diff, 0, atol=1e-2).astype(int)  # 0.01 for floating point errors

    self.spiked_input_w_sums = self.network_weight.T.dot(s_flag).T

  def __run_stdp_on_all_connected_pairs(self)-> float:
    """Checks all connected pairs and update weights based on STDP scheme.

    This double-nested for-loop checks through all pairs, and the four cases 
    (A, B, C, D) allows for checking of a spiked neuron's pre- and post-synaptic
    partners. 

    Case B is specifically for when synaptic delay = 0, which usually would not
    happen in our current model as all neurons in this current model are 
    homogenous.

    Returns:
      delta_w_sum (float): The sum of all the delta_w.
    """
    dw = 0

    if self.flag_wUpdate.any():
      for i in range(self.n_neurons):
        if (self.flag_wUpdate[i] == 1):  ### SPOTLIGHT ###
          self.spikeRecord = np.append(self.spikeRecord,
                                        np.array([i, self.t_current]))
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
            if self.network_conn[i, j] == 1:

              # Check if j has a spike in transit, and if so, use the spike before last:
              # Smallest value is syn_delay; range: [syn_delay, t+syn_delay]
              temporal_diff = self.t_currentSpike[i] - self.t_currentSpike[j] + self.synaptic_delay
              # <<<<<<<<<< IGNORE - DEBUG USE
              # print(f"{' '*10} {self.t_currentSpike[i]} - {self.t_currentSpike[j]} + {self.synaptic_delay}")
              # >>>>>>>>>> IGNORE - DEBUG USE
              
              # Case A
              if temporal_diff > 0:  # ??? temporal_diff >= 0 ??? - Why not triage like Case C and D? 
                # - i has spike in transit (both spiked at the same time or j spiked no more than delay-time ago)
                # - LTD always, regardless of when j spiked
                dw += self.__stdp_w_update(time_diff=temporal_diff, pre_idx=i, post_idx=j)
                # <<<<<<<<<< IGNORE - DEBUG USE
                # print(f"A: STDP on {i} -> {j} at eulerstep {t} of time {self.t} ms")
                # print(f"{' '*10} {self.t_currentSpike[i]} - {self.t_currentSpike[j]} + {self.synaptic_delay}")
                # print(f"{' '*10} t_currentSpike[{i}]-t_currentSpike[{j}]: temporal_diff: {temporal_diff} ms")
                # >>>>>>>>>> IGNORE - DEBUG USE
              
              # Case B
              else:
                # - This can only happen is synaptic delay = 0 
                # - And this is always LTD
                temporal_diff = self.t_currentSpike[i] - self.t_prevSpike[j] + self.synaptic_delay
                dw += self.__stdp_w_update(time_diff=temporal_diff, pre_idx=i, post_idx=j)
                # <<<<<<<<<< IGNORE - DEBUG USE
                # print(f"B: STDP on {i} -> {j} at eulerstep {t} of time {self.t} ms")
                # print(f"{' '*10} {self.t_currentSpike[i]} - {self.t_prevSpike[j]} + {self.synaptic_delay}")
                # print(f"{' '*10} t_currentSpike[{i}]-t_prevSpike[{j}]: temporal_diff: {temporal_diff} ms")
                # >>>>>>>>>> IGNORE - DEBUG USE

            ### Spotlight is on i ### SPIKED neuron receiving connection
            # if j is pre-synaptic to i, update W(j,i)
            if self.network_conn[j, i] == 1: 
              
              # check if j has a spike in transit, and if so, use the spike before last:
              # Largest value is syn_delay; range: [syn_delay-t-10000, syn_delay]
              temporal_diff =  self.t_currentSpike[j] - self.t_currentSpike[i] + self.synaptic_delay
                # <<<<<<<<<< IGNORE - DEBUG USE
              # print(f"{' '*10} {self.t_currentSpike[j]} - {self.t_currentSpike[i]} + {self.synaptic_delay}")
                # >>>>>>>>>> IGNORE - DEBUG USE
              
              # Case C
              if temporal_diff < 0: 
                # - j's spike arrived at i before i spiked, thus LTP
                dw += self.__stdp_w_update(time_diff=temporal_diff, pre_idx=j, post_idx=i)
                # <<<<<<<<<< IGNORE - DEBUG USE
                # print(f"C: STDP on {j} -> {i} at eulerstep {t} of time {self.t} ms")
                # print(f"{' '*10} t_currentSpike[{j}]-t_currentSpike[{i}]: temporal_diff: {temporal_diff} ms")
                # print(f"{' '*10} {self.t_currentSpike[j]} - {self.t_currentSpike[i]} + {self.synaptic_delay}")
                # >>>>>>>>>> IGNORE - DEBUG USE
              
              # Case D
              else: 
                # - j has spike in transit (both spiked at the same time or j spiked no more than delay-time ago)
                # - Can be LTP or LTD, really depends when the last time j spiked.
                #   - LTD if j's previous spike is less than delay-time ago from i's current spike.
                #   - LTP if j's previous spike is more than delay-time ago from i's current spike.
                temporal_diff = self.t_prevSpike[j] - self.t_currentSpike[i] + self.synaptic_delay
                dw += self.__stdp_w_update(time_diff=temporal_diff, pre_idx=j, post_idx=i)
                # <<<<<<<<<< IGNORE - DEBUG USE
                # print(f"D: STDP on {j} -> {i} at eulerstep {t} of time {self.t} ms")
                # print(f"{' '*10} {self.t_prevSpike[j]} - {self.t_currentSpike[i]} + {self.synaptic_delay}")
                # print(f"{' '*10} t_prevSpike[{j}]-t_currentSpike[{i}]: temporal_diff: {temporal_diff} ms")
                # >>>>>>>>>> IGNORE - DEBUG USE
    return dw

  def __update_g_noise(self) -> None:
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

    ISSUE (Tony): In Ali's code, the Poisson input calculation assumes the nn
    to be strongly connected, instead, the Poisson noise should be originating
    from each neuron and it is unclear where he got g_poisson=1.3.
    """
    # Generate Poisson noise
    self.__generate_poisson_noise()

    # Update conductivity (denoted g)
    self.g_noise = (self.g_noise * np.exp(-self.dt/self.tau_syn) 
                    + self.g_poisson * self.poisson_noise_spike_flag)  # Poisson conductivity * poisson_input_flag makes sense because poisson_input_flag is binary outcome.
  
  def __update_g_noise_tony(self, kappa_noise:float=0.026) -> None:
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
        kappa_noise (float, optional): 
          [mS/cm^2] Noise conductivity. Defaults to 0.026.
    """
    # Generate Poisson noise
    self.__generate_poisson_noise()
    poisson_noise_spiked_input_count = np.matmul(self.poisson_noise_spike_flag, 
                                                 self.network_conn)
    # Update conductivity (denoted g) - 
    # Integrate inputs from noise and synapses (Equation 6 from paper)
    del_g_noise = (np.exp(-self.dt/self.tau_syn)
                   * (- self.g_noise 
                      + kappa_noise 
                        * self.tau_syn * poisson_noise_spiked_input_count))
    new_g_noise = (self.g_noise + del_g_noise)
    self.g_noise = new_g_noise

  def __update_g_syn(self, kappa:float=400, 
                     external_stim_coup_strength:float=None, 
                     step:int=None, external_spiked_input_w_sums:npt.NDArray=None) -> None: 
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

    tau_syn is missing from the equation, and would be okay if tau_syn is always
    1ms. If tau_syn were to ever change, the equation here would no longer be 
    accurate.

    NOTE (Tony): This method is calculating the dynamic conductivity for the
     current euler-step, whereas the `_update_g_syn_tony` calculates the next
     step's conductivity. The difference is minor in the grand scheme of things
     when we allow some time for the neural network to stablize.

    TODO (Tony): Check with Jesse whether the missing of tau_syn here in
      __update_g_syn is a concern.

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
    
    per_neuron_coup_strength = kappa/self.n_neurons

    # Update synaptic conductivity
    self.g_syn = (self.g_syn * np.exp(-self.dt/self.tau_syn)
                  + per_neuron_coup_strength * self.spiked_input_w_sums
                  + external_stim_coup_strength * external_spiked_input_w_sums_step)
       
  def __update_g_syn_tony(self, kappa:float=8, 
                          external_stim_coup_strength:float=None, 
                          step:int=None, external_spiked_input_w_sums:npt.NDArray=None) -> None: 
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
    new_g_syn = (self.g_syn + del_g_syn)
    self.g_syn = new_g_syn
    
  def __update_v(self, capacitance:npt.NDArray=None, g_leak:float=10) -> None:
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
    del_v = (self.dt / capacitance) * (g_leak * (self.v_rest - self.v)
                                       + self.g_syn * (self.v_syn - self.v)
                                       + external_current_stim_step
                                       + I_noise)
    new_v = (self.v + del_v)
    self.v = new_v

  def __update_thr(self) -> None:
    """Updates neuron spiking threshold with method from Ali's code.

    This method does not exponentially decay and can cause issue if the user 
    were to increase the timestep of the neural network.

    For a exponentially decaying version, use `__update_thr_tony()` instead.
    """
    # Determine method of dyanmic threshold update implementation
    self.v_thr = (self.v_thr + self.dt/self.tau_rf_thr * (self.v_thr_rest - self.v_thr))

  def __update_thr_tony(self) -> None:
    """Updates the neuron spiking threshold with method from Ali's paper.

    This method further enhanced the method laid out in Ali's paper by including
    exponential decay to provide more accurate calculations if the timestep of 
    the neural network were to increase.
    """
    # Determine method of dyanmic threshold update implementation
    del_v_thr = self.dt/self.tau_rf_thr * (self.v_thr_rest - self.v_thr)
    self.v_thr = (self.v_thr + del_v_thr)

  def plot_spikeTrain(self, lookBack:float=None, first_n_neurons:int=5, purge:bool=False):
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
    spikeRecord = np.reshape(self.spikeRecord,newshape = [-1,2])
    spikeRecord = np.delete(spikeRecord, obj=0, axis=0)               # Delete first row - the placeholder
    idx_spikeRecord = np.argmax(spikeRecord[:, 1] >= strt_timestamp)  # idx of first spikeRecord since timestamp == loopBack
    spikeRecord = spikeRecord[idx_spikeRecord:, :]                   # Lookback subset (NDarray View)
    spikeRecord = spikeRecord[spikeRecord[:, 0] <= first_n_neurons]  # first_n_neuron subset
    
    # Plot 
    fig, ax = plt.subplots()
    ax.plot([strt_timestamp, self.t_current], [0, first_n_neurons], "white")
    for entry in spikeRecord:
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
      self.spikeRecord = np.empty(shape=[1,2])
    
    return spikeRecord

  def kuramato_vect(self, period: float = None, lookback: float = None, r_cutoff = 0.3):
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
    SR = np.reshape(self.spikeRecord,newshape = [-1,2])  
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

  def kuramato(self, period:float=None, lookback:float=None):
    """Return the phase mean of all neurons in the spike record.

    circmean() instead of trigonometry is used to find the mean phase of all 
    the spikes of a neuron, thus, there is higher precision, hence, eliminating
    the need of a r_cutoff argument to trim off the phase noise created by
    rounding floating points.

    Essentially, this is a more accurate version of the `kuramato_vect` method.

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
    SR = np.reshape(self.spikeRecord,newshape = [-1,2])
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


  def simulate_not_readlly_working(self, 
               sim_duration: float = 1, 
               external_current_stim: npt.NDArray = None,  # Used for __update_v_tony (feature expansion)
               external_spiked_input_w_sums: npt.NDArray = None,):
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
    # Set Variables
    euler_steps = int(sim_duration/self.dt)   # Number of Euler-method steps
    euler_step_idx_start = self.t_current / self.dt  # Euler-step starting index

    # Output variable placeholders
    holder_epoch_timestamps    = np.zeros((euler_steps, ))
    holder_v                   = np.zeros((euler_steps, self.n_neurons))
    holder_g_syn               = np.zeros((euler_steps, self.n_neurons))
    holder_poi_noise_flags     = np.zeros((euler_steps, self.n_neurons))
    holder_spiked_input_w_sums = np.zeros((euler_steps, self.n_neurons))
    holder_dw                  = np.zeros((euler_steps, ))

    # Euler-step Loop
    for step in range(euler_steps):
      # Update Poisson noise's conductivity - Ali's code method and variables
      self.__update_g_noise()
      # Update synaptic conductivity - Ali's code method and varaibles
      self.__update_g_syn(kappa=self.kappa_ali_code,
                          step = step, 
                          external_spiked_input_w_sums=external_spiked_input_w_sums)
      
      # Reset inputs
      self.spiked_input_w_sums = np.zeros(self.n_neurons)
      self.flag_wUpdate = np.zeros(self.n_neurons)
      dw = 0

      # Update membrane potential
      self.__update_v(capacitance=self.tau_m, g_leak=self.g_leak_ali_code)
      # Update spike threshold
      self.__update_thr()
      
      # Check if the neurons spike and mark them as needing to update conn weight
      self.__check_if_spikes()
      # Depolarization and Hyperpolarization (rectangular spike shape)
      self.__rectangular_spiking()
      # Update the variable needed for next step's g_syn calculation
      self.__calc_spiked_input_w_sums()
      # Updates the network_W and dW
      dw_sum = self.__run_stdp_on_all_connected_pairs()
    

      # End of Epoch:
      # NOTE: Used so that multiple simulation runs have continuity.
      tix =                             int(self.euler_step_idx - euler_step_idx_start)
      holder_epoch_timestamps[tix] =    self.t_current
      holder_v[tix] =                   self.v   
      holder_g_syn[tix] =               self.g_syn + self.g_noise
      holder_poi_noise_flags[tix] =     self.poisson_noise_spike_flag
      holder_spiked_input_w_sums[tix] = self.spiked_input_w_sums
      holder_dw[tix] =                  dw_sum
      
  
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
  
  def simulate(self, 
               sim_duration: float = 1, 
               external_current_stim: npt.NDArray = None,  # Used for __update_v_tony (feature expansion)
               external_spiked_input_w_sums: npt.NDArray = None,):
    
    euler_steps = int(sim_duration / self.dt)

    if external_spiked_input_w_sums is None: 
      external_spiked_input_w_sums = np.zeros(shape=(euler_steps, self.n_neurons))

    # Varaible exporters:
    t_holder =    np.zeros([euler_steps,])
    v_holder =    np.zeros([euler_steps,self.n_neurons])
    gsyn_holder = np.zeros([euler_steps,self.n_neurons])
    pois_holder = np.zeros([euler_steps,self.n_neurons])
    in_holder =   np.zeros([euler_steps,self.n_neurons])
    dW_holder =   np.zeros([euler_steps,])

    euler_step_idx_start = self.t_current/self.dt

    # Time loop:
    ii = 0
    for t in range(euler_steps):

      # Calculate Poisson noise
      self.__generate_poisson_noise()

      # Noise conductivity update
      # self.noise_g = (1-self.dt) * self.noise_g + self.g_poisson * self.poisson_input  # Non exp decay
      self.g_noise = self.g_noise * np.exp(-self.dt/self.tau_syn) + self.g_poisson * self.poisson_noise_spike_flag
      
      # Synaptic conductivity update
      # self.g_syn = (1-self.dt) * self.g_syn + self.network_coupling * self.network_input  # Non exp decay
      network_coupling = self.kappa_ali_code / self.n_neurons
      external_strength = self.kappa_ali_code / 5
      self.g_syn = (self.g_syn * np.exp(-self.dt/self.tau_syn) 
                    + network_coupling * self.spiked_input_w_sums 
                    + external_strength * external_spiked_input_w_sums[ii][:])

      # Variable reset
      self.network_input = np.zeros([self.n_neurons,])
      self.flag_wUpdate = np.zeros([self.n_neurons,])
      dW = 0

      # Membrane potential and action-potential threshold update
      # NOTE (Tony): Ali used a g_leak value that is 500x stated in the paper 0.02 * 500 = 10
      # Case 1: g_leak (paper's value) = 0.02
      # self.v = (self.v 
      #           + (self.dt/self.tau_m) * (self.g_leak * (self.v_rest - self.v) 
      #                                     - self.v * (self.g_noise + self.g_syn)))
      # Case 2: g_leak_ali_code (Ali's code value) = 10
      self.v = (self.v 
                + (self.dt/self.tau_m) * (self.g_leak_ali_code * (self.v_rest - self.v) 
                                          - self.v * (self.g_noise + self.g_syn)))
      self.v_thr = self.v_thr + self.dt * (self.v_thr_rest - self.v_thr) / self.tau_m

      # Evaluate if reached AP voltage threshold
      sp = np.logical_and(np.greater_equal(self.v, self.v_thr), 
                          np.equal(self.flag_spike, 0))
      self.flag_spike[sp] = 1
      self.flag_wUpdate[sp] = 1

      # Time tracker switcharoo
      self.t_prevSpike[sp] = self.t_currentSpike[sp]
      self.t_currentSpike[sp] = self.t_current

      reached_spike_thresh = np.equal(self.flag_spike, 1)
      self.v[reached_spike_thresh] = self.v_spike

      t_offset = np.less_equal(self.t_currentSpike, self.t_current-self.tau_spike)
      not_in_abs_ref_and_passed_ap_threshold = np.logical_and(t_offset, 
                                                              reached_spike_thresh)
      self.flag_spike[not_in_abs_ref_and_passed_ap_threshold] = 0
      self.v[not_in_abs_ref_and_passed_ap_threshold]          = self.v_reset
      self.v_thr[not_in_abs_ref_and_passed_ap_threshold]      = self.v_thr_spike

      s_difference = np.subtract(self.t_current - self.synaptic_delay, self.t_currentSpike)
      s_flag = np.isclose(s_difference, 0, atol=1e-2).astype(int)
      self.network_input = self.network_weight.T.dot(s_flag).T

      # STDP:
      if self.flag_wUpdate.any():
        for i in range(self.n_neurons):
          if (self.flag_wUpdate[i] == 1):  ### SPOTLIGHT ###
            self.spikeRecord = np.append(self.spikeRecord,np.array([i, self.t_current]))
            for j in range(self.n_neurons):
              
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
              if self.network_conn[i, j] == 1:

                # check if j has a spike in transit, and if so, use the spike before last:
                # Smallest value is syn_delay; range: [syn_delay, t+syn_delay]
                temporal_diff = self.t_currentSpike[i] - self.t_currentSpike[j]   + self.synaptic_delay
                # print(f"{' '*10} {self.t_currentSpike[i]} - {self.t_currentSpike[j]} + {self.synaptic_delay}")
                
                # Case A
                if temporal_diff > 0:  # ??? temporal_diff >= 0 ??? - Why not triage like Case C and D? 
                  # - i has spike in transit (both spiked at the same time or j spiked no more than delay-time ago)
                  # - LTD always, regardless of when j spiked
                #   print(f"A: STDP on {i} -> {j} at eulerstep {t} of time {self.t} ms")
                #   print(f"{' '*10} {self.t_currentSpike[i]} - {self.t_currentSpike[j]} + {self.synaptic_delay}")
                #   print(f"{' '*10} t_currentSpike[{i}]-t_currentSpike[{j}]: temporal_diff: {temporal_diff} ms")
                  dW = dW + self.__stdp_w_update(temporal_diff,i,j)
                # Case B
                else:
                  # - This can only happen is synaptic delay = 0 
                  # - And this is always LTD
                  temporal_diff = self.t_currentSpike[i] - self.t_prevSpike[j] + self.synaptic_delay
                #   print(f"B: STDP on {i} -> {j} at eulerstep {t} of time {self.t} ms")
                #   print(f"{' '*10} {self.t_currentSpike[i]} - {self.t_prevSpike[j]} + {self.synaptic_delay}")
                #   print(f"{' '*10} t_currentSpike[{i}]-t_prevSpike[{j}]: temporal_diff: {temporal_diff} ms")
                  dW = dW + self.__stdp_w_update(temporal_diff,i,j)

              ### Spotlight is on i ### SPIKED neuron receiving connection
              # if j is pre-synaptic to i, update W(j,i)
              if self.network_conn[j, i] == 1: 
                
                # check if j has a spike in transit, and if so, use the spike before last:
                # Largest value is syn_delay; range: [syn_delay-t-10000, syn_delay]
                temporal_diff =  self.t_currentSpike[j] - self.t_currentSpike[i] + self.synaptic_delay
                # print(f"{' '*10} {self.t_currentSpike[j]} - {self.t_currentSpike[i]} + {self.synaptic_delay}")
                
                # Case C
                if temporal_diff < 0: 
                  # - j's spike arrived at i before i spiked, thus LTP
                #   print(f"C: STDP on {j} -> {i} at eulerstep {t} of time {self.t} ms")
                #   print(f"{' '*10} t_currentSpike[{j}]-t_currentSpike[{i}]: temporal_diff: {temporal_diff} ms")
                #   print(f"{' '*10} {self.t_currentSpike[j]} - {self.t_currentSpike[i]} + {self.synaptic_delay}")
                  dW = dW + self.__stdp_w_update(temporal_diff,j,i)
                # Case D
                else: 
                  # - j has spike in transit (both spiked at the same time or j spiked no more than delay-time ago)
                  # - Can be LTP or LTD, really depends when the last time j spiked.
                  #   - LTD if j's previous spike is less than delay-time ago from i's current spike.
                  #   - LTP if j's previous spike is more than delay-time ago from i's current spike.
                  temporal_diff = self.t_prevSpike[j] - self.t_currentSpike[i] + self.synaptic_delay
                #   print(f"D: STDP on {j} -> {i} at eulerstep {t} of time {self.t} ms")
                #   print(f"{' '*10} {self.t_prevSpike[j]} - {self.t_currentSpike[i]} + {self.synaptic_delay}")
                #   print(f"{' '*10} t_prevSpike[{j}]-t_currentSpike[{i}]: temporal_diff: {temporal_diff} ms")
                  dW = dW + self.__stdp_w_update(temporal_diff,j,i)
                              
      # End of Epoch:
      tix = int(self.euler_step_idx-euler_step_idx_start)
      t_holder[tix] = self.t_current
      v_holder[:][tix] = self.v   
      gsyn_holder[:][tix] = self.g_syn + self.g_noise
      pois_holder[:][tix] = self.poisson_noise_spike_flag
      in_holder[:][tix] = self.spiked_input_w_sums
      dW_holder[tix] = dW

      self.euler_step_idx += 1
      self.t_current += self.dt
      ii += 1

      # print(f"self.network_W: {self.network_W}")
      # print(f"self.network_conn: {self.network_conn}")
      # print(np.mean(self.network_W[self.network_conn.astype(bool)]))
      
    return v_holder, gsyn_holder, pois_holder, t_holder, in_holder, dW_holder