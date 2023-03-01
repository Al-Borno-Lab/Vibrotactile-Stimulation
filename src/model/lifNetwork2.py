import matplotlib.pyplot as plt    # MatPlotLib is a plotting package. 
import numpy as np                 # NumPy is a numerical types package.
from scipy import stats            # ScPy is a scientific computing package. We just want the stats, because Ca2+ imaging is always calculated in z-score.
from scipy.stats import circmean
import math
from posixpath import join
import numpy.typing as npt
import time

class LIF_Network:
  def __init__(self, n_neurons = 1000, dimensions = [[0,100],[0,100],[0,100]]):
    """Leaky Intergrate-and-Fire (LIF) Neuron network model.

    Args:
      n_neurons (int): number of neurons; default to 1000 neurons
      dimensions: Spatial dimensions for plotting; this plots the neurons in a 
        100x100x100 3D space by default.

    Returns:

    """
    # NOTE (Tony): ((Discrepancy between code vs paper))
    # - tau_m is an odd value and we speculate it to be the capacitance rv from 
    #   eqaution (2) of the paper, however, even then, its generated values are
    #   off. The paper states the capacitance is drawn from a different dist.
    # - Initial value of membrane potential according to the paper is drawn from
    #   unif(-38, -40), however, Ali's code has unif(-55, -35)
    # - Regarding dynamic voltage threshold, it wasn't clear from the paper if 
    #   the initial value was set to 0mV or -40mV. However, it seems like from
    #   Ali's code, the initial value is -40mV.
    # - tau_spike was mentioned in the paper, but the value wasn't.
    # - Poisson noise distribution was using binomial distribution estimation, 
    #   which would break down as the poisson frequency increases, or when n is 
    #   small. Changed the implementation to actual Poisson distribution for 
    #   robustness.

    # Neuron count
    self.n_neurons = n_neurons
    
    # Spatial organization
    self.dimensions = dimensions
    self.x = np.random.uniform(*self.dimensions[0], size=self.n_neurons)
    self.z = np.random.uniform(*self.dimensions[2], size=self.n_neurons)
    self.y = np.random.uniform(*self.dimensions[1], size=self.n_neurons)

    # Internal time trackers
    self.t = 0                                                                              # [ms] Current time 
    self.dt = 0.1                                                                           # [ms] Timestep length
    self.euler_step_idx = 0                                                                 # [idx] Euler step index
    self.relax_time = 10000                                                                 # [ms] Length of relaxation phase 

    # Electrophysiology values
    self.v = np.random.uniform(low=-10, high=10, size=(self.n_neurons,)) - 45               # [mV] Current membrane potential - Initialized with Unif(-55, -35)
    self.v_rest = -38                                                                       # [mV] Resting voltage, equilibrium voltage. Default: -38 mV
    self.v_thr = np.zeros(self.n_neurons) - 40                                              # [mV] Reversal potential for spiking; equation (3)
    self.v_thr_rest =  - 40                                                                 # [mV] Reversal potential for spiking during refractory period. Defaults to -40mV; equation (3).
    self.v_spike = 20                                                                       # [mV] The voltage that spikes rise to ((AP Peak))
    self.v_syn = 0                                                                          # [mV] Reversal potential; equation (2) in paper
    self.tau_syn = 1                                                                        # [ms] Synaptic time-constant; equation (4) in paper
    self.tau_rf_thr = 5                                                                     # [ms] Timescale tau between refractory and normal thresholds relaxation period
    self.tau_spike = 1                                                                      # [ms] Time for an AP spike, also the length of the absolute refractory period
    self.g_leak = 0.02                                                                      # [mS/cm^2] Conductance of the leak channels; equation (2) in paper
    self.g_syn_initial_value = 0                                                            # [mS/cm^2] Initial value of synaptic conductance; equation (2) in paper
    # Connectivity parameters (connection weight = conductance)
    self.synaptic_delay = 3                                                                 # [ms] Synaptic transmission delay from soma to soma (default: 3 ms), equation (4) in paper
    # Threshold and Membrane Potential are set to following values after spiking ((Equation 3 in paper))
    self.v_reset = -67                                                                      # [mV] Membrane potential right after spikeing ((Hyperpolarization)); equation (3) in paper
    self.v_rf_spike = 0                                                                     # [mV] Threshold during relative refractory period; V_th_spike of equation (3) in paper
    self.capacitance = 0.001 * np.random.normal(loc=3,                                      # [mF/cm^2] Converted from micro-Farad to mF; equation (2) in paper
                                                scale=(0.05*3), 
                                                size=self.n_neurons)

    # Input noise of Poisson distribution (input is generated with I = G*V)
    self.g_poisson = 1.3                                                                    # [mS/cm^2] Conductance of the extrinsic poisson inputs
    self.poisson_freq = (20 * 0.001) * self.dt                                              # [count] Poisson input frequency (2e-3 times [per time-interval [dt = 0.1 ms]])

    # Internal Trackers
    self.spike_flag = np.zeros(self.n_neurons)                                              # Tracker of whether a neuron spiked
    self.t_spike1 = np.zeros(self.n_neurons) - 10000                                        # Tracker of spike timestamp; Default to -10000 for mathematical computation convenience; keep track of the previous timestamps of spiked neurons.
    self.t_spike2 = np.zeros(self.n_neurons) - 10000                                        # Tracker of spike timestamp; Default to -10000 for mathematical computation convenience; keep track of the current timestamps of spiked neurons.
    self.spike_record = np.empty(shape=(1, 2))                                              # Tracker of Spike record recorded as a list: [neuron_number, spike_time]
    self.g_syn = np.zeros(self.n_neurons) + self.g_syn_initial_value                        # Tracker of dynamic synaptic conductance. Initial value of 0.; equation (2)
    self.g_noise = np.zeros(self.n_neurons)                                                 # Tracker of dynamic noise conductance
    # self.poisson_input_flag = np.zeros(self.n_neurons)                                      # Tracker of Input vector of external noise to each neuron
    self.w_update_flag = np.zeros(self.n_neurons)                                           # Tracker of connetion weight updates. When a neuron spikes, it is flagged as needing update on its connection weight.
    self.spiked_input_w_sums = np.zeros(self.n_neurons)                                     # Tracker of the connected presynaptic weight sum for each neuron (Eq 4: weight * Dirac Delta Distribution)
    self.network_conn = np.zeros([self.n_neurons, self.n_neurons])                          # Tracker of Neuron connection matrix: from row-th neuron to column-th neuron
    self.network_W = np.random.random(size=(self.n_neurons, self.n_neurons))                # Tracker of Neuron connection weight matrix: from row-th neuron to column-th neuron

    # STDP paramters
    self.stdp_beta = 1.4                                                                    # Balance factor (ratio of LTD to LTP); equation (7)
    self.stdp_tau_r = 4                                                                     # When desynchronized and synchronized states coexist; equation (7)
    self.stdp_tau_plus = 10                                                                 # For the postive half of STDP; equation (7)
    self.stdp_tau_neg = self.stdp_tau_r * self.stdp_tau_plus                                # For the negative half of STDP; equation (7)
    self.eta = 0.02                                                                         # Scales the weight update per spike; 0.02 for "slow STDP"; equation (7)

    # ??? Speculate to be capacitance random value drawn from a normal distribution ???
    tau_c1 = np.sqrt(-2 * np.log(np.random.random(size=(self.n_neurons,))))                 # ??? membrane time-constant component 1 ??? - Jesse and Tony are unsure what this is 
    tau_c2 = np.cos(2 * np.pi * np.random.random(size=(self.n_neurons,)))                   # ??? membrane time-constant component 2 ??? - Jesse and Tony are unsure what this is  
    self.tau_m = 7.5 * tau_c1 * tau_c2 + 150                                                # ??? Unsure what this is, but through usage seems like the capacitance random-variable in equation (2) (However, values are off...)
    
    # Generate neuron connection matrix
    self.random_conn()  # Create random neuron connections

  def random_conn(self, mean_w: float=0.5, proba_conn: float=0.07):
    """Randomly create connections between neurons.

    Using LIF neuron objects intrinsic probability of presynaptic connections
    from other neurons (`proba_conn`), and then mean conductance of synapses 
    (mean_w) to normalize the randomly generated value.

    Update `network_W` matrix to binary values indicating the connections
    between neurons.

    Args:
        mean_w (float, optional): The mean connection weight to normalize each  
          connection in the network to. Defaults 0.5
        proba_conn (float, optional): Probability of connection between neurons.
          Defaults 0.07.
    """
    # NOTES (Tony): 
    # - `pc` is a connectivity probability matrix randomly generated with the 
    #   dimension of n_neurons * n_neurons representing the combinations of 
    #   connections between the neurons.
    # - Mark the connections that are below the "probability of presynaptic
    #   connection" threshold as 0
    # - Normalize to the mean conudctance of synapses.
    # - Set connections with normalized weight greater than 1 as 1; and set 
    #   connections with normalized weight less than 0 as 0.

    pc = np.random.random(size=(self.n_neurons,self.n_neurons))  # Connectivity probability matrix
    self.network_conn = pc < proba_conn  # Mask - Check if the connectivity probability meets the threshold `proba_conn`
    self.network_W = np.random.random(size=(self.n_neurons,self.n_neurons))  # Connectivity weight matrix
    # == FALSE; mark connections lower than probability of presynaptic connection as 0
    self.network_W[self.network_conn == 0] = 0
    # Normalized to mean conductance (i.e., `mean_w`)
    self.network_W = (self.network_W * 
                      (mean_w / np.mean(self.network_W[self.network_W > 0])))          
    self.network_W[self.network_W > 1] = 1
    self.network_W[self.network_W < 0] = 0

  def structured_conn(self, LIF, mean_w: float=0.5):
    """_summary_

    Args:
        LIF (_type_): _description_
        mean_w (float, optional): The mean connection weight to normalize each
        connection in the network to. Defaults 0.5.

    Returns:
        _type_: _description_
    """

    self.network_conn = np.zeros([self.n_neurons,self.n_neurons])  
    dist=np.empty([self.n_neurons,self.n_neurons])
    dist[:] = np.nan
    dist1=[]
    c=[]
    for f in range(LIF.n_neurons-1):
      i=f
      for j in range(f+1,LIF.n_neurons,1):
        a=(LIF.x[i]-LIF.x[j])*(LIF.x[i]-LIF.x[j])+(LIF.y[i]-LIF.y[j])*(LIF.y[i]-LIF.y[j])+(LIF.z[i]-LIF.z[j])*(LIF.z[i]-LIF.z[j])
        b=np.sqrt(a)
        c.append(b)
      #print('distance between neurons', i+1, 'and', j+1, ': ', b)
    d=sum(c)/len(c)
    print('The average distance between neurons in this network is:', d)
    print('The base of the exponent is:', LIF.p_conn**(1/d))
    bb=[]
    cc=[]
    for p in range(LIF.n_neurons):
      for p2 in range(LIF.n_neurons):
        if(p!=p2):
          a=(LIF.x[p]-LIF.x[p2])*(LIF.x[p]-LIF.x[p2])+(LIF.y[p]-LIF.y[p2])*(LIF.y[p]-LIF.y[p2])+(LIF.z[p]-LIF.z[p2])*(LIF.z[p]-LIF.z[p2])
          b=np.sqrt(a)
          dist1.append(b)
    for p in range(LIF.n_neurons):
      for p2 in range(LIF.n_neurons):
        if(p!=p2):
          a=(LIF.x[p]-LIF.x[p2])*(LIF.x[p]-LIF.x[p2])+(LIF.y[p]-LIF.y[p2])*(LIF.y[p]-LIF.y[p2])+(LIF.z[p]-LIF.z[p2])*(LIF.z[p]-LIF.z[p2])
          b=np.sqrt(a)
          #aa=((LIF.p_conn**(1/d)) ** b)
          aa=2.71828**(-b/(LIF.p_conn*max(dist1)))
          pc = np.random.random(size=(1,))
          if(pc<aa):
            self.network_conn[p][p2] = 1
            dist[p][p2]=b
    ## Tony - Verify that this block below is indeed not needed.
    # self.network_W = np.random.random(size=(self.n_neurons,self.n_neurons))
    # self.network_W[self.network_conn == 0] = 0
    # self.network_W = self.network_W/np.mean(self.network_W[self.network_W > 0]) * mean_w
    # self.network_W[self.network_W > 1] = 1
    # self.network_W[self.network_W < 0] = 0
    return dist

    
  def simulate_poisson(self, ):
    """Calculate and return Poisson spike flags for noise input calculation."""
    # NOTES (Tony):
    # - Poisson frequency is hardcoded as (20 * 0.001 * 0.1) = 20e-4 = 0.002 times 
    #   per time interval, which is set as dt = 0.1.
    # - With rate of 2e-3, there is only probability of 2e-3 that a neuron will 
    #   spike, thus multiplying the value 1 by the boolean values.
    # - Essentially the same as estimating a Poisson process with Binomial trials,
    #   thus, may make more sense to just determine spiking with a Binomial random
    #   variable.

    poisson_noise_input_flag = 1 * (np.random.rand(self.n_neurons) < self.poisson_freq)

    return poisson_noise_input_flag


  def assay_stdp(self):
    """Plot the STDP scheme assay.

    Y-axis being the connection weight update (delta w).
    X-axis being the time diff of presynaptic spike timestamp less postsynaptic
    spike timestamp. The definition of time_diff is opposite of time_lag (termed
    in the original paper).
    """
    # %matplotlib inline
    fig = plt.figure()

    for i in range(-100,100,1):
      plt.scatter(i, self.stdp_weight_update(i, 0, 0), 
                  s=2,
                  c='k')

    plt.ylabel('dW')
    plt.xlabel('time offset (pre - post)')
    plt.title('STDP curve')
    plt.show()
    self.random_conn()
    
  def stdp_weight_update(self, time_diff, pre_idx, post_idx):
    """Update and return connection weight change in STDP scheme.

    Spike-timing-dependent plasticity (STDP) scheme by updating the connection
    weights from presynaptic neuron (pre_idx) to postsynaptic neuron (post_idx)

    Args: 
      time_diff: t_{pre} - t_{post}; opposite of time_lag.
      pre_idx: index of the presynaptic neuron.
      post_idx: index of the postsynaptic neuron.

    Returns: 
      dW: Connection weight change with the time diff of time_diff.
    """
    
    # NOTES (Tony):
    # - Nearest neighbor STDP scheme by updating the weights whenever a 
    #   postsynaptic neuron spikes. 
    # - Equation (7) in the original paper; however differ in that time_diff is
    #   used instead of the time_lag as stated in equation 7 of the paper.
    # - time_diff < 0 is when the presynaptic neuron spikes before the 
    #   postsynaptic resulting in a potentiation (through connection weight 
    #   increase) of the two neurons.
    # - time_diff > 0 is when the presynaptic neuron spikes AFTER the 
    #   poststynaptic resulting in a depression (through connection weight
    #   decrease) of the two neurons.

    # TODO (TONY): 
    # - [ ] Modulize out the part of updating the connection weight matrix and 
    #   keep this method to just calculating the weight change.

    # QUESTIONS (Tony):
    # - ??? Why use -0.01 and 0.01 instead of zero as stated in eq.7 of the paper?
    # - ??? Why is the hard bound of [0, 1] used for the weights? (Reason not 
    #   specified in the paper.)


    dW = 0

    ## Case: LTP (Long-term potentiation)
    if time_diff < -0.01:
      dW = (self.eta 
            * np.exp( time_diff / self.stdp_tau_plus ))
    
    ## Case: LTD (Long-term depression)
    if time_diff > 0.01:
      dW = (self.eta 
            * -(self.stdp_beta/self.stdp_tau_r) 
            * np.exp( -time_diff / self.stdp_tau_neg ))

    ## Update connection weight
    self.network_W[pre_idx][post_idx] = self.network_W[pre_idx][post_idx] + dW

    ## Limit connection weights to hard bound [1, 0]
    self.network_W[self.network_W > 1] = 1
    self.network_W[self.network_W < 0] = 0

    return dW


  def spikeTrain(self,lookBack=None, nNeurons = 5, purge=False):
    """Plot spiketrain plot of specified neuron counts and lookBack range.

    Args: 
      lookback: length of time [ms] to backtrack for plotting the spikeTrain; 
        default None results in the entire time duration.
      nNeurons: number of neurons to plot spike train; should be <= n_neurons 
        in the LIF_Network; default = 5
      purge (boolean): Clears the spike record in the LIF_Network object

    Returns:
      SR (np.ndarray): n-by-2 ndarray recording the neuron and its spike time. 
    """

    # NOTES (Tony): 
    # - Spike record is subsetted to be loopBack onwards.
    # - Interesting way of utilizing argmax to find the first instance of 
    #   timestamp matching loopBack.
    # - Spike record: first-column: The i-th neuron,
    #                 second-column: Time that spiked.
    # - argmax returns the first instance of matched condition.
    # - np.where returns the indices of items with satisfied conditions if second
    #   and third arguments are missing for the function.
    # - The beginning attempt to subset SR with the first instance of condition
    #   match can be optimized as this only subsets partially and thus still
    #   requires the if-statement in the plotting calls. (Just not optimized.)
    # - lookBack variable name can be better named as the author used the same 
    #   variable for two purposes. One to specify the length to look back in time,
    #   two to specify the starting point of spiketrain plotting timestamp.
    # - `result` is a list of indices in the spike_record that matches that of 
    #   the specified neuron.
    # - The method starts by creating a blank canvas using `plt.plot()` with 
    #   arguments such as the number of neurons and then fill in the spiketrain
    #   information in subsequent code.

    # QUESTIONS (Tony): 
    # - ??? Why does it delete the first row of spike record?
    #   - Answer: The first row was the placeholder placed by object __init__.
    #     Since the `simulation` method only appends spike records to the 
    #     spike_record variable, the original placeholder is still occupying the 
    #     first entry of the variable.
    # - ??? Why is loopBack logic written in such convoluted way?
    #   - Answer: This may just be a preference, but the code is unpythonic.
    # - ??? Why is reshaping needed if the spike_record is already in the format?


    
    ## lookBack == 0 if None
    if lookBack is None:
      lookBack = self.t
    lookBack = self.t - lookBack  # Spiketrain plot starting timestamp
    # ## More pythonic way to accomplish the same thing
    # if lookBack is None:
    #   strt_timestamp = 0
    # else: 
    #   strt_timestamp = self.t - lookBack 
    # lookBack = strt_timestamp
    
    ## Subsetting spike record since lookBack onwards
    SR = np.reshape(self.spike_record,newshape = [-1,2])
    SR = np.delete(SR, 0, 0)               # Delete first row - the placeholder
    SRix = np.argmax(SR[:,1] >= lookBack)  # idx of first SR since timestamp == loopBack
    SR = SR[SRix:,:]                       # Subset using index
    
    ## Plotting spike records one neuron at a time
    # %matplotlib inline
    fig = plt.figure()
    plt.plot([lookBack, self.t],[0,nNeurons],'white')
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
    """

    # NOTES (Tony):
    # - The variable `lb` should be renamed as the starting point.
    # - 1000 periods if time step (dt) is 0.1 (1000 / 0.1 = 1e4)
    # - Converting the
    # - The held_neuron is a reference neuron that we are referring all other 
    #   neurons to. The reference neuron is the reference frame that we compare 
    #   other neurons to.
    # - The held_neuron is reset by the starting timestamp so that the held_neuron
    #   is viewed in the proper reference with respect to the interval we are
    #   looking back to view or plot.
    # - The held_neuron is just the neuron that we are calculating. It is set as
    #   the held_neuron when a new neuron is first encountered when iterating 
    #   through the sorted spike record.
    # - When held_neuron is reset to the starting timestamp, we are just changing
    #   the reference frame to that of the lookBack time interval.
    # - The period of a periodic function is the value at which the function
    #   repeat itself, thus we are stating that at value (100 / dt) = 1000 is when 
    #   the function repeats.
    # - The case when the ix neuron IS an held_neuron, it is limiting the frame
    #   to that specified by the lookBack and then projecting the entire spike
    #   acitivity of that neuron onto a single period. Each of the spike would be
    #   an arm on the polar coordinate thus has a radian, and this radian is 
    #   recorded into the phase_entries array to be later processed.
    # - `myarm` is named because each of the spike can be projected onto the polar
    #   coordinate as an arm.
    # - held_neuron is a variable that is outside of the for-loop and is reset 
    #   when finished processing all the phases for the previous neuron by setting
    #   to the current neuron number (i.e., ix). The phase_entries array is also
    #   cleared to make room for the next neuron number.
    # - The phase medians are like each neuron's center of gravity (COG) in the 
    #   polar coordinate, however, I am not sure why we are not taking the length
    #   from center to the COG into the calculation.
    # - Summing the Euler's equation is equivalent to summing each of the complex
    #   numbers representing the Kuramato vector denoting the mean phase of each
    #   neuron. Summing each neuron's Kuramato vector and then dividing by the 
    #   number of neuron becomes finding the phase mean of all the neurons.
    # - The absolute mean phase in radian is returned because at the midpoint of 
    #   the period, the periodic function is just opposite of that of t=0. We do
    #   not care whether the mean phase is lagging or ahead of the the periodic
    #   function at t=0, thus we only care about the magnitude of the phase, hence
    #   taking the absolute value of the mean phase of all neurons.
    # - The cutoff value is tuned to adjust for floating point errors when 
    #   calculating the the phase angle using trigonometry. Seen below when
    #   finding the phase angle [radian] using scipy's circmean, we are no longer
    #   using trigonometry to find the phase angle, thus floating point errors are
    #   eliminated.

    # QUESTIONS (Tony):
    # - SR is already in n-rows of length-2 arrays, why do we need to reshape it?
    # - Why is the time length for calculating the period variable fixed at 100ms?
    # - Why add 1 to the priod during linespace creation?
    #   - Answer: Because the evenly spaced samples includes the start and 
    #     endpoints. For example, if we would like 4 sections, we would need 5 
    #     marking points (3x points at the center and then the two ends).
    # - What is held_neuron (smallest neuron in the spike record?)
    # - What unit is the `period` variable? I think it should be ms, but it is not
    #   clear if there is even a unit.
    # - Why is the radius/hypotenus or the distance from origin to the COG only
    #   filtered at the cutoff of 0.3? What is the significance of 0.3? Why not 
    #   consider this distance in calculating the Kuramato vector? 
    # - ??? The `wraps` variable doesn't really make sense as it is dividing
    #   lookBack of unit [ms] by period of unit [count], in addition the variable 
    #   is not used nor returned, what is the point of this variable?


    # Convert period and lookback from ms to euler-steps
    if period is None:
      period=100  # 100ms
    steps_in_period = period / self.dt  # Number of Euler steps in a period
    if lookback is None:
      lookback = self.t
    steps_to_lookback = lookback / self.dt

    lb = self.t - steps_to_lookback  # Analysis starting-point timestamp [ms]

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

    """
    
    ## TODO (Tony): 
    # - [ ] Rename the period argument, name is misleading.

    ## NOTES (Tony): 
    # - The higher the period value, the higher the resolution in the calculation
    #  especially when converting to radians. Perhaps period of 100ms is adequate
    #  as it would divide 2pi into 1000 sections.

    # Convert period and lookback from ms to euler-steps
    if period is None:
      period=100  # 100ms
    steps_in_period = period / self.dt  # Number of Euler steps in a period
    if lookback is None:
      lookback = self.t
    steps_to_lookback = lookback / self.dt

    lb = self.t - steps_to_lookback  # Analysis starting-point timestamp [ms]

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
               kappa_noise: float = 0.026):
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
        2D matrix of the conductance of each neuron at each epoch (Euler-step).
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
    # NOTES (Tony):
    # - `spike_flag` seems to be an array of n_neuron length that marks whether
    #   the neuron has spiked?
    # - The `spike_flag` array marks whether the neurons fired at the specific 
    #   time step being iterated through because a neuron cannot be spiking twice
    #   at the same time slice (time step).
    # - `sp` is a vector masking the neurons that are ineligible for spiking.
    # - `t_spike1` and `t_spike2` are default to -10000 so that we are able to 
    #   filter by timestamp, and -10000 is just an arbitrary number. It can be any
    #   negative number IMO because negative timestamp does not exist and is 
    #   sufficient for the purpose of filtering by timestamp.
    # - `sp` checking if the `spike_flag` is 0 seems to tell that `spike_flag` 
    #   marks whether the neuron is spiking or not. If it has a value of 1, it
    #   indicates that it is ineligible to spike, thus even if the membrane
    #   potential passes the threshold, it still won't spike. This seems to tell
    #   that if a neuron has a FALSE spike_flag, it is in an absolute refractory
    #   period and thus can't spike no-matter the membrane potential.
    #   - `spike_flag == 0` =?= "neuron in aboslute refractory period"
    # - Upon consideration, the only reason one needs `t_offset * f` is because
    #   `spike_flag` is actually tracking whether the neuron is in abs-refractory
    #   period and `f` marks the neurons that have `spike_flag == 1`.
    # - Typical abs-refractory periods are 1-2 ms, which matches the code because
    #   our timesteps are dt = 0.1ms and thus it probable that multiple timesteps
    #   would still be within a neuron's aboslute refractory period.
    # - [ ] Propose changing name of variable `spike_flag` to `abs_rf_flag`.
    # - When the neuron spikes, it's connection weight is also due to update due
    #   to the STDP scheme. Thus, we keep track of the connections that needs to
    #   be updated with the `w_update_flag`.
    # ` `f` seems similar to `sp` at first but they are drastically different for
    #   the following reason.
    #   - Each for loop iteration is only one timestep, thus it moves the time by
    #     0.1 ms, thus there will be cases when a neuron has not spiked
    #     (`sp == 0`), but still in absolute-refractory period (`spike_flag == 1`)
    #     thus making the neuron unable to spike again.
    #   - For the above reason, I am proposing changing the name `spike_flag` to 
    #     `abs_rf_flag` as it helps with the understanding of the code.
    # - `tau_spike` same as `abs_rf_time_length`, in [ms]
    # - ##### I AM WRONG! `spike_flag` should be `rel_rf_flag`,              #####
    #   ##### and `t_offset` should be `abs_rf_flag`.                        #####
    # - ##### I AM WRONG AGAIN!!! `spike_flag` should be `rf_flag`           #####

    # TODO (Tony):
    # - [ ] Check if Delta_W method has a 0.01 threshold instead of 0 because of 
    #   the 0.1 implementation in STDP below. Check the Fortran code for this.
    # - [ ] Optimize the "End of Epoch" section using the step variable of the
    #   for-loop.
    # - [ ] Run simulation with kappa = 400, which was found from Ali's Fortran code.

    # QUESTIONS (Tony)
    # - ??? Why was the matrix I declared twice `I = I = np.zeros([n_times, self.n_neurons])`
    # - ??? Why are the voltage thresholds of the neurons that fired (i.e., 
    #   `v_thr[f]`) set to the refractory period potential (`v_rf_spike = 0`).
    #     - This tells me that this is just a relative refractory period and thus 
    #       it is still possible to elicit an AP, just harder because the 
    #       threshold is now 0mV instead of -40mV.
    # - ??? `spike_flag == 0` =?= "neuron in aboslute refractory period"
    # - ??? Makes more sense to rename `timesteps` argument as `time` because time
    #   divided by timestep (dt) would yield the number of steps. The actual
    #   timestamp is being tracked with `self.t` and moved forward with
    #   `self.t += self.dt`?
    # - ??? Is the "Informing post-synaptic neuron partner" backpropagation?
    
    # Set variables
    per_neuron_coup_strength = kappa / self.n_neurons # [mS/cm^2] Neuron coupling strength (Network-coupling-strength / number-of-neurons)
    external_stim_coup_strength = kappa / 5  # Coupling strength of input external inputs (i.e., vibrotactile stimuli); value 5 is arbitrary for a strong coupling strength.

    euler_steps = int(sim_duration/self.dt)   # Number of Euler-method steps
    euler_step_idx_start = self.t / self.dt  # Euler-step starting index

    ## External stimulation current input matrix
    if I_stim == None:
      I_stim = np.zeros(shape=(euler_steps, self.n_neurons))
    # Weight sums of external spiked and connected input (conductance)
    if external_spiked_input_w_sums == None: 
      external_spiked_input_w_sums = np.zeros(shape=(euler_steps, self.n_neurons))

    ## Output variable placeholders
    holder_epoch_timestamps = np.zeros((euler_steps, ))
    holder_v = np.zeros((euler_steps, self.n_neurons))
    holder_g_syn = np.zeros((euler_steps, self.n_neurons))
    holder_poi_noise_flags = np.zeros((euler_steps, self.n_neurons))
    holder_spiked_input_w_sums = np.zeros((euler_steps, self.n_neurons))
    holder_dw = np.zeros((euler_steps, ))


    ## Euler-step Loop
    for step in range(euler_steps):  # Step-loop: because (time_duration/dt = steps OR sections)
      start = time.perf_counter()
      # <<<<<<< DEBUG (Original noise generation)
      ## Generate Poisson noise input flags and input current
      poisson_noise_spike_flag = self.simulate_poisson()
      # poisson_noise_spiked_input_count = np.matmul(poisson_noise_spike_flag, self.network_conn)

      # Update Conductance (denoted g) - Integrate inputs from noise and synapses
      # # Method 1: Original method from Fortran code
      self.g_noise = self.g_noise * np.exp(-self.dt/self.tau_syn) + self.g_poisson * poisson_noise_spike_flag
      # Method 2: According to the paper's equations (equation 6)
      # del_g_noise = (-self.g_noise 
      #                + kappa_noise * self.tau_syn * poisson_noise_spiked_input_count) * np.exp(-self.dt/self.tau_syn)
      # self.g_noise = (self.g_noise + del_g_noise)
      # =======
      # ## Generate Poisson noise input flags and input current
      # poisson_noise_spike_flag = self.simulate_poisson()
      # poisson_noise_spiked_input_count = np.matmul(poisson_noise_spike_flag, self.network_conn)

      # # Update Conductance (denoted g) - Integrate inputs from noise and synapses
      # # # Method 1: Original method from Fortran code
      # # self.g_noise = self.g_noise * np.exp(-self.dt/self.tau_syn) 
      # # Method 2: According to the paper's equations (equation 6)
      # del_g_noise = (-self.g_noise 
      #                + kappa_noise * self.tau_syn * poisson_noise_spiked_input_count) * np.exp(-self.dt/self.tau_syn)
      # self.g_noise = (self.g_noise + del_g_noise)
      # >>>>>>> DEBUG (Poisson Noise generation)

      # # Method 1: Original method from Fortran code
      # self.syn_g = (self.syn_g * np.exp(-self.dt/self.tau_syn)
      #               + per_neuron_coup_strength * self.spiked_input_w_sums
      #               + external_stim_coup_strength * external_spiked_input_w_sums[step, :]
      #               )
      # Method 2: According to the paper's equation (equation 4)
      del_g_syn = (-self.g_syn 
                   + kappa/self.n_neurons * self.tau_syn * self.spiked_input_w_sums 
                   + external_stim_coup_strength * external_spiked_input_w_sums[step, :]) * np.exp(-self.dt/self.tau_syn) 
      self.syn_g = (self.g_syn + del_g_syn)
    

      ## Reset variables
      self.spiked_input_w_sums = np.zeros(self.n_neurons)     # Weight sum of all spiked-connected presynaptic neurons
      self.w_update_flag = np.zeros(self.n_neurons)           # Connection weight update tracker
      dW = 0                                                  # Net connection weight change per epoch

      ## Update membrane-potential, spiking-threshold
      # # Dynamic membrane potential - Version 1 - Replicated from Ali's Fortran code - 
      # self.v = (self.v + (self.dt/self.tau_m) * ((self.g_leak) * (self.v_rest - self.v) 
      #                                            + (self.g_noise + self.g_syn) * (self.v_rest - self.v)))
      # # Dynamic membrane potential - Version 2 - Replicated from Ali's Fortran code - with reversal potential of Na+ (20mV) instead of v_rest=0mV
      # self.v = (self.v + (self.dt/self.tau_m) * ((self.g_leak) * (self.v_rest - self.v) 
      #                                            + (self.g_noise + self.g_syn) * (20 - self.v)))
      # Dynamic membrane potential - Version 3 - Formula from the paper (equation 2)
      I_noise = self.g_noise * (self.v_syn - self.v)
      del_v = (self.g_leak * (self.v_reset - self.v)
               + self.g_syn * (self.v_syn - self.v)
               + I_stim[step] 
               + I_noise) * (self.dt / self.capacitance)
      self.v = (self.v + del_v)

      # Dynamic spiking threshold
      # # Method 1: From Ali's Fortran code
      # self.v_thr = (self.v_thr
      #         + (self.dt/self.tau_rf_thr) * (self.v_thr_rest-self.v_thr))
      # Method 2: Paper's formula (equation 3) - This also implements the exponential decay as that in other Euler Scheme formulas
      del_v_thr = (self.v_thr_rest - self.v_thr) * np.exp(-self.dt/self.tau_rf_thr)
      self.v_thr = (self.v_thr + del_v_thr)

      ## Depolarizing to meet spiking-threshold
      spike = ((self.v >= self.v_thr) *  # Met dynamic spiking threshold
               (self.spike_flag == 0))   # Not in abs_refractory period because not recently spiked
      self.spike_flag[spike] = 1
      self.w_update_flag[spike] = 1                # Connection-weight update tracker
      self.t_spike1[spike] = self.t_spike2[spike]  # Moves the t_spike2 array into t_spike1 for placeholding
      self.t_spike2[spike] = self.t                # t_spike2 keeps track of each neuron's most recent spike's timestamp

      ## Spiking phase
      spiked = (self.spike_flag == 1)
      self.v[spiked] = self.v_spike         # Rectangle spike shape by setting voltage to V_spike for duration of tau_spike (equation 3)
      self.v_thr[spiked] = self.v_rf_spike  # Threshold is reset to V_th_spike=0mV right after spiking (equation 3)
      in_abs_rf_period = (self.t_spike2 + self.tau_spike) > self.t

      ## Hyperpolarization phase
      self.v[(~in_abs_rf_period) * spiked] = self.v_reset
      # Reset spike tracker
      self.spike_flag[(~in_abs_rf_period) * spiked] = 0
      

      ## Dirac Delta Distribution (equation 4 in paper)
      spike_t_diff = self.t - (self.t_spike2 + self.synaptic_delay)  # [n, ] array
      s_flag = 1.0 * (abs(spike_t_diff) < 0.01)  # 0.01 for floating point errors
      # Presynaptic neurons' weight sum for each neuron
      self.spiked_input_w_sums = np.matmul(s_flag,
                                     self.network_W * self.network_conn)
      stop = time.perf_counter()
      print(f"Dynamic functions' total processing time: {stop-start} s")
      start = time.perf_counter()
      ## STDP (Spike-timing-dependent plasticity)
      ## Note: Iterates over all pairs of connections using double-nested loops
      if self.w_update_flag.any():
        print(sum(self.w_update_flag))
        for pre_idx in range(self.n_neurons):
          
          if (self.w_update_flag[pre_idx] == 1):
            # Add spike record
            self.spike_record = np.append(self.spike_record,
                                          np.array([pre_idx, self.t]))

            for post_idx in range(self.n_neurons):

              # Check last spike of pre-synaptic partners (forward propagation):
              if self.network_conn[pre_idx][post_idx] == 1:
                temporal_diff = (self.t_spike2[pre_idx] + self.synaptic_delay 
                                 - self.t_spike2[post_idx])

                if temporal_diff > 0:  # LTD
                  dW = dW + self.stdp_weight_update(temporal_diff, pre_idx, post_idx)
                else:                  # LTP (temporal_diff >= 0)
                  # Addresses the case when temporal_diff = 0
                  # NOTE: t_spike1 for neurons that spiked would only differ
                  #       from t_spike2 by dt=0.1
                  temporal_diff = (self.t_spike2[pre_idx] + self.synaptic_delay 
                                   - self.t_spike1[post_idx])
                  dW = dW + self.stdp_weight_update(temporal_diff, pre_idx, post_idx)

              # Inform post-synaptic partners about spike (backpropagation):
              elif self.network_conn[post_idx][pre_idx] == 1:  
                temporal_diff =  (self.t_spike2[post_idx] + self.synaptic_delay 
                                  - self.t_spike2[pre_idx])
                dW = dW + self.stdp_weight_update(temporal_diff, post_idx, pre_idx)
                # self.stdp_weight_update(temporal_diff, post_idx, pre_idx)
      stop = time.perf_counter()
      print(f"Total time to iterate through conn matrix with stdp calls: {stop-start} s")
      # End of Epoch:
      # NOTE: Used so that multiple simulation runs have continuity.
      tix = int(self.euler_step_idx - euler_step_idx_start)
      holder_epoch_timestamps[tix] = self.t
      holder_v[tix] = self.v   
      holder_g_syn[tix] = self.g_syn
      holder_poi_noise_flags[tix] = poisson_noise_spike_flag
      holder_spiked_input_w_sums[tix] = self.spiked_input_w_sums
      holder_dw[tix] = dW

      # Increment Euler-step index
      self.euler_step_idx += 1

      ## Increment time tracker
      self.t += self.dt
    
    return (holder_v, 
            holder_g_syn, 
            holder_poi_noise_flags, 
            holder_epoch_timestamps, 
            holder_spiked_input_w_sums, 
            holder_dw)
 
