import matplotlib.pyplot as plt    # MatPlotLib is a plotting package. 
import numpy as np                 # NumPy is a numerical types package.
from scipy import stats            # ScPy is a scientific computing package. We just want the stats, because Ca2+ imaging is always calculated in z-score.
from scipy.stats import circmean
import math
from posixpath import join

class LIF_Network:
  def __init__(self, n_neurons = 1000, dimensions = [[0,100],[0,100],[0,100]]):
    """Leaky Intergrate-and-Fire (LIF) Neuron network model.

    Args:
      n_neurons (int): number of neurons; default to 1000 neurons
      dimensions: Spatial dimensions for plotting; this plots the neurons in a 
        100x100x100 3D space by default.

    Attr:
      n_neurons: 
      dimensions: 
      x: 
      y:
      z: 
      t: current time [ms]
      dt: time step [ms]
      euler_step_idx: time index
      relax_time: length of relaxation phase [ms]
      v: internal voltage [mV]
      v_rest: resting voltage, equilibrium [mV] def: -38
      v_thr: 
      v_rf_thr: reversal potential for spiking during refractory period [mV]
      v_rf_tau: 

    Returns:

    NOTES:
    - Maxima

    TODO (Tony):
    - [ ] Revisit the parameters of STDP and understand what they are doing.
    """

    # Neuron count
    self.n_neurons = n_neurons
    
    # Spatial organization
    self.dimensions = dimensions;
    self.x = np.random.uniform(low=self.dimensions[0][0], high=self.dimensions[0][1], size=(self.n_neurons,))
    self.y = np.random.uniform(low=self.dimensions[1][0], high=self.dimensions[1][1], size=(self.n_neurons,))
    self.z = np.random.uniform(low=self.dimensions[2][0], high=self.dimensions[2][1], size=(self.n_neurons,))

    # Internal time trackers
    self.t = 0                                                                              # current time [ms]
    self.dt = .1                                                                            # timestep leng [ms]
    self.euler_step_idx = 0                                                                 # time index
    self.relax_time = 10000                                                                 # length of relaxation phase [ms]

    # Electrophysiology values
    self.v = np.random.uniform(low=-10, high=10, size=(self.n_neurons,)) - 45               # internal voltage [mV]
    self.v_rest = -38                                                                       # the resting voltage, equilibrium [mV] def: -38

    self.v_thr = np.ones([self.n_neurons,]) * -40                                           # Potential Threshold for spiking [mV]
    self.v_rf_thr = np.ones([self.n_neurons,]) * -40                                        # Potential Threshold for spiking during (relative) refractory period [mV]
    self.v_rf_tau = 5                                                                       # the relaxation tau between refractory and normal thresholds.

    self.v_reset = -67                                                                      # the reset, overshoot voltage [mV] ((??? Hyperpolarization))
    self.v_spike = 20                                                                       # the voltage that spikes rise to [mV] ((??? AP Peak))
    self.v_rf_spike = 0                                                                     # relative refractory period potential threshold [mV]

    self.spike_length = 1                                                                   # [ms] Time for an AP spike, also the length of the absolute refractory period
    self.spike_flag = np.zeros([self.n_neurons,])                                           # Tracker of whether a neuron spiked
    self.t_spike1 = np.zeros([self.n_neurons,]) - 10000                                     # Tracker of spike timestamp; Default to -10000 for mathematical computation convenience; keep track of the previous timestamps of spiked neurons.
    self.t_spike2 = np.zeros([self.n_neurons,]) - 10000                                     # Tracker of spike timestamp; Default to -10000 for mathematical computation convenience; keep track of the current timestamps of spiked neurons.
    self.spike_record = np.empty(shape=[1,2])                                               # Spike record recorded as a list: [neuron_number, spike_time]
    self.g_leak = 10                                                                        # [nS] Conductance of the leak channels

    tau_c1 = np.sqrt(-2 * np.log(np.random.random(size=(self.n_neurons,))))                 # ???membrane time-constant component 1 ???
    tau_c2 = np.cos(2 * np.pi * np.random.random(size=(self.n_neurons,)))                   # ???membrane time-constant component 2 ???                                           
    self.m_tau = 7.5 * tau_c1 * tau_c2 + 150                                                # [ms] Membrane time-constant

    self.syn_tau = 1                                                                        # [ms] Synaptic time-constant
    self.v_syn = 0                                                                          # [mV] Reversal potential; equation (2) of paper
    self.syn_g = np.zeros([self.n_neurons,])                                                # Tracker of dynamic synaptic conductance
    
    self.g_poisson = 1.3                                                                    # [mS/cm^2] Conductance of the extrinsic poisson inputs
    self.poisson_freq = (20 * 0.001) * self.dt                                              # [count] Poisson input frequency (2e-3 times [per time-interval [dt = 0.1 ms]])
    self.poisson_input_flag = np.zeros([self.n_neurons,])                                   # Input vector of external noise to each neuron
    self.noise_g = np.zeros([self.n_neurons,])                                              # Tracker of dynamic noise conductance

    # STDP paramters
    self.stdp_beta = 1.4                                                                    # Balance factor for LTP and LTD
    self.stdp_tau_R = 4                                                                     # When desynchronized and synchronized states coexist
    self.stdp_tau_plus = 10                                                                 # For the postive half of STDP
    self.stdp_tau_neg = self.stdp_tau_R * self.stdp_tau_plus                                # For the negative half of STDP
    self.eta = 0.02                                                                         # Scales the weight update per spike; 0.02 for "slow STDP" (eq 7 in paper)
    self.w_update_flag = np.zeros([self.n_neurons,])                                        # Tracker of connetion weight updates
                                                                                            # When a neuron spikes, it is flagged as needing update on its connection weight.
    
    # Connectivity parameters (connection weight = conductance)
    self.proba_conn = .07                                                                   # Probability of presynaptic connections from other neurons
    self.mean_w = 0.5                                                                       # [mS/cm^2] Mean conductance of synapses; (conductance = connection weight)
    self.synaptic_delay = 3                                                                 # [ms] Time for an AP to propogate from a pre- to post-synaptic neuron (default: 3 ms)
    max_coup_strength = 400                                                                 # [mS/cm^2] Maximal coupling strength; maximal conductance (Equation 4 in paper)
    self.per_neuron_coup_strength = max_coup_strength / self.n_neurons                      # [mS/cm^2] Neuron coupling strength (Network-coupling-strength / number-of-neurons)
    self.connected_input_w_sum = np.zeros([self.n_neurons,])                                # Tracker of the connected presynaptic weight sum for each neuron (Eq 4: weight * Dirac Delta Distribution)
    self.external_stim_coup_strength = max_coup_strength / 5                                # Coupling strength of input external inputs (i.e., vibrotactile stimuli); value 5 is arbitrary for a strong coupling strength.
    self.network_conn = np.zeros([self.n_neurons,self.n_neurons])                           # Neuron connection matrix: from row-th neuron to column-th neuron
    self.network_W = np.random.random(size=(self.n_neurons,self.n_neurons))                 # Neuron connection weight matrix: from row-th neuron to column-th neuron
    
    # Generate neuron connection matrix
    self.random_conn()  # Create random neuron connections

  def random_conn(self,):
    """Randomly create connections between neurons.

    Using LIF neuron objects intrinsic probability of presynaptic connections
    from other neurons (`proba_conn`), and then mean conductance of synapses 
    (mean_w) to normalize the randomly generated value.

    Update `network_W` matrix to binary values indicating the connections
    between neurons.

    NOTES (Tony): 
    - `pc` is a connectivity probability matrix randomly generated with the 
      dimension of n_neurons * n_neurons representing the combinations of 
      connections between the neurons.
    - Mark the connections that are below the "probability of presynaptic
      connection" threshold as 0
    - Normalize to the mean conudctance of synapses.
    - Set connections with normalized weight greater than 1 as 1; and set 
      connections with normalized weight less than 0 as 0.
    """

    pc = np.random.random(size=(self.n_neurons,self.n_neurons))  # Connectivity probability matrix
    self.network_conn = pc < self.proba_conn  # Mask - Check if the connectivity probability meets the threshold `proba_conn`
    self.network_W = np.random.random(size=(self.n_neurons,self.n_neurons))  # Connectivity weight matrix
    # == FALSE; mark connections lower than probability of presynaptic connection as 0
    self.network_W[self.network_conn == 0] = 0
    # Normalized to mean conductance (i.e., `mean_w`)
    self.network_W = (self.network_W * 
                      (self.mean_w / np.mean(self.network_W[self.network_W > 0])))          
    self.network_W[self.network_W > 1] = 1
    self.network_W[self.network_W < 0] = 0

  def structured_conn(self,LIF,):
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
    # self.network_W = self.network_W/np.mean(self.network_W[self.network_W > 0]) * self.mean_w
    # self.network_W[self.network_W > 1] = 1
    # self.network_W[self.network_W < 0] = 0
    return dist

    
  def simulate_poisson(self,):
    """Generate Poisson spike trains.
    
    NOTES (Tony):
    - Poisson frequency is hardcoded as (20 * 0.001 * 0.1) = 20e-4 = 0.002 times 
      per time interval, which is set as dt = 0.1.
    - With rate of 2e-3, there is only probability of 2e-3 that a neuron will 
      spike, thus multiplying the value 1 by the boolean values.
    - Essentially the same as estimating a Poisson process with Binomial trials,
      thus, may make more sense to just determine spiking with a Binomial random
      variable.
    """

    self.poisson_input_flag = 1 * (np.random.rand(self.n_neurons,) < self.poisson_freq)
  
  def assaySTDP(self):
    """Plot the STDP function curve.

    Y-axis being the connection weight update (delta w).
    X-axis being the time diff of presynaptic spike timestamp less postsynaptic
    spike timestamp. The definition of time_diff is opposite of time_lag (termed
    in the original paper).
    """
    # %matplotlib inline
    fig = plt.figure()

    for i in range(-100,100,1):
      plt.scatter(i, self.Delta_W_tau(i, 0, 0), 
                  s=2,
                  c='k')

    plt.ylabel('dW')
    plt.xlabel('time offset (pre - post)')
    plt.title('STDP curve')
    plt.show()
    self.random_conn()
    
  def Delta_W_tau(self, time_diff, pre_idx, post_idx):
    """Update and return connection weight change in STDP scheme.

    Spike-timing-dependent plasticity (STDP) scheme by updating the connection
    weights from presynaptic neuron (pre_idx) to postsynaptic neuron (post_idx)

    Args: 
      time_diff: t_{pre} - t_{post}; opposite of time_lag.
      pre_idx: index of the presynaptic neuron.
      post_idx: index of the postsynaptic neuron.

    Returns: 
      dW: Connection weight change with the time diff of time_diff.

    ---
    
    NOTES (Tony):
    - Nearest neighbor STDP scheme by updating the weights whenever a 
      postsynaptic neuron spikes. 
    - Equation (7) in the original paper; however differ in that time_diff is
      used instead of the time_lag as stated in equation 7 of the paper.
    - time_diff < 0 is when the presynaptic neuron spikes before the 
      postsynaptic resulting in a potentiation (through connection weight 
      increase) of the two neurons.
    - time_diff > 0 is when the presynaptic neuron spikes AFTER the 
      poststynaptic resulting in a depression (through connection weight
      decrease) of the two neurons.

    TODO (TONY): 
    - [ ] Modulize out the part of updating the connection weight matrix and 
      keep this method to just calculating the weight change.

    QUESTIONS (Tony):
    - ??? Why use -0.01 and 0.01 instead of zero as stated in eq.7 of the paper?
    - ??? Why is the hard bound of [0, 1] used for the weights? (Reason not 
      specified in the paper.)
    """

    dW = 0

    ## Case: LTP (Long-term potentiation)
    if time_diff < -0.01:
      dW = (self.eta 
            * np.exp( time_diff / self.stdp_tau_plus ))
    
    ## Case: LTD (Long-term depression)
    if time_diff > 0.01:
      dW = (self.eta 
            * -(self.stdp_beta/self.stdp_tau_R) 
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


    NOTES (Tony): 
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

    QUESTIONS (Tony): 
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

  def vect_kuramato(self,period=None,lookBack=None, r_cutoff = .3):
    """Return the phase mean of all neurons in the spike record.

    Kuramato vectors and trigonometry are used to calculate the phase mean,
    which has potential for rounding errors due to floating points. Thus, 
    r_cuttoff is used to eliminate the potential errors. The larger the
    potential rounding error, the larger the r_cutoff is needed.

    Args: 
      period: [count] The number of sections to split a period into.
        If set to the None, we are assuming a period of 100ms and timesteps of 
        0.1ms, thus yielding 1000 sections in a single cycle.
      lookBack: [ms] The length of time we are looking back to analyze.
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

    NOTES (Tony):
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

    QUESTIONS (Tony):
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

    if period is None:
      period=100/self.dt    # 100 milliseconds; period has unit [count]
    if lookBack is None:
      lookBack = self.t     # [ms]
    lb = self.t - lookBack  # Starting-point timestamp [ms]

    # Spike record
    SR = np.reshape(self.spike_record,newshape = [-1,2])  
    SR = np.delete(SR, 0, 0)         # Delete first row - The placeholder
    SRix = np.argmax(SR[:,1] >= lb)  # Index of the starting point
    SR = SR[SRix:,:]
    SR = sorted(SR,key=lambda x: x[0])  # Sort based on 1st column - neuron
    wraps = lookBack/period

    N = self.n_neurons
  
    theta = np.random.normal(size=[N,])
    phasespace = np.linspace(0,2*np.pi,int(period+1))  # includes both ends

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
        while myarm > period:  # Project onto one period
          myarm = myarm - period
        myarm = phasespace[int(np.round(myarm))]  # Phase of the spikes
        phase_entries.append(myarm)

    ## Filter out the NaN
    phase_medians = phase_medians[~np.isnan(phase_medians)]

    # e^{xi} = cos(x) + (i)*sin(x) -- Euler's Equation
    z = 1/N * np.sum(np.exp(1.0j * phase_medians))  # Phase mean of all neurons
    r = np.abs(z)  # Only care about the magnitude
    return r  # Mean phase of all neurons in [radian]

  
  def kuramato(self,period=None,lookBack=None):
    """Return the phase mean of all neurons in the spike record.

    circmean() instead of trigonometry is used to find the mean phase of all 
    the spikes of a neuron, thus, there is higher precision, hence, eliminating
    the need of a r_cutoff argument to trim off the phase noise created by
    rounding floating points.

    Essentially, this is a more accurate version of the `vect_kuramato` method.

    ## TODO (Tony): 
    - [ ] Rename the period argument, name is misleading.

    ## NOTES (Tony): 
    - The higher the period value, the higher the resolution in the calculation
      especially when converting to radians. Perhaps period of 100ms is adequate
      as it would divide 2pi into 1000 sections.

    Args: 
      period: [count] The number of sections to split a period into.
        If set to the None, we are assuming a period of 100ms and timesteps of 
        0.1ms, thus yielding 1000 sections in a single cycle.
      lookBack: [ms] The length of time we are looking back to analyze.
        If set to the default `None`, we are considering the entire span of the
        spike record.

    Returns: 
      r: [radian] Magnitude of the mean Kuramato vectors of all neurons.

    """

    if period is None:
      period=100/self.dt  # 100ms for a cycle; given dt=0.1, thus 1000 sections.
    if lookBack is None:
      lookBack = self.t
    lb = self.t - lookBack  # Analysis starting-point timestamp [ms]

    # Spike record
    SR = np.reshape(self.spike_record,newshape = [-1,2])
    SR = np.delete(SR, 0, 0)         # Placeholder value in first row
    SRix = np.argmax(SR[:,1] >= lb)  # Index of first instance
    SR = SR[SRix:,:]
    SR = sorted(SR,key=lambda x: x[0])
    wraps = lookBack/period          # ??? What is this for? 

    N = self.n_neurons
  
    theta = np.random.normal(size=[N,])
    phasespace = np.linspace(0,2*np.pi,int(period+1))  # Includes both ends

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
        while myarm > period:  # Project onto the first period
          myarm = myarm - period
        myarm = phasespace[int(np.round(myarm))]
        phase_entries.append(myarm)

    phase_medians = phase_medians[~np.isnan(phase_medians)]

    z = 1/N * np.sum(np.exp(1.0j * phase_medians))
    r = np.abs(z)  # Only care about the magnitude
    return r


  def simulate(self, 
               sim_duration:float = 1, 
               epoch_current_input:"np.NDarray" = None):
    """Run simulation

    Args: 
      sim_duration: [ms] Duration of time in milliseconds.
      epoch_current_input (ndarray): 
        2D ndarray matrix denoting the current input for each neuron per each
        epoch (Euler-step); current in mA.

    Returns: 
      v_holder (ndarray): 
        2D matrix of the membrane potential of each neuron at 
        each epoch (Euler-step).
      gsyn_holder (ndarray): 
        2D matrix of the conductance of each neuron at each epoch (Euler-step).
      pois_holder (ndarray): 
        2D matrix of binary outcomes (spike vs not-spike) of each neuron 
        at each epoch (Euler-step).
      t_holder (ndarray): Timestamps of each epoch (Euler-step).
      in_holder (ndarray): 
        2D matrix of connected presynaptic weight sum for each neuron at 
        each epoch (Euler-step). Rows are for each epoch, and columns are for
        each post-synaptic neuron.
      dW_holder (ndarray): 
        1D array of the net connection weight change of the entire network 
        at each epoch (Euler-step).

    NOTES (Tony):
    - `spike_flag` seems to be an array of n_neuron length that marks whether
      the neuron has spiked?
    - The `spike_flag` array marks whether the neurons fired at the specific 
      time step being iterated through because a neuron cannot be spiking twice
      at the same time slice (time step).
    - `sp` is a vector masking the neurons that are ineligible for spiking.
    - `t_spike1` and `t_spike2` are default to -10000 so that we are able to 
      filter by timestamp, and -10000 is just an arbitrary number. It can be any
      negative number IMO because negative timestamp does not exist and is 
      sufficient for the purpose of filtering by timestamp.
    - `sp` checking if the `spike_flag` is 0 seems to tell that `spike_flag` 
      marks whether the neuron is spiking or not. If it has a value of 1, it
      indicates that it is ineligible to spike, thus even if the membrane
      potential passes the threshold, it still won't spike. This seems to tell
      that if a neuron has a FALSE spike_flag, it is in an absolute refractory
      period and thus can't spike no-matter the membrane potential.
      - `spike_flag == 0` =?= "neuron in aboslute refractory period"
    - Upon consideration, the only reason one needs `t_offset * f` is because
      `spike_flag` is actually tracking whether the neuron is in abs-refractory
      period and `f` marks the neurons that have `spike_flag == 1`.
    - Typical abs-refractory periods are 1-2 ms, which matches the code because
      our timesteps are dt = 0.1ms and thus it probable that multiple timesteps
      would still be within a neuron's aboslute refractory period.
    - [ ] Propose changing name of variable `spike_flag` to `abs_rf_flag`.
    - When the neuron spikes, it's connection weight is also due to update due
      to the STDP scheme. Thus, we keep track of the connections that needs to
      be updated with the `w_update_flag`.
    ` `f` seems similar to `sp` at first but they are drastically different for
      the following reason.
      - Each for loop iteration is only one timestep, thus it moves the time by
        0.1 ms, thus there will be cases when a neuron has not spiked
        (`sp == 0`), but still in absolute-refractory period (`spike_flag == 1`)
        thus making the neuron unable to spike again.
      - For the above reason, I am proposing changing the name `spike_flag` to 
        `abs_rf_flag` as it helps with the understanding of the code.
    - ##### I AM WRONG! `spike_flag` should be `rel_rf_flag`,              #####
      ##### and `t_offset` should be `abs_rf_flag`.                        #####
    - `spike_length` same as `abs_rf_time_length`, in [ms]
    - ##### I AM WRONG AGAIN!!! `spike_flag` should be `rf_flag`           #####

    TODO (Tony):
    - [ ] Check if Delta_W method has a 0.01 threshold instead of 0 because of 
      the 0.1 implementation in STDP below. Check the Fortran code for this.
    - [ ] Optimize the "End of Epoch" section using the step variable of the
      for-loop.

    QUESTIONS (Tony)
    - ??? Why was the matrix I declared twice `I = I = np.zeros([n_times, self.n_neurons])`
    - ??? Why are the voltage thresholds of the neurons that fired (i.e., 
      `v_thr[f]`) set to the refractory period potential (`v_rf_spike = 0`).
        - This tells me that this is just a relative refractory period and thus 
          it is still possible to elicit an AP, just harder because the 
          threshold is now 0mV instead of -40mV.
    - ??? `spike_flag == 0` =?= "neuron in aboslute refractory period"
    - ??? Makes more sense to rename `timesteps` argument as `time` because time
      divided by timestep (dt) would yield the number of steps. The actual
      timestamp is being tracked with `self.t` and moved forward with
      `self.t += self.dt`?
    - ??? Is the "Informing post-synaptic neuron partner" backpropagation?



    """
    euler_steps = int(sim_duration/self.dt)   # Number of Euler-method steps
    euler_step_idx_start = self.t / self.dt  # Euler-step starting index

    ## External input current matrix
    if epoch_current_input == None:
      epoch_current_input = np.zeros(shape = [euler_steps, self.n_neurons])

    ## Output variable placeholders
    t_holder = np.zeros([euler_steps, ])
    v_holder = np.zeros([euler_steps, self.n_neurons])
    gsyn_holder = np.zeros([euler_steps, self.n_neurons])
    pois_holder = np.zeros([euler_steps, self.n_neurons])
    in_holder = np.zeros([euler_steps, self.n_neurons])
    dW_holder = np.zeros([euler_steps, ])


    ## Euler-step Loop
    for step in range(euler_steps):  # Step-loop: because (time_duration/dt = steps OR sections)
      
      ## Generate poisson inputs
      self.simulate_poisson()  # Saved in `self.poisson_input_flag`

      ## Update Conductance (denoted g) - Integrate inputs from noise and synapses
      ## Option 1: Non-exponential decay ##
      # self.noise_g = ((1-self.dt) * self.noise_g + 
      #                 self.g_poisson * self.poisson_input_flag)
      # self.syn_g = ((1-self.dt) * self.syn_g + 
      #               self.per_neuron_coup_strength * self.connected_input_w_sum)
      ## Option 2: Exponential decay ##
      self.noise_g = (self.noise_g * np.exp(-self.dt/self.syn_tau) 
                      + self.g_poisson * self.poisson_input_flag)  # Poisson conductance * poisson_input_flag makes sense because poisson_input_flag is binary outcome.
      self.syn_g = (self.syn_g * np.exp(-self.dt/self.syn_tau)
                    + self.per_neuron_coup_strength * self.connected_input_w_sum
                    + self.external_stim_coup_strength * epoch_current_input[step][:])


      ## Reset inputs
      self.connected_input_w_sum = np.zeros([self.n_neurons,])  # Connected presnaptic input weight sum for each neuron
      self.w_update_flag = np.zeros([self.n_neurons,])  # Connection weight update tracker
      dW = 0                                            # Net connection weight change per epoch

      ## Update membrane-potential, spiking-threshold
      # Dynamic membrane potential
      self.v = self.v + self.dt * (((self.v_rest - self.v)
                                    - (self.noise_g + self.syn_g) 
                                    * self.v)
                                   / self.m_tau)
      # Dynamic spiking threshold
      self.v_thr = (self.v_thr
                    + self.dt 
                      * (self.v_rf_thr - self.v_thr) 
                      / self.v_rf_tau)

      ## Depolarizing to meet spiking-threshold
      spike = ((self.v >= self.v_thr) *  # Met dynamic spiking threshold
               (self.spike_flag == 0))   # Not in abs_refractory period because not recently spiked
      self.spike_flag[spike] = 1
      self.w_update_flag[spike] = 1                # Connection-weight update tracker
      self.t_spike1[spike] = self.t_spike2[spike]  # Moves the t_spike2 array into t_spike1 for placeholding
      self.t_spike2[spike] = self.t                # t_spike2 keeps track of each neuron's most recent spike's timestamp

      ## Spiking phase
      spiked = (self.spike_flag == 1)
      self.v[spiked] = self.v_spike         # Update membrane potential to AP peak
      self.v_thr[spiked] = self.v_rf_spike  # Update spiking-threshold to the relative-refractory period threshold

      ## Hyperpolarization phase
      in_abs_rf_period = (self.t_spike2 + self.spike_length) > self.t
      self.v[(~in_abs_rf_period) * spiked] = self.v_reset
      # Reset spike tracker
      self.spike_flag[(~in_abs_rf_period) * spiked] = 0
      

      ## Dirac Delta Distribution (equation 4 in paper)
      spike_t_diff = self.t - (self.t_spike2 + self.synaptic_delay)  # [n, ] array
      s_flag = 1.0 * (abs(spike_t_diff) < 0.01)  # 0.01 for floating point errors
      # Presynaptic neurons' weight sum for each neuron
      self.connected_input_w_sum = np.matmul(s_flag,
                                     self.network_W * self.network_conn)

      ## STDP (Spike-timing-dependent plasticity)
      ## Note: Iterates over all pairs of connections using double-nested loops
      if self.w_update_flag.any():

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
                  dW = dW + self.Delta_W_tau(temporal_diff, pre_idx, post_idx)
                else:                  # LTP (temporal_diff >= 0)
                  # Addresses the case when temporal_diff = 0
                  # NOTE: t_spike1 for neurons that spiked would only differ
                  #       from t_spike2 by dt=0.1
                  temporal_diff = (self.t_spike2[pre_idx] + self.synaptic_delay 
                                   - self.t_spike1[post_idx])
                  dW = dW + self.Delta_W_tau(temporal_diff, pre_idx, post_idx)

              # Inform post-synaptic partners about spike (backpropagation):
              elif self.network_conn[post_idx][pre_idx] == 1:  
                temporal_diff =  (self.t_spike2[post_idx] + self.synaptic_delay 
                                  - self.t_spike2[pre_idx])
                dW = dW + self.Delta_W_tau(temporal_diff, post_idx, pre_idx)
                # self.Delta_W_tau(temporal_diff, post_idx, pre_idx)

      # End of Epoch:
      # NOTE: Used so that multiple simulation runs have continuity.
      tix = int(self.euler_step_idx - euler_step_idx_start)
      t_holder[tix] = self.t
      v_holder[tix] = self.v   
      gsyn_holder[tix] = self.syn_g + self.noise_g
      pois_holder[tix] = self.poisson_input_flag
      in_holder[tix] = self.connected_input_w_sum
      dW_holder[tix] = dW

      # Increment Euler-step index
      self.euler_step_idx += 1

      ## Increment time tracker
      self.t += self.dt
    
    return v_holder, gsyn_holder, pois_holder, t_holder, in_holder, dW_holder
 
