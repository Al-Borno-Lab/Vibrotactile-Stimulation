#@title LIF Model

# Define LIF neurons:

from numba import njit, jit, prange
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import circmean
import numpy as np
from tqdm import tqdm

class LIF_Network:
  def __init__(self, n_neurons = 1000, dimensions = [[0,100],[0,100],[0,100]], plotGraphs=False):
    
    # plots data is set true
    self.plotGraphs = plotGraphs
    
    # Neuron count
    self.n_neurons = n_neurons
    
    # spatial organization.
    self.dimensions = dimensions
    self.x = np.random.uniform(low=self.dimensions[0][0], high=self.dimensions[0][1], size=(self.n_neurons,))
    self.y = np.random.uniform(low=self.dimensions[1][0], high=self.dimensions[1][1], size=(self.n_neurons,))
    self.z = np.random.uniform(low=self.dimensions[2][0], high=self.dimensions[2][1], size=(self.n_neurons,))

    # internal time trackers
    self.t = 0                 # current time [ms]
    self.dt = .1               # time step [ms]
    self.t_itt = 0             # time index
    self.relax_time = 10000    # length of relaxation phase [ms]

    # electrophysiology values
    self.v = np.random.uniform(low=-10, high=10, size=(self.n_neurons,)) - 45               #  internal voltage [mV]
    self.v_rest = -38                                                                       #  the resting voltage, equilibrium [mV] def: -38

    self.v_thr = np.ones([self.n_neurons,]) * -40                                           #  reversal potential for spiking [mV]
    self.v_rf_thr = np.ones([self.n_neurons,]) * -40                                        #  reversal potential for spiking during refractory period [mV]
    self.v_rf_tau = 5                                                                       #  the relaxation tau between refractory and normal thresholds.

    self.v_reset = -67                                                                      #  the reset, overshoot voltage [mV]
    self.v_spike = 20                                                                       #  the voltage that spikes rise to [mV]
    self.v_rf_spike = 0                                                                     #  the threshold that changes in the refractory period [mV]
    self.spike_length = 1
    self.spike_flag = np.zeros([self.n_neurons,])
    self.t_spike1 = np.zeros([self.n_neurons,]) - 10000
    self.t_spike2 = np.zeros([self.n_neurons,]) - 10000
    self.spike_record = np.empty(shape=[1,2])
    self.g_leak = 10                                                                        #  the conductance of the leak channels [nS]

    tau_c1 = np.sqrt(-2 * np.log(np.random.random(size=(self.n_neurons,))))                 #  membrane time-constant component 1
    tau_c2 = np.cos(2 * np.pi * np.random.random(size=(self.n_neurons,)))                   #  membrane time-constant component 2                                               
    self.m_tau = 7.5 * tau_c1 * tau_c2 + 150                                                #  the membrane time-constant [ms]

    self.syn_tau = 1                                                                        #  the synaptic time-constant [ms]
    self.v_syn = 0                                                                          #  the voltage synapses push the cell membrane towards [mV]
    self.syn_g = np.zeros([self.n_neurons,])                                                #  dynamic tracker of synaptic conductance.
    
    self.g_poisson = 1.3                                                                    #  the conductance of the extrinsic poisson inputs.
    self.poisson_freq = 20 * self.dt * .001                                                 #  poisson input frequency
    self.poisson_input = np.zeros([self.n_neurons,])                                        #  the input vector from external noise
    self.noise_g = np.zeros([self.n_neurons,])                                              #  dynamic tracker of noise conductance.

    # STDP paramters
    self.stdp_beta = 1.4                                                                    #  the balance factor for LTP and LTD
    self.stdp_tau_R = 4                                                                     #  used for the negative half of STDP
    self.stdp_tau_plus = 10                                                                 #  used for the postive half of STDP
    self.stdp_tau_neg = self.stdp_tau_R * self.stdp_tau_plus                                #  used for the negative half of STDP

    self.lamda = 0.02
    self.w_flag = np.zeros([self.n_neurons,])
    
    # Connectivity parameters
    self.p_conn = .07                                                                       #  probability of presynaptic connections from other neurons.
    self.mean_w = 0.5                                                                       #  mean conductance of synapses.
    self.synaptic_delay = 3                                                                 #  the amount of time an AP takes to propogate [ms]. def: 3
    C = 400
    self.network_coupling = C/self.n_neurons                                                #  coupling strength of extrinsic input noise?
    self.network_input = np.zeros([self.n_neurons,])                                        #  dynamic tracker of synaptic inputs
    self.external_strength = C/5
    self.network_conn = np.zeros([self.n_neurons,self.n_neurons])                           #  network connectivity
    self.network_W = np.random.random(size=(self.n_neurons,self.n_neurons)) 
    self.random_conn()

  def random_conn(self,):
    pc = np.random.random(size=(self.n_neurons, self.n_neurons))
    self.network_conn = pc < self.p_conn
    list_of_connection = list(zip(*np.where(self.network_conn == 1)))



    self.network_W = np.random.random(size=(self.n_neurons,self.n_neurons))
    self.network_W[self.network_conn == 0] = 0
    self.network_W = self.network_W/np.mean(self.network_W[self.network_W > 0]) * self.mean_w
    self.network_W[self.network_W > 1] = 1
    self.network_W[self.network_W < 0] = 0

  # Jesse note: this function can be matrix-ized to save processing time. # To do.
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
    # print('The average distance between neurons in this network is:', d)
    # print('The base of the exponent is:', LIF.p_conn**(1/d))
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
            dist[p][p2] = b
    return dist
    self.network_W = np.random.random(size=(self.n_neurons,self.n_neurons))
    self.network_W[self.network_conn == 0] = 0
    self.network_W = self.network_W/np.mean(self.network_W[self.network_W > 0]) * self.mean_w
    self.network_W[self.network_W > 1] = 1
    self.network_W[self.network_W < 0] = 0

  def simulate_poisson(self,):
    self.poisson_input = 1 * (np.random.rand(self.n_neurons,) < self.poisson_freq)
  
  def assaySTDP(self):

    if self.plotGraphs:
        fig = plt.figure()

        for i in range(-100,100,1):
            plt.scatter(i,self.Delta_W_tau(i,0,0),s=2,c='k')

        plt.ylabel('dW')
        plt.xlabel('time offset (pre - post)')
        plt.title('STDP curve')
        plt.show()
    
    self.random_conn()
    
  def Delta_W_tau(self,time_diff,i,j):
    dW = 0
    if time_diff < -0.01:
      dW = self.lamda * np.exp( time_diff / self.stdp_tau_plus )
    
    if time_diff > 0.01:
      dW = -(self.stdp_beta/self.stdp_tau_R) * self.lamda * np.exp( -time_diff / self.stdp_tau_neg)

    self.network_W[i][j] = self.network_W[i][j] + dW
    self.network_W[self.network_W > 1] = 1
    self.network_W[self.network_W < 0] = 0
    return dW

  def spikeTrain(self,lookBack=None, nNeurons = 5, purge=False):
    if lookBack is None:
      lookBack = self.t
    lookBack = self.t - lookBack

    SR = np.reshape(self.spike_record, newshape = [-1,2])
    SR = np.delete(SR, 0, 0)
    SRix = np.argmax(SR[:,1] >= lookBack)
    SR = SR[SRix:,:]
    
    if self.plotGraphs:
        fig = plt.figure()
        plt.plot([lookBack, self.t],[0,nNeurons],'white')
        for i in range(nNeurons):
            result = np.array(np.where(SR[:,0] == i)).flatten()
            for q in range(len(result)):
                loc = result[q]
                if (SR[loc,1]) >= lookBack:
                    plt.plot([SR[loc,1], SR[loc,1]],[i,i+.9],'k',linewidth=.5)
        
        plt.xlabel('time (ms)')
        plt.ylabel('neuron #')
        fig.set_size_inches(5, 4)
        plt.show()

    if purge:
      self.spike_record = np.empty(shape=[1,2])
    
    return SR

  def vect_kuramato(self,period=None,lookBack=None, r_cutoff = .3):
    if period is None:
      period=100/self.dt # 100 milliseconds.
    if lookBack is None:
      lookBack = self.t
    lb = self.t - lookBack

    # Spike record
    SR = np.reshape(self.spike_record,newshape = [-1,2])
    SR = np.delete(SR, 0, 0)
    SRix = np.argmax(SR[:,1] >= lb)
    SR = SR[SRix:,:]
    SR = sorted(SR,key=lambda x: x[0])
    wraps = lookBack/period

    N = self.n_neurons
  
    theta = np.random.normal(size=[N,])
    phasespace = np.linspace(0,2*np.pi,int(period+1))

    held_neuron = np.min(SR[:][0])
    phase_entries = []
    phase_medians = np.zeros(shape=[N,])
    for i in range(len(SR)):
      ix = SR[i][0]
      if ix != held_neuron:
        x = np.cos(phase_entries)
        y = np.sin(phase_entries)
        mx = np.mean(x)
        my = np.mean(y)
        rho = np.sqrt(mx**2 + my**2)
        phi = np.arctan2(my, mx)
        if rho >= r_cutoff:
          phase_medians[int(ix)] = phi
        else:
          phase_medians[int(ix)] = np.NaN
        held_neuron = ix
        phase_entries = []
      else:
        myarm = SR[i][1]-lb
        while myarm > period:
          myarm = myarm - period
        myarm = phasespace[int(np.round(myarm))]
        phase_entries.append(myarm)

    phase_medians = phase_medians[~np.isnan(phase_medians)]

    z = 1/N * np.sum(np.exp(1.0j * phase_medians))
    r = np.abs(z)
    return r

  
  def kuramato(self,period=None,lookBack=None):
    if period is None:
      period=100/self.dt # 100 milliseconds.
    if lookBack is None:
      lookBack = self.t
    lb = self.t - lookBack

    # Spike record
    SR = np.reshape(self.spike_record,newshape = [-1,2])
    SR = np.delete(SR, 0, 0)
    SRix = np.argmax(SR[:,1] >= lb)
    SR = SR[SRix:,:]
    SR = sorted(SR,key=lambda x: x[0])
    wraps = lookBack/period

    N = self.n_neurons
  
    theta = np.random.normal(size=[N,])
    phasespace = np.linspace(0,2*np.pi,int(period+1))

    held_neuron = np.min(SR[:][0])
    phase_entries = []
    phase_medians = np.zeros(shape=[N,])
    for i in range(len(SR)):
      ix = SR[i][0]
      if ix != held_neuron:

        phase_medians[int(ix)] = circmean(phase_entries)
        held_neuron = ix
        phase_entries = []
      else:
        myarm = SR[i][1]-lb
        while myarm > period:
          myarm = myarm - period
        myarm = phasespace[int(np.round(myarm))]
        phase_entries.append(myarm)

    phase_medians = phase_medians[~np.isnan(phase_medians)]

    z = 1/N * np.sum(np.exp(1.0j * phase_medians))
    r = np.abs(z)
    return r

  # @numba.jit(nopython=True, parallel=True)
  def simulate(self, timesteps = 1 ,I = None):
    
    n_time = int(timesteps/self.dt)

    if I is None:
      I = I = np.zeros(shape = [n_time,self.n_neurons])

    # Varaible exporters:
    t_holder = np.zeros([n_time,])
    v_holder = np.zeros([n_time,self.n_neurons])
    gsyn_holder = np.zeros([n_time,self.n_neurons])
    pois_holder = np.zeros([n_time,self.n_neurons])
    in_holder = np.zeros([n_time,self.n_neurons])
    dW_holder = np.zeros([n_time,])

    init_time = self.t/self.dt

    # Time loop:
    ii = 0
    for t in tqdm(range(n_time)):

      # Get poisson inputs:
      self.simulate_poisson()

      # Integrate inputs from noise and synapses
      # Updating to exp decay...
      # self.noise_g = (1-self.dt) * self.noise_g + self.g_poisson * self.poisson_input
      self.noise_g = self.noise_g * np.exp(-self.dt/self.syn_tau) + self.g_poisson * self.poisson_input
      # Updating to exp decay...
      # self.syn_g = (1-self.dt) * self.syn_g + self.network_coupling * self.network_input
      self.syn_g = self.syn_g * np.exp(-self.dt/self.syn_tau) + self.network_coupling * self.network_input + self.external_strength*I[ii][:]

      # Input reset
      self.network_input = np.zeros([self.n_neurons,])
      self.w_flag = np.zeros([self.n_neurons,])
      dW = 0

      # Update V and Thr
      self.v = self.v + self.dt * ( ( (self.v_rest - self.v) - (self.noise_g + self.syn_g) * self.v) / self.m_tau )
      self.v_thr = self.v_thr + self.dt * (self.v_rf_thr - self.v_thr) / self.v_rf_tau

      # Do spike calculations:
      sp = (self.v >= self.v_thr) * (self.spike_flag == 0)
      self.spike_flag[sp] = 1
      self.w_flag[sp] = 1
      self.t_spike1[sp] = self.t_spike2[sp]
      self.t_spike2[sp] = self.t

      f = (self.spike_flag == 1)
      self.v[f] = self.v_spike
      self.v_thr[f] = self.v_rf_spike

      t_offset = self.t_spike2+self.spike_length <= self.t
      self.spike_flag[t_offset * f] = 0
      self.v[t_offset * f] = self.v_reset

      s_difference = self.t-(self.t_spike2+self.synaptic_delay)
      s_flag = 1.0 * (abs(s_difference) < .01)
      self.network_input = np.matmul(s_flag.T, self.network_W * self.network_conn)

      # STDP:
      dW, self.spike_record = self.loopOverNeurons(
        self.network_conn,
        self.network_W,
        self.n_neurons,
        self.spike_record,
        self.w_flag,
        self.t_spike1,
        self.t_spike2,
        self.synaptic_delay,
        self.stdp_beta,
        self.stdp_tau_R,
        self.stdp_tau_plus,
        self.stdp_tau_neg,
        self.lamda,
        t
      )
      # TODO: REMVOVE COMMENT BELOW AFTER {loopOverNeurons} is tested
      # if self.w_flag.any():
      #   for i in range(self.n_neurons):
      #     if (self.w_flag[i] == 1):
      #       self.spike_record = np.append(self.spike_record,np.array([i,self.t]))
      #       for j in range(self.n_neurons):

      #         # Check for last spike of pre-synaptic partners:
      #         if self.network_conn[i][j] == 1:
                
      #           # TD       =   (post-synaptic spike - pre-synaptic spike (+) offset by delay)
      #           temporal_diff = self.t_spike2[i] - self.t_spike2[j]  + self.synaptic_delay
      #           if temporal_diff > 0:
      #             dW = dW + self.Delta_W_tau(temporal_diff,i,j)
      #           else:
      #             temporal_diff = self.t_spike2[i] - self.t_spike1[j] + self.synaptic_delay
      #             dW = dW + self.Delta_W_tau(temporal_diff,i,j)

      #         # Now inform post-synaptic parters about spike.
      #         if self.network_conn[j][i] == 1: 
      #           temporal_diff =  self.t_spike2[j] - self.t_spike2[i] + self.synaptic_delay
      #           dW = dW + self.Delta_W_tau(temporal_diff,j,i)



      # End of Epoch:
      tix = int(self.t_itt-init_time)
      t_holder[tix] = self.t
      v_holder[:][tix] = self.v   
      gsyn_holder[:][tix] = self.syn_g + self.noise_g
      pois_holder[:][tix] = self.poisson_input
      in_holder[:][tix] = self.network_input
      dW_holder[tix] = dW

      self.t_itt += 1
      self.t += self.dt
      ii += 1
    
    return v_holder, gsyn_holder, pois_holder, t_holder, in_holder, dW_holder

  @staticmethod
  @njit(parallel=True) # turned into static method to paralize loops
  def loopOverNeurons(network_conn,
                      network_W, 
                      n_neurons, 
                      spike_record, 
                      w_flag,
                      t_spike1,
                      t_spike2, 
                      synaptic_delay,
                      stdp_beta,
                      stdp_tau_R,
                      stdp_tau_plus,
                      stdp_tau_neg,
                      lamda,
                      t
                      ):

    def Delta_W_tau(time_diff,i,j):
      dW = 0
      if time_diff < -0.01:
        dW = lamda * np.exp( time_diff / stdp_tau_plus )
      
      if time_diff > 0.01:
        dW = -(stdp_beta/stdp_tau_R) * lamda * np.exp( -time_diff / stdp_tau_neg)

      network_W[i][j] = network_W[i][j] + dW
      
      if network_W[i][j] > 1:
        network_W[i][j] = 1
      elif network_W[i][j] < 0:
        network_W[i][j] = 0

      # loop over to fix range
      # for ii in range(network_W.shape[0]):
      #   for jj in range(network_W.shape[1]):
          
      #     if network_W[ii][jj] > 1:
      #       network_W[ii][jj] = 1
      #     elif network_W[ii][jj] < 0:
      #       network_W[ii][jj] = 0

      return dW

    dW = 0
    if w_flag.any():
      
      for i in range(n_neurons):
        if (w_flag[i] == 1):
          spike_record = np.append(spike_record, np.array([[i,t]], dtype=np.float64), axis=0)
          
      for i in prange(n_neurons):
          for j in range(n_neurons):

            # Check for last spike of pre-synaptic partners:
            if network_conn[i][j] == 1:
              
              # TD       =   (post-synaptic spike - pre-synaptic spike (+) offset by delay)
              temporal_diff = t_spike2[i] - t_spike2[j]  + synaptic_delay
              if temporal_diff > 0:
                dW = dW + Delta_W_tau(temporal_diff,i,j)
              else:
                temporal_diff = t_spike2[i] - t_spike1[j] + synaptic_delay
                dW = dW + Delta_W_tau(temporal_diff,i,j)

            # Now inform post-synaptic parters about spike.
            if network_conn[j][i] == 1: 
              temporal_diff =  t_spike2[j] - t_spike2[i] + synaptic_delay
              dW = dW + Delta_W_tau(temporal_diff,j,i)

    return dW, spike_record
            

 