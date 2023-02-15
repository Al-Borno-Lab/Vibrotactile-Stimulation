class StimulusInput:
  def __init__(self):
    self.name = "Stimulus Input"
    self.n_states = 11  # Lets start with 11 stats,range(0,.1,1.1) kuramato orders
    self.n_actions = 5*5 # Lets have 10 frequencies and 10 sites
    self.init_state = 0

  def get_outcome(self, state, action, LIF, ConnStimNetwork):
    reward = 0
    next_state = 0

    #basically we want the agent to pick a stimulation based on the current state.
    # To get the outcome lets run the simulation and get the order...
    # Let's take MAtteo's work as the baseline experiment:
    dt = LIF.dt
    
    T = int(50000/10) # This is 100 seconds, I divided it by 10 because the stimorder has nof actions in it (10).
    stim_length = 50000
    
    n = LIF.n_neurons
    I = np.zeros(shape = [int(T/dt),n])
    Wpre = np.copy(LIF.network_W) # Starting weights.

    freq = np.floor(action/10) * 10 + 10
    stimorder = action%10
    num = int(1000/freq/dt)
    held_i = 0    
    
    while held_i<T/dt:
        for i in range(0,stim_length,num): 
            if i+held_i<T/dt:                
                I[i+held_i][ConnStimNetwork[stimorder,:]>0] = 1      
        held_i = held_i + stim_length

    LIF.simulate(timesteps = T,I=I)
    LIF.simulate(timesteps = 2000)
    Wpost = np.copy(LIF.network_W) # Starting weights.
    ord = LIF.kuramato(period = 250, lookBack = 2000)

    next_state = np.round(ord*10)
    reward = 1-ord
    return int(next_state) if next_state is not None else None, reward

  def get_all_outcomes(self):
    outcomes = {}
    for state in range(self.n_states):
      for action in range(self.n_actions):
        next_state, reward = self.get_outcome(state, action)
        outcomes[state, action] = [(1, next_state, reward)]
    return outcomes
