class LearningWorld:
  def __init__(self):
    self.name = "Stimulus Input"
    self.n_states = 1  # Lets start with 11 stats,range(0,.1,1.1) kuramato orders
    self.n_actions = 4 # Lets have 10 frequencies and 10 sites
    self.init_state = 0

  def get_outcome(self, state, action, LIF, ConnStimNetwork):
    pre_w = np.mean(LIF.network_W.flatten())
    if(action==0):
        LIF.network_W = LIF.network_W + 0.03
    if(action==1):
        LIF.network_W = LIF.network_W
    if(action==2):
        LIF.network_W = LIF.network_W - 0.015
    if(action==3):
        LIF.network_W = LIF.network_W - 0.06

    next_state = 0
    reward = pre_w - np.mean(LIF.network_W.flatten())
    return int(next_state) if next_state is not None else None, reward

  def get_all_outcomes(self):
    outcomes = {}
    for state in range(self.n_states):
      for action in range(self.n_actions):
        next_state, reward = self.get_outcome(state, action)
        outcomes[state, action] = [(1, next_state, reward)]
    return outcomes
