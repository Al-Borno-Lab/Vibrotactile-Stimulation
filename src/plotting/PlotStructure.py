def plot_structure(LIF, conn= False, conn_target = None):
  if conn == True and conn_target is None:
    conn_target = range(LIF.n_neurons)
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(LIF.x,LIF.y,LIF.z)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('Location of cells')
  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  cm = plt.cm.get_cmap('viridis', LIF.n_neurons)

  if conn:
    C = LIF.network_conn
    for i in range(LIF.n_neurons):
      for j in range(LIF.n_neurons):
        if C[i][j] > 0 and i in conn_target:
          ax.plot([LIF.x[i], LIF.x[j]], [LIF.y[i], LIF.y[j]], [LIF.z[i], LIF.z[j]], color=cm(i),linewidth=.75)
  plt.show()

def plot_connectivity(LIF):
  fig = plt.figure()
  ax = fig.add_subplot()
  plt.imshow(LIF.network_conn,aspect='equal',interpolation='none')
  ax.set_aspect('auto')
  plt.ylabel('Neuron index')
  plt.xlabel('Connectivity to other neurons')
  plt.title('Connectivity map')
  
  plt.show()

def plot_voltage(sim, n = 5):

  [v,g,p,t,inp,dw] = sim
  fig = plt.figure()
  for i in range(n):
    plt.plot(t[:-1],v[:-1,i]-i*100)
  plt.xlabel('time [ms]')
  plt.ylabel('neural voltage, by index x 100')
  plt.title('example voltages')
  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.show()

def plotter(LIF,time,pN = 5):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(LIF.x,LIF.y,LIF.z)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('Location of cells')
  plt.show()

  fig = plt.figure()
  ax = fig.add_subplot()
  plt.imshow(LIF.network_conn,aspect='equal',interpolation='none')
  ax.set_aspect('auto')
  plt.ylabel('Neuron index')
  plt.xlabel('Connectivity to other neurons')
  plt.title('Connectivity map')
  plt.show()
  print(np.mean(LIF.network_conn.flatten()))

  fig = plt.figure()
  ax = fig.add_subplot()
  plt.imshow(LIF.network_W,aspect='equal',interpolation='none')
  plt.colorbar()
  ax.set_aspect('auto')
  plt.ylabel('Neuron index')
  plt.xlabel('Weight to other neurons')
  plt.title('Weights pre-training')
  plt.show()

  W = np.copy(LIF.network_W)
  Wf = W.flatten()
  fig = plt.figure()
  plt.plot(np.sort(Wf[Wf>0]))
  plt.xlabel('order of weights')
  plt.ylabel('weight')
  plt.title('Sorted weights pre-training')
  plt.show()
  print(np.mean(W[W > 0].flatten()))

  h = LIF.simulate(sim_duration = time)
  [v,g,p,t,inp,dw] = h
  W2 = LIF.network_W
  
  fig = plt.figure()
  ax = fig.add_subplot()
  plt.imshow(W2,aspect='equal',interpolation='none')
  plt.colorbar()
  ax.set_aspect('auto')
  plt.ylabel('Neuron index')
  plt.xlabel('Weight to other neurons')
  plt.title('Weights after-training')
  plt.show()

  fig = plt.figure()
  ax = fig.add_subplot()
  plt.imshow((W2-W),aspect='equal',interpolation='none')
  plt.colorbar()
  ax.set_aspect('auto')
  plt.ylabel('Neuron index')
  plt.xlabel('Weight to other neurons')
  plt.title('Change in Weights after-training')
  plt.show()

  print(np.mean(W2[W2 > 0].flatten()))
  Wf2 = W2.flatten()
  fig = plt.figure()
  plt.plot(np.sort(Wf[Wf>0]))
  plt.plot(np.sort(Wf2[Wf>0]))
  plt.xlabel('order of weights')
  plt.ylabel('weight')
  plt.show()

  fig = plt.figure()
  for i in range(pN):
    plt.plot(t,v[:,i]-i*100)
  plt.xlabel('time [ms]')
  plt.ylabel('neural voltage (offset by index)')
  plt.title('example neural activity')
  plt.show()

  fig = plt.figure()
  for i in range(pN):
    plt.plot(t,g[:,i]-i*50)  # ??? What is this 50??? 
  plt.xlabel('time [ms]')
  plt.ylabel('syaptic current (offset by index)')
  plt.title('example neural currents')
  plt.show()

  fig = plt.figure()
  for i in range(pN):
    plt.plot(t,p[:,i]-i*1)
  plt.xlabel('time [ms]')
  plt.ylabel('poisson spikes (offset by index)')
  plt.title('example external inputs')
  plt.show()

  fig = plt.figure()
  for i in range(2):
    plt.plot(t,inp[:,i])
  plt.xlabel('time [ms]')
  plt.ylabel('poisson spikes (offset by index)')
  plt.title('example internal inputs')
  plt.show()

  fig = plt.figure()
  plt.plot(t,dw)
  plt.xlabel('time [ms]')
  plt.ylabel('weight changes')
  plt.title('dW')
  plt.show()







def plotter2(LIF:LIF_Network, 
             sim_duration, 
             I:"ndarray", 
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