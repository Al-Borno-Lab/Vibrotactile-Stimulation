import numpy as np
from src.model import lifNetwork
from numpy import typing as npt
import matplotlib.pyplot as plt

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
    raise AssertionError("tau_r, tau_plus, tau_neg: two of the three have to be provided.")
  elif (tau_r is None) & (tau_plus is None):
    raise AssertionError("Either tau_r or tau_plus is needed.")
  elif (tau_plus is None) & (tau_neg is None):
    raise AssertionError("Either tau_plus or tau_neg is needed.")
  elif (tau_neg is None) & (tau_r is None):
    raise AssertionError("Either tau_neg or tau_r is needed.")
  
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
    if dw < 0:
      raise AssertionError(f"During LTP, dw should be >= 0, it is {dw}")
  
  ## Case: LTD (Long-term depression)
  if np.greater(time_diff, 0):
    dw = (scale_factor 
          * -(stdp_beta / tau_r) 
          * np.exp( -time_diff / tau_neg ))
    if dw > 0:
      raise AssertionError(f"During LTD, dw should be <= 0, it is {dw}")

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


###################### IGNORE BELOW: WORK IN PROGRESS ##########################

# def stdp_scheme_logic(time_diff:float,
#                       scale_factor:float, stdp_beta:float,
#                       stdp_tau_r:float=None, 
#                       stdp_tau_plus:float=None, stdp_tau_neg:float=None) -> float:
    
#     # print(f"{'>'*10} stdp_scheme_logic TRACING {'>'*10}")
    
#     assert ((stdp_tau_plus is not None) 
#         | (stdp_tau_neg is not None)), "Either tau_plus or tau_neg have to be supplied."
#     assert ((stdp_tau_r is not None) | 
#             ((stdp_tau_plus is not None) & (stdp_tau_neg is not None))), "tau_r is required if either tau_plus or tau_neg is not supplied."

#     # Calculate stdp_tau_neg
#     if stdp_tau_neg is None: 
#         stdp_tau_neg = stdp_tau_r * stdp_tau_plus

#     # Logic
#     dw = tf.cast(0, tf.float32)
#     if (time_diff >= -0.01) | (time_diff <= 0.01):
#         return dw
#     if time_diff < -0.01:  # LTP
#         dw = (scale_factor * tf.math.exp( time_diff / stdp_tau_plus))
#     elif time_diff > 0.01: # LTD
#         dw = (scale_factor 
#               * -(stdp_beta/stdp_tau_r) 
#               * tf.math.exp(-time_diff/stdp_tau_neg))
        
#     return dw


# # def graph_update_weights(network_w:tf.Variable, 
# #                          conn_matrix:tf.Tensor,
# #                          w_update_flag:tf.Tensor,
# #                          temporal_diff:tf.Tensor,
# #                          stdp_scheme_param:dict) -> tf.Variable:    
    
# #     # print(f"{'>'*10} graph_update_weights TRACING {'>'*10}")
    
# #     # Update the network weight matrix
# #     updates = conn_matrix * w_update_flag * temporal_diff
# #     fn = lambda time_diff: stdp_scheme_logic(time_diff=time_diff,
# #                                              **stdp_scheme_param,)
    
# #     elems = tf.reshape(updates, shape=(-1, 1))
# #     dw = tf.map_fn(fn=fn, elems=elems)
# #     dw = tf.reshape(dw, shape=tf.shape(updates))
# #     network_w.assign_add(dw)
    
# #     return network_w


# def conn_update_STDP(network_w:npt.NDArray, conn_matrix:npt.NDArray,
#                      w_update_flag:npt.NDArray, syn_delay:float,
#                      t_spike1:npt.NDArray, t_spike2:npt.NDArray, 
#                      stdp_scheme_param:dict=None,) -> npt.NDArray:
#     """_summary_

#     Args:
#         nn (lifNetwork.LIF_Network): _description_

#     Returns:
#         _type_: _description_
#     """
#     if stdp_scheme_param is None: 
#         stdp_scheme_param = {"scale_factor": 0.02,
#                              "stdp_tau_plus": 10, 
#                              "stdp_tau_neg": 40,
#                              "stdp_tau_r": 4,
#                              "stdp_beta": 1.4}

#     # Convert to tensors
#     network_w = tf.Variable(tf.cast(tf.convert_to_tensor(network_w), tf.float32))
#     conn_matrix = tf.cast(tf.convert_to_tensor(conn_matrix), tf.float32)
#     w_update_flag = tf.cast(tf.convert_to_tensor(w_update_flag), tf.float32)
#     t_spike1 = tf.cast(tf.convert_to_tensor(t_spike1), tf.float32)
#     t_spike2 = tf.cast(tf.convert_to_tensor(t_spike2), tf.float32)
#     syn_delay = tf.cast(tf.convert_to_tensor(syn_delay), tf.float32)
    

#     def graph_calculations(network_w, conn_matrix, w_update_flag, t_spike1, t_spike2, syn_delay):
#         print(f"{'>'*10} graph_calculations TRACING {'>'*10}")
#         # Reshape tensors
#         w_update_flag = tf.reshape(w_update_flag, shape=(-1, 1))
#         t_spike1 = tf.reshape(t_spike1, shape=(-1, 1))
#         temporal_diff = t_spike2 + syn_delay - t_spike1

#         # Update the network weight matrix
#         updates = conn_matrix * w_update_flag * temporal_diff
#         fn = lambda time_diff: stdp_scheme_logic(time_diff=time_diff,
#                                                 **stdp_scheme_param,)
        
#         elems = tf.reshape(updates, shape=(-1, 1))
#         dw = tf.map_fn(fn=fn, elems=elems)

#         dw = tf.reshape(dw, shape=tf.shape(updates))
#         network_w.assign_add(dw)

#         return network_w

#     network_w = graph_calculations(network_w, conn_matrix, w_update_flag, t_spike1, t_spike2, syn_delay)

#     # Hard bound connection weights to [1, 0]
#     # tf.cond(tf.math.greater(network_w, 1), lambda: 1)
#     # tf.cond(tf.math.less(network_w, 0), lambda: 0)
    
#     return network_w