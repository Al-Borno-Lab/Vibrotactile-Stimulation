import numpy as np
import tensorflow as tf
from src.model import lifNetwork
from numpy import typing as npt

def stdp_scheme_logic(time_diff:float,
                      scale_factor:float, stdp_beta:float,
                      stdp_tau_r:float=None, 
                      stdp_tau_plus:float=None, stdp_tau_neg:float=None) -> float:
    
    # print(f"{'>'*10} stdp_scheme_logic TRACING {'>'*10}")
    
    assert ((stdp_tau_plus is not None) 
        | (stdp_tau_neg is not None)), "Either tau_plus or tau_neg have to be supplied."
    assert ((stdp_tau_r is not None) | 
            ((stdp_tau_plus is not None) & (stdp_tau_neg is not None))), "tau_r is required if either tau_plus or tau_neg is not supplied."

    # Calculate stdp_tau_neg
    if stdp_tau_neg is None: 
        stdp_tau_neg = stdp_tau_r * stdp_tau_plus

    # Logic
    dw = tf.cast(0, tf.float32)
    if (time_diff >= -0.01) | (time_diff <= 0.01):
        return dw
    if time_diff < -0.01:  # LTP
        dw = (scale_factor * tf.math.exp( time_diff / stdp_tau_plus))
    elif time_diff > 0.01: # LTD
        dw = (scale_factor 
              * -(stdp_beta/stdp_tau_r) 
              * tf.math.exp(-time_diff/stdp_tau_neg))
        
    return dw


# def graph_update_weights(network_w:tf.Variable, 
#                          conn_matrix:tf.Tensor,
#                          w_update_flag:tf.Tensor,
#                          temporal_diff:tf.Tensor,
#                          stdp_scheme_param:dict) -> tf.Variable:    
    
#     # print(f"{'>'*10} graph_update_weights TRACING {'>'*10}")
    
#     # Update the network weight matrix
#     updates = conn_matrix * w_update_flag * temporal_diff
#     fn = lambda time_diff: stdp_scheme_logic(time_diff=time_diff,
#                                              **stdp_scheme_param,)
    
#     elems = tf.reshape(updates, shape=(-1, 1))
#     dw = tf.map_fn(fn=fn, elems=elems)
#     dw = tf.reshape(dw, shape=tf.shape(updates))
#     network_w.assign_add(dw)
    
#     return network_w


def conn_update_STDP(network_w:npt.NDArray, conn_matrix:npt.NDArray,
                     w_update_flag:npt.NDArray, syn_delay:float,
                     t_spike1:npt.NDArray, t_spike2:npt.NDArray, 
                     stdp_scheme_param:dict=None,) -> npt.NDArray:
    """_summary_

    Args:
        nn (lifNetwork.LIF_Network): _description_

    Returns:
        _type_: _description_
    """
    if stdp_scheme_param is None: 
        stdp_scheme_param = {"scale_factor": 0.02,
                             "stdp_tau_plus": 10, 
                             "stdp_tau_neg": 40,
                             "stdp_tau_r": 4,
                             "stdp_beta": 1.4}

    # Convert to tensors
    network_w = tf.Variable(tf.cast(tf.convert_to_tensor(network_w), tf.float32))
    conn_matrix = tf.cast(tf.convert_to_tensor(conn_matrix), tf.float32)
    w_update_flag = tf.cast(tf.convert_to_tensor(w_update_flag), tf.float32)
    t_spike1 = tf.cast(tf.convert_to_tensor(t_spike1), tf.float32)
    t_spike2 = tf.cast(tf.convert_to_tensor(t_spike2), tf.float32)
    syn_delay = tf.cast(tf.convert_to_tensor(syn_delay), tf.float32)
    

    def graph_calculations(network_w, conn_matrix, w_update_flag, t_spike1, t_spike2, syn_delay):
        print(f"{'>'*10} graph_calculations TRACING {'>'*10}")
        # Reshape tensors
        w_update_flag = tf.reshape(w_update_flag, shape=(-1, 1))
        t_spike1 = tf.reshape(t_spike1, shape=(-1, 1))
        temporal_diff = t_spike2 + syn_delay - t_spike1

        # Update the network weight matrix
        updates = conn_matrix * w_update_flag * temporal_diff
        fn = lambda time_diff: stdp_scheme_logic(time_diff=time_diff,
                                                **stdp_scheme_param,)
        
        elems = tf.reshape(updates, shape=(-1, 1))
        dw = tf.map_fn(fn=fn, elems=elems)

        dw = tf.reshape(dw, shape=tf.shape(updates))
        network_w.assign_add(dw)

        return network_w

    network_w = graph_calculations(network_w, conn_matrix, w_update_flag, t_spike1, t_spike2, syn_delay)

    # Hard bound connection weights to [1, 0]
    # tf.cond(tf.math.greater(network_w, 1), lambda: 1)
    # tf.cond(tf.math.less(network_w, 0), lambda: 0)
    
    return network_w