from src.model import lifNetwork as lif

lif_ = lif.LIF_Network(n_neurons=300)

choose_model_setup = {"update_g_noise_method": "Ali",
                      "update_g_syn_method": "Ali",
                      "update_v_method": "Ali",
                      "update_v_capacitance_method": "Ali",
                      "update_thr_method": "Ali"
                      }

lif_.simulate(temp_param=choose_model_setup)
