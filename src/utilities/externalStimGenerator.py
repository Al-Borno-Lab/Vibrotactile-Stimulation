import numpy.typing as npt
import numpy as np
from tqdm import tqdm

# def rvs_cr_stimulation(dt:float=0.1, 
#                        freq_cr:float=17.5,
#                        A_stim:float = 1,
#                        amplitude_excitatory:float,
#                        amplitude_inhibitory:float, 
#                        v_th:float,
#                        v_th_reset: float) -> npt.NDArray:
#     """Generates the stimulation matrix according to the RVS CR protocol.

#     Args:
#         amplitude_excitatory (float): _description_
#         amplitude_inhibitory (float): _description_
#         v_th (float): _description_
#         v_th_reset (float): _description_
#         dt (float, optional): (ms) timestep for the Euler-scheme. Defaults to 0.1.
#         freq_cr (float, optional): Coordinated Reset stimulation frequency (Hz).
#           Defaults to 17.5.
#         A_stim (float, optional): Dimensionless Stimulation Strength. 
#           Defaults to 1.

#     Returns:
#         npt.NDArray: _description_
#     """

def manual_rvs_cr_stim_generation(n_neurons:int = 300, 
                                  freq_cr: float=17.5,
                                  stim_amplitude: float = 1,
                                  dt: float = 0.1,
                                  stim_duration: float = 100,
                                  n_sim_iter: int = 20000, 
                                  rvs_cr_start_sec: int = 500, 
                                  rvs_cr_duration_sec: int = 500):
  """This manually generates the external stimulation according to the RVS CR.

  Args:
      n_neurons (int, optional): _description_. Defaults to 300.
      freq_cr (float, optional): [Hz] Value that Ali used in paper. Defaults to 17.5.
      stim_amplitude (float, optional): _description_. Defaults to 1.
      dt (float, optional): _description_. Defaults to 0.1.
      stim_duration (float, optional): _description_. Defaults to 100.
      n_sim_iter (int, optional): _description_. Defaults to 20000.
      rvs_cr_start_sec (int, optional): _description_. Defaults to 500.
      rvs_cr_duration_sec (int, optional): _description_. Defaults to 500.
  """

  dt_decimals = str(dt)[::-1].find(".")
  euler_steps = int(stim_duration * n_sim_iter / dt)
  
  rvs_cr_start_ms = rvs_cr_start_sec * 1000  # [ms]
  rvs_cr_duration_ms = rvs_cr_duration_sec * 1000  # [ms]
  
  rvs_cr_end_sec = rvs_cr_start_sec + rvs_cr_duration_sec  # [seconds]
  rvs_cr_end_ms = rvs_cr_end_sec * 1000  # [ms]

  rvs_cr_start_idx = int(rvs_cr_start_ms / dt - 1)
  rvs_cr_end_idx = int(rvs_cr_end_ms / dt - 1)

  print((rvs_cr_end_idx - rvs_cr_start_idx))
  assert(rvs_cr_end_idx - rvs_cr_start_idx) == (rvs_cr_duration_ms / dt), "RVS CR euler_steps_count mismatch!"

  ## Placeholder variable
  stim_matrix = np.zeros((euler_steps, n_neurons))

  ## Checking if dt is enough resolution
  ## TODO (Tony): Not very accurate due to the rounding, fix the ISI for RVS CR stim matrix generation.
  tau_cr = (freq_cr/1000)**(-1)  # [ms] timescale of RVS CR 
  (samples_idx, isi_rvs_cr) = np.linspace(0, tau_cr, num=n_neurons, endpoint=False, retstep=True)  # [ms] interspike-interval during one period of the RVS CR
  isi_rvs_cr = round(isi_rvs_cr, dt_decimals)
  assert isi_rvs_cr >= dt, "dt is not small enough, we don't have enough resolution."


  for i, idx_euler_step in tqdm(enumerate(np.arange(rvs_cr_start_idx, rvs_cr_end_idx+1)), desc="Generating matrix..."):
    if i == 0:
      stim_order = np.random.choice(np.arange(n_neurons), size=n_neurons, replace=False)
    if len(stim_order) == 0:
      stim_order = np.random.choice(np.arange(n_neurons), size=n_neurons, replace=False)
    idx_neuron, stim_order = stim_order[0], stim_order[1:]

    stim_matrix[idx_euler_step, idx_neuron] = stim_amplitude

  # print(stim_matrix.shape)
  return(stim_matrix)


# if __name__ == "__main__": 
#   manual_rvs_cr_stim_generation()