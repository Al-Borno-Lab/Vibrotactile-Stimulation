import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from tqdm import tqdm

def generate_poisson_spike_train(lam:int,
                                 n_neurons:int, duration:float,
                                 timestep:float=0.1,
                                 neuron_timescale:float=1) -> npt.NDArray:
    """Generate and return a Poisson Spike Train.

    The Poisson Spike Train assumes the independent spike hypothesis. This 
    generator takes into consideration of the absolute refractory period of a
    neuron and thus prohibits a spike generation during the period.

    Args:
        lam (int): [Hz] Rate or the lambda parameter in a Poisson distribution.
        n_neurons (int): Number of neurons
        duration (float): [ms] The duration or length of the spike train.
        timestep (float, optional): [ms] The timestep used during the  
            Euler-scheme simulation. Defaults to 0.1 [ms].
        neuron_timescale (float, optional): [ms] The time constant of the 
            neuron, also known as the absolute refractory period. Defaults to 1.

    Returns:
        npt.NDArray: euler_steps by n_neurons numpy.ndarray, with 1 denoting a 
            spike and 0 denoting non-spike.
    """
    euler_steps = int(duration / timestep)
    abs_ref_steps = int(neuron_timescale / timestep)
    steps_in_abs_ref = np.zeros((n_neurons,), dtype=int)
    spike_train = np.zeros((euler_steps, n_neurons), dtype=int)

    binomial_n = 1000 / timestep
    binomial_p = lam / binomial_n

    for step in tqdm(np.arange(euler_steps), desc="Generating spike train..."):
        
        # Check if in absolute refractory period of the neuron
        in_abs_ref = np.where(steps_in_abs_ref>0, True, False)
        not_in_abs_ref = np.logical_not(in_abs_ref)

        # Binomial generator (simulating Poisson Spike Train)
        step_spike_trigger = np.random.binomial(n=1, p=binomial_p, size=(n_neurons,))

        # Spike vs No-spike at current Euler-step
        step_spike = np.multiply(step_spike_trigger, not_in_abs_ref, dtype=int)
        spike_train[step, ] = step_spike  # Append to output spike-train placeholder

        # In absolute refractory counter update
        steps_in_abs_ref += np.multiply(step_spike, abs_ref_steps, dtype=int)
        steps_in_abs_ref -= in_abs_ref  # Decrement those that are in abs ref period

        # # DEBUG USE, IGNORE
        # if (np.where(steps_in_abs_ref<0, 1, 0).any()):
        #     raise Exception(f"Negative value detected, "\
        #                     f"something is wrong at euler-step: {step}")
    
    return spike_train