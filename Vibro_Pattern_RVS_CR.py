################################################################################
## Generates the RVS CR stimulation schedule
##
## Written by: Anthony Lee 
## Question?: anthony.y.lee@ucdenver.edu
##
## Reference: https://www.frontiersin.org/articles/10.3389/fphys.2021.624317/full
################################################################################
import numpy as np

## NOTES:
# Vibration rest: 
# The vibration rest is the time between the end of one finger's vibration and 
# beginning of another finger's vibration. The vibration is currently set to 
# 100ms or 1/10 sec.


## Inputs
treatment_duration = 60  # [s]
freq_cr = 1.5            # [Hz] Frequency of coordinated-reset periods
n_sites = 5              # Number of vibration sites
n_on_periods = 3         # Number of on periods
n_off_periods = 2        # Number of off periods
burst_duration = 100     # [ms]
freq_burst = 250         # [Hz]


## Calculated variables
tau_cr = int(freq_cr**(-1)*1000)                           # [ms] Time length per CR period; Round-down to prior integer
treatment_duration_ms = int(treatment_duration * 1000)     # [ms]; Round-down to prior integer
burst_rest = (tau_cr - n_sites*burst_duration) // n_sites  # [ms] Floor-division


## Calculated variables -Number of on/off periods for entire treatment
tau_cr_approx = (burst_duration + burst_rest) * n_sites
tau_cycle = tau_cr_approx * (n_on_periods + n_off_periods)
n_cycles_in_treatment = treatment_duration_ms // tau_cycle  # [count] Floor-division


## Stimulation schedule generation
for cycle_idx in range(n_cycles_in_treatment):
    for on_idx in range(n_on_periods):
        for site_idx in np.random.randint(0, n_sites, n_sites):
            for millisecond in range(burst_duration):
                print(f"{' ' * site_idx} 1")  # Vibrate
            for millisecond in range(burst_rest):
                print(f"{' ' * site_idx} 0")  # Rest

    for off_idx in range(n_off_periods):
        for millisecond in range(tau_cr_approx):
            # print(f" {'0'*n_sites}")  # Off-period rest
            pass
