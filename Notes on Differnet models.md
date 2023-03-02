## Coaleasing the notes: 
- `self.poisson_freq` is mis-leading because this is actually the `p` parameter in a Binomial distribution as opposed to the `lambda` parameter in a Poisson distribution. `lambda` is the rate, thus is the same as the frequency. However, `lambda = n * p` thus `p = lambda / n` and thus `p` and `lambda` needs to be clearly differentiated.
  - Renamed the variables, added docstring into the `stimulate_poisson` method to indicate the difference.
  - Migrated the conversion between `lambda` and `p` into the method with the ability to call the method using different `lambda` value. 
  - `lambda` is defaulted to 20Hz as set by the variables set in the original code inherited from Ali.
- `v_spike` is not seen in the paper, but set to 20 in Ali's code, thus we are also using it.
- The spiking phase utilizes a rectangular spike, which is more similar to voltage clamping or neuronal communications when the extracelluar resistance is high. However, if STN neurons are more like typical neuronal communications using NT more, then the spiking would be more like a current injection and thus have a exponential decay on the depolarization and repolarizing stages.
  - [ ] Check literature how STN neurons behave, more like voltage or current input?




## Notes:
- `1lifNetworkOFlogicMatteoUsed.py`
  - epoch current input matrix is a current input, it was incorrectly implemented in the dynamic conductance update function.
  - Subsequent model would have this fixed by migrating to the dynamic membrane potential function.
  - The code here has significant change in organization and variable names from the original class definition in the ipynb on Google Colab.
  - g_leak set to 10 instead of 0.02 as according to the paper
- `2lifNetworkFixedInputFromConductanceToVoltage.py`
  - This model has fixed the previous model's incorrect implementation of the epoch current input matrix by migrating the logic from the dynamic conductance update function to the dynamic membrane potnetial update function, aligning with equation (2) of Ali's paper.
  - All other variables remain the same from preivous model (i.e., `1lifNetworkOFlogicMatteoUsed.py`)
  - g_leak set to 10 instead of 0.02 as according to the paper
  - The dynamic voltage function is completely off from the paper.
  - **Abandoning iteratively fixing the model to show the difference.**
- `lifNetwork2.py`
  - Implements the "epoch_current_input" matrix as "external_spiked_input_w_sums".
  - Fixed various off variables such as g_leak set to 0.02 aligning with paper as opposed to 10 from Ali's code.
  - Unclear what and why tau_m is used: Jesse recommends Tony to look into STN firing frequency to validate which one to use for PD's STN.
    - To fix this, go into the `stimulate` method to comment out one of the value.


## lifNetwork2 issues:
- Dynamic voltage function: 
  - The dynamic voltage function is missing a variable, the g_leak variable.
  - Using the function with missing variable (OG), I get a plot that goes up briefly and then comes back down, so this is what I want to see. However, because the function is faulty with the missing g_leak, this is not a realiable result unless g_leak=1 and thus can be missed.
  - However, g_leak is set to 10 in the code and 0.02 in the paper, thus it isn't a value that can be ignored.
  - When I add the missing g_leak variable back into the dynamic function and set to Ali's original code value of 10, we get very slow calculations, but I have not been able to run long enough to see if the result is the desired "going up briefly and then come back down" thus needs testing. ([ ] Test to run this longer and see if desired result is produced.)

### Baseline (Ali's functions for all)
- Ali's function (missing g_leak)
  - g_leak = 10 >>> lifNetwork11.py
    - g_syn does NOT approach zero
    - Runs relatively fast
    - Dip at 75-th iteration
    - Peak at around 180-th iteration
  - g_leak = 0.02 >>> lifNetwork12.py
    - Seems to be running fine without the g_syn approaching zero
    - Seeing the decrease, increase, then decrease
    - Dip at 40-th iteration
    - Peak at 140-th iteration
- Ali's function (added g_leak)
  - g_leak = 10 >>> lifNetwork13.py
  - g_leak = 0.02 >>> lifNetwork14.py

### Dynamic g_noise
- Ali's function (missing g_leak)
  - g_leak = 10 >>> lifNetwork21.py
  - g_leak = 0.02 >>> lifNetwork22.py
- Ali's function (added g_leak)
  - g_leak = 10 >>> lifNetwork23.py
  - g_leak = 0.02 >>> lifNetwork24.py
- Tony's code
  - g_leak = 10 (Ali's code) >>> lifNetwork25.py
    <!-- - g_syn does NOT appraoch zero in an insane way, however fluctuates
    - Increase slightly and very briefly, then constantly decrease
    - peaks around 50 -->
  - g_leak = 0.02 (paper) >>> lifNetwork26.py
    <!-- - g_syn does NOT approach zero in an insane way, 10e-200
    - increases, then decreases
    - peak at around 55-th iteration -->

### Dynamic g_syn
- Ali's function (missing g_leak)
  - g_leak = 10 >>> lifNetwork31.py
  - g_leak = 0.02 >>> lifNetwork32.py
- Ali's function (added g_leak)
  - g_leak = 10 >>> lifNetwork33.py
  - g_leak = 0.02 >>> lifNetwork34.py
- Tony's code
  - g_leak = 10 (Ali's code) >>> lifNetwork35.py
  - g_leak = 0.02 (paper) >>> lifNetwork36.py

### Dynamic Voltage Function:
- With OG function (missing g_leak) 
  - g_leak = 10 (Ali's code) >>> lilfNetwork41.py
  - g_leak = 0.02 (paper) >>> lilfNetwork42.py
- With OG function (added missing g_leak)
  - g_leak = 10 (Ali's code) >>> lilfNetwork43.py
  - g_leak = 0.02 (paper) >>> lilfNetwork44.py
- With MY function
  - g_leak = 10 (Ali's code) >>> lilfNetwork45.py
  - g_leak = 0.02 (paper) >>> lilfNetwork46.py

### Dynamic voltage threshold function
- Ali's function (missing g_leak)
  - g_leak = 10 >>> lifNetwork51.py
  - g_leak = 0.02 >>> lifNetwork52.py
- Ali's function (added g_leak)
  - g_leak = 10 >>> lifNetwork53.py
  - g_leak = 0.02 >>> lifNetwork54.py
- Tony's code
  - g_leak = 10 (Ali's code) >>> lifNetwork55.py
  - g_leak = 0.02 (paper) >>> lifNetwork56.py
