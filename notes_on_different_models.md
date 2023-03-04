## CONCLUSION: 
- Ali's code is somewhat deviating from the step-wise change method from his paper, on top of that, Jesse updated the exponential decay (partially) to accomodate for if dt is set to values other than 0.1.
- Matteo's code is missing g_leak in the dynamic v function, and this may be the reason why he is seeing constant tendency of oversynchronization (in my todo to test this hypothesis).
- Some of the variable differences may be Ali's attempt to speed up the computation speed, however, it is not proven mathematically whether this scaling is equivalent to the paper's model (in todo to test).
- Model 11 is showing a general tendency that matches Ali's paper, contrary to Matteo's result, however, due to some code mistake, g_leak is set to 0.02 instead of 10. Perhaps fixing g_leak would reproduce Ali's results completely (in todo, high priority)
- Cluster works, and works well! :)



## TODO
- [ ] Check the units of capacitance from the paper - micro-Farad (I might have confused conductance vs conductivity and thus pressume equation 2 to be equating the charge on each side of the equation.) -- Continue the work on my paper notebook.
- [ ] Think of Jesse's decaying function in terms of approaching asymptote. If I were to implement the del-method, would that cause values to decrease into negative? 
- [ ] Check if g_poisson = 1.3 is collapsed from K_noise and tau_syn
- [ ] Fix (1) and (2)'s g_leak variable usage and resimulate.
- [ ] See (2) and test if kappa=400 that Ali used does in fact increase the exponential decay speed by a factor of 50x from the paper's kappa=8.
- [ ] Validate this hypothesis: On how Matteo's missing ov g_leak caused a tendency to oversynchronize.
- [ ] !!! Test model 11 to see if g_leak is fixed, we'd see what we want to see in Ali's paper.

## Variable deviations:
| <div style="width:300px">Paper</div> | <div style="width:300px">Ali's code (translated from Fortran)</div> |
| --- | --- |
| g_leak=0.02 | g_leak=10 |
| kappa=8 | kappa=400 |
| v_initial value ~ unif(-38, -40) | v_initial_value ~ unif(-55, -35) |
| v_spike = ??? (value not mentioned) | v_spike = 20 |
| C_i ~ N(3, 0.05*3) micro-Farad | tau_m, which is a normal distribution of mean = 150 |
| K_noise = 0.026 mS/cm^2 | g_poisson = 1.3 (instead of using K_noise and tau_syn to calculate g_noise updates, Ali may have collapsed the value because K_noise and g_leak are very close thus didn't declare the variable and just collapsed the value into g_poisson = 1.3)

### Speculated reasonings:
- Ali may have increased kappa from 8 to 400 to increase the g_syn update speed thus saving computation time.
- Ali may have decrease capacitance from 300 (my speculated value, need verification) to 150 to increase the membrane potential update speed.

## ModelTests in Alderaan (Cluster): modelTestingCodes
Here I ran the model (lifNetwork2) using different model deviations and parameters.
Specifically, we tested different dynamic functions (i.e. Ali's, Matteo's, and Tony's). 
The first parameter that we varied is the variable kappa (i.e., max coupling strength in mS/cm^2) where Ali's code used kappa = 400, however, the paper used kappa = 8.
The other parameter that we varied here is using tau_m (from Ali's code) vs the value from the paper 0.026 for the capacitance variable value from equation (4) of the paper. Ali's code used tau_m which is approximately 150, and the paper 3 mF/cm^2.
Another parameter: g_leak (Ali's code uses 10, whereas the paper uses 0.026.)

The various models are testing by adjusting the dictionary value of each of the model testing entry script. The value is then taken into the model (i.e., lifNetwork2) that will handle the switch case to change up the implementation during `simulate()` calls.

### 1_modelTest_ali_kappa8
- Dynamic functions g_noise, g_syn, v, v_th all uses Ali's code's implementation (translated from Fortran by Jesse).
- Conductivity: g_leak is added and uses the value 0.02 >>> INCORRECT, NEED TO REDO BECAUSE ALI SET THIS TO 10.
- Capacitance: tau_m is used (as per Ali's code) instead of the capacitance r.v. from the paper. (NEED TO VERIFY - SEE TODO on conductance vs conductivity)
- The plot outcome is a linear line, not what we expected. Perhaps kappa=8 is too small, thus the neuronal learning (weight change) is too slow, thus it looks like a linear line. Let's see how this contrast with kappa=400. 
- Additional reason may be g_leak was accidentally set to 0.02 instead of the value 10 which was what Ali had in his code. The impact on this (based on my intuition) is that dynamic membrane potential will be VERY low, hence not much firing, thus we see a desynchronization happening, and because kappa is 8, the synaptic conductance update is also very very slow.
![PlotExport](ignored_dir/cluster_runs/modelTest_ali_kappa8/export_2023-03-02-16-35-51/iteration_9990.png)

### 2_modelTest_ali_kappa400
- Similar to above, all Ali's code implementation (translated by Jesse from Fortran)
- Conductivity: g_leak is set to 0.02 (per the value of the paper) BY MISTAKE BECAUSE ALI'S CODE USED 10. JUST LIKE ABOVE, NEED FIXING.
- Capacitance: tau_m is used isntead of the capacitance r.v. from the paper. (NEED TO VERIFY - SEE TODO on conductance vs conductivity)
- The plot outcome is a exponential decay decrease and reaches an asymptotic value.
- Comparing to (1), this have a decrease because like (1), g_leak was accidentally set to 0.02 by Tony, hence the voltage update small hence keeps getting desychornized. However, contrasting to (1), we do see an exponential decay because kappa=400 allows g_syn to update faster.
- ((It seems like Ali set kappa=400 from the paper's kappa=8 to increase the change speed by a factor of almost 50x))
![PlotExport](ignored_dir/cluster_runs/modelTest_ali_kappa400/export_2023-03-02-16-36-36/iteration_9990.png)

### 3_modelTest_ali_paperCapacitance_kappa8
- DISREGARD THIS: The result is wrong because I accidentally set g_leak = 10 for the paper method, which should be set to 0.026 (very close to Ali's) as stated in the paper (Error in lifNetwork2 under the `update_v()` method definition).
- Similar to above, just used the paper's capacitance calculation, which needs verifying (TODO about conductance vs conductivity).
- Not going to interpret this plot because the g_leak conductivity is set to incorrect value.
- TAKES FOREVER: The potential reason for this plot to take forever might be due to my incorrect conversion of micro-Farad the capacitance r.v. that Ali used tau_m instead. Perhaps proper conversion would yield the varaible to be 300, which is only twice that of Ali's value, and Ali may have decreased the membrane capacitance from 300 to 150 because create less resistance to membrane potential change during injection of current and thus allows him to simulate faster.
- [ ] TODO (Tony): Fix g_leak and confirm capacitance unit (micro-Farad) and run this simulation.

### 4_modelTest_ali_paperCapacitance_kappa400
- DISREGARD THIS: The result is wrong because I accidentally set g_leak = 10 for the paper method, which should be set to 0.026 (very close to Ali's) as stated in the paper (Error in lifNetwork2 under the `update_v()` method definition).
- Similar to above, just used the paper's capacitance calculation, which needs verifying (TODO about conductance vs conductivity).
- Not going to interpret this plot because the g_leak conductivity is set to incorrect value.
- TAKES FOREVER: The potential reason for this plot to take forever might be due to my incorrect conversion of micro-Farad the capacitance r.v. that Ali used tau_m instead. Perhaps proper conversion would yield the varaible to be 300, which is only twice that of Ali's value, and Ali may have decreased the membrane capacitance from 300 to 150 because create less resistance to membrane potential change during injection of current and thus allows him to simulate faster.
- [ ] TODO (Tony): Fix g_leak and confirm capacitance unit (micro-Farad) and run this simulation.

### 5_modelTest_tony_aliCapacitance_kappa8
- Uses lifNetwork3, which moved the g_syn and g_noise calculation after the weight update. Additionally, uses the del-method to calculate the dynamic function's Euler-step changes.
- capacitance: tau_m
- g_leak: 0.02 (MISTAKE, should be Ali's method of 10, but I messed up).
- Plot is a linear line, and I speculate it is because kappa=8, which makes g_syn update very very slowly. In addition, I messed up and set g_leak to 0.02 instead of 10, hence voltage update is very slow as we can see the largest mean_weight value vs the smallest is 0.3990-0.3978.
- Fix: If changing g_leak to 10 (Ali's value) and kappa=400, we should see more significant change. Thus, let's see the next plot where kappa=400 and see if there is more signficiant g_syn change. 
![PlotExport](ignored_dir/cluster_runs/5_modelTest_tony_aliCapacitance_kappa8/export_2023-03-02-21-44-13/iteration_9990.png)

### 6_modelTest_tony_aliCapacitance_kappa400
- Uses lifNetwork3, which moved the g_syn and g_noise calculation after the weight update. Additionally, uses the del-method to calculate the dynamic function's Euler-step changes.
- capacitance: tau_m
- g_leak: 0.02 (MISTAKE, should be Ali's method of 10, but I messed up).
- Compared to (5), we can see the exponential decay a little bit here and it is due to the change of kappa=8 to kappa=400, thus, this is seemingly affirming my hypothesis more and more that Ali changed kappa from 8 to 400 to speed things up.
![PlotExport](ignored_dir/cluster_runs/6_modelTest_tony_aliCapacitance_kappa400/export_2023-03-02-21-43-33/iteration_9990.png)


### 7_modelTest_tony_paperCapacitance_kappa8
- capacitance: paper's value (NEED VERIFICATION, in todo already)
- g_leak = 10  (MISTAKE, meant to be 0.02, the value from the paper)
  - Even though large g_leak (compared to 0.02) should leads to large changes in membrane potential, but with zero conductivity (kappa being so small and g_syn starting value of 0), nothing is happening.
- kappa = 8
  - Small kappa leads to very slow g_syn value change, and g_syn has a initial value of 0, thus kind of stays at zero.
- The plotting here takes a VERY LONG TIME.
  - Not sure why this takes so long: 
    - Potentiall because kappa and g_leak's value, but other models did not have this issue.
    - Potentially because of the way I implemented the dynamic functions?
      - Answer: Yeah, the plot below is just as slow, perhaps my way (paper's method) of implementing the calculation is a lot more computationally intensive.
- [x] See if when kappa=400 is just as slow.

![PlotExport](ignored_dir/cluster_runs/7_modelTest_tony_paperCapacitance_kappa8/export_2023-03-02-21-44-27/iteration_70.png)

### 8_modelTest_tony_paperCapacitance_kappa400
- capacitance: paper's value (NEED VERIFICATION, in todo already)
- g_leak = 10  (MISTAKE, meant to be 0.02, the value from the paper)
- kappa = 400
- The plotting here takes a VERY LONG TIME.
- Even with large kappa, with the initial value of g_syn being zero and very slow g_syn change, we don't see anything.

![PlotExport](ignored_dir/cluster_runs/8_modelTest_tony_paperCapacitance_kappa400/export_2023-03-02-21-43-52/iteration_80.png)

### 9_modelTest_matteo_v_ali_rest_kappa8
- capacitance = tau_m
- g_leak: 0.02 (MISTAKE, just like all the other ones)
  - Doesn't matter for this model because Matteo's dynamic v function missed g_leak
- kappa = 8
- Matteo's v update is missing the value g_leak, and the exponential decay is not updated, thus the two may combined and cause what we are seeing here.
- I've updated to have the exponential decay and added the missing g_leak in the dynamic membrane potential function.
- **Learning lesson: When code is merged manually and rewrote over different notebooks, this can happen, and this will be mitigated when we are pulling the code base from a central repo (github).**

![PlotExport](ignored_dir/cluster_runs/9_modelTest_matteo_v_ali_rest_kappa8/export_2023-03-02-18-45-47/iteration_5500.png)

### 10_modelTest_matteo_v_ali_rest_kappa400
- capacitance: tau_m
- g_leak: 0.02 (MISTAKE, just like any other ones)
- kappa = 400
- The g_leak mistake doesn't really matter here because Matte's dynamic v function missed it.
- Plot: Even though g_leak is missing, we are still seeing an exponential decay, and I speculate that this is due to the exponential decay of g_syn as we see it in (9) and (10) this one.
- Impact of missing g_leak: I'm curious what is the bias caused by the missing of g_leak as g_leak=0.02 scales down a section of the dynamic v function and g_leak=10 scales up a section of the function.
- Ali's g_leak = 10 would scale up the del_v (think of it as decaying faster) and thus he can see a bistablization. However, with Matteo's missing of g_leak, this may cause the del_v to be smaller and thus leading to the network to oversynchronize.
  - [ ] Validate this hypothesis: On how Matteo's missing ov g_leak caused a tendency to oversynchronize.

![PlotExport](ignored_dir/cluster_runs/10_modelTest_matteo_v_ali_rest_kappa400/export_2023-03-02-19-29-22/iteration_6660.png)

### 11_modelTest_ali_various_initial_weight_kappa400
- capacitance: tau_m
- g_leak 0.02 (MISTAKE!!! See other models' notes)
- kappa: 400
- Interpretation: 
  - g_leak being small is causing membrane potential update to be small, and hence, we may be seeing a tendency of desynchronization. I speculate that if it is fixed to Ali's value of 10 and thus the entire model matches his, we'd see his results.
  - kappa=400, allows for g_syn to update faster, thus speeding up the computation speed.
- [ ] Test model 11 to see if g_leak is fixed, we'd see what we want to see in Ali's paper.

![PlotExport](ignored_dir/cluster_runs/11_modelTest_ali_various_initial_weight_kappa400/export_2023-03-02-23-14-08/iteration_9990.png)











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
