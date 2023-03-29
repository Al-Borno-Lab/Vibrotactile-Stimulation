## Manually import each module in
# from src.model import lifNetwork as lif
# lif_ = lif.LIF_Network(n_neurons=700)

## Testing the sub-package init module
import src.model as mod
lif_ = mod.LIF_Network(n_neurons=700)

print(lif_)

lif_.simulate(sim_duration=100)

lif_.spikeTrain()