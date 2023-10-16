#!/bin/bash
# A simple single core job template
#SBATCH --job-name=22_BasedOn16_All_ali_but_thr_method_Tony
#SBATCH --partition=math-alderaan
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=slurm_%u_%x_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.y.lee@ucdenver.edu

# Activate the proper conda env
source /home/leean9/.bashrc
conda activate nest

# Call model testing script
export PYTHONPATH=$PYTHONPATH:/data001/projects/leean9/Vibrotactile-Stimulation/AYL_Vibro_22/  # So Python knows where to find src
python3 -m src.modelTestingScripts.22_BasedOn16_All_ali_but_thr_method_Tony
