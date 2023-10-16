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
conda activate vibro

# Pull latest codebase
git -C /data001/projects/leean9/Vibrotactile-Stimulation pull 
# git -C ~/Vibrotactile-Stimulation checkout model_testing
echo "**git pull completed; project directory up-to-date.**"

# Call model testing script
export PYTHONPATH=$PYTHONPATH:/data001/projects/leean9/Vibrotactile-Stimulation  # So Python knows where to find src
python3 -m src.modelTestingScripts.22_BasedOn16_All_ali_but_thr_method_Tony
