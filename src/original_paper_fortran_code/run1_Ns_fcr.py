import os,sys
from time import sleep
import numpy as np 

program=sys.argv[1]
Fortran_program='/'+str(program)

program2=sys.argv[2]
Fortran_program2='/'+str(program2)

jobIndex=0  # this will increase at every loop iteration to and generate many job names
cwd = os.getcwd() # this gets the path for the current directory
per_taks_job=1
#A= np.round( np.logspace(np.log10(0.05),0,40) , 4)

A=np.asarray([1])
fCR=np.linspace(1,24,24).astype(int)
Ms=np.linspace(2,50,25).astype(int)
nets=np.array([1,2,3])
#Ms=np.linspace(1,25,25).astype(int)
k=-1
lxpr=len(Ms)*len(nets)
xpr=np.zeros([5,lxpr])
for M1 in (Ms):
    for n1 in (nets):
        k=k+1
        xpr[0,k]=1 # fcr
        xpr[1,k]=1 # Astim
        xpr[2,k]=M1 #M_electrode
        xpr[3,k]=1000 # stim duration
        xpr[4,k]=n1 # net_index
    
lb1=930


for i in range (0,lxpr):
    print(i)
    jobIndex+=1
    Astim1=xpr[1,i]
    Mel=xpr[2,i]
    stim_dur=xpr[3,i]
    nindx=xpr[4,i]
    jobName='CCR_'+str(jobIndex)
    with open(jobName+'.sh', 'w') as file:  # write whatever you want to the file
        file.write(str('#!/bin/bash'+'\n'))
        file.write(str('#'+'\n'))
        file.write( str('#SBATCH --job-name='+str( jobName )+'\n' ) )
        file.write( str('#SBATCH --mail-type=FAIL'+'\n' ) )
        file.write( str('#SBATCH --mail-user=khaledi@stanford.edu'+'\n' ) )
        file.write(str('#SBATCH --time=48:00:00'+'\n'))
        file.write(str('#SBATCH --ntasks=1'+'\n'))
        file.write(str('#SBATCH --cpus-per-task=1'+'\n'))
        file.write(str('#SBATCH --mem-per-cpu=4G'+'\n'))
        file.write(str('#SBATCH --array=1'+'\n'))
              
#        if jobIndex <= lb1 :
#            file.write(str('#SBATCH -p normal'+'\n'))
#        if jobIndex > lb1  :
#            file.write(str('#SBATCH -p ptass'+'\n'))
        #file.write(str('#SBATCH -p ptass'+'\n'))
        file.write(str('#SBATCH -p normal'+'\n'))
        file.write(str('#'+'\n'))
        file.write( str('PER_TASK='+str( per_taks_job )+'\n' ) )
        file.write( str('START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))'+'\n' ) )
        file.write( str('END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))'+'\n' ) )
        
        file.write('srun ifort -o '+jobName+'.o  ' +  '-fast -c '+str(str(cwd)+Fortran_program+' \n'))
        file.write('srun ifort -o ' +jobName+'.x  '+str(str(cwd)+'/'+jobName+'.o  '+' \n'))
        file.write( str('for (( ff1=1; ff1<=24; ff1=ff1+1 )); do'+'\n' ) )
        file.write('    srun '+str(str(cwd)+'/'+jobName+'.x'+' '+'$ff1'+' '+str(Astim1)+'  '+str(Mel)+'  '+str(stim_dur) +'  '+str(nindx)+'\n'))
        file.write( str('done'+'\n' ) )
        
    print(jobName,Mel,nindx)
    os.system("sbatch "+jobName+'.sh') # this will submit the job
    
    
jobIndex=0 
for i in range (0,lxpr):
    print(i)
    jobIndex+=1
    Astim1=xpr[1,i]
    Mel=xpr[2,i]
    stim_dur=xpr[3,i]
    nindx=xpr[4,i]
    jobName='SCCR_'+str(jobIndex)
    with open(jobName+'.sh', 'w') as file:  # write whatever you want to the file
        file.write(str('#!/bin/bash'+'\n'))
        file.write(str('#'+'\n'))
        file.write( str('#SBATCH --job-name='+str( jobName )+'\n' ) )
        file.write( str('#SBATCH --mail-type=FAIL'+'\n' ) )
        file.write( str('#SBATCH --mail-user=khaledi@stanford.edu'+'\n' ) )
        file.write(str('#SBATCH --time=48:00:00'+'\n'))
        file.write(str('#SBATCH --ntasks=1'+'\n'))
        file.write(str('#SBATCH --cpus-per-task=1'+'\n'))
        file.write(str('#SBATCH --mem-per-cpu=4G'+'\n'))
        file.write(str('#SBATCH --array=1'+'\n'))
              
#        if jobIndex <= lb1 :
#            file.write(str('#SBATCH -p normal'+'\n'))
#        if jobIndex > lb1  :
#            file.write(str('#SBATCH -p ptass'+'\n'))
        #file.write(str('#SBATCH -p ptass'+'\n'))
        file.write(str('#SBATCH -p normal'+'\n'))
        file.write(str('#'+'\n'))
        file.write( str('PER_TASK='+str( per_taks_job )+'\n' ) )
        file.write( str('START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))'+'\n' ) )
        file.write( str('END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))'+'\n' ) )
        
        file.write('srun ifort -o '+jobName+'.o  ' +  '-fast -c '+str(str(cwd)+Fortran_program2+' \n'))
        file.write('srun ifort -o ' +jobName+'.x  '+str(str(cwd)+'/'+jobName+'.o  '+' \n'))
        file.write( str('for (( ff1=1; ff1<=24; ff1=ff1+1 )); do'+'\n' ) )
        file.write('    srun '+str(str(cwd)+'/'+jobName+'.x'+' '+'$ff1'+' '+str(Astim1)+'  '+str(Mel)+'  '+str(stim_dur) +'  '+str(nindx)+'\n'))
        file.write( str('done'+'\n' ) )
        
    print(jobName,Mel,nindx)
    os.system("sbatch "+jobName+'.sh') # this will submit the job


