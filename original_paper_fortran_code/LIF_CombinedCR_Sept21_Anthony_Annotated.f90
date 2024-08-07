! 2023-07-26 - Commented by Anthony, original code written by Ali Khaledi-Nasab

!!! Discrepencies between code and Paper !!!
! - D=1.3d0 >>> This param looks like kappa_noise, but is 50x the value stated 
!   in the paper (assuming kappa_noise).
!   - 2023-09-09 (Anthony) - This value matches that of kappa_noise, and the 50x
!     scale aligns with how the paper scales the membrane capacitance value by 50x
!     from what was stated in the paper.
! - tau=150.d0 >>> This is the mean of the membrane capacitance random variable
!   of which the paper used 3 instead; thus the code is 50x the paper's value.
!   Here the code is implemented as Box Mueller Method.
! - sigtau=7.5 >>> This is the stdev of the membrane capacitance random variable
!   of which the paper used 0.15 instead; thus the code is 50x the paper's value.
!   Here the code is implemented as Box Mueller Method.


!! Setting the parameters
implicit none
!! Some model parameters
real*8,parameter:: ti=1.d-1        ! [ms] Time per Euler scheme step
real*8,parameter:: tu=1000.d0      ! [ms] Network warmup/relax duration
real*8,parameter:: sparcity=0.07   ! Unused > seems like connection probability
real*8,parameter:: sigw=0.1d0      ! ??? Unused / Unknown
integer,parameter:: maxspk=500     ! ??? Unknown
integer,parameter:: seed_index=1   ! ??? Unused / Unknown

!! Neuron physiology parameters
real*8,parameter:: D=1.3d0         ! ??? [mS/cm^2] See notes above
real*8,parameter:: vth_rest=-40.d0 ! [mV] Spiking threshold when at rest
real*8,parameter:: v_rest=-38.d0   ! [mV] Resting potential
real*8,parameter:: v_syn=0.d0      ! [mV] Reversal potential for noise current (eq. 5)
real*8,parameter:: tau_th=5.d0     ! [ms] tau for threshold dynamic func (eq. 3)
real*8,parameter:: tau_syn=1.d0    ! [ms] Synaptic time constant (eq. 3, 4, 6)
real*8,parameter:: t_d=3.d0        ! [ms] Synaptic transmission delay (eq. 4)
real*8,parameter:: tau=150.d0      ! ??? [micro-Farad/cm^2] See notes above
real*8,parameter:: sigtau=7.5      ! ??? [micro-Farad/cm^2] See notes above

!! STDP Dynamics
real*8,parameter:: beta=1.4d0      ! Depression to potentitation ratio (eq. 7)
real*8,parameter:: mean_Weight=0.5d0
real*8,parameter:: tau_R=4.d0      ! LTD-to-LTP time-constant ratio (Depression time scale)
real*8,parameter:: taup=10.d0      ! [ms] STDP decay time for long-term potentiation (tau_positive)

!! Poisson Noise input
real*8,parameter:: fnoise=20.d0    ! [Hz] Poisson noisy input

!! Rectangular spike parameters
real*8,parameter:: vth_spike=0.d0  ! [mV] Spike threshold right after rect spike
real*8,parameter:: v_reset=-67.d0  ! [mV] Pontential right after rect spike
real*8,parameter:: t_spike=1.d0    ! [ms] Rect spike duration
real*8,parameter:: v_spike=20.d0   ! [mV] Rect spike potential
real*8,parameter:: Delta_V=vth_spike-v_reset

!! Data read from terminal input
real*8:: f_CR          ! [Hz] Coordinate Reset frequency
real*8:: A_stim        ! [unitless] Dimensionless stimulation strength
integer:: M_electrode  ! [count] Numbers of stimulation sites
integer:: stim_dur     ! [s] Stimulation duration in seconds
integer:: net_index    ! ??? Network drive index? - Looks to be used to identify file writing

!! Read in from model_input.in
real*8:: lamda  ! Scales the weight update per spike, eta in the paper (default = 2.d-2)
real*8:: tmax   ! [s] Total simulation time (default = 2500.d0)
real*8:: tchun  ! [s] Save data and calculate the Kuramoto order parameter every tchunk seconds (default = 10.d0)
real*8:: capa   ! [mS/cm^2] Max coupling strength (default = 400.d0)
integer:: Nr    ! Total number of neurons (defaults = {200, 500, 1000, 2000})

integer:: Npre
integer:: Npost
integer:: ntmax
integer:: ntu  ! [count] Number of Euler scheme steps to relax / warm-up the network
integer:: jtime
integer:: i
integer:: j
integer:: k
integer:: i1
integer:: j1
integer:: i2
integer:: j2
integer:: k1
integer:: i3
integer:: postsyn
integer:: start_or_continue
integer:: seeds(10)
integer:: posti
integer:: window
integer:: maxN_window
integer:: nr_index
integer:: syn_tchunk
integer:: itime_save
integer:: post,prei
integer:: pre,time_i
integer:: time_j
integer:: itime
integer:: stdp_max
integer:: stdp_index
integer:: index1
integer:: index2
integer:: index3
integer:: pre1
integer:: pre2
CHARACTER(len=255):: cwd
CHARACTER(len=255):: outputpath
CHARACTER(len=255):: output_dir
CHARACTER(len=255):: out_file(10)
CHARACTER(len=255):: net_path
CHARACTER(len=255):: terminal_inputs(6)
CHARACTER(len=255):: net_file(11)
CHARACTER(len=255):: stater_path

real*8:: taun  ! [ms] STDP decay time for long-term depression (tau_negative)
real*8:: fr
real*8:: pi2
real*8:: t2
real*8:: t
real*8:: dt2
real*8:: odpar
real*8:: dti
real*8:: total_syn
real*8:: delta_time(2)
real*8:: sum1
real*8:: beta_tau_R_lamndba  ! Variable to hold the LTD weight scaling scalar
real*8:: sw_outCR
real*8:: Astim1
real*8:: Astim2
real*8:: SD_CR
integer:: CR_win
integer:: ub
integer:: lb
integer:: CR_win2
integer:: ist

!! Allocatable arrays
real*8,allocatable, dimension(:):: v0
real*8,allocatable, dimension(:):: v
real*8,allocatable, dimension(:):: vth
real*8,allocatable, dimension(:):: vth0
real*8,allocatable, dimension(:):: un1
real*8,allocatable, dimension(:):: un2
real*8,allocatable, dimension(:):: taum ! [micro-Farad/cm^2] Membrane capacitance
real*8,allocatable, dimension(:):: gs0  ! [mS/cm^2] Synaptic conductance placeholder (for calculation)
real*8,allocatable, dimension(:):: gs   ! [mS/cm^2] Synaptic conductance - Eq. 4
real*8,allocatable, dimension(:):: dgs  ! [ms/cm^2] Synaptic conductance delta
real*8,allocatable, dimension(:):: gn0  ! [mS/cm^2] Noise conductance placeholder (for calculation)
real*8,allocatable, dimension(:):: gn   ! [mS/cm^2] Noise conductance - Eq. 6
real*8,allocatable, dimension(:):: sw_save
real*8,allocatable, dimension(:):: unpoi
real*8,allocatable, dimension(:):: odparw
real*8,allocatable, dimension(:):: stdp_poten
real*8,allocatable, dimension(:):: stdp_dep
real*8,allocatable, dimension(:):: cpl
real*8,allocatable, dimension(:):: x1D
real*8,allocatable, dimension(:):: sw_inCR
real*8,allocatable, dimension(:,:):: tspk
real*8,allocatable, dimension(:,:):: sw  ! [scale] Scales the coupling strength, which is analogous to conductance - It is also clipped between [0, 1].
real*8,allocatable, dimension(:,:):: sw0
real*8,allocatable, dimension(:,:):: dw
real*8,allocatable, dimension(:,:):: tsh
real*8,allocatable, dimension(:,:):: tshw
real*8,allocatable, dimension(:,:):: un3
real*8,allocatable, dimension(:,:):: un4
real*8,allocatable, dimension(:,:):: hat_post
real*8,allocatable, dimension(:,:):: delays
real*8,allocatable, dimension(:,:):: dis_post
real*8,allocatable, dimension(:,:):: I_stim
real*8,allocatable, dimension(:,:):: CR_stim
real*8,allocatable, dimension(:,:):: CR_means
integer,allocatable, dimension(:):: flag
integer,allocatable, dimension(:):: itrch
integer,allocatable, dimension(:):: itrchw
integer,allocatable, dimension(:):: presyn
integer,allocatable, dimension(:):: poi1
integer,allocatable, dimension(:):: x1Dc
integer,allocatable, dimension(:):: CR_groups_count
integer,allocatable, dimension(:):: CR_groups
integer,allocatable, dimension(:):: CR_n
integer,allocatable,dimension(:,:):: prelink1
integer,allocatable,dimension(:,:):: prelink2
integer,allocatable,dimension(:,:):: postlink,adj
integer,allocatable,dimension(:,:):: window_nodes
integer,allocatable,dimension(:,:):: widnow_CR_nodes
integer,allocatable,dimension(:,:):: sw_CR1
integer,allocatable, dimension(:):: Syn_CR
integer,allocatable, dimension(:):: CR_neuron,coor
integer,allocatable, dimension(:,:,:):: sw_CR


!! Getting terminal inputs - f_CR, A_stim, M_electrode, stim_dur, net_index
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CALL GET_COMMAND_ARGUMENT(1,terminal_inputs(1))
READ(terminal_inputs(1),*)f_CR

CALL GET_COMMAND_ARGUMENT(2,terminal_inputs(2))
READ(terminal_inputs(2),*)A_stim

CALL GET_COMMAND_ARGUMENT(3,terminal_inputs(3))
READ(terminal_inputs(3),*)M_electrode

CALL GET_COMMAND_ARGUMENT(4,terminal_inputs(4))
READ(terminal_inputs(4),*)stim_dur

CALL GET_COMMAND_ARGUMENT(5,terminal_inputs(5))
READ(terminal_inputs(5),*)net_index

!CALL GET_COMMAND_ARGUMENT(3,terminal_inputs(3))
!READ(terminal_inputs(3),*)SD_CR

call getcwd(cwd)  ! Get current working directory


!!!!!!!!!! Read parameters from intput file !!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
open(11,file=trim(cwd)//"/model_input.in")  ! Read the file model_input.in
read(11,*)lamda  ! [unitless] Scales the weight update per spike - variable eta in the paper
read(11,*)tmax   ! [s] Total simulation time
read(11,*)tchunk ! [s] Save data and Calculate the Kuramoto order parameter every tchunk seconds
read(11,*)capa   ! [mS/cm^2] Max coupling strength
read(11,*)Nr     ! [count] Total number of neurons
close(11)

!!!!!!!!!! Constants !!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Variables for output storage
net_path=trim(cwd)//trim('/')//trim('net_reg')//trim('/') ! Path to the network directory
call create_directory( 'CCR' )                            ! mkdir CCR
outputpath=trim(cwd)//trim('/')//trim('CCR')//trim('/')   ! Path to the output folder
!! Other calculated constants
ntmax=int(tchunk*1000/ti)+1  ! [count] Number of Euler scheme steps
ntu=int(tu/ti)               ! [count] Number of Euler scheme steps to relax / warm-up the network
taun=tau_R*taup              ! [ms] STDP deay times for LTD (tau_negative)
fr=fnoise*ti*1.d-3           ! [count/timestep] Poisson spike train noisy input, converted from Hz
jtime=int(tmax/tchunk)       ! [count] Total number of times to save data and calculate Kuramoto order
pi2=8*atan(1.d0)             ! ???
stdp_max=int(1000.d0/ti)     ! [count] ???
dti=1.d0-ti                  ! [ms] ???
beta_tau_R_lamndba=0.d0      ! Init
beta_tau_R_lamndba=(beta/tau_R)*lamda  ! Scalar for LTD weight update
CR_win=0  ! ??? 
CR_win2=0 ! ???

!!!!!!!!!! Create all the names you need to use !!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
write (net_file(1),"(i0,'_N_',i0,'_reg_network_dimension.net')") net_index,Nr
write (net_file(2),"(i0,'_N_',i0,'_reg_postlink.net')") net_index,Nr
write (net_file(3),"(i0,'_N_',i0,'_reg_presyn.net')") net_index,Nr
write (net_file(4),"(i0,'_N_',i0,'_reg_prelink1.net')") net_index,Nr
write (net_file(5),"(i0,'_N_',i0,'_reg_prelink2.net')") net_index,Nr
write (net_file(6),"(i0,'_N_',i0,'_reg_hat_post.net')") net_index,Nr
write (net_file(7),"(i0,'_N_',i0,'_reg_dis_post.net')") net_index,Nr
write (net_file(8),"(i0,'_N_',i0,'_reg_adj.net')") net_index,Nr
write (net_file(9),"(i0,'_N_',i0,'_reg_x_per_windowc.net')") net_index,Nr
write (net_file(10),"(i0,'_N_',i0,'_reg_N_window.net')") net_index,Nr
write (net_file(11),"(i0,'_N_',i0,'_reg_positions.net')") net_index,Nr


write (out_file(1),"('1_odpar_fCR_',i0,'_M_',i0,'_Astim_',i0,'_net_',i0,'_dur',i0,'.dat')") int(f_CR*10),M_electrode,int(A_stim*1000),net_index,stim_dur
!write (out_file(2),"('2_spk_fCR_',i0,'_M_',i0,'_sigma_CR_',i0,'_Astim_',i0,'_net_',i0,'_dur',i0,'.dat')") int(f_CR*10),M_electrode,int(SD_CR*1000),int(A_stim*1000),net_index,stim_dur

       
open(unit=111,status='unknown',file=trim(outputpath)//trim(out_file(1)))  !! Output file 111 for things like mean synaptic weight
!open(unit=224,status='unknown',file=trim(outputpath)//trim(out_file(2)))

!!!!!!!!!! Open and read network_dimension data to variables !!!!!!!!!!
open(unit=1, status='old',file=trim(net_path)//trim(net_file(1)))
read(1,*)Nr,Npre,Npost,window,maxN_window  ! Setting the network sparsity implicitly
close(1)
!! Calculate the parameters
total_syn=dble(Nr*Npost)
CR_win=int(window/M_electrode)
CR_win2=(CR_win+1)*maxN_window


!!!!!!!!!! Allocate arrays !!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
allocate(v0(Nr), v(Nr), vth(Nr), vth0(Nr), un1(Nr), un2(Nr), flag(Nr), taum(Nr))
allocate(tspk(Nr,2), tsh(Nr,maxspk), itrch(Nr), presyn(Nr), poi1(Nr), unpoi(Nr))
allocate(adj(Nr,Nr), gs0(Nr), gs(Nr), dgs(Nr), gn0(Nr), gn(Nr), cpl(Nr))
allocate(prelink1(Nr,Npre), prelink2(Nr,Npre), postlink(Nr,Npost))
allocate(hat_post(Nr,Npost), delays(Nr,Npost), dis_post(Nr,Npost), un3(Nr,Npost))
allocate(un4(Nr,Npost), sw(Nr,Npost), dw(Nr,Npost), sw_save(Nr*Npost), sw0(Nr,Nr))
allocate(stdp_poten(stdp_max), stdp_dep(stdp_max), window_nodes(window,maxN_window))
allocate(x1Dc(window), x1D(window), odparw(window), I_stim(M_electrode,ntmax))
allocate(CR_stim(Nr,ntmax), CR_groups_count(M_electrode))
allocate(widnow_CR_nodes(M_electrode,CR_win2), CR_groups(window), CR_n(M_electrode))
allocate(sw_CR(M_electrode,Nr,Npost), sw_CR1(Nr,Npost), Syn_CR(M_electrode))
allocate(sw_inCR(M_electrode), CR_neuron(Nr), coor(Nr), CR_means(M_electrode,ntmax))

window_nodes=0
x1Dc=0
x1D=0.d0
odparw=0.d0
I_stim=0.d0
v0=0.d0
v=0.d0
vth=0.d0
vth0=0.d0
un1=0.d0  ! Pseudo random number, drawn from Unif(0, 1)
un2=0.d0  ! Pseudo random number, drawn from Unif(0, 1)
flag=0
taum=0.d0 ! [micro-Farad/cm^2] Membrane capacitance
tspk=0.d0
tsh=0
itrch=0
stdp_poten=0.d0
stdp_dep=0.d0
postlink=0
prelink1=0
prelink2=0
presyn=0
hat_post=0.d0
delays=0.d0
dis_post=0.d0
cpl=0.d0
CR_groups_count=0
ub=0;lb=0
widnow_CR_nodes=0
CR_groups=0
CR_n=0
sw_CR=0
Syn_CR=0
sw_inCR=0.d0
sw_outCR=0.d0
CR_neuron=0
CR_means=0
sw_CR1=0
coor=0

!!!!!!!!!! Reading network files !!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
open(unit=2, status='old',action='read',file=trim(net_path)//trim(net_file(2)))
open(unit=3, status='old',action='read',file=trim(net_path)//trim(net_file(3)))
open(unit=4, status='old',action='read',file=trim(net_path)//trim(net_file(4)))
open(unit=5, status='old',action='read',file=trim(net_path)//trim(net_file(5)))
open(unit=6, status='old',action='read',file=trim(net_path)//trim(net_file(6)))
open(unit=7, status='old',action='read',file=trim(net_path)//trim(net_file(7)))
open(unit=8, status='old',action='read',file=trim(net_path)//trim(net_file(8)))
open(unit=9, status='old',action='read',file=trim(net_path)//trim(net_file(9)))
open(unit=10, status='old',action='read',file=trim(net_path)//trim(net_file(10)))
open(unit=11, status='old',action='read',file=trim(net_path)//trim(net_file(11)))

!! Read the positional parameter for each neuron
do i=1,Nr
    read(2,*)postlink(i,:)
    read(3,*)presyn(i)
    read(4,*)prelink1(i,:)  ! Presynaptic check's pre - read in line by line.
    read(5,*)prelink2(i,:)  ! Presynaptic check's post - read in line by line.
    read(6,*)hat_post(i,:)
    read(7,*)dis_post(i,:)
    read(11,*)coor(i)
enddo

!! ???
do i=1,window
    read(9,*)X1Dc(i), x1D(i)
    read(10,*)window_nodes(i,:)
enddo

!! Close all the open files
close(2)
close(3)
close(4)
close(5)
close(6)
close(7)
close(8)
close(9)
close(10)
close(11)

!!!!!!!!!! Calculate some variables !!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
capa=capa/Nr  ! [mS/cm^2]  Equation 4 in the paper, scales the max coupling strength by N

!open(unit=11,status='unknown',file=trim(outputpath)//trim(out_file(1)))
!open(unit=12,status='unknown',file=trim(outputpath)//trim(out_file(2)))
!open(unit=16,status='unknown',file=trim(outputpath)//trim(out_file(6)))
!open(unit=17,status='unknown',file=trim(outputpath)//trim(out_file(7)))
!open(unit=18,status='unknown',file=trim(outputpath)//trim(out_file(8)))
!open(unit=19,status='unknown',file=trim(outputpath)//trim(out_file(9)))

call random_number(un1)  ! Subroutine to draw pseudo random number from Unif(0, 1) and save to variable `un1`
call random_number(un2)  ! Subroutine to draw pseudo random number from Unif(0, 1) and save to variable `un2`

taum=sigtau*sqrt(-2.d0*log(un1))*cos(pi2*un2)+tau ! [micro-Farad/cm^2] Membrane capacitance generated via Box Muller Method to generate normal random variable at mean `tau` and stdev `sigtau`
vth=vth_rest*un1
v0=v_reset*un2
gs0=0.d0
gs=0.d0
gn0=0.d0
gn=0.d0
flag=0
tsh=0.d0      ! ???
itrch=0       ! ???
tspk=0.d0
itime_save=0
poi1=0
sw=0          ! Synaptic weight
!call init_weights_Nr(Nr,Npost,mean_weight,adj,sw0)
call init_weights(Nr,Npost,mean_weight,sw)  ! Initialize weight to one within the Npre and Npost until mean-weight is achieved.



!call create_CR_groups(Nr,window,M_electrode,maxN_window,CR_win2,x1Dc,window_nodes,CR_groups,CR_groups_count,CR_n,widnow_CR_nodes,CR_neuron)

!call CR_sw_group(M_electrode,Nr,Npost,CR_win2,postlink,CR_n, widnow_CR_nodes,Syn_CR,sw_CR,sw_CR1)
!call CR_reg_NR (Nr,f_CR,A_stim,M_electrode,window,ntmax,ti,tau,Delta_V,maxN_window,x1Dc,window_nodes,CR_groups,CR_stim,CR_means)
!call sw_inout_CR(Nr,Npost,M_electrode,sw_CR,sw,sw_CR1,sw_inCR,sw_outCR)
call CR_groups_distance (Nr,coor,M_electrode,CR_neuron,CR_n)
!do i=1,2
!    call Noisy_CR3 (Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,SD_CR,CR_stim)
!    do i1=1,ntmax
!        write(14,1)i1*ti+(i-1)*tchunk*1000, CR_stim(1,i1), CR_stim(100,i1)
!    enddo
!enddo
!stop

!!!!!!!!!! Relax / Warming up the neural network simulation !!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dgs=0.d0
do i=1,ntu
    t=i*ti;
    call poi_array(Nr,fr,poi1)  ! Generate Poisson noise spike train
    gn=dti*gn0 + D*poi1                       ! eq. 6
    gs=dti*gs0 + capa*dgs                     ! eq. 4
    v=v0+ti*(((v_rest-v0) -(gs+gn)*v0)/taum)  ! eq. 2
    vth=vth0+ti*((vth_rest-vth0))/tau_th      ! eq. 3
    do j=1,Nr
          if (v(j) .gt. vth(j) .and. flag(j) .eq. 0) then
             flag(j)=1
             tspk(j,1)=tspk(j,2)
             tspk(j,2)=t
             v(j)=v_spike; vth(j)=vth_spike
          endif
          if (flag(j) .eq. 1 ) then
             if (t .le. tspk(j,2)+t_spike)then; v(j)=v_spike; vth(j)=vth_spike;
             else ; flag(j)=0; v(j)=v_reset
             endif
          endif
    enddo

       v0=v; vth0=vth; gs0=gs; gn0=gn
enddo

    t2=tu-tchunk*1000


!!!!!!!!!! Main loop !!!!!!!!!!
!!!!!!!!!! Achieved bistable states (synchronized and desynchronized)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
lb=int(501/tchunk)
ub=lb+int(stim_dur/tchunk)-1

do itime=1,jtime
    t2=t2+tchunk*1000
    itrch=0; tsh=0
     
     CR_stim=0.d0
    !if (itime .ge. lb .and. itime .le. ub ) call NCR_Repeat(Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,SD_CR,CR_stim)
    if (itime .ge. lb .and. itime .le. ub ) call CCR (Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,CR_stim)
!    if (itime .ge. lb .and. itime .le. ub ) call SCCR(Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,CR_stim)

    do i=1,ntmax
        t=i*ti+t2;
        call poi_array(Nr,fr,poi1)
        gn=dti*gn0 + D*poi1
        gs=dti*gs0 + capa*dgs
        v=v0+ti*(((v_rest-v0) -(gs+gn)*v0  + CR_stim(:,i))/taum)
        vth=vth0+ti*((vth_rest-vth0))/tau_th
        dgs=0.d0
        do j=1,Nr
            if (v(j) .gt. vth(j) .and. flag(j) .eq. 0) then
                flag(j)=1
                itrch(j)=itrch(j)+1;
                tsh(j,itrch(j))=t-t2;
                tspk(j,1)=tspk(j,2)
                tspk(j,2)=t
                v(j)=v_spike; vth(j)=vth_spike

                ! NOTE: postlink, prelink1, and prelink2 corresponds together.
                ! postlink - marks the postsynaptic partner of each neuron specified by the row index of the matrix.
                ! prelink1 - marks the presynaptic partner of the neuron specified by the row index.
                ! prelink2 - marks the column-index in postlink for each of the corresponding entry in prelink1.
                do prei=1,presyn(j)              ! presyn is read in from a file that is 200x1 with each value no greater than 24
                    pre=prelink1(j,prei)         ! prelink1 is a matrix of 200x24
                    pre1=prelink2(j,prei)        ! prelink2 is a matrix of 200x24
                    dt2= t-tspk(pre,2)-t_d
                    if (dt2 .ge. 0.d0) then
                        !sw0(pre,j)= sw0(pre,j)+lamda*exp(-dt2/taup)
                        index1=int(dt2/ti)
                        sw(pre,pre1)=sw(pre,pre1)+ lamda*exp(-dt2/taup)
                        else
                        dt2=t-tspk(pre,1)-t_d
                        !sw0(pre,j)= sw0(pre,j)+lamda*exp(-dt2/taup) !
                        sw(pre,pre1)=sw(pre,pre1)+lamda*exp(-dt2/taup)
                    endif
                enddo
            endif
            if (flag(j) .eq. 1 ) then
                if (t .le. tspk(j,2)+t_spike)then; v(j)=v_spike; vth(j)=vth_spike;
                else; flag(j)=0; v(j)=v_reset
                endif
            endif

            if (abs(t-(tspk(j,2)+t_d)) .lt. 1.d-6)then
                do posti=1,Npost
                        post=postlink(j,posti)
                        dt2=t-tspk(post,2)
                        !sw0(j,post)= sw0(j,post) -(beta/tau_R)*lamda*exp(-dt2/taun)
                        sw(j,posti)=sw(j,posti)-(beta/tau_R)*lamda*exp(-dt2/taun)
                enddo
            endif

           do prei=1,presyn(j)
               pre=prelink1(j,prei)
               pre1=prelink2(j,prei)
               if (abs(t - (tspk(pre,2)+t_d)) .lt. 1.d-6) dgs(j)=dgs(j)+sw(pre,pre1)
            enddo

        enddo  ! loop over all nodes

        do i1=1,Nr
            do j1=1,Npost
                !! Clipping the synaptic weight between 0 and 1
                if ( sw(i1,j1) .gt. 1.d0) sw(i1,j1)=1.d0
                if ( sw(i1,j1) .lt. 0.d0) sw(i1,j1)=0.d0
            enddo
        enddo
        v0=v; vth0=vth; gs0=gs; gn0=gn
    enddo  ! loop over time=2 secons

    !if (mod (itime , 500) .eq. 1 ) then
        call sync_net(Nr,maxspk,itrch,tsh,tchunk,odpar)
        write(111,3)t/1000,odpar,sum(sw)/total_syn,sum(itrch)/(dble(Nr)*tchunk),f_CR, M_electrode,A_stim,net_index,stim_dur
!if (itime .ge. lb-3 ) then
!    do i=1, Nr
!        do j=1,itrch(i)
!            write(224,3) i,coor(i),tsh(i,j)+t2
!        enddo
!    enddo
!endif
enddo ! end time


1     format(1001(f18.5))
2     format(700(g12.5,1x))
3     format(1000(g18.8))
4     format(20(i5))

end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine CR_regular (Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,CR_stim)
    !! Multisite CR Stimulation (Paper Section V and subsectino C) !!
implicit none
    integer, intent(in):: Nr
    integer, intent(in):: M_electrode
    integer, intent(in):: ntmax
    real*8,intent(in):: ti
    real*8,intent(in):: f_CR
    real*8,intent(in):: tau      ! [micro-Farad/cm^2] Membrane capacitance Gaussian distribution mean ??? The paper defaults=3 whereas the code here defaults=150
    real*8,intent(in):: Delta_V  ! [mV] V_th_spike = V_reset
    real*8,intent(in):: A_stim
    integer, intent(in)::CR_neuron(Nr)
    real*8,intent(out):: CR_stim(Nr,ntmax)
    real*8,parameter:: stim_length1=0.4  ! [ms] Excitatory pulse duration
    real*8,parameter:: tsilent=0.2       ! [ms] Time separation between Excitatory- and Inhibitory-pulse
    real*8,parameter:: stim_length2=3.d0 ! [ms] Inhibitory pulse duration
    real*8:: T_CR
    real*8:: Delta_CR
    real*8:: t
    real*8:: As1
    real*8:: As2
    real*8:: S1
    !real*8:: I_stim(M_electrode,ntmax)
    integer:: CR_period_dt
    integer:: Delta_CR_dt
    integer:: st1
    integer:: st2
    integer:: st3
    integer:: electrodes(M_electrode)
    integer:: M_shuffled(M_electrode)
    integer::electrode
    integer::k1,i,j, lb, ub,k2,i1,j1,k3,sigma_CR,ntmax1,k1ub
    real*8,allocatable,dimension(:,:):: I_stim

    
    CR_stim=0.d0
    T_CR=1000.d0/f_CR
    CR_period_dt=int(T_CR/ti) ! time frame for stimulating all sub-populations
    ntmax1=ntmax+2*CR_period_dt
    allocate (I_stim(M_electrode,ntmax1))
    I_stim=0.d0
    Delta_CR=T_CR/M_electrode  ! ms time between two CR stimultion
    Delta_CR_dt=int(Delta_CR/ti)
    As1=A_stim*tau*Delta_V/(stim_length1)
    As2=A_stim*tau*Delta_V/(stim_length2)


    st1=int(stim_length1/ti)
    st2=st1+int(tsilent/ti)
    st3=int(stim_length2/ti)+st2+st1
    k1ub=CR_period_dt-(st1+st2+st3)
    I_stim=0.d0
    k1=0; k2=0; k3=0

    do i=1,ntmax,CR_period_dt
        k1=i
        call shuffle_int_array(M_electrode,electrodes)
        do j=1,M_electrode
            !electrodes(j)=j
            I_stim(electrodes(j),k1:k1+st1)=As1;
            I_stim(electrodes(j),k1+st2:k1+st3)=-As1
!            CR_means(electrodes(j),k1)=dble(k1)/10.d0
            k1=k1+Delta_CR_dt
            enddo
    enddo

!    do i=1,ntmax,CR_period_dt
!        k1=i
!        k3=k3+1
!        !call shuffle_int_array(M_electrode,M_shuffled)
!        call rnadom_replace_int(M_electrode,M_shuffled)
!        do j=1,M_electrode
!            electrode=j!M_shuffled(j)
!            call random_int(k1,k1+k1ub,k2)
!            I_stim(electrode,k2:k2+st1)=As1;
!            I_stim(electrode,k2+st2:k2+st3)=-As2
!        enddo
!    enddo

    do i=1,Nr
        CR_stim(i,1:ntmax)=I_stim(CR_neuron(i),1:ntmax)
    enddo
deallocate(I_stim)
return
end



subroutine SCCR (Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,CR_stim)
!Sept 2021, compound CR
implicit none
    integer, intent(in):: Nr,M_electrode,ntmax
    real*8,intent(in)::ti,f_CR,tau,Delta_V,A_stim
    integer, intent(in)::CR_neuron(Nr)
    real*8,intent(out):: CR_stim(Nr,ntmax)
    real*8,parameter:: stim_length1=0.4,tsilent=0.2,stim_length2=3.d0
    real*8:: T_CR,Delta_CR,t,As1,As2,S1
    !real*8:: I_stim(M_electrode,ntmax)
    integer:: CR_period_dt,Delta_CR_dt,st1,st2,st3, electrodes(M_electrode),M_shuffled(M_electrode),electrode
    integer::k1,i,j, lb, ub,k2,i1,j1,k3,sigma_CR,ntmax1,k1ub
    real*8,allocatable,dimension(:,:):: I_stim,CR_means

    
    CR_stim=0.d0
    T_CR=1000.d0/f_CR
    CR_period_dt=int(T_CR/ti) ! time frame for stimulating all sub-populations
    ntmax1=ntmax+2*CR_period_dt
    allocate (I_stim(M_electrode,ntmax1),CR_means(M_electrode,ntmax))
    I_stim=0.d0
    Delta_CR=T_CR/M_electrode  ! ms time between two CR stimultion
    Delta_CR_dt=int(Delta_CR/ti)
    As1=A_stim*tau*Delta_V/(stim_length1)
    As2=A_stim*tau*Delta_V/(stim_length2)


    st1=int(stim_length1/ti)
    st2=st1+int(tsilent/ti)
    st3=int(stim_length2/ti)+st2+st1
    k1ub=CR_period_dt-(st1+st2+st3)
    I_stim=0.d0
    k1=0; k2=0; k3=0
    do i=1,ntmax,CR_period_dt
        k1=i
        k3=k3+1
        !call shuffle_int_array(M_electrode,M_shuffled)
        call rnadom_replace_int(M_electrode,M_shuffled)
        do j=1,M_electrode
            electrode=M_shuffled(j)
            call random_int(k1,k1+k1ub,k2)
            I_stim(electrode,k2:k2+st1)=As1;
            I_stim(electrode,k2+st2:k2+st3)=-As2
        enddo
    enddo

    do i=1,Nr
        CR_stim(i,1:ntmax)=I_stim(CR_neuron(i),1:ntmax)
    enddo
deallocate(I_stim,CR_means)
return
end

subroutine CCR (Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,CR_stim)
!Sept 2021, compound CR
implicit none
    integer, intent(in):: Nr,M_electrode,ntmax
    real*8,intent(in)::ti,f_CR,tau,Delta_V,A_stim
    integer, intent(in)::CR_neuron(Nr)
    real*8,intent(out):: CR_stim(Nr,ntmax)
    real*8,parameter:: stim_length1=0.4,tsilent=0.2,stim_length2=3.d0
    real*8:: T_CR,Delta_CR,t,As1,As2,S1
    !real*8:: I_stim(M_electrode,ntmax)
    integer:: CR_period_dt,Delta_CR_dt,st1,st2,st3, electrodes(M_electrode),M_shuffled(M_electrode),electrode
    integer::k1,i,j, lb, ub,k2,i1,j1,k3,sigma_CR,ntmax1,k1ub
    real*8,allocatable,dimension(:,:):: I_stim,CR_means

    
    CR_stim=0.d0
    T_CR=1000.d0/f_CR
    CR_period_dt=int(T_CR/ti) ! time frame for stimulating all sub-populations
    ntmax1=ntmax+2*CR_period_dt
    allocate (I_stim(M_electrode,ntmax1),CR_means(M_electrode,ntmax))
    I_stim=0.d0
    Delta_CR=T_CR/M_electrode  ! ms time between two CR stimultion
    Delta_CR_dt=int(Delta_CR/ti)
    As1=A_stim*tau*Delta_V/(stim_length1)
    As2=A_stim*tau*Delta_V/(stim_length2)


    st1=int(stim_length1/ti)
    st2=st1+int(tsilent/ti)
    st3=int(stim_length2/ti)+st2+st1
    k1ub=CR_period_dt-(st1+st2+st3)
    I_stim=0.d0
    k1=0; k2=0; k3=0
    do i=1,ntmax,CR_period_dt
        k1=i
        k3=k3+1
        !call shuffle_int_array(M_electrode,M_shuffled)
        call rnadom_replace_int(M_electrode,M_shuffled)
        do j=1,M_electrode
            electrode=j!M_shuffled(j)
            call random_int(k1,k1+k1ub,k2)
            I_stim(electrode,k2:k2+st1)=As1;
            I_stim(electrode,k2+st2:k2+st3)=-As2
        enddo
    enddo

    do i=1,Nr
        CR_stim(i,1:ntmax)=I_stim(CR_neuron(i),1:ntmax)
    enddo
deallocate(I_stim,CR_means)
return
end


subroutine NCR_Repeat (Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,SD_CR,CR_stim)
!Dec 2020, Noisy CR with repeated sites
implicit none
    integer, intent(in):: Nr,M_electrode,ntmax
    real*8,intent(in)::ti,f_CR,tau,Delta_V,A_stim,SD_CR
    integer, intent(in)::CR_neuron(Nr)
    real*8,intent(out):: CR_stim(Nr,ntmax)
    real*8,parameter:: stim_length1=0.4,tsilent=0.2,stim_length2=3.d0
    real*8:: T_CR,Delta_CR,t,As1,As2,S1
    !real*8:: I_stim(M_electrode,ntmax)
    integer:: CR_period_dt,Delta_CR_dt,st1,st2,st3, electrodes(M_electrode),M_shuffled(M_electrode),electrode
    integer::k1,i,j, lb, ub,k2,i1,j1,k3,sigma_CR,ntmax1
    real*8,allocatable,dimension(:,:):: I_stim,CR_means

    
    CR_stim=0.d0
    T_CR=1000.d0/f_CR
    CR_period_dt=int(T_CR/ti) ! time frame for stimulating all sub-populations
    ntmax1=ntmax+2*CR_period_dt
    allocate (I_stim(M_electrode,ntmax1),CR_means(M_electrode,ntmax))
    I_stim=0.d0
    Delta_CR=T_CR/M_electrode  ! ms time between two CR stimultion
    Delta_CR_dt=int(Delta_CR/ti)
    As1=A_stim*tau*Delta_V/(stim_length1)
    As2=A_stim*tau*Delta_V/(stim_length2)
    sigma_CR=int(SD_CR*Delta_CR_dt/2)

    st1=int(stim_length1/ti)
    st2=st1+int(tsilent/ti)
    st3=int(stim_length2/ti)+st2+st1

    I_stim=0.d0
    k1=0; k2=0; k3=0
    do i=1,ntmax,CR_period_dt
        k1=i+sigma_CR
        k3=k3+1
        !call shuffle_int_array(M_electrode,M_shuffled)
        call rnadom_replace_int(M_electrode,M_shuffled)
        do j=1,M_electrode
            electrode=M_shuffled(j)
            call random_int(k1-sigma_CR,k1+sigma_CR,k2)
            I_stim(electrode,k2:k2+st1)=As1;
            I_stim(electrode,k2+st2:k2+st3)=-As2
            !CR_means(electrode,k3)=dble(k2)
            k1=k1+Delta_CR_dt
        enddo
    enddo

    do i=1,Nr
        CR_stim(i,1:ntmax)=I_stim(CR_neuron(i),1:ntmax)
    enddo
deallocate(I_stim,CR_means)
return
end

subroutine rnadom_replace_int(size,array)
implicit none
    integer,intent(in) :: size
    integer,intent(out):: array(size)
    real*8:: un4(size)

    un4=0.d0
    !call random_seed()
    call random_number(un4)
    array=1 + FLOOR((size)*un4)

return
end


subroutine Noisy_CR3 (Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,SD_CR,CR_stim)
!Sept 2020
implicit none
    integer, intent(in):: Nr,M_electrode,ntmax
    real*8,intent(in)::ti,f_CR,tau,Delta_V,A_stim,SD_CR
    integer, intent(in)::CR_neuron(Nr)
    real*8,intent(out):: CR_stim(Nr,ntmax)
    real*8,parameter:: stim_length1=0.4,tsilent=0.2,stim_length2=3.d0
    real*8:: T_CR,Delta_CR,t,As1,As2,S1
    !real*8:: I_stim(M_electrode,ntmax)
    integer:: CR_period_dt,Delta_CR_dt,st1,st2,st3, electrodes(M_electrode),M_shuffled(M_electrode),electrode
    integer::k1,i,j, lb, ub,k2,i1,j1,k3,sigma_CR,ntmax1
    real*8,allocatable,dimension(:,:):: I_stim,CR_means

    
    CR_stim=0.d0
    T_CR=1000.d0/f_CR
    CR_period_dt=int(T_CR/ti) ! time frame for stimulating all sub-populations
    ntmax1=ntmax+2*CR_period_dt
    allocate (I_stim(M_electrode,ntmax1),CR_means(M_electrode,ntmax))
    I_stim=0.d0
    Delta_CR=T_CR/M_electrode  ! ms time between two CR stimultion
    Delta_CR_dt=int(Delta_CR/ti)
    As1=A_stim*tau*Delta_V/(stim_length1)
    As2=A_stim*tau*Delta_V/(stim_length2)
    sigma_CR=int(SD_CR*Delta_CR_dt/2)

    st1=int(stim_length1/ti)
    st2=st1+int(tsilent/ti)
    st3=int(stim_length2/ti)+st2+st1

    I_stim=0.d0
k1=0; k2=0; k3=0
    do i=1,ntmax,CR_period_dt
        k1=i+sigma_CR
        k3=k3+1
        call shuffle_int_array(M_electrode,M_shuffled)
        do j=1,M_electrode
            electrode=M_shuffled(j)
            call random_int(k1-sigma_CR,k1+sigma_CR,k2)
            I_stim(electrode,k2:k2+st1)=As1;
            I_stim(electrode,k2+st2:k2+st3)=-As2
            !CR_means(electrode,k3)=dble(k2)
            k1=k1+Delta_CR_dt
        enddo
    enddo

    do i=1,Nr
        CR_stim(i,1:ntmax)=I_stim(CR_neuron(i),1:ntmax)
    enddo
deallocate(I_stim,CR_means)
return
end



subroutine Noisy_CR2 (Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,SD_CR,CR_stim)
! Sept 2020 fixed the loop
implicit none
    integer, intent(in):: Nr,M_electrode,ntmax
    real*8,intent(in)::ti,f_CR,tau,Delta_V,A_stim,SD_CR
    integer, intent(in)::CR_neuron(Nr)
    real*8,intent(out):: CR_stim(Nr,ntmax)
    real*8,parameter:: stim_length1=0.4,tsilent=0.2,stim_length2=3.d0
    real*8:: T_CR,Delta_CR,t,As1,As2,S1,un2
    !real*8:: I_stim(M_electrode,ntmax)
    integer:: CR_period_dt,Delta_CR_dt,st1,st2,st3, electrodes(M_electrode),M_shuffled(M_electrode),electrode
    integer::k1,i,j, lb, ub,k2,i1,j1,k3,sigma_CR,time1,count1,ranmax,ntmax1,k0
    real*8,allocatable,dimension(:,:):: I_stim,CR_means
    real*8,allocatable,dimension(:):: stim_tm
    

 
    T_CR=1000.d0/f_CR
    ntmax1=0; ranmax=0
    ranmax=int((ntmax/10000)*f_CR*M_electrode*1000)
    CR_period_dt=int(T_CR/ti) ! time frame for stimulating all sub-populations
    Delta_CR=T_CR/M_electrode  ! ms time between two CR stimultion
    Delta_CR_dt=int(Delta_CR/ti)
    As1=A_stim*tau*Delta_V/(stim_length1)
    As2=A_stim*tau*Delta_V/(stim_length2)
    sigma_CR=int(SD_CR*Delta_CR_dt/2)
    ntmax1=ntmax+CR_period_dt
    allocate(I_stim(M_electrode,ntmax1),stim_tm(ranmax),CR_means(M_electrode,ntmax))
    I_stim=0.d0
    CR_stim=0.d0
    stim_tm=0.d0
    st1=int(stim_length1/ti)
    st2=st1+int(tsilent/ti)
    st3=int(stim_length2/ti)+st2+st1
    print*,'Ali',ranmax
    call random_number(stim_tm)
    stim_tm=(stim_tm*2*sigma_CR)-sigma_CR
    I_stim=0.d0
    k2=0;k0=0; k3=0
    time1=0; count1=0
    k1=int(Delta_CR_dt/2)
    CR_means=0.d0
    do while (time1 .lt. ntmax1)
        call shuffle_int_array(M_electrode,M_shuffled)
        k3=k3+1
        do j=1,M_electrode
            electrode=M_shuffled(j)
            count1=count1+1
            k2=k1+int(stim_tm(count1))
            I_stim(electrode,int(k2):int(k2+st1))=As1;
            I_stim(electrode,int(k2+st2):int(k2+st3))=-As2
            k1=k1+Delta_CR_dt
            !print*,k2
            !CR_means(electrode,k3)=dble(k2)
        enddo
        time1=k2+M_electrode *Delta_CR_dt
    enddo
    !do i=1,k3
    !    write(19,*)CR_means(:,i)
    !enddo
    do i=1,Nr
        CR_stim(i,1:ntmax)=I_stim(CR_neuron(i),1:ntmax)
    enddo
deallocate(I_stim,stim_tm,CR_means)

return
end




subroutine CR_groups_distance (Nr,coor,M_electrode,CR_neuron,CR_n)
implicit none
integer,intent(in):: Nr,M_electrode
integer,intent(in):: coor(Nr)
integer,intent(out):: CR_neuron(Nr), CR_n(M_electrode)
integer:: max1,min1
integer:: k1,i,i1
integer::sx1,slb,sub,stepx(M_electrode+1)


max1=maxval(coor); min1=minval(coor)
CR_neuron=0; CR_n=0

stepx(1)=min1; stepx(M_electrode+1)=max1
sx1=1+int((max1-min1)/M_electrode)
do i=1,M_electrode-1
    stepx(i+1)=stepx(i)+sx1
enddo

do i=1,M_electrode
    if (i .lt. M_electrode) then
        do i1=1,Nr
            if (coor(i1) .ge. stepx(i) .and. coor(i1) .lt. stepx(i+1)) then
                CR_n(i)= CR_n(i)+1; CR_neuron(i1)=i
            endif
        enddo
    else
        do i1=1,Nr
            if (coor(i1) .ge. stepx(i) .and. coor(i1) .le. stepx(i+1)) then
            CR_n(i)= CR_n(i)+1; CR_neuron(i1)=i
            endif
        enddo
     endif
enddo

return
end

subroutine sync_net(node,maxspk,itrh,tsh,tmax,odpar)
      implicit none
      ! fixed Spet 2021, now it goes close to zero for desync state
      integer,intent(in):: node, maxspk
      integer,intent(in):: itrh(node)
      real*8, intent(in) :: tmax,tsh(node,maxspk)
      real*8,intent(out):: odpar

      real*8, parameter :: tsamp=0.1d0
      integer nh,itrc, i, j, j1, i2, i1, nt1,nt2,nti,ntf
      real*8 t, pi2, odpar1, odpar_dif, odpar_dif1,odpar2,syn1, fspk, lspk
      real*8, dimension (maxspk) :: ts1
      real*8, allocatable, dimension (:,:) ::  phn,psin
      complex*16 im, odp,odpave,odpave1,odp1
      parameter (im=(0.d0,1.d0))
      
     odpar=0
      pi2=8*atan(1.d0)
      nt1=tmax*1000.d0/tsamp ! sampling, we sample every one millisecond
      fspk=0.d0
      lspk=0.d0
      nt2=0; nti=0; ntf=0
      allocate(phn(node,nt1))
      phn=0.0; odpar=0.d0; odp=0.d0
! calculate phase: only if all nodes are spiking
    lspk=tsh(1,itrh(1))
    do j=1,node
          ts1=tsh(j,:)
          do i=1,nt1
            t=(i-1)*tsamp
            call locate(ts1,itrh(j),t,i2)
            if(i2.ge.1.and.i2.le.itrh(j)-1) phn(j,i)=pi2*(t-ts1(i2))/(ts1(i2+1)-ts1(i2))+pi2*i2
         enddo
         if (tsh(j,itrh(j)) .lt. lspk) lspk=tsh(j,itrh(j))
     enddo
     fspk =maxval(tsh(:,1))
     nti=int(fspk/tsamp)+100
     ntf=int(lspk/tsamp)-100
     
!do i=1,nt1
!   odp=sum(exp(im*phn(:,i)))/node
!   odpar=odpar+abs(odp)
!enddo
!15  continue
!odpar=odpar/(nt1)
      do i=nti,ntf
         odp=sum(exp(im*phn(:,i)))/node
         odpar=odpar+abs(odp)
      enddo
      odpar=(odpar/(ntf-nti))
    deallocate(phn)
return
end
     


SUBROUTINE locate(xx,n,x,j)
      implicit none
      INTEGER j,n
      real*8 x,xx(n)
      INTEGER jl,jm,ju
      jl=0
      ju=n+1
10    if(ju-jl.gt.1)then
        jm=(ju+jl)/2
        if((xx(n).ge.xx(1)).eqv.(x.ge.xx(jm)))then
          jl=jm
        else
          ju=jm
        endif
      goto 10
      endif
      if(x.eq.xx(1))then
        j=1
      else if(x.eq.xx(n))then
        j=n-1
      else
        j=jl
      endif
 return
END


subroutine poi_array(Nr,mean,poi)
    !! Poisson Noise generation !!
    !! When n > 20 and np < 5, Binomial(n, p) and Poi(mean = np) are similar
    !! This is why the code below simulates Poisson RV in a Binomial fasion.
    integer,intent(in):: Nr
    real*8,intent(in):: mean
    integer,intent(out):: poi(Nr)
    real*8:: un1(Nr)
    integer:: j

    un1=0.d0
    call random_number(un1)  ! Pseudo random number ~ Unif(0, 1)
    poi=0
    do j=1,Nr
       if (un1(j) .lt. mean) poi(j)=1
    enddo

return
end

subroutine sw_inout_CR(Nr,Npost,M_electrode,sw_CR,sw,sw_CR1,sw_inCR,sw_outCR)
implicit none
integer,intent(in):: Nr,Npost,M_electrode
integer,intent(in):: sw_CR(M_electrode,Nr,Npost),sw_CR1(Nr,Npost)
real*8,intent(in):: sw(Nr,Npost)
real*8,intent(out):: sw_inCR(M_electrode),sw_outCR
integer:: i1,i2,j

    sw_inCR=0.d0
    sw_outCR=0.d0
    do j=1,M_electrode
            sw_inCR(j)=sum(sw*sw_CR(j,:,:))
    enddo
    sw_outCR=sum(sw*sw_CR1)
return
end

subroutine Noisy_CR1 (Nr,f_CR,A_stim,M_electrode,ntmax,ti,tau,Delta_V,CR_neuron,SD_CR,CR_stim,I_stim)
implicit none
    integer, intent(in):: Nr,M_electrode,ntmax
    real*8,intent(in)::ti,f_CR,tau,Delta_V,A_stim,SD_CR
    integer, intent(in)::CR_neuron(Nr)
    real*8,intent(out):: CR_stim(Nr,ntmax),I_stim(M_electrode,ntmax)
    real*8,parameter:: stim_length1=0.4,tsilent=0.2,stim_length2=3.d0
    real*8:: T_CR,Delta_CR,t,As1,As2,S1,CR_means(M_electrode,ntmax)
    !real*8:: I_stim(M_electrode,ntmax)
    integer:: CR_period_dt,Delta_CR_dt,st1,st2,st3, electrodes(M_electrode),M_shuffled(M_electrode),electrode
    integer::k1,i,j, lb, ub,k2,i1,j1,k3,sigma_CR

    I_stim=0.d0
    CR_stim=0.d0
    T_CR=1000.d0/f_CR
    CR_period_dt=int(T_CR/ti) ! time frame for stimulating all sub-populations
    Delta_CR=T_CR/M_electrode  ! ms time between two CR stimultion
    Delta_CR_dt=int(Delta_CR/ti)
    As1=A_stim*tau*Delta_V/(stim_length1)
    As2=A_stim*tau*Delta_V/(stim_length2)
    sigma_CR=int(SD_CR*Delta_CR_dt/2)

    st1=int(stim_length1/ti)
    st2=st1+int(tsilent/ti)
    st3=int(stim_length2/ti)+st2+st1

    I_stim=0.d0
    k1=0; k2=0
    do i=1,ntmax-Delta_CR_dt,CR_period_dt
        k1=i+sigma_CR
        call shuffle_int_array(M_electrode,M_shuffled)
        do j=1,M_electrode
            electrode=M_shuffled(j)
            call random_int(k1-sigma_CR,k1+sigma_CR-st3,k2)
            I_stim(electrode,k2:k2+st1)=As1;
            I_stim(electrode,k2+st2:k2+st3)=-As2
            k1=k1+Delta_CR_dt
        enddo
    enddo
    do i=1,Nr
        CR_stim(i,1:ntmax)=I_stim(CR_neuron(i),1:ntmax)
    enddo

return
end



subroutine RVS_Repalce (Nr,f_CR,A_stim,M_electrode,window,ntmax,ti,tau,Delta_V,maxN_window,x1Dc,window_nodes,CR_groups,CR_stim)
implicit none
    integer, intent(in):: Nr,M_electrode,ntmax,window,maxN_window
    real*8,intent(in)::ti,f_CR,tau,Delta_V,A_stim
    integer, intent(in)::x1Dc(window), window_nodes(window,maxN_window),CR_groups(window)
    real*8,intent(out):: CR_stim(Nr,ntmax)
    real*8,parameter:: stim_length=0.4,tsilent=0.2
    real*8:: T_CR,Delta_CR,t,As1,S1,CR_means(M_electrode,ntmax)
    real*8:: I_stim(M_electrode,ntmax)
    integer:: CR_period_dt,Delta_CR_dt,st1,st2,st3, electrodes(M_electrode)
    integer::k1,i,j, lb, ub,k2,i1,j1,k3


    electrodes=0; CR_means=0.d0
    T_CR=1000.d0/f_CR
    CR_period_dt=int(T_CR/ti) ! time frame for stimulating all sub-populations
    Delta_CR=T_CR/M_electrode  ! ms time between two CR stimultion
    Delta_CR_dt=int(Delta_CR/ti)
    st1=int(stim_length/ti)
    st2=st1+int(tsilent/ti)
    st3=st2+st1
    As1=A_stim*tau*Delta_V/(stim_length)
    I_stim=0.d0
    k1=0

    do i=1,ntmax-Delta_CR_dt,CR_period_dt
        k1=i
        call rnadom_replace_int(M_electrode,electrodes)
        do j=1,M_electrode
            I_stim(electrodes(j),k1:k1+st1)=As1;
            I_stim(electrodes(j),k1+st2:k1+st3)=-As1
            CR_means(electrodes(j),k1)=dble(k1)/10.d0
            k1=k1+Delta_CR_dt
        enddo
    enddo

    do i1=1,window
        do j1=1,x1Dc(i1)
            CR_stim(window_nodes(i1,j1),:)=I_stim(CR_groups(i1),:)
        enddo
    enddo

return
end





subroutine CR_reg_NR (Nr,f_CR,A_stim,M_electrode,window,ntmax,ti,tau,Delta_V,maxN_window,x1Dc,window_nodes,CR_groups,CR_stim)
implicit none
    integer, intent(in):: Nr,M_electrode,ntmax,window,maxN_window
    real*8,intent(in)::ti,f_CR,tau,Delta_V,A_stim
    integer, intent(in)::x1Dc(window), window_nodes(window,maxN_window),CR_groups(window)
    real*8,intent(out):: CR_stim(Nr,ntmax)
    real*8,parameter:: stim_length=0.4,tsilent=0.2
    real*8:: T_CR,Delta_CR,t,As1,S1,CR_means(M_electrode,ntmax)
    real*8:: I_stim(M_electrode,ntmax)
    integer:: CR_period_dt,Delta_CR_dt,st1,st2,st3, electrodes(M_electrode)
    integer::k1,i,j, lb, ub,k2,i1,j1,k3


    electrodes=0; CR_means=0.d0
    T_CR=1000.d0/f_CR
    CR_period_dt=int(T_CR/ti) ! time frame for stimulating all sub-populations
    Delta_CR=T_CR/M_electrode  ! ms time between two CR stimultion
    Delta_CR_dt=int(Delta_CR/ti)
    st1=int(stim_length/ti)
    st2=st1+int(tsilent/ti)
    st3=st2+st1
    As1=A_stim*tau*Delta_V/(stim_length)
    I_stim=0.d0
    k1=0

    do i=1,ntmax-CR_period_dt,CR_period_dt
        k1=i
        call shuffle_int_array(M_electrode,electrodes)
        do j=1,M_electrode
            !electrodes(j)=j
            I_stim(electrodes(j),k1:k1+st1)=As1;
            I_stim(electrodes(j),k1+st2:k1+st3)=-As1
            CR_means(electrodes(j),k1)=dble(k1)/10.d0
            k1=k1+Delta_CR_dt
            enddo
    enddo

    do i1=1,window
        do j1=1,x1Dc(i1)
            CR_stim(window_nodes(i1,j1),:)=I_stim(CR_groups(i1),:)
        enddo
    enddo

return
end



subroutine shuffle_int_array(size,array)
implicit none
    integer,intent(in) :: size
    integer,intent(out):: array(size)
    integer:: k,i,j,n,m,itemp
    real*8:: u

    array=[(i,i=1,size)]
    n=1;m=size
    do k=1,2
     do i=1,m
      call random_number(u)
      j = n + FLOOR((m+1-n)*u)
      itemp=array(j); array(j)=array(i); array(i)=itemp
     enddo
    enddo

return
end


subroutine init_weights(Nr,Npost,mean_weight,sw)
    integer,intent(in):: Nr,Npost
    real*8,intent(in):: mean_weight
    real*8,intent(out):: sw(Nr,Npost)
    integer:: i1,j1
    real*8:: u2(2),msw

    sw=0.d0; msw=0.d0
    do while (msw <mean_weight)
        !call random_seed()
        call random_number(u2)
        i1=1+floor(Nr*u2(1));j1=1+floor(Npost*u2(2))
        sw(i1,j1)=1.d0
        msw=sum(sw)/dble(Nr*Npost)
    enddo

return
end

subroutine CR_sw_group(M_electrode,Nr,Npost,CR_win2,postlink,CR_n, widnow_CR_nodes,Syn_CR,sw_CR,sw_CRout)
implicit none
integer, intent(in):: Nr,M_electrode,Npost,CR_win2
integer,intent(in):: postlink(Nr,Npost),widnow_CR_nodes(M_electrode,CR_win2),CR_n(M_electrode)
integer,intent(out)::Syn_CR(M_electrode),sw_CR(M_electrode,Nr,Npost), sw_CRout(Nr,Npost)
integer::i1,i2,i3,i4,j,j1,j2,k1,k2,pre,post

Syn_CR=0; sw_CR=0; sw_CRout=0
    
    do j=1,M_electrode
        k1=0
        do i1=1,Nr
            pre=i1
            do j1=1,Npost
                post=postlink(i1,j1)
                do i2=1,CR_n(j)
                    if (widnow_CR_nodes(j,i2)==pre) then
                        do i3=1,CR_n(j)
                                if (widnow_CR_nodes(j,i3)==post .and. widnow_CR_nodes(j,i3) .ne. pre) then
                                k1=k1+1
                                sw_CR(j,pre,j1)=1
                            endif
                        enddo
                    endif
                enddo
            enddo
        enddo
        Syn_CR(j)=k1
    enddo
    
    sw_CRout=1
    do j=1,M_electrode
        do i1=1,Nr
            do j1=1,Npost
                if (sw_CR(j,i1,j1)==1) sw_CRout(i1,j1)=0
            enddo
        enddo
    enddo
   
return
end


subroutine create_CR_groups(Nr,window,M_electrode,maxN_window,CR_win2,x1Dc,window_nodes,CR_groups,CR_groups_count,CR_n,widnow_CR_nodes,CR_neuron)
implicit none
    integer,intent(in):: window,M_electrode,CR_win2,Nr,maxN_window
    integer,intent(in)::X1Dc(window),window_nodes(window,maxN_window)
    integer,intent(out):: CR_groups(window), CR_groups_count(M_electrode),CR_n(M_electrode),widnow_CR_nodes(M_electrode,CR_win2),CR_neuron(Nr)

    integer::win_CR,k2,lb,ub,j,i,k1,i1,i2,j1
    
    win_CR=0; k2=0; lb=0; ub=0; j=0; CR_n=0
    win_CR=int(window/M_electrode)
    CR_groups_count=0; CR_groups=0
    widnow_CR_nodes=0
    if (M_electrode .gt. window) win_CR=1
    k2=mod(window,M_electrode)
    CR_groups_count(1:M_electrode)=win_CR
    if (k2 .eq. 0) then
        lb=1; ub=win_CR
        do j=1,M_electrode
            CR_groups(lb:ub)=j
            lb=ub+1; ub=ub+win_CR
        enddo
    elseif (k2 .gt. 0 ) then
        do i=1,k2
            CR_groups_count(i)=win_CR+1
        enddo
        ub=0
        do j=1,M_electrode
            lb=ub+1
            ub=lb+CR_groups_count(j)-1
            CR_groups(lb:ub)=j
        enddo
    endif

    ub=0
    do j=1,M_electrode
        lb=ub+1
        ub=ub+CR_groups_count(j)
        CR_n(j)= sum(X1Dc(lb:ub))
        k1=0
        do i1=lb,ub
            do i2=1,x1Dc(i1)
            k1=k1+1
            widnow_CR_nodes(j,k1)=window_nodes(i1,i2)
            enddo
        enddo
    enddo
    
    do i1=1,M_electrode
        do i2=1,CR_n(i1)
            j1=widnow_CR_nodes(i1,i2)
            CR_neuron(j1)=i1
        enddo
    enddo

return
end


 subroutine create_directory( folder_name)
    implicit none
    character(len=*), intent(in) :: folder_name
    character(len=256)           :: mkdirCmd
    logical                      :: dirExists

    mkdirCmd = 'mkdir -p '//trim(folder_name)
    inquire( directory=folder_name, exist=dirExists )
    if ( .not. dirExists) call system( mkdirCmd )
    
end subroutine create_directory


subroutine random_int(lb,ub,k2)
implicit none
    integer,intent(in):: lb,ub
    integer,intent(out)::k2
    real*8:: u

    !call random_seed()
    call random_number(u)
    k2 = lb + FLOOR((ub+1-lb)*u)

return
end
