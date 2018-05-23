changequote(`[', `]')dnl
define([SCRIPT], dnl
[dnl
#!/bin/bash
#SBATCH -p regular
#SBATCH -N $2 
#SBATCH -C haswell
#SBATCH -t 00:10:00
#SBATCH -o stage_$6_$7_$3.txt
#DW jobdw capacity=1289GiB access_mode=striped type=scratch pool=sm_pool
echo "IO_DRIVER:STAGE"
echo "IO_MODE:$6"
echo "N_NODE:$2"
echo "N_PROC:$3"
echo ${BB_JOB_STRIPED}stage
mkdir -p $1
mkdir -p $1/stage_$6_$7_$3
mkdir -p ${BB_JOB_STRIPED}stage
export MPICH_MPIIO_HINTS="*:romio_cb_write=disable"
export stageout_bb_path="${BB_JOB_STRIPED}stage"
export stageout_pfs_path="$1/stage_$6_$7_$3"
for i in 1 2 3
do
    echo "Round " $i ":"
    rm -f $1/stage_$6_$7_$3/*
    rm -f ${BB_JOB_STRIPED}stage/*
    srun -n $3 --export=MPICH_MPIIO_HINTS ./flash_benchmark_io ${BB_JOB_STRIPED}stage/flash_ 
    echo "ls -lah $1/stage_$6_$7_$3/*"
    ls -lah $1/stage_$6_$7_$3/*
    echo "ls -lah ${BB_JOB_STRIPED}stage/*"
    ls -lah ${BB_JOB_STRIPED}stage/*
    echo '-----+-----++------------+++++++++--+---'
done
echo '--++---+----+++-----++++---+++--+-++--+---'
unset stageout_bb_path
unset stageout_pfs_path
echo "BB Info: "
module load dws
sessID=$(dwstat sessions | grep $SLURM_JOBID | awk '{print $5}')
echo "session ID is: "${sessID}
instID=$(dwstat instances | grep $sessID | awk '{print $5}')
echo "instance ID is: "${instID}
echo "fragments list:"
echo "frag state instID capacity gran node"
dwstat fragments | grep ${instID}
])dnl
SCRIPT(EXPDIR,NNODE,NPROC,LVL,[$1],IOMODEA, IOMODEB)

