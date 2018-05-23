#!/bin/bash
#SBATCH -p regular
#SBATCH -N 1 
#SBATCH -C haswell
#SBATCH -t 00:10:00
#SBATCH -o flash_1.txt
#DW jobdw capacity=1289GiB access_mode=striped type=scratch pool=sm_pool

OUTDIR=/global/cscratch1/sd/khl7265/FS_64_8M/flash
BBDIR=${BB_JOB_STRIPED}flash
NN=${SLURM_NNODES}
let NP=NN*32

echo "mkdir -p ${OUTDIR}"
mkdir -p ${OUTDIR}
echo "mkdir -p ${BBDIR}"
mkdir -p ${BBDIR}

# Bb
export PNETCDF_HINTS="pnetcdf_bb=enable;pnetcdf_bb_del_on_close=disable;pnetcdf_bb_overwrite=enable;pnetcdf_bb_dirname=${BBDIR}"
for i in 1 2 3
do
    echo "#%$: experiment: bb_timing_breakdown"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*
    echo "rm -f ${BBDIR}/*"
    rm -f ${BBDIR}/*
    
    echo "srun -n ${NP} ./flash_benchmark_io ${OUTDIR}/flash_ blocking coll"
    srun -n ${NP} ./flash_benchmark_io ${OUTDIR}/flash_ blocking coll

    echo "#%$: io_driver: bb"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: io_mode: ${u}_${v}"

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
                
    echo '-----+-----++------------+++++++++--+---'
done
unset PNETCDF_HINTS
echo '--++---+----+++-----++++---+++--+-++--+---'

echo "BB Info: "
module load dws
sessID=$(dwstat sessions | grep $SLURM_JOBID | awk '{print $1}')
echo "session ID is: "${sessID}
instID=$(dwstat instances | grep $sessID | awk '{print $1}')
echo "instance ID is: "${instID}
echo "fragments list:"
echo "frag state instID capacity gran node"
dwstat fragments | grep ${instID}


