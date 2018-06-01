#SBATCH -p debug
#SBATCH -N 1 
#SBATCH -C haswell
#SBATCH -t 00:10:00
#SBATCH -o flash_1.txt
#SBATCH -L scratch
#DW jobdw capacity=1289GiB access_mode=striped type=scratch
#DW jobdw capacity=1289GiB access_mode=private type=scratch

RUNS=(1) # Number of runs
OUTDIR=/global/cscratch1/sd/khl7265/FS_64_8M/flash
NN=${SLURM_NNODES}
let NP=NN*32

echo "mkdir -p ${OUTDIR}"
mkdir -p ${OUTDIR}

for i in ${RUNS[@]}
do
    # Ncmpio
    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*
    
    srun -n ${NP} ./flash_benchmark_io ${OUTDIR}/flash_ blocking coll

    echo "#%$: io_driver: ncmpi"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
    
    echo '-----+-----++------------+++++++++--+---'

    # BB LPP P

    echo "#%$: io_driver: bb_lpp_private"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*

    export PNETCDF_HINTS="nc_burst_buf_driver=enable;nc_burst_buf_del_on_close=disable;nc_burst_buf_overwrite=enable;nc_burst_buf_dirname=${DW_JOB_PRIVATE}"
    srun -n ${NP} ./flash_benchmark_io ${OUTDIR}/flash_ blocking coll
    unset PNETCDF_HINTS

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
    echo "ls -lah ${DW_JOB_PRIVATE}"
    ls -lah ${DW_JOB_PRIVATE}

    echo '-----+-----++------------+++++++++--+---'

    # BB LPP S

    echo "#%$: io_driver: bb_lpn_striped"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*

    export PNETCDF_HINTS="nc_burst_buf_driver=enable;nc_burst_buf_del_on_close=disable;nc_burst_buf_overwrite=enable;nc_burst_buf_dirname=${DW_JOB_STRIPED}"
    srun -n ${NP} ./flash_benchmark_io ${OUTDIR}/flash_ blocking coll
    unset PNETCDF_HINTS

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
    echo "ls -lah ${DW_JOB_STRIPED}"
    ls -lah ${DW_JOB_STRIPED}

    echo '-----+-----++------------+++++++++--+---'

    # BB LPN S

    echo "#%$: io_driver: bb_lpn_striped"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*

    export PNETCDF_HINTS="nc_burst_buf_driver=enable;nc_burst_buf_del_on_close=disable;nc_burst_buf_overwrite=enable;nc_burst_buf_sharedlog=enable;nc_burst_buf_dirname=${DW_JOB_STRIPED}"
    srun -n ${NP} ./flash_benchmark_io ${OUTDIR}/flash_ blocking coll
    unset PNETCDF_HINTS

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}

    echo '-----+-----++------------+++++++++--+---'

    # Staging

    echo "#%$: io_driver: stage"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*
    echo "rm -f ${DW_JOB_STRIPED}/*"
    rm -f ${DW_JOB_STRIPED}/*

    export stageout_burst_buf_path="${DW_JOB_STRIPED}"
    export stageout_pfs_path="${OUTDIR}"
    srun -n ${NP} ./flash_benchmark_io ${DW_JOB_STRIPED}/flash_ blocking coll
    unset stageout_burst_buf_path
    unset stageout_pfs_path

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
    echo "ls -lah ${DW_JOB_STRIPED}"
    ls -lah ${DW_JOB_STRIPED}

    echo '-----+-----++------------+++++++++--+---'

    # LogFS
    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*
    
    export PNETCDF_HINTS="logfs_replayonclose=true;logfs_info_logbase=${DW_JOB_PRIVATE};logfs_flushblocksize=268435456"
    srun -n ${NP} ./flash_benchmark_io_logfs ${OUTDIR}/flash_ blocking coll
    unset PNETCDF_HINTS

    echo "#%$: io_driver: logfs"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
    
    echo '-----+-----++------------+++++++++--+---'

    # Data Elevator
    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*
    
    srun -n ${NP} ./flash_benchmark_io_de ${OUTDIR}/flash_ blocking coll

    echo "#%$: io_driver: de"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
    
    echo '-----+-----++------------+++++++++--+---'

echo "BB Info: "
module load dws
sessID=$(dwstat sessions | grep $SLURM_JOBID | awk '{print $1}')
echo "session ID is: "${sessID}
instID=$(dwstat instances | grep $sessID | awk '{print $1}')
echo "instance ID is: "${instID}
echo "fragments list:"
echo "frag state instID capacity gran node"
dwstat fragments | grep ${instID}


