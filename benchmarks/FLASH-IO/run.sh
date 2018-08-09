#!/bin/bash
#COBALT -t 10
#COBALT -n 1
#COBALT --attrs mcdram=cache:numa=quad:ssds=required:ssd_size=16
#COBALT -A ecp-testbed-01
#COBALT -q debug-flat-quad
#COBALT -o flash_1.txt
#COBALT -e flash_1.txt

echo "Starting Cobalt job script"

export n_nodes=$COBALT_JOBSIZE
export n_mpi_ranks_per_node=${PPN}
export n_mpi_ranks=$(($n_nodes * $n_mpi_ranks_per_node))
export n_openmp_threads_per_rank=1
export n_hyperthreads_per_core=1
export n_hyperthreads_skipped_between_ranks=7

RUNS=(1) # Number of runs
OUTDIR=/projects/radix-io/khou/FS_64_8M/flash
BBDIR=/local/scratch
PPN=4
#NN=16
NN=${COBALT_JOBSIZE}
let NP=NN*PPN
TL=300

echo "mkdir -p ${OUTDIR}"
mkdir -p ${OUTDIR}

TSTARTTIME=`date +%s.%N`

for i in ${RUNS[@]}
do
    # Ncmpio

    echo "========================== NCMPI =========================="
    >&2 echo "========================== NCMPI =========================="

    echo "#%$: io_driver: ncmpi"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*
    
    STARTTIME=`date +%s.%N`

    aprun -n ${NP} -N ${PPN} -t ${TL} ./flash_benchmark_io ${OUTDIR}/flash_ blocking coll

    ENDTIME=`date +%s.%N`
    TIMEDIFF=`echo "$ENDTIME - $STARTTIME" | bc | awk -F"." '{print $1"."$2}'`

    echo "#%$: exe_time: $TIMEDIFF"

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
    
    echo '-----+-----++------------+++++++++--+---'

    # Ncmpio NB
    
    echo "========================== NCMPI NB =========================="
    >&2 echo "========================== NCMPI NB =========================="

    echo "#%$: io_driver: ncmpi"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: nonblocking_coll"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*
    
    STARTTIME=`date +%s.%N`

    aprun -n ${NP} -N ${PPN} -t ${TL} ./flash_benchmark_io ${OUTDIR}/flash_ nonblocking coll

    ENDTIME=`date +%s.%N`
    TIMEDIFF=`echo "$ENDTIME - $STARTTIME" | bc | awk -F"." '{print $1"."$2}'`

    echo "#%$: exe_time: $TIMEDIFF"

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
    
    echo '-----+-----++------------+++++++++--+---'

    # BB LPP P
    
    echo "========================== BB LPP P =========================="
    >&2 echo "========================== BB LPP P =========================="

    echo "#%$: io_driver: bb_lpp_private"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*

    STARTTIME=`date +%s.%N`
    
    aprun -n ${NP} -N ${PPN} -t ${TL} -e PNETCDF_HINTS="nc_burst_buf=enable;nc_burst_buf_del_on_close=disable;nc_burst_buf_overwrite=enable;nc_burst_buf_dirname=${BBDIR}" ./flash_benchmark_io ${OUTDIR}/flash_ blocking coll
    
    ENDTIME=`date +%s.%N`
    TIMEDIFF=`echo "$ENDTIME - $STARTTIME" | bc | awk -F"." '{print $1"."$2}'`

    echo "#%$: exe_time: $TIMEDIFF"

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}

    echo '-----+-----++------------+++++++++--+---'

    # BB LPN S
    
    echo "========================== BB LPN S =========================="
    >&2 echo "========================== BB LPN S =========================="

    echo "#%$: io_driver: bb_lpn_private"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*

    STARTTIME=`date +%s.%N`

    aprun -n ${NP} -N ${PPN} -t ${TL} -e PNETCDF_HINTS="nc_burst_buf=enable;nc_burst_buf_del_on_close=disable;nc_burst_buf_shared_logs=enable;nc_burst_buf_overwrite=enable;nc_burst_buf_dirname=${BBDIR}" ./flash_benchmark_io ${OUTDIR}/flash_ blocking coll

    ENDTIME=`date +%s.%N`
    TIMEDIFF=`echo "$ENDTIME - $STARTTIME" | bc | awk -F"." '{print $1"."$2}'`

    echo "#%$: exe_time: $TIMEDIFF"

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}

    echo '-----+-----++------------+++++++++--+---'

    # LogFS
        
    echo "========================== Logfs =========================="
    >&2 echo "========================== Logfs =========================="

    echo "#%$: io_driver: logfs"
    echo "#%$: number_of_nodes: ${NN}"
    echo "#%$: number_of_proc: ${NP}"
    echo "#%$: io_mode: blocking_coll"

    echo "rm -f ${OUTDIR}/*"
    rm -f ${OUTDIR}/*
    
    export PNETCDF_HINTS="logfs_replayonclose=true;logfs_info_logbase=${DW_JOB_PRIVATE};logfs_flushblocksize=268435456"

    STARTTIME=`date +%s.%N`

    aprun -n ${NP} -N ${PPN} -t ${TL} ./flash_benchmark_io_logfs logfs:${OUTDIR}/flash_ blocking coll

    ENDTIME=`date +%s.%N`
    TIMEDIFF=`echo "$ENDTIME - $STARTTIME" | bc | awk -F"." '{print $1"."$2}'`

    unset PNETCDF_HINTS

    echo "#%$: exe_time: $TIMEDIFF"

    echo "ls -lah ${OUTDIR}"
    ls -lah ${OUTDIR}
    
    echo '-----+-----++------------+++++++++--+---'

done

ENDTIME=`date +%s.%N`
TIMEDIFF=`echo "$ENDTIME - $TSTARTTIME" | bc | awk -F"." '{print $1"."$2}'`
echo "-------------------------------------------------------------"
echo "total_exe_time: $TIMEDIFF"
echo "-------------------------------------------------------------"
