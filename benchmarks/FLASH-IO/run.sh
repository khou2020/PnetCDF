#!/bin/bash
#COBALT -t 3
#COBALT -n 1
#COBALT --attrs mcdram=cache:numa=quad:ssds=required:ssd_size=16
#COBALT -A ecp-testbed-01
#COBALT -q debug-flat-quad
#COBALT -o flash_1.txt
#COBALT -e flash_1.txt

echo "Starting Cobalt job script"

RUNS=(1) # Number of runs
OUTDIR=/projects/radix-io/khou/FS_64_8M/flash
BBDIR=/local/scratch
PPN=4
NN=${COBALT_JOBSIZE}
let NP=NN*PPN

export n_nodes=$COBALT_JOBSIZE
export n_mpi_ranks_per_node=${PPN}
export n_mpi_ranks=$(($n_nodes * $n_mpi_ranks_per_node))
export n_openmp_threads_per_rank=1
export n_hyperthreads_per_core=1
export n_hyperthreads_skipped_between_ranks=7

echo "mkdir -p ${OUTDIR}"
mkdir -p ${OUTDIR}

for i in ${RUNS[@]}
do
    for u in blocking nonblocking
    do
        for v in coll indep
        do
            # Ncmpi
            if [ "x${v}" = "xcoll" ]; then
                echo "rm -f ${OUTDIR}/*"
                rm -f ${OUTDIR}/*
                
                aprun -n ${NP} -N ${PPN} ./flash_benchmark_io ${OUTDIR}/flash_ ${u} ${v}

                echo "#%$: io_driver: ncmpi"
                echo "#%$: platform: theta"
                echo "#%$: number_of_nodes: ${NN}"
                echo "#%$: io_mode: ${u}_${v}"

                echo "ls -lah ${OUTDIR}"
                ls -lah ${OUTDIR}
                
                echo '-----+-----++------------+++++++++--+---'
            fi

            # Dw
            if [ "x${u}" = "xblocking" ] && [ "x${v}" = "xcoll" ]; then
                echo "rm -f ${OUTDIR}/*"
                rm -f ${OUTDIR}/*

                aprun -n ${NP} -N ${PPN} -e PNETCDF_HINTS="nc_bb_driver=enable;nc_bb_del_on_close=disable;nc_bb_overwrite=enable;nc_bb_dirname=${BBDIR}" ./flash_benchmark_io ${OUTDIR}/flash_ ${u} ${v}

                echo "#%$: io_driver: bb"
                echo "#%$: platform: theta"
                echo "#%$: number_of_nodes: ${NN}"
                echo "#%$: io_mode: ${u}_${v}"

                echo "ls -lah ${OUTDIR}"
                ls -lah ${OUTDIR}
                if ["${NP}" -lt 33]; then
                    echo "ls -lah ${BBDIR}"
                    ls -lah ${BBDIR}
                fi
                            
                echo '-----+-----++------------+++++++++--+---'
            fi

            # Dw shared
            if [ "x${u}" = "xblocking" ] && [ "x${v}" = "xcoll" ]; then
                echo "rm -f ${OUTDIR}/*"
                rm -f ${OUTDIR}/*
                
                aprun -n ${NP} -N ${PPN} -e PNETCDF_HINTS="nc_bb_driver=enable;nc_bb_del_on_close=disable;nc_bb_overwrite=enable;nc_bb_sharedlog=enable;nc_bb_dirname=${BBDIR}" ./flash_benchmark_io ${OUTDIR}/flash_ ${u} ${v}

                echo "#%$: io_driver: bb_shared"
                echo "#%$: platform: theta"
                echo "#%$: number_of_nodes: ${NN}"
                echo "#%$: io_mode: ${u}_${v}"

                echo "ls -lah ${OUTDIR}"
                ls -lah ${OUTDIR}
                if ["${NP}" -lt 33]; then
                    echo "ls -lah ${BBDIR}"
                    ls -lah ${BBDIR}
                fi
                            
                echo '-----+-----++------------+++++++++--+---'
            fi
        done
    done
done
