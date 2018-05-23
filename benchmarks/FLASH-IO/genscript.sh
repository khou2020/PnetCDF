#!/bin/bash
#WD=$4
WD="/global/cscratch1/sd/khl7265/FS_64_1M/flash"

for i in blocking nonblocking
do
    for j in coll indep
    do
        m4 -D EXPDIR=${WD} -D NNODE=$1 -D NPROC=$2 -D IOMODE=$m bb.m4 > bb_${i}_${j}_$2.sl
        m4 -D EXPDIR=${WD} -D NNODE=$1 -D NPROC=$2 -D IOMODE=$m ncmpi.m4 > ncmpi_${i}_${j}_$2.sl
        m4 -D EXPDIR=${WD} -D NNODE=$1 -D NPROC=$2 -D IOMODE=$m stage.m4 > stage_${i}_${j}_$2.sl
    done
done
