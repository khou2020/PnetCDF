define(`SCRIPT', dnl
`dnl
#!/bin/bash
#SBATCH -p regular
#SBATCH -N $2
#SBATCH -C haswell
#SBATCH -t 00:10:00
#SBATCH -o ncmpi_$4_$5_$3.txt

mkdir -p $1
mkdir -p $1/ncmpi_$4_$5_$3
for i in 1 2 3
do
    echo "Round " $i ":"
    rm f $1/ncmpi_$4_$5_$3/*
    srun -n $3 ./flash_benchmark_io $1/ncmpi_$4_$5_$3/flash_
    echo "io_driver: NCMPIO"
    echo "n_nodes: $2" 
    echo 'ls -lah $1/ncmpi_$4_$5_$3'
    ls -lah $1/ncmpi_$4_$5_$3
    echo '-----+-----++------------+++++++++--+---'
done
echo '--++---+----+++-----++++---+++--+-++--+---'
')dnl
SCRIPT(EXPDIR,NNODE,NPROC,IOMODEA,IOMODEB)
