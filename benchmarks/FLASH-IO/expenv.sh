#!/bin/bash
#WD=$4
#module load datawarp 
#cc stageout.c -o stageout `pkg-config --cflags --libs cray-datawarp`
#./genscript.sh 1 32 
#./genscript.sh 2 64 
#./genscript.sh 4 128 
#./genscript.sh 8 256 
#./genscript.sh 16 512 
#./genscript.sh 32 1024 
./genscript.sh 64 2048
./genscript.sh 128 4096
#./genscript.sh 256 8192

