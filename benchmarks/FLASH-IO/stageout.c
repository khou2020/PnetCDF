#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <mpi.h>
#include <datawarp.h>

/* build with: *   gcc dirstageandwait.c -o dirstageandwait `pkg-config --cflags \ *     
--libs cray-datawarp` */

int main(int argc, char **argv) {
    int ret;   
    int comp, pend, defer, fail;
    double st, et;

    if (argc < 3) {      
        printf("Error: Expected usage:  \n%s src dst\n", argv[0]);
        return 0;  
    }

    MPI_Init(&argc, &argv);

    st = MPI_Wtime();

    /* perform stage out */   
    ret = bb_stage_directory_out(argv[1], argv[2], BB_STAGE_IMMEDIATE);    
    if (ret != 0) {
        printf("%s: bb_stage_file error - %d %s\n", argv[0], ret, strerror(-ret));
        return ret;
    }

    /* wait stage out */  
    ret = bb_wait_directory_stage(argv[1]);
    if (ret != 0) {
        printf("%s: bb_wait_dir_stage error %d %s\n", argv[0], ret, strerror(-ret));
        return ret;
    }

    et = MPI_Wtime();

    /* query final stage state of bb target */
    ret = bb_query_directory_stage(argv[1], &comp, &pend, &defer, &fail);
    if (ret != 0) {
        printf("%s: query_file_stage error %d %s\n", argv[0], ret, strerror(-ret));
        return ret;
    }
    if (comp > 0){
        printf("Staging done\n");
    }
    else{
        printf("Stagging failed\n");
    }
    
    printf("Time stagging: %17.2f sec\n", (et  -st));

    MPI_Finalize();

    return 0;
}

