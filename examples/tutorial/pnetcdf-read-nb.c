/*********************************************************************
 *
 *  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id$ */

/* simple demonstration of pnetcdf:
 * knowing nothing about the file, read in the variables. 
 *
 * This example demonstrates the non-blocking read interface */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pnetcdf.h>

static void handle_error(int status, int lineno)
{
    fprintf(stderr, "Error at line %d: %s\n", lineno, ncmpi_strerror(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char **argv) {

    int i, j, rank, nprocs, ret;
    int ncfile, ndims, nvars, ngatts, unlimited, var_ndims, var_natts;;
    MPI_Offset *dim_sizes, var_size, *start, *count;
    int *requests, *statuses, dimids[NC_MAX_VAR_DIMS], **data; 
    char varname[NC_MAX_NAME+1];
    nc_type type;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s filename\n", argv[0]);
        MPI_Finalize();
        exit(-1);
    }

    ret = ncmpi_open(MPI_COMM_WORLD, argv[1], NC_NOWRITE, MPI_INFO_NULL,
                     &ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* reader knows nothing about dataset, but we can interrogate with query
     * routines: ncmpi_inq tells us how many of each kind of "thing"
     * (dimension, variable, attribute) we will find in the file  */

    /* no commnunication needed after ncmpi_open: all processors have a cached
     * veiw of the metadata once ncmpi_open returns */

    ret = ncmpi_inq(ncfile, &ndims, &nvars, &ngatts, &unlimited);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* we do not really need the name of the dimension or the variable for
     * reading in this example.  we could, in a different example, take the
     * name of a variable on the command line and read just that one */

    dim_sizes = (MPI_Offset*) calloc(ndims, sizeof(MPI_Offset));
    /* netcdf dimension identifiers are allocated sequentially starting
     * at zero; same for variable identifiers */
    for(i=0; i<ndims; i++)  {
        ret = ncmpi_inq_dimlen(ncfile, i, &(dim_sizes[i]) );
        if (ret != NC_NOERR) handle_error(ret, __LINE__);
    }

    requests = (int*) calloc(nvars, sizeof(int));
    statuses = (int*) calloc(nvars, sizeof(int));

    data = (int**) calloc(nvars, sizeof(int*));

    for(i=0; i<nvars; i++) { 
        /* much less coordination in this case compared to rank 0 doing all
         * the i/o: everyone already has the necessary information */
        ret = ncmpi_inq_var(ncfile, i, varname, &type, &var_ndims, dimids,
                            &var_natts);
        if (ret != NC_NOERR) handle_error(ret, __LINE__);

        start = (MPI_Offset*) calloc(var_ndims, sizeof(MPI_Offset));
        count = (MPI_Offset*) calloc(var_ndims, sizeof(MPI_Offset));

        /* we will simply decompose along one dimension.  Generally the
         * application has some algorithim for domain decomposistion.  Note
         * that data decomposistion can have an impact on i/o performance.
         * Often it's best just to do what is natural for the application,
         * but something to consider if performance is not what was
         * expected/desired */

        start[0] = (dim_sizes[dimids[0]]/nprocs)*rank;
        count[0] = (dim_sizes[dimids[0]]/nprocs);
        var_size = count[0];

        for (j=1; j<var_ndims; j++) {
            start[j] = 0;
            count[j] = dim_sizes[dimids[j]];
            var_size *= count[j];
        }

        switch(type) {
            case NC_INT:
                data[i] = (int*) calloc(var_size, sizeof(int));
                /* as with the writes, this call is independent: we
                 * will do any coordination (if desired) in a
                 * subsequent ncmpi_wait_all() call */
                ret = ncmpi_iget_vara(ncfile, i, start, count, data[i],
                                      var_size, MPI_INT, &requests[i]);
                if (ret != NC_NOERR) handle_error(ret, __LINE__);
                break;
            default:
                /* we can do this for all the known netcdf types but this
                 * example is already getting too long  */
                fprintf(stderr, "unsupported NetCDF type \n");
        }

        free(start);
        free(count);
    }

    ret = ncmpi_wait_all(ncfile, nvars, requests, statuses);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* check status of each nonblocking call */
    for (i=0; i<nvars; i++)
        if (statuses[i] != NC_NOERR) handle_error(statuses[i], __LINE__);

    /* now that the ncmpi_wait_all has returned, the caller can do stuff with
     * the buffers passed in to the non-blocking operations.  The buffer resue
     * rules are similar to MPI non-blocking messages */

    for (i=0; i<nvars; i++) {
        if (data[i] != NULL) free(data[i]);
    }
    free(data);
    free(dim_sizes);
    free(requests);
    free(statuses);

    ret = ncmpi_close(ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    MPI_Finalize();
    return 0;
}

/*
 *vim: ts=8 sts=4 sw=4 noexpandtab */
