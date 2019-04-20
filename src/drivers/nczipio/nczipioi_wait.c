/*
 *  Copyright (C) 2019, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the following PnetCDF APIs.
 *
 * ncmpi_get_var<kind>_all()        : dispatcher->get_var()
 * ncmpi_put_var<kind>_all()        : dispatcher->put_var()
 * ncmpi_get_var<kind>_<type>_all() : dispatcher->get_var()
 * ncmpi_put_var<kind>_<type>_all() : dispatcher->put_var()
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <mpi.h>

#include <pnc_debug.h>
#include <common.h>
#include <nczipio_driver.h>
#include "nczipio_internal.h"


/* Out drive currently can handle only one variable at a time
 * We pack all request as a large varn request
 */
int nczipioi_wait_put_reqs(NC_zip *nczipp, int nreq, int *reqids, int *stats){
    int err;
    int i;
    int nvar;
    int *varids;
    int *dirty;  // Number of reqids in each variable
    NC_zip_req *req;

    // Build a skip list of touched vars
    dirty = (int*)NCI_Malloc(sizeof(int) * nczipp->vars.cnt);
    memset(dirty, 0, sizeof(int) * nczipp->vars.cnt);
    for(i = 0; i < nreq; i++){
        req = nczipp->putlist.reqs + reqids[i];
        dirty[req->varid] = 1;
    }
    for(i = 0; i < nczipp->vars.cnt; i++){
        if (dirty[i]){
            nvar++;
        }
    }
    varids = (int*)NCI_Malloc(sizeof(int) * nvar);
    nvar = 0;
    for(i = 0; i < nczipp->vars.cnt; i++){
        if (dirty[i]){
            varids[nvar++] = i;
        }
    }

    // Perform collective buffer
    if (nczipp->comm_unit == NC_ZIP_COMM_CHUNK){
        nczipioi_iput_cb_chunk(nczipp, nreq, reqids, stats);
    }
    else{
        nczipioi_iput_cb_proc(nczipp, nreq, reqids, stats);
    }

    // Perform I/O for comrpessed variables
    nczipioi_save_nvar(nczipp, nvar, varids);

    // Free buffers
    NCI_Free(varids);
    NCI_Free(dirty);

    return NC_NOERR;
}

/* Out drive currently can handle only one variable at a time
 * We pack all request as a large varn request
 */
int nczipioi_wait_get_reqs(NC_zip *nczipp, int nreq, int *reqids, int *stats){
    int err;
    int i;
    int nvar;
    int *varids;
    int *dirty;  // Number of reqids in each variable
    NC_zip_req *req;

    // Build a skip list of touched vars
    dirty = (int*)NCI_Malloc(sizeof(int) * nczipp->vars.cnt);
    memset(dirty, 0, sizeof(int) * nczipp->vars.cnt);
    for(i = 0; i < nreq; i++){
        req = nczipp->getlist.reqs + reqids[i];
        dirty[req->varid] = 1;
    }
    for(i = 0; i < nczipp->vars.cnt; i++){
        if (dirty[i]){
            nvar++;
        }
    }
    varids = (int*)NCI_Malloc(sizeof(int) * nvar);
    nvar = 0;
    for(i = 0; i < nczipp->vars.cnt; i++){
        if (dirty[i]){
            varids[nvar++] = i;
        }
    }

    // Perform I/O for comrpessed variables
    nczipioi_load_nvar(nczipp, nvar, varids);

    // Perform collective buffer
    if (nczipp->comm_unit == NC_ZIP_COMM_CHUNK){
        nczipioi_iget_cb_chunk(nczipp, nreq, reqids, stats);
    }
    else{
        nczipioi_iget_cb_proc(nczipp, nreq, reqids, stats);
        //nczipioi_iget_cb_chunk(nczipp, nreq, reqids, stats);
    }

    // Free buffers
    NCI_Free(varids);
    NCI_Free(dirty);
}

int
nczipioi_wait(NC_zip *nczipp, int nreqs, int *reqids, int *stats, int reqMode){
    int err;
    int i;
    int nput = 0, nget = 0;
    int *putreqs = NULL, *getreqs = NULL;
    int *putstats = NULL, *getstats = NULL;

    if (nreqs == NC_REQ_ALL || nreqs == NC_PUT_REQ_ALL){
        nput = nczipp->putlist.nused;
        putreqs = (int*)NCI_Malloc(sizeof(int) * nput);
        memcpy(putreqs, nczipp->putlist.ids, nput * sizeof(int));
    }
    if(nreqs == NC_REQ_ALL || nreqs == NC_GET_REQ_ALL){
        nget = nczipp->getlist.nused;
        getreqs = (int*)NCI_Malloc(sizeof(int) * nget);
        memcpy(getreqs, nczipp->getlist.ids, nget * sizeof(int));
    }

    if (nreqs > 0){
        // Count number of get and put requests
        for(i = 0; i < nreqs; i++){
            if (reqids[i] & 1){
                nput++;
            }
        }

        // Allocate buffer
        nget = nreqs - nput;
        putreqs = (int*)NCI_Malloc(sizeof(int) * nput);
        getreqs = (int*)NCI_Malloc(sizeof(int) * nget);
        
        // Build put and get req list
        nput = nget = 0;
        for(i = 0; i < nreqs; i++){
            if (reqids[i] & 1){
                putreqs[nput++] = reqids[i] >> 1;
            }
            else{
                getreqs[nget++] = reqids[i] >> 1;
            }
        }
    }

    if (stats != NULL){
        putstats = (int*)NCI_Malloc(sizeof(int) * nput);
        getstats = (int*)NCI_Malloc(sizeof(int) * nget);
    }
    else{
        putstats = NULL;
        getstats = NULL;
    }

    if (nczipp->recdim >= 0){
        MPI_Allreduce(MPI_IN_PLACE, &(nczipp->recsize), 1, MPI_LONG_LONG, MPI_MAX, nczipp->comm);   // Sync number of recs
        // Expand all variables
        for(i = 0; i < nczipp->vars.cnt; i++){
            if (nczipp->vars.data[i].isrec && (nczipp->vars.data[i].dimsize[0] < nczipp->recsize)){
                nczipioi_var_resize(nczipp, nczipp->vars.data + i);
            }
        }
    }

    if (nput > 0){
        nczipioi_wait_put_reqs(nczipp, nput, putreqs, putstats);
    }
    
    if (nget > 0){
        nczipioi_wait_get_reqs(nczipp, nget, getreqs, getstats);
    }

    // Assign stats
    if (stats != NULL){
        nput = nget = 0;
        for(i = 0; i < nreqs; i++){
            if (reqids[i] & 1){
                stats[i] = putstats[nput++];
            }
            else{
                stats[i] = getstats[nget++];
            }
        }

        NCI_Free(putstats);
        NCI_Free(getstats);
    }
    
    // Remove from req list
    for(i = 0; i < nput; i++){
        nczipioi_req_list_remove(&(nczipp->putlist), putreqs[i]);
    }
    for(i = 0; i < nget; i++){
        nczipioi_req_list_remove(&(nczipp->getlist), getreqs[i]);
    }

    NCI_Free(putreqs);
    NCI_Free(getreqs);
    
    return NC_NOERR;
}
