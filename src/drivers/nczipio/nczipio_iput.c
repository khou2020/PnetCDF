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

static inline int
nczipioi_init_put_req( NC_zip *nczipp,
                        NC_zip_req *req,
                        int        varid,
                        MPI_Offset *start,
                        MPI_Offset *count,
                        MPI_Offset *stride, 
                        const void *xbuf,
                        const void *buf) {
    int err;
    int i, j, k, l;
    int *tsize, *tssize, *tstart;   // Size for sub-array type
    int *cstart, *cend, *citr; // Bounding box for chunks overlapping my own write region
    int overlapsize, packoff;
    MPI_Datatype ptype; // Pack datatype

    // Record request
    req.start = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim);
    memcpy(req.start, start, sizeof(MPI_Offset) * varp->ndim);
    req.count = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim);
    memcpy(req.count, count, sizeof(MPI_Offset) * varp->ndim);
    if (stride != NULL){
        req.stride = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim);
        memcpy(req.stride, stride, sizeof(MPI_Offset) * varp->ndim);
    }

    req.varid = varid;
    req.buf = buf;
    req.xbuf = xbuf;
    req.nreq = 1;

    return NC_NOERR;
}

int
nczipioi_iput_var(NC_zip        *nczipp,
              int               varid,
              MPI_Offset        *starts,
              MPI_Offset        *counts,
              const void        *xbuf,
              const void        *buf,
              int               *reqid)
{
    int err;
    int req_id;
    NC_zip_req req;

    err = nczipioi_init_put_req(nczipp, &req, varid, start, count, stride, xbuf, buf);

    // Add to req list
    nczipioi_list_add(&(nczipp->putlist), &req_id);
    ncadp->putlist.reqs[req_id] = req;
    
    if (reqid != NULL){
        *reqid = req_id;
    }

    return NC_NOERR;
}

static inline int
nczipioi_init_put_varn_req( NC_zip *nczipp,
                        NC_zip_req *req,
                        int        varid,
                        int        nreq,
                        MPI_Offset *const*starts,
                        MPI_Offset *const*counts, 
                        const void *xbuf,
                        const void *buf) {
    int i;

    // Record request
    req.starts = (MPI_Offset**)NCI_Malloc(sizeof(MPI_Offset*) * nreq);
    req.starts[0] = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim * nreq);
    for(i = 0; i < nreq; i++){
        req.starts[i] = req.starts[0] + i * varp->ndim;
        memcpy(req.starts[i], starts[i], sizeof(MPI_Offset) * varp->ndim);
    }
    req.counts = (MPI_Offset**)NCI_Malloc(sizeof(MPI_Offset*) * nreq);
    req.counts[0] = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim * nreq);
    for(i = 0; i < nreq; i++){
        req.counts[i] = req.counts[0] + i * varp->ndim;
        memcpy(req.counts[i], counts[i], sizeof(MPI_Offset) * varp->ndim);
    }

    req.varid = varid;
    req.buf = buf;
    req.xbuf = xbuf;
    req.nreq = nreq;

    return NC_NOERR;
}

int
nczipioi_iput_varn(NC_zip        *nczipp,
              int               varid,
              int               nreq,
              MPI_Offset        *starts,
              MPI_Offset        *counts,
              const void        *xbuf,
              const void        *buf,
              int               *reqid)
{
    int err;
    int req_id;
    NC_zip_req req;

    if (nreq > 1){
        err = nczipioi_init_put_varn_req(nczipp, &req, varid, nreq, start, count, stride, xbuf, buf);
    }
    else{
        err = nczipioi_init_put_var_req(nczipp, &req, varid, starts[0], counts[0], NULL, xbuf, buf);
    }

    // Add to req list
    nczipioi_list_add(&(nczipp->putlist), &req_id);
    ncadp->putlist.reqs[req_id] = req;
    
    if (reqid != NULL){
        *reqid = req_id;
    }

    return NC_NOERR;
}
