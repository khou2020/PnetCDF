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

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

static int get_chunk_idx(NC_zip_var *varp, int* cord){
    int i, ret;
    
    ret = cord[0];
    for(i = 1; i < varp->ndim; i++){
        ret = ret * varp->chunkdim[i - 1] + cord[i];
    }

    return ret;
}

static int get_chunk_cord(NC_zip_var *varp, int idx, int* cord){
    int i, ret;
    
    ret = cord[0];
    for(i = 1; i < varp->ndim; i++){
        ret = ret * varp->chunkdim[i - 1] + cord[i];
    }

    for(i = varp->ndim - 1; i > 0; i--){
        cord[i] = idx % varp->chunkdim[i - 1];
        idx /= varp->chunkdim[i - 1];
    }
    cord[0] = idx;

    return 0;
}

static int get_chunk_overlap(NC_zip_var *varp, int* cord, const MPI_Offset *start, const MPI_Offset *count, const MPI_Offset *stride, int *ostart, int *ocount){
    int i, ret;

    for(i = 0; i < varp->ndim; i++){
        ostart[i] = max(start[i], cord[i] * varp->chunkdim[i]);
        ocount[i] = (min(start[i] + count[i] * stride[i], (cord[i] + 1) * varp->chunkdim[i]) - ostart[i]) / stride[i];
        if (ocount[i] < 0){
            ocount[i] = 0;
        }
    }

    return 0;
}

int
nczipioi_init_put_req( NC_zip *nczipp,
                        NC_zip_req *req,
                        NC_zip_var *varp,
                        MPI_Offset *start,
                        MPI_Offset *count,
                        MPI_Offset *stride, 
                        const void *xbuf,
                        const void *buf) {
    int err;
    int i, j, k, l;
    int *tsize, *tssize, *tstart;   // Size for sub-array type
    int *cstart, *cend, *citr; // Bounding box for chunks overlapping my own write region
    int *wcnt_local, *wcnt_all;   // Number of processes that writes to each chunk
    char *sbuf_base, *sbuf_cur; // Send buffer, exactly the same size as buf
    int put_size_total, put_size;  // Total size of buf and size of data of a single req
    char **bufs; // Location of data in buf for every req
    int overlapsize, packoff;
    MPI_Datatype ptype; // Pack datatype
    MPI_Request *sreqs, *rreqs;    // Send and recv req
    char **sbufs, **rbufs;   // Send and recv buffer
    MPI_Status *sstats, *rstats;    // Send and recv status
    int nsend, nrecv;
    int *rsizes;
    MPI_Message rmsg;   // Receive message
    int *zsizes, *zsizes_all, *zoffs;
    MPI_Offset zstart, zcount;
    char **zbufs;
    int zdimid;
    char name[128]; // Name of objects

    // Allocate buffering for location of data in buf for every req
    bufs = (char**)NCI_Malloc(sizeof(char*) * nreq);

    // Allocate buffering for write count
    wcnt_local = (int*)NCI_Malloc(sizeof(int) * varp->nchunks);
    wcnt_all = (int*)NCI_Malloc(sizeof(int) * varp->nchunks);

    // Allocate buffering for overlaping index
    tsize = (int*)NCI_Malloc(sizeof(int) * varp->ndim);
    tssize = (int*)NCI_Malloc(sizeof(int) * varp->ndim);
    tstart = (int*)NCI_Malloc(sizeof(int) * varp->ndim);

    // Starting, ending, current chunk position
    cstart = (int*)NCI_Malloc(sizeof(int) * varp->ndim);
    citr = (int*)NCI_Malloc(sizeof(int) * varp->ndim);
    cend = (int*)NCI_Malloc(sizeof(int) * varp->ndim);

    // Record request
    req.start = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim);
    memcpy(req.start, start, sizeof(MPI_Offset) * varp->ndim);
    req.count = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim);
    memcpy(req.count, count, sizeof(MPI_Offset) * varp->ndim);
    if (stride != NULL){
        req.stride = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim);
        memcpy(req.stride, stride, sizeof(MPI_Offset) * varp->ndim);
    }

    req.nsend = nczipioi_chunk_itr_init(varp, start, count, stride, c start, cend, citr);

    req.widx = (int*)NCI_Malloc(sizeof(int) * req.nsend);
    req.sbuf = (char**)NCI_Malloc(sizeof(char*) * req.nsend);
    req.sreqs = (MPI_Request*)NCI_Malloc(sizeof(MPI_Request) * req.nsend);
    req.sstats = (MPI_Status*)NCI_Malloc(sizeof(MPI_Status) * req.nsend);

    req.buf = buf;
    req.xbuf = xbuf;
    req.nreq = 1;

    //Calculate local write count, we calculate offset and size of each req by the way
    memset(wcnt_local, 0, sizeof(int) * nczipp->np);

    // Iterate through chunk
    req.nsend = 0;  // Previous estimate contains our own chunks. Now, we count real chunk
    do{
        // Chunk index
        i = get_chunk_idx(varp, citr);

        if (varp->chunk_owner[i] != nczipp->rank){
            // Overlapping size
            get_chunk_overlap(varp, citr, start, count, tstart, tssize);
            overlapsize = varp->esize;
            for(j = 0; j < varp->ndim; j++){
                overlapsize *= tssize[j];                     
            }
            printf("overlapsize = %d\n", overlapsize); fflush(stdout);
            
            // Pack type
            for(j = 0; j < varp->ndim; j++){
                tstart[j] -= start[j];
                tsize[j] = (int)count[j];
            }
            MPI_Type_create_subarray(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, etype, &ptype);
            MPI_Type_commit(&ptype);
            
            // Allocate buffer
            req.sbuf[req.nsend] = (char*)NCI_Malloc(overlapsize + sizeof(int) * (varp->ndim * 2 + 1));

            // Pack data
            packoff = 0;
            MPI_Pack(starts[i], varp->ndim, MPI_INT, req.sbuf[req.nsend], packoff + sizeof(int) * varp->ndim, &packoff, nczipp->comm);
            MPI_Pack(counts[i], varp->ndim, MPI_INT, req.sbuf[req.nsend], packoff + sizeof(int) * varp->ndim, &packoff, nczipp->comm);
            MPI_Pack(bufs[i], 1, ptype, req.sbuf[req.nsend], packoff + overlapsize, &packoff, nczipp->comm);

            // Free packtype
            MPI_Type_free(&ptype);

            // Send data to owner
            MPI_Isend(req.sbuf[req.nsend], packoff, MPI_BYTE, varp->chunk_owner[i], i, nczipp->comm, req.sreqs + req.nsend);

            // Record chunk write
            req.widx[req.nsend] = i;
            
            //Count send request
            req.nsend++;
        }
    // Move to next chunk        
    } while (nczipioi_chunk_itr_next(varp, start, count, stride, c start, cend, citr));

    // Free buffers
    NCI_Free(tsize);
    NCI_Free(tssize);
    NCI_Free(tstart);

    NCI_Free(cstart);
    NCI_Free(ccord);
    NCI_Free(cend);

    return NC_NOERR;
}

int
nczipioi_iput_var(NC_zip        *nczipp,
              NC_zip_var        *varp,
              MPI_Offset        *starts,
              MPI_Offset        *counts,
              const void        *xbuf,
              const void        *buf,
              int               *reqid)
{
    int err;
    int req_id;
    NC_zip_req req;

    err = nczipioi_init_put_req(nczipp, &req, varp, start, count, stride, xbuf, buf);

      // Release var info
    adios_free_varinfo (v);

    // Add to req list
    nczipioi_list_add(&(nczipp->putlist), &req_id);
    ncadp->putlist.reqs[req_id] = req;
    
    if (reqid != NULL){
        *reqid = req_id;
    }

    return NC_NOERR;
}
