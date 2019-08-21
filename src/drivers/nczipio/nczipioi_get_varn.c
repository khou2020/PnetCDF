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
#include <stdint.h> 

#include <mpi.h>

#include <pnc_debug.h>
#include <common.h>
#include <nczipio_driver.h>
#include "nczipio_internal.h"

int
nczipioi_get_varn_cb_chunk(NC_zip          *nczipp,
                    NC_zip_var      *varp,
                    int              nreq,
                    MPI_Offset* const *starts,
                    MPI_Offset* const *counts,
                    MPI_Offset* const *strides,
                    void            **bufs)
{
    int err;
    int i, j, k, l;
    int cid, req;   // Chunk iterator

    MPI_Offset *ostart, *osize;
    int *tsize, *tssize, *tstart, *tsizep, *tssizep, *tstartp;   // Size for sub-array type
    MPI_Offset *citr; // Chunk iterator
    
    int *rcnt_local, *rcnt_all;   // Number of processes that writes to each chunk

    int overlapsize;    // Size of overlaping region of request and chunk
    int overlapsize_total, overlapcnt;
    char *cbuf = NULL;     // Intermediate continuous buffer
    
    int packoff, unpackoff; // Pack offset
    MPI_Datatype ptype; // Pack datatype

    int nread;  // # chunks to read form file
    int *rids;  // Id of chunks to read from file

    int nsend, nrecv;   // Number of send and receive
    MPI_Request *sreqs, *rreqs;    // Send and recv req
    MPI_Status *sstats, *rstats;    // Send and recv status
    char **sbufs, **rbufs;   // Send and recv buffer
    int *rsizes;    // recv size of each message
    MPI_Message rmsg;   // Receive message

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_INIT)

    // Allocate buffering for write count
    rcnt_local = (int*)NCI_Malloc(sizeof(int) * varp->nchunk * 2);
    rcnt_all = rcnt_local + varp->nchunk;

    // Allocate buffering for overlaping index
    tsize = (int*)NCI_Malloc(sizeof(int) * varp->ndim * 3);
    tssize = tsize + varp->ndim;
    tstart = tssize + varp->ndim;
    ostart = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim * 3);
    osize = ostart + varp->ndim;

    // Chunk iterator
    citr = osize + varp->ndim;

    // We need to calculate the size of message of each chunk
    // This is just for allocating send buffer
    // We do so by iterating through all request and all chunks they cover
    // If we are not the owner of a chunk, we need to send message
    memset(rcnt_local, 0, sizeof(int) * varp->nchunk);
    nsend = 0;
    for(req = 0; req < nreq; req++){
        // Iterate through chunks
        nczipioi_chunk_itr_init(varp, starts[req], counts[req], citr, &cid);
        do{
            if (varp->chunk_owner[cid] != nczipp->rank && rcnt_local[cid] == 0){
                // Count number of mnessage we need to send
                nsend++;    
            }

            rcnt_local[cid] = 1;
        } while (nczipioi_chunk_itr_next(varp, starts[req], counts[req], citr, &cid));
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_INIT)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SYNC)

    // Sync number of messages of each chunk
    CHK_ERR_ALLREDUCE(rcnt_local, rcnt_all, varp->nchunk, MPI_INT, MPI_SUM, nczipp->comm);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SYNC)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_INIT)

    // We need to prepare chunk in the chunk cache
    // For chunks not yet allocated, we need to read them form file collectively
    // We collect chunk id of those chunks
    // Calculate number of recv request
    // This is for all the chunks
    rids = (int*)NCI_Malloc(sizeof(int) * varp->nmychunk);
    nread = 0;
    nrecv = 0;
    for(i = 0; i < varp->nmychunk; i++){
        cid = varp->mychunks[i];
        // We don't need message for our own data
        nrecv += rcnt_all[cid] - rcnt_local[cid];
        // Count number of chunks we need to prepare
        // We read only chunks that is required
        if (rcnt_all[cid] || rcnt_local[cid]){
            if (varp->chunk_cache[cid] == NULL){
                err = nczipioi_cache_alloc(nczipp, varp->chunksize, varp->chunk_cache + cid);
                //varp->chunk_cache[cid] = (NC_zip_cache*)NCI_Malloc(varp->chunksize);
                if (varp->data_lens[cid] > 0){
                    rids[nread++] = cid;
                }
            }
            else{
                nczipioi_cache_visit(nczipp, varp->chunk_cache[cid]);
            }
        }
    }
    // Increase batch number to indicate allocated chunk buffer can be freed for future allocation
    (nczipp->cache_serial)++;

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB)  // I/O time count separately

    // Decompress chunks into chunk cache
    nczipioi_load_var(nczipp, varp, nread, rids);

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB)

    // Allocate buffer for send and recv
    // We need to accept nrecv requests and receive nsend of replies
    rreqs = (MPI_Request*)NCI_Malloc(sizeof(MPI_Request) * (nrecv + nsend));
    rstats = (MPI_Status*)NCI_Malloc(sizeof(MPI_Status) * (nrecv + nsend));
    rbufs = (char**)NCI_Malloc(sizeof(char*) * (nrecv + nsend));
    rsizes = (int*)NCI_Malloc(sizeof(int) * (nrecv + nsend));
    // We need to send nsend requests and reply nrecv of requests
    sbufs = (char**)NCI_Malloc(sizeof(char*) * (nrecv + nsend));
    sreqs = (MPI_Request*)NCI_Malloc(sizeof(MPI_Request) * (nrecv + nsend));
    sstats = (MPI_Status*)NCI_Malloc(sizeof(MPI_Status) * (nrecv + nsend));

    // Post send
    k = l = 0;
    for(cid = 0; cid < varp->nchunk; cid++){
        if (varp->chunk_owner[cid] == nczipp->rank){
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REQ)

            // We are the owner of the chunk
            // Receive data from other process
            for(j = 0; j < rcnt_all[cid] - rcnt_local[cid]; j++){
                // Get message size, including metadata
                CHK_ERR_MPROBE(MPI_ANY_SOURCE, cid, nczipp->comm, &rmsg, rstats);
                CHK_ERR_GET_COUNT(rstats, MPI_BYTE, rsizes + k);

                // Allocate buffer
                rbufs[k] = (char*)NCI_Malloc(rsizes[k]);

                // Post irecv
                CHK_ERR_IMRECV(rbufs[k], rsizes[k], MPI_BYTE, &rmsg, rreqs + k);
                k++;
            }

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REQ)
        }
        else{
            // We have some request to send
            if (rcnt_local[cid] > 0){
                get_chunk_itr(varp, cid, citr);
                rsizes[nrecv + l] = overlapcnt = 0;

                NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_PACK_REQ)
                
                // Calculate send buffer size
                for(req = 0; req < nreq; req++){
                    // Calculate chunk overlap
                    overlapsize = get_chunk_overlap(varp, citr, starts[req], counts[req], ostart, osize);
                    
                    rsizes[nrecv + l]  += overlapsize;

                    if (overlapsize > 0){
                        overlapcnt++;
                    }
                }

                // Allocate buffer
                // Faster to request the entire chunk
                if (rsizes[nrecv + l]  >= varp->chunksize){
                    rsizes[nrecv + l]  = varp->chunksize;
                    overlapcnt = 1;
                }
                sbufs[l] = (char*)NCI_Malloc(sizeof(int) * (overlapcnt * varp->ndim * 2) + 1);
                rbufs[nrecv + l] = (char*)NCI_Malloc(rsizes[nrecv + l]);

                // Metadata
                *((int*)sbufs[l]) = rsizes[nrecv + l];  packoff = sizeof(int);
                if (rsizes[nrecv + l] == varp->chunksize){  // Request the entire chunk directly if need more than that
                    tstartp = (int*)(sbufs[l] + packoff); packoff += varp->ndim * sizeof(int);
                    tsizep = (int*)(sbufs[l] + packoff); packoff += varp->ndim * sizeof(int);
                    memset(tstartp, 0, sizeof(int) * varp->ndim);
                    memcpy(tsizep, varp->chunkdim, sizeof(int) * varp->ndim);
                }
                else{
                    for(req = 0; req < nreq; req++){
                        // Calculate chunk overlap
                        overlapsize = get_chunk_overlap(varp, citr, starts[req], counts[req], ostart, osize);

                        if (overlapsize > 0){
                            tstartp = (int*)(sbufs[l] + packoff); packoff += varp->ndim * sizeof(int);
                            tsizep = (int*)(sbufs[l] + packoff); packoff += varp->ndim * sizeof(int);
                            // Metadata
                            for(j = 0; j < varp->ndim; j++){
                                tstartp[j] = (uintptr_t)(ostart[j] - citr[j]);
                                tsizep[j] = (uintptr_t)osize[j];
                            }
                        }
                    }
                }

                NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_PACK_REQ)
                NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REQ)

                // Send request
                CHK_ERR_ISEND(sbufs[l], packoff, MPI_BYTE, varp->chunk_owner[cid], cid, nczipp->comm, sreqs + l);

                NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REQ)
                NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REP)

                //printf("Rank: %d, CHK_ERR_IRECV(%d, %d, %d, %d)\n", nczipp->rank, overlapsize, varp->chunk_owner[cid], cid + 1024, nrecv + k); fflush(stdout);
                CHK_ERR_IRECV(rbufs[l + nrecv], rsizes[nrecv + l] , MPI_BYTE, varp->chunk_owner[cid], cid + 1024, nczipp->comm, rreqs + nrecv + l);

                NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REP)

                l++;
            } 
        }
    }

    // Allocate intermediate buffer
    cbuf = (char*)NCI_Malloc(varp->chunksize);

    // For each chunk we own, we need to reply to incoming reqeust
    k = 0;
    for(i = 0; i < varp->nmychunk; i++){
        cid = varp->mychunks[i];
        
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SELF)
        
        // Handle our own data first if we have any
        if (rcnt_local[cid] > 0){
            // Convert chunk id to iterator
            get_chunk_itr(varp, cid, citr);

            for(req = 0; req < nreq; req++){
                // Calculate overlapping region
                overlapsize = get_chunk_overlap(varp, citr, starts[req], counts[req], ostart, osize);

                if (overlapsize > 0){
                    // Pack type from chunk buffer to (contiguous) intermediate buffer
                    for(j = 0; j < varp->ndim; j++){
                        tstart[j] = (uintptr_t)(ostart[j] - citr[j]);
                        tsize[j] = varp->chunkdim[j];
                        tssize[j] = (uintptr_t)osize[j];
                    }                   
                    CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
                    CHK_ERR_TYPE_COMMIT(&ptype);

                    // Pack data into intermediate buffer
                    packoff = 0;
                    CHK_ERR_PACK(varp->chunk_cache[cid]->buf, 1, ptype, cbuf, varp->chunksize, &packoff, nczipp->comm);
                    overlapsize = packoff;
                    MPI_Type_free(&ptype);

                    // Pack type from (contiguous) intermediate buffer to user buffer
                    for(j = 0; j < varp->ndim; j++){
                        tstart[j] = (uintptr_t)(ostart[j] - starts[req][j]);
                        tsize[j] = (uintptr_t)counts[req][j];
                    }                 
                    CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype); 
                    CHK_ERR_TYPE_COMMIT(&ptype);

                    // Pack data into user buffer
                    packoff = 0;
                    CHK_ERR_UNPACK(cbuf, overlapsize, &packoff, bufs[req], 1, ptype, nczipp->comm);
                    MPI_Type_free(&ptype);
                }
            }
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SELF)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REQ)

        // Wait for all send requests related to this chunk
        // We remove the impact of -1 mark in rcnt_local[cid]
        CHK_ERR_WAITALL(rcnt_all[cid] - rcnt_local[cid], rreqs + k, rstats + k);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REQ)

        // Now, it is time to process data from other processes
        for(j = 0; j < varp->ndim; j++){
            tsize[j] = varp->chunkdim[j];
        }
        // Process data received
        for(j = k; j < k + rcnt_all[cid] - rcnt_local[cid]; j++){
            packoff = 0; 

            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)
            
            // Allocate buffer 
            overlapsize = *((int*)rbufs[j]); unpackoff = sizeof(int);
            sbufs[j + nsend] = (char*)NCI_Malloc(overlapsize); // For reply

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)

            // Pack data
            while(unpackoff < rsizes[j]){
                NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)

                // Get metadata
                tstartp = (int*)(rbufs[j] + unpackoff); unpackoff += varp->ndim * sizeof(int);
                tssizep = (int*)(rbufs[j] + unpackoff); unpackoff += varp->ndim * sizeof(int);

                NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)
                NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_PACK_REP)

                // Pack type
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssizep, tstartp, MPI_ORDER_C, varp->etype, &ptype);
                CHK_ERR_TYPE_COMMIT(&ptype);

                // Pack data
                CHK_ERR_PACK(varp->chunk_cache[cid]->buf, 1, ptype, sbufs[j + nsend], overlapsize, &packoff, nczipp->comm);
                MPI_Type_free(&ptype);    

                NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_PACK_REP)   
            }

            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REP)

            // Send reply
            //printf("Rank: %d, CHK_ERR_ISEND(%d, %d, %d, %d)\n", nczipp->rank, packoff, varp->chunk_owner[cid], cid + 1024, k + nsend); fflush(stdout);
            CHK_ERR_ISEND(sbufs[j + nsend], packoff, MPI_BYTE, rstats[j].MPI_SOURCE, cid + 1024, nczipp->comm, sreqs + j + nsend);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REP)
        }
        k += rcnt_all[cid] - rcnt_local[cid];
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REQ)

    // Wait for all request sent
    //printf("Rank: %d, CHK_ERR_WAITALL_send(%d, %d)\n", nczipp->rank, nsend, 0); fflush(stdout);
    CHK_ERR_WAITALL(nsend, sreqs, sstats);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REQ)

    // Receive replies from the owners and update the user buffer
    k = 0;
    for(cid = 0; cid < varp->nchunk; cid++){
        if (rcnt_local[cid] > 0 && varp->chunk_owner[cid] != nczipp->rank){
            get_chunk_itr(varp, cid, citr);

            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REP)

            // Wait for reply
            //printf("Rank: %d, MPI_Wait_recv(%d)\n", nczipp->rank, nrecv + k); fflush(stdout);
            MPI_Wait(rreqs + nrecv + k, rstats + nrecv + k);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REP)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_UNPACK_REP)

            packoff = 0;
            for(req = 0; req < nreq; req++){
                // Calculate chunk overlap
                overlapsize = get_chunk_overlap(varp, citr, starts[req], counts[req], ostart, osize);

                if (overlapsize > 0){
                    // Pack type from recv buffer to user buffer
                    for(j = 0; j < varp->ndim; j++){
                        tstart[j] = (uintptr_t)(ostart[j] - starts[req][j]);
                        tsize[j] = (uintptr_t)counts[req][j];
                        tssize[j] = (uintptr_t)osize[j];
                    }
                    CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
                    CHK_ERR_TYPE_COMMIT(&ptype);

                    // Pack data                
                    CHK_ERR_UNPACK(rbufs[nrecv + k], rsizes[nrecv + k], &packoff, bufs[req], 1, ptype, nczipp->comm);                
                    MPI_Type_free(&ptype);
                }
            }
            k++;

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_UNPACK_REP)
        }
    }


    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REP)

    // Wait for all send replies
    CHK_ERR_WAITALL(nrecv, sreqs + nsend, sstats + nsend);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REP)
    
    // Free buffers
    NCI_Free(rcnt_local);

    NCI_Free(rids);

    NCI_Free(tsize);

    NCI_Free(ostart);

    for(i = 0; i < nsend + nrecv; i++){
        NCI_Free(sbufs[i]);
        NCI_Free(rbufs[i]);
    }
    NCI_Free(sreqs);
    NCI_Free(sstats);
    NCI_Free(sbufs);
    NCI_Free(rreqs);
    NCI_Free(rstats);
    NCI_Free(rbufs);
    NCI_Free(rsizes);

    if (cbuf != NULL){
        NCI_Free(cbuf);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB)

    return NC_NOERR;
}

int nczipioi_get_varn_cb_proc(  NC_zip          *nczipp,
                                NC_zip_var      *varp,
                                int              nreq,
                                MPI_Offset* const *starts,
                                MPI_Offset* const *counts,
                                void            **bufs) {
    int err;
    int i, j, k, l;
    int cid, cown; // Chunk iterator and owner
    int vid;
    int r;
    MPI_Offset *ostart, *osize;
    int *tsize, *tssize, *tstart, *tssizep, *tstartp; // Size for sub-array type
    MPI_Offset *citr;                                 // Bounding box for chunks overlapping my own write region

    int *scnt, *ssize;
    int *rcnt, *rsize;

    int nread;
    int *rids;
    int wrange[4];   // Number of processes that writes to each chunk


    int overlapsize;   // Size of overlaping region of request and chunk
    char *tbuf = NULL; // Intermediate buffer

    int packoff;        // Pack offset
    MPI_Datatype ptype; // Pack datatype

    int nsend, npack;
    char **sbuf, **sbufp;
    MPI_Datatype *stype;
    MPI_Datatype **stypes, **stypesp;
    MPI_Aint **soffs, **soffsp;
    int **slens, **slensp;
    int *sntypes;
    int *sdst; // recv size of each message
    int *smap;
    MPI_Request *sreq;
    MPI_Status *sstat;

    int nrecv, nunpack;
    char **rbuf, *rbufp;
    MPI_Datatype *rtype;
    MPI_Datatype **rtypes, *rtypesp;
    MPI_Aint **roffs, *roffsp;
    int **rlens, *rlensp;
    int *rmap;
    int *rsrc;
    MPI_Request *rreq;
    MPI_Status *rstat;

    MPI_Message rmsg; // Receive message

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_INIT)

    // Allocate buffering for overlaping index
    tstart = (int *)NCI_Malloc(sizeof(int) * nczipp->max_ndim * 3);
    tssize = tstart + nczipp->max_ndim;
    tsize = tssize + nczipp->max_ndim;
    ostart = (MPI_Offset *)NCI_Malloc(sizeof(MPI_Offset) * nczipp->max_ndim * 3);
    osize = ostart + nczipp->max_ndim;

    // Chunk iterator
    citr = osize + nczipp->max_ndim;

    // Access range
    wrange[0] = varp->nchunk;
    wrange[1] = 0;

    // Allocate buffering for write count
    scnt = (int *)NCI_Malloc(sizeof(int) * nczipp->np * 7);
    ssize = scnt + nczipp->np;
    rcnt = ssize + nczipp->np;
    rsize = rcnt + nczipp->np;
    smap = rsize + nczipp->np;
    rmap = smap + nczipp->np;
    rsrc = rmap + nczipp->np;

    memset(scnt, 0, sizeof(int) * nczipp->np * 2);
    npack = 0;
    // Count total number of messages and build a map of accessed chunk to list of comm datastructure
    for (r = 0; r < nreq; r++) {
        nczipioi_chunk_itr_init(varp, starts[r], counts[r], citr, &cid); // Initialize chunk iterator
        do {
            // Chunk owner
            cown = varp->chunk_owner[cid];

            // Mapping to skip list of send requests
            if ((scnt[cown] == 0) && (cown != nczipp->rank)) {
                smap[cown] = npack++;
            }
            scnt[cown]++; // Need to send message if not owner

            if (wrange[0] > cid) {
                wrange[0] = cid;
            }
            if (wrange[1] < cid) {
                wrange[1] = cid;
            }
        } while (nczipioi_chunk_itr_next(varp, starts[r], counts[r], citr, &cid));
    }
    nsend = npack;
    if (scnt[nczipp->rank]){
        smap[nczipp->rank] = npack++;
    }

    // Allocate data structure for sending
    sbuf = (char **)NCI_Malloc(sizeof(char *) * npack * 2);
    sbufp = sbuf + npack;
    sreq = (MPI_Request *)NCI_Malloc(sizeof(MPI_Request) * nsend * 2);
    sstat = (MPI_Status *)NCI_Malloc(sizeof(MPI_Status) * nsend * 2);
    stype = (MPI_Datatype *)NCI_Malloc(sizeof(MPI_Datatype) * npack);
    stypes = (MPI_Datatype **)NCI_Malloc(sizeof(MPI_Datatype *) * npack * 2);
    stypesp = stypes + npack;
    soffs = (MPI_Aint **)NCI_Malloc(sizeof(MPI_Aint *) * npack * 2);
    soffsp = soffs + npack;
    slens = (int **)NCI_Malloc(sizeof(int *) * npack * 2);
    slensp = slens + npack;
    sntypes = (int *)NCI_Malloc(sizeof(int) * npack * 2);
    sdst = sntypes + npack;

    if (npack > 0){
        k = l = 0;
        for (i = 0; i < nczipp->np; i++) {
            if (scnt[i] > 0) {
                j = smap[i];
                sdst[j] = i;
                sntypes[j] = scnt[i];
                ssize[i] = scnt[i] * sizeof(int) * (varp->ndim * 2 + 2);
                l += sntypes[j];
                k += ssize[i];
            }
        }

        stypes[0] = (MPI_Datatype *)NCI_Malloc(sizeof(MPI_Datatype) * l);
        soffs[0] = (MPI_Aint *)NCI_Malloc(sizeof(MPI_Aint) * l);
        slens[0] = (int *)NCI_Malloc(sizeof(int) * l);
        sbuf[0] = sbufp[0] = (char *)NCI_Malloc(k);

        for (i = 1; i < npack; i++) {
            stypes[i] = stypes[i - 1] + sntypes[i - 1];
            soffs[i] = soffs[i - 1] + sntypes[i - 1];
            slens[i] = slens[i - 1] + sntypes[i - 1];

            sbuf[i] = sbufp[i] = sbuf[i - 1] + ssize[sdst[i - 1]];
        }
        for (i = 0; i < npack; i++) {
            stypesp[i] = stypes[i];
            soffsp[i] = soffs[i];
            slensp[i] = slens[i];
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_INIT)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_PACK_REQ)

    // Pack requests
    for (r = 0; r < nreq; r++){
        nczipioi_chunk_itr_init_ex(varp, starts[r], counts[r], citr, &cid, ostart, osize); // Initialize chunk iterator
        do {
            // Chunk index and owner
            cown = varp->chunk_owner[cid];

            j = smap[cown];
    
            // Pack metadata
            *((int *)(sbufp[j])) = cid; sbufp[j] += sizeof(int);
            tstartp = (int *)sbufp[j];  sbufp[j] += varp->ndim * sizeof(int);
            tssizep = (int *)sbufp[j];  sbufp[j] += varp->ndim * sizeof(int);

            for (i = 0; i < varp->ndim; i++){
                tstartp[i] = (uintptr_t)(ostart[i] - citr[i]);
                tssizep[i] = (uintptr_t)osize[i];
            }

            // Pack type from user buffer to send buffer
            for (i = 0; i < varp->ndim; i++){
                tsize[i] = (uintptr_t)counts[r][i];
                tstart[i] = (uintptr_t)(ostart[i] - starts[r][i]);
            }

            err = nczipioi_subarray_off_len(varp->ndim, tsize, tssizep, tstart, soffsp[j], slensp[j]);
            if (err){
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssizep, tstart, MPI_ORDER_C, varp->etype, stypesp[j]);
                CHK_ERR_TYPE_COMMIT(stypesp[j]);
                *(soffsp[j]) = (uintptr_t)(bufs[r]);
                *(slensp[j]) = 1;
            }
            else{
                *(stypesp[j]) = MPI_BYTE;
                *(soffsp[j]) = (*(soffsp[j])) * varp->esize + (uintptr_t)(bufs[r]);
                *(slensp[j]) *= varp->esize;
            }
            stypesp[j]++;
            soffsp[j]++;
            slensp[j]++;

#ifdef PNETCDF_PROFILING
            nczipp->nsend++;
#endif
        } while (nczipioi_chunk_itr_next_ex(varp, starts[r], counts[r], citr, &cid, ostart, osize));
    }

    // Construct compound type
    for (i = 0; i < npack; i++) {
        MPI_Type_struct(sntypes[i], slens[i], soffs[i], stypes[i], stype + i);
        CHK_ERR_TYPE_COMMIT(stype + i);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_PACK_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SYNC)

    // Sync number of write in message
    CHK_ERR_ALLTOALL(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, nczipp->comm);

    // Access range
    wrange[1] *= -1;
    CHK_ERR_ALLREDUCE(wrange, wrange + 2, 2, MPI_INT, MPI_MIN, nczipp->comm);
    wrange[3] *= -1;

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SYNC)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_INIT)

    j = k = 0;
    nunpack = 0; // We don't need to receive request form self
    l = rcnt[nczipp->rank];	// It will be overwritten by rearrangement, save it first
    for (i = 0; i < nczipp->np; i++){
        if ((rcnt[i] > 0) && (i != nczipp->rank)){
            rmap[i] = nunpack;
            rsrc[nunpack]=i;            
            rsize[nunpack] = rcnt[i] * sizeof(int) * (varp->ndim * 2 + 2);
            k += rsize[nunpack];
            rcnt[nunpack++] = rcnt[i];
            j += rcnt[i];
        }
    }
    nrecv = nunpack;
    if (l){
        rmap[nczipp->rank] = nunpack;
        rsrc[nunpack]=nczipp->rank;            
        rsize[nunpack] = l * sizeof(int) * (varp->ndim * 2 + 2);
        rcnt[nunpack++] = l;
        j += l;
    }

    // Allocate data structure for receving
    rbuf = (char **)NCI_Malloc(sizeof(char *) * nunpack);
    rreq = (MPI_Request *)NCI_Malloc(sizeof(MPI_Request) * nrecv * 2);
    rstat = (MPI_Status *)NCI_Malloc(sizeof(MPI_Status) * nrecv * 2);
    rtype = (MPI_Datatype*)NCI_Malloc(sizeof(MPI_Datatype) * nunpack);
    rtypes = (MPI_Datatype**)NCI_Malloc(sizeof(MPI_Datatype*) * nunpack);
    roffs = (MPI_Aint**)NCI_Malloc(sizeof(MPI_Aint*) * nunpack);
    rlens = (int**)NCI_Malloc(sizeof(int*) * nunpack);

    if (nunpack > 0) {
        rtypes[0] = (MPI_Datatype*)NCI_Malloc(sizeof(MPI_Datatype) * j);
        roffs[0] = (MPI_Aint*)NCI_Malloc(sizeof(MPI_Aint) * j);
        rlens[0] = (int*)NCI_Malloc(sizeof(int) * j);
        rbuf[0] = (char *)NCI_Malloc(k);
        for (i = 1; i < nunpack; i++) {
            rtypes[i] = rtypes[i - 1] + rcnt[i - 1];
            roffs[i] = roffs[i - 1] + rcnt[i - 1];
            rlens[i] = rlens[i - 1] + rcnt[i - 1];

            rbuf[i] = rbuf[i - 1] + rsize[i - 1];
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_INIT)

    // Post send req
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SEND_REQ)
    for (i = 0; i < nsend; i++) {
        CHK_ERR_ISEND(sbuf[i], ssize[sdst[i]], MPI_BYTE, sdst[i], 0, nczipp->comm, sreq + i);
    }
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SEND_REQ)

    // Post recv rep
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SEND_REQ)
    for (i = 0; i < nsend; i++) {
        CHK_ERR_IRECV(MPI_BOTTOM, 1, stype[i], sdst[i], 1, nczipp->comm, sreq + nsend + i);
    }
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SEND_REQ)

    // Post recv req
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_RECV_REQ)
    for (i = 0; i < nrecv; i++) {
        CHK_ERR_IRECV(rbuf[i], rsize[i], MPI_BYTE, rsrc[i], 0, nczipp->comm, rreq + i);
    }
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_RECV_REQ)

    // Prepare chunk buffer
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_INIT)
    for(j = 0; j < varp->nmychunk && varp->mychunks[j] < wrange[2]; j++);
    for(k = j; k < varp->nmychunk && varp->mychunks[k] <= wrange[3]; k++);
    rids = (int*)NCI_Malloc(sizeof(int) * (k - j));
    nread = 0;
    for(i = j; i < k; i++){
        cid = varp->mychunks[i];
        if (varp->chunk_cache[cid] == NULL){
            err = nczipioi_cache_alloc(nczipp, varp->chunksize, varp->chunk_cache + cid);
            if (varp->data_lens[cid] > 0){
                rids[nread++] = cid;
            }
        }
        else{
            nczipioi_cache_visit(nczipp, varp->chunk_cache[cid]);
        }
    }
    // Increase batch number to indicate allocated chunk buffer can be freed for future allocation
    (nczipp->cache_serial)++;
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_INIT)

    err = nczipioi_load_var(nczipp, varp, nread, rids);    CHK_ERR

    // Handle our own data
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SELF)
    
    if (scnt[nczipp->rank] > 0){
        j = smap[nczipp->rank];
        l = rmap[nczipp->rank];
        
        // Pack into continuous buffer
        rbufp = sbuf[j];
        rtypesp = rtypes[l];
        roffsp = roffs[l];
        rlensp = rlens[l];
        for (k = 0; k < rcnt[l]; k++) {
            // Retrieve metadata
            cid = *((int *)(rbufp));            rbufp += sizeof(int);  
            tstartp = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);            
            tssizep = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);

            // Subarray type
            err = nczipioi_subarray_off_len(varp->ndim, varp->chunkdim, tssizep, tstartp, roffsp, rlensp);
            if (err){
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, varp->chunkdim, tssizep, tstartp, MPI_ORDER_C, varp->etype, rtypesp);
                CHK_ERR_TYPE_COMMIT(rtypesp);
                *roffsp = (((uintptr_t)(varp->chunk_cache[cid]->buf)));
                *rlensp = 1;
            }
            else{
                *rtypesp = MPI_BYTE;
                *roffsp = (*roffsp) * varp->esize + (((uintptr_t)(varp->chunk_cache[cid]->buf)));
                *rlensp *= varp->esize;
            }
            rtypesp++;
            roffsp++;
            rlensp++;

            // Mark chunk as dirty
            varp->dirty[cid] = 1;
#ifdef PNETCDF_PROFILING
            nczipp->nrecv++;
#endif
        }

        // Pack type
        MPI_Type_struct(rcnt[l], rlens[l], roffs[l], rtypes[l], rtype + l);
        CHK_ERR_TYPE_COMMIT(rtype + l);
        MPI_Type_size(rtype[l], rsize + l);

        // Allocate intermediate buffer for our own data
        tbuf = (char *)NCI_Malloc(rsize[l]);

        // Pack data into contiguous buffer
        packoff = 0;
        CHK_ERR_PACK(MPI_BOTTOM, 1, rtype[l], tbuf, rsize[l], &packoff, nczipp->comm);

        // Unpack into user buffer
        packoff = 0;
        CHK_ERR_UNPACK(tbuf, rsize[l], &packoff, MPI_BOTTOM, 1, stype[j], nczipp->comm);

        // Free temporary buffer
        NCI_Free(tbuf);
    }
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SELF)

    //Handle incoming requests
    for (i = 0; i < nrecv; i++) {
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_RECV_REQ)

        // Will wait any provide any benefit?
        MPI_Waitany(nrecv, rreq, &j, rstat);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_RECV_REQ)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_UNPACK_REQ)

        rbufp = rbuf[j];
        rtypesp = rtypes[j];
        roffsp = roffs[j];
        rlensp = rlens[j];
        for (k = 0; k < rcnt[j]; k++) {
            // Retrieve metadata
            cid = *((int *)(rbufp));            rbufp += sizeof(int);
            tstartp = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);
            tssizep = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);

            // Subarray type
            err = nczipioi_subarray_off_len(varp->ndim, varp->chunkdim, tssizep, tstartp, roffsp, rlensp);
            if (err){
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, varp->chunkdim, tssizep, tstartp, MPI_ORDER_C, varp->etype, rtypesp);
                CHK_ERR_TYPE_COMMIT(rtypesp);
                *roffsp = (((uintptr_t)(varp->chunk_cache[cid]->buf)));
                *rlensp = 1;
            }
            else{
                *rtypesp = MPI_BYTE;
                *roffsp = (*roffsp) * varp->esize + (((uintptr_t)(varp->chunk_cache[cid]->buf)));
                *rlensp *= varp->esize;
            }
            rtypesp++;
            roffsp++;
            rlensp++;

            // Mark chunk as dirty
            varp->dirty[cid] = 1;
#ifdef PNETCDF_PROFILING
            nczipp->nrecv++;
#endif
        }

        // Send data
        MPI_Type_struct(rcnt[j], rlens[j], roffs[j], rtypes[j], rtype + j);
        CHK_ERR_TYPE_COMMIT(rtype + j);
        CHK_ERR_ISEND(MPI_BOTTOM, 1, rtype[j], rstat->MPI_SOURCE, 1, nczipp->comm, rreq + nrecv + i);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_UNPACK_REQ)
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_RECV_REP)
    CHK_ERR_WAITALL(nsend, sreq + nsend, sstat);
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_RECV_REP)

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SEND_REQ)
    CHK_ERR_WAITALL(nsend, sreq, sstat);
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SEND_REQ)

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SEND_REP)
    CHK_ERR_WAITALL(nrecv, rreq + nrecv, rstat);
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SEND_REP)

    // Free type
    for (i = 0; i < npack; i++) {
        for (j = 0; j < sntypes[i]; j++) {
            if (stypes[i][j] != MPI_BYTE) {
                MPI_Type_free(stypes[i] + j);
            }
        }
        MPI_Type_free(stype + i);
    }

    for (i = 0; i < nunpack; i++) {
        for (j = 0; j < rcnt[i]; j++) {
            if (rtypes[i][j] != MPI_BYTE) {
                MPI_Type_free(rtypes[i] + j);
            }
        }
        MPI_Type_free(rtype + i);
    }

    // Free buffers
    NCI_Free(scnt);

    NCI_Free(tstart);

    NCI_Free(ostart);
    
    NCI_Free(rids);

    NCI_Free(sreq);
    NCI_Free(sstat);
    if (npack > 0) {
        NCI_Free(sbuf[0]);
        NCI_Free(stypes[0]);
        NCI_Free(soffs[0]);
        NCI_Free(slens[0]);
    }
    NCI_Free(sbuf);
    NCI_Free(stypes);
    NCI_Free(stype);
    NCI_Free(soffs);
    NCI_Free(slens);
    NCI_Free(sntypes);

    NCI_Free(rreq);
    NCI_Free(rstat);
    if (nunpack > 0){
        NCI_Free(rbuf[0]);
        NCI_Free(rtypes[0]);
        NCI_Free(roffs[0]);
        NCI_Free(rlens[0]);
    }
    NCI_Free(rbuf);
    NCI_Free(rtype);
    NCI_Free(rtypes);
    NCI_Free(roffs);
    NCI_Free(rlens);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB)

    return NC_NOERR;
}


int
nczipioi_get_varn(NC_zip        *nczipp,
              NC_zip_var       *varp,
              int              nreq,
              MPI_Offset* const *starts,
              MPI_Offset* const *counts,
		  const void       *buf) {
    int i, j;
    MPI_Offset rsize;
    char *bptr = (char*)buf;
    char **bufs;
    
    // Calculate buffer offset of each request
    bufs = (char**)NCI_Malloc(sizeof(char*) * nreq);
    for(i = 0; i < nreq; i++){
        bufs[i] = bptr;
        rsize = varp->esize;
        for(j = 0; j < varp->ndim; j++){
            rsize *= counts[i][j];
        }
        bptr += rsize;
    }

    // Collective buffer
    switch (nczipp->comm_unit){
        case NC_ZIP_COMM_CHUNK:
            nczipioi_get_varn_cb_chunk(nczipp, varp, nreq, starts, counts, NULL, (void**)bufs);
            break;
        case NC_ZIP_COMM_PROC:
            nczipioi_get_varn_cb_proc(nczipp, varp, nreq, starts, counts, (void**)bufs);
            break;
    }
    NCI_Free(bufs);

    return NC_NOERR;
}
