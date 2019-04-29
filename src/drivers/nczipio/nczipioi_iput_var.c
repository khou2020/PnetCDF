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
int nczipioi_iput_cb_chunk(NC_zip *nczipp, int nreq, int *reqids, int *stats){
    int err;
    int i, j;
    int vid;   // Iterators for variable id
    int *nreqs;  // Number of reqids in each variable
    int *nums;  // Number of reqs in each varn
    int **vreqids;
    int num, maxnum = 0;
    MPI_Offset **starts, **counts, **strides;
    MPI_Offset rsize;
    char **bufs;
    NC_zip_req *req;

    // Count total number of request in per variable for packed varn request
    nums = (int*)NCI_Malloc(sizeof(int) * nczipp->vars.cnt * 2);
    nreqs = nums + nczipp->vars.cnt;
    memset(nums, 0, sizeof(int) * nczipp->vars.cnt);
    memset(nreqs, 0, sizeof(int) * nczipp->vars.cnt);
    for(i = 0; i < nreq; i++){
        req = nczipp->putlist.reqs + reqids[i];
        nreqs[req->varid]++;
        nums[req->varid] += req->nreq;
    }

    /* Allocate a skip list of reqids for each vriable
     * At the same time, we find out the number of starts and counts we need to allocate
     */
    vreqids = (int**)NCI_Malloc(sizeof(int*) * nczipp->vars.cnt);
    vreqids[0] = (int*)NCI_Malloc(sizeof(int) * nreq);
    maxnum = 0;
    i = 0;
    for(vid = 0; vid < nczipp->vars.cnt; vid++){
        if (nreqs[vid] > 0){
            // Assign buffer to reqid skip list
            vreqids[vid] = vreqids[0] + i;
            i += nreqs[vid];

            // maximum number of starts and counts we need across all variables
            if (maxnum < nums[vid]){
                maxnum = nums[vid];
            }
        }
    }

    // Fill up the skip list
    memset(nreqs, 0, sizeof(int) * nczipp->vars.cnt);
    for(i = 0; i < nreq; i++){
        req = nczipp->putlist.reqs + reqids[i];
        vreqids[req->varid][nreqs[req->varid]++] = reqids[i];
    }
    
    // Allocate parameters
    starts = (MPI_Offset**)NCI_Malloc(sizeof(MPI_Offset*) * maxnum * 2);
    counts = starts + maxnum;
    bufs =  (char**)NCI_Malloc(sizeof(char*) * maxnum);

    /* Pack requests variable by variable
     */
    for(vid = 0; vid < nczipp->vars.cnt; vid++){
        if (nreqs[vid] > 0){

            // Collect parameters
            num = 0;
            for(j = 0; j < nreqs[vid]; j++){
                req = nczipp->putlist.reqs + vreqids[vid][j];

                for(i = 0; i < req->nreq; i++){
                    starts[num] = req->starts[i];
                    counts[num] = req->counts[i];
                    bufs[num++] = req->xbufs[i];
                }
            }

            // Perform collective buffering
            nczipioi_put_varn_cb_chunk(nczipp, nczipp->vars.data + vid, num, starts, counts, NULL, (void**)bufs);
        }
    }


    // Free buffers
    NCI_Free(nums);

    NCI_Free(vreqids[0]);
    NCI_Free(vreqids);
    
    NCI_Free(starts);
    NCI_Free(bufs);

    return NC_NOERR;
}

/* Out drive currently can handle only one variable at a time
 * We pack all request as a large varn request
 */
int nczipioi_iput_cb_proc(NC_zip *nczipp, int nreq, int *reqids, int *stats){
    int err;
    int i, j, k;
    int cid, cown;   // Chunk iterator and owner
    int vid;
    int r;
    MPI_Offset *ostart, *osize;
    int *tsize, *tssize, *tstart;   // Size for sub-array type
    MPI_Offset *citr; // Bounding box for chunks overlapping my own write region
    
    int *wcnt_local, *wcnt_all;   // Number of processes that writes to each chunk

    int overlapsize;    // Size of overlaping region of request and chunk
    char *tbuf = NULL;     // Intermediate buffer
    
    int packoff; // Pack offset
    MPI_Datatype ptype; // Pack datatype

    int nsend, nrecv;   // Number of send and receive
    MPI_Request *sreq, *rreq;    // Send and recv req
    MPI_Status *sstat, rstat;    // Send and recv status
    char **sbuf, **rbuf;   // Send and recv buffer
    int *rsize, *ssize;    // recv size of each message
    int *roff, *soff;    // recv size of each message
    int *sdst;    // recv size of each message
    int *smap;
    MPI_Message rmsg;   // Receive message
    NC_zip_var *varp;
    NC_zip_req *req;

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB_INIT)

    // Allocate buffering for write count
    wcnt_local = (int*)NCI_Malloc(sizeof(int) * nczipp->np * 3);
    wcnt_all = wcnt_local + nczipp->np;
    smap = wcnt_all + nczipp->np;

    // Intermediate buffer for our own data
    tbuf = (char*)NCI_Malloc(nczipp->max_chunk_size);

    // Allocate buffering for overlaping index
    tstart = (int*)NCI_Malloc(sizeof(int) * nczipp->max_ndim * 3);
    tssize = tstart + nczipp->max_ndim;
    tsize = tssize + nczipp->max_ndim;
    ostart = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * nczipp->max_ndim * 3);
    osize = ostart + nczipp->max_ndim;

    // Chunk iterator
    citr = osize + nczipp->max_ndim;

    // We need to calculate the size of message of each processes
    // This is just for allocating send buffer
    // We do so by iterating through all request and all chunks they cover
    // If we are not the owner of a chunk, we need to send message
    memset(wcnt_local, 0, sizeof(int) * nczipp->np);
    nsend = 0;

    // Count total number of messages and build a map of accessed chunk to list of comm datastructure
    for(i = 0; i < nreq; i++){
        req = nczipp->putlist.reqs + reqids[i];
        varp = nczipp->vars.data + req->varid;
        for(r = 0; r < req->nreq; r++){
            nczipioi_chunk_itr_init(varp, req->starts[r], req->counts[r], citr, &cid); // Initialize chunk iterator
            do{
                // Chunk index and owner
                cid = get_chunk_id(varp, citr);
                cown = varp->chunk_owner[cid];

                // Mapping to skip list of send requests 
                if (wcnt_local[cown] == 0 && cown != nczipp->rank){
                    smap[cown] = nsend++;
                }
                wcnt_local[cown] = 1;   // Need to send message if not owner       
            } while (nczipioi_chunk_itr_next(varp, req->starts[r], req->counts[r], citr, &cid));
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB_INIT)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB_SYNC)

    // Sync number of messages of each chunk
    CHK_ERR_ALLREDUCE(wcnt_local, wcnt_all, nczipp->np, MPI_INT, MPI_SUM, nczipp->comm);
    nrecv = wcnt_all[nczipp->rank] - wcnt_local[nczipp->rank];  // We don't need to receive request form self

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB_SYNC)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB_PACK_REQ)

    // Allocate data structure for messaging
    sbuf = (char**)NCI_Malloc(sizeof(char*) * nsend);
    ssize = (int*)NCI_Malloc(sizeof(int) * nsend * 3);
    soff = ssize + nsend;
    sdst = soff + nsend;
    sreq = (MPI_Request*)NCI_Malloc(sizeof(MPI_Request) * nsend);
    sstat = (MPI_Status*)NCI_Malloc(sizeof(MPI_Status) * nsend);

    rbuf = (char**)NCI_Malloc(sizeof(char*) * nrecv);
    rsize = (int*)NCI_Malloc(sizeof(int) * nrecv);
    rreq = (MPI_Request*)NCI_Malloc(sizeof(MPI_Request) * nrecv);

    // Count size of each request
    memset(ssize, 0, sizeof(int) * nsend);
    for(k = 0; k < nreq; k++){
        req = nczipp->putlist.reqs + reqids[k];
        varp = nczipp->vars.data + req->varid;

        for(r = 0; r < req->nreq; r++){
            nczipioi_chunk_itr_init(varp, req->starts[r], req->counts[r], citr, &cid); // Initialize chunk iterator
            do{
                // Chunk index and owner
                cid = get_chunk_id(varp, citr);
                cown = varp->chunk_owner[cid];
                if (cown != nczipp->rank){
                    j = smap[cown];
                    sdst[j] = cown; // Record a reverse map by the way

                    // Count overlap
                    get_chunk_overlap(varp, citr, req->starts[r], req->counts[r], ostart, osize);
                    overlapsize = varp->esize;
                    for(i = 0; i < varp->ndim; i++){
                        overlapsize *= osize[i];                     
                    }
                    ssize[j] += overlapsize + sizeof(int) * (varp->ndim * 2 + 2);
                }
            } while (nczipioi_chunk_itr_next(varp, req->starts[r], req->counts[r], citr, &cid));
        }
    }
    // Allocate buffer for send
    for(i = 0; i < nsend; i++){
        sbuf[i] = (char*)NCI_Malloc(ssize[i]);
    }

    // Pack requests
    memset(soff, 0, sizeof(int) * nsend);
    for(k = 0; k < nreq; k++){
        req = nczipp->putlist.reqs + reqids[k];
        varp = nczipp->vars.data + req->varid;

        for(r = 0; r < req->nreq; r++){
            nczipioi_chunk_itr_init(varp, req->starts[r], req->counts[r], citr, &cid); // Initialize chunk iterator
            do{
                // Chunk index and owner
                cid = get_chunk_id(varp, citr);
                cown = varp->chunk_owner[cid];
                if (cown != nczipp->rank){
                    j = smap[cown];

                    // Get overlap region
                    get_chunk_overlap(varp, citr, req->starts[r], req->counts[r], ostart, osize);

                    // Pack type from user buffer to (contiguous) intermediate buffer
                    for(i = 0; i < varp->ndim; i++){
                        tstart[i] = (int)(ostart[i] - req->starts[r][i]);
                        tsize[i] = (int)req->counts[r][i];
                        tssize[i] = (int)osize[i];
                    }
                    //printf("Rank: %d, CHK_ERR_TYPE_CREATE_SUBARRAY_send([%d, %d], [%d, %d], [%d, %d]\n", nczipp->rank, tsize[0], tsize[1], tssize[0], tssize[1], tstart[0], tstart[1]); fflush(stdout);
                    CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
                    CHK_ERR_TYPE_COMMIT(&ptype);

                    // Pack metadata
                    for(i = 0; i < varp->ndim; i++){
                        tstart[i] = (int)(ostart[i] - citr[i]);
                    }
                    CHK_ERR_PACK(&(req->varid), 1, MPI_INT, sbuf[j], ssize[j], soff + j, nczipp->comm);
                    CHK_ERR_PACK(&cid, 1, MPI_INT, sbuf[j], ssize[j], soff + j, nczipp->comm);
                    CHK_ERR_PACK(tstart, varp->ndim, MPI_INT, sbuf[j], ssize[j], soff + j, nczipp->comm);
                    CHK_ERR_PACK(tssize, varp->ndim, MPI_INT, sbuf[j], ssize[j], soff + j, nczipp->comm);
                    // Pack data
                    CHK_ERR_PACK(req->xbufs[r], 1, ptype, sbuf[j], ssize[j], soff + j, nczipp->comm);
                    MPI_Type_free(&ptype);
                }
            } while (nczipioi_chunk_itr_next(varp, req->starts[r], req->counts[r], citr, &cid));
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB_PACK_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB_SEND_REQ)

    // Post send
    for(i = 0; i < nsend; i++){
        CHK_ERR_ISEND(sbuf[i], soff[i], MPI_BYTE, sdst[i], 0, nczipp->comm, sreq + i);
    }    

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB_SEND_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB_RECV_REQ)

    // Post recv
   for(i = 0; i < nrecv; i++){
        // Get message size, including metadata
        CHK_ERR_MPROBE(MPI_ANY_SOURCE, 0, nczipp->comm, &rmsg, &rstat);
        CHK_ERR_GET_COUNT(&rstat, MPI_BYTE, rsize + i);

        // Allocate buffer
        rbuf[i] = (char*)NCI_Malloc(rsize[i]);

        // Post irecv
        CHK_ERR_IMRECV(rbuf[i], rsize[i], MPI_BYTE, &rmsg, rreq + i);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB_RECV_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB_SELF)

    // Handle our own data
    for(k = 0; k < nreq; k++){
        req = nczipp->putlist.reqs + reqids[k];
        varp = nczipp->vars.data + req->varid;

        for(r = 0; r < req->nreq; r++){
            nczipioi_chunk_itr_init(varp, req->starts[r], req->counts[r], citr, &cid); // Initialize chunk iterator
            do{
                // Chunk index and owner
                cid = get_chunk_id(varp, citr);

                if (varp->chunk_owner[cid] == nczipp->rank){
                    // Get overlap region
                    get_chunk_overlap(varp, citr, req->starts[r], req->counts[r], ostart, osize);
                    overlapsize = varp->esize;
                    for(i = 0; i < varp->ndim; i++){
                        overlapsize *= osize[i];                     
                    }

                    if (overlapsize > 0){
                        // Pack type from user buffer to (contiguous) intermediate buffer
                        for(j = 0; j < varp->ndim; j++){
                            tstart[j] = (int)(ostart[j] - req->starts[r][j]);
                            tsize[j] = (int)req->counts[r][j];
                            tssize[j] = (int)osize[j];
                        }
                        //printf("Rank: %d, CHK_ERR_TYPE_CREATE_SUBARRAY_self([%d, %d], [%d, %d], [%d, %d]\n", nczipp->rank, tsize[0], tsize[1], tssize[0], tssize[1], tstart[0], tstart[1]); fflush(stdout);
                        CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
                        CHK_ERR_TYPE_COMMIT(&ptype);

                        // Pack data into intermediate buffer
                        packoff = 0;
                        CHK_ERR_PACK(req->xbufs[r], 1, ptype, tbuf, varp->chunksize, &packoff, nczipp->comm);
                        MPI_Type_free(&ptype);
                        overlapsize = packoff;

                        // Pack type from (contiguous) intermediate buffer to chunk buffer
                        for(j = 0; j < varp->ndim; j++){
                            tstart[j] = (int)(ostart[j] - citr[j]);
                            tsize[j] = varp->chunkdim[j];
                        }
                        //printf("Rank: %d, CHK_ERR_TYPE_CREATE_SUBARRAY_self2([%d, %d], [%d, %d], [%d, %d]\n", nczipp->rank, tsize[0], tsize[1], tssize[0], tssize[1], tstart[0], tstart[1]); fflush(stdout);
                        CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
                        CHK_ERR_TYPE_COMMIT(&ptype);
                        
                        // Unpack data into chunk buffer
                        packoff = 0;
                        CHK_ERR_UNPACK(tbuf, overlapsize, &packoff, varp->chunk_cache[cid], 1, ptype, nczipp->comm);
                        MPI_Type_free(&ptype);    
                    }
                }
            } while (nczipioi_chunk_itr_next(varp, req->starts[r], req->counts[r], citr, &cid));
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB_SELF)

    //Handle incoming requests
    for(i = 0; i < varp->ndim; i++){
        tsize[i] = varp->chunkdim[i];
    }
    for(i = 0; i < nrecv; i++){
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB_RECV_REQ)

        // Will wait any provide any benefit?
        MPI_Waitany(nrecv, rreq, &j, &rstat);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB_RECV_REQ)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB_UNPACK_REQ)

        packoff = 0;
        //printf("rsize_2 = %d\n", rsizes[j]); fflush(stdout);
        while(packoff < rsize[j]){
            // Retrieve metadata
            CHK_ERR_UNPACK(rbuf[j], rsize[j], &packoff, &vid, 1, MPI_INT, nczipp->comm);
            CHK_ERR_UNPACK(rbuf[j], rsize[j], &packoff, &cid, 1, MPI_INT, nczipp->comm);
            varp = nczipp->vars.data + vid;
            CHK_ERR_UNPACK(rbuf[j], rsize[j], &packoff, tstart, varp->ndim, MPI_INT, nczipp->comm);
            CHK_ERR_UNPACK(rbuf[j], rsize[j], &packoff, tssize, varp->ndim, MPI_INT, nczipp->comm);

            //printf("Rank: %d, cid = %d, CHK_ERR_TYPE_CREATE_SUBARRAY_recv([%d, %d], [%d, %d], [%d, %d]\n", nczipp->rank, cid, tsize[0], tsize[1], tssize[0], tssize[1], tstart[0], tstart[1]); fflush(stdout);
            // Pack type
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
            CHK_ERR_TYPE_COMMIT(&ptype);

            // Pack data
            CHK_ERR_UNPACK(rbuf[j], rsize[j], &packoff, varp->chunk_cache[cid], 1, ptype, nczipp->comm);
            //printf("cache[0] = %d, cache[1] = %d\n", ((int*)(varp->chunk_cache[cid]))[0], ((int*)(varp->chunk_cache[cid]))[1]); fflush(stdout);
            MPI_Type_free(&ptype);
        }
        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB_UNPACK_REQ)
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_CB_SEND_REQ)

    CHK_ERR_WAITALL(nsend, sreq, sstat);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB_SEND_REQ)

    // Free buffers
    NCI_Free(wcnt_local);

    NCI_Free(tstart);

    NCI_Free(ostart);

    NCI_Free(sreq);
    NCI_Free(sstat);
    NCI_Free(ssize);
    for(i = 0; i < nsend; i++){
        NCI_Free(sbuf[i]);
    }
    NCI_Free(sbuf);

    NCI_Free(rreq);
    for(i = 0; i < nrecv; i++){
        NCI_Free(rbuf[i]);
    }
    NCI_Free(rbuf);
    NCI_Free(rsize);

    if (tbuf != NULL){
        NCI_Free(tbuf);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_CB)

    return NC_NOERR;
}