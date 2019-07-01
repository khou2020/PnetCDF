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

int
nczipioi_get_var_cb_chunk(NC_zip          *nczipp,
                    NC_zip_var      *varp,
                    const MPI_Offset      *start,
                    const MPI_Offset      *count,
                    const MPI_Offset      *stride,
                    void            *buf)
{
    int err;
    int i, j, k;
    int cid;   // Chunk iterator

    MPI_Offset *ostart, *osize;
    int *tsize, *tssize, *tstart, *tsizep, *tssizep, *tstartp;   // Size for sub-array type
    MPI_Offset *citr; // Chunk iterator
    
    int *rcnt_local, *rcnt_all;   // Number of processes that writes to each chunk

    int overlapsize;    // Size of overlaping region of request and chunk
    char *cbuf = NULL;     // Intermediate continuous buffer
    
    int packoff; // Pack offset
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

    // Iterate through chunks
    nczipioi_chunk_itr_init(varp, start, count, citr, &cid);
    do{
        rcnt_local[cid] = 1;

        if (varp->chunk_owner[cid] != nczipp->rank){
            // Count number of mnessage we need to send
            nsend++;    
        }
    } while (nczipioi_chunk_itr_next(varp, start, count, citr, &cid));

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
        if (rcnt_all[cid] > 0 && varp->chunk_cache[cid] == NULL){
            rids[nread++] = cid;
        }
    }

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
    k = 0;
    // Initialize chunk iterator
    nczipioi_chunk_itr_init_ex(varp, start, count, citr, &cid, ostart, osize);
    // Iterate through chunks
    do{
        // We got something to send if we are not owner
        if (varp->chunk_owner[cid] != nczipp->rank){
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_PACK_REQ)

            // Calculate chunk overlap
            overlapsize = varp->esize;
            for(j = 0; j < varp->ndim; j++){
                overlapsize *= osize[j];                     
            }

            // Allocate buffer
            sbufs[k] = (char*)NCI_Malloc(sizeof(int) * varp->ndim * 2); // For request
            rbufs[k + nrecv] = (char*)NCI_Malloc(overlapsize);   // For reply, first nrecv are for request

            // Metadata
            tstartp = (int*)sbufs[k]; packoff = varp->ndim * sizeof(int);
            tsizep = (int*)(sbufs[k] + packoff); packoff += varp->ndim * sizeof(int);
            for(j = 0; j < varp->ndim; j++){
                tstartp[j] = (int)(ostart[j] - citr[j]);
                tsizep[j] = (int)osize[j];
            }

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_PACK_REQ)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REQ)

            // Send request
            CHK_ERR_ISEND(sbufs[k], packoff, MPI_BYTE, varp->chunk_owner[cid], cid, nczipp->comm, sreqs + k);
            
            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REQ)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REP)

            // Post recv reply
            CHK_ERR_IRECV(rbufs[k + nrecv], overlapsize, MPI_BYTE, varp->chunk_owner[cid], cid + 1024, nczipp->comm, rreqs + nrecv + k);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REP)

            k++;
        }
    } while (nczipioi_chunk_itr_next_ex(varp, start, count, citr, &cid, ostart, osize));

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REQ)

    // Post recv
    k = 0;
    for(i = 0; i < varp->nmychunk; i++){
        cid = varp->mychunks[i];
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
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REQ)

    // Allocate intermediate buffer
    cbuf = (char*)NCI_Malloc(varp->chunksize);

    // For each chunk we own, we need to receive incoming data
    k = 0;
    for(i = 0; i < varp->nmychunk; i++){
        cid = varp->mychunks[i];

        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SELF)

        // Handle our own data first if we have any
        if (rcnt_local[cid] > 0){
            // Convert chunk id to iterator
            get_chunk_itr(varp, cid, citr);

            // Calculate overlapping region
            get_chunk_overlap(varp, citr, start, count, ostart, osize);
            
            // Pack type from chunk buffer to (contiguous) intermediate buffer
            for(j = 0; j < varp->ndim; j++){
                tstart[j] = (int)(ostart[j] - citr[j]);
                tsize[j] = varp->chunkdim[j];
                tssize[j] = (int)osize[j];
            }
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
            CHK_ERR_TYPE_COMMIT(&ptype);

            // Pack data into intermediate buffer
            packoff = 0;
            CHK_ERR_PACK(varp->chunk_cache[cid], 1, ptype, cbuf, varp->chunksize, &packoff, nczipp->comm);
            overlapsize = packoff;
            MPI_Type_free(&ptype);

            // Pack type from (contiguous) intermediate buffer to user buffer
            for(j = 0; j < varp->ndim; j++){
                tstart[j] = (int)(ostart[j] - start[j]);
                tsize[j] = (int)count[j];
            }
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
            CHK_ERR_TYPE_COMMIT(&ptype);

            // Pack data into user buffer
            packoff = 0;
            CHK_ERR_UNPACK(cbuf, overlapsize, &packoff, buf, 1, ptype, nczipp->comm);
            MPI_Type_free(&ptype);
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SELF)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REQ)

        // Wait for all send requests related to this chunk
        // We remove the impact of -1 mark in rcnt_local[cid]
        //printf("Rank: %d, CHK_ERR_WAITALL_recv(%d, %d)\n", nczipp->rank, rcnt_all[cid] - rcnt_local[cid], k); fflush(stdout);
        CHK_ERR_WAITALL(rcnt_all[cid] - rcnt_local[cid], rreqs + k, rstats + k);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REQ)

        // Now, it is time to process data from other processes
        for(j = 0; j < varp->ndim; j++){
            tsize[j] = varp->chunkdim[j];
        }

        // Process data received
        //printf("nrecv = %d, rcnt_all = %d, rcnt_local = %d\n", nrecv, rcnt_all[cid], rcnt_local[cid]); fflush(stdout);
        for(j = k; j < k + rcnt_all[cid] - rcnt_local[cid]; j++){
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)
            
            // Metadata
            tstartp = (int*)rbufs[j]; packoff = varp->ndim * sizeof(int);
            tssizep = (int*)(rbufs[j] + packoff); packoff += varp->ndim * sizeof(int);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_PACK_REP)

            // Pack type
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssizep, tstartp, MPI_ORDER_C, varp->etype, &ptype);
            CHK_ERR_TYPE_COMMIT(&ptype);

            // Allocate buffer 
            MPI_Type_size(ptype, &overlapsize);
            sbufs[j + nsend] = (char*)NCI_Malloc(overlapsize); // For reply
            
            // Data
            packoff = 0;
            CHK_ERR_PACK(varp->chunk_cache[cid], 1, ptype, sbufs[j + nsend], overlapsize, &packoff, nczipp->comm);
            MPI_Type_free(&ptype);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_PACK_REP)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REP)

            // Send reply
            CHK_ERR_ISEND(sbufs[j + nsend], packoff, MPI_BYTE, rstats[j].MPI_SOURCE, cid + 1024, nczipp->comm, sreqs + j + nsend);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REP)
        }
        k += rcnt_all[cid] - rcnt_local[cid];        

        //princbuf(nczipp->rank, varp->chunk_cache[cid], varp->chunksize);
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REQ)

    // Wait for all send request
    //printf("Rank: %d, CHK_ERR_WAITALL_send(%d, %d)\n", nczipp->rank, nsend, 0); fflush(stdout);
    CHK_ERR_WAITALL(nsend, sreqs, sstats);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REQ)

    // Receive replies from the owners and update the user buffer
    k = 0;
    // Initialize chunk iterator
    nczipioi_chunk_itr_init_ex(varp, start, count, citr, &cid, ostart, osize);
    // Iterate through chunks
    do{
        // We got something to recv if we are not owner
        if (varp->chunk_owner[cid] != nczipp->rank){
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_UNPACK_REP)
            
            // Pack type from recv buffer to user buffer
            for(j = 0; j < varp->ndim; j++){
                tstart[j] = (int)(ostart[j] - start[j]);
                tsize[j] = (int)count[j];
                tssize[j] = (int)osize[j];
            }
            //printf("Rank: %d, ostart=[%lld, %lld], osize=[%lld, %lld]\n", nczipp->rank, ostart[0], ostart[1], osize[0], osize[1]); fflush(stdout);
            //printf("Rank: %d, CHK_ERR_TYPE_CREATE_SUBARRAY4([%d, %d], [%d, %d], [%d, %d]\n", nczipp->rank, tsize[0], tsize[1], tssize[0], tssize[1], tstart[0], tstart[1]); fflush(stdout);
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
            //printf("Rank: %d, commit\n", nczipp->rank); fflush(stdout);
            CHK_ERR_TYPE_COMMIT(&ptype);
            MPI_Type_size(ptype, &overlapsize);

            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REP)

            //printf("Rank: %d, wait recv, nrecv = %d, k = %d, nsend = %d\n", nczipp->rank, nrecv, k, nsend); fflush(stdout);
            // Wait for reply
            //printf("Rank: %d, MPI_Wait_recv(%d)\n", nczipp->rank, nrecv + k); fflush(stdout);
            MPI_Wait(rreqs + nrecv + k, rstats + nrecv + k);

            NC_ZIP_TIMER_STOPEX(NC_ZIP_TIMER_GET_CB_RECV_REP, NC_ZIP_TIMER_GET_CB_UNPACK_REP)

            // Pack data
            //printf("Rank: %d, pack\n", nczipp->rank); fflush(stdout);
            packoff = 0;
            CHK_ERR_UNPACK(rbufs[nrecv + k], overlapsize, &packoff, buf, 1, ptype, nczipp->comm);
            MPI_Type_free(&ptype);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_UNPACK_REP)

            k++;
        }
    } while (nczipioi_chunk_itr_next_ex(varp, start, count, citr, &cid, ostart, osize));

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REP)

    //printf("Rank: %d, wait_final\n", nczipp->rank); fflush(stdout);
    // Wait for all send replies
    //printf("Rank: %d, CHK_ERR_WAITALL_send(%d, %d)\n", nczipp->rank, nrecv, nsend); fflush(stdout);
    CHK_ERR_WAITALL(nrecv, sreqs + nsend, sstats + nsend);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REP)

    //printf("Rank: %d, exiting\n", nczipp->rank); fflush(stdout);

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

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REP)

    return NC_NOERR;
}


int
nczipioi_get_var_cb_proc(      NC_zip          *nczipp,
                            NC_zip_var      *varp,
                            const MPI_Offset      *start,
                            const MPI_Offset      *count,
                            const MPI_Offset      *stride,
                            void            *buf)
{
    int err;
    int i, j, k;
    int cid, cown;   // Chunk iterator

    MPI_Offset *ostart, *osize;
    int *tsize, *tssize, *tstart, *tssizep, *tstartp;   // Size for sub-array type
    MPI_Offset *citr; // Chunk iterator
    
    int *rcnt_local, *rcnt_all;   // Number of processes that writes to each proc
    int *rcnt_local_chunk, *rcnt_all_chunk;   // Number of processes that writes to each chunk

    int overlapsize;    // Size of overlaping region of request and chunk
    int max_tbuf = 0;   // Size of intermediate buffer
    char *tbuf = NULL;     // Intermediate buffer
    
    int packoff; // Pack offset
    MPI_Datatype ptype; // Pack datatype

    int nread;  // # chunks to read form file
    int *rids;  // Id of chunks to read from file

    int nsend, nrecv;   // Number of send and receive
    MPI_Request *sreq, *rreq, *sreq_re, *rreq_re;    // Send and recv req
    MPI_Status *sstat, rstat, *sstat_re;    // Send and recv status
    char **sbuf, **rbuf, **sbufp, **rbufp, **sbuf_re, **rbuf_re;   // Send and recv buffer
    int *rsize, *ssize, *rsize_re, *ssize_re;    // recv size of each message
    int *sdst;    // recv size of each message
    int *smap;
    MPI_Message rmsg;   // Receive message

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_INIT)

    // Allocate buffering for write count
    rcnt_local = (int*)NCI_Malloc(sizeof(int) * (nczipp->np * 3 + varp->nchunk * 2));
    rcnt_local_chunk = rcnt_local + nczipp->np;
    rcnt_all = rcnt_local_chunk + varp->nchunk;
    rcnt_all_chunk = rcnt_all + nczipp->np;
    smap = rcnt_all_chunk + varp->nchunk;

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
    memset(rcnt_local, 0, sizeof(int) * (nczipp->np + varp->nchunk));
    nsend = 0;

    // Count total number of messages and build a map of accessed chunk to list of comm datastructure
    nczipioi_chunk_itr_init(varp, start, count, citr, &cid); // Initialize chunk iterator
    do{
        // Chunk owner
        cown = varp->chunk_owner[cid];

        // Mapping to skip list of send requests 
        if (rcnt_local[cown] == 0 && cown != nczipp->rank){
            smap[cown] = nsend++;
        }
        rcnt_local[cown] = 1;   // Need to send message if not owner     
        rcnt_local_chunk[cid] = 1;  // This tells the owner to prepare the chunks  
    } while (nczipioi_chunk_itr_next(varp, start, count, citr, &cid));

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_INIT)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SYNC)

    // Sync number of messages of each chunk
    CHK_ERR_ALLREDUCE(rcnt_local, rcnt_all, nczipp->np + varp->nchunk, MPI_INT, MPI_SUM, nczipp->comm);
    nrecv = rcnt_all[nczipp->rank] - rcnt_local[nczipp->rank];  // We don't need to receive request form self

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SYNC)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_INIT)

    // We need to prepare chunk in the chunk cache
    // For chunks not yet allocated, we need to read them form file collectively
    // We collect chunk id of those chunks
    // Calculate number of recv request
    // This is for all the chunks
    rids = (int*)NCI_Malloc(sizeof(int) * varp->nmychunk);
    nread = 0;
    for(i = 0; i < varp->nmychunk; i++){
        cid = varp->mychunks[i];
        // Count number of chunks we need to prepare
        // We read only chunks that is required
        if (rcnt_all_chunk[cid] > 0 && varp->chunk_cache[cid] == NULL){
            rids[nread++] = cid;
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB)  // I/O time count separately

    // Decompress chunks into chunk cache
    nczipioi_load_var(nczipp, varp, nread, rids);
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_PACK_REQ)

    // Allocate data structure for messaging
    sbuf = (char**)NCI_Malloc(sizeof(char*) * (2 * nsend + nrecv));
    ssize = (int*)NCI_Malloc(sizeof(int) * (nsend * 2 + nrecv * 1));
    sdst = ssize + (nsend + nrecv);
    sreq = (MPI_Request*)NCI_Malloc(sizeof(MPI_Request) * (nsend + nrecv));
    sstat = (MPI_Status*)NCI_Malloc(sizeof(MPI_Status) * (nsend + nrecv));

    rbuf = (char**)NCI_Malloc(sizeof(char*) * (nsend + nrecv * 2));
    rsize = (int*)NCI_Malloc(sizeof(int) * (nsend + nrecv));
    rreq = (MPI_Request*)NCI_Malloc(sizeof(MPI_Request) * (nsend + nrecv));

    sbuf_re = sbuf + nsend;
    sbufp = sbuf_re + nrecv;
    ssize_re = ssize + nsend;
    sreq_re = sreq + nsend;
    sstat_re = sstat + nsend;

    rbuf_re = rbuf + nrecv;
    rbufp = rbuf_re + nsend;
    rsize_re = rsize + nrecv;
    rreq_re = rreq + nrecv;

    // Count size of each request
    memset(ssize, 0, sizeof(int) * nsend);
    memset(rsize_re, 0, sizeof(int) * nsend);
    nczipioi_chunk_itr_init_ex(varp, start, count, citr, &cid, ostart, osize); // Initialize chunk iterator
    do{
        // Chunk owner               
        cown = varp->chunk_owner[cid];
        if (cown != nczipp->rank){
            j = smap[cown];
            sdst[j] = cown; // Record a reverse map by the way

            // Count overlap
            overlapsize = varp->esize;
            for(i = 0; i < varp->ndim; i++){
                overlapsize *= osize[i];                     
            }
            ssize[j] += sizeof(int) * (varp->ndim * 2 + 1);
            rsize_re[j] += overlapsize;
        }
    } while (nczipioi_chunk_itr_next_ex(varp, start, count, citr, &cid, ostart, osize));

    // Allocate buffer for send
    for(i = 0; i < nsend; i++){
        ssize[i] += sizeof(int);
        sbuf[i] = sbufp[i] = (char*)NCI_Malloc(ssize[i]);
        *((int*)sbufp[i]) = rsize_re[i];    sbufp[i] += sizeof(int);
        rbuf_re[i] = (char*)NCI_Malloc(rsize_re[i]);
    }

    // Pack requests
    nczipioi_chunk_itr_init_ex(varp, start, count, citr, &cid, ostart, osize); // Initialize chunk iterator
    do{
        // Chunk owner
        cown = varp->chunk_owner[cid];
        if (cown != nczipp->rank){
            j = smap[cown];

            // Metadata
            *((int*)sbufp[j]) = cid;    sbufp[j] += sizeof(int);
            tstartp = (int*)sbufp[j];    sbufp[j] += sizeof(int) * varp->ndim;
            tssizep = (int*)sbufp[j];    sbufp[j] += sizeof(int) * varp->ndim;
            for(i = 0; i < varp->ndim; i++){
                tstartp[i] = (int)(ostart[i] - citr[i]);
                tssizep[i] = (int)osize[i];
            }
        }
    } while (nczipioi_chunk_itr_next_ex(varp, start, count, citr, &cid, ostart, osize));

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_PACK_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REQ)

    // Post send 
    for(i = 0; i < nsend; i++){
        CHK_ERR_ISEND(sbuf[i], ssize[i], MPI_BYTE, sdst[i], 0, nczipp->comm, sreq + i);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REP)

    // Post receive  
    for(i = 0; i < nsend; i++){
        CHK_ERR_IRECV(rbuf_re[i], rsize_re[i], MPI_BYTE, sdst[i], 1, nczipp->comm, rreq_re + i);
    }   

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REP)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REQ)

    // Post recv
    for(i = 0; i < nrecv; i++){
        // Get message size, including metadata
        CHK_ERR_MPROBE(MPI_ANY_SOURCE, 0, nczipp->comm, &rmsg, &rstat);
        CHK_ERR_GET_COUNT(&rstat, MPI_BYTE, rsize + i);

        // Allocate buffer
        rbuf[i] = rbufp[i] = (char*)NCI_Malloc(rsize[i]);

        // Post irecv
        CHK_ERR_IMRECV(rbuf[i], rsize[i], MPI_BYTE, &rmsg, rreq + i);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SELF)

    tbuf = (char*)NCI_Malloc(varp->chunksize);

    // Handle our own data
    nczipioi_chunk_itr_init_ex(varp, start, count, citr, &cid, ostart, osize); // Initialize chunk iterator
    do{
        if (varp->chunk_owner[cid] == nczipp->rank){
            // Pack type from chunk cache to (contiguous) intermediate buffer
            for(j = 0; j < varp->ndim; j++){
                tstart[j] = (int)(ostart[j] - citr[j]);
                tsize[j] = varp->chunkdim[j];
                tssize[j] = (int)osize[j];
            }
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
            CHK_ERR_TYPE_COMMIT(&ptype);

            // Pack data into intermediate buffer
            packoff = 0;
            CHK_ERR_PACK(varp->chunk_cache[cid], 1, ptype, tbuf, varp->chunksize, &packoff, nczipp->comm);
            MPI_Type_free(&ptype);
            overlapsize = packoff;

            // Pack type from (contiguous) intermediate buffer to chunk buffer
            for(j = 0; j < varp->ndim; j++){
                tstart[j] = (int)(ostart[j] - start[j]);
                tsize[j] = (int)count[j];
            }            
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
            CHK_ERR_TYPE_COMMIT(&ptype);
            
            // Unpack data into chunk buffer
            packoff = 0;
            CHK_ERR_UNPACK(tbuf, overlapsize, &packoff, buf, 1, ptype, nczipp->comm);
            MPI_Type_free(&ptype);    
        }
    } while (nczipioi_chunk_itr_next_ex(varp, start, count, citr, &cid, ostart, osize));

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SELF)

    //Handle incoming requests
    for(i = 0; i < varp->ndim; i++){
        tsize[i] = varp->chunkdim[i];
    }
    for(i = 0; i < nrecv; i++){
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REQ)

        // Will wait any provide any benefit?
        MPI_Waitany(nrecv, rreq, &j, &rstat);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REQ)

        packoff = 0;
        ssize_re[j] = *((int*)rbufp[j]);    rbufp[j] += sizeof(int);
        sbuf_re[j] = (char*)NCI_Malloc(ssize_re[j]);
        while(rbufp[j] < rbuf[j] + rsize[j]){
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)

            // Metadata
            cid = *((int*)rbufp[j]);    rbufp[j] += sizeof(int);
            tstartp = (int*)rbufp[j];    rbufp[j] += sizeof(int) * varp->ndim;
            tssizep = (int*)rbufp[j];    rbufp[j] += sizeof(int) * varp->ndim;

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_PACK_REP)

            // Pack type
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssizep, tstartp, MPI_ORDER_C, varp->etype, &ptype);
            CHK_ERR_TYPE_COMMIT(&ptype);

            // Data
            CHK_ERR_PACK(varp->chunk_cache[cid], 1, ptype, sbuf_re[j], ssize_re[j], &packoff, nczipp->comm);
            MPI_Type_free(&ptype);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_PACK_REP)
        }

        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REQ)

        // Send Response
        CHK_ERR_ISEND(sbuf_re[j], packoff, MPI_BYTE, rstat.MPI_SOURCE, 1, nczipp->comm, sreq_re + j);
        
        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REQ)
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REQ)

    // Wait for all request
    CHK_ERR_WAITALL(nsend, sreq, sstat);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REQ)

    //Handle reply
    for(i = 0; i < varp->ndim; i++){
        tsize[i] = count[i];
    }
    for(i = 0; i < nsend; i++){
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REP)

        // Will wait any provide any benefit?
        MPI_Waitany(nsend, rreq_re, &j, &rstat);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REP)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_UNPACK_REP)

        sbufp[j] = sbuf[j] + sizeof(int);  // Skip reply size
        packoff = 0;
        while(packoff < rsize_re[j]){
            // Retrieve metadata from the request we sent
            cid = *((int*)sbufp[j]);    sbufp[j] += sizeof(int);
            tstartp = (int*)sbufp[j];   sbufp[j] += sizeof(int) * varp->ndim;
            tssizep = (int*)sbufp[j];    sbufp[j] += sizeof(int) * varp->ndim;

            // Bring back the request
            get_chunk_itr(varp, cid, citr);
            for(k = 0; k < varp->ndim; k++){
                tstartp[k] += (int)(citr[k] - start[k]);
            }

            // Pack type
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssizep, tstartp, MPI_ORDER_C, varp->etype, &ptype);
            CHK_ERR_TYPE_COMMIT(&ptype);

            // Pack data
            CHK_ERR_UNPACK(rbuf_re[j], rsize_re[j], &packoff, buf, 1, ptype, nczipp->comm);
            MPI_Type_free(&ptype);
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_UNPACK_REP)
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REP)

    // Wait for all Response
    CHK_ERR_WAITALL(nrecv, sreq_re, sstat_re);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REP)

    // Free buffers
    NCI_Free(rcnt_local);

    NCI_Free(rids);

    NCI_Free(tsize);

    NCI_Free(ostart);

    NCI_Free(sreq);
    NCI_Free(sstat);
    NCI_Free(ssize);
    for(i = 0; i < nsend + nrecv; i++){
        NCI_Free(sbuf[i]);
        NCI_Free(rbuf[i]);
    }
    NCI_Free(sbuf);

    NCI_Free(rreq);
    NCI_Free(rbuf);
    NCI_Free(rsize);

    if (tbuf != NULL){
        NCI_Free(tbuf);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB)

    return NC_NOERR;
}
