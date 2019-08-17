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
            CHK_ERR_PACK(varp->chunk_cache[cid]->buf, 1, ptype, cbuf, varp->chunksize, &packoff, nczipp->comm);
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
            CHK_ERR_PACK(varp->chunk_cache[cid]->buf, 1, ptype, sbufs[j + nsend], overlapsize, &packoff, nczipp->comm);
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


int nczipioi_get_var_cb_proc(      NC_zip          *nczipp,
                            NC_zip_var      *varp,
                            const MPI_Offset      *start,
                            const MPI_Offset      *count,
                            const MPI_Offset      *stride,
                            void            *buf) {
    int err;
    int i, j, k, l;
    int cid, cown; // Chunk iterator and owner
    int vid;
    int r;
    MPI_Offset *ostart, *osize;
    int *tsize, *tssize, *tstart, *tssizep, *tstartp; // Size for sub-array type
    MPI_Offset *citr;                                 // Bounding box for chunks overlapping my own write region

    int nread;
    int *rids;
    int wrange[4];   // Number of processes that writes to each chunk

    int *scnt, *ssize;
    int *rcnt, *rsize;

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
    MPI_Request *rreq;
    MPI_Status *rstat;

    MPI_Message rmsg; // Receive message

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_INIT)

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
    scnt = (int *)NCI_Malloc(sizeof(int) * nczipp->np * 6);
    ssize = scnt + nczipp->np;
    rcnt = ssize + nczipp->np;
    rsize = rcnt + nczipp->np;
    smap = rsize + nczipp->np;
    rmap = smap + nczipp->np;

    // Count total number of messages and build a map of accessed chunk to list of comm datastructure
    memset(scnt, 0, sizeof(int) * nczipp->np * 2);
    npack = 0;
    nczipioi_chunk_itr_init(varp, start, count, citr, &cid); // Initialize chunk iterator
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
    } while (nczipioi_chunk_itr_next(varp, start, count, citr, &cid));
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
                l += sntypes[j];
                ssize[i] = sizeof(int) * scnt[j] * (varp->ndim * 2 + 1);
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

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_INIT)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_PACK_REQ)

    // Pack requests
    nczipioi_chunk_itr_init_ex(varp, start, count, citr, &cid, ostart, osize); // Initialize chunk iterator
    do {
        // Chunk index and owner
        cown = varp->chunk_owner[cid];

        j = smap[cown];

        // Pack metadata
        *((int *)(sbufp[j])) = cid; sbufp[j] += sizeof(int);
        tstartp = (int *)sbufp[j];  sbufp[j] += varp->ndim * sizeof(int);
        tssizep = (int *)sbufp[j];  sbufp[j] += varp->ndim * sizeof(int);

        for (i = 0; i < varp->ndim; i++){
            tstartp[i] = (int)(ostart[i] - citr[i]);
            tssizep[i] = (int)osize[i];
        }

        // Pack type from user buffer to send buffer
        for (i = 0; i < varp->ndim; i++){
            tsize[i] = (int)count[i];
            tstart[i] = (int)(ostart[i] - start[i]);
        }

        err = nczipioi_subarray_off_len(varp->ndim, tsize, tssizep, tstart, soffsp[j], slensp[j]);
        if (err){
            CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssizep, tstart, MPI_ORDER_C, varp->etype, stypesp[j]);
            CHK_ERR_TYPE_COMMIT(stypesp[j]);
            *(soffsp[j]) = buf;
            *(slensp[j]) = 1;
        }
        else{
            *(stypesp[j]) = MPI_BYTE;
            *(soffsp[j]) = (*(soffsp[j])) * varp->esize + buf;
            *(slensp[j]) *= varp->esize;
        }
        stypesp[j]++;
        soffsp[j]++;
        slensp[j]++;

#ifdef PNETCDF_PROFILING
        nczipp->nsend++;
#endif
    } while (nczipioi_chunk_itr_next_ex(varp, start, count, citr, &cid, ostart, osize));

    // Construct compound type
    for (i = 0; i < npack; i++) {
        MPI_Type_struct(sntypes[i], slens[i], soffs[i], stypes[i], stype + i);
        CHK_ERR_TYPE_COMMIT(stype + i);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_PACK_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SYNC)

    // Sync number of write in message
    CHK_ERR_ALLTOALL(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, nczipp->comm);

    // Sync message size
    CHK_ERR_ALLTOALL(ssize, 1, MPI_INT, rsize, 1, MPI_INT, nczipp->comm);

    // Access range
    wrange[1] *= -1;
    CHK_ERR_ALLREDUCE(wrange, wrange + 2, 2, MPI_INT, MPI_MIN, nczipp->comm);
    wrange[3] *= -1;

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SYNC)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_INIT)

    j = k = 0;
    nunpack = 0; // We don't need to receive request form self
    for (i = 0; i < nczipp->np; i++){
        if ((rcnt[i] > 0) && (i != nczipp->rank)){
            rmap[nunpack] = i;
            rsize[nunpack] = rsize[i];
            rcnt[nunpack++] = rcnt[i];

            k += rsize[i];
            j += rcnt[i];
        }
    }
    nrecv = nunpack;
    if (rcnt[nczipp->rank]){
        rmap[nunpack] = nczipp->rank;
        rsize[nunpack] = rsize[nczipp->rank];
        rcnt[nunpack++] = rcnt[nczipp->rank];

        j += rcnt[nczipp->rank];
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

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_INIT)

    // Post send req
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REQ)
    for (i = 0; i < nsend; i++) {
        CHK_ERR_ISEND(sbuf[i], ssize[sdst[i]], MPI_BYTE, sdst[i], 0, nczipp->comm, sreq + i);
    }
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REQ)

    // Post recv rep
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REP)
    for (i = 0; i < nsend; i++) {
        CHK_ERR_IRECV(MPI_BOTTOM, 1, stype[i], sdst[i], 1, nczipp->comm, sreq + nsend + i);
    }
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REP)

    // Post recv req
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REQ)
    for (i = 0; i < nrecv; i++) {
        CHK_ERR_IRECV(rbuf[i], rsize[i], MPI_BYTE, rmap[i], 0, nczipp->comm, rreq + i);
    }
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REQ)

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
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SELF)
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
                *roffsp = varp->chunk_cache[cid]->buf;
                *rlensp = 1;
            }
            else{
                *rtypesp = MPI_BYTE;
                *roffsp = (*roffsp) * varp->esize + varp->chunk_cache[cid]->buf;
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
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SELF)

    //Handle incoming requests
    for (i = 0; i < nrecv; i++) {
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REQ)

        // Will wait any provide any benefit?
        MPI_Waitany(nrecv, rreq, &j, rstat);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REQ)


        rbufp = rbuf[j];
        rtypesp = rtypes[j];
        roffsp = roffs[j];
        rlensp = rlens[j];
        for (k = 0; k < rcnt[j]; k++) {
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)
            // Retrieve metadata
            cid = *((int *)(rbufp));            rbufp += sizeof(int);
            tstartp = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);
            tssizep = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_UNPACK_REQ)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_PACK_REP)

            // Subarray type
            err = nczipioi_subarray_off_len(varp->ndim, varp->chunkdim, tssizep, tstartp, roffsp, rlensp);
            if (err){
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, varp->chunkdim, tssizep, tstartp, MPI_ORDER_C, varp->etype, rtypesp);
                CHK_ERR_TYPE_COMMIT(rtypesp);
                *roffsp = varp->chunk_cache[cid]->buf;
                *rlensp = 1;
            }
            else{
                *rtypesp = MPI_BYTE;
                *roffsp = (*roffsp) * varp->esize + varp->chunk_cache[cid]->buf;
                *rlensp *= varp->esize;
            }
            rtypesp++;
            roffsp++;
            rlensp++;

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_PACK_REP)

            // Mark chunk as dirty
            varp->dirty[cid] = 1;
#ifdef PNETCDF_PROFILING
            nczipp->nrecv++;
#endif
        }

        // Send data
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REP)
        MPI_Type_struct(rcnt[j], rlens[j], roffs[j], rtypes[j], rtype + j);
        CHK_ERR_TYPE_COMMIT(rtype + j);
        CHK_ERR_ISEND(MPI_BOTTOM, 1, rtype[j], rstat->MPI_SOURCE, 1, nczipp->comm, rreq + nrecv + i);
        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REP)
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_RECV_REP)
    CHK_ERR_WAITALL(nsend, sreq + nsend, sstat);
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_RECV_REP)

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REQ)
    CHK_ERR_WAITALL(nsend, sreq, sstat);
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REQ)

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_CB_SEND_REP)
    CHK_ERR_WAITALL(nrecv, rreq + nrecv, rstat);
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB_SEND_REP)

    // Free type
    for (i = 0; i < npack; i++) {
        for (j = 0; j < sntypes[i]; j++) {
            if (stypes[i][j] != MPI_BYTE) {
                MPI_Type_free(stypes[i] + j);
            }
        }
        MPI_Type_free(stype + i);
    }

    for (i = 0; i < nrecv; i++) {
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
    NCI_Free(rtypes);
    NCI_Free(rtype);
    NCI_Free(roffs);
    NCI_Free(rlens);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_CB)

    return NC_NOERR;
}
