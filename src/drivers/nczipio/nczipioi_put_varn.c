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

int printbuf(int rank, char* buf, int len){
    int i;
    printf("Rank %d: ", rank);
    for(i = 0; i < len; i++){
        printf("%02x ", buf[i]);
    }
    printf("\n");
}

int
nczipioi_put_varn_cb_chunk(  NC_zip        *nczipp,
                    NC_zip_var       *varp,
                    int              nreq,
                    MPI_Offset* const *starts,
                    MPI_Offset* const *counts,
                    MPI_Offset* const *strides,
                    void              **bufs)
{
    int err;
    int i, j, k;
    int cid, req;   // Chunk and request iterator

    int *tsize, *tssize, *tstart, *tsizep, *tstartp;   // Size for sub-array type
    MPI_Offset *ostart, *osize;
    MPI_Offset *citr;
    
    int *wcnt_local, *wcnt_all;   // Number of processes that writes to each chunk

    int nread;  // Chunks to read for background
    int *rids;

    int overlapsize;    // Size of overlaping region of request and chunk
    int max_tbuf;   // Size of intermediate buffer
    char *tbuf = NULL;     // Intermediate buffer
    
    int packoff; // Pack offset
    MPI_Datatype ptype; // Pack datatype

    int nsend, nrecv;   // Number of send and receive
    MPI_Request *sreqs, *rreqs;    // Send and recv req
    MPI_Status *sstats, *rstats;    // Send and recv status
    char **sbufs, **rbufs;   // Send and recv buffer
    int *rsizes;    // recv size of each message
    MPI_Message rmsg;   // Receive message

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_INIT)

    // Allocate buffering for write count
    wcnt_local = (int*)NCI_Malloc(sizeof(int) * varp->nchunk * 2);
    wcnt_all = wcnt_local + varp->nchunk;

    // Allocate buffering for overlaping index
    tstart = (int*)NCI_Malloc(sizeof(int) * varp->ndim * 3);
    tsize = tstart + varp->ndim;
    tssize = tsize + varp->ndim;
    ostart = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * varp->ndim * 3);
    osize = ostart + varp->ndim;

    // Chunk iterator
    citr = osize + varp->ndim;

    // We need to calculate the size of message of each chunk
    // This is just for allocating send buffer
    // We do so by iterating through all request and all chunks they cover
    // If we are not the owner of a chunk, we need to send message
    memset(wcnt_local, 0, sizeof(int) * varp->nchunk);
    nsend = 0;
    max_tbuf = 0;
    for(req = 0; req < nreq; req++){
        // Initialize chunk iterator
        nczipioi_chunk_itr_init_ex(varp, starts[req], counts[req], citr, &cid, ostart, osize); // Initialize chunk iterator

        // Iterate through chunks
        do{
            // Calculate overlapping
            overlapsize = varp->esize;
            for(j = 0; j < varp->ndim; j++){
                overlapsize *= osize[j];                     
            }

            if (varp->chunk_owner[cid] != nczipp->rank){
                // Count number of mnessage we need to send
                if (wcnt_local[cid] == 0){
                    nsend++;
                }
                wcnt_local[cid] += overlapsize + sizeof(int) * 2 * varp->ndim;
            }
            else{
                // We mark covered chunk of our own to prevent unnecessary calculation of overlap
                // -1 is purely a mark, we need to add 1 back to global message count
                wcnt_local[cid] = -1;

                // Record max overlapsize so we know how large the intermediate buffer is needed later
                if (max_tbuf < overlapsize){
                    max_tbuf = overlapsize;
                }
            }

        } while (nczipioi_chunk_itr_next_ex(varp, starts[req], counts[req], citr, &cid, ostart, osize));
    }

    // Allocate buffer for sending
    sbufs = (char**)NCI_Malloc(sizeof(char*) * nsend);
    sreqs = (MPI_Request*)NCI_Malloc(sizeof(MPI_Request) * nsend);
    sstats = (MPI_Status*)NCI_Malloc(sizeof(MPI_Status) * nsend);
    j = 0;
    // Allocate buffer for data
    for(cid = 0; cid < varp->nchunk; cid++){
        // Count number of mnessage we need to send
        if (wcnt_local[cid] > 0){
            // Add space for number of reqs
            sbufs[j++] = (char*)NCI_Malloc(wcnt_local[cid]);
            // We don't need message size anymore, wcnt_local is used to track number of message from now on 
            wcnt_local[cid] = 1;
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_INIT)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SYNC)

    // Sync number of messages of each chunk
    CHK_ERR_ALLREDUCE(wcnt_local, wcnt_all, varp->nchunk, MPI_INT, MPI_SUM, nczipp->comm);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SYNC)

    // Calculate number of recv request
    // This is for all the chunks
    nrecv = 0;
    for(i = 0; i < varp->nmychunk; i++){
        cid = varp->mychunks[i];
        // We don't need message for our own data
        nrecv += wcnt_all[cid] - wcnt_local[cid];
    }
    rreqs = (MPI_Request*)NCI_Malloc(sizeof(MPI_Request) * nrecv);
    rstats = (MPI_Status*)NCI_Malloc(sizeof(MPI_Status) * nrecv);
    rbufs = (char**)NCI_Malloc(sizeof(char*) * nrecv);
    rsizes = (int*)NCI_Malloc(sizeof(int) * nrecv);

    // Post send and recv
    nrecv = 0;
    nsend = 0;
    for(cid = 0; cid < varp->nchunk; cid++){
        if (varp->chunk_owner[cid] == nczipp->rank){
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_RECV_REQ)

            // We are the owner of the chunk
            // Receive data from other process
            for(i = 0; i < wcnt_all[cid] - wcnt_local[cid]; i++){
                // Get message size, including metadata
                CHK_ERR_MPROBE(MPI_ANY_SOURCE, cid, nczipp->comm, &rmsg, rstats);
                CHK_ERR_GET_COUNT(rstats, MPI_BYTE, rsizes + nrecv);

                // Allocate buffer
                rbufs[nrecv] = (char*)NCI_Malloc(rsizes[nrecv]);

                // Post irecv
                CHK_ERR_IMRECV(rbufs[nrecv], rsizes[nrecv], MPI_BYTE, &rmsg, rreqs + nrecv);
                nrecv++;
            }
            
            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_RECV_REQ)
        }
        else{
            // If we any of our request overlap with this chunk, we need to send data
            // We send only 1 message for 1 chunk
            if (wcnt_local[cid] > 0){
                NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_PACK_REQ)
                
                packoff = 0;
                // Get chunk iterator
                get_chunk_itr(varp, cid, citr);  
                for(req = 0; req < nreq; req++){
                    // Calculate chunk overlap
                    overlapsize = get_chunk_overlap(varp, citr, starts[req], counts[req], ostart, osize);

                    // If current request have any overlap with the chunk, we pack the data and metadata
                    if (overlapsize > 0){
                        // Metadata
                        tstartp = (int*)(sbufs[nsend] + packoff); packoff += varp->ndim * sizeof(int);
                        tsizep = (int*)(sbufs[nsend] + packoff); packoff += varp->ndim * sizeof(int);
                        for(j = 0; j < varp->ndim; j++){
                            tstartp[j] = (int)(ostart[j] - citr[j]);
                            tsizep[j] = (int)osize[j];
                        }

                        // Pack type
                        for(j = 0; j < varp->ndim; j++){
                            tstart[j] = (int)(ostart[j] - starts[req][j]);
                            tsize[j] = (int)counts[req][j];
                            tssize[j] = (int)osize[j];
                        }
                        CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
                        CHK_ERR_TYPE_COMMIT(&ptype);
                        
                        // Data
                        CHK_ERR_PACK(bufs[req], 1, ptype, sbufs[nsend], packoff + overlapsize, &packoff, nczipp->comm);
                        MPI_Type_free(&ptype);
                    }
                }

                NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_PACK_REQ)
                NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SEND_REQ)

                // Send the request
                CHK_ERR_ISEND(sbufs[nsend], packoff, MPI_BYTE, varp->chunk_owner[cid], cid, nczipp->comm, sreqs + nsend);
                nsend++;

                NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SEND_REQ)
            }
        }
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SEND_REQ)

    // Wait for all send
    CHK_ERR_WAITALL(nsend, sreqs, sstats);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SEND_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_INIT)

    // Preparing chunk cache
    nread = 0;
    for(i = 0; i < varp->nmychunk; i++){
        cid = varp->mychunks[i];
        if (wcnt_all[cid] && varp->chunk_cache[cid] == NULL){
            if (varp->data_lens[cid] > 0){
                nread++;
            }
        }
    }
    rids = (int*)NCI_Malloc(sizeof(int) * nread);
    nread = 0;
    for(i = 0; i < varp->nmychunk; i++){
        cid = varp->mychunks[i];
        if (wcnt_all[cid] || wcnt_local[cid]){
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

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_INIT)

    // Read background
    nczipioi_load_var_bg(nczipp, varp, nread, rids);

    // Allocate intermediate buffer
    if (max_tbuf > 0){
        tbuf = (char*)NCI_Malloc(max_tbuf);
    }

    // For each chunk we own, we need to receive incoming data
    nrecv = 0;
    for(i = 0; i < varp->nmychunk; i++){
        cid = varp->mychunks[i];

        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SELF)

        // Handle our own data first if we have any
        if (wcnt_local[cid] < 0){
            for(req = 0; req < nreq; req++){
                // Convert chunk id to iterator
                get_chunk_itr(varp, cid, citr);

                // Calculate overlapping region
                overlapsize = get_chunk_overlap(varp, citr, starts[req], counts[req], ostart, osize);

                // If anything overlaps
                if (overlapsize > 0){
                    // Pack type from user buffer to (contiguous) intermediate buffer
                    for(j = 0; j < varp->ndim; j++){
                        tstart[j] = (int)(ostart[j] - starts[req][j]);
                        tsize[j] = (int)counts[req][j];
                        tssize[j] = (int)osize[j];
                    }
                    
                    CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
                    CHK_ERR_TYPE_COMMIT(&ptype);

                    // Pack data into intermediate buffer
                    packoff = 0;
                    CHK_ERR_PACK(bufs[req], 1, ptype, tbuf, overlapsize, &packoff, nczipp->comm);

                    MPI_Type_free(&ptype);

                    // Pack type from (contiguous) intermediate buffer to chunk buffer
                    for(j = 0; j < varp->ndim; j++){
                        tstart[j] = (int)(ostart[j] - citr[j]);
                        tsize[j] = varp->chunkdim[j];
                    }
                    CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssize, tstart, MPI_ORDER_C, varp->etype, &ptype);
                    CHK_ERR_TYPE_COMMIT(&ptype);
                    
                    // Unpack data into chunk buffer
                    packoff = 0;
                    CHK_ERR_UNPACK(tbuf, overlapsize, &packoff, varp->chunk_cache[cid]->buf, 1, ptype, nczipp->comm);

                    MPI_Type_free(&ptype);

                    // Mark chunk as dirty
                    varp->dirty[cid] = 1;
                }
            }
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SELF)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_RECV_REQ)

        // Now, it is time to process data from other processes

        // Wait for all send requests related to this chunk
        // We remove the impact of -1 mark in wcnt_local[cid]
        CHK_ERR_WAITALL(wcnt_all[cid] - wcnt_local[cid], rreqs + nrecv, rstats + nrecv);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_RECV_REQ)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_UNPACK_REQ)

        // Process data received
        for(j = nrecv; j < nrecv + wcnt_all[cid] - wcnt_local[cid]; j++){
            packoff = 0;
            while(packoff < rsizes[j]){
                // Metadata
                tstartp = (int*)(rbufs[j] + packoff); packoff += varp->ndim * sizeof(int);
                tsizep = (int*)(rbufs[j] + packoff); packoff += varp->ndim * sizeof(int);

                // Packtype
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, varp->chunkdim, tsizep, tstartp, MPI_ORDER_C, varp->etype, &ptype);
                CHK_ERR_TYPE_COMMIT(&ptype);

                // Data
                CHK_ERR_UNPACK(rbufs[j], rsizes[j], &packoff, varp->chunk_cache[cid]->buf, 1, ptype, nczipp->comm);
                MPI_Type_free(&ptype);

                // Mark chunk as dirty
                varp->dirty[cid] = 1;
            }
        }
        nrecv += wcnt_all[cid] - wcnt_local[cid];     

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_UNPACK_REQ)
    }

    // Free buffers
    NCI_Free(wcnt_local);

    NCI_Free(tstart);

    NCI_Free(ostart);

    NCI_Free(sreqs);
    NCI_Free(sstats);
    for(i = 0; i < nsend; i++){
        NCI_Free(sbufs[i]);
    }
    NCI_Free(sbufs);

    NCI_Free(rreqs);
    NCI_Free(rstats);
    for(i = 0; i < nrecv; i++){
        NCI_Free(rbufs[i]);
    }
    NCI_Free(rbufs);
    NCI_Free(rsizes);

    if (tbuf != NULL){
        NCI_Free(tbuf);
    }
    
    if (rids != NULL){
        NCI_Free(rids);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB)

    return NC_NOERR;
}

int nczipioi_put_varn_cb_proc(  NC_zip        *nczipp,
                                    NC_zip_var       *varp,
                                    int              nreq,
                                    MPI_Offset* const *starts,
                                    MPI_Offset* const *counts,
                                    void              **bufs) {
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

    int nrecv;
    char **rbuf, *rbufp;
    MPI_Request *rreq;
    MPI_Status rstat;
    MPI_Datatype rtype;
    MPI_Datatype *rtypes;
    MPI_Aint *roffs;
    int *rlens;
    int *rmap;

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
    scnt = (int *)NCI_Malloc(sizeof(int) * nczipp->np * 6);
    ssize = scnt + nczipp->np;
    rcnt = ssize + nczipp->np;
    rsize = rcnt + nczipp->np;
    smap = rsize + nczipp->np;
    rmap = smap + nczipp->np;

    // Count total number of messages and build a map of accessed chunk to list of comm datastructure
    memset(scnt, 0, sizeof(int) * nczipp->np * 2);
    npack = 0;
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
    sreq = (MPI_Request *)NCI_Malloc(sizeof(MPI_Request) * nsend);
    sstat = (MPI_Status *)NCI_Malloc(sizeof(MPI_Status) * nsend);
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
                sntypes[j] = scnt[i] + 1;
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
        for (i = 0; i < nsend; i++) {
            stypesp[i] = stypes[i] + 1;
            soffsp[i] = soffs[i] + 1;
            slensp[i] = slens[i] + 1;

            stypes[i][0] = MPI_BYTE;
            soffs[i][0] = sbuf[i];
            slens[i][0] = ssize[sdst[i]];
        }
        if (npack > i){
            stypesp[i] = stypes[i];
            soffsp[i] = soffs[i];
            slensp[i] = slens[i];
            sntypes[i]--;
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
                tstartp[i] = (int)(ostart[i] - citr[i]);
                tssizep[i] = (int)osize[i];
            }

            // Pack type from user buffer to send buffer
            for (i = 0; i < varp->ndim; i++){
                tsize[i] = (int)counts[r][i];
                tstart[i] = (int)(ostart[i] - starts[r][i]);
            }

            err = nczipioi_subarray_off_len(varp->ndim, tsize, tssizep, tstart, soffsp[j], slensp[j]);
            if (err){
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssizep, tstart, MPI_ORDER_C, varp->etype, stypesp[j]);
                CHK_ERR_TYPE_COMMIT(stypesp[j]);
                *(soffsp[j]) = bufs[r];
                *(slensp[j]) = 1;
            }
            else{
                *(stypesp[j]) = MPI_BYTE;
                *(soffsp[j]) = (*(soffsp[j])) * varp->esize + bufs[r];
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
        MPI_Type_size(stype[i], ssize + sdst[i]);
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_PACK_REQ)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SYNC)

    // Sync number of write in message
    CHK_ERR_ALLTOALL(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, nczipp->comm);

    // Sync message size
    CHK_ERR_ALLTOALL(ssize, 1, MPI_INT, rsize, 1, MPI_INT, nczipp->comm);

    // Access range
    wrange[1] *= -1;
    CHK_ERR_ALLREDUCE(wrange, wrange + 2, 2, MPI_INT, MPI_MIN, nczipp->comm);
    wrange[3] *= -1;

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SYNC)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_INIT)

    j = k = 0;
    nrecv = 0; // We don't need to receive request form self
    for (i = 0; i < nczipp->np; i++){
        if (rsize[i] > 0){
            if (i != nczipp->rank){
                rmap[nrecv] = i;
                rsize[nrecv] = rsize[i];
                rcnt[nrecv++] = rcnt[i];

                k += rsize[i];
            }
            
            if (j < rcnt[i]) {
                j = rcnt[i];
            }
        }
    }

    // Allocate data structure for receving
    rbuf = (char **)NCI_Malloc(sizeof(char *) * nrecv);
    rreq = (MPI_Request *)NCI_Malloc(sizeof(MPI_Request) * nrecv);
    rtypes = (MPI_Datatype *)NCI_Malloc(sizeof(MPI_Datatype) * j);
    roffs = (MPI_Aint *)NCI_Malloc(sizeof(MPI_Aint) * j);
    rlens = (int *)NCI_Malloc(sizeof(int) * j);
    if (nrecv > 0) {
        rbuf[0] = (char *)NCI_Malloc(k);
        for (i = 1; i < nrecv; i++)
        {
            rbuf[i] = rbuf[i - 1] + rsize[i - 1];
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_INIT)

    // Post send
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SEND_REQ)
    for (i = 0; i < nsend; i++) {
        CHK_ERR_ISEND(MPI_BOTTOM, 1, stype[i], sdst[i], 0, nczipp->comm, sreq + i);
    }
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SEND_REQ)

    // Post recv
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_RECV_REQ)
    for (i = 0; i < nrecv; i++) {
        CHK_ERR_IRECV(rbuf[i], rsize[i], MPI_BYTE, rmap[i], 0, nczipp->comm, rreq + i);
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

    err = nczipioi_load_var_bg(nczipp, varp, nread, rids);    CHK_ERR

    // Handle our own data
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SELF)
    if (scnt[nczipp->rank] > 0){
        j = smap[nczipp->rank];
        
        // Allocate intermediate buffer for our own data
        tbuf = (char *)NCI_Malloc(ssize[nczipp->rank]);

        // Pack into continuous buffer
        packoff = 0;
        CHK_ERR_PACK(MPI_BOTTOM, 1, stype[j], tbuf, ssize[nczipp->rank], &packoff, nczipp->comm);

        rbufp = sbuf[j];
        for (k = 0; k < scnt[nczipp->rank]; k++) {
            // Retrieve metadata
            cid = *((int *)(rbufp));            rbufp += sizeof(int);
            tstartp = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);
            tssizep = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);

            // Subarray type
            err = nczipioi_subarray_off_len(varp->ndim, varp->chunkdim, tssizep, tstartp, roffs + k, rlens + k);
            if (err){
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, varp->chunkdim, tssizep, tstartp, MPI_ORDER_C, varp->etype, rtypes + k);
                CHK_ERR_TYPE_COMMIT(rtypes + k);
                roffs[k] = varp->chunk_cache[cid]->buf;
                rlens[k] = 1;
            }
            else{
                rtypes[k] = MPI_BYTE;
                roffs[k] = roffs[k] * varp->esize + varp->chunk_cache[cid]->buf;
                rlens[k] *= varp->esize;
            }

            // Mark chunk as dirty
            varp->dirty[cid] = 1;
#ifdef PNETCDF_PROFILING
            nczipp->nrecv++;
#endif
        }

        // Unpack data
        MPI_Type_struct(scnt[nczipp->rank], rlens, roffs, rtypes, &rtype);
        CHK_ERR_TYPE_COMMIT(&rtype);
        packoff = 0;
        CHK_ERR_UNPACK(tbuf, ssize[nczipp->rank], &packoff, MPI_BOTTOM, 1, rtype, nczipp->comm);

        // Free type
        for (k = 0; k < scnt[nczipp->rank]; k++) {
            if (rtypes[k] != MPI_BYTE) {
                MPI_Type_free(rtypes + k);
            }
        }
        MPI_Type_free(&rtype);

        NCI_Free(tbuf);
    }
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SELF)

    //Handle incoming requests
    for (i = 0; i < nrecv; i++) {
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_RECV_REQ)

        // Will wait any provide any benefit?
        MPI_Waitany(nrecv, rreq, &j, &rstat);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_RECV_REQ)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_UNPACK_REQ)

        rbufp = rbuf[j];
        for (k = 0; k < rcnt[j]; k++) {
            // Retrieve metadata
            cid = *((int *)(rbufp));            rbufp += sizeof(int);
            tstartp = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);
            tssizep = (int *)rbufp;            rbufp += varp->ndim * sizeof(int);

            // Subarray type
            err = nczipioi_subarray_off_len(varp->ndim, varp->chunkdim, tssizep, tstartp, roffs + k, rlens + k);
            if (err){
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, varp->chunkdim, tssizep, tstartp, MPI_ORDER_C, varp->etype, rtypes + k);
                CHK_ERR_TYPE_COMMIT(rtypes + k);
                roffs[k] = varp->chunk_cache[cid]->buf;
                rlens[k] = 1;
            }
            else{
                rtypes[k] = MPI_BYTE;
                roffs[k] = roffs[k] * varp->esize + varp->chunk_cache[cid]->buf;
                rlens[k] *= varp->esize;
            }

            // Mark chunk as dirty
            varp->dirty[cid] = 1;
#ifdef PNETCDF_PROFILING
            nczipp->nrecv++;
#endif
        }

        // Unpack data
        MPI_Type_struct(rcnt[j], rlens, roffs, rtypes, &rtype);
        CHK_ERR_TYPE_COMMIT(&rtype);
        packoff = 0;
        CHK_ERR_UNPACK(rbufp, rsize[j], &packoff, MPI_BOTTOM, 1, rtype, nczipp->comm);

        // Free type
        for (k = 0; k < rcnt[j]; k++) {
            if (rtypes[k] != MPI_BYTE) {
                MPI_Type_free(rtypes + k);
            }
        }
        MPI_Type_free(&rtype);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_UNPACK_REQ)
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_SEND_REQ)

    CHK_ERR_WAITALL(nsend, sreq, sstat);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_SEND_REQ)

    // Free type
    for (i = 0; i < npack; i++) {
        for (j = 0; j < sntypes[i]; j++) {
            if (stypes[i][j] != MPI_BYTE) {
                MPI_Type_free(stypes[i] + j);
            }
        }
        MPI_Type_free(stype + i);
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
    if (nrecv > 0){
        NCI_Free(rbuf[0]);
    }
    NCI_Free(rbuf);
    NCI_Free(rtypes);
    NCI_Free(roffs);
    NCI_Free(rlens);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB)

    return NC_NOERR;
}

int
nczipioi_put_varn(NC_zip        *nczipp,
              NC_zip_var       *varp,
              int              nreq,
              MPI_Offset* const *starts,
              MPI_Offset* const *counts,
              const void       *buf)
{
    int err;
    int i, j;
    MPI_Offset rsize;
    char *bptr = (char*)buf;
    char **bufs;
    
    if (varp->isrec){
        for(i = 0; i < nreq; i++){
            if (nczipp->recsize < starts[i][0] + counts[i][0]){
                nczipp->recsize = starts[i][0] + counts[i][0];
            }
        }
        CHK_ERR_ALLREDUCE(MPI_IN_PLACE, &(nczipp->recsize), 1, MPI_LONG_LONG, MPI_MAX, nczipp->comm);   // Sync number of recs
        if (varp->dimsize[0] < nczipp->recsize){
            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_RESIZE)

            nczipioi_var_resize(nczipp, varp);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_RESIZE)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT)
        }
    }

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
            nczipioi_put_varn_cb_chunk(nczipp, varp, nreq, starts, counts, NULL, (void**)bufs);
            break;
        case NC_ZIP_COMM_PROC:
            nczipioi_put_varn_cb_proc(nczipp, varp, nreq, starts, counts, (void**)bufs);
            break;
    }
    
    // Write the compressed variable
    nczipioi_save_var(nczipp, varp);

    NCI_Free(bufs);

    return NC_NOERR;
}
