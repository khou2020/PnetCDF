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
#include <config.h>
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
int nczipioi_iput_cb_chunk(NC_zip *nczipp, int nreq, int *reqids, int *stats) {
    int err;
    int i, j;
    int vid;    // Iterators for variable id
    int *nreqs; // Number of reqids in each variable
    int *nums;  // Number of reqs in each varn
    int **vreqids;
    int num, maxnum = 0;
    MPI_Offset **starts, **counts, **strides;
    MPI_Offset rsize;
    char **bufs;
    NC_zip_req *req;

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_INIT)

    // Count total number of request in per variable for packed varn request
    nums = (int *)NCI_Malloc(sizeof(int) * nczipp->vars.cnt * 2);
    nreqs = nums + nczipp->vars.cnt;
    memset(nums, 0, sizeof(int) * nczipp->vars.cnt);
    memset(nreqs, 0, sizeof(int) * nczipp->vars.cnt);
    for (i = 0; i < nreq; i++){
        req = nczipp->putlist.reqs + reqids[i];
        nreqs[req->varid]++;
        nums[req->varid] += req->nreq;
    }

    /* Allocate a skip list of reqids for each vriable
     * At the same time, we find out the number of starts and counts we need to allocate
     */
    vreqids = (int **)NCI_Malloc(sizeof(int *) * nczipp->vars.cnt);
    vreqids[0] = (int *)NCI_Malloc(sizeof(int) * nreq);
    maxnum = 0;
    i = 0;
    for (vid = 0; vid < nczipp->vars.cnt; vid++){
        if (nreqs[vid] > 0) {
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
    for (i = 0; i < nreq; i++){
        req = nczipp->putlist.reqs + reqids[i];
        vreqids[req->varid][nreqs[req->varid]++] = reqids[i];
    }

    // Allocate parameters
    starts = (MPI_Offset **)NCI_Malloc(sizeof(MPI_Offset *) * maxnum * 2);
    counts = starts + maxnum;
    bufs = (char **)NCI_Malloc(sizeof(char *) * maxnum);

    /* Pack requests variable by variable
     */
    for (vid = 0; vid < nczipp->vars.cnt; vid++){
        if (nreqs[vid] > 0){
            // Collect parameters
            num = 0;
            for (j = 0; j < nreqs[vid]; j++){
                req = nczipp->putlist.reqs + vreqids[vid][j];

                for (i = 0; i < req->nreq; i++){
                    starts[num] = req->starts[i];
                    counts[num] = req->counts[i];
                    bufs[num++] = req->xbufs[i];
                }
            }

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB)
            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_INIT)

            // Perform collective buffering
            nczipioi_put_varn_cb_chunk(nczipp, nczipp->vars.data + vid, num, starts, counts, NULL, (void **)bufs);

            NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB)
            NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_CB_INIT)
        }
    }

    // Free buffers
    NCI_Free(nums);

    NCI_Free(vreqids[0]);
    NCI_Free(vreqids);

    NCI_Free(starts);
    NCI_Free(bufs);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB)
    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB_INIT)

    return NC_NOERR;
}

/* Out drive currently can handle only one variable at a time
 * We pack all request as a large varn request
 */
int nczipioi_iput_cb_proc(NC_zip *nczipp, int nreq, int *reqids, int *stats){
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
    int *rlo_local, *rhi_local;
    int *rlo_all, *rhi_all;
    int *rids;

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
    NC_zip_var *varp;
    NC_zip_req *req;

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
    rlo_local = (int *)NCI_Malloc(sizeof(int) * nczipp->vars.cnt * 5);
    rhi_local = rlo_local + nczipp->vars.cnt;
    rlo_all = rhi_local + nczipp->vars.cnt;
    rhi_all = rlo_all + nczipp->vars.cnt;
    rids = rhi_all + nczipp->vars.cnt;

    for (i = 0; i < nczipp->vars.cnt; i++){
        rlo_local[i] = 2147483647;
        rhi_local[i] = -1;
    }

    // Allocate buffering for write count
    scnt = (int *)NCI_Malloc(sizeof(int) * nczipp->np * 6);
    ssize = scnt + nczipp->np;
    rcnt = ssize + nczipp->np;
    rsize = rcnt + nczipp->np;
    smap = rsize + nczipp->np;
    rmap = smap + nczipp->np;

    memset(scnt, 0, sizeof(int) * nczipp->np * 2);
    npack = 0;
    // Count total number of messages and build a map of accessed chunk to list of comm datastructure
    for (i = 0; i < nreq; i++) {
        req = nczipp->putlist.reqs + reqids[i];
        varp = nczipp->vars.data + req->varid;
        for (r = 0; r < req->nreq; r++) {
            nczipioi_chunk_itr_init(varp, req->starts[r], req->counts[r], citr, &cid); // Initialize chunk iterator
            do {
                // Chunk owner
                cown = varp->chunk_owner[cid];

                // Mapping to skip list of send requests
                if ((scnt[cown] == 0) && (cown != nczipp->rank)) {
                    smap[cown] = npack++;
                }
                scnt[cown]++; // Need to send message if not owner
                ssize[cown] += sizeof(int) * (varp->ndim * 2 + 2);

                if (rlo_local[req->varid] > cid) {
                    rlo_local[req->varid] = cid;
                }
                if (rhi_local[req->varid] < cid) {
                    rhi_local[req->varid] = cid;
                }
            } while (nczipioi_chunk_itr_next(varp, req->starts[r], req->counts[r], citr, &cid));
        }
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
            soffs[i][0] = (uintptr_t)(sbuf[i]);
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
    for (k = 0; k < nreq; k++){
        req = nczipp->putlist.reqs + reqids[k];
        varp = nczipp->vars.data + req->varid;

        for (r = 0; r < req->nreq; r++){
            nczipioi_chunk_itr_init_ex(varp, req->starts[r], req->counts[r], citr, &cid, ostart, osize); // Initialize chunk iterator
            do {
                // Chunk index and owner
                cown = varp->chunk_owner[cid];

                j = smap[cown];
		
                // Pack metadata
                *((int *)(sbufp[j])) = req->varid;  sbufp[j] += sizeof(int);
                *((int *)(sbufp[j])) = cid; sbufp[j] += sizeof(int);
                tstartp = (int *)sbufp[j];  sbufp[j] += varp->ndim * sizeof(int);
                tssizep = (int *)sbufp[j];  sbufp[j] += varp->ndim * sizeof(int);

                for (i = 0; i < varp->ndim; i++){
                    tstartp[i] = (uintptr_t)(ostart[i] - citr[i]);
                    tssizep[i] = (uintptr_t)osize[i];
                }

                // Pack type from user buffer to send buffer
                for (i = 0; i < varp->ndim; i++){
                    tsize[i] = (uintptr_t)req->counts[r][i];
                    tstart[i] = (uintptr_t)(ostart[i] - req->starts[r][i]);
                }

                err = nczipioi_subarray_off_len(varp->ndim, tsize, tssizep, tstart, soffsp[j], slensp[j]);
                if (err){
                    CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, tsize, tssizep, tstart, MPI_ORDER_C, varp->etype, stypesp[j]);
                    CHK_ERR_TYPE_COMMIT(stypesp[j]);
                    *(soffsp[j]) = (uintptr_t)req->xbufs[r];
                    *(slensp[j]) = 1;
                }
                else{
                    *(stypesp[j]) = MPI_BYTE;
                    *(soffsp[j]) = (*(soffsp[j])) * varp->esize + (uintptr_t)req->xbufs[r];
                    *(slensp[j]) *= varp->esize;
                }
                stypesp[j]++;
                soffsp[j]++;
                slensp[j]++;

#ifdef PNETCDF_PROFILING
                nczipp->nsend++;
#endif
            } while (nczipioi_chunk_itr_next_ex(varp, req->starts[r], req->counts[r], citr, &cid, ostart, osize));
        }
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

    for (i = 0; i < nczipp->vars.cnt; i++) {
        rhi_local[i] *= -1;
    }
    CHK_ERR_ALLREDUCE(rlo_local, rlo_all, nczipp->vars.cnt * 2, MPI_INT, MPI_MIN, nczipp->comm);
    for (i = 0; i < nczipp->vars.cnt; i++) {
        rhi_all[i] *= -1;
    }

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
    nread = 0;
    for (i = 0; i < nczipp->vars.cnt; i++) {
        if (rhi_all[i] >= rlo_all[i]) {
            varp = nczipp->vars.data + i;
            rids[nread] = i;
            for (j = 0; j < varp->nmychunk && varp->mychunks[j] < rlo_all[i]; j++) ;
            for (k = j; k < varp->nmychunk && varp->mychunks[k] <= rhi_all[i]; k++) ;
            rlo_all[nread] = j;
            rhi_all[nread++] = k;
        }
    }
    err = nczipioi_load_nvar_bg(nczipp, nread, rids, rlo_all, rhi_all); CHK_ERR
    (nczipp->cache_serial)++;

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
            vid = *((int *)(rbufp));
            rbufp += sizeof(int);
            cid = *((int *)(rbufp));
            rbufp += sizeof(int);
            varp = nczipp->vars.data + vid;
            tstartp = (int *)rbufp;
            rbufp += varp->ndim * sizeof(int);
            tssizep = (int *)rbufp;
            rbufp += varp->ndim * sizeof(int);

            // Subarray type
            err = nczipioi_subarray_off_len(varp->ndim, varp->chunkdim, tssizep, tstartp, roffs + k, rlens + k);
            if (err){
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, varp->chunkdim, tssizep, tstartp, MPI_ORDER_C, varp->etype, rtypes + k);
                CHK_ERR_TYPE_COMMIT(rtypes + k);
                roffs[k] = (((uintptr_t)(varp->chunk_cache[cid]->buf)));
                rlens[k] = 1;
            }
            else{
                rtypes[k] = MPI_BYTE;
                roffs[k] = roffs[k] * varp->esize + (((uintptr_t)(varp->chunk_cache[cid]->buf)));
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
            vid = *((int *)(rbufp));
            rbufp += sizeof(int);
            cid = *((int *)(rbufp));
            rbufp += sizeof(int);
            varp = nczipp->vars.data + vid;
            tstartp = (int *)rbufp;
            rbufp += varp->ndim * sizeof(int);
            tssizep = (int *)rbufp;
            rbufp += varp->ndim * sizeof(int);

            // Subarray type
            err = nczipioi_subarray_off_len(varp->ndim, varp->chunkdim, tssizep, tstartp, roffs + k, rlens + k);
            if (err){
                CHK_ERR_TYPE_CREATE_SUBARRAY(varp->ndim, varp->chunkdim, tssizep, tstartp, MPI_ORDER_C, varp->etype, rtypes + k);
                CHK_ERR_TYPE_COMMIT(rtypes + k);
                roffs[k] = (((uintptr_t)(varp->chunk_cache[cid]->buf)));
                rlens[k] = 1;
            }
            else{
                rtypes[k] = MPI_BYTE;
                roffs[k] = roffs[k] * varp->esize + (((uintptr_t)(varp->chunk_cache[cid]->buf)));
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

    NCI_Free(rlo_local);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_CB)

    return NC_NOERR;
}
