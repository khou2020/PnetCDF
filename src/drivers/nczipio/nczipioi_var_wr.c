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
#include "../ncmpio/ncmpio_NC.h"

int nczipioi_save_var(NC_zip *nczipp, NC_zip_var *varp) {
    int i, j, k, l, err;
    int *zsizes, *zsizes_all;
    MPI_Datatype mtype, ftype;  // Memory and file datatype
    int wcnt;
    int reqids[2];
    int *lens;
    MPI_Aint *disps;
    MPI_Status status;
    MPI_Offset *zoffs;
    MPI_Offset start, count, oldzoff;
    void **zbufs;
    int zdimid, mdimid;
    int put_size;
    char name[128]; // Name of objects
    NC *ncp = (NC*)(nczipp->ncp);
    NC_var *ncvarp;

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO)

    // Allocate buffer for compression
    zsizes = (int*)NCI_Malloc(sizeof(int) * varp->nchunk);
    zbufs = (void**)NCI_Malloc(sizeof(void*) * varp->nmychunk);
    zsizes_all = (int*)NCI_Malloc(sizeof(int) * varp->nchunk);
    zoffs = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * (varp->nchunk + 1));

    //zsizes_all = varp->data_lens;
    //zoffs = varp->data_offs;
    oldzoff = zoffs[varp->nchunk];

    // Allocate buffer for I/O
    wcnt = 0;
    for(l = 0; l < varp->nmychunk; l++){
        k = varp->mychunks[l];
        if (varp->dirty[k]){
            wcnt++;
        }
    }
    if (nczipp->rank == varp->chunk_owner[0]){
        wcnt += 2;
    }
    lens = (int*)NCI_Malloc(sizeof(int) * wcnt);
    disps = (MPI_Aint*)NCI_Malloc(sizeof(MPI_Aint) * wcnt);

    memset(zsizes, 0, sizeof(int) * varp->nchunk);

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_COM)

    // Compress each chunk we own
    if (varp->zip != NULL){
        varp->zip->init(MPI_INFO_NULL);
        for(l = 0; l < varp->nmychunk; l++){
            k = varp->mychunks[l];

            if (varp->dirty[k]){
                // Apply compression
                varp->zip->compress_alloc(varp->chunk_cache[k], varp->chunksize, zbufs + l, zsizes + k, varp->ndim, varp->chunkdim, varp->etype);
            }
        }
        varp->zip->finalize();
    }
    else{
        for(l = 0; l < varp->nmychunk; l++){
            k = varp->mychunks[l];
            if (varp->dirty[k]){
                zbufs[l] = varp->chunk_cache[k];
                zsizes[k] = varp->chunksize;
            }
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_COM)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_SYNC)

    // Sync compressed data size with other processes
    CHK_ERR_ALLREDUCE(zsizes, zsizes_all, varp->nchunk, MPI_INT, MPI_MAX, nczipp->comm);
    zoffs[0] = 0;
    for(i = 0; i < varp->nchunk; i++){
        zoffs[i + 1] = zoffs[i] + zsizes_all[i];
    }
    //zsizes_all[i] = zoffs[i];   // Remove valgrind warning

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_SYNC)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_INIT)

    /* Write comrpessed variable
     * We start by defining data variable and writing metadata
     * Then, we create buffer type and file type for data
     * Finally MPI collective I/O is used for writing data
     */

    // Enter redefine mode
    nczipp->driver->redef(nczipp->ncp);

    // Prepare metadata variable
    if (varp->offvarid < 0 || varp->expanded){    // Check if we need new metadata vars
        // Define dimension for metadata variable
        sprintf(name, "_compressed_meta_dim_%d_%d", varp->varid, varp->metaserial);
        err = nczipp->driver->def_dim(nczipp->ncp, name, varp->nchunkalloc, &mdimid);
        if (err != NC_NOERR) return err;

        // Define off variable
        sprintf(name, "_compressed_offset_%d_%d", varp->varid, varp->metaserial);
        err = nczipp->driver->def_var(nczipp->ncp, name, NC_INT64, 1, &mdimid, &(varp->offvarid));
        if (err != NC_NOERR) return err;

        // Define lens variable
        sprintf(name, "_compressed_size_%d_%d", varp->varid, varp->metaserial);
        err = nczipp->driver->def_var(nczipp->ncp, name, NC_INT, 1, &mdimid, &(varp->lenvarid));
        if (err != NC_NOERR) return err;

        // Mark as meta variable
        i = NC_ZIP_VAR_META;
        err = nczipp->driver->put_att(nczipp->ncp, varp->offvarid, "_varkind", NC_INT, 1, &i, MPI_INT);
        if (err != NC_NOERR) return err;

        err = nczipp->driver->put_att(nczipp->ncp, varp->lenvarid, "_varkind", NC_INT, 1, &i, MPI_INT);
        if (err != NC_NOERR) return err;

        // Record lens variable id
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_lenvarid", NC_INT, 1, &(varp->lenvarid), MPI_INT);
        if (err != NC_NOERR) return err;
        
        // Record offset variable id
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_offvarid", NC_INT, 1, &varp->offvarid, MPI_INT);
        if (err != NC_NOERR) return err;

        // Record serial
        varp->metaserial++;
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_metaserial", NC_INT, 1, &(varp->metaserial), MPI_INT);
        if (err != NC_NOERR) return err;
    }

    // Prepare data variable
    if (1 || varp->datavarid < 0|| varp->expanded || zoffs[varp->nchunk] > oldzoff){ // Check if we need new data vars
        // Define dimension for data variable
        sprintf(name, "_compressed_data_dim_%d_%d", varp->varid, varp->dataserial);
        err = nczipp->driver->def_dim(nczipp->ncp, name, zoffs[varp->nchunk], &zdimid);
        if (err != NC_NOERR) return err;

        // Define data variable
        sprintf(name, "_compressed_data_%d_%d", varp->varid, varp->dataserial);
        err = nczipp->driver->def_var(nczipp->ncp, name, NC_BYTE, 1, &zdimid, &(varp->datavarid));
        if (err != NC_NOERR) return err;

        // Record data variable id
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_datavarid", NC_INT, 1, &(varp->datavarid), MPI_INT);
        if (err != NC_NOERR) return err;

        // Mark as data variable
        i = NC_ZIP_VAR_DATA;
        err = nczipp->driver->put_att(nczipp->ncp, varp->datavarid, "_varkind", NC_INT, 1, &i, MPI_INT);
        if (err != NC_NOERR) return err;

        // Record serial
        varp->dataserial++;
        err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_dataserial", NC_INT, 1, &(varp->dataserial), MPI_INT);
        if (err != NC_NOERR) return err;
    }

    // unset expand flag
    varp->expanded = 0;

    // Switch to data mode
    err = nczipp->driver->enddef(nczipp->ncp);
    if (err != NC_NOERR) return err;

    // Update metadata
    ncvarp = ncp->vars.value[varp->datavarid];
    for(i = 0; i < varp->nchunk; i++){
        if (zsizes_all[i] > 0){
            varp->data_lens[i] = zsizes_all[i];
            varp->data_offs[i] = zoffs[i] + ncvarp->begin - ncp->begin_var;
        }
    }

    /* Carry out coll I/O
     * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when count is 0
     * We use a dummy call inplace of type with 0 count
     */
    if (wcnt > 0){
        // Create file type
        l = 0;
        if (nczipp->rank == varp->chunk_owner[0]){  // First chunk owner writes metadata
            ncvarp = ncp->vars.value[varp->offvarid];
            lens[l] = (varp->nchunk) * sizeof(long long);
            disps[l++] = (MPI_Aint)ncvarp->begin;
            
            ncvarp = ncp->vars.value[varp->lenvarid];
            lens[l] = (varp->nchunk) * sizeof(int);
            disps[l++] = (MPI_Aint)ncvarp->begin;
        }
        for(i = 0; i < varp->nmychunk; i++){
            k = varp->mychunks[i];

            // Record compressed size
            if (varp->dirty[k]){
                lens[l] = zsizes[k];
                disps[l++] = (MPI_Aint)varp->data_offs[k] + ncp->begin_var;
            }
        }
        MPI_Type_create_hindexed(wcnt, lens, disps, MPI_BYTE, &ftype);
        CHK_ERR_TYPE_COMMIT(&ftype);

        // Create memory buffer type
        l = 0;
        if (nczipp->rank == varp->chunk_owner[0]){  // First chunk owner writes metadata
            lens[l] = (varp->nchunk) * sizeof(long long);
            disps[l++] = (MPI_Aint)varp->data_offs;

            lens[l] = (varp->nchunk) * sizeof(int);
            disps[l++] = (MPI_Aint)varp->data_lens;
        }
        for(i = 0; i < varp->nmychunk; i++){
            k = varp->mychunks[i];

            // Record compressed size
            if (varp->dirty[k]){
                lens[l] = zsizes[k];
                disps[l++] = (MPI_Aint)zbufs[i];
            }
        }
        err = MPI_Type_create_hindexed(wcnt, lens, disps, MPI_BYTE, &mtype);
        CHK_ERR_TYPE_COMMIT(&mtype);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_INIT)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_WR)

#ifndef WORDS_BIGENDIAN // NetCDF data is big endian
        if (nczipp->rank == varp->chunk_owner[0]){
            ncmpii_in_swapn(varp->data_offs, varp->nchunk + 1, sizeof(long long));
            ncmpii_in_swapn(varp->data_lens, varp->nchunk + 1, sizeof(int));
        }
#endif

        // Perform MPI-IO
        // Set file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
        // Write data
        CHK_ERR_WRITE_AT_ALL(ncp->collective_fh, 0, MPI_BOTTOM, 1, mtype, &status);
        // Restore file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

#ifndef WORDS_BIGENDIAN // Switch back to little endian
        if (nczipp->rank == varp->chunk_owner[0]){
            ncmpii_in_swapn(varp->data_offs, varp->nchunk + 1, sizeof(long long));
            ncmpii_in_swapn(varp->data_lens, varp->nchunk + 1, sizeof(int));
        }
#endif

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_WR)

#ifdef _USE_MPI_GET_COUNT
        MPI_Get_count(&status, MPI_BYTE, &put_size);
#else
        MPI_Type_size(mtype, &put_size);
#endif
        nczipp->putsize += put_size;

        // Free type
        MPI_Type_free(&ftype);
        MPI_Type_free(&mtype);
    }
    else{
        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_INIT)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_WR)

        // Follow coll I/O with dummy call
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
        CHK_ERR_WRITE_AT_ALL(ncp->collective_fh, 0, MPI_BOTTOM, 0, MPI_BYTE, &status);
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_WR)
    }

    // Free buffers
    NCI_Free(zsizes);
    NCI_Free(zsizes_all);
    NCI_Free(zoffs);
    for(l = 0; l < varp->nmychunk; l++){
        k = varp->mychunks[l];
        if (varp->dirty[k]){
            if (varp->zip != NULL){
                free(zbufs[l]);
            }
            // Clear dirty flag
            varp->dirty[k] = 0;
        }
    }
    NCI_Free(zbufs);

    NCI_Free(lens);
    NCI_Free(disps);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO)

    return NC_NOERR;
}

int nczipioi_save_nvar(NC_zip *nczipp, int nvar, int *varids) {
    int i, j, k, l, err;
    int vid;    // Iterator for variable id
    int cid;    // Iterator for chunk id
    int total_nchunks = 0;
    int *zsizes, *zsizes_all, *zsizesp, *zsizes_allp;
    int nreq;
    MPI_Offset *zoffs, *zoffsp;
    MPI_Offset start, count, oldzoff;
    MPI_Datatype mtype, ftype;  // Memory and file datatype
    int wcnt, ccnt, wcur, ccur;
    int *mlens, *flens;
    MPI_Aint *mdisps, *fdisps;
    MPI_Status status;
    MPI_Request *reqs;
    int put_size;
    void **zbufs;
    int *zdels;
    int zdimid, mdimid;
    char name[128]; // Name of objects
    NC_zip_var *varp;
    NC *ncp = (NC*)(nczipp->ncp);
    NC_var *ncvarp;

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_INIT)

    wcnt = 0;
    ccnt = 0;
    for(i = 0; i < nvar; i++){
        varp = nczipp->vars.data + varids[i];
        if (nczipp->rank == varp->chunk_owner[0]){
            wcnt += 2;
        }
        for(l = 0; l < varp->nmychunk; l++){
            k = varp->mychunks[l];
            if (varp->dirty[k]){
                ccnt++;
            }
        }
        total_nchunks += varp->nchunk;
    }
    wcnt += ccnt;

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_INIT)

    // Allocate reqid for metadata
    reqs = (int*)NCI_Malloc(sizeof(MPI_Request) * nvar);

    // Allocate buffer for compression
    zsizes = (int*)NCI_Malloc(sizeof(int) * total_nchunks);
    zsizes_all = (int*)NCI_Malloc(sizeof(int) * total_nchunks);
    zbufs = (void**)NCI_Malloc(sizeof(void*) * ccnt);
    zdels = (int*)NCI_Malloc(sizeof(int) * ccnt);
    zoffs = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * (total_nchunks + nvar));

    // Allocate buffer file type
    mlens = (int*)NCI_Malloc(sizeof(int) * wcnt);
    mdisps = (MPI_Aint*)NCI_Malloc(sizeof(MPI_Aint) * wcnt);
    flens = (int*)NCI_Malloc(sizeof(int) * wcnt);
    fdisps = (MPI_Aint*)NCI_Malloc(sizeof(MPI_Aint) * wcnt);

    // Enter redefine mode
    nczipp->driver->redef(nczipp->ncp);

    ccur = 0;
    zsizesp = zsizes;
    zsizes_allp = zsizes_all;
    zoffsp = zoffs;
    for(vid = 0; vid < nvar; vid++){
        varp = nczipp->vars.data + varids[vid];

        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_COM)

        //zsizes_all = varp->data_lens;
        //zoffs = varp->data_offs;
        oldzoff = zoffs[varp->nchunk];

        memset(zsizesp, 0, sizeof(int) * varp->nchunk);

        // Compress each chunk we own
        if (varp->zip != NULL){
            varp->zip->init(MPI_INFO_NULL);
            for(l = 0; l < varp->nmychunk; l++){
                cid = varp->mychunks[l];

                // Apply compression
                if (varp->dirty[cid]){
                    zdels[ccur] = 1;
                    varp->zip->compress_alloc(varp->chunk_cache[cid], varp->chunksize, zbufs + (ccur++), zsizesp + cid, varp->ndim, varp->chunkdim, varp->etype);
                }
            }
            varp->zip->finalize();
        }
        else{
            for(l = 0; l < varp->nmychunk; l++){
                cid = varp->mychunks[l];
                if (varp->dirty[cid]){
                    zsizesp[cid] = varp->chunksize;
                    zdels[ccur] = 0;
                    zbufs[ccur++] = varp->chunk_cache[cid];
                }
            }
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_COM)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_SYNC)

        // Sync compressed data size with other processes
        CHK_ERR_IALLREDUCE(zsizesp, zsizes_allp, varp->nchunk, MPI_INT, MPI_MAX, nczipp->comm, reqs + vid);
        
        //zsizes_all[cid] = zoffs[cid];   // Remove valgrind warning

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_SYNC)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_INIT)

        /* Write comrpessed variable
        * We start by defining data variable and writing metadata
        * Then, we create buffer type and file type for data
        * Finally MPI collective I/O is used for writing data
        */

        // Prepare metadata variable
        if (varp->offvarid < 0 || varp->expanded){    // Check if we need new metadata vars
            // Define dimension for metadata variable
            sprintf(name, "_compressed_meta_dim_%d_%d", varp->varid, varp->metaserial);
            err = nczipp->driver->def_dim(nczipp->ncp, name, varp->nchunkalloc, &mdimid);
            if (err != NC_NOERR) return err;

            // Define off variable
            sprintf(name, "_compressed_offset_%d_%d", varp->varid, varp->metaserial);
            err = nczipp->driver->def_var(nczipp->ncp, name, NC_INT64, 1, &mdimid, &(varp->offvarid));
            if (err != NC_NOERR) return err;

            // Define lens variable
            sprintf(name, "_compressed_size_%d_%d", varp->varid, varp->metaserial);
            err = nczipp->driver->def_var(nczipp->ncp, name, NC_INT, 1, &mdimid, &(varp->lenvarid));
            if (err != NC_NOERR) return err;

            // Mark as meta variable
            i = NC_ZIP_VAR_META;
            err = nczipp->driver->put_att(nczipp->ncp, varp->offvarid, "_varkind", NC_INT, 1, &i, MPI_INT);
            if (err != NC_NOERR) return err;

            err = nczipp->driver->put_att(nczipp->ncp, varp->lenvarid, "_varkind", NC_INT, 1, &i, MPI_INT);
            if (err != NC_NOERR) return err;

            // Record lens variable id
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_lenvarid", NC_INT, 1, &(varp->lenvarid), MPI_INT);
            if (err != NC_NOERR) return err;
            
            // Record offset variable id
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_offvarid", NC_INT, 1, &varp->offvarid, MPI_INT);
            if (err != NC_NOERR) return err;

            // Record serial
            varp->metaserial++;
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_metaserial", NC_INT, 1, &(varp->metaserial), MPI_INT);
            if (err != NC_NOERR) return err;

            // unset expand flag
            varp->expanded = 0;
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_INIT)

        zsizesp += varp->nchunk;
        zsizes_allp += varp->nchunk;
        zoffsp += varp->nchunk + 1;
    }

    zsizes_allp = zsizes_all;
    zoffsp = zoffs;
    for(vid = 0; vid < nvar; vid++){
        varp = nczipp->vars.data + varids[vid];

        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_SYNC)

        CHK_ERR_WAIT(reqs + vid, &status);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_SYNC)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_INIT)

        zoffsp[0] = 0;
        for(cid = 0; cid < varp->nchunk; cid++){
            zoffsp[cid + 1] = zoffsp[cid] + zsizes_allp[cid];
        }

        // Prepare data variable
        if (1 || varp->datavarid < 0|| varp->expanded || zoffsp[varp->nchunk] > oldzoff){ // Check if we need new data vars
            // Define dimension for data variable
            sprintf(name, "_compressed_data_dim_%d_%d", varp->varid, varp->dataserial);
            err = nczipp->driver->def_dim(nczipp->ncp, name, zoffsp[varp->nchunk], &zdimid);
            if (err != NC_NOERR) return err;

            // Define data variable
            sprintf(name, "_compressed_data_%d_%d", varp->varid, varp->dataserial);
            err = nczipp->driver->def_var(nczipp->ncp, name, NC_BYTE, 1, &zdimid, &(varp->datavarid));
            if (err != NC_NOERR) return err;

            // Record data variable id
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_datavarid", NC_INT, 1, &(varp->datavarid), MPI_INT);
            if (err != NC_NOERR) return err;

            // Mark as data variable
            i = NC_ZIP_VAR_DATA;
            err = nczipp->driver->put_att(nczipp->ncp, varp->datavarid, "_varkind", NC_INT, 1, &i, MPI_INT);
            if (err != NC_NOERR) return err;

            // Record serial
            varp->dataserial++;
            err = nczipp->driver->put_att(nczipp->ncp, varp->varid, "_dataserial", NC_INT, 1, &(varp->dataserial), MPI_INT);
            if (err != NC_NOERR) return err;
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_INIT)

        zsizes_allp += varp->nchunk;
        zoffsp += varp->nchunk + 1;
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_INIT)

    // Switch back to data mode
    err = nczipp->driver->enddef(nczipp->ncp);
    if (err != NC_NOERR) return err;

    wcur = ccur = 0;
    zsizes_allp = zsizes_all;
    zoffsp = zoffs;
    for(vid = 0; vid < nvar; vid++){
        varp = nczipp->vars.data + varids[vid];
        
        ncvarp = ncp->vars.value[varp->datavarid];
        for(cid = 0; cid < varp->nchunk; cid++){
            if (zsizes_allp[cid] > 0){
                varp->data_lens[cid] = zsizes_allp[cid];
                varp->data_offs[cid] = zoffsp[cid] + ncvarp->begin - ncp->begin_var;
            }
        }

        /* Paramemter for file and memory type 
         * We do not know variable file offset until the end of define mode
         * We will add the displacement later
         */
        if (nczipp->rank == varp->chunk_owner[0]){  // First chunk owner writes metadata
            ncvarp = ncp->vars.value[varp->offvarid];
            flens[wcur] = mlens[wcur] = varp->nchunk * sizeof(long long);
            fdisps[wcur] = (MPI_Aint)ncvarp->begin;
            mdisps[wcur++] = (MPI_Aint)(varp->data_offs);
            
            ncvarp = ncp->vars.value[varp->lenvarid];
            flens[wcur] = mlens[wcur] = varp->nchunk * sizeof(int);
            fdisps[wcur] = (MPI_Aint)ncvarp->begin;
            mdisps[wcur++] = (MPI_Aint)(varp->data_lens);
        }
        for(i = 0; i < varp->nmychunk; i++){
            cid = varp->mychunks[i];

            // Record parameter
            if (varp->dirty[cid]){
                flens[wcur] = mlens[wcur] = varp->data_lens[cid];
                fdisps[wcur] = (MPI_Aint)varp->data_offs[cid] + ncp->begin_var;
                mdisps[wcur++] = (MPI_Aint)zbufs[ccur++];
            }
        }

        // Clear dirty flag
        memset(varp->dirty, 0, varp->nchunk * sizeof(int));

        zsizes_allp += varp->nchunk;
        zoffsp += varp->nchunk + 1;

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_INIT)
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_INIT)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_PUT_IO_WR)

    /* Carry our coll I/O
     * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when count is 0
     * We use a dummy call inplace of type with 0 count
     */
    if (wcnt > 0){
         // Create file type
        MPI_Type_create_hindexed(wcnt, flens, fdisps, MPI_BYTE, &ftype);
        CHK_ERR_TYPE_COMMIT(&ftype);

        // Create memmory type
        MPI_Type_create_hindexed(wcnt, mlens, mdisps, MPI_BYTE, &mtype);
        CHK_ERR_TYPE_COMMIT(&mtype);

#ifndef WORDS_BIGENDIAN // NetCDF data is big endian
        for(vid = 0; vid < nvar; vid++){
            varp = nczipp->vars.data +  varids[vid];
            if (nczipp->rank == varp->chunk_owner[0]){
                ncmpii_in_swapn(varp->data_offs, varp->nchunk + 1, sizeof(long long));
                ncmpii_in_swapn(varp->data_lens, varp->nchunk + 1, sizeof(int));
            }
        }
#endif     

        // Perform MPI-IO
        // Set file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
        // Write data
        CHK_ERR_WRITE_AT_ALL(ncp->collective_fh, 0, MPI_BOTTOM, 1, mtype, &status);
        // Restore file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

#ifndef WORDS_BIGENDIAN // Switch back to little endian
        for(vid = 0; vid < nvar; vid++){
            varp = nczipp->vars.data +  varids[vid];
            if (nczipp->rank == varp->chunk_owner[0]){
                ncmpii_in_swapn(varp->data_offs, varp->nchunk + 1, sizeof(long long));
                ncmpii_in_swapn(varp->data_lens, varp->nchunk + 1, sizeof(int));
            }
        }
#endif    

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_WR)

#ifdef _USE_MPI_GET_COUNT
        MPI_Get_count(&status, MPI_BYTE, &put_size);
#else
        MPI_Type_size(mtype, &put_size);
#endif
        nczipp->putsize += put_size;

        // Free type
        MPI_Type_free(&ftype);
        MPI_Type_free(&mtype);
    }
    else{
        // Follow coll I/O with dummy call
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
        CHK_ERR_WRITE_AT_ALL(ncp->collective_fh, 0, MPI_BOTTOM, 0, MPI_BYTE, &status);
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO_WR)
    }

    // Free buffers
    NCI_Free(zsizes);
    NCI_Free(zsizes_all);
    NCI_Free(zoffs);
    ccur = 0;
    for(i = 0; i < ccnt; i++){
        if (zdels[i]){
            free(zbufs[i]);
        }
    }
    NCI_Free(zbufs);
    NCI_Free(zdels);

    NCI_Free(flens);
    NCI_Free(fdisps);
    NCI_Free(mlens);
    NCI_Free(mdisps);

    NCI_Free(reqs);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_PUT_IO)

    return NC_NOERR;
}
