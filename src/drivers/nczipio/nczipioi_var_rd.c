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

int nczipioi_load_var(NC_zip *nczipp, NC_zip_var *varp, int nchunk, int *cids) {
    int err;
    int i;
    int cid;
    int get_size;

    int dsize;
    MPI_Offset bsize;

    int *lens;
    MPI_Aint *disps;
    MPI_Status status;
    MPI_Datatype ftype;  // Memory and file datatype

    int *zsizes;
    MPI_Offset *zoffs;
    char **zbufs;

    NC *ncp = (NC*)(nczipp->ncp);
    NC_var *ncvarp;

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_INIT)

    // -1 means all chunks
    if (nchunk < 0){
        nchunk = varp->nmychunks;
        cids = varp->mychunks;
    }

    zsizes = varp->data_lens;
    zoffs = varp->data_offs;

    // Allocate buffer for I/O
    lens = (int*)NCI_Malloc(sizeof(int) * nchunk);
    disps = (MPI_Aint*)NCI_Malloc(sizeof(MPI_Aint) * nchunk);
    zbufs = (char**)NCI_Malloc(sizeof(char*) * nchunk);

    /* Carry our coll I/O
     * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when count is 0
     * We use a dummy call inplace of type with 0 count
     */
    if (nchunk > 0){
        // Create file type
        ncvarp = ncp->vars.value[varp->datavarid];
        bsize = 0;
        for(i = 0; i < nchunk; i++){
            cid = cids[i];
            // offset and length of compressed chunks
            lens[i] = zsizes[cid];
            disps[i] = (MPI_Aint)zoffs[cid] + (MPI_Aint)ncvarp->begin;
            // At the same time, we record the size of buffer we need
            bsize += (MPI_Offset)lens[i];
        }
        MPI_Type_create_hindexed(nchunk, lens, disps, MPI_BYTE, &ftype);
        CHK_ERR_TYPE_COMMIT(&ftype);

        // Allocate buffer for compressed data
        // We allocate it continuously so no mem type needed
        zbufs[0] = (char*)NCI_Malloc(bsize);
        for(i = 1; i < nchunk; i++){
            zbufs[i] = zbufs[i - 1] + zsizes[cids[i - 1]];
        }

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_INIT)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_RD)

        // Perform MPI-IO
        // Set file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
        // Write data
        CHK_ERR_READ_AT_ALL(ncp->collective_fh, 0, zbufs[0], bsize, MPI_BYTE, &status);
        // Restore file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_RD)

#ifdef _USE_MPI_GET_COUNT
        MPI_Get_count(&status, MPI_BYTE, &get_size);
#else
        MPI_Type_size(ftype, &get_size);
#endif
        nczipp->getsize += get_size;

        // Free type
        MPI_Type_free(&ftype);
    }
    else{
        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_INIT)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_RD)

        // Follow coll I/O with dummy call
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
        CHK_ERR_READ_AT_ALL(ncp->collective_fh, 0, &i, 0, MPI_BYTE, &status);
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_RD)
    }

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_DECOM)

    // Decompress each chunk
    // Allocate chunk cache if not allocated
    if (varp->zip != NULL){
        varp->zip->init(MPI_INFO_NULL);
        dsize = varp->chunksize;
        for(i = 0; i < nchunk; i++){
            cid = cids[i];
            if (varp->chunk_cache[cid] == NULL){
                varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);
            }
            
            varp->zip->decompress(zbufs[i], lens[i], varp->chunk_cache[cid], &dsize, varp->ndim, varp->chunkdim, varp->etype);

            if(dsize != varp->chunksize){
                printf("Decompress Error\n");
            }
        }
        varp->zip->finalize();
    }
    else{
        for(i = 0; i < nchunk; i++){
            cid = cids[i];
            if (varp->chunk_cache[cid] == NULL){
                varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);
            }
            
            memcpy(varp->chunk_cache[cid], zbufs[i], lens[i]);
        }
    }

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_DECOM)

    // Free buffers
    if (nchunk > 0){
        NCI_Free(zbufs[0]);
    }
    NCI_Free(zbufs);

    NCI_Free(lens);
    NCI_Free(disps);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO)

    return NC_NOERR;
}

int nczipioi_load_nvar(NC_zip *nczipp, int nvar, int *varids) {
    int err;
    int i, j, k;
    int cid, vid;
    int get_size;

    int nchunk;

    int dsize;
    MPI_Offset bsize;

    int *lens;
    MPI_Aint *disps;
    MPI_Status status;
    MPI_Datatype ftype;  // Memory and file datatype

    char **zbufs;

    NC *ncp = (NC*)(nczipp->ncp);
    NC_zip_var *varp;
    NC_var *ncvarp;

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO)
    NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_INIT)

    // -1 means all chunks
    nchunk = 0;
    for(i = 0; i < nvar; i++){
        varp = nczipp->vars.data + varids[i];
    
        for(j = 0; j < varp->nmychunks; j++){
            cid = varp->mychunks[j];

            // We only need to read when it is not in cache
            if (varp->chunk_cache[cid] == NULL){
                nchunk++;
            }
        }
    }

    // Allocate buffer for I/O
    lens = (int*)NCI_Malloc(sizeof(int) * nchunk);
    disps = (MPI_Aint*)NCI_Malloc(sizeof(MPI_Aint) * nchunk);
    zbufs = (char**)NCI_Malloc(sizeof(char*) * nchunk);

    /* Carry our coll I/O
     * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when count is 0
     * We use a dummy call inplace of type with 0 count
     */
    if (nchunk > 0){
        // Create file type
        bsize = 0;
        k = 0;
        for(i = 0; i < nvar; i++){
            varp = nczipp->vars.data + varids[i];
            ncvarp = ncp->vars.value[varp->datavarid];
        
            for(j = 0; j < varp->nmychunks; j++){
                cid = varp->mychunks[j];

                // We only need to read when it is not in cache
                if (varp->chunk_cache[cid] == NULL){
                    // offset and length of compressed chunks
                    lens[k] = varp->data_lens[cid];
                    disps[k] = (MPI_Aint)(varp->data_offs[cid]) + (MPI_Aint)ncvarp->begin;
                    // At the same time, we record the size of buffer we need
                    bsize += (MPI_Offset)lens[k++];
                }
            }
        }

        MPI_Type_create_hindexed(nchunk, lens, disps, MPI_BYTE, &ftype);
        CHK_ERR_TYPE_COMMIT(&ftype);

        // Allocate buffer for compressed data
        // We allocate it continuously so no mem type needed
        zbufs[0] = (char*)NCI_Malloc(bsize);
        for(j = 1; j < nchunk; j++){
            zbufs[j] = zbufs[j - 1] + lens[j - 1];
        }    

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_INIT)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_RD)

        // Perform MPI-IO
        // Set file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
        // Write data
        CHK_ERR_READ_AT_ALL(ncp->collective_fh, 0, zbufs[0], bsize, MPI_BYTE, &status);
        // Restore file view
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_RD)

#ifdef _USE_MPI_GET_COUNT
        MPI_Get_count(&status, MPI_BYTE, &get_size);
#else
        MPI_Type_size(ftype, &get_size);
#endif
        nczipp->getsize += get_size;

        // Free type
        MPI_Type_free(&ftype);

        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_DECOM)

        k = 0;
        for(i = 0; i < nvar; i++){
            varp = nczipp->vars.data + varids[i];
            dsize = varp->chunksize;

            // Decompress each chunk
            if (varp->zip != NULL){
                varp->zip->init(MPI_INFO_NULL);
                for(j = 0; j < varp->nmychunks; j++){
                    cid = varp->mychunks[j];

                    // Allocate chunk cache if not allocated
                    if (varp->chunk_cache[cid] == NULL){
                        varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);

                        // Perform decompression
                        varp->zip->decompress(zbufs[k], lens[k], varp->chunk_cache[cid], &dsize, varp->ndim, varp->chunkdim, varp->etype);
                        if(dsize != varp->chunksize){
                            printf("Decompress Error\n");
                        }

                        k++;
                    }            
                }
                varp->zip->finalize();
            }
            else{
                for(j = 0; j < varp->nmychunks; j++){
                    cid = varp->mychunks[j];

                    // Allocate chunk cache if not allocated
                    if (varp->chunk_cache[cid] == NULL){
                        varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);

                        memcpy(varp->chunk_cache[cid], zbufs[k], lens[k]);

                        k++;
                    }            
                }
            }
        }
        
        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_DECOM)
    }
    else{
        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_INIT)
        NC_ZIP_TIMER_START(NC_ZIP_TIMER_GET_IO_RD)

        // Follow coll I/O with dummy call
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
        CHK_ERR_READ_AT_ALL(ncp->collective_fh, 0, &i, 0, MPI_BYTE, &status);
        CHK_ERR_SET_VIEW(ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

        NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO_RD)
    }

    // Free buffers
    if (nchunk > 0){
        NCI_Free(zbufs[0]);
    }
    NCI_Free(zbufs);

    NCI_Free(lens);
    NCI_Free(disps);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_GET_IO)

    return NC_NOERR;
}