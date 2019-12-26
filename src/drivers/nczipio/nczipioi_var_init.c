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

int nczipioi_var_init(NC_zip *nczipp, NC_zip_var *varp, int nreq, MPI_Offset **starts, MPI_Offset **counts) {
    int i, j, err;
    int valid;
    MPI_Offset len;
    NC_zip_var *var;

    if (varp->varkind == NC_ZIP_VAR_COMPRESSED){
        if (varp->chunkdim == NULL){    // This is a new uninitialized variable 
            // Update dimsize on rec dim
            if (nczipp->recdim >= 0){
                if (varp->dimsize[0] < nczipp->recsize){
                    varp->dimsize[0] = nczipp->recsize;
                }
            }

            // Determine its block size
            varp->chunkdim = (int*)NCI_Malloc(sizeof(int) * varp->ndim);
            varp->nchunks = (int*)NCI_Malloc(sizeof(int) * varp->ndim);

            // First check attribute
            valid = 1;
            err = nczipp->driver->inq_att(nczipp->ncp, varp->varid, "_chunkdim", NULL, &len);
            if (err == NC_NOERR && len == varp->ndim){
                err = nczipp->driver->get_att(nczipp->ncp, varp->varid, "_chunkdim", varp->chunkdim, MPI_INT);
                if (err != NC_NOERR){
                    valid = 0;
                }
                //chunkdim must be at leasst 1
                for(j = 0; j < varp->ndim; j++){ 
                    if (varp->chunkdim[j] <= 0){
                        valid = 0;
                        printf("Warning: chunk size invalid, use default");
                        break;
                    }
                }
            }
            else{
                valid = 0;
            }

            // Now, try global default
            if ((!valid) && nczipp->chunkdim){
                valid = 1;
                for(i = 0; i < varp->ndim; i++){
                    if (nczipp->chunkdim[varp->dimids[i]] > 0){
                        varp->chunkdim[i] = nczipp->chunkdim[varp->dimids[i]];
                    }
                    else{
                        valid = 0;
                        break;
                    }
                }
            }

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_INIT_META)
            
            // Still no clue, try to infer form I/O pattern (expensive)
            // If there is no I/O records, the default is just set to entire variable (only 1 chunk)
            if (!valid){
                err = nczipioi_calc_chunk_size(nczipp, varp, nreq, starts, counts);
                if (err != NC_NOERR){
                    return err;
                }
            }

            NC_ZIP_TIMER_START(NC_ZIP_TIMER_INIT_META)

            // Calculate total # chunks, # chunks along each dim, chunksize
            varp->nchunkrec = 1;
            varp->chunksize = NC_Type_size(varp->xtype);
            for(i = 0; i < varp->ndim; i++){ //chunkdim must be at leasst 1
                if (varp->dimsize[i] % varp->chunkdim[i] == 0){
                    varp->nchunks[i] = (int)varp->dimsize[i] / varp->chunkdim[i];
                }
                else{
                    varp->nchunks[i] = (int)varp->dimsize[i] / varp->chunkdim[i] + 1;
                }
                if (i > 0){
                    varp->nchunkrec *= varp->nchunks[i];
                }
                varp->chunksize *= varp->chunkdim[i];
            }
            if (varp->isrec){
                varp->nrec = varp->nchunks[0];
                varp->nrecalloc = nczipp->default_recnalloc;
                while(varp->nrecalloc < varp->nchunks[0]){
                    varp->nrecalloc *= NC_ZIP_REC_MULTIPLIER;
                }
            }
            else{
                varp->nrec = 1;
                varp->nrecalloc = 1;
                varp->nchunkrec *= varp->nchunks[0];
            }
            varp->nchunk = varp->nchunkrec * varp->nrec;
            varp->nchunkalloc = varp->nrecalloc * varp->nchunkrec;

            // Calculate number of chunks below each dimension
            varp->cidsteps = (int*)NCI_Malloc(sizeof(int) * varp->ndim);
            varp->cidsteps[varp->ndim - 1] = 1;
            for(i = varp->ndim - 2; i >= 0; i--){
                varp->cidsteps[i] = varp->cidsteps[i + 1] * varp->nchunks[i + 1];
            }

            // Determine block ownership
            varp->chunk_owner = (int*)NCI_Malloc(sizeof(int) * varp->nchunkalloc);
            varp->dirty = (int*)NCI_Malloc(sizeof(int) * varp->nchunkalloc);
            varp->chunk_cache = (NC_zip_cache**)NCI_Malloc(sizeof(char*) * varp->nchunkalloc);
            memset(varp->chunk_cache, 0, sizeof(char*) * varp->nchunkalloc);
            memset(varp->dirty, 0, sizeof(int) * varp->nchunkalloc);

            NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_INIT_META)

            // We infer owners by reqs
            err = nczipioi_calc_chunk_owner(nczipp, varp, nreq, starts, counts);
            if (err != NC_NOERR){
                return err;
            }

            NC_ZIP_TIMER_START(NC_ZIP_TIMER_INIT_META)

            // Build skip list of my own chunks
            if (varp->nchunk > 0){
                varp->nmychunkrec = 0;
                for(j = 0; j < varp->nchunkrec; j++){ 
                    if (varp->chunk_owner[j] == nczipp->rank){
                        varp->nmychunkrec++;
                    }
                }
                varp->nmychunk = varp->nmychunkrec * varp->nrec;
                varp->mychunks = (int*)NCI_Malloc(sizeof(int) * varp->nmychunkrec * varp->nrecalloc);
                varp->nmychunk = 0;
                for(j = 0; j < varp->nchunk; j++){ 
                    if (varp->chunk_owner[j] == nczipp->rank){
                        varp->mychunks[varp->nmychunk++] = j;
                        if (varp->isnew){   // Only apply to new var, old var will be read when it is needed
                            // varp->chunk_cache[j] = (void*)NCI_Malloc(varp->chunksize);  // Allocate buffer for blocks we own
                            //memset(varp->chunk_cache[j], 0 , varp->chunksize);
                        }
                    }
                }
            }
            else{
                varp->nmychunk = varp->nmychunkrec = 0;
                varp->mychunks = NULL;
            }
            
            // Update global chunk count
            nczipp->nmychunks += varp->nmychunk;

            // Determine block offset
            varp->chunk_index = (NC_zip_chunk_index_entry*)NCI_Malloc(sizeof(NC_zip_chunk_index_entry) * (varp->nchunkalloc + 1));

            // Try if there are offset recorded in attributes, it can happen after opening a file
            if (varp->isnew){
                varp->metaoff = -1;;
                memset(varp->chunk_index, 0, sizeof(NC_zip_chunk_index_entry) * (varp->nchunk + 1));
            }
           
            /* Select compression driver based on attribute */
            err = nczipp->driver->inq_att(nczipp->ncp, varp->varid, "_zipdriver", NULL, &len);
            if (err == NC_NOERR && len == 1){
                err = nczipp->driver->get_att(nczipp->ncp, varp->varid, "_zipdriver", &(varp->zipdriver), MPI_INT);
                if (err != NC_NOERR){
                    return err;
                }
            }
            else{
                varp->zipdriver = nczipp->default_zipdriver;
            }
            switch (varp->zipdriver){
                case NC_ZIP_DRIVER_NONE:
                    varp->zip = NULL;
                    break;
                case NC_ZIP_DRIVER_DUMMY:
                    varp->zip = nczip_dummy_inq_driver();
                    break;
#ifdef ENABLE_ZLIB
                case NC_ZIP_DRIVER_ZLIB:
                    varp->zip = nczip_zlib_inq_driver();
                    break;
#endif
#ifdef ENABLE_SZ
                case NC_ZIP_DRIVER_SZ:
                    varp->zip = nczip_sz_inq_driver();
                    break;
#endif
                default:
                    if (nczipp->rank == 0){
                        printf("Warning: Unknown zip driver id %d, use NC_ZIP_DRIVER_DUMMY\n", varp->zipdriver);
                    }
                    varp->zip = nczip_dummy_inq_driver();
                    break;
                break;
            }

            // Update max ndim and chunksize
            if (nczipp->max_ndim < varp->ndim){
                nczipp->max_ndim = varp->ndim;
            }
            if (nczipp->max_chunk_size < varp->chunksize){
                nczipp->max_chunk_size = varp->chunksize;
            }

            if (nczipp->cache_limit_hint == -1){
                nczipp->cache_limit += (size_t)(varp->nmychunkrec) * (size_t)(varp->chunksize);
            }
        }   
    }

    return NC_NOERR;
}

void nczipioi_var_free(NC_zip_var *varp) {
    int i;

    if (varp->chunkdim != NULL){
        NCI_Free(varp->dimsize);
        NCI_Free(varp->chunkdim);
        NCI_Free(varp->dimids);
        NCI_Free(varp->nchunks);
        NCI_Free(varp->cidsteps);
        NCI_Free(varp->chunk_index);
        NCI_Free(varp->chunk_owner);
        NCI_Free(varp->dirty);
        //for(i = 0; i < varp->nmychunk; i++){
        //    if (varp->chunk_cache[varp->mychunks[i]] != NULL){
        //        NCI_Free(varp->chunk_cache[varp->mychunks[i]]);
        //    }
        //}
        NCI_Free(varp->chunk_cache);
        NCI_Free(varp->mychunks);
    }
}

int nczipioi_init_nvar(NC_zip *nczipp, int nput, int *putreqs, int nget, int *getreqs){
    int err;
    int i, j;
    int nflag;
    unsigned int *flag, *flag_all;
    int nvar;
    int *rcnt, *roff;
    int *vids, *vmap;
    int nread;
    int *lens;
    MPI_Aint *fdisps, *mdisps;
    MPI_Datatype ftype, mtype;
    MPI_Status status;
    MPI_Offset **starts, **counts;
    NC_zip_req *req;
    NC_zip_var *varp;

    NC_ZIP_TIMER_START(NC_ZIP_TIMER_INIT_META)

    CHK_ERR_ALLREDUCE(MPI_IN_PLACE, &(nczipp->recsize), 1, MPI_LONG_LONG, MPI_MAX, nczipp->comm);   // Sync number of recs

    // Flag of touched vars
    nflag = nczipp->vars.cnt / 32 + 1;
    flag = (unsigned int*)NCI_Malloc(sizeof(int) * nflag * 2);
    flag_all = flag + nflag;
    memset(flag, 0, sizeof(int) * nflag);
    for(i = 0; i < nput; i++){
        req = nczipp->putlist.reqs + putreqs[i];
        flag[req->varid >> 5] |= 1u << (req->varid % 32);
    }
    for(i = 0; i < nget; i++){
        req = nczipp->getlist.reqs + getreqs[i];
        flag[req->varid >> 5] |= 1u << (req->varid % 32);
    }

    // Sync flag
    CHK_ERR_ALLREDUCE(flag, flag_all, nflag, MPI_UNSIGNED, MPI_BOR, nczipp->comm);

    // Build a skip list of touched vars
    nvar = 0;
    for(i = 0; i < nczipp->vars.cnt; i++){
        if (flag_all[i >> 5] & (1u << (i % 32))) {
            if ((nczipp->vars.data + i)->chunkdim == NULL){   // If not yet inited
                nvar++;
            }
            else{   
                flag_all[i >> 5] ^= (1u << (i % 32));
                if ((nczipp->vars.data + i)->dimsize[0] < nczipp->recsize){
                    nczipioi_var_resize(nczipp, nczipp->vars.data + i);
                }
            }
        }
    }
    vids = (int*)NCI_Malloc(sizeof(int) * nvar);
    vmap = (int*)NCI_Malloc(sizeof(int) * nczipp->vars.cnt);
    nvar = 0;
    for(i = 0; i < nczipp->vars.cnt; i++){
        if (flag_all[i >> 5] & (1u << (i % 32))) {
            vids[nvar] = i;
            vmap[i] = nvar++;
        }
    }

    // Count reqs for each var
    roff = (int*)NCI_Malloc(sizeof(int) * (nvar + 1));
    rcnt = (int*)NCI_Malloc(sizeof(int) * nvar);
    memset(rcnt, 0, sizeof(int) * nvar);
    for(i = 0; i < nput; i++){
        req = nczipp->putlist.reqs + putreqs[i];
        j = req->varid;
        if (flag_all[j >> 5] & (1u << (j % 32))) {
            rcnt[vmap[j]] += req->nreq;
        }
    }
    for(i = 0; i < nget; i++){
        req = nczipp->getlist.reqs + getreqs[i];
        j = req->varid;
        if (flag_all[j >> 5] & (1u << (j % 32))) {
            rcnt[vmap[j]] += req->nreq;
        }
    }
    roff[0] = 0;
    for(i = 0; i < nvar; i++){
        roff[i + 1] = roff[i] + rcnt[i];
    }

    // Gather starts and counts
    starts = (MPI_Offset**)NCI_Malloc(sizeof(MPI_Offset*) * roff[nvar] * 2);
    counts = starts + roff[nvar];
    memset(rcnt, 0, sizeof(int) * nvar);
    for(i = 0; i < nput; i++){
        req = nczipp->putlist.reqs + putreqs[i];
        j = req->varid;
        if (flag_all[j >> 5] & (1u << (j % 32))) {
            j = vmap[req->varid];
            if (req->nreq > 1){
                memcpy(starts + roff[j] + rcnt[j], req->starts, sizeof(MPI_Offset*) * req->nreq);
                memcpy(counts + roff[j] + rcnt[j], req->counts, sizeof(MPI_Offset*) * req->nreq);
                rcnt[j] += req->nreq;
            }
            else{
                starts[roff[j] + rcnt[j]] = req->start;
                counts[roff[j] + (rcnt[j]++)] = req->count;     
            }
        }
    }
    for(i = 0; i < nget; i++){
        req = nczipp->getlist.reqs + getreqs[i];
        j = req->varid;
        if (flag_all[j >> 5] & (1u << (j % 32))) {
            j = vmap[req->varid];
            if (req->nreq > 1){
                memcpy(starts + roff[j] + rcnt[j], req->starts, sizeof(MPI_Offset*) * req->nreq);
                memcpy(counts + roff[j] + rcnt[j], req->counts, sizeof(MPI_Offset*) * req->nreq);
                rcnt[j] += req->nreq;
            }
            else{
                starts[roff[j] + rcnt[j]] = req->start;
                counts[roff[j] + (rcnt[j]++)] = req->count;     
            }
        }
    }

    lens = NCI_Malloc(sizeof(int) * nvar);
    fdisps = NCI_Malloc(sizeof(MPI_Aint) * nvar * 2);
    mdisps = fdisps + nvar;

    nread = 0;
    for(i = 0; i < nvar; i++){
        varp = nczipp->vars.data + vids[i];

        err = nczipioi_var_init(nczipp, varp, rcnt[i], starts + roff[i], counts + roff[i]);
        if (err != NC_NOERR){
            return err;
        }

        if(!(varp->isnew)){
            err = nczipp->driver->get_att(nczipp->ncp, varp->varid, "_metaoffset", &(varp->metaoff), MPI_LONG_LONG);
            if (err == NC_NOERR){
                lens[nread] = sizeof(NC_zip_chunk_index_entry) * (varp->nchunk);
                fdisps[nread] = varp->metaoff;
                mdisps[nread++] = (MPI_Aint)(varp->chunk_index);
            }
            else{
                varp->metaoff = -1;
                memset(varp->chunk_index, 0, sizeof(NC_zip_chunk_index_entry) * (varp->nchunk + 1));
            }
        }
    }

    if (nread){
        nczipioi_sort_file_offset(nread, fdisps, mdisps, lens);

        MPI_Type_create_hindexed(nread, lens, fdisps, MPI_BYTE, &ftype);
        CHK_ERR_TYPE_COMMIT(&ftype);

        MPI_Type_create_hindexed(nread, lens, mdisps, MPI_BYTE, &mtype);
        CHK_ERR_TYPE_COMMIT(&mtype);

        // Set file view
        CHK_ERR_SET_VIEW(((NC*)(nczipp->ncp))->collective_fh, ((NC*)(nczipp->ncp))->begin_var, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
        
        // Read data
        CHK_ERR_READ_AT_ALL(((NC*)(nczipp->ncp))->collective_fh, 0, MPI_BOTTOM, 1, mtype, &status);
        
        // Restore file view
        CHK_ERR_SET_VIEW(((NC*)(nczipp->ncp))->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

#ifdef WORDS_BIGENDIAN // Switch back to little endian
        nczipioi_idx_in_swapn(varp-chunk_index, varp->nchunk + 1);
#endif

        MPI_Type_free(&ftype);
        MPI_Type_free(&mtype);
    }

    NCI_Free(lens);
    NCI_Free(fdisps);

    NCI_Free(flag);
    NCI_Free(vids);
    NCI_Free(vmap);
    NCI_Free(roff);
    NCI_Free(rcnt);
    NCI_Free(starts);

    NC_ZIP_TIMER_STOP(NC_ZIP_TIMER_INIT_META)

    return NC_NOERR;
}
