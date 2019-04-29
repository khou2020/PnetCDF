/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

#include <pnc_debug.h>
#include <nczipio_driver.h>
#include <zip_driver.h>
#include <common.h>

#include <zlib.h>

int nczip_zlib_init(MPI_Info info) {
    return NC_NOERR;
}

int nczip_zlib_finalize() {
    return NC_NOERR;
}

/* Return an estimated compressed data size
 * Actual compressed size should not exceed the estimation
 */
int nczip_zlib_inq_cpsize(void *in, int in_len, int *out_len, int ndim, int *dims, MPI_Datatype dtype) {
    return NC_ENOTSUPPORT;  // Zlib has no size estimation
}

/* If out_len is large enough, compress the data at in and save it to out. out_len is set to actual compressed data size
 * If out_len is NULL, we assume out is large enough for compressed data
 */
int nczip_zlib_compress(void *in, int in_len, void *out, int *out_len, int ndim, int *dims, MPI_Datatype dtype) {
    int err;

    // zlib struct
    z_stream defstream;
    defstream.zalloc = Z_NULL;
    defstream.zfree = Z_NULL;
    defstream.opaque = Z_NULL;
    defstream.avail_in = (uInt)(in_len); // input size
    defstream.next_in = (Bytef*)in; // input
    if (out_len != NULL){
        defstream.avail_out = (uInt)(*out_len); // output buffer size
    }
    else{
        defstream.avail_out = (uInt)1000000000; // Assume it is large enough
    }
    defstream.next_out = (Bytef *)out; // output buffer

    // the actual compression work.
    err = deflateInit(&defstream, Z_BEST_COMPRESSION);
    if (err != Z_OK){
        printf("deflateInit fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }
    err = deflate(&defstream, Z_FINISH);
    if (err != Z_OK){
        printf("deflate fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }
    err = deflateEnd(&defstream);
    if (err != Z_OK){
        printf("deflateEnd fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // If buffer not large enough
    if (defstream.avail_in > 0){
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    // Size of comrpessed data
    if (out_len != NULL){
        *out_len = defstream.total_out;
    }

    return NC_NOERR;
}

/* Compress the data at in and save it to a newly allocated buffer at out. out_len is set to actual compressed data size
 * The caller is responsible to free the buffer
 * If out_len is not NULL, it will be set to buffer size allocated
 */
int nczip_zlib_compress_alloc(void *in, int in_len, void **out, int *out_len, int ndim, int *dims, MPI_Datatype dtype) {
    int err;
    int bsize = in_len >> 3; // Start by 1/8 of the in_len
    char *buf;

    buf = (char*)NCI_Malloc(bsize); 

    // zlib struct
    z_stream defstream;
    defstream.zalloc = Z_NULL;
    defstream.zfree = Z_NULL;
    defstream.opaque = Z_NULL;
    defstream.avail_in = (uInt)(in_len); // input size
    defstream.next_in = (Bytef*)in; // input
    defstream.avail_out = (uInt)(bsize); // output buffer size
    defstream.next_out = (Bytef *)buf; // output buffer

    // Initialize deflat stream
    err = deflateInit(&defstream, Z_BEST_COMPRESSION);
    if (err != Z_OK){
        printf("deflateInit fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // The actual compression work
    while (defstream.avail_in != 0){
        // Compress data
        err = deflate(&defstream, Z_NO_FLUSH);
        if (err != Z_OK){
            printf("deflate fail: %d: %s\n", err, defstream.msg);
            DEBUG_RETURN_ERROR(NC_EIO)
        }

        // If we run out of buffer
        if (defstream.avail_out == 0){
            // Enlarge buffer
            buf = (char*)NCI_Realloc(buf, bsize << 1); 

            // Reset buffer info in stream
            defstream.next_out = buf + bsize;
            defstream.avail_out = bsize;

            // Reocrd new buffer size
            bsize <<= 1;
        }
    }

    // Finalize deflat stream
    err = deflateEnd(&defstream);
    if (err != Z_OK){
        printf("deflateEnd fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // Size of comrpessed data
    if (out_len != NULL){
        *out_len = defstream.total_out;
    }

    // Compressed data
    *out = buf;

    return NC_NOERR;
}

/* Return an estimated decompressed data size
 * Actual decompressed size should not exceed the estimation
 */
int nczip_zlib_inq_dcsize(void *in, int in_len, int *out_len, int ndim, int *dims, MPI_Datatype dtype) {
    return NC_ENOTSUPPORT;  // Zlib has no size estimation
}

/* If out_len is large enough, decompress the data at in and save it to out. out_len is set to actual decompressed size
 * If out_len is NULL, we assume out is large enough for decompressed data
 */
int nczip_zlib_decompress(void *in, int in_len, void *out, int *out_len, int ndim, int *dims, MPI_Datatype dtype) {
    int err;

    // zlib struct
    z_stream infstream;
    infstream.zalloc = Z_NULL;
    infstream.zfree = Z_NULL;
    infstream.opaque = Z_NULL;
    infstream.avail_in = (unsigned long) in_len; // input size
    infstream.next_in = (Bytef *)in; // input
    if (out_len != NULL){
        infstream.avail_out = (uInt)(*out_len); // output buffer size
    }
    else{
        infstream.avail_out = (uInt)1000000000; // Assume it is large enough
    }
    infstream.next_out = (Bytef *)out; // buffer size
    
    // the actual decompression work.
    err = inflateInit(&infstream);
    if (err != Z_OK){
        printf("inflateInit fail: %d: %s\n", err, infstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }
    err = inflate(&infstream, Z_NO_FLUSH);
    if (err != Z_OK){
        printf("inflate fail: %d: %s\n", err, infstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }
    err = inflateEnd(&infstream);
    if (err != Z_OK){
        printf("inflateEnd fail: %d: %s\n", err, infstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // If buffer not large enough
    if (infstream.avail_in > 0){
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    // Size of decomrpessed data
    if (out_len != NULL){
        *out_len = infstream.total_out;
    }

    return NC_NOERR;
}

/* Decompress the data at in and save it to a newly allocated buffer at out. out_len is set to actual decompressed data size
 * The caller is responsible to free the buffer
 * If out_len is not NULL, it will be set to buffer size allocated
 */
int nczip_zlib_decompress_alloc(void *in, int in_len, void **out, int *out_len, int ndim, int *dims, MPI_Datatype dtype) {
    int err;
    int bsize = in_len << 1; // Start by 2 times of the in_len
    char *buf;

    buf = (char*)NCI_Malloc(bsize); 

    // zlib struct
    z_stream infstream;
    infstream.zalloc = Z_NULL;
    infstream.zfree = Z_NULL;
    infstream.opaque = Z_NULL;
    infstream.avail_in = (uInt)(in_len); // input size
    infstream.next_in = (Bytef*)in; // input
    infstream.avail_out = (uInt)(bsize); // output buffer size
    infstream.next_out = (Bytef *)buf; // output buffer

    // Initialize deflat stream
    err = inflateInit(&infstream);
    if (err != Z_OK){
        printf("inflateInit fail: %d: %s\n", err, infstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // The actual compression work
    while (infstream.avail_in != 0){
        // Compress data
        err = inflate(&infstream, Z_NO_FLUSH);
        if (err != Z_OK){
            printf("deflate fail: %d: %s\n", err, infstream.msg);
            DEBUG_RETURN_ERROR(NC_EIO)
        }

        // If we run out of buffer
        if (infstream.avail_out == 0){
            // Enlarge buffer
            buf = (char*)NCI_Realloc(buf, bsize << 1); 

            // Reset buffer info in stream
            infstream.next_out = buf + bsize;
            infstream.avail_out = bsize;

            // Reocrd new buffer size
            bsize <<= 1;
        }
    }

    // Finalize deflat stream
    err = inflateEnd(&infstream);
    if (err != Z_OK){
        printf("inflateEnd fail: %d: %s\n", err, infstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // Size of comrpessed data
    if (out_len != NULL){
        *out_len = infstream.total_out;
    }

    // Compressed data
    *out = buf;

    return NC_NOERR;
}

static NCZIP_driver nczip_driver_zlib = {
    nczip_zlib_init,
    nczip_zlib_finalize,
    nczip_zlib_inq_cpsize,
    nczip_zlib_compress,
    nczip_zlib_compress_alloc,
    nczip_zlib_inq_dcsize,
    nczip_zlib_decompress,
    nczip_zlib_decompress_alloc
};

NCZIP_driver* nczip_zlib_inq_driver(void) {
    return &nczip_driver_zlib;
}
