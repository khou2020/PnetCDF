/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <common.h>
#include <nczipio_driver.h>
#include "nczipio_internal.h"

/* return internal size for values of specified netCDF type */
MPI_Offset NC_Type_size(nc_type type){			/* netCDF type code */
    switch (type) {
      case NC_BYTE:
      return sizeof(char);
      case NC_CHAR:
      return sizeof(char);
      case NC_SHORT:
      return sizeof(short);
      case NC_INT:
      return sizeof(int);
      case NC_FLOAT:
      return sizeof(float);
      case NC_DOUBLE:
      return sizeof(double);
      case NC_UBYTE:
      return sizeof(unsigned char);
      case NC_USHORT:
      return sizeof(unsigned short);
      case NC_UINT:
      return sizeof(unsigned int);
      case NC_INT64:
      return sizeof(long long);
      case NC_UINT64:
      return sizeof(unsigned long long);
      default:

      return 0;
    }
}

/*
 * Convert NC type to MPI type
 */
MPI_Datatype nczipioi_nc_to_mpi_type(nc_type atype){
    switch (atype) {
        case NC_BYTE:
            return MPI_BYTE;
        case NC_CHAR:
            return MPI_CHAR;
        case NC_SHORT:
            return MPI_SHORT;
        case NC_INT:
            return MPI_INT;
        case NC_FLOAT:
            return MPI_FLOAT;
        case NC_DOUBLE:
            return MPI_DOUBLE;
    }

    return NC_NAT;
}

/*
 * Extract mpi hints and set up the flags
 */
int nczipioi_extract_hint(NC_zip *nczipp, MPI_Info info){
    int flag;
    char value[MPI_MAX_INFO_VAL];

    // Block assignment
    MPI_Info_get(info, "nc_zip_block_mapping", MPI_MAX_INFO_VAL - 1,
                 value, &flag);
    if (flag) {
        if (strcmp(value, "static") == 0){
            nczipp->blockmapping = NC_ZIP_MAPPING_STATIC;  
        }
        else{
            printf("Warning: Unknown zip method %s, using dummy\n", value);
            nczipp->blockmapping = NC_ZIP_MAPPING_STATIC;    
        }
    }
    else {
        nczipp->blockmapping = NC_ZIP_MAPPING_STATIC;    
    }

    // Messaging unit
    MPI_Info_get(info, "nc_zip_comm_unit", MPI_MAX_INFO_VAL - 1,
                 value, &flag);
    if (flag) {
        if (strcmp(value, "chunk") == 0){
            nczipp->comm_unit = NC_ZIP_COMM_CHUNK;  
        }
        else if (strcmp(value, "proc") == 0){
            nczipp->comm_unit = NC_ZIP_COMM_PROC;  
        }
        else{
            printf("Warning: Unknown messaging unit %s, using proc\n", value);
            nczipp->comm_unit = NC_ZIP_COMM_PROC;  
        }
    }
    else { 
        nczipp->comm_unit = NC_ZIP_COMM_PROC;   
    }

    // Delay init
    nczipp->delay_init = 0;  
    MPI_Info_get(info, "nc_zip_delay_init", MPI_MAX_INFO_VAL - 1, value, &flag);
    if (flag) {
        if (strcmp(value, "1") == 0){
            nczipp->delay_init = 1;  
        }
    }

    // Default zipdriver
    nczipp->default_zipdriver = NC_ZIP_DRIVER_NONE;  
    MPI_Info_get(info, "nc_zip_driver", MPI_MAX_INFO_VAL - 1, value, &flag);
    if (flag) {
        if (strcmp(value, "none") == 0){
            nczipp->default_zipdriver = NC_ZIP_DRIVER_NONE;  
        }
        else if (strcmp(value, "dummy") == 0){
            nczipp->default_zipdriver = NC_ZIP_DRIVER_DUMMY;  
        }
        else if (strcmp(value, "zlib") == 0){
            nczipp->default_zipdriver = NC_ZIP_DRIVER_ZLIB;  
        }
        else if (strcmp(value, "sz") == 0){
            nczipp->default_zipdriver = NC_ZIP_DRIVER_SZ;  
        }
        else{
            printf("Warning: Unknown zip driver %s, use none\n", value);
        }
    }

    return NC_NOERR;
}

/*
 * Export hint based on flag
 * NOTE: We only set up the hint if it is not the default setting
 *       user hint maching the default behavior will be ignored
 */
int nczipioi_export_hint(NC_zip *nczipp, MPI_Info info){
    char value[MPI_MAX_INFO_VAL];

    MPI_Info_set(info, "nc_compression", "enable");

    switch (nczipp->blockmapping){
        case NC_ZIP_MAPPING_STATIC:
            MPI_Info_set(info, "nc_zip_block_mapping", "static");
            break;
    }

    switch (nczipp->comm_unit){
        case NC_ZIP_COMM_CHUNK:
            MPI_Info_set(info, "nc_zip_comm_unit", "chunk");
            break;
        case NC_ZIP_COMM_PROC:
            MPI_Info_set(info, "nc_zip_comm_unit", "proc");
            break;
    }

    if (nczipp->delay_init){
        MPI_Info_set(info, "nc_zip_delay_init", "1");
    }
    else{
        MPI_Info_set(info, "nc_zip_delay_init", "0");
    }

    sprintf(value, "%d", nczipp->default_zipdriver);
    MPI_Info_set(info, "nc_zip_driver", value);

    return NC_NOERR;
}

int nczipioi_print_buffer(char *prefix, unsigned char* buf, int len){
    int i;
    int plen;
    char *out, *outp;

    plen = strlen(prefix);
    out = outp = (char*)NCI_Malloc(len * 3 + 1 + plen);

    sprintf(outp, "%s ", prefix);   outp += plen + 1;
    for(i = 0; i < len; i++){
        sprintf(outp, "%02x ", (unsigned int)buf[i]); outp += 3;
    }

    printf("%s\n", out); fflush(stdout);

    NCI_Free(out);

    return NC_NOERR;
}

