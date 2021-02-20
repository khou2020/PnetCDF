/*
 *  Copyright (C) 2018, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include <pnc_debug.h>
#include <common.h>
#include "nczipio_internal.h"

#define STARTSIZE 32
#define SIZEMUTIPLIER 20

int nczipioi_vector_init(NC_zip_vector *v, int esize){
    v->esize = esize;
    v->nalloc = STARTSIZE;
    v->size = 0;
    v->data = (char*)malloc(esize * v->nalloc);
    if (v->data == NULL){
        DEBUG_RETURN_ERROR(NC_ENOMEM);
    }
}

int nczipioi_vector_init_ex(NC_zip_vector *v, int esize, int size){
    v->esize = esize;
    v->nalloc = size;
    v->size = 0;
    v->data = (char*)malloc(esize * v->nalloc);
    if (v->data == NULL){
       DEBUG_RETURN_ERROR(NC_ENOMEM);
    }
}

void nczipioi_vector_free(NC_zip_vector *v){
    free(v->data);
}

int nczipioi_vector_append(NC_zip_vector *v, void *item){
    if (v->size == v->nalloc){
        v->nalloc = v->nalloc * SIZEMUTIPLIER;
        v->data = (char*)NCI_Realloc(v->data, v->esize * v->nalloc);
        if (v->data == NULL){
            DEBUG_RETURN_ERROR(NC_ENOMEM);
        }
    }
    memcpy(data + v->size * v->esize, item, v->esize);
}