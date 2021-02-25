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
#include <nczipio_driver.h>
#include "nczipio_internal.h"

int nczipioi_var_list_init(NC_zip_var_list *list) {
    list->cnt = 0;
    list->nalloc = 0;
    return NC_NOERR;
}

int nczipioi_var_list_free(NC_zip_var_list *list) {
    int i, j;
    if (list->nalloc > 0){
        for(i = 0; i < list->cnt; i++){
            nczipioi_var_free(list->data + i);
        }
        NCI_Free(list->data);
    }
    return NC_NOERR;
}

int nczipioi_var_list_add(NC_zip_var_list *list) {
    if (list->nalloc == 0){
        list->nalloc = 16;
        list->data = NCI_Malloc(list->nalloc * sizeof(NC_zip_var));
        CHK_ALLOC(list->data)
    }
    else if (list->nalloc == list->cnt){
        list->nalloc *= 2;
        list->data = NCI_Realloc(list->data, list->nalloc * sizeof(NC_zip_var));
        CHK_ALLOC(list->data)
    }

    return ((list->cnt)++);
}
