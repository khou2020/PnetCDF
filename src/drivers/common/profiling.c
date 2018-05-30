/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <pnetcdf.h>
#include <pnc_debug.h>
#include <common.h>

static double ncmpii_log_flush_replay_time;
static double ncmpii_log_flush_data_rd_time;
static double ncmpii_log_flush_put_time;
static double ncmpii_log_flush_wait_time;
static double ncmpii_log_flush_time;
static double ncmpii_log_put_data_wr_time;
static double ncmpii_log_put_meta_wr_time;
static double ncmpii_log_put_num_wr_time;
static double ncmpii_log_put_time;
static double ncmpii_log_create_time;
static double ncmpii_log_enddef_time;
static double ncmpii_log_close_time;
static double ncmpii_log_total_time;
static MPI_Offset ncmpii_log_total_data;
static MPI_Offset ncmpii_log_total_meta;
static MPI_Offset ncmpii_log_max_buffer;

void ncmpii_update_bb_counter(double total_time, double create_time, double enddef_time, double put_time, double flush_time, double close_time,
                              double flush_replay_time, double flush_data_rd_time, double flush_put_time, double flush_wait_time,
                              double put_data_wr_time, double put_meta_wr_time, double put_num_wr_time,
                              MPI_Offset total_data, MPI_Offset total_meta, MPI_Offset max_buffer){
    ncmpii_log_flush_replay_time += flush_replay_time;
    ncmpii_log_flush_data_rd_time += flush_data_rd_time;
    ncmpii_log_flush_put_time += flush_put_time;
    ncmpii_log_flush_wait_time += flush_wait_time;
    ncmpii_log_flush_time += flush_time;
    ncmpii_log_put_data_wr_time += put_data_wr_time;
    ncmpii_log_put_meta_wr_time += put_meta_wr_time;
    ncmpii_log_put_num_wr_time += put_num_wr_time;
    ncmpii_log_put_time += put_time;
    ncmpii_log_create_time += create_time;
    ncmpii_log_enddef_time += enddef_time;
    ncmpii_log_close_time += close_time;
    ncmpii_log_total_time += total_time;
    ncmpii_log_total_data += total_data;
    ncmpii_log_total_meta += total_meta;
    if (ncmpii_log_max_buffer < max_buffer) {
        ncmpii_log_max_buffer = max_buffer; 
    }
}

/*----< ncmpi_inq_bb_time() >--------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_bb_time(double *total_time, double *create_time, double *enddef_time, double *put_time, double *flush_time, double *close_time)
{
    // Fill time info
    if (total_time != NULL){
        *total_time = ncmpii_log_total_time;
    }
    if (create_time != NULL){
        *create_time = ncmpii_log_create_time;
    }
    if (enddef_time != NULL){
        *enddef_time = ncmpii_log_enddef_time;
    }
    if (put_time != NULL){
        *put_time = ncmpii_log_put_time;
    }
    if (flush_time != NULL){
        *flush_time = ncmpii_log_flush_time;
    }
    if (close_time != NULL){
        *close_time = ncmpii_log_close_time;
    }

    return NC_NOERR;
}

/*----< ncmpi_inq_bb_time_put() >--------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_bb_time_put(double *put_data_wr_time, double *put_meta_wr_time, double *put_num_wr_time)
{
    // Fill time info
    if (put_data_wr_time != NULL){
        *put_data_wr_time = ncmpii_log_put_data_wr_time;
    }
    if (put_meta_wr_time != NULL){
        *put_meta_wr_time = ncmpii_log_put_meta_wr_time;
    }
    if (put_num_wr_time != NULL){
        *put_num_wr_time = ncmpii_log_put_num_wr_time;
    }
    
    return NC_NOERR;
}

/*----< ncmpi_inq_bb_time_put() >--------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_bb_time_flush(double *flush_replay_time, double *flush_data_rd_time, double *flush_put_time, double *flush_wait_time)
{
    // Fill time info
    if (flush_replay_time != NULL){
        *flush_replay_time = ncmpii_log_flush_replay_time;
    }
    if (flush_data_rd_time != NULL){
        *flush_data_rd_time = ncmpii_log_flush_data_rd_time;
    }
    if (flush_put_time != NULL){
        *flush_put_time = ncmpii_log_flush_put_time;
    }
    if (flush_wait_time != NULL){
        *flush_wait_time = ncmpii_log_flush_wait_time;
    }

    return NC_NOERR;
}

/*----< ncmpi_inq_buffer_size() >--------------------------------------------*/
/* This is an independent subroutine. */
int
ncmpi_inq_bb_size(MPI_Offset *datasize, MPI_Offset *metasize, MPI_Offset *buffersize)
{
    /* Fill size info */
    if (datasize != NULL){
        *datasize = ncmpii_log_total_data;
    }
    if (metasize != NULL){
        *metasize = ncmpii_log_total_meta;
    }
    if (buffersize != NULL){
        *buffersize = ncmpii_log_max_buffer;
    }

    return NC_NOERR;
}