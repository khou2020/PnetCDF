/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the following PnetCDF APIs.
 *
 * ncmpi_def_var()                  : dispatcher->def_var()
 * ncmpi_inq_varid()                : dispatcher->inq_varid()
 * ncmpi_inq_var()                  : dispatcher->inq_var()
 * ncmpi_rename_var()               : dispatcher->rename_var()
 *
 * ncmpi_get_var<kind>()            : dispatcher->get_var()
 * ncmpi_put_var<kind>()            : dispatcher->put_var()
 * ncmpi_get_var<kind>_<type>()     : dispatcher->get_var()
 * ncmpi_put_var<kind>_<type>()     : dispatcher->put_var()
 * ncmpi_get_var<kind>_all()        : dispatcher->get_var()
 * ncmpi_put_var<kind>_all()        : dispatcher->put_var()
 * ncmpi_get_var<kind>_<type>_all() : dispatcher->get_var()
 * ncmpi_put_var<kind>_<type>_all() : dispatcher->put_var()
 *
 * ncmpi_iget_var<kind>()           : dispatcher->iget_var()
 * ncmpi_iput_var<kind>()           : dispatcher->iput_var()
 * ncmpi_iget_var<kind>_<type>()    : dispatcher->iget_var()
 * ncmpi_iput_var<kind>_<type>()    : dispatcher->iput_var()
 *
 * ncmpi_buffer_attach()            : dispatcher->buffer_attach()
 * ncmpi_buffer_detach()            : dispatcher->buffer_detach()
 * ncmpi_bput_var<kind>_<type>()    : dispatcher->bput_var()
 *
 * ncmpi_get_varn_<type>()          : dispatcher->get_varn()
 * ncmpi_put_varn_<type>()          : dispatcher->put_varn()
 *
 * ncmpi_iget_varn_<type>()         : dispatcher->iget_varn()
 * ncmpi_iput_varn_<type>()         : dispatcher->iput_varn()
 * ncmpi_bput_varn_<type>()         : dispatcher->bput_varn()
 *
 * ncmpi_get_vard()                 : dispatcher->get_vard()
 * ncmpi_put_vard()                 : dispatcher->put_vard()
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include <pnc_debug.h>
#include <common.h>
#include <nczipio_driver.h>

int
nczipio_def_var(void       *ncdp,
              const char *name,
              nc_type     xtype,
              int         ndims,
              const int  *dimids,
              int        *varidp)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;
    NC_zip_var var;

    var.ndims = ndims;
    var.stripesize = NULL;
    var.offset = NULL;
    var.owner = NULL;

    if (ndims > 3 || ndims < 1) { // Does not support higher dimensional vars
        err = nczipp->driver->def_var(nczipp->ncp, name, xtype, ndims, dimids, &var.varid);  // We use it to save the id of data variable
        if (err != NC_NOERR) return err;
        
        err = nczipp->driver->put_att(nczipp->ncp, var.varid, "_vartype", NC_INT, 1, NC_ZIP_VAR_RAW, MPI_INT);   // Comressed var?
        if (err != NC_NOERR) return err;

        var.type = NC_ZIP_VAR_RAW;
        var.dimsize = NULL;
    }
    else{
        err = nczipp->driver->def_var(nczipp->ncp, name, NC_INT, 0, NULL, &var.varid);  // We use it to save the id of data variable
        if (err != NC_NOERR) return err;
        
        var.type = NC_ZIP_VAR_COMPRESSED;
        var.dimsize = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset) * ndims);
        for(i = 0; i < ndims; i++){
            nczipp->driver->inq_dim(nczipp->ncp, dimids[i], NULL, var.dimsize + i);
        }

        err = nczipp->driver->put_att(nczipp->ncp, var.varid, "_ndim", NC_INT, 1, &ndims, MPI_INT); // Original dimensions
        if (err != NC_NOERR) return err;
        err = nczipp->driver->put_att(nczipp->ncp, var.varid, "_dimids", NC_INT, ndims, dimids, MPI_INT);   // Dimensiona IDs
        if (err != NC_NOERR) return err;
        err = nczipp->driver->put_att(nczipp->ncp, var.varid, "_datatype", NC_INT, 1, &xtype, MPI_INT); // Original datatype
        if (err != NC_NOERR) return err;
        err = nczipp->driver->put_att(nczipp->ncp, var.varid, "_vartype", NC_INT, 1, NC_ZIP_VAR_COMPRESSED, MPI_INT);   // Comressed var?
        if (err != NC_NOERR) return err;
    }

    err = nczipioi_var_list_add(&(nczipp->vars), var);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_inq_varid(void       *ncdp,
                const char *name,
                int        *varid)
{
    int i, vid, err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    if (varid != NULL){
        err = nczipp->driver->inq_varid(nczipp->ncp, name, &vid);
        if (err != NC_NOERR) return err;

        for(i = 0; i < nczipp->vars.cnt; i++){
            if (nczipp->vars.data[i].varid == vid){
                *varid = i;
                break;
            }
        }
    }

    return NC_NOERR;
}

int
nczipio_inq_var(void       *ncdp,
              int         varid,
              char       *name,
              nc_type    *xtypep,
              int        *ndimsp,
              int        *dimids,
              int        *nattsp,
              MPI_Offset *offsetp,
              int        *no_fillp,
              void       *fill_valuep)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;
    NC_var *varp;

    if (varid < 0 || varid >= nczipp->vars.cnt){
        DEBUG_RETURN_ERROR(NC_EINVAL);
    }

    varp = nczipp->vars.data + varid;

    err = nczipp->driver->inq_var(nczipp->ncp, varp->varid, name, xtypep, ndimsp, dimids,
                               nattsp, offsetp, no_fillp, fill_valuep);
    if (err != NC_NOERR) return err;

    if (ndimsp != NULL){
        *ndimsp = varp->ndim;
    }

    if (dimids != NULL){
        memcpy(dimids, varp->dimids, sizeof(int) * varp->ndim);
    }

    return NC_NOERR;
}

int
nczipio_rename_var(void       *ncdp,
                 int         varid,
                 const char *newname)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;
    NC_var *varp;

    if (varid < 0 || varid >= nczipp->vars.cnt){
        DEBUG_RETURN_ERROR(NC_EINVAL);
    }
    varp = nczipp->vars.data + varid;

    err = nczipp->driver->rename_var(nczipp->ncp, varp->varid, newname);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_iget_var(void             *ncdp,
               int               varid,
               const MPI_Offset *start,
               const MPI_Offset *count,
               const MPI_Offset *stride,
               const MPI_Offset *imap,
               void             *buf,
               MPI_Offset        bufcount,
               MPI_Datatype      buftype,
               int              *reqid,
               int               reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->iget_var(nczipp->ncp, varid, start, count, stride, imap,
                                buf, bufcount, buftype, reqid, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_iput_var(void             *ncdp,
               int               varid,
               const MPI_Offset *start,
               const MPI_Offset *count,
               const MPI_Offset *stride,
               const MPI_Offset *imap,
               const void       *buf,
               MPI_Offset        bufcount,
               MPI_Datatype      buftype,
               int              *reqid,
               int               reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->iput_var(nczipp->ncp, varid, start, count, stride, imap,
                                buf, bufcount, buftype, reqid, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_buffer_attach(void       *ncdp,
                    MPI_Offset  bufsize)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->buffer_attach(nczipp->ncp, bufsize);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_buffer_detach(void *ncdp)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->buffer_detach(nczipp->ncp);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_bput_var(void             *ncdp,
               int               varid,
               const MPI_Offset *start,
               const MPI_Offset *count,
               const MPI_Offset *stride,
               const MPI_Offset *imap,
               const void       *buf,
               MPI_Offset        bufcount,
               MPI_Datatype      buftype,
               int              *reqid,
               int               reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->bput_var(nczipp->ncp, varid, start, count, stride, imap,
                                buf, bufcount, buftype, reqid, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}
int
nczipio_get_varn(void              *ncdp,
               int                varid,
               int                num,
               MPI_Offset* const *starts,
               MPI_Offset* const *counts,
               void              *buf,
               MPI_Offset         bufcount,
               MPI_Datatype       buftype,
               int                reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->get_varn(nczipp->ncp, varid, num, starts, counts, buf,
                                bufcount, buftype, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_put_varn(void              *ncdp,
               int                varid,
               int                num,
               MPI_Offset* const *starts,
               MPI_Offset* const *counts,
               const void        *buf,
               MPI_Offset         bufcount,
               MPI_Datatype       buftype,
               int                reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->put_varn(nczipp->ncp, varid, num, starts, counts, buf,
                                bufcount, buftype, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_iget_varn(void               *ncdp,
                int                 varid,
                int                 num,
                MPI_Offset* const  *starts,
                MPI_Offset* const  *counts,
                void               *buf,
                MPI_Offset          bufcount,
                MPI_Datatype        buftype,
                int                *reqid,
                int                 reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->iget_varn(nczipp->ncp, varid, num, starts, counts, buf,
                                 bufcount, buftype, reqid, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_iput_varn(void               *ncdp,
                int                 varid,
                int                 num,
                MPI_Offset* const  *starts,
                MPI_Offset* const  *counts,
                const void         *buf,
                MPI_Offset          bufcount,
                MPI_Datatype        buftype,
                int                *reqid,
                int                 reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->iput_varn(nczipp->ncp, varid, num, starts, counts, buf,
                                 bufcount, buftype, reqid, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_bput_varn(void               *ncdp,
                int                 varid,
                int                 num,
                MPI_Offset* const  *starts,
                MPI_Offset* const  *counts,
                const void         *buf,
                MPI_Offset          bufcount,
                MPI_Datatype        buftype,
                int                *reqid,
                int                 reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->bput_varn(nczipp->ncp, varid, num, starts, counts, buf,
                                 bufcount, buftype, reqid, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_get_vard(void         *ncdp,
               int           varid,
               MPI_Datatype  filetype,
               void         *buf,
               MPI_Offset    bufcount,
               MPI_Datatype  buftype,
               int           reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->get_vard(nczipp->ncp, varid, filetype, buf, bufcount,
                                buftype, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

int
nczipio_put_vard(void         *ncdp,
               int           varid,
               MPI_Datatype  filetype,
               const void   *buf,
               MPI_Offset    bufcount,
               MPI_Datatype  buftype,
               int           reqMode)
{
    int err;
    NC_zip *nczipp = (NC_zip*)ncdp;

    DEBUG_RETURN_ERROR(NC_ENOTSUPPORT);

    err = nczipp->driver->put_vard(nczipp->ncp, varid, filetype, buf, bufcount,
                                buftype, reqMode);
    if (err != NC_NOERR) return err;

    return NC_NOERR;
}

