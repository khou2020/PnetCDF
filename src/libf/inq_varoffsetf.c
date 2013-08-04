/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*  
 *  (C) 2003 by Argonne National Laboratory and Northwestern University.
 *      See COPYRIGHT in top-level directory.
 *
 * This file is automatically generated by ./buildiface -infile=../lib/pnetcdf.h -deffile=defs
 * DO NOT EDIT
 */
#include "mpinetcdf_impl.h"


#ifdef F77_NAME_UPPER
#define nfmpi_inq_varoffset_ NFMPI_INQ_VAROFFSET
#elif defined(F77_NAME_LOWER_2USCORE)
#define nfmpi_inq_varoffset_ nfmpi_inq_varoffset__
#elif !defined(F77_NAME_LOWER_USCORE)
#define nfmpi_inq_varoffset_ nfmpi_inq_varoffset
/* Else leave name alone */
#endif


/* Prototypes for the Fortran interfaces */
#include "mpifnetcdf.h"
FORTRAN_API int FORT_CALL nfmpi_inq_varoffset_ ( int *v1, int *v2, MPI_Offset *v3 ){
    int ierr;
    int l2 = *v2 - 1;
    ierr = ncmpi_inq_varoffset( *v1, l2, v3 );
    return ierr;
}
