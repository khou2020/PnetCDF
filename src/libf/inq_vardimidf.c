/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*  
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 * This file is automatically generated by buildiface -infile=../lib/pnetcdf.h -deffile=defs
 * DO NOT EDIT
 */
#include "mpinetcdf_impl.h"


#ifdef F77_NAME_UPPER
#define nfmpi_inq_vardimid_ NFMPI_INQ_VARDIMID
#elif defined(F77_NAME_LOWER_2USCORE)
#define nfmpi_inq_vardimid_ nfmpi_inq_vardimid__
#elif !defined(F77_NAME_LOWER_USCORE)
#define nfmpi_inq_vardimid_ nfmpi_inq_vardimid
/* Else leave name alone */
#endif


/* Prototypes for the Fortran interfaces */
#include "mpifnetcdf.h"
FORTRAN_API void FORT_CALL nfmpi_inq_vardimid_ ( int *v1, int *v2, MPI_Fint *v3, MPI_Fint *ierr ){
    int *l3=0, ln3;

    ln3 = ncmpixVardim(*v1,*v2-1);
    if (ln3 > 0) {
        l3 = (size_t *)malloc( ln3 * sizeof(int) );
    }
    *ierr = ncmpi_inq_vardimid( *v1, *v2, l3 );

    if (l3) { 
	int li;
        for (li=0; li<ln3; li++) 
            v3[li] = l3[ln3-1-li] + 1;
        free(l3); }
}
