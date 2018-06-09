#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <pnetcdf.h>
#include <pnc_debug.h>
#include <common.h>
#include <mpi.h>

/*----< ncmpi_inq_buffer_size() >--------------------------------------------*/
/* This is an independent subroutine. Wrapper for MPI_Finalize */
int
ncmpi_mpi_finalize()
{
    return MPI_Finalize();
}