dnl Process this m4 file to produce 'C' language file.
dnl
dnl If you see this line, you can ignore the next one.
/* Do not edit this file. It is produced from the corresponding .m4 source */
dnl
/*
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

#if HAVE_CONFIG_H
# include <ncconfig.h>
#endif

#include <stdio.h>
#include <unistd.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_INTTYPES_H
#include <inttypes.h> /* uint16_t, uint32_t, uint64_t */
#elif defined(HAVE_STDINT_H)
#include <stdint.h>   /* uint16_t, uint32_t, uint64_t */
#endif
#include <assert.h>

#include <mpi.h>

#include "nc.h"
#include "ncx.h"
#include "macro.h"

/*
 *  Datatype Mapping:
 *
 *  NETCDF    <--> MPI                    Description
 *   NC_BYTE       MPI_SIGNED_CHAR        signed 1-byte integer
 *   NC_CHAR       MPI_CHAR               char, text (cannot convert to other types)
 *   NC_SHORT      MPI_SHORT              signed 2-byte integer
 *   NC_INT        MPI_INT                signed 4-byte integer
 *   NC_FLOAT      MPI_FLOAT              single precision floating point
 *   NC_DOUBLE     MPI_DOUBLE             double precision floating point
 *   NC_UBYTE      MPI_UNSIGNED_CHAR      unsigned 1-byte int
 *   NC_USHORT     MPI_UNSIGNED_SHORT     unsigned 2-byte int
 *   NC_UINT       MPI_UNSIGNED           unsigned 4-byte int
 *   NC_INT64      MPI_LONG_LONG_INT      signed 8-byte int
 *   NC_UINT64     MPI_UNSIGNED_LONG_LONG unsigned 8-byte int
 *
 *  Assume: MPI_Datatype and nc_type are both enumerable types
 *          (this might not conform with MPI, as MPI_Datatype is intended to be
 *           an opaque data type.)
 *
 *  In OpenMPI, this assumption will fail
 */

inline MPI_Datatype
ncmpii_nc2mpitype(nc_type xtype)
{
    switch(xtype){
        case NC_BYTE :   return MPI_SIGNED_CHAR;
        case NC_CHAR :   return MPI_CHAR;
        case NC_SHORT :  return MPI_SHORT;
        case NC_INT :    return MPI_INT;
        case NC_FLOAT :  return MPI_FLOAT;
        case NC_DOUBLE : return MPI_DOUBLE;
        case NC_UBYTE :  return MPI_UNSIGNED_CHAR;
        case NC_USHORT : return MPI_UNSIGNED_SHORT;
        case NC_UINT :   return MPI_UNSIGNED;
        case NC_INT64 :  return MPI_LONG_LONG_INT;
        case NC_UINT64 : return MPI_UNSIGNED_LONG_LONG;
        default:         return MPI_DATATYPE_NULL;
    }
}

inline nc_type
ncmpii_mpi2nctype(MPI_Datatype itype)
{
    if (itype == MPI_SIGNED_CHAR)        return NC_BYTE ;
    if (itype == MPI_CHAR)               return NC_CHAR ;
    if (itype == MPI_SHORT)              return NC_SHORT ;
    if (itype == MPI_INT)                return NC_INT ;
    if (itype == MPI_FLOAT)              return NC_FLOAT ;
    if (itype == MPI_DOUBLE)             return NC_DOUBLE ;
    if (itype == MPI_UNSIGNED_CHAR)      return NC_UBYTE ;
    if (itype == MPI_UNSIGNED_SHORT)     return NC_USHORT ;
    if (itype == MPI_UNSIGNED)           return NC_UINT ;
    if (itype == MPI_LONG_LONG_INT)      return NC_INT64 ;
    if (itype == MPI_UNSIGNED_LONG_LONG) return NC_UINT64 ;
    return NC_EBADTYPE;
}

/*----< ncmpii_need_convert() >----------------------------------------------*/
/* netCDF specification makes a special case for type conversion between
 * uchar and NC_BYTE: do not check for range error. See
 * http://www.unidata.ucar.edu/software/netcdf/docs/data_type.html#type_conversion
 */
inline int
ncmpii_need_convert(int          format, /* 1, 2, or 5 (CDF format number) */
                    nc_type      xtype,  /* external NC type */
                    MPI_Datatype itype)  /* internal MPI type */
{

    if (format > 2) { /* NC_BYTE is considered signed 1-byte integer */
        if ((xtype == NC_BYTE  && itype == MPI_UNSIGNED_CHAR)
#if defined(__CHAR_UNSIGNED__) && __CHAR_UNSIGNED__ != 0
            || (xtype == NC_BYTE  && itype == MPI_CHAR)
#endif
           )
       return 1;
    }

    return !( (xtype == NC_CHAR   && itype == MPI_CHAR)           ||
              (xtype == NC_BYTE   && itype == MPI_SIGNED_CHAR)    ||
              (xtype == NC_BYTE   && itype == MPI_UNSIGNED_CHAR)  ||
#if defined(__CHAR_UNSIGNED__) && __CHAR_UNSIGNED__ != 0
              (xtype == NC_BYTE   && itype == MPI_CHAR)           ||
#endif
              (xtype == NC_SHORT  && itype == MPI_SHORT)          ||
              (xtype == NC_INT    && itype == MPI_INT)            ||
              (xtype == NC_INT    && itype == MPI_LONG &&
               X_SIZEOF_INT == SIZEOF_LONG)                       ||
              (xtype == NC_FLOAT  && itype == MPI_FLOAT)          ||
              (xtype == NC_DOUBLE && itype == MPI_DOUBLE)         ||
              (xtype == NC_UBYTE  && itype == MPI_UNSIGNED_CHAR)  ||
#if defined(__CHAR_UNSIGNED__) && __CHAR_UNSIGNED__ != 0
              (xtype == NC_UBYTE  && itype == MPI_CHAR)           ||
#endif
              (xtype == NC_USHORT && itype == MPI_UNSIGNED_SHORT) ||
              (xtype == NC_UINT   && itype == MPI_UNSIGNED)       ||
              (xtype == NC_INT64  && itype == MPI_LONG_LONG_INT)  ||
              (xtype == NC_UINT64 && itype == MPI_UNSIGNED_LONG_LONG)
            );
}

/*----< ncmpii_need_swap() >-------------------------------------------------*/
inline int
ncmpii_need_swap(nc_type      xtype,  /* external NC type */
                 MPI_Datatype itype)  /* internal MPI type */
{
#ifdef WORDS_BIGENDIAN
    return 0;
#else
    if ((xtype == NC_CHAR  && itype == MPI_CHAR)           ||
        (xtype == NC_BYTE  && itype == MPI_SIGNED_CHAR)    ||
        (xtype == NC_UBYTE && itype == MPI_UNSIGNED_CHAR))
        return 0;

    return 1;
#endif
}

/* Endianness byte swap: done in-place */
#define SWAP(x,y) {tmp = (x); (x) = (y); (y) = tmp;}

/*----< ncmpii_swap() >-------------------------------------------------------*/
/* out-place byte swap, i.e. dest_p != src_p */
void
ncmpii_swapn(void       *dest_p,  /* destination array */
             const void *src_p,   /* source array */
             MPI_Offset  nelems,  /* number of elements in buf[] */
             int         esize)   /* byte size of each element */
{
    int  i;

    if (esize <= 1 || nelems <= 0) return;  /* no need */

    if (esize == 4) { /* this is the most common case */
              uint32_t *dest = (uint32_t*)       dest_p;
        const uint32_t *src  = (const uint32_t*) src_p;
        for (i=0; i<nelems; i++) {
            dest[i] = src[i];
            dest[i] =  ((dest[i]) << 24)
                    | (((dest[i]) & 0x0000ff00) << 8)
                    | (((dest[i]) & 0x00ff0000) >> 8)
                    | (((dest[i]) >> 24));
        }
    }
    else if (esize == 8) {
              uint64_t *dest = (uint64_t*)       dest_p;
        const uint64_t *src  = (const uint64_t*) src_p;
        for (i=0; i<nelems; i++) {
            dest[i] = src[i];
            dest[i] = ((dest[i] & 0x00000000000000FFULL) << 56) | 
                      ((dest[i] & 0x000000000000FF00ULL) << 40) | 
                      ((dest[i] & 0x0000000000FF0000ULL) << 24) | 
                      ((dest[i] & 0x00000000FF000000ULL) <<  8) | 
                      ((dest[i] & 0x000000FF00000000ULL) >>  8) | 
                      ((dest[i] & 0x0000FF0000000000ULL) >> 24) | 
                      ((dest[i] & 0x00FF000000000000ULL) >> 40) | 
                      ((dest[i] & 0xFF00000000000000ULL) >> 56);
        }
    }
    else if (esize == 2) {
              uint16_t *dest =       (uint16_t*) dest_p;
        const uint16_t *src  = (const uint16_t*) src_p;
        for (i=0; i<nelems; i++) {
            dest[i] = src[i];
            dest[i] = ((dest[i] & 0xff) << 8) |
                      ((dest[i] >> 8) & 0xff);
        }
    }
    else {
              uchar *op = (uchar*) dest_p;
        const uchar *ip = (uchar*) src_p;
        /* for esize is not 1, 2, or 4 */
        while (nelems-- > 0) {
            for (i=0; i<esize; i++)
                op[i] = ip[esize-1-i];
            op += esize;
            ip += esize;
        }
    }
}

/* Other options to in-place byte-swap
htonl() is for 4-byte swap
htons() is for 2-byte swap

#include <arpa/inet.h>
    dest[i] = htonl(dest[i]);
    dest[i] = htons(dest[i]);

Or

#include <byteswap.h>

        for (i=0; i<nelems; i++)
            dest[i] = __bswap_32(dest[i]);

*/

/*----< ncmpii_in_swap() >---------------------------------------------------*/
/* in-place byte swap */
void
ncmpii_in_swapn(void       *buf,
                MPI_Offset  nelems,  /* number of elements in buf[] */
                int         esize)   /* byte size of each element */
{
    int i;

    if (esize <= 1 || nelems <= 0) return;  /* no need */

    if (esize == 4) { /* this is the most common case */
        uint32_t *dest = (uint32_t*) buf;
        for (i=0; i<nelems; i++)
            dest[i] =  ((dest[i]) << 24)
                    | (((dest[i]) & 0x0000ff00) << 8)
                    | (((dest[i]) & 0x00ff0000) >> 8)
                    | (((dest[i]) >> 24));
    }
    else if (esize == 8) {
        uint64_t *dest = (uint64_t*) buf;
        for (i=0; i<nelems; i++)
            dest[i] = ((dest[i] & 0x00000000000000FFULL) << 56) | 
                      ((dest[i] & 0x000000000000FF00ULL) << 40) | 
                      ((dest[i] & 0x0000000000FF0000ULL) << 24) | 
                      ((dest[i] & 0x00000000FF000000ULL) <<  8) | 
                      ((dest[i] & 0x000000FF00000000ULL) >>  8) | 
                      ((dest[i] & 0x0000FF0000000000ULL) >> 24) | 
                      ((dest[i] & 0x00FF000000000000ULL) >> 40) | 
                      ((dest[i] & 0xFF00000000000000ULL) >> 56);
    }
    else if (esize == 2) {
        uint16_t *dest = (uint16_t*) buf;
        for (i=0; i<nelems; i++)
            dest[i] = ((dest[i] & 0xff) << 8) |
                      ((dest[i] >> 8) & 0xff);
    }
    else {
        uchar tmp, *op = (uchar*)buf;
        /* for esize is not 1, 2, or 4 */
        while (nelems-- > 0) {
            for (i=0; i<esize/2; i++)
                SWAP(op[i], op[esize-1-i])
            op += esize;
        }
    }
}

dnl
dnl PUTN_XTYPE(xtype)
dnl
define(`PUTN_XTYPE',dnl
`dnl
/*----< ncmpii_x_putn_$1() >--------------------------------------------------*/
inline int
ncmpii_x_putn_$1(ifelse(`$1',`NC_BYTE',`int cdf_ver,/* 1,2,or 5 CDF format */')
              void         *xp,     /* buffer of external type $1 */
              const void   *buf,    /* user buffer of internal type, itype */
              MPI_Offset    nelems,
              MPI_Datatype  itype,  /* internal data type (MPI_Datatype) */
              void         *fillp)  /* in internal representation */
{
    if (itype == MPI_CHAR || itype == MPI_SIGNED_CHAR)
        /* This is for 1-byte integer, assuming ECHAR has been checked before */
        return ncmpix_putn_$1_schar(&xp, nelems, (signed char*)buf, fillp);
    else if (itype == MPI_UNSIGNED_CHAR) {
        ifelse(`$1',`NC_BYTE',
       `if (cdf_ver < 5)
            return ncmpix_putn_NC_UBYTE_uchar(&xp, nelems,(const uchar*)buf, fillp);
        else')
            return ncmpix_putn_$1_uchar(&xp, nelems, (const uchar*)     buf, fillp);
    }
    else if (itype == MPI_SHORT)
        return ncmpix_putn_$1_short    (&xp, nelems, (const short*)     buf, fillp);
    else if (itype == MPI_UNSIGNED_SHORT)
        return ncmpix_putn_$1_ushort   (&xp, nelems, (const ushort*)    buf, fillp);
    else if (itype == MPI_INT)
        return ncmpix_putn_$1_int      (&xp, nelems, (const int*)       buf, fillp);
    else if (itype == MPI_UNSIGNED)
        return ncmpix_putn_$1_uint     (&xp, nelems, (const uint*)      buf, fillp);
    else if (itype == MPI_LONG)
        return ncmpix_putn_$1_long     (&xp, nelems, (const long*)      buf, fillp);
    else if (itype == MPI_FLOAT)
        return ncmpix_putn_$1_float    (&xp, nelems, (const float*)     buf, fillp);
    else if (itype == MPI_DOUBLE)
        return ncmpix_putn_$1_double   (&xp, nelems, (const double*)    buf, fillp);
    else if (itype == MPI_LONG_LONG_INT)
        return ncmpix_putn_$1_longlong (&xp, nelems, (const longlong*)  buf, fillp);
    else if (itype == MPI_UNSIGNED_LONG_LONG)
        return ncmpix_putn_$1_ulonglong(&xp, nelems, (const ulonglong*) buf, fillp);
    DEBUG_RETURN_ERROR(NC_EBADTYPE)
}
')dnl

PUTN_XTYPE(NC_UBYTE)
PUTN_XTYPE(NC_SHORT)
PUTN_XTYPE(NC_USHORT)
PUTN_XTYPE(NC_INT)
PUTN_XTYPE(NC_UINT)
PUTN_XTYPE(NC_FLOAT)
PUTN_XTYPE(NC_DOUBLE)
PUTN_XTYPE(NC_INT64)
PUTN_XTYPE(NC_UINT64)

/* In CDF-2, NC_BYTE is considered a signed 1-byte integer in signed APIs, and
 * unsigned 1-byte integer in unsigned APIs. In CDF-5, NC_BYTE is always a
 * signed 1-byte integer. See
 * http://www.unidata.ucar.edu/software/netcdf/docs/data_type.html#type_conversion
 */
PUTN_XTYPE(NC_BYTE)


dnl
dnl GETN_XTYPE(xtype)
dnl
define(`GETN_XTYPE',dnl
`dnl
/*----< ncmpii_x_getn_$1() >-------------------------------------------------*/
inline int
ncmpii_x_getn_$1(ifelse(`$1',`NC_BYTE',`int cdf_ver,/* 1,2,or 5 CDF format */')
              const void   *xp,     /* buffer of external type $1 */
              void         *ip,     /* user buffer of internal type, itype */
              MPI_Offset    nelems,
              MPI_Datatype  itype,  /* internal data type (MPI_Datatype) */
              void         *fillp)  /* in internal representation */
{
    if (itype == MPI_CHAR || itype == MPI_SIGNED_CHAR)
        /* This is for 1-byte integer, assuming ECHAR has been checked before */
        return ncmpix_getn_$1_schar(&xp, nelems, (signed char*)ip, *(schar*)fillp);
    else if (itype == MPI_UNSIGNED_CHAR) {
        ifelse(`$1',`NC_BYTE',`if (cdf_ver < 5)
            return ncmpix_getn_NC_UBYTE_uchar(&xp, nelems,(uchar*)ip, *(uchar*)fillp);
        else')
            return ncmpix_getn_$1_uchar(&xp, nelems,      (uchar*)ip, *(uchar*)fillp);
    }
    else if (itype == MPI_SHORT)
        return ncmpix_getn_$1_short    (&xp, nelems,      (short*)ip, *(short*)fillp);
    else if (itype == MPI_UNSIGNED_SHORT)
        return ncmpix_getn_$1_ushort   (&xp, nelems,     (ushort*)ip, *(ushort*)fillp);
    else if (itype == MPI_INT)
        return ncmpix_getn_$1_int      (&xp, nelems,        (int*)ip, *(int*)fillp);
    else if (itype == MPI_UNSIGNED)
        return ncmpix_getn_$1_uint     (&xp, nelems,       (uint*)ip, *(uint*)fillp);
    else if (itype == MPI_LONG)
        return ncmpix_getn_$1_long     (&xp, nelems,       (long*)ip, *(long*)fillp);
    else if (itype == MPI_FLOAT)
        return ncmpix_getn_$1_float    (&xp, nelems,      (float*)ip, *(float*)fillp);
    else if (itype == MPI_DOUBLE)
        return ncmpix_getn_$1_double   (&xp, nelems,     (double*)ip, *(double*)fillp);
    else if (itype == MPI_LONG_LONG_INT)
        return ncmpix_getn_$1_longlong (&xp, nelems,   (longlong*)ip, *(longlong*)fillp);
    else if (itype == MPI_UNSIGNED_LONG_LONG)
        return ncmpix_getn_$1_ulonglong(&xp, nelems,  (ulonglong*)ip, *(ulonglong*)fillp);
    DEBUG_RETURN_ERROR(NC_EBADTYPE)
}
')dnl

GETN_XTYPE(NC_UBYTE)
GETN_XTYPE(NC_SHORT)
GETN_XTYPE(NC_USHORT)
GETN_XTYPE(NC_INT)
GETN_XTYPE(NC_UINT)
GETN_XTYPE(NC_FLOAT)
GETN_XTYPE(NC_DOUBLE)
GETN_XTYPE(NC_INT64)
GETN_XTYPE(NC_UINT64)

/* In CDF-2, NC_BYTE is considered a signed 1-byte integer in signed APIs, and
 * unsigned 1-byte integer in unsigned APIs. In CDF-5, NC_BYTE is always a
 * signed 1-byte integer. See
 * http://www.unidata.ucar.edu/software/netcdf/docs/data_type.html#type_conversion
 */
GETN_XTYPE(NC_BYTE)

