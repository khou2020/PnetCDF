#
# Copyright (C) 2003, Northwestern University and Argonne National Laboratory
# See COPYRIGHT notice in top-level directory.
#
# $Id$
#
# @configure_input@

SUFFIXES = .o .c

AM_CPPFLAGS  = -I$(top_srcdir)/src/include
AM_CPPFLAGS += -I$(srcdir)/../common
AM_CPPFLAGS += -I$(top_builddir)/src/include

LDADD = $(top_builddir)/src/libs/libpnetcdf.la ../common/libtestutils.la

if DECL_MPI_OFFSET
   # Do not add to AM_CPPFLAGS, as it will also be used by Fortran programs
   # AM_CPPFLAGS += -DHAVE_DECL_MPI_OFFSET
   AM_CFLAGS = -DHAVE_DECL_MPI_OFFSET
   # AM_FFLAGS += $(FC_DEFINE)HAVE_DECL_MPI_OFFSET
endif

TESTPROGRAMS = test_inq_format \
               cdf_type \
               dim_cdf12

check_PROGRAMS = $(TESTPROGRAMS) tst_open_cdf5 tst_corrupt

# autimake 1.11.3 has not yet implemented AM_TESTS_ENVIRONMENT
# For newer versions, we can use AM_TESTS_ENVIRONMENT instead
# AM_TESTS_ENVIRONMENT  = TESTPROGRAMS="$(TESTPROGRAMS)" ; export TESTPROGRAMS;
# AM_TESTS_ENVIRONMENT += TESTSEQRUN="$(TESTSEQRUN)" ; export TESTSEQRUN;
# AM_TESTS_ENVIRONMENT += TESTOUTDIR="$(TESTOUTDIR)" ; export TESTOUTDIR;
TESTS_ENVIRONMENT  = TESTPROGRAMS="$(TESTPROGRAMS)" ; export TESTPROGRAMS;
TESTS_ENVIRONMENT += TESTSEQRUN="$(TESTSEQRUN)" ; export TESTSEQRUN;
TESTS_ENVIRONMENT += TESTOUTDIR="$(TESTOUTDIR)" ; export TESTOUTDIR;
TESTS_ENVIRONMENT += BAD_FILES="$(BAD_FILES)" ; export BAD_FILES;

TESTS = cdf_type dim_cdf12 seq_runs.sh xfail_runs.sh
TEST_EXTENSIONS = .sh
LOG_COMPILER = $(srcdir)/wrap_runs.sh
SH_LOG_COMPILER =

if BUILD_DRIVER_BB
TESTBB = 1
TESTS_ENVIRONMENT += TESTBB=1 ; export TESTBB;
endif

BAD_FILES = bad_begin.nc5 \
            bad_xtype.nc1 bad_xtype.nc2 bad_xtype.nc5 \
            bad_ndims.nc1 bad_ndims.nc2 bad_ndims.nc5 \
            bad_dimid.nc1 bad_dimid.nc2 bad_dimid.nc5 \
            bad_nattrs.nc1 bad_nattrs.nc2 bad_nattrs.nc5

XFAIL_TESTS = xfail_runs.sh

CLEANFILES = $(TESTOUTDIR)/cdf_type.nc $(TESTOUTDIR)/dim_cdf12.nc \
             core core.* *.gcda *.gcno *.gcov gmon.out

EXTRA_DIST = wrap_runs.sh seq_runs.sh xfail_runs.sh \
             test_cdf1.nc test_cdf2.nc test_cdf5.nc \
             $(BAD_FILES)

../common/libtestutils.la:
	set -e; cd ../common && $(MAKE) $(MFLAGS) tests

TESTMPIRUN4  = `echo $(TESTMPIRUN) | $(SED) -e 's/NP/4/g'`

ptest ptests ptest4: $(TESTPROGRAMS)
	export PNETCDF_SAFE_MODE=0 && $(TESTMPIRUN4) ./test_inq_format $(srcdir)
	export PNETCDF_SAFE_MODE=0 && $(TESTMPIRUN4) ./cdf_type $(TESTOUTDIR)/cdf_type.nc
	export PNETCDF_SAFE_MODE=0 && $(TESTMPIRUN4) ./dim_cdf12 $(TESTOUTDIR)/dim_cdf12.nc
	export PNETCDF_SAFE_MODE=1 && $(TESTMPIRUN4) ./test_inq_format $(srcdir)
	export PNETCDF_SAFE_MODE=1 && $(TESTMPIRUN4) ./cdf_type $(TESTOUTDIR)/cdf_type.nc
	export PNETCDF_SAFE_MODE=1 && $(TESTMPIRUN4) ./dim_cdf12 $(TESTOUTDIR)/dim_cdf12.nc
	if [ -n "$(TESTBB)" ]; then ( \
	export PNETCDF_HINTS="nc_burst_buf=enable;nc_burst_buf_dirname=${TESTOUTDIR};" ; \
	export PNETCDF_SAFE_MODE=0 && $(TESTMPIRUN4) ./test_inq_format $(srcdir) ; \
	export PNETCDF_SAFE_MODE=0 && $(TESTMPIRUN4) ./cdf_type $(TESTOUTDIR)/cdf_type.nc ; \
	export PNETCDF_SAFE_MODE=0 && $(TESTMPIRUN4) ./dim_cdf12 $(TESTOUTDIR)/dim_cdf12.nc ; \
	export PNETCDF_SAFE_MODE=1 && $(TESTMPIRUN4) ./test_inq_format $(srcdir) ; \
	export PNETCDF_SAFE_MODE=1 && $(TESTMPIRUN4) ./cdf_type $(TESTOUTDIR)/cdf_type.nc ; \
	export PNETCDF_SAFE_MODE=1 && $(TESTMPIRUN4) ./dim_cdf12 $(TESTOUTDIR)/dim_cdf12.nc ; \
	unset PNETCDF_HINTS ; \
	) ; fi ;

ptest2 ptest6 ptest8 ptest10:

# build check targets but not invoke
tests-local: all $(check_PROGRAMS)

.PHONY: ptest ptests ptest2 ptest4 ptest6 ptest8 ptest10

