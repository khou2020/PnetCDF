#
# Copyright (C) 2014, Northwestern University and Argonne National Laboratory
# See COPYRIGHT notice in top-level directory.
#
# $Id$
#
# @configure_input@

SUFFIXES = .o .cpp

AM_DEFAULT_SOURCE_EXT = .cpp

AM_CPPFLAGS  = -I$(top_srcdir)/src/include
AM_CPPFLAGS += -I$(top_builddir)/src/binding/cxx
AM_CPPFLAGS += -I$(srcdir)/../common
AM_CPPFLAGS += -I$(top_builddir)/src/include
LDADD = $(top_builddir)/src/libs/libpnetcdf.la ../common/libtestutils.la

if DECL_MPI_OFFSET
   AM_CPPFLAGS += -DHAVE_DECL_MPI_OFFSET
endif

TESTPROGRAMS = nctst \
               test_classic

check_PROGRAMS = $(TESTPROGRAMS)

# autimake 1.11.3 has not yet implemented AM_TESTS_ENVIRONMENT
# For newer versions, we can use AM_TESTS_ENVIRONMENT instead
TESTS_ENVIRONMENT  = TESTSEQRUN="$(TESTSEQRUN)" ; export TESTSEQRUN;
TESTS_ENVIRONMENT += TESTOUTDIR="$(TESTOUTDIR)" ; export TESTOUTDIR;

if BUILD_DRIVER_BB
TESTBB = 1
TESTS_ENVIRONMENT += TESTBB=1 ; export TESTBB;
endif

TESTS = $(TESTPROGRAMS)
LOG_COMPILER = $(srcdir)/wrap_runs.sh

EXTRA_DIST = wrap_runs.sh

CLEANFILES = $(TESTOUTDIR)/nctst.nc $(TESTOUTDIR)/test_classic.nc \
             core core.* *.gcda *.gcno *.gcov gmon.out a.out

../common/libtestutils.la:
	set -e; cd ../common && $(MAKE) $(MFLAGS) tests

TESTMPIRUN4  = `echo $(TESTMPIRUN) | $(SED) -e 's/NP/4/g'`

ptest ptests ptest4: $(TESTPROGRAMS)
	for j in 0 1 ; do { \
	export PNETCDF_SAFE_MODE=$$j ; \
	set -e ; for i in $(TESTPROGRAMS); do ( \
	$(TESTMPIRUN4) ./$$i $(TESTOUTDIR)/$$i.nc ; \
	if [ -n "$(TESTBB)" ]; then ( \
	export PNETCDF_HINTS="nc_burst_buf=enable;nc_burst_buf_dirname=${TESTOUTDIR};" ; \
	$(TESTMPIRUN4) ./$$i $(TESTOUTDIR)/$$i.nc ; \
	unset PNETCDF_HINTS ; \
	) ; fi ; \
	) ; done ; } ; done

ptest2 ptest6 ptest8 ptest10:

# build check targets but not invoke
tests-local: all $(check_PROGRAMS)

.PHONY: ptest ptests ptest2 ptest4 ptest6 ptest8 ptest10

