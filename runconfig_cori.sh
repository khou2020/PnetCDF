./configure --prefix=/global/homes/k/khl7265/local/ncmpi_eval \
                cross_compiling="yes" \
                CFLAGS="-fast -no-ipo -O2" CXXFLAGS="-fast -no-ipo -O2" \
                FFLAGS="-fast -no-ipo -O2"  FCFLAGS="-fast -no-ipo -O2" \
                TESTMPIRUN="srun -n NP" TESTSEQRUN="srun -n 1" \
                TESTOUTDIR="$SCRATCH" \
		--disable-shared --disable-debug \
                --enable-profiling --enable-burst-buffering \
                --enable-staging
