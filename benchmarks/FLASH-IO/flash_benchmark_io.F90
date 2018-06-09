      program flash_benchmark_io          
            use pnetcdf
!
! This is a sample program that setups the FLASH data structures and
! drives the I/O routines.  It is intended for benchmarking the I/O
! performance.
!

! #ifndef MPI_OFFSET
! #define MPI_OFFSET MPI_INTEGER8
! #endif

#ifdef NAGFortran
      USE F90_UNIX_ENV, only : iargc, getarg
#endif

! the main data structures are contained in common blocks, defined in the
! include files

#include "common.fh"

#ifndef NAGFortran
      integer iargc
#endif
      integer i, argc, ierr
      character(len=128) executable
      logical verbose, isArgvRight

      double precision time_io(3), time_begin
      double precision chk_io, corner_io, nocorner_io
      double precision checkpoint_wr_ncmpi_par
      double precision plotfile_ncmpi_par

      character*16 tmp

      integer, parameter :: local_blocks = INT(0.8*maxblocks)

! initialize MPI and get the rank and size
      call MPI_INIT(ierr)
      call MPI_Comm_Rank (MPI_Comm_World, MyPE, ierr)
      call MPI_Comm_Size (MPI_Comm_World, NumPEs, ierr)

      MasterPE = 0
      verbose = .TRUE.
      use_indep_io = .FALSE.
      use_nonblocking_io = .TRUE.

      ! root process reads command-line arguments
      if (MyPE .EQ. MasterPE) then
            isArgvRight = .TRUE.
            argc = IARGC()   ! IARGC() does not count the executable name
            call getarg(0, executable)
            call getarg(1, basenm)

            if (argc .GT. 1) then
                  call getarg(2, tmp)
                  if (TRIM(tmp) .EQ. 'blocking') then
                        use_nonblocking_io = .FALSE.
                  endif
            endif

            if (argc .GT. 2) then
                  call getarg(3, tmp)
                  if (TRIM(tmp) .EQ. 'indep') then
                        use_indep_io = .TRUE.
                  endif
            endif
      endif

      ! Bcast io mode
      call MPI_Bcast(use_nonblocking_io, 1, MPI_LOGICAL, MasterPE, MPI_COMM_WORLD, ierr)
      call MPI_Bcast(use_indep_io, 1, MPI_LOGICAL, MasterPE, MPI_COMM_WORLD, ierr)

      ! broadcast if command-line arguments are valid
      call MPI_Bcast(isArgvRight, 1, MPI_LOGICAL, MasterPE, &
                     MPI_COMM_WORLD, ierr)
      if (.NOT. isArgvRight) goto 999

      ! broadcast file base name prefix
      call MPI_Bcast(basenm, 128, MPI_CHARACTER, MasterPE, &
                     MPI_COMM_WORLD, ierr)

! put ~100 blocks on each processor -- make it vary a little, since it does
! in the real application.  This is the maximum that we can fit on Blue
! Pacific comfortably.
      lnblocks = local_blocks + mod(MyPE,3)

! just fill the tree stucture with dummy information -- we are just going to
! dump it out
      size(:,:) = 0.5e0
      lrefine(:) = 1
      nodetype(:) = 1
      refine(:) = .FALSE.
      derefine(:) = .FALSE.
      parent(:,:) = -1
      child(:,:,:) = -1
      coord(:,:) = 0.25e0
      bnd_box(:,:,:) = 0.e0
      neigh(:,:,:) = -1
      empty(:) = 0

      ! use nonblocking APIs
      ! use_nonblocking_io = .TRUE.
      ! use_nonblocking_io = .FALSE.

! initialize the unknowns with the index of the variable
      do i = 1, nvar
        unk(i,:,:,:,:) = float(i)
      enddo

!---------------------------------------------------------------------------
! netCDF checkpoint file
!---------------------------------------------------------------------------
      time_begin = MPI_Wtime()
      chk_io = checkpoint_wr_ncmpi_par(0,0.e0) - barr_time
      time_io(1) = MPI_Wtime() - time_begin

!---------------------------------------------------------------------------
! netCDF plotfile -- no corners
!---------------------------------------------------------------------------
      time_begin = MPI_Wtime()
      nocorner_io = plotfile_ncmpi_par(0,0.e0,.false.) - barr_time
      time_io(2) = MPI_Wtime() - time_begin

!---------------------------------------------------------------------------
! netCDF plotfile -- corners
!---------------------------------------------------------------------------
      time_begin = MPI_Wtime()
      corner_io = plotfile_ncmpi_par(0,0.e0,.true.) - barr_time
      time_io(3) = MPI_Wtime() - time_begin

      call report_io_performance(verbose, local_blocks, time_io, chk_io, &
                                 corner_io, nocorner_io)

 999  ierr = nf90mpi_mpi_finalize()

      end program flash_benchmark_io

! ---------------------------------------------------------------------------
! print mpi info
! ---------------------------------------------------------------------------
      subroutine print_info(info_used)
          use pnetcdf
          
          implicit none
          include "mpif.h"

          integer info_used

          ! local variables
          character*(MPI_MAX_INFO_VAL) key, value
          integer nkeys, i, err
          logical flag

1010 format('#%$: ', A32, ': ', A64)

          call MPI_Info_get_nkeys(info_used, nkeys, err)
          print *, 'MPI File Info: nkeys =', nkeys
          do i=0, nkeys-1
              call MPI_Info_get_nthkey(info_used, i, key, err)
              call MPI_Info_get(info_used, key, MPI_MAX_INFO_VAL, value, flag, err)
              print 1010, key, value
          enddo
          print *

          return
      end ! subroutine print_info


! ---------------------------------------------------------------------------
! get the file striping information from the MPI info objects
! ---------------------------------------------------------------------------
      subroutine get_file_striping(info, striping_factor, striping_unit)
          implicit none
          include 'mpif.h'
          integer, intent(in)  :: info
          integer, intent(out) :: striping_factor
          integer, intent(out) :: striping_unit

          ! local variables
          character*(MPI_MAX_INFO_VAL) key, value
          integer                      i, nkeys, valuelen, ierr
          logical                      flag

          call MPI_Info_get_nkeys(info, nkeys, ierr)
          do i=0, nkeys-1
              key(:) = ' '
              call MPI_Info_get_nthkey(info, i, key, ierr)
              call MPI_Info_get(info, key, MPI_MAX_INFO_VAL, value, &
                                flag, ierr)
              call MPI_Info_get_valuelen(info, key, valuelen, flag, &
                                ierr)
              value(valuelen+1:) = ' '
              if (key(len_trim(key):len_trim(key)) .EQ. char(0)) &
                  key(len_trim(key):) = ' '
              if (trim(key) .EQ. 'striping_factor') &
                  read(value, '(i10)') striping_factor
              if (trim(key) .EQ. 'striping_unit') &
                  read(value, '(i10)') striping_unit
          enddo
      end subroutine get_file_striping

!---------------------------------------------------------------------------
! print I/O performance numbers
!---------------------------------------------------------------------------
      subroutine report_io_performance(verbose, local_blocks, time_io, &
                                       chk_io, corner_io, nocorner_io)
       use pnetcdf
#include "common.fh"

       logical verbose
       integer local_blocks, i
       double precision time_io(3)
       double precision chk_io, corner_io, nocorner_io

       ! local variables
       integer ierr, striping_factor, striping_unit, MaxPE, err
       double precision tmax(3), ttotal(2), time_total, io_amount, bw
       integer(kind=MPI_OFFSET_KIND) malloc_size, sum_size
       integer(kind=MPI_OFFSET_KIND) bb_data, bb_meta, bb_buffer
       integer(kind=MPI_OFFSET_KIND) bb_meta_all, bb_data_all, bb_buffer_all
       double precision time_io_max(3), time_io_min(3), time_io_mean(3), time_io_var(3)
       double precision chk_t_max(3), chk_t_min(3), chk_t_mean(3), chk_t_var(3)
       double precision corner_t_max(3), corner_t_min(3), corner_t_mean(3), corner_t_var(3)
       double precision nocorner_t_max(3), nocorner_t_min(3), nocorner_t_mean(3), nocorner_t_var(3)
       double precision bb_time(13), bb_time_max(13), bb_time_min(13), bb_time_mean(13), bb_time_var(13)
       double precision var(13), total_max, total_min, total_mean, total_var
       double precision time_staging

      err = nfmpi_inq_bb_time( bb_time(1), bb_time(2), bb_time(3), bb_time(4), bb_time(5), bb_time(6))
      err = nfmpi_inq_bb_time_put(bb_time(7), bb_time(8), bb_time(9))
      err = nfmpi_inq_bb_time_flush(bb_time(10), bb_time(11), bb_time(12), bb_time(13))
      err = nfmpi_inq_bb_size(bb_data, bb_meta, bb_buffer)

      ttotal(1) = time_io(1) + time_io(2) + time_io(3)
      ttotal(2) = MyPE
      call MPI_Allreduce(ttotal, tmax, 1, MPI_2DOUBLE_PRECISION, MPI_MAXLOC, MPI_COMM_WORLD, ierr)
      MaxPE = tmax(2)
      ! MaxPE = 0

      call MPI_Reduce(time_io, time_io_max, 3, MPI_DOUBLE_PRECISION, MPI_max, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Reduce(time_io, time_io_min, 3, MPI_DOUBLE_PRECISION, MPI_min, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Allreduce(time_io, time_io_mean, 3, MPI_DOUBLE_PRECISION, MPI_sum, MPI_COMM_WORLD, ierr)
      do 100 i = 1, 3
            time_io_mean(i) = time_io_mean(i) / NumPEs
            var(i) = (time_io(i) - time_io_mean(i)) * (time_io(i) - time_io_mean(i))
100   continue
      call MPI_Reduce(var, time_io_var, 3, MPI_DOUBLE_PRECISION, MPI_sum, MaxPE, MPI_COMM_WORLD, ierr)
      do 110 i = 1, 3
            time_io_var(i) = time_io_var(i) / NumPEs
110   continue

      call MPI_Reduce(chk_t, chk_t_max, 3, MPI_DOUBLE_PRECISION, MPI_max, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Reduce(chk_t, chk_t_min, 3, MPI_DOUBLE_PRECISION, MPI_min, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Allreduce(chk_t, chk_t_mean, 3, MPI_DOUBLE_PRECISION, MPI_sum, MPI_COMM_WORLD, ierr)
      do 120 i = 1, 3
            chk_t_mean(i) = chk_t_mean(i) / NumPEs
            var(i) = (chk_t(i) - chk_t_mean(i)) * (chk_t(i) - chk_t_mean(i))
120   continue
      call MPI_Reduce(var, chk_t_var, 3, MPI_DOUBLE_PRECISION, MPI_sum, MaxPE, MPI_COMM_WORLD, ierr)
      do 130 i = 1, 3
            chk_t_var(i) = chk_t_var(i) / NumPEs
130   continue

      call MPI_Reduce(corner_t, corner_t_max, 3, MPI_DOUBLE_PRECISION, MPI_max, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Reduce(corner_t, corner_t_min, 3, MPI_DOUBLE_PRECISION, MPI_min, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Allreduce(corner_t, corner_t_mean, 3, MPI_DOUBLE_PRECISION, MPI_sum, MPI_COMM_WORLD, ierr)
      do 140 i = 1, 3
            corner_t_mean(i) = corner_t_mean(i) / NumPEs
            var(i) = (corner_t(i) - corner_t_mean(i)) * (corner_t(i) - corner_t_mean(i))
140   continue
      call MPI_Reduce(var, corner_t_var, 3, MPI_DOUBLE_PRECISION, MPI_sum, MaxPE, MPI_COMM_WORLD, ierr)
      do 150 i = 1, 3
            corner_t_var(i) = corner_t_var(i) / NumPEs
150   continue

      call MPI_Reduce(nocorner_t, nocorner_t_max, 3, MPI_DOUBLE_PRECISION, MPI_max, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Reduce(nocorner_t, nocorner_t_min, 3, MPI_DOUBLE_PRECISION, MPI_min, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Allreduce(nocorner_t, nocorner_t_mean, 3, MPI_DOUBLE_PRECISION, MPI_sum, MPI_COMM_WORLD, ierr)
      do 160 i = 1, 3
            nocorner_t_mean(i) = nocorner_t_mean(i) / NumPEs
            var(i) = (nocorner_t(i) - nocorner_t_mean(i)) * (nocorner_t(i) - nocorner_t_mean(i))
160   continue
      call MPI_Reduce(var, nocorner_t_var, 3, MPI_DOUBLE_PRECISION, MPI_sum, MaxPE, MPI_COMM_WORLD, ierr)
      do 170 i = 1, 3
            nocorner_t_var(i) = nocorner_t_var(i) / NumPEs
170   continue

      call MPI_Reduce(bb_time, bb_time_max, 13, MPI_DOUBLE_PRECISION, MPI_max, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Reduce(bb_time, bb_time_min, 13, MPI_DOUBLE_PRECISION, MPI_min, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Allreduce(bb_time, bb_time_mean, 13, MPI_DOUBLE_PRECISION, MPI_sum, MPI_COMM_WORLD, ierr)
      do 180 i = 1, 13
            bb_time_mean(i) = bb_time_mean(i) / NumPEs
            var(i) = (bb_time(i) - bb_time_mean(i)) * (bb_time(i) - bb_time_mean(i))
180   continue
      call MPI_Reduce(var, bb_time_var, 13, MPI_DOUBLE_PRECISION, MPI_sum, MaxPE, MPI_COMM_WORLD, ierr)
      do 190 i = 1, 13
            bb_time_var(i) = bb_time_var(i) / NumPEs
190   continue

      time_total = time_io(1) + time_io(2) + time_io(3)
      call MPI_Reduce(time_total, total_max, 1, MPI_DOUBLE_PRECISION, MPI_max, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Reduce(time_total, total_min, 1, MPI_DOUBLE_PRECISION, MPI_min, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Allreduce(time_total, total_mean, 1, MPI_DOUBLE_PRECISION, MPI_sum, MPI_COMM_WORLD, ierr)
      total_mean = total_mean / NumPEs
      var(1) = (time_total - total_mean) * (time_total - total_mean)
      call MPI_Reduce(var(1), total_var, 1, MPI_DOUBLE_PRECISION, MPI_sum, MaxPE, MPI_COMM_WORLD, ierr)
      total_var = total_var / NumPEs

      call MPI_Reduce(bb_meta, bb_meta_all, 1, MPI_OFFSET, MPI_SUM, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Reduce(bb_data, bb_data_all, 1, MPI_OFFSET, MPI_SUM, MaxPE, MPI_COMM_WORLD, ierr)
      call MPI_Reduce(bb_buffer, bb_buffer_all, 1, MPI_OFFSET, MPI_max, MaxPE, MPI_COMM_WORLD, ierr)

      call MPI_Reduce(chk_io, bw, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                  MaxPE, MPI_COMM_WORLD, ierr)
      chk_io = bw
      call MPI_Reduce(nocorner_io, bw, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                  MaxPE, MPI_COMM_WORLD, ierr)
      nocorner_io = bw
      call MPI_Reduce(corner_io, bw, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                  MaxPE, MPI_COMM_WORLD, ierr)
      corner_io = bw

      if (MyPE .EQ. MaxPE) then
           
            call nfmpi_stage_out_env(time_staging)

            call print_info(info_used)

            striping_factor = 0
            striping_unit   = 0
            call get_file_striping(info_used, striping_factor, striping_unit)

            io_amount = chk_io + nocorner_io + corner_io

1001 format(A,I13)
1002 format(A,I13,A)
1003 format(A,F16.2,A)
1004 format('-------------------------------------------------------')
1005 format(' NumPEs    array size      exec (sec)   bandwidth (MiB/s)')
1006 format(I5, 3x, i3,' x ',i3,' x ',i3, 3x, F7.2 , 2x,F10.2 /)
1007 format(A,A)
1008 format('#%$: ', A, ': ', F16.2)
1009 format('#%$: ', A, ': ', I13)
1010 format('#%$: ', A, ': ', A)
1011 format('#%$: ', A, ': ', F16.4)

            print 1009,' number_of_guards',nguard
            print 1009,' number_of_blocks',local_blocks
            print 1009,' number_of_variables',nvar
            print 1008,' checkpoint_time_max        ',time_io_max(1)
            print 1008,' checkpoint_time_header_max  ',chk_t_max(1)
            print 1008,' checkpoint_time_other_max  ',chk_t_max(2)
            print 1008,' checkpoint_time_close_max  ',chk_t_max(3)
            print 1008,' checkpoint_time_min        ',time_io_min(1)
            print 1008,' checkpoint_time_header_min  ',chk_t_min(1)
            print 1008,' checkpoint_time_other_min  ',chk_t_min(2)
            print 1008,' checkpoint_time_close_min  ',chk_t_min(3)
            print 1008,' checkpoint_time_mean        ',time_io_mean(1)
            print 1008,' checkpoint_time_header_mean  ',chk_t_mean(1)
            print 1008,' checkpoint_time_other_mean  ',chk_t_mean(2)
            print 1008,' checkpoint_time_close_mean  ',chk_t_mean(3)
            print 1011,' checkpoint_time_var        ',time_io_var(1)
            print 1011,' checkpoint_time_header_var  ',chk_t_var(1)
            print 1011,' checkpoint_time_other_var  ',chk_t_var(2)
            print 1011,' checkpoint_time_close_var  ',chk_t_var(3)
            print 1008,' checkpoint_io_size     ',chk_io / (1024 * 1024 * 1024)
            
            print 1008,' plot_no_corner_time_max        ',time_io_max(2)
            print 1008,' plot_no_corner_time_header_max  ',nocorner_t_max(1)
            print 1008,' plot_no_corner_time_other_max  ',nocorner_t_max(2)
            print 1008,' plot_no_corner_time_close_max  ',nocorner_t_max(3)
            print 1008,' plot_no_corner_time_min        ',time_io_min(2)
            print 1008,' plot_no_corner_time_header_min  ',nocorner_t_min(1)
            print 1008,' plot_no_corner_time_other_min  ',nocorner_t_min(2)
            print 1008,' plot_no_corner_time_close_min  ',nocorner_t_min(3)
            print 1008,' plot_no_corner_time_mean        ',time_io_mean(2)
            print 1008,' plot_no_corner_time_header_mean  ',nocorner_t_mean(1)
            print 1008,' plot_no_corner_time_other_mean  ',nocorner_t_mean(2)
            print 1008,' plot_no_corner_time_close_mean  ',nocorner_t_mean(3)
            print 1011,' plot_no_corner_time_var        ',time_io_var(2)
            print 1011,' plot_no_corner_time_header_var  ',nocorner_t_var(1)
            print 1011,' plot_no_corner_time_other_var  ',nocorner_t_var(2)
            print 1011,' plot_no_corner_time_close_var  ',nocorner_t_var(3)
            print 1008,' plot_no_corner_io_amount  ',nocorner_io / (1024 * 1024 * 1024)

            print 1008,' plot_corner_time_max        ',time_io_max(2)
            print 1008,' plot_corner_time_header_max  ',corner_t_max(1)
            print 1008,' plot_corner_time_other_max  ',corner_t_max(2)
            print 1008,' plot_corner_time_close_max  ',corner_t_max(3)
            print 1008,' plot_corner_time_min        ',time_io_min(2)
            print 1008,' plot_corner_time_header_min  ',corner_t_min(1)
            print 1008,' plot_corner_time_other_min  ',corner_t_min(2)
            print 1008,' plot_corner_time_close_min  ',corner_t_min(3)
            print 1008,' plot_corner_time_mean        ',time_io_mean(2)
            print 1008,' plot_corner_time_header_mean  ',corner_t_mean(1)
            print 1008,' plot_corner_time_other_mean  ',corner_t_mean(2)
            print 1008,' plot_corner_time_close_mean  ',corner_t_mean(3)
            print 1011,' plot_corner_time_var        ',time_io_var(2)
            print 1011,' plot_corner_time_header_var  ',corner_t_var(1)
            print 1011,' plot_corner_time_other_var  ',corner_t_var(2)
            print 1011,' plot_corner_time_close_var  ',corner_t_var(3)
            print 1008,' plot_corner_io_size ',corner_io / (1024 * 1024 * 1024)
            print 1004
            print 1010,' file_base_name        ', trim(basenm)
            if (striping_factor .GT. 0) then
                  print 1009,'   file_striping_count  ',striping_factor
                  print 1009,'   file_striping_size   ',striping_unit
            endif
            print 1008,' total_io_size       ', io_amount / (1024 * 1024 * 1024)
            print 1009,' number_of_processes', NumPEs
            print 1009,' dim_x', nxb
            print 1009,' dim_y', nyb
            print 1009,' dim_z', nzb

            print 1008,' flash_time_max        ', total_max
            print 1008,' flash_time_min        ', total_min
            print 1008,' flash_time_mean        ', total_mean
            print 1008,' flash_time_var        ', total_var

            print 1008,' total_time_max        ', total_max + time_staging
            print 1008,' total_time_min        ', total_min + time_staging
            print 1008,' total_time_mean        ', total_mean + time_staging
            print 1008,' total_time_var        ', total_var + time_staging

            print 1008,' stage_time       ', time_staging
            if (use_nonblocking_io) then
                print 1010,' nonblocking_io', '1'
            else
                print 1010,' nonblocking_io', '0'
            endif
            if (use_indep_io) then
                print 1010,' indep_io','1' 
            else
                print 1010,' indep_io', '0'
            endif

            print 1008,' bb_total_time_max       ', bb_time_max(1)
            print 1008,' bb_create_time_max       ', bb_time_max(2)
            print 1008,' bb_enddef_time_max       ', bb_time_max(3)
            print 1008,' bb_put_time_max       ', bb_time_max(4)
            print 1008,' bb_flush_time_max       ', bb_time_max(5)
            print 1008,' bb_close_time_max       ', bb_time_max(6)
            print 1008,' bb_put_data_wr_time_max       ', bb_time_max(7)
            print 1008,' bb_put_meta_wr_time_max       ', bb_time_max(8)
            print 1008,' bb_put_num_wr_time_max       ', bb_time_max(9)
            print 1008,' bb_flush_replay_time_max       ', bb_time_max(10)
            print 1008,' bb_flush_data_rd_time_max       ', bb_time_max(11)
            print 1008,' bb_flush_put_time_max       ', bb_time_max(12)
            print 1008,' bb_flush_wait_time_max       ', bb_time_max(13)

            print 1008,' bb_total_time_min       ', bb_time_min(1)
            print 1008,' bb_create_time_min       ', bb_time_min(2)
            print 1008,' bb_enddef_time_min       ', bb_time_min(3)
            print 1008,' bb_put_time_min       ', bb_time_min(4)
            print 1008,' bb_flush_time_min       ', bb_time_min(5)
            print 1008,' bb_close_time_min       ', bb_time_min(6)
            print 1008,' bb_put_data_wr_time_min       ', bb_time_min(7)
            print 1008,' bb_put_meta_wr_time_min       ', bb_time_min(8)
            print 1008,' bb_put_num_wr_time_min       ', bb_time_min(9)
            print 1008,' bb_flush_replay_time_min       ', bb_time_min(10)
            print 1008,' bb_flush_data_rd_time_min       ', bb_time_min(11)
            print 1008,' bb_flush_put_time_min       ', bb_time_min(12)
            print 1008,' bb_flush_wait_time_min       ', bb_time_min(13)

            print 1008,' bb_total_time_mean       ', bb_time_mean(1)
            print 1008,' bb_create_time_mean       ', bb_time_mean(2)
            print 1008,' bb_enddef_time_mean       ', bb_time_mean(3)
            print 1008,' bb_put_time_mean       ', bb_time_mean(4)
            print 1008,' bb_flush_time_mean       ', bb_time_mean(5)
            print 1008,' bb_close_time_mean       ', bb_time_mean(6)
            print 1008,' bb_put_data_wr_time_mean       ', bb_time_mean(7)
            print 1008,' bb_put_meta_wr_time_mean       ', bb_time_mean(8)
            print 1008,' bb_put_num_wr_time_mean       ', bb_time_mean(9)
            print 1008,' bb_flush_replay_time_mean       ', bb_time_mean(10)
            print 1008,' bb_flush_data_rd_time_mean       ', bb_time_mean(11)
            print 1008,' bb_flush_put_time_mean       ', bb_time_mean(12)
            print 1008,' bb_flush_wait_time_mean       ', bb_time_mean(13)

            print 1011,' bb_total_time_var       ', bb_time_var(1)
            print 1011,' bb_create_time_var       ', bb_time_var(2)
            print 1011,' bb_enddef_time_var       ', bb_time_var(3)
            print 1011,' bb_put_time_var       ', bb_time_var(4)
            print 1011,' bb_flush_time_var       ', bb_time_var(5)
            print 1011,' bb_close_time_var       ', bb_time_var(6)
            print 1011,' bb_put_data_wr_time_var       ', bb_time_var(7)
            print 1011,' bb_put_meta_wr_time_var       ', bb_time_var(8)
            print 1011,' bb_put_num_wr_time_var       ', bb_time_var(9)
            print 1011,' bb_flush_replay_time_var       ', bb_time_var(10)
            print 1011,' bb_flush_data_rd_time_var       ', bb_time_var(11)
            print 1011,' bb_flush_put_time_var       ', bb_time_var(12)
            print 1011,' bb_flush_wait_time_var       ', bb_time_var(13)

            print 1009,' bb_metadata_size       ', bb_meta_all
            print 1009,' bb_data_size       ', bb_data_all
            print 1009,' bb_flush_buffer_size       ', bb_buffer_all
      endif
      call MPI_Info_free(info_used, ierr)

      ! print info about PnetCDF internal malloc usage
      ierr = nfmpi_inq_malloc_max_size(malloc_size)
      if (ierr .EQ. NF_NOERR) then
          call MPI_Reduce(malloc_size, sum_size, 1, MPI_OFFSET, MPI_SUM, &
                          MaxPE, MPI_COMM_WORLD, ierr)
          if (verbose .AND. MyPE .EQ. MaxPE) &
              print 1002, &
              'maximum heap memory allocted by PnetCDF internally is', &
              sum_size/1048576, ' MiB'

          ierr = nfmpi_inq_malloc_size(malloc_size)
          call MPI_Reduce(malloc_size, sum_size, 1, MPI_OFFSET, MPI_SUM, &
                          MaxPE, MPI_COMM_WORLD, ierr)
          if (verbose .AND. MyPE .EQ. MaxPE .AND. &
              sum_size .GT. 0_MPI_OFFSET_KIND) &
              print 1002, &
              'heap memory allocated by PnetCDF internally has ', &
              sum_size/1048576, ' MiB yet to be freed'
      endif

      end subroutine report_io_performance




