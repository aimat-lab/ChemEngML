c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/stopnow.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine stopnow(i)

      implicit none

#ifdef MPI
      include 'mpif.h'
#endif

      integer i

#ifdef MPI
      integer my_node,ierror
#endif

#ifdef MPI
      call mpi_comm_rank(mpi_comm_world,my_node,ierror)
      write(*,*) '*** STOP *** at location (node ',my_node,'):',i
      call mpi_barrier(mpi_comm_world,ierror)
      call mpi_finalize(ierror)
#else
      write(*,*) '*** STOP *** at location:',i
#endif      

      stop

      end subroutine stopnow
