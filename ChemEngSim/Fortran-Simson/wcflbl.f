c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/wcflbl.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine wcflbl(dt,cfl,cflp,rot,dstar,my_node_world)
c     
c     Accumulates the cfl and writes to logfile
c
      implicit none

      include 'par.f'
#ifdef MPI
      include 'mpif.h'
#endif
      real dt,cfl,cflp(memny),rot,dstar
      integer i,ym
      real pi
      parameter (pi = 3.1415926535897932385)
c
c     MPI
c
      integer my_node_world
#ifdef MPI
      real cfl1
      integer ym1,ip,ierror
      integer status1(mpi_status_size)
#endif

c
c     Find the local max and y pos
c
      cfl=cflp(1)
      ym=1
      do i=2,nby
         if (cflp(i).gt.cfl) ym=i
         cfl=max(cflp(i),cfl)
      end do
c
c     Communicate cfl and fine the max
c
#ifdef MPI
c      if (my_node_world.ne.0) then
c         call mpi_ssend(cfl,1,mpi_double_precision,
c     &        0,1,mpi_comm_world,ierror)
c         call mpi_ssend(ym,1,mpi_integer4,
c     &        0,2,mpi_comm_world,ierror)
c      else
c         if (nproc.gt.1) then
c            do ip=1,nproc-1
c               call mpi_recv(cfl1,1,mpi_double_precision,
c     &              ip,1,mpi_comm_world,status1,ierror)
c               call mpi_recv(ym1,1,mpi_integer4,
c     &              ip,2,mpi_comm_world,status1,ierror)
c               if (cfl.lt.cfl1) then
c                  cfl=cfl1
c                  ym = ym1
c               end if
c            end do
c         end if
c      end if
#endif

c      if (my_node_world.eq.0) then
c         cfl=(cfl*pi+2.*abs(rot))*dt
c         write(*,'(a,f22.16,a,f22.16,a,i4)') 
c     &        'CFL*dt ',cfl,' dt ',dt/dstar,' in box',ym
c      end if
      
#ifdef MPI
c      if (nproc.gt.1) then
c         call mpi_bcast(cfl,1,mpi_double_precision,0,mpi_comm_world,
c     &        ierror)
c      end if
#endif

c
c     New implementation using global communication
c
#ifdef MPI
      call mpi_allreduce(cfl,cfl1,1,mpi_double_precision,
     &     mpi_max,mpi_comm_world,ierror)
#endif
      
      cfl=(cfl1*pi+2.*abs(rot))*dt
      if (my_node_world.eq.0) then
         write(*,'(a,f22.16,a,f22.16)') 
     &        'CFL*dt ',cfl,' dt ',dt/dstar
      end if

      if (cfl.gt.10.) then
         if (my_node_world.eq.0) then
            write(*,*) 'CFL*dt too high. Stop'
         end if
         call stopnow(324234)
      end if

      end subroutine wcflbl
