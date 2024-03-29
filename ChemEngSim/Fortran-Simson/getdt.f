c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/getdt.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine getdt(dt,cflmax,rot,deta,xl,zl,prex,prez,pres,prea,
     &     ur,ui,u2r,u2i,bu1,bu2,pert,lin,wr,wi,
     &     realg1,realg2,realg3,realg4,wbci,my_comm_z,my_node_z,
     &     my_comm_x,my_node_x,my_node_world,
     &     boxr_z,boxi_z,wr_z,wi_z,wr_x,wi_x)
c
c     Computes CFL and dt for the first time step
c
      implicit none

      include 'par.f'
#ifdef MPI
      include 'mpif.h'
#endif

      real dt,cflmax,deta(nyp),xl,zl,rot
      real prex(nxp+15)
      real prez(nzp*2+15),pres(nzst*2+15),prea(nzat*3/2+15)
      real ur(memnx,memny,memnz,memnxyz),ui(memnx,memny,memnz,memnxyz)
      real u2r((nxp/2+1)*mby,nzd_new/nprocz,3)
      real u2i((nxp/2+1)*mby,nzd_new/nprocz,3)
      real wr(nxp/2+1,mby,nzd_new/nprocz,3)
      real wi(nxp/2+1,mby,nzd_new/nprocz,3)
      real cfl,cflp(memny)
      logical sym
      integer yb,i,myb,wbci
      real pi
      parameter (pi = 3.1415926535897932385)

      logical lin,pert
      real cflp1,cflp2   
      real bu1(nxp/2+1,nyp,3+scalar),bu2(nxp/2+1,nyp,3+scalar)
c
c     MPI
c
      integer my_node_world,realg1,realg2,realg3,realg4
      integer my_node_z,my_node_x,my_comm_z,my_comm_x
      integer yp,nypp
#ifdef MPI
      integer status1(mpi_status_size)
      real cfl1
      integer ierror,ip
      real boxr_z(memnx,nzd_new,3)
      real boxi_z(memnx,nzd_new,3)
      real wr_z(memnx,nzd_new)
      real wi_z(memnx,nzd_new)
      real wr_x(nxp/2+1,nzd_new/nprocz)
      real wi_x(nxp/2+1,nzd_new/nprocz)
#endif
c
c     Calculates CFL/dt from a timestep
c
      if (nproc.eq.1) then
         nypp = nyp
      else
         nypp = nyp/nprocz+1
      end if

      do yp=1,nypp
         yb =(yp-1)*nprocz+my_node_world/nprocx+1
         myb=(yp-1)/mby+1
c         yb = (yp-1)*nproc+my_node_world+1

c         myb=(yb-1)/mby+1
         do i=1,3
            if (nproc.eq.1) then
               call getxz(u2r(1,1,i),u2i(1,1,i),yb,i,1,ur,ui)
c
c     u,v symmetric, w antisymmetric if this is a 'symmetric' case
c
               sym=i.le.2
               if (yb.le.nyp) then
                  call fft2db(u2r(1,1,i),u2i(1,1,i),sym,
     &                 1,prex,prez,pres,prea,wr,wi)
               end if

            else
#ifdef MPI
               call getpxz_z(boxr_z(1,1,i),boxi_z(1,1,i),yb,i,1,ur,ui,
     &              realg1,realg2,my_node_z,my_comm_z,my_node_world)
      
c     
c     Backward Fourier transform in z
c     
               call vcfftb(boxr_z(1,1,i),boxi_z(1,1,i),wr_z,wi_z,
     &              nzp,memnx,memnx,1,prez)
       
               call getpxz_x(u2r(1,1,i),u2i(1,1,i),yb,i,1,
     &              boxr_z(1,1,i),boxi_z(1,1,i),
     &              realg3,realg4,my_node_x,my_comm_x,my_node_world)
c     
c     Backward Fourier transform in x
c     
               call vrfftb(u2r(1,1,i),u2i(1,1,i),wr_x,wi_x,
     &              nxp,nzd_new/nprocz,1,nxp/2+1,prex)
#endif
            end if
         end do
         if (yb.le.nyp) then

            if (pert)then
               call boxcflbf(cflp1,bu1,bu2,yb,deta,xl,zl)
            else
               cflp1=0.
            end if
            if (.not.lin) then 
               call boxcfl(cflp2,u2r,u2i,yb,deta,xl,zl,wbci)
            else
               cflp2=0.
            end if
            cflp(yp)=cflp1+cflp2
         else
            cflp(yp) = 0.
         end if
      end do
c      
c     Communicate CFL
c
      cfl=0.
      do i=1,nypp
         cfl=max(cfl,cflp(i))
      end do
#ifdef MPI
c      if (my_node_world.ne.0) then
c         call mpi_ssend(cfl,1,mpi_double_precision,
c     &        0,1,mpi_comm_world,ierror)
c      else
c         if (nproc.gt.1) then
c            do ip=1,nproc-1
c               call mpi_recv(cfl1,1,mpi_double_precision,
c     &              ip,1,mpi_comm_world,status1,ierror)
c               if (cfl.lt.cfl1) cfl=cfl1
c            end do
c         end if
c      end if
#endif      
c      cfl=cfl*pi+2.*abs(rot)
c      dt=cflmax/cfl
#ifdef MPI
c      if (nproc.gt.1) then
c         call mpi_bcast(cfl,1,mpi_double_precision,0,mpi_comm_world,
c     &        ierror)
c         call mpi_bcast(dt,1,mpi_double_precision,0,mpi_comm_world,
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

      cfl=cfl1*pi+2.*abs(rot)
      dt=cflmax/cfl

      end subroutine getdt
