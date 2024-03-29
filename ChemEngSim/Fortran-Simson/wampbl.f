c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/wampbl.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine wampbl(t,it,amp,campw,kx,kz,mwave,nwave,wint,re,
     &     xl,zl,longli,dstar,
     &     rlam,xbl,my_node_world,eta,fbla,dybla,nbla,spanv,fileurms,
     &     my_node_x,my_node_z,my_comm_x,fltype,px)
c
c     Accumulates and writes the amplitude file
c
c     Function
c     
c
c amp(y,1)  streamwise velocity average (squared)
c amp(y,2)  normal velocity average (squared)
c amp(y,3)  spanwise velocity average (squared)
c amp(y,4)  streamwise vorticity average (squared)
c amp(y,5)  normal vorticity average (squared)
c amp(y,6)  spanwise vorticity average (squared) 
c
c amp(y,7)  empty
c amp(y,8)  reynolds stress average
c
c amp(y,9)  mean streamwise velocity
c amp(y,10) mean spanwise velocity
c amp(y,11) mean streamwise vorticity component
c amp(y,12) mean spanwise vorticity component (to calculate wall shear)
c
c amp(y,13-20) empty 
c
c     
c     campw(y,i,wave) complex normal velocity and normal vorticity averages
c                     from selected wavenumbers
c     
c     Communicate amp
c     Note that the method is very inefficient since every amp(y,i) is
c     sent seperately. Could be improved by defining a strided 
c     MPI datatype.
c     The communication of campw is not yet implemented.
c
      implicit none

      include 'par.f'
#ifdef MPI
      include 'mpif.h'
#endif

      integer it,nwave,mwave,kx(nwave),kz(nwave)
      real t,wint(nyp),re,xl,zl,dstar,rlam,xbl
      real amp(nyp,20),spanv,tamp(nyp,20)
      complex campw(nyp,4,nwave)
      real ybl,etabl
      real eta(nyp),dybla
      real fbla(mbla,7+3*scalar)
      integer nbla
c
c     Accumulates the amplitudes from the values for each xz-box
c     and writes to logfile and ampfile (unit 15)
c     If longli = .true. then write out y-dependence of statistics
c     first accumulate
c
      logical longli,fileurms
      real sum(100),sumw1,sumw2,sumw3,sumw4,hplus,e0,c,um,omm,dum,wm
      real tau1,tau2
      real u0upp,u0low,w0low
      integer i,j,y

      real px
      integer fltype

c
c     MPI
c
      integer my_node,my_node_x,my_node_z,my_comm_x
#ifdef MPI
      integer ip,ierror,nypp,ybp,yb,my_node_world
      integer status1(mpi_status_size)
#endif
c
c     Functions
c
      real,external :: cubip

      real pi
      parameter (pi = 3.1415926535897932385)

      real tampp


      amp = amp / real(nxp*nzp)

c
c     first, reduce in spanwise direction
c
      tamp = 0.
      do i=1,20
         do y=1,nyp/nprocz+1
            call mpi_reduce(amp(y,i),tampp,1,
     &           mpi_double_precision,mpi_sum,0,
     &           my_comm_x,ierror)



            
            do ip=0,nprocz-1
               yb=ip+1+(y-1)*nprocz
               if (yb.le.nyp) then
                  
                  if (ip.ge.1) then
                     
                     if (my_node_world.eq.ip*nprocz) then
                        call mpi_ssend(tampp,1,
     &                       mpi_double_precision,0,
     &                       ip,
     &                       mpi_comm_world,ierror)
                        
                     end if
                     
                     
                     if (my_node_world.eq.0) then
                        call mpi_recv(tampp,1,
     &                       mpi_double_precision,
     &                       ip*nprocz,ip,
     &                       mpi_comm_world,status1,ierror)
                        
                        
                     end if
                     
                  else
c
c     keep what you have in txys, will later be overwritten
c     
                  end if
               end if
               
               call mpi_barrier(mpi_comm_world,ierror)
               if (my_node_world.eq.0) then
                  if (yb.le.nyp) then
                     tamp(yb,i) = tampp
                  end if
               end if
            end do
         end do
      end do

      if (my_node_world.gt.0) then
         call mpi_barrier(mpi_comm_world,ierror)
         return
      end if

      amp=tamp

      u0upp=amp(1,9)
      u0low=amp(nyp,9)
      w0low=amp(nyp,10)

      do j=1,20
         sum(j)=0.
      end do
c
c     This loop makes the amplitude files "compatible" with previous ones
c     by adding in the (0,0) component
c     In addition the turbulence production is calculated
c     this is for Blasius boundary layer
c
      c=-sqrt(re*(rlam+1.)/(2.*xbl))
      do y=1,nyp

         amp(y,1) = amp(y,1)-(amp(y,9))**2
         amp(y,2) = amp(y,2)
         amp(y,3) = amp(y,3)-(amp(y,10))**2

         amp(y,4) = amp(y,4)-amp(y,11)**2
         amp(y,5) = amp(y,5)
         amp(y,6) = amp(y,6)-(amp(y,12))**2

         amp(y,8) = -amp(y,8)
         amp(y,13)= amp(y,9)
         amp(y,9) = amp(y,9)**2
         amp(y,10) = amp(y,10)**2

         amp(y,15) = amp(y,15)-amp(y,14)**2
         amp(y,17) = amp(y,17)-amp(y,16)**2

      end do
      do i=1,10
         do y=1,nyp
            sum(i)=sum(i)+amp(y,i)*wint(y)
         end do
      end do
      do i=13,13
         do y=1,nyp
            sum(i)=sum(i)+amp(y,i)*wint(y)
         end do
      end do
      do i=14,17
         do y=1,nyp
            sum(i)=sum(i)+amp(y,i)*wint(y)
         end do
      end do



c
c     Then scale
c
      do j=1,6
         sum(j)=sqrt(max(sum(j),0.)*.5)
      end do
      do j=15,17,2
         sum(j)=sqrt(max(sum(j),0.)*.5)
      end do
      e0=sqrt((sum(9)+sum(10))*.5)
      tau1=amp(1,12)
      tau2=amp(nyp,12)
c
c     For channel and Couette flow
c
      if (fltype.eq.1.or.fltype.eq.2) then
         hplus=sqrt((abs(tau1)+abs(tau2))*re/2.)
      else
c
c     For boundary-layer cases
c
         hplus=sqrt(abs(tau2)*re)
      end if
c
c     Write screen output
c
c      write(*,*) 't',t/dstar,' it ',it
      write(*,'(A,3(g23.16,tr1))') ' velocity rms            ',
     &                       (sum(j),j=1,3)
      write(*,'(A,3(g23.16,tr1))') ' vorticity rms           ',
     &                       (sum(j)*dstar,j=4,6)
      write(*,'(A,4(g23.16,tr1))') ' omy**2/k2, dUuv, e0, h+ ',
     &                       sum(7),sum(8),e0,hplus

c
c     Note that the pressure gradient px might not be
c     correctly computed due to a mismatch of RK substeps etc.
c
      write(*,'(A,4(g23.16,tr1))') ' px, Reb, tau1/2 ',
     &                       px,sum(13)*re/2.,tau1,tau2
      if (scalar.gt.0) then
         write(*,'(A,3(g23.16,tr1))') ' thrms, dtdy1/2 ',
     &        sum(15),amp(1,16),amp(nyp,16)
      end if
c
c     Write file output
c
c      write(15,*) t/dstar,(sum(j),j=1,3)
c      write(15,*) (sum(j)*dstar,j=4,6),sum(7)
c      write(15,*) sum(8),e0,hplus
      if (scalar.eq.0) then
         write(15,'(15(g23.16,tr1))') t/dstar,(sum(j),j=1,3),
     &        (sum(j)*dstar,j=4,6),sum(7),sum(8),e0,hplus,
     &        px,sum(13)*re/2.,tau1,tau2
      else
         write(15,'(18(g23.16,tr1))') t/dstar,(sum(j),j=1,3),
     &        (sum(j)*dstar,j=4,6),sum(7),sum(8),e0,hplus,
     &        px,sum(13)*re/2.,tau1,tau2,sum(15),amp(1,16),amp(nyp,16)
      end if


      if (longli) then
         do i=1,10
            write(15,*) (amp(y,i),y=1,nyp)
         end do
      end if

      if (fileurms) then
         open(file='energy_urms',position='append',unit=78)
         write(78,'(15e15.7)')t/dstar,(sum(j),j=1,3)
         close(78)
      end if
c
c     Complex amplitudes are saved for selected waves
c
      if (mwave.gt.0) then
c
c     Not implemented for MPI
c
         call stopnow(4343657)

      end if

#ifdef MPI
      call mpi_barrier(mpi_comm_world,ierror)
#endif

      end subroutine wampbl
