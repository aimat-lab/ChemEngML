c ***********************************************************************
c
c $LastChangedDate: 2013-12-16 04:01:04 +0100 (Mo, 16. Dez 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 33 $
c
c ***********************************************************************
      subroutine oc(oc_y,oc_ampl,oc_xstart,oc_xend,oc_xblend,oc_vflux,
     &     oc_xstart_cor,oc_xend_cor,oc_tw,oc_tgrad,
     &     b2r,b2i,wr,wi,ur,ui,oc_amplx_tmp,
     &     xs,dstar,fltype,it,eta,unblow,
     &     prexn,prezn,u0low,w0low,alfa,zs,
     &     beta,my_node,realg1,realg2,oc_plane,oc_type,
     &     sc_wx,sc_wz,sc_wfunc,dxsc,dzsc)

      implicit none

#ifdef MPI
      include 'mpif.h'
#endif
      include 'par.f'
      real ur(memnx,memny,memnz,memnxyz),ui(memnx,memny,memnz,memnxyz)
      integer z,y,x,yb,i,xi,zi
      integer fltype,npl,it
      real dstar
      integer ivar,zp,ybp,nypp,myb
      real b2r(nxp/2+1,mby,nzd),b2i(nxp/2+1,mby,nzd)
      real wr(nxp/2+1,mby,nzd),wi(nxp/2+1,mby,nzd)
      real prexn(nx+15),prezn(nz*2+15)
      real alfa(nx/2*mbz),zs,xs,beta(nz),u0low,w0low
      real pi,step
      real eta(nyp)
      parameter (pi = 3.1415926535897932385)
c
c     Temporary storage arrays (only used on master)
c
      real uxz(nx,nz)

C     Oposition control parameters
      real oc_amplx(nx)
      real oc_amplx_tmp(nx)
      real oc_ampl,meanflux,oc_time
      integer oc_y,oc_xstart,oc_xend,oc_xblend
      integer oc_xstart_cor,oc_xend_cor,oc_tgrad
      logical oc_vflux,oc_tw,unblow
      integer k,j
      real oc_plane(nx,nz)      
      integer oc_type

C     Suboptimal control parameters
      real sc_wfunc(3*nx,3*nz),rms,part
      real*8, dimension(:,:),allocatable :: sc_wtmp
      integer sc_wx,sc_wz
      LOGICAL :: file_exists
      real*8, dimension(:,:),allocatable :: dwdy
      real*8, intent(inout) :: dxsc,dzsc
      real val

c
c     MPI
c
      integer my_node
      integer realg1,realg2
#ifdef MPI
      integer ierror
#endif

      if(it.eq.1)then
      
      if(oc_type.gt.2)then
        write(*,*)'Control type not implemented!'
        stop
      endif
      if(oc_tw)then
      do  k=1,nx
      oc_amplx_tmp(k)=1.0D0
      enddo

      else
      do  k=1,nx
      oc_amplx_tmp(k)=0.0D0
      enddo
C some test cases below      
C test sinus      
C      do  k=oc_xstart,oc_xend 
C        oc_amplx_tmp(k)=sin(2*pi*(k-oc_xstart)/(oc_xend-oc_xstart))
C      enddo
C sinus
C      do  k=oc_xstart,oc_xend 
C        if(k.lt.oc_xstart+oc_xblend)then
C       oc_amplx_tmp(k)=
C     & 0.5D0+0.5D0*(sin(pi*(k-oc_xstart)/(oc_xblend)-pi/2.D0))
C        else if(k.gt.oc_xend-oc_xblend)then
C       oc_amplx_tmp(k)=0.5D0-0.5D0*(sin(pi*(k-oc_xend+oc_xblend)/
C     & (oc_xblend)-pi/2.0D0))
C        else
C       oc_amplx_tmp(k)=1.0D0
C        endif
C      enddo
C tanh      
C      do  k=oc_xstart,oc_xend 
C        if(k.lt.oc_xstart+oc_xblend)then
C       oc_amplx_tmp(k)=
C     & 0.5D0+0.5D0*(tanh(pi*(k-oc_xstart)/(oc_xblend/2)-pi))
C        else if(k.gt.oc_xend-oc_xblend)then
C       oc_amplx_tmp(k)=0.5D0-0.5D0*(tanh(pi*(k-oc_xend+oc_xblend)/
C     & (oc_xblend/2)-pi))
C        else
C       oc_amplx_tmp(k)=1.0D0
C        endif
C      enddo
C
C using step function      

      do  k=oc_xstart,oc_xend
        if(k.lt.oc_xstart+oc_xblend)then
       oc_amplx_tmp(k)=
     & step(dble(k-oc_xstart)/(oc_xblend))
        else if(k.gt.oc_xend-oc_xblend)then
       oc_amplx_tmp(k)=
     & step(1.0-dble(k-oc_xend+oc_xblend)/(oc_xblend))
        else
       oc_amplx_tmp(k)=1.0D0
        endif
      enddo

C      if (my_node.eq.0) then
C      write(ios,*) 'Amplification profile oc_amplx_tmp computed'
C      do  k=1,nx
C        write(730,'(1024E15.7)')oc_amplx_tmp(k)
C      enddo
C      write(ios,*) 'Amplification profile oc_amplx_tmp computed'
C      do  k=1,nx
C        write(731,'(1024E15.7)')oc_amplx(k)
C      enddo
C      endif  

      endif

      if(oc_type.eq.2)then
        sc_wfunc=0.0D0
        sc_wtmp=0.0D0
      if (my_node.eq.0) then
        write(*,*)'1st iteration for suboptimal control'
      endif

      INQUIRE(FILE="weight.func", EXIST=file_exists)
      if(file_exists)then

      if (my_node.eq.0) then
      write(*,*)"Loading weight function..."
      endif
      OPEN(750,FILE='weight.func',STATUS='UNKNOWN')
      read(750,'(2I15)')sc_wx,sc_wz
      read(750,'(2E15.7)')dxsc,dzsc
      allocate(sc_wtmp(sc_wx,sc_wz))
      if (my_node.eq.0) then
      write(*,*)'WX=',sc_wx,', WZ=',sc_wz
      write(*,*)'DX=',dxsc,', DZ=',dzsc
      endif
      do x=1,sc_wx
        read(750,'(1024E15.7)')sc_wtmp(x,1:sc_wz)
      enddo
      CLOSE(750)
      if (my_node.eq.0) then
      write(*,*)"Done."
      endif

      else

      write(*,*)"Generating weight function..."

C Function from Lee et al 1997      
      sc_wx=3
      sc_wz=7
      dxsc=40.0D0
      dzsc=13.0D0
      allocate(sc_wtmp(sc_wx,sc_wz))

      sc_wtmp(1,1)=0.0D0
      sc_wtmp(1,2)=1.0D0
      sc_wtmp(1,3)=-0.1039D0
      sc_wtmp(1,4)=0.2679D0
      sc_wtmp(1,5)=-0.0852D0
      sc_wtmp(1,6)=0.1419D0
      sc_wtmp(1,7)=-0.0671
      sc_wtmp(2,1)=0.0086D0
      sc_wtmp(2,2)=0.0537D0
      sc_wtmp(2,3)=0.0503D0
      sc_wtmp(2,4)=0.0310D0
      sc_wtmp(2,5)=0.0340D0
      sc_wtmp(2,6)=0.0148D0
      sc_wtmp(2,7)=0.0237D0
      sc_wtmp(3,1)=0.0001D0
      sc_wtmp(3,2)=-0.0104D0
      sc_wtmp(3,3)=0.0059D0
      sc_wtmp(3,4)=0.0051D0
      sc_wtmp(3,5)=0.0100D0
      sc_wtmp(3,6)=0.0074D0
      sc_wtmp(3,7)=0.0092D0

      if (my_node.eq.0) then
      write(*,*)"Writing weight function..."
      OPEN(750,FILE='weight.func',STATUS='UNKNOWN')
      write(750,'(2I15)')sc_wx,sc_wz
      write(750,'(2E15.7)')dxsc,dzsc
      do x=1,sc_wx
        write(750,'(1024E15.7)')sc_wtmp(x,1:sc_wz)
      enddo
      CLOSE(750)
      write(*,*)"Done."
      endif

      endif

C creating temporal weight function
      do x=1,sc_wx
        do z=1,sc_wz
          sc_wfunc(nx+x,nz+z)=sc_wtmp(x,z)
        enddo
      enddo
      do x=1,nx-1
        do z=1,nz
          sc_wfunc(nx-x+1,nz+z)=sc_wfunc(nx+x+1,nz+z)
        enddo
      enddo
      do x=1,2*nx
        do z=1,nz-1
          sc_wfunc(x,nz-z+1)=-sc_wfunc(x,nz+z+1)
        enddo
      enddo
      deallocate(sc_wtmp)

C             if (my_node.eq.0) then
C      do x=1,3*nx
C        write(700,'(1024E15.7)')sc_wfunc(x,1:3*nz)
C      enddo
C             endif
      endif

      endif

      if(it.le.oc_tgrad)then
      oc_time=dble(it)/dble(oc_tgrad)
      do  k=1,nx/2
       oc_amplx(k)=oc_time*oc_amplx_tmp(k+nx/2)
       oc_amplx(k+nx/2)=oc_time*oc_amplx_tmp(k)
      enddo
      else
      do  k=1,nx/2
       oc_amplx(k)=oc_amplx_tmp(k+nx/2)
       oc_amplx(k+nx/2)=oc_amplx_tmp(k)
      enddo
      endif
 

      if(unblow)then

      do  j=1,nz
       do  k=1,nx
           oc_plane(k,j)=oc_amplx(k)
       enddo
      enddo


      else

      if(oc_type.eq.1)then
      write(*,*)'Opposition control active'
C        wall-normal velocity
         ivar=2

c
c     xz plane
c
            ybp=nyp-oc_y
            yb=(ybp-1)/mby*mby+1+my_node
    

c
c     get xz plane
c
            if (nproc.eq.1) then
               call getxz(b2r,b2i,yb,ivar,0,ur,ui)
            else
#ifdef MPI
               call getpxz(b2r,b2i,yb,ivar,0,ur,ui,
     &              realg1,realg2,my_node)
#endif
            end if
    
             if (my_node.eq.0) then
c
c     Shift and remove moving wall
c
               if (fltype.ne.2.and.fltype.ne.5) then
                  call xzsh(b2r,b2i,xs,zs,alfa,beta,yb)
                  if (ivar.eq.1) then
                     b2r(1,1,1)=b2r(1,1,1)-u0low
                  end if
                  if (ivar.eq.3) then
                     b2r(1,1,1)=b2r(1,1,1)-w0low
                  end if
               end if
c
c     Transform to physical space
c
               call vcfftb(b2r(1,1,1),b2i(1,1,1),wr,wi,nz,
     &              nx/2,(nxp/2+1)*mby,1,prezn)
               call vrfftb(b2r(1,1,1),b2i(1,1,1),wr,wi,
     &              nx,nzpc,1,mby*(nxp/2+1),prexn)
c
c     Copy relevant section
c
               do z=1,nz
                  do x=1,nx/2
                     uxz(2*x-1,z)=b2r(x,ybp-yb+1,z)
                     uxz(2*x  ,z)=b2i(x,ybp-yb+1,z)
                  end do
               end do
      do  j=1,nz
       do  k=1,nx
          oc_plane(k,j)=-oc_amplx(k)*oc_ampl*uxz(k,j)
       enddo
      enddo

      endif

      end if

      if(oc_type.eq.2)then
C      write(*,*)'Suboptimal control active'
      allocate(dwdy(3*nx,3*nz))
      dwdy=0.0D0

C get velocity data      
C        spanwise velocity
         ivar=3
c
c     xz plane for y=1
c
            ybp=nyp-1
            yb=(ybp-1)/mby*mby+1+my_node
c
c     get xz plane
c
            if (nproc.eq.1) then
               call getxz(b2r,b2i,yb,ivar,0,ur,ui)
            else
#ifdef MPI
               call getpxz(b2r,b2i,yb,ivar,0,ur,ui,
     &              realg1,realg2,my_node)
#endif
            end if
    
             if (my_node.eq.0) then
c
c     Shift and remove moving wall
c
               if (fltype.ne.2.and.fltype.ne.5) then
                  call xzsh(b2r,b2i,xs,zs,alfa,beta,yb)
                  if (ivar.eq.1) then
                     b2r(1,1,1)=b2r(1,1,1)-u0low
                  end if
                  if (ivar.eq.3) then
                     b2r(1,1,1)=b2r(1,1,1)-w0low
                  end if
               end if
c
c     Transform to physical space
c
               call vcfftb(b2r(1,1,1),b2i(1,1,1),wr,wi,nz,
     &              nx/2,(nxp/2+1)*mby,1,prezn)
               call vrfftb(b2r(1,1,1),b2i(1,1,1),wr,wi,
     &              nx,nzpc,1,mby*(nxp/2+1),prexn)
c
c     Copy relevant section
c
               do z=1,nz
                  do x=1,nx/2
                     uxz(2*x-1,z)=b2r(x,ybp-yb+1,z)
                     uxz(2*x  ,z)=b2i(x,ybp-yb+1,z)
                  end do
               end do


      do x=1,nx
        do z=1,nz
          dwdy(nx+x,nz+z)=uxz(x,z)
        enddo
      enddo

C      if (my_node.eq.0) then
C      do i=1,3*nx
C        write(701,'(1024E15.7)')dwdy(i,1:3*nz)
C      enddo
C      endif

      do x=1,nx
        do z=1,nz
          dwdy(2*nx+x,nz+z)=dwdy(nx+x,nz+z)
        enddo
      enddo
      do x=1,nx
        do z=1,nz
          dwdy(x,nz+z)=dwdy(nx+x,nz+z)
        enddo
      enddo
      do x=1,3*nx
        do z=1,nz
          dwdy(x,2*nz+z)=dwdy(x,nz+z)
        enddo
      enddo
      do x=1,3*nx
        do z=1,nz
          dwdy(x,z)=dwdy(x,nz+z)
        enddo
      enddo

C      if (my_node.eq.0) then
C      do i=1,3*nx
C        write(702,'(1024E15.7)')dwdy(i,1:3*nz)
C      enddo
C      endif

      do xi=1,nx
        do zi=1,nz
          do x=nx-sc_wx+2,nx+sc_wx
            do z=nz-sc_wz+2,nz+sc_wz

C take into account dx dz:
C      call findshift(dwdy,23.9246D0,5.9016D0,dxsc,dzsc,
C     &      xi,zi,x-nx-1,z-nz-1,nx,nz,val)

C         oc_plane(xi,zi)=oc_plane(xi,zi)+
C     &                   sc_wfunc(x,z)*val

         oc_plane(xi,zi)=oc_plane(xi,zi)+
     &                   sc_wfunc(x,z)*dwdy(xi+x-1,zi+z-1)
            enddo
          enddo
        enddo
      enddo

      if (my_node.eq.0) then
      do i=1,nx
        write(703,'(1024E15.7)')oc_plane(i,1:nz)
      enddo
      endif

      do x=1,nx
       do z=1,nz        
         oc_plane(x,z)=oc_amplx(x)*oc_plane(x,z)
       enddo
      enddo

      if (my_node.eq.0) then
      do i=1,nx
        write(704,'(1024E15.7)')oc_plane(i,1:nz)
      enddo
      endif
C      rms=0.0D0
C      do x=1,nx
C        do z=1,nz
C            rms=rms+oc_plane(x,z)**2
C        enddo
C      enddo
C      rms=rms/(oc_xend-oc_xstart-1)/nz
C      rms=sqrt(rms)
C      write(*,*)'RMS:',rms

      if(oc_xstart.le.nx/2)then
        oc_xstart_cor=nx/2+oc_xstart
      else
        oc_xstart_cor=oc_xstart-nx/2
      endif
      if(oc_xend.le.nx/2)then
        oc_xend_cor=nx/2+oc_xend
      else
        oc_xend_cor=oc_xend-nx/2
      endif
   
         if(oc_xend_cor.ge.oc_xstart_cor)then
         rms=0.0D0
         do k=oc_xstart_cor,oc_xend_cor
          do j=1,nz
          rms=rms+oc_plane(k,j)**2
          enddo
         enddo
         rms=rms/nz/(oc_xend_cor-oc_xstart_cor+1)
  
C         do k=oc_xstart_cor,oc_xend_cor
C         do  j=1,nz
C          oc_plane(k,j)=oc_plane(k,j)-meanflux
C         enddo
C         enddo
 
         else

         rms=0.0D0
         do j=1,nz
          do k=1,oc_xend_cor
          rms=rms+oc_plane(k,j)**2
          enddo
          do k=oc_xstart_cor,nx
          rms=rms+oc_plane(k,j)**2
          enddo
         enddo
         rms=rms/nz/(oc_xend_cor+(nx-oc_xstart_cor)+1)
  
C         do  j=1,nz
C          do k=1,oc_xend_cor
C          oc_plane(k,j)=oc_plane(k,j)-meanflux
C          enddo
C          do k=oc_xstart_cor,nx
C          oc_plane(k,j)=oc_plane(k,j)-meanflux
C          enddo
C         enddo
       endif
       rms=sqrt(rms)
C      write(*,*)'RMS:',rms

      do x=1,nx
       do z=1,nz        
         oc_plane(x,z)=(oc_ampl/rms)*oc_plane(x,z)
       enddo
      enddo

C      if (my_node.eq.0) then
C      do i=1,nx
C        write(705,'(1024E15.7)')oc_plane(i,1:nz)
C      enddo
C      endif


      endif

      deallocate(dwdy)
      endif

      if(oc_vflux)then
 
      if(oc_xstart.le.nx/2)then
        oc_xstart_cor=nx/2+oc_xstart
      else
        oc_xstart_cor=oc_xstart-nx/2
      endif
      if(oc_xend.le.nx/2)then
        oc_xend_cor=nx/2+oc_xend
      else
        oc_xend_cor=oc_xend-nx/2
      endif
   
         if(oc_xend_cor.ge.oc_xstart_cor)then
C         write(ios,*) 'Case 1 '
 
         meanflux=0.0D0
         do k=oc_xstart_cor,oc_xend_cor
          do j=1,nz
          meanflux=meanflux+oc_plane(k,j)
          enddo
         enddo
         meanflux=meanflux/nz/(oc_xend_cor-oc_xstart_cor+1)

C          write(ios,*) 'Current v-mean-flux: ',meanflux
C          write(ios,*) 'Limits: ',oc_xstart_cor,oc_xend_cor
  
         do k=oc_xstart_cor,oc_xend_cor
         do  j=1,nz
          oc_plane(k,j)=oc_plane(k,j)-meanflux
         enddo
         enddo
 
         else
C         write(ios,*) 'Case 2 '

         meanflux=0.0D0
         do j=1,nz
          do k=1,oc_xend_cor
          meanflux=meanflux+oc_plane(k,j)
          enddo
          do k=oc_xstart_cor,nx
          meanflux=meanflux+oc_plane(k,j)
          enddo
         enddo
         meanflux=meanflux/nz/(oc_xend_cor+(nx-oc_xstart_cor)+1)

C          write(ios,*) 'Devided by: ',(oc_xend_cor+(nx-oc_xstart_cor)+1)
C          write(ios,*) 'Current v-mean-flux: ',meanflux
C          write(ios,*) 'Limits: ',oc_xstart_cor,oc_xend_cor
  
         do  j=1,nz
          do k=1,oc_xend_cor
          oc_plane(k,j)=oc_plane(k,j)-meanflux
          enddo
          do k=oc_xstart_cor,nx
          oc_plane(k,j)=oc_plane(k,j)-meanflux
          enddo
         enddo
 
         endif

C
C uncomment for checking mean flux after correction         
C
C         meanflux=0.0D0
C         do k=1,nx
C          do j=1,nz
C          meanflux=meanflux+oc_plane(k,j)
C          enddo
C         enddo
C         meanflux=meanflux/nz/nx
 
C         write(ios,*) 'v-mean-flux after correction: ',meanflux

         end if

      end if
 
#ifdef MPI
             call mpi_bcast(oc_plane,nx*nz,mpi_double_precision,
     &                      0,mpi_comm_world,ierror) 
#endif

      end subroutine oc

      subroutine findshift(dwdy,dxorig,dzorig,dx,dz,
     &                     xorig,zorig,x,z,xn,zn,val)
      !
      implicit none

      integer :: x,z,xn,zn,xorig,zorig
      real*8 :: xtmp,ztmp,xpart,zpart
      integer :: xntmp,zntmp
      real*8, dimension(3*xn,3*zn,2) :: dwdy
      real*8 :: dx,dz,dxorig,dzorig
      real*8 :: val

      xtmp=dx*x/dxorig
      xntmp=int(xtmp)
      xpart=xtmp-xntmp
      ztmp=dz*z/dzorig
      zntmp=int(ztmp)
      zpart=ztmp-zntmp

C      write(*,*)'xtmp:',xtmp
C      write(*,*)'xntmp:',xntmp
C      write(*,*)'xpart:',xpart
C      write(*,*)'xtmp:',ztmp
C      write(*,*)'xtmp:',zntmp
C      write(*,*)'xtmp:',zpart

      val=(1.0D0-xpart)*(1.0D0-zpart)*
     & dwdy(xn+xorig+xntmp,zn+zorig+zntmp,1)+
     &       (xpart)*(1.0D0-zpart)*
     & dwdy(xn+xorig+xntmp+1,zn+zorig+zntmp,1)+
     &       (zpart)*(1.0D0-xpart)*
     & dwdy(xn+xorig+xntmp,zn+zorig+zntmp+1,1)+
     &       (xpart)*(zpart)*
     & dwdy(xn+xorig+xntmp+1,zn+zorig+zntmp+1,1)

C      write(*,*)'xn,zn:',xn,zn
C      write(*,*)'f(0,0):',xn+xorig+xntmp,zn+zorig+zntmp
C      write(*,*)'f(1,0):',xn+xorig+xntmp+1,zn+zorig+zntmp
C      write(*,*)'f(0,1):',xn+xorig+xntmp,zn+zorig+zntmp+1
C      write(*,*)'f(1,1):',xn+xorig+xntmp+1,zn+zorig+zntmp+1
C      write(*,*)'f(0,0):',dwdy(xn+xorig+xntmp,zn+zorig+zntmp,1)
C      write(*,*)'f(1,0):',dwdy(xn+xorig+xntmp+1,zn+zorig+zntmp,1)
C      write(*,*)'f(0,1):',dwdy(xn+xorig+xntmp,zn+zorig+zntmp+1,1)
C      write(*,*)'f(1,1):',dwdy(xn+xorig+xntmp+1,zn+zorig+zntmp+1,1)
C      write(*,*)'val:',val(1)


      end subroutine findshift
