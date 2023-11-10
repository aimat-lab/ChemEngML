c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/locf.f $
c $LastChangedDate: 2016-02-08 13:53:41 +0100 (Mon, 08. Feb 2016) $
c $LastChangedBy: alex $
c $LastChangedRevision: 94 $
c
c ***********************************************************************
      subroutine locf(om2r,om2i,yb,xl,zl,xsc,zsc,eta,tc,
     &     loctyp,fp1,fpds1,fpds2,fpds3,fpds4,fpds5,fpds6,fpds7,fpds8,
     &     fpdds1,fpdds2,fpdds3,fpdds4,fpdds5,g1,g2,th2r,th2i,
     &     my_node_world,rgh_h,rgh_ampl,rgh_maxyn,u2r,u2i,rgh_hy,
     &     fd,fd1,Intu2r,Intu2i,Ints2r,Ints2i,an0,bn0)

c      
c     Localized forcing
c
c==== loctyp=1: 
c
c     Adding a volume force of the form
c     F=(ampx,ampy,ampz)*exp(-(y/yscale)**2)*g(x,z)*f(t)
c
c     zscale>0   g(x,z)=exp(-(x-xloc0)/xscale**2-(z/zscale)**2)
c     zscale<0   g(x,z)=exp(-(x-xloc0)/xscale**2)*cos((z-x*lskew)/zscale*2*pi)
c     
c     tscale>0 f(t) is a smooth turn on   : f(t)=exp(-(t/tscale)**2)
c     tscale<0 f(t) is a smooth turn off  : f(t)=step(-t/tscale))
c     tscale=0 f(t)=1.
c 
c     where step is defined in step.f
c 
c     the volume force is only calculated if locfor is true
c     and the time is in the interval [0-5 tscale] or tscale<0
c
c
c==== loctyp=2:
c
c     Adding a volume force of the form
c     F=(ampx,ampy,ampz).*(g(1,z't),g(2,z't),g(3,z't))**fy(y),fx(x)
c     
c     g(1,z't)=cos(zbet*z)*cos(tomeg*t)/(2*tomeg)
c     g(2,z't)=cos(zbet*z)*sin(tomeg*t)
c     g(3,z't)=-sin(zbet*z)*sin(tomeg*t)/(2*zbet)
c     
c     fx(x)=step((x-xstart)/xrise)-step((x-xend)/xfall+1)
c     
c     fy(y)=step((y-ystart)/yrise)-step((y-yend)/yfall+1)
c     
c     where step is defined in step.f
c     
c==== loctyp=3:
c     
c     Adding a volume force of the form
c     F=(ampx,ampy,ampz)*exp(-(y/yscale)**2)*fx(x)*f(t)
c     
c     xtype=0  f(x)=                 exp(-((x-xloc0)/xscale)**2)
c     xtype=1  f(x)=(x-xloc0)/xscale*exp(-((x-xloc0)/xscale)**2)
c     
c     f(t)=sin(tomeg*t)
c
c==== loctyp=4:
c     
c     Adding a volume force of the form
c     F=(ampx,ampy,ampz)*exp(-y/yscale)*ft(t)
c     
c     f(t)=(step((t-tstart)/tscale))-step((t-tend)/tscale+1))*cos(tomeg*t)
c     
c==== loctyp=5: 
c     
c     Related to loctyp=1      
c     
c     adding a volume force of the form
c     F=(ampx,ampy,ampz)*exp(-(y/yscale)**2)*g(x,z)*f(t)
c     
c     
c     zscale>0   g(x,z)=exp(-(x-xloc0)/xscale**2-(z/zscale)**2)
c     zscale<0   g(x,z)=cos((z-x*lskew)/zscale*2*pi)*exp(-(x-xloc0)/xscale**2)
c     
c     tscale>0 f(t) is a smooth turn on   : f(t)=exp(-(t/tscale)**2)
c     tscale<0 f(t) is a smooth turn off  : f(t)=step(-t/tscale))
c     tscale=0 f(t)=1.
c     
c     h1(t)=aomeg*sin(tomeg*tc) (note: aomeg is relative amplitude
c     between oscillation and stationary force)
c     
c     where step is defined in step.f
c     
c     the volume force is only calculated if locfor is true
c     and the time is in the interval [0-5 tscale] or tscale<0
c     
c==== loctyp=6:
c     
c     Adding a volume force of the form
c     F=(ampx,ampy,ampz)*exp(-(y/yscale)**2)*fx(x)*f(t)
c     useful for TS waves and corresponding secondary instability (K/H-type)
c     
c     f(x)=exp(-((x-xloc0)/xscale)**2)
c
c     g(z) = cos(2pi/zl)
c     
c     f2d(t)   = sin(tomeg*t)
c     f3d(t) = sin(tomeg3D*t)
c
c     F=(0,1,0)*exp(-(yc/yscale)**2)*(amp2d*f2d(t)*f(x) + 
c                                     amp3d*f3d(t)*f(x)*g(z) )
c
c==== loctyp=7:
c
c     Adding a localised forcing of the temperature (line source) 
c     
c==== loctyp=8:
c     
c     Approximated an impulse input
c
c     Adding a volume force of the form 
c     F=(ampx,ampy,ampz)*exp(-((y-yloc)/yscale)**2)*fx(x)*f(t)
c     
c     xtype=0  f(x)=                 exp(-((x-xloc0)/xscale)**2)
c     
c     f(t)=exp(-((t-tstart)/tscale)**2)
c

      implicit none

      include 'par.f'

      integer yb,loctyp,ith
      real om2r(nxp/2+1,mby,nzd_new/nprocz,3)
      real om2i(nxp/2+1,mby,nzd_new/nprocz,3)
      real th2r(nxp/2+1,mby,nzd_new/nprocz,4*scalar)
      real th2i(nxp/2+1,mby,nzd_new/nprocz,4*scalar)
      real xl,zl,xsc,zsc,tc
      real eta(nyp)
      real g1(nxp/2,nzd),g2(nxp/2,nzd)
      real fp1,fpds1,fpds2,fpds3,fpds4,fpds5,fpds6,fpds7
      real fpds8,fpdds1,fpdds2,fpdds3,fpdds4,fpdds5

      real ampx,ampy,ampz,xscale,yscale,zscale,tscale,xloc0,lskew
      real xtype,xstart,xend,xrise,xfall,ystart,yend,yrise,yfall
      real zbet,tomeg,tstart,tend,xo,aomeg,y0,yscale1
      real amp,yloc0
      real coeff,coeff1

      real xc1(nxp/2),xc2(nxp/2)
      integer x,y,z      
      real xc11,xc22,fx1(nxp/2),fx2(nxp/2),f2x1(nxp/2),f2x2(nxp/2)
      real fy,dfy,f2y,ft,f2t,yc,zc,k2
      real pi     
      real amp2d,amp3d,tomeg3d,ft3d
      integer my_node_world,zp,zb
      parameter (pi = 3.1415926535897932385)

C     Roughness with direct forcing (loctyp==9,10)
      real u2r((nxp/2+1),mby,nzd_new/nprocz,3)
      real u2i((nxp/2+1),mby,nzd_new/nprocz,3)
      real an0,bn0
      real Intu2r((nxp/2+1),nyp+1,nzd_new/nprocz,3)
      real Intu2i((nxp/2+1),nyp+1,nzd_new/nprocz,3)
      real Ints2r((nxp/2+1),nyp+1,nzd_new/nprocz,scalar)
      real Ints2i((nxp/2+1),nyp+1,nzd_new/nprocz,scalar)
      real rgh_h(nxp,nzp)
      real rgh_hy(nxp,nyp)
      real rgh_ampl,rgh_ampl2
      integer rgh_maxyn,k
      integer rgh_zshift
c     Prescribed body force (loctyp==11)
      real fd(nyp),fd1(nyp)

c     Functions
c
      real,external :: step,dstep
CC      rgh_ampl2=300 
CC      rgh_ampl=250
CC      rgh_zshift=96; 
CC      rgh_zshift=288; 
CC      rgh_zshift=72 
CC      rgh_zshift=0
CC      rgh_zshift=324
CC      rgh_zshift=144; 
C      if (loctyp.ne.3) then
C         write(*,*) 'not yet implemented'
C         call stopnow(342432)
C      end if


c
c Note that we need a coordinate that runs between -xl/2 and xl/2
c regardless of the shift xsc, otherwise the force would be turned off
c abruptly when shifted out of the box
c

C roughness with IBM based on x-z-distribution
      if (loctyp.eq.9) then

C         write(*,*) 'loctyp = ',loctyp,' running.'
C         write(*,*) 'an = ',an0,' bn = ',bn0
C         write(*,*) 'rgh_ampl = ',rgh_ampl,' rgh_ampl2 = ',rgh_ampl2
C (lower wall)
      if (yb.ge.(nyp-1-rgh_maxyn)) then

      zb=mod(my_node_world,nprocx)*nzd_new/nprocz
      do y=1,min(mby,nyp-yb+1)
C compute wall-normal coordinate
        yc = 1.+eta(y+yb-1)
C        write(*,*)yb,yc
        do zp=1,(nzd_new/nprocz)
         z=zp+zb
         do x=1,nxp/2

           if(yc.le.rgh_h(2*x-1,z)) then
            do k=1,3
       Intu2r(x,y+yb,zp,k)=Intu2r(x,y+yb,zp,k)+
     & (an0+bn0)*u2r(x,y,zp,k)
c       if (abs(Intu2r(x,y+yb,zp,k))>0.0001) then
c       write(*,*)my_node_world,x,z,zp,y+yb,k,Intu2r(x,y+yb,zp,k)
c       endif
       om2r(x,y,zp,k)=om2r(x,y,zp,k)-(u2r(x,y,zp,k))
     & *rgh_ampl/2.0D0-Intu2r(x,y+yb,zp,k)*rgh_ampl2
            end do
            do ith=1,scalar
       Ints2r(x,y+yb,zp,ith)=Ints2r(x,y+yb,zp,ith)+
     & (an0+bn0)*th2r(x,y,zp,1+4*(ith-1))
       th2r(x,y,zp,4+4*(ith-1))=th2r(x,y,zp,4+4*(ith-1))
     & -(th2r(x,y,zp,1+4*(ith-1)))*rgh_ampl/2.0D0
     & -(Ints2r(x,y+yb,zp,ith))*rgh_ampl2
            end do
          endif
           if(yc.le.rgh_h(2*x,z)) then
            do k=1,3
       Intu2i(x,y+yb,zp,k)= Intu2i(x,y+yb,zp,k)+
     & (an0+bn0)*u2i(x,y,zp,k)
c       if (abs(Intu2i(x,y+yb,zp,k))>0.0001) then
c       write(*,*)my_node_world,x,z,zp,y+yb,Intu2i(x,y+yb,zp,k)
c       endif
       om2i(x,y,zp,k)=om2i(x,y,zp,k)-(u2i(x,y,zp,k))
     & *rgh_ampl/2.0D0-Intu2i(x,y+yb,zp,k)*rgh_ampl2
            end do
            do ith=1,scalar
       Ints2i(x,y+yb,zp,ith)=Ints2i(x,y+yb,zp,ith)+
     & (an0+bn0)*th2i(x,y,zp,1+4*(ith-1))
       th2i(x,y,zp,4+4*(ith-1))=th2i(x,y,zp,4+4*(ith-1))
     & -(th2i(x,y,zp,1+4*(ith-1)))*rgh_ampl/2.0D0
     & -(Ints2i(x,y+yb,zp,ith))*rgh_ampl2
            end do

          endif

         end do
       end do
      end do

      end if
C (upper wall)
      if (yb.le.(rgh_maxyn)) then

      zb=mod(my_node_world,nprocx)*nzd_new/nprocz
      do y=1,min(mby,nyp-yb+1)
C compute wall-normal coordinate
        yc = 1.+eta(y+yb-1)
C        write(*,*)yb,yc
        do zp=1,(nzd_new/nprocz)
         z=zp+zb
         do x=1,nxp/2
           if(x.ge.(rgh_zshift/2+1)) then

           if(yc.ge.(2.0D0-rgh_h(2*x-1-rgh_zshift,z))) then
            do k=1,3
       Intu2r(x,y+yb,zp,k)=Intu2r(x,y+yb,zp,k)+
     & (an0+bn0)*u2r(x,y,zp,k)
c       if (abs(Intu2r(x,y+yb,zp,k))>0.0001) then
c       write(*,*)my_node_world,x,z,zp,y+yb,k,Intu2r(x,y+yb,zp,k)
c       endif
       om2r(x,y,zp,k)=om2r(x,y,zp,k)-(u2r(x,y,zp,k))
     & *rgh_ampl/2.0D0-Intu2r(x,y+yb,zp,k)*rgh_ampl2
            end do
CCC SCALAR
            do ith=1,scalar
       Ints2r(x,y+yb,zp,ith)=Ints2r(x,y+yb,zp,ith)+
     & (an0+bn0)*(th2r(x,y,zp,1+4*(ith-1))-1.0D0)
       th2r(x,y,zp,4+4*(ith-1))=th2r(x,y,zp,4+4*(ith-1))
     & -(th2r(x,y,zp,1+4*(ith-1))-1.0D0)*rgh_ampl/2.0D0
     & -(Ints2r(x,y+yb,zp,ith))*rgh_ampl2
            end do
CCCCCCCCCC
          endif
           if(yc.ge.(2.0D0-rgh_h(2*x-rgh_zshift,z))) then
            do k=1,3
       Intu2i(x,y+yb,zp,k)= Intu2i(x,y+yb,zp,k)+
     & (an0+bn0)*u2i(x,y,zp,k)
c       if (abs(Intu2i(x,y+yb,zp,k))>0.0001) then
c       write(*,*)my_node_world,x,z,zp,y+yb,Intu2i(x,y+yb,zp,k)
c       endif
       om2i(x,y,zp,k)=om2i(x,y,zp,k)-(u2i(x,y,zp,k))
     & *rgh_ampl/2.0D0-Intu2i(x,y+yb,zp,k)*rgh_ampl2
            end do
CCC SCALAR
            do ith=1,scalar
       Ints2i(x,y+yb,zp,ith)=Ints2i(x,y+yb,zp,ith)+
     & (an0+bn0)*(th2i(x,y,zp,1+4*(ith-1))-1.0D0)
       th2i(x,y,zp,4+4*(ith-1))=th2i(x,y,zp,4+4*(ith-1))
     & -(th2i(x,y,zp,1+4*(ith-1))-1.0D0)*rgh_ampl/2.0D0
     & -(Ints2i(x,y+yb,zp,ith))*rgh_ampl2
            end do
CCCCCCCCCC
          endif
           else
           if(yc.ge.(2.0D0-rgh_h(2*x-1+nxp-rgh_zshift,z))) then
            do k=1,3
       Intu2r(x,y+yb,zp,k)=Intu2r(x,y+yb,zp,k)+
     & (an0+bn0)*u2r(x,y,zp,k)
c       if (abs(Intu2r(x,y+yb,zp,k))>0.0001) then
c       write(*,*)my_node_world,x,z,zp,y+yb,k,Intu2r(x,y+yb,zp,k)
c       endif
       om2r(x,y,zp,k)=om2r(x,y,zp,k)-(u2r(x,y,zp,k))
     & *rgh_ampl/2.0D0-Intu2r(x,y+yb,zp,k)*rgh_ampl2
            end do
CCC SCALAR
            do ith=1,scalar
       Ints2r(x,y+yb,zp,ith)=Ints2r(x,y+yb,zp,ith)+
     & (an0+bn0)*(th2r(x,y,zp,1+4*(ith-1))-1.0D0)
       th2r(x,y,zp,4+4*(ith-1))=th2r(x,y,zp,4+4*(ith-1))
     & -(th2r(x,y,zp,1+4*(ith-1))-1.0D0)*rgh_ampl/2.0D0
     & -(Ints2r(x,y+yb,zp,ith))*rgh_ampl2
            end do
CCCCCCCCCC
          endif
           if(yc.ge.(2.0D0-rgh_h(2*x+nxp-rgh_zshift,z))) then
            do k=1,3
       Intu2i(x,y+yb,zp,k)= Intu2i(x,y+yb,zp,k)+
     & (an0+bn0)*u2i(x,y,zp,k)
c       if (abs(Intu2i(x,y+yb,zp,k))>0.0001) then
c       write(*,*)my_node_world,x,z,zp,y+yb,Intu2i(x,y+yb,zp,k)
c       endif
       om2i(x,y,zp,k)=om2i(x,y,zp,k)-(u2i(x,y,zp,k))
     & *rgh_ampl/2.0D0-Intu2i(x,y+yb,zp,k)*rgh_ampl2
            end do
CCC SCALAR
            do ith=1,scalar
       Ints2i(x,y+yb,zp,ith)=Ints2i(x,y+yb,zp,ith)+
     & (an0+bn0)*(th2i(x,y,zp,1+4*(ith-1))-1.0D0)
       th2i(x,y,zp,4+4*(ith-1))=th2i(x,y,zp,4+4*(ith-1))
     & -(th2i(x,y,zp,1+4*(ith-1))-1.0D0)*rgh_ampl/2.0D0
     & -(Ints2i(x,y+yb,zp,ith))*rgh_ampl2
            end do
CCCCCCCCCC
          endif
          endif

         end do
       end do
      end do

      end if
        
      end if

C damping based on x-y-distribution
      if (loctyp.eq.10) then

      zb=mod(my_node_world,nprocx)*nzd_new/nprocz
      do y=1,min(mby,nyp-yb+1)
        yc = 1.+eta(y+yb-1)
        do zp=1,(nzd_new/nprocz)
         z=zp+zb
         do x=1,nxp/2

            do k=1,3
       om2r(x,y,zp,k)=om2r(x,y,zp,k)-(u2r(x,y,zp,k))
     & *rgh_ampl*rgh_hy(2*x-1,nyp-yb+1)
       om2i(x,y,zp,k)=om2i(x,y,zp,k)-(u2i(x,y,zp,k))
     & *rgh_ampl*rgh_hy(2*x,nyp-yb+1)

            end do
CC Scalar            
      if(yc.ge.1.0D0) then
       do ith=1,scalar
       th2r(x,y,zp,4+4*(ith-1))=th2r(x,y,zp,4+4*(ith-1))
     & -(th2r(x,y,zp,1+4*(ith-1))-1.0D0)*rgh_ampl
     &  *rgh_hy(2*x,nyp-yb+1)
       end do
       else
       do ith=1,scalar
       th2i(x,y,zp,4+4*(ith-1))=th2i(x,y,zp,4+4*(ith-1))
     & -(th2i(x,y,zp,1+4*(ith-1)))*rgh_ampl
     &  *rgh_hy(2*x,nyp-yb+1)
       end do

      end if

         end do
       end do
      end do

      end if

C roughness with direct forcing based on x-z-distribution
      if (loctyp.eq.11) then

C only one wall (lower wall)
c      if (yb.ge.(nyp-1-92)) then

      zb=mod(my_node_world,nprocx)*nzd_new/nprocz
      do y=1,min(mby,nyp-yb+1)
C        write(*,*)yb,yc
        coeff=fd(nyp-yb-y+2)
        coeff1=fd1(nyp-yb-y+1)
        do zp=1,(nzd_new/nprocz)
         z=zp+zb
         do x=1,nxp/2
C           if(yc.le.rgh_h(2*x-1,z)) then
            do k=1,3
C               if (k.ne.2) then
                  om2r(x,y,zp,k)=om2r(x,y,zp,k)+coeff*
     &            u2r(x,y,zp,k)*abs(u2r(x,y,zp,k))
     &            +coeff1*u2r(x,y,zp,k)
C                end if
            end do
C                if(yc.le.rgh_h(2*x,z)) then
            do k=1,3
C                if (k.ne.2) then
                   om2i(x,y,zp,k)=om2i(x,y,zp,k)+coeff*
     &             u2i(x,y,zp,k)*abs(u2i(x,y,zp,k))
     &             +coeff1*u2i(x,y,zp,k)
C                end if
            end do
         end do
       end do
      end do

c      end if
      end if


      if (loctyp.gt.11) then
         write(*,*) 'loctyp = ',loctyp,' not implemented.'
         call stopnow(5433332)
      end if
      if (yb.eq.120) then
      endif
      end subroutine locf
