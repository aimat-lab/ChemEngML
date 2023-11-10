c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/xzsh.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine xzsh(b2r,b2i,xs,zs,alfa,beta,yb)
c
c     Shift a Fourier-transformed xz box so that xs=0 and zs=0
c
      implicit none

      include 'par.f'

      integer yb
      real b2r(nxp/2+1,mby,nzd),b2i(nxp/2+1,mby,nzd)
      real alfa(nx/2*mbz),beta(nz)
      real xs,zs

      integer y,x,z
      real hr,argx,argz
      real cx(nx/2),sx(nx/2)
      real ca(nx/2),sa(nx/2)

c     There is a problem here with the sizes of the arrays in 
c     the 2D parallel version
      call stopnow(545334)

      do x=1,nx/2
         argx=-xs*alfa(x)
         cx(x)=cos(argx)
         sx(x)=sin(argx)
      end do
      do z=1,nzc
         argz=-zs*beta(z)
         do x=1,nx/2
c     arg=argx+argz
c     exp(i*arg)=exp(i*argx)*exp(i*argz)=
c     cos(argx)*cos(argz)-sin(argx)*sin(argz)+
c     i*(cos(argx)*sin(argz)+sin(argx)*cos(argz))
            ca(x)=cx(x)*cos(argz)-sx(x)*sin(argz)
            sa(x)=cx(x)*sin(argz)+sx(x)*cos(argz)
         end do
         do y=1,min(mby,nyp-yb+1)
            do x=1,nx/2
c     b2=b2*exp(i*arg)
c     b2r=b2r*cos(arg)-b2i*sin(arg)
c     b2i=b2i*cos(arg)+b2r*sin(arg)
               hr=b2r(x,y,z)*ca(x)-b2i(x,y,z)*sa(x)
               b2i(x,y,z)=b2i(x,y,z)*ca(x)+b2r(x,y,z)*sa(x)
               b2r(x,y,z)=hr
            end do
         end do
      end do

      end subroutine xzsh
