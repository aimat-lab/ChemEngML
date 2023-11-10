c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/xysh.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine xysh(boxr,boxi,xs,zs,alfa,beta,zb,my_node_world)
c
c     Shift a fourier-transformed box so that xs=0 and zs=0
c
      implicit none

      include 'par.f'

      integer zb
      real boxr(memnx*mbz),boxi(memnx*mbz)
      real alfa(nx/2*mbz),beta(nz)
      real xs,zs

      integer xz,y,x,z,my_node_world
      real hr,arg,bbeta(nx/2*mbz)
c
c     Compute the full beta plane for vectorisation
c
      do z=1,mbz
         do x=1,nx/2
            xz=x+nx/2*(z-1)
            bbeta(xz)=beta(z+zb-1)
         end do
      end do
c
c     Perform the shifting using
c     arg  = - xs*alfa - zs*beta
c     box  = box*exp(i*arg) 
c          = box*(cos(arg)+i*sin(arg))
c     boxr = boxr*cos(arg) - boxi*sin(arg)
c     boxi = boxi*cos(arg) + boxr*sin(arg)
c
c      x=mod(my_node_world,nprocx)*memnx
      x=my_node_world*memnx
      do xz=1,memnx*mbz
         arg = - xs*alfa(xz+x) - zs*bbeta(xz+x) 
         hr         = boxr(xz)*cos(arg) - boxi(xz)*sin(arg)
         boxi(xz) = boxi(xz)*cos(arg) + boxr(xz)*sin(arg)
         boxr(xz) = hr
      end do

      end subroutine xysh
