c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/getxy.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine getxy(boxr,boxi,zb,i,ur,ui)
c
c     Get an xy box from ur 
c     
c     size of xy box:  memnx * mbz * nyp (real and imaginary part)
c     
      implicit none
      
      include 'par.f'

      integer zb,i
      real boxr(memnx,mbz,nyp),boxi(memnx,mbz,nyp)
      real ur(memnx,memny,memnz,memnxyz),ui(memnx,memny,memnz,memnxyz)

      integer x,y,z

      if (mbz.eq.1) then
         do y=1,nyp
            do x=1,memnx
               boxr(x,1,y)=ur(x,y,zb,i)
               boxi(x,1,y)=ui(x,y,zb,i)
            end do
         end do
      else
         do z=zb,zb+mbz-1
            do y=1,nyp
               do x=1,memnx
                  boxr(x,z-zb+1,y)=ur(x,y,z,i)
                  boxi(x,z-zb+1,y)=ui(x,y,z,i)
               end do
            end do
         end do
      end if
      
      end subroutine getxy