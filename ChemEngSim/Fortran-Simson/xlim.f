c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/xlim.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      real function xlim(x,xlow)
c
c     Returns the argument if argument is larger than xlow
c     If less than this it returns a value which
c     is always at least xlow/2
c     xlim has two continuous derivatives
c
      implicit none

      real x,xlow
      if (x.ge.xlow) then
         xlim=x
      else
         if (x.le.0.) then
            xlim=xlow*0.5
         else
            xlim=xlow*(0.5+(x/xlow)**3-0.5*(x/xlow)**4)
         end if
      end if

      return

      end function xlim
