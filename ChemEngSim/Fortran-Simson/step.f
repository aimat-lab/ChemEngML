c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/step.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      real function step(x)
c     
c     Smooth step function:
c     x<=0 : step(x) = 0
c     x>=1 : step(x) = 1
c     Non-continuous derivatives at x=0.02 and x=0.98
c     
      implicit none
      
      real x
      
      if (x.le.0.02) then
         step = 0.0
      else
         if (x.le.0.98) then
            step = 1./( 1. + exp(1./(x - 1.) + 1./x) )
         else
            step = 1.
         end if
      end if
      
      end function step
