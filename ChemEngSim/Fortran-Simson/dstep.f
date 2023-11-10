c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/dstep.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      real function dstep(x)
c
c     Computes first derivative of smooth step function (see step.f)
c     
      implicit none
      
      real x,t1,t5,t7,t9,t11

      if (x.le.0.98.and.x.ge.0.02) then
         t1 = x-1.
         t5 = exp(1./t1+1./x)
         t7 = (1.+t5)*(1.+t5)
         t9 = t1*t1
         t11= x**2
         dstep = 1./t7*(1./t9+1./t11)*t5
      else
         dstep = 0.
      end if
      
      end function dstep
