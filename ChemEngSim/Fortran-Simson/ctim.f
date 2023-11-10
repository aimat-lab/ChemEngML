c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/ctim.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine ctim(ctime,wtime)
c
c     Get wall time and CPU time
c
      implicit none

      real wtime,ctime

      call cpu_time(ctime)
      call wall_time(wtime)
      
      end subroutine ctim
