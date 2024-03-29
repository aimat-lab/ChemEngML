c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/splfre.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine splfre(uu1,ue1,uu2,ue2,bstart,dstar,xl,xsc,namfre)
c      
c     Reads a freestream velocity table from file namfre
c     and finds the spline interpolated velocity distribution
c
      implicit none

      include 'par.f'

      character*80 namfre
      real uu1(nxp/2),uu2(nxp/2),ue1(nxp/2),ue2(nxp/2)
      real bstart,dstar,xl,xsc

      integer mtab
      parameter (mtab=2049)
      real xtab(mtab),utab(mtab),u2tab(mtab),w(mtab)
      real xblu1,xblu2,xble1,xble2,xc
      integer i,x,n
c
c     Read freestream table 
c      
      open(unit=17,file=namfre,status='old')
      read(17,*) n
      if (n.gt.mtab) then
         write(*,*) 'The freestream table is too large'
         write(*,*) 'Increase mtab in splfre.f to at least ',n
         write(*,*) 'and recompile'
         stop
      end if
      do i=1,n
         read(17,*) xtab(i),utab(i)
      end do
      close(unit=17)
c
c     Spline interpolate
c      
      call spline(xtab,utab,n,1.e30,1.e30,u2tab,w)
      do x=1,nxp/2
c
c     Odd points
c
         xc=real(2*x-1-nxp/2-1)/real(nxp)*xl+xsc
c
c     Calculate coordinate in interval [bstart,bstart+xl[
c
         xblu1=xc-(int((xc-bstart)/xl+1.)-1.)*xl
c
c     Calculate coordinate in interval [bstart+xl,bstart+2*xl[
c
         xblu2=xblu1+xl
c
c     Rescale to boundary layer coordinates
c
         xblu1=xblu1/dstar
         xblu2=xblu2/dstar
         call spliet(xtab,utab,u2tab,n,xblu1,uu1(x))
         call spliet(xtab,utab,u2tab,n,xblu2,uu2(x))
c
c     Even points
c
         xc=real(2*x-nxp/2-1)/real(nxp)*xl+xsc
c
c     Calculate coordinate in interval [bstart,bstart+xl[
c
         xble1=xc-(int((xc-bstart)/xl+1.)-1.)*xl
c
c     Calculate coordinate in interval [bstart+xl,bstart+2*xl[
c
         xble2=xble1+xl
c
c     Rescale to boundary layer coordinates
c
         xble1=xble1/dstar
         xble2=xble2/dstar
         call spliet(xtab,utab,u2tab,n,xble1,ue1(x))
         call spliet(xtab,utab,u2tab,n,xble2,ue2(x))
      end do

      end subroutine splfre
