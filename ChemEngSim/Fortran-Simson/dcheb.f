c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/dcheb.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine dcheb(b,a,n,m,md)
c
c     Calculates the first derivative in Chebyshev space
c     
c     a is the input array, the result is returned in b
c     n number of Chebyshev coefficients
c     m number of derivatives
c     md first index length on data array
c     
      implicit none

      integer n,m,md
      real a(md,n),b(md,n)
      integer i,k

      do k=1,m
         b(k,n-2)=2.*real(n-2)*a(k,n-1)
         b(k,n-1)=2.*real(n-1)*a(k,n)
      end do
      do i=n-3,1,-1
         do k=1,m
            b(k,i)=b(k,i+2)+2.*real(i)*a(k,i+1)
         end do
      end do
      do k=1,m
         b(k,n)=0.0
         b(k,1)=.5*b(k,1)
      end do
      
      end subroutine dcheb
