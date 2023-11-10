c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/sgs_stress.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine sgs_stress(taur,taui,zb,zbp,alfa,beta,
     &     gur,gui,gu3r,gu3i,w3,prey,sigma3r,sigma3i,tau3r,tau3i,
     &     gsr,gsi,my_node_world)    
c
c     Used by (HPF) eddy-viscosity models to
c     compute the subgrid-scale force sigma_i = dtau_ij/dx_j
c     and for the scalars: sigma = dtau_j/dx_j
c
c     Input tau_ij is on taur,taui
c     Output sigma_i is on gur,gui and gsr,gsi
c     tau_ij and sigma_i are in Fourier/Physical/Fourier space
c
      implicit none

      include 'par.f'

      integer nxz
      parameter (nxz=memnx*mbz)

      integer i,x,y,z,xz,zb
      real alfa(nx/2),beta(nz),bbeta(nx/2)
      real c1
      real w3(memnx,mbz,nyp)
      real prey(nyp*2+15)

      real gur (memnx,memny,memnz,5)
      real gui (memnx,memny,memnz,5)
      real gsr (memnx,memny,memnz,scalar)
      real gsi (memnx,memny,memnz,scalar)
      real taur(memnx,memny,memnz,6+3*scalar)
      real taui(memnx,memny,memnz,6+3*scalar)

      real gu3r   (nxz,nyp  ),gu3i   (nxz,nyp  )
      real sigma3r(nxz,nyp,3),sigma3i(nxz,nyp,3)
      real tau3r  (nxz,nyp,6),tau3i  (nxz,nyp,6)

      integer zbp,ith,xb,my_node_world

      xb=mod(my_node_world,nprocx)*memnx
c
c     Compute spanwise wavenumber for all planes
c
      do z=1,mbz
         do x=1,nx/2
            xz = x+nx/2*(z-1)
            bbeta(xz) = beta(z+zb-1)
         end do
      end do
c
c     Get an xy box of the SGS stresses tau_ij, transform to 
c     Chebyshev space and normalise transform
c
      c1 = 2./real(nyp-1)
      do i=1,6
         call getxy(tau3r(1,1,i),tau3i(1,1,i),zbp,i,taur,taui)
         call vchbf(tau3r(1,1,i),w3,nyp,nxz,nxz,1,prey)
         call vchbf(tau3i(1,1,i),w3,nyp,nxz,nxz,1,prey)
         do y=1,ny
            do xz=1,nxz
               tau3r(xz,y,i) = c1*tau3r(xz,y,i)
               tau3i(xz,y,i) = c1*tau3i(xz,y,i)
            end do
         end do
      end do
c
c     Streamwise and spanwise derivatives
c
      do y=1,ny
         do xz=1,nxz
            x=xz+xb
            sigma3r(xz,y,1) = -alfa(x)*tau3i(xz,y,1)
     &           -bbeta(x)*tau3i(xz,y,3)
            sigma3i(xz,y,1) =  alfa(x)*tau3r(xz,y,1)
     &           +bbeta(x)*tau3r(xz,y,3)

            sigma3r(xz,y,2) = -alfa(x)*tau3i(xz,y,2)
     &           -bbeta(x)*tau3i(xz,y,5)
            sigma3i(xz,y,2) =  alfa(x)*tau3r(xz,y,2)
     &           +bbeta(x)*tau3r(xz,y,5)

            sigma3r(xz,y,3) = -alfa(x)*tau3i(xz,y,3)
     &           -bbeta(x)*tau3i(xz,y,6)
            sigma3i(xz,y,3) =  alfa(x)*tau3r(xz,y,3)
     &           +bbeta(x)*tau3r(xz,y,6)
         end do
      end do
c
c     Wall-normal derivatives
c
      call dcheb(gu3r,tau3r(1,1,2),ny,nxz,nxz)
      call dcheb(gu3i,tau3i(1,1,2),ny,nxz,nxz)
      do y=1,ny
         do xz=1,nxz
            sigma3r(xz,y,1) = sigma3r(xz,y,1) + gu3r(xz,y)
            sigma3i(xz,y,1) = sigma3i(xz,y,1) + gu3i(xz,y)
         end do
      end do

      call dcheb(gu3r,tau3r(1,1,4),ny,nxz,nxz)
      call dcheb(gu3i,tau3i(1,1,4),ny,nxz,nxz)
      do y=1,ny
         do xz=1,nxz
            sigma3r(xz,y,2) = sigma3r(xz,y,2) + gu3r(xz,y)
            sigma3i(xz,y,2) = sigma3i(xz,y,2) + gu3i(xz,y)
         end do
      end do

      call dcheb(gu3r,tau3r(1,1,5),ny,nxz,nxz)
      call dcheb(gu3i,tau3i(1,1,5),ny,nxz,nxz)
      do y=1,ny
         do xz=1,nxz
            sigma3r(xz,y,3) = sigma3r(xz,y,3) + gu3r(xz,y)
            sigma3i(xz,y,3) = sigma3i(xz,y,3) + gu3i(xz,y)
         end do
      end do
c
c     Transform back to Fourier/physical/Fourier space
c
      do i=1,3
         call vchbb(sigma3r(1,1,i),w3,nyp,nxz,nxz,1,prey)
         call vchbb(sigma3i(1,1,i),w3,nyp,nxz,nxz,1,prey)
         call putxy(sigma3r(1,1,i),sigma3i(1,1,i),zbp,i,gur,gui)
      end do


c
c     *******************************
c     ** Now do it for the scalars **
c     *******************************
c
c     Get an xy box of the SGS stresses tau_ij, transform to 
c     Chebyshev space and normalise transform
c
      do ith=1,scalar
         c1 = 2./real(nyp-1)
         do i=1,3
            call getxy(tau3r(1,1,i),tau3i(1,1,i),zbp,
     &           6+i+(ith-1)*3,taur,taui)
            call vchbf(tau3r(1,1,i),w3,nyp,nxz,nxz,1,prey)
            call vchbf(tau3i(1,1,i),w3,nyp,nxz,nxz,1,prey)
            do y=1,ny
               do xz=1,nxz
                  tau3r(xz,y,i) = c1*tau3r(xz,y,i)
                  tau3i(xz,y,i) = c1*tau3i(xz,y,i)
               end do
            end do
         end do
c
c     Streamwise and spanwise derivatives
c
         do y=1,ny
            do xz=1,nxz
               x=xz+xb
               sigma3r(xz,y,1) = -alfa(x)*tau3i(xz,y,1)
     &              -bbeta(x)*tau3i(xz,y,3)
               sigma3i(xz,y,1) =  alfa(x)*tau3r(xz,y,1)
     &              +bbeta(x)*tau3r(xz,y,3)
            end do
         end do
c
c     Wall-normal derivatives
c     
         call dcheb(gu3r,tau3r(1,1,2),ny,nxz,nxz)
         call dcheb(gu3i,tau3i(1,1,2),ny,nxz,nxz)
         do y=1,ny
            do xz=1,nxz
               sigma3r(xz,y,1) = sigma3r(xz,y,1) + gu3r(xz,y)
               sigma3i(xz,y,1) = sigma3i(xz,y,1) + gu3i(xz,y)
            end do
         end do
c     
c     Transform back to Fourier/physical/Fourier space
c
         call vchbb(sigma3r(1,1,1),w3,nyp,nxz,nxz,1,prey)
         call vchbb(sigma3i(1,1,1),w3,nyp,nxz,nxz,1,prey)
         call putxy(sigma3r(1,1,1),sigma3i(1,1,1),zbp,ith,gsr,gsi)
      end do

      end subroutine sgs_stress