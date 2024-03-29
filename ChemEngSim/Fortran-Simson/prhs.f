c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/prhs.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine prhs(ur,ui,puw,alfa,beta,re,pr,an,prey,
     &     u3r,u3i,om3r,om3i,th3r,th3i,pomyr,pomyi,
     &     pvr,pvi,pthr,pthi,phr,phi,w3,dthr,dthi,
     &     iles,gur,gui,taur,taui,gu3r,gu3i,ggu3r,ggu3i,gsr,gsi,
     &     gth3r,gth3i,ggth3r,ggth3i,gewy1,gewy2,gewy12,gewy22,iord,
     &     diags,diags2,gplane,
     &     filtxz,filtxz2,lpfxz,lpfxz2,my_node_world,ihighorder,cs)
c
c     Calculates the partial right hand side for the first Euler step
c     or first RK substep. Further computes the streamwise and spanwise
c     vorticity omx,omz.
c     
c     The equations read in general form:
c     pomyn=-(2*re/(an+bn)+(d2-k2))(i*beta*u-i*alfa*w)-bn*2*re/(an+bn)*homynm1
c     pvn=-(2*re/(an+bn)+(d2-k2))phin-bn*2*re/(an+bn)*hvnm1
c     phin=(d2-k2)v
c     wavenumber zero
c     pun=-(2*re/(an+bn)+d2)un-bn*2*re/(an+bn)*hunm1
c     pwn=-(2*re/(an+bn)+d2)wn-bn*2*re/(an+bn)*hwnm1
c
c     For Euler/RK stage n=1 bn=0 and these reduce to
c     pomyn=-(2*re/an+(d2-k2))(i*beta*u-i*alfa*w)
c     pvn=-(2*re/an+(d2-k2))phin
c     phin=(d2-k2)vn
c     wavenumber zero
c     pun=-(2*re/an+d2)un
c     pwn=-(2*re/an+d2)wn
c
      implicit none

      include 'par.f'

      real ur(memnx,memny,memnz,memnxyz),ui(memnx,memny,memnz,memnxyz)
      real prey(nyp*2+15)
      real u3r(memnx,mbz,nyp,3),u3i(memnx,mbz,nyp,3)
      real th3r(memnx,mbz,nyp,scalar),th3i(memnx,mbz,nyp,scalar)
      real om3r(memnx,mbz,nyp,3),om3i(memnx,mbz,nyp,3)
      real pvr(memnx,mbz,nyp),pvi(memnx,mbz,nyp)
      real pomyr(memnx,mbz,nyp),pomyi(memnx,mbz,nyp)
      real phr(memnx,mbz,nyp),phi(memnx,mbz,nyp)
      real pthr(memnx,mbz,nyp,scalar),pthi(memnx,mbz,nyp,scalar)
      real puw(ny,2+scalar)
      real w3(memnx,mbz,nyp)
c
c     Additional working space
c      
      real dthr(memnx,mbz,nyp,scalar),dthi(memnx,mbz,nyp,scalar)
      real alfa(nx/2*mbz),beta(nz)
      real re,an,pr(scalar)

      real tuw(ny,2+scalar)
      integer y,z,zb,i,x,mzb,nxz,ll,ith
      real k2,lam
c
c     LES
c
      real cs
      integer iles,iord,ihighorder
      real gur (memnx,memny,memnz,5+2*scalar)
      real gui (memnx,memny,memnz,5+2*scalar)
      real gsr (memnx,memny,memnz,scalar)
      real gsi (memnx,memny,memnz,scalar)
      real taur(memnx,memny,memnz,6+2*scalar)
      real taui(memnx,memny,memnz,6+2*scalar)
      real gu3r  (memnx,mbz,nyp),gu3i  (memnx,mbz,nyp)
      real ggu3r (memnx,mbz,nyp),ggu3i (memnx,mbz,nyp)
      real gth3r  (memnx,mbz,nyp),gth3i  (memnx,mbz,nyp)
      real ggth3r (memnx,mbz,nyp),ggth3i (memnx,mbz,nyp)
      real gplane(memnx,mbz,nyp)
      real gewy1 (nyp,5),gewy2 (nyp,5)
      real gewy12(nyp,5),gewy22(nyp,5)
      real diags (nyp,5)
      real diags2(nyp,5)
      real filtxz (memnx,nz),lpfxz (memnx,nz)
      real filtxz2(memnx,nz),lpfxz2(memnx,nz)

      integer my_node_world,zbp,xp,xb
c
c     Loop over all xy-boxes
c
      nxz=memnx*mbz
      xb=mod(my_node_world,nprocx)*memnx
      do zbp=1,memnz
         zb = (my_node_world/nprocx)*memnz+zbp
         do ll=1,3
c
c     Get velocities onto u3r(memnx,mbz,nyp) in physical space
c
            call getxy(u3r(1,1,1,ll),u3i(1,1,1,ll),zbp,ll,ur,ui)
c
c     Compute HPF filtered velocity gu3 from u3
c
            if (iles.eq.1.or.iles.eq.3) then
c               write(*,*)'Not Implemented'
c               call stopnow(2233445)
c
c     Get gu3
c
               do y=1,nyp
                  do z=1,mbz
                     do x=1,memnx
                        gu3r(x,z,y) = u3r(x,z,y,ll)
                        gu3i(x,z,y) = u3i(x,z,y,ll)
                     end do
                  end do
               end do
               if (ihighorder.eq.0) then
c
c     Only two-dimensional filtering
c
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gu3r(x,z,y) = u3r(x,z,y,ll) - 
     &                          lpfxz(x,zb)*gu3r(x,z,y)
                           gu3i(x,z,y) = u3i(x,z,y,ll) - 
     &                          lpfxz(x,zb)*gu3i(x,z,y)
                        end do
                     end do
                  end do   
               else if (ihighorder.eq.1) then
c
c     Do the multiple filtering via the iterative way
c
                  do i=1,iord+1
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              ggu3r(x,z,y) = filtxz(x,zb)*gu3r(x,z,y)
                              ggu3i(x,z,y) = filtxz(x,zb)*gu3i(x,z,y)
                           end do
                        end do
                     end do
                     call lpf_y(ggu3r,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     call lpf_y(ggu3i,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              gu3r(x,z,y) = gu3r(x,z,y)-ggu3r(x,z,y)
                              gu3i(x,z,y) = gu3i(x,z,y)-ggu3i(x,z,y)
                           end do
                        end do
                     end do
                  end do
               else if (ihighorder.eq.2) then
c
c     Do the multiple filtering via the old way
c     High-pass filter velocities in y
c
                  do i=1,iord+1
                     call hpf_y(gu3r,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     call hpf_y(gu3i,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                  end do
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Store low-pass filtered velocities
c
                           gu3r(x,z,y) = u3r(x,z,y,ll)-gu3r(x,z,y)
                           gu3i(x,z,y) = u3i(x,z,y,ll)-gu3i(x,z,y)
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gu3r(x,z,y) = u3r(x,z,y,ll) - 
     &                          lpfxz(x,zb)*gu3r(x,z,y)
                           gu3i(x,z,y) = u3i(x,z,y,ll) - 
     &                          lpfxz(x,zb)*gu3i(x,z,y)
                        end do
                     end do
                  end do 
               else
                  call stopnow(47432)
               end if
c
c     Put into gur,gui
c
               call putxy(gu3r,gu3i,zbp,ll,gur,gui)
            end if


            if (iles.eq.3.and.cs.eq.0) then
c               write(*,*)'Not Implemented'
c               call stopnow(22334455)
c
c     Get the filtered velocities with the secondary filter
c     Get gu3
c
               do y=1,nyp
                  do z=1,mbz
                     do x=1,memnx
                        gu3r(x,z,y) = u3r(x,z,y,ll)
                        gu3i(x,z,y) = u3i(x,z,y,ll)
                     end do
                  end do
               end do

               if (ihighorder.eq.0) then
c
c     Only two-dimensional filtering
c
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gu3r(x,z,y) = u3r(x,z,y,ll) - 
     &                          lpfxz2(x,zb)*gu3r(x,z,y)
                           gu3i(x,z,y) = u3i(x,z,y,ll) - 
     &                          lpfxz2(x,zb)*gu3i(x,z,y)
                        end do
                     end do
                  end do 
               else if (ihighorder.eq.1) then
c
c     Do it the iterative way
c
                  do i=1,iord+1
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              ggu3r(x,z,y) = filtxz2(x,zb)*gu3r(x,z,y)
                              ggu3i(x,z,y) = filtxz2(x,zb)*gu3i(x,z,y)
                           end do
                        end do
                     end do
                     call lpf_y(ggu3r,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     call lpf_y(ggu3i,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              gu3r(x,z,y) = gu3r(x,z,y)-ggu3r(x,z,y)
                              gu3i(x,z,y) = gu3i(x,z,y)-ggu3i(x,z,y)
                           end do
                        end do
                     end do
                  end do
               else if (ihighorder.eq.2) then
c
c     Do it the old way, high-pass filter velocities in y
c
                  do i=1,iord+1
                     call hpf_y(gu3r,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     call hpf_y(gu3i,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                  end do
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Store low-pass filtered velocities
c
                           gu3r(x,z,y) = u3r(x,z,y,ll)-gu3r(x,z,y)
                           gu3i(x,z,y) = u3i(x,z,y,ll)-gu3i(x,z,y)
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gu3r(x,z,y) = u3r(x,z,y,ll) - 
     &                          lpfxz2(x,zb)*gu3r(x,z,y)
                           gu3i(x,z,y) = u3i(x,z,y,ll) - 
     &                          lpfxz2(x,zb)*gu3i(x,z,y)
                        end do
                     end do
                  end do 
               else
                  call stopnow(47432)
               end if
c
c     Put into gur,gui
c
               call putxy(gu3r,gu3i,zbp,ll,taur,taui)
            end if
c
c     Transform the velocities to Chebyshev space
c
            call vchbf(u3r(1,1,1,ll),w3,nyp,nxz,nxz,1,prey)
            call vchbf(u3i(1,1,1,ll),w3,nyp,nxz,nxz,1,prey)
c
c     Normalize Chebyshev transform (on ny grid)
c
            do z=zb,zb+mbz-1
               mzb=z-zb+1
               do y=1,ny
                  do xp=1,memnx
                     u3r(xp,mzb,y,ll)=u3r(xp,mzb,y,ll)*(2./real(nyp-1))
                     u3i(xp,mzb,y,ll)=u3i(xp,mzb,y,ll)*(2./real(nyp-1))
                  end do
               end do
            end do
         end do

         do ith=1,scalar
c
c     Get the scalar onto th3r(memnx,mbz,nyp) in physical space
c
            call getxy(th3r(1,1,1,ith),th3i(1,1,1,ith),zbp,
     &           8+pressure+3*(ith-1),ur,ui)

c
c     Compute HPF filtered scalar gth3 from th3
c
            if (iles.eq.1.or.iles.eq.3) then
c
c     Get gth3
c
               do y=1,nyp
                  do z=1,mbz
                     do x=1,memnx
                        gth3r(x,z,y) = th3r(x,z,y,ith)
                        gth3i(x,z,y) = th3i(x,z,y,ith)
                     end do
                  end do
               end do
               if (ihighorder.eq.0) then
c
c     Only two-dimensional filtering
c
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Low-pass filter scalar in x/z and store high-pass filter
c
                           gth3r(x,z,y) = th3r(x,z,y,ith) -
     &                          lpfxz(x,zb)*gth3r(x,z,y)
                           gth3i(x,z,y) = th3i(x,z,y,ith) -
     &                          lpfxz(x,zb)*gth3i(x,z,y)
                        end do
                     end do
                  end do
               else if (ihighorder.eq.1) then
c
c     Do the multiple filtering via the iterative way
c
                  do i=1,iord+1
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              ggth3r(x,z,y) = filtxz(x,zb)*gth3r(x,z,y)
                              ggth3i(x,z,y) = filtxz(x,zb)*gth3i(x,z,y)
                           end do
                        end do
                     end do
                     call lpf_y(ggth3r,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     call lpf_y(ggth3i,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              gth3r(x,z,y) = gth3r(x,z,y)-ggth3r(x,z,y)
                              gth3i(x,z,y) = gth3i(x,z,y)-ggth3i(x,z,y)
                           end do
                        end do
                     end do
                  end do
               else if (ihighorder.eq.2) then
c
c     Do the multiple filtering via the old way
c     High-pass filter scalar in y
c
                  do i=1,iord+1
                     call hpf_y(gth3r,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     call hpf_y(gth3i,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                  end do
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Store low-pass filtered scalar
c
                           gth3r(x,z,y) = th3r(x,z,y,ith)-gth3r(x,z,y)
                           gth3i(x,z,y) = th3i(x,z,y,ith)-gth3i(x,z,y)
c
c     Low-pass filter scalar in x/z and store high-pass filter
c
                           gth3r(x,z,y) = th3r(x,z,y,ith) -
     &                          lpfxz(x,zb)*gth3r(x,z,y)
                           gth3i(x,z,y) = th3i(x,z,y,ith) -
     &                          lpfxz(x,zb)*gth3i(x,z,y)
                        end do
                     end do
                  end do
               else
                  call stopnow(47432)
               end if
c
c     Put into gur,gui
c
               if (iles.eq.3) then
                  call putxy(gth3r,gth3i,zbp,5+ith,gur,gui)
               else
                  call putxy(gth3r,gth3i,zbp,ith,gsr,gsi)
               end if
            end if


            if (iles.eq.3.and.cs.eq.0) then
c
c     Get the filtered scalar with the secondary filter
c     Get gth3
c
               do y=1,nyp
                  do z=1,mbz
                     do x=1,memnx
                        gth3r(x,z,y) = th3r(x,z,y,ith)
                        gth3i(x,z,y) = th3i(x,z,y,ith)
                     end do
                  end do
               end do

               if (ihighorder.eq.0) then
c
c     Only two-dimensional filtering
c
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gth3r(x,z,y) = th3r(x,z,y,ith) -
     &                          lpfxz2(x,zb)*gth3r(x,z,y)
                           gth3i(x,z,y) = th3i(x,z,y,ith) -
     &                          lpfxz2(x,zb)*gth3i(x,z,y)
                        end do
                     end do
                  end do
               else if (ihighorder.eq.1) then
c
c     Do it the iterative way
c
                  do i=1,iord+1
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              ggth3r(x,z,y) = filtxz2(x,zb)*gth3r(x,z,y)
                              ggth3i(x,z,y) = filtxz2(x,zb)*gth3i(x,z,y)
                           end do
                        end do
                     end do
                     call lpf_y(ggth3r,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     call lpf_y(ggth3i,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              gth3r(x,z,y) = gth3r(x,z,y)-ggth3r(x,z,y)
                              gth3i(x,z,y) = gth3i(x,z,y)-ggth3i(x,z,y)
                           end do
                        end do
                     end do
                  end do
               else if (ihighorder.eq.2) then
c
c     Do it the old way, high-pass filter velocities in y
c
                  do i=1,iord+1
                     call hpf_y(gth3r,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     call hpf_y(gth3i,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                  end do
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Store low-pass filtered velocities
c
                           gth3r(x,z,y) = th3r(x,z,y,ith)-gth3r(x,z,y)
                           gth3i(x,z,y) = th3i(x,z,y,ith)-gth3i(x,z,y)
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gth3r(x,z,y) = th3r(x,z,y,ith) -
     &                          lpfxz2(x,zb)*gth3r(x,z,y)
                           gth3i(x,z,y) = th3i(x,z,y,ith) -
     &                          lpfxz2(x,zb)*gth3i(x,z,y)
                        end do
                     end do
                  end do
               else
                  call stopnow(47432)
               end if
c
c     Put into taur,taui
c
               call putxy(gth3r,gth3i,zbp,6+ith,taur,taui)
            end if
c
c     Transform the velocities to Chebyshev space
c
            call vchbf(th3r(1,1,1,ith),w3,nyp,nxz,nxz,1,prey)
            call vchbf(th3i(1,1,1,ith),w3,nyp,nxz,nxz,1,prey)
c
c     Normalize Chebyshev transform (on ny grid)
c
            do z=zb,zb+mbz-1
               mzb=z-zb+1
               do y=1,ny
                  do xp=1,memnx
                     th3r(xp,mzb,y,ith)=th3r(xp,mzb,y,ith)*
     &                    (2./real(nyp-1))
                     th3i(xp,mzb,y,ith)=th3i(xp,mzb,y,ith)*
     &                    (2./real(nyp-1))
                  end do
               end do
            end do
         end do
c
c     Wavenumber zero
c
         if (zb.eq.1.and.my_node_world.eq.0) then
            do y=1,ny
               puw(y,1)=u3r(1,1,y,1)
               puw(y,2)=u3r(1,1,y,3)
            end do
            call dcheb(tuw(1,1),puw(1,1),ny,1,1)
            call dcheb(puw(1,1),tuw(1,1),ny,1,1)
            call dcheb(tuw(1,2),puw(1,2),ny,1,1)
            call dcheb(puw(1,2),tuw(1,2),ny,1,1)
            do y=1,ny
               puw(y,1)=-2.*re/an*u3r(1,1,y,1)-puw(y,1)
               puw(y,2)=-2.*re/an*u3r(1,1,y,3)-puw(y,2)
            end do

            do ith=1,scalar
               do y=1,ny
                  puw(y,2+ith)=th3r(1,1,y,ith)
               end do
               call dcheb(tuw(1,2+ith),puw(1,2+ith),ny,1,1)
               call dcheb(puw(1,2+ith),tuw(1,2+ith),ny,1,1)
               do y=1,ny
                  puw(y,2+ith)=-2.*(pr(ith)*re)/an*
     &                 th3r(1,1,y,ith)-puw(y,2+ith)
               end do
            end do
         end if
c
c     Construct vorticities
c
         do y=1,ny
            do z=zb,zb+mbz-1
               do xp=1,memnx
                  x=xp+xb
                  mzb=z-zb+1
                  om3r(xp,mzb,y,2)=-u3i(xp,mzb,y,1)*beta(z)+
     &                 u3i(xp,mzb,y,3)*alfa(x)
                  om3i(xp,mzb,y,2)=u3r(xp,mzb,y,1)*beta(z)-
     &                 u3r(xp,mzb,y,3)*alfa(x)
               end do
            end do
         end do

         call dcheb(om3r,u3r(1,1,1,3),ny,nxz,nxz)
         call dcheb(om3i,u3i(1,1,1,3),ny,nxz,nxz)
         call dcheb(om3r(1,1,1,3),u3r,ny,nxz,nxz)
         call dcheb(om3i(1,1,1,3),u3i,ny,nxz,nxz)

         do y=1,ny
            do z=zb,zb+mbz-1
               do xp=1,memnx
                  x=xp+xb
                  mzb=z-zb+1
                  om3r(xp,mzb,y,1)=om3r(xp,mzb,y,1)
     &                 +u3i(xp,mzb,y,2)*beta(z)
                  om3i(xp,mzb,y,1)=om3i(xp,mzb,y,1)
     &                 -u3r(xp,mzb,y,2)*beta(z)
                  om3r(xp,mzb,y,3)=-om3r(xp,mzb,y,3)-
     &                 u3i(xp,mzb,y,2)*alfa(x)
                  om3i(xp,mzb,y,3)=-om3i(xp,mzb,y,3)+
     &                 u3r(xp,mzb,y,2)*alfa(x)
               end do
            end do
         end do
c
c     Contruct pomy and pv
c
c     phi = d2v,pomy=d2omy 
         call dcheb(w3,u3r(1,1,1,2),ny,nxz,nxz)
         call dcheb(phr,w3,ny,nxz,nxz)
         call dcheb(w3,u3i(1,1,1,2),ny,nxz,nxz)
         call dcheb(phi,w3,ny,nxz,nxz)
         call dcheb(w3,om3r(1,1,1,2),ny,nxz,nxz)
         call dcheb(pomyr,w3,ny,nxz,nxz)
         call dcheb(w3,om3i(1,1,1,2),ny,nxz,nxz)
         call dcheb(pomyi,w3,ny,nxz,nxz)


         do ith=1,scalar
            call dcheb(w3, th3r(1,1,1,ith),ny,nxz,nxz)
            call dcheb(dthr(1,1,1,ith),th3i(1,1,1,ith),ny,nxz,nxz)

            call dcheb(pthr(1,1,1,ith),w3 ,ny,nxz,nxz)
            call dcheb(pthi(1,1,1,ith),dthr(1,1,1,ith),ny,nxz,nxz)
c
c     Transform and put dtheta/dy into position 9
c
            call vchbb(w3 ,dthi(1,1,1,ith),nyp,nxz,nxz,1,prey)
            call vchbb(dthr(1,1,1,ith),dthi(1,1,1,ith),
     &           nyp,nxz,nxz,1,prey)
            call putxy(w3,dthr(1,1,1,ith),zbp,
     &           9+pressure+3*(ith-1),ur,ui)
            if (iles.eq.3) then
               do y=1,nyp
                  do z=1,mbz
                     do x=1,memnx
                        dthi(x,z,y,ith) = dthr(x,z,y,ith)
                        dthr(x,z,y,ith) = w3(x,z,y)
                     end do
                  end do
               end do
            end if
         end do
c
c     Complete pomy and phi
c
         do y=1,ny
            do z=zb,zb+mbz-1
               do xp=1,memnx
                  x=xp+xb
                  mzb=z-zb+1
                  k2=alfa(x)**2+beta(z)**2
                  lam=k2-2.*re/an
                  phr(xp,mzb,y)=phr(xp,mzb,y)-k2*u3r(xp,mzb,y,2)
                  phi(xp,mzb,y)=phi(xp,mzb,y)-k2*u3i(xp,mzb,y,2)
                  pomyr(xp,mzb,y)=-pomyr(xp,mzb,y)+lam*om3r(xp,mzb,y,2)
                  pomyi(xp,mzb,y)=-pomyi(xp,mzb,y)+lam*om3i(xp,mzb,y,2)

                  do ith=1,scalar
                     pthr(xp,mzb,y,ith)=-pthr(xp,mzb,y,ith)
     &                    +(k2-2.*(pr(ith)*re)/an)*th3r(xp,mzb,y,ith)
                     pthi(xp,mzb,y,ith)=-pthi(xp,mzb,y,ith)
     &                    +(k2-2.*(pr(ith)*re)/an)*th3i(xp,mzb,y,ith)
                  end do
               end do
            end do
         end do
c
c     pv=d2phi
c
         call dcheb(w3,phr,ny,nxz,nxz)
         call dcheb(pvr,w3,ny,nxz,nxz)
         call dcheb(w3,phi,ny,nxz,nxz)
         call dcheb(pvi,w3,ny,nxz,nxz)
c
c     Complete pv
c
         do y=1,ny
            do z=zb,zb+mbz-1
               do xp=1,memnx
                  x=xp+xb
                  mzb=z-zb+1
                  lam=alfa(x)**2+beta(z)**2-2.*re/an
                  pvr(xp,mzb,y)=-pvr(xp,mzb,y)+lam*phr(xp,mzb,y)
                  pvi(xp,mzb,y)=-pvi(xp,mzb,y)+lam*phi(xp,mzb,y)
               end do
            end do
         end do
c
c     Pad, transform and store
c
         if (ny+1.le.nyp) then
            do y=ny+1,max(nyp,ny+1)
               do z=zb,zb+mbz-1
                  do xp=1,memnx
                     mzb=z-zb+1
                     om3r(xp,mzb,y,1)=0.0
                     om3i(xp,mzb,y,1)=0.0
                     om3r(xp,mzb,y,3)=0.0
                     om3i(xp,mzb,y,3)=0.0
                  end do
               end do
            end do
         end if
c
c     Streamwise vorticity
c
         call vchbb(om3r,w3,nyp,nxz,nxz,1,prey)
         call vchbb(om3i,w3,nyp,nxz,nxz,1,prey)
         call putxy(om3r,om3i,zbp,4,ur,ui)
c
c     Spanwise vorticity
c
         call vchbb(om3r(1,1,1,3),w3,nyp,nxz,nxz,1,prey)
         call vchbb(om3i(1,1,1,3),w3,nyp,nxz,nxz,1,prey)
         call putxy(om3r(1,1,1,3),om3i(1,1,1,3),zbp,5,ur,ui)
c
c     Partial right-hand sides
c
         call putxy(pvr,pvi,zbp,6,ur,ui)
         call putxy(pomyr,pomyi,zbp,7,ur,ui)
         do ith=1,scalar
            call putxy(pthr(1,1,1,ith),pthi(1,1,1,ith),
     &           zbp,10+pressure+3*(ith-1),ur,ui)
         end do

         if (iles.eq.3) then
c
c     Save also the HPF vorticity
c
            do ll=1,2
c
c     Compute HPF filtered velocity gu3 from u3
c     Get gu3
c
               do y=1,nyp
                  do z=1,mbz
                     do x=1,memnx
                        gu3r(x,z,y) = om3r(x,z,y,ll*2-1)
                        gu3i(x,z,y) = om3i(x,z,y,ll*2-1)
                     end do
                  end do
               end do
               
               if (ihighorder.eq.0) then
c
c     Only two-dimensional filtering
c
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gu3r(x,z,y) = om3r(x,z,y,ll*2-1) - 
     &                          lpfxz(x,zb)*gu3r(x,z,y)
                           gu3i(x,z,y) = om3i(x,z,y,ll*2-1) - 
     &                          lpfxz(x,zb)*gu3i(x,z,y)
                        end do
                     end do
                  end do 
               else if (ihighorder.eq.1) then
c
c     Do it the iterative way
c
                  do i=1,iord+1
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              ggu3r(x,z,y) = filtxz(x,zb)*gu3r(x,z,y)
                              ggu3i(x,z,y) = filtxz(x,zb)*gu3i(x,z,y)
                           end do
                        end do
                     end do
                     call lpf_y(ggu3r,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     call lpf_y(ggu3i,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              gu3r(x,z,y) = gu3r(x,z,y)-ggu3r(x,z,y)
                              gu3i(x,z,y) = gu3i(x,z,y)-ggu3i(x,z,y)
                           end do
                        end do
                     end do
                  end do
               else if (ihighorder.eq.2) then
c
c     Do it the old way, high-pass filter velocities in y
c
                  do i=1,iord+1
                     call hpf_y(gu3r,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     call hpf_y(gu3i,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                  end do
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Store low-pass filtered velocities
c
                           gu3r(x,z,y) = om3r(x,z,y,ll*2-1)-gu3r(x,z,y)
                           gu3i(x,z,y) = om3i(x,z,y,ll*2-1)-gu3i(x,z,y)
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gu3r(x,z,y) = om3r(x,z,y,ll*2-1) - 
     &                          lpfxz(x,zb)*gu3r(x,z,y)
                           gu3i(x,z,y) = om3i(x,z,y,ll*2-1) - 
     &                          lpfxz(x,zb)*gu3i(x,z,y)
                        end do
                     end do
                  end do 
               else
                  call stopnow(47432)
               end if
c
c     Put into gur,gui
c
               call putxy(gu3r,gu3i,zbp,ll+3,gur,gui)
            end do
         end if
         
         if (iles.eq.3.and.cs.eq.0) then
c
c     Save also the HPF vorticity with secondary HPF
c
            do ll=1,2
c
c     Compute HPF filtered velocity gu3 from u3
c     Get gu3
c
               do y=1,nyp
                  do z=1,mbz
                     do x=1,memnx
                        gu3r(x,z,y) = om3r(x,z,y,ll*2-1)
                        gu3i(x,z,y) = om3i(x,z,y,ll*2-1)
                     end do
                  end do
               end do
               
               if (ihighorder.eq.0) then
c
c     Only two-dimensional filtering
c
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gu3r(x,z,y) = om3r(x,z,y,ll*2-1) - 
     &                          lpfxz2(x,zb)*gu3r(x,z,y)
                           gu3i(x,z,y) = om3i(x,z,y,ll*2-1) - 
     &                          lpfxz2(x,zb)*gu3i(x,z,y)
                        end do
                     end do
                  end do 
               else if (ihighorder.eq.1) then
c
c     Do it the iterative way
c
                  do i=1,iord+1
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              ggu3r(x,z,y) = filtxz2(x,zb)*gu3r(x,z,y)
                              ggu3i(x,z,y) = filtxz2(x,zb)*gu3i(x,z,y)
                           end do
                        end do
                     end do
                     call lpf_y(ggu3r,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     call lpf_y(ggu3i,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              gu3r(x,z,y) = gu3r(x,z,y)-ggu3r(x,z,y)
                              gu3i(x,z,y) = gu3i(x,z,y)-ggu3i(x,z,y)
                           end do
                        end do
                     end do
                  end do
               else if (ihighorder.eq.2) then
c
c     Do it the old way, high-pass filter velocities in y
c
                  do i=1,iord+1
                     call hpf_y(gu3r,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     call hpf_y(gu3i,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                  end do
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Store low-pass filtered velocities
c
                           gu3r(x,z,y) = om3r(x,z,y,ll*2-1)-gu3r(x,z,y)
                           gu3i(x,z,y) = om3i(x,z,y,ll*2-1)-gu3i(x,z,y)
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gu3r(x,z,y) = om3r(x,z,y,ll*2-1) - 
     &                          lpfxz2(x,zb)*gu3r(x,z,y)
                           gu3i(x,z,y) = om3i(x,z,y,ll*2-1) - 
     &                          lpfxz2(x,zb)*gu3i(x,z,y)
                        end do
                     end do
                  end do 
               else
                  call stopnow(47432)
               end if
c
c     Put into gur,gui
c
               call putxy(gu3r,gu3i,zbp,ll+3,taur,taui)
            end do
         end if

         if (iles.eq.3) then
c
c     Save also the HPF dtheta/dy
c
            do ith=1,scalar
c
c     Compute HPF filtered dtheta/dy gth3 from dth
c     Get gth3
c
               do y=1,nyp
                  do z=1,mbz
                     do x=1,memnx
                        gth3r(x,z,y) = dthr(x,z,y,ith)
                        gth3i(x,z,y) = dthi(x,z,y,ith)
                     end do
                  end do
               end do

               if (ihighorder.eq.0) then
c
c     Only two-dimensional filtering
c
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Low-pass filter dtheta/dy in x/z and store high-pass filter
c
                           gth3r(x,z,y) = dthr(x,z,y,ith) -
     &                          lpfxz(x,zb)*gth3r(x,z,y)
                           gth3i(x,z,y) = dthi(x,z,y,ith) -
     &                          lpfxz(x,zb)*gth3i(x,z,y)
                        end do
                     end do
                  end do
               else if (ihighorder.eq.1) then
c
c     Do it the iterative way
c
                  do i=1,iord+1
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              ggth3r(x,z,y) = filtxz(x,zb)*gth3r(x,z,y)
                              ggth3i(x,z,y) = filtxz(x,zb)*gth3i(x,z,y)
                           end do
                        end do
                     end do
                     call lpf_y(ggth3r,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     call lpf_y(ggth3i,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              gth3r(x,z,y) = gth3r(x,z,y)-ggth3r(x,z,y)
                              gth3i(x,z,y) = gth3i(x,z,y)-ggth3i(x,z,y)
                           end do
                        end do
                     end do
                  end do
               else if (ihighorder.eq.2) then
c
c     Do it the old way, high-pass filter scalar in y
c
                  do i=1,iord+1
                     call hpf_y(gth3r,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                     call hpf_y(gth3i,gewy1,gewy2,memnx,nyp,mbz,
     &                    diags,gplane)
                  end do
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Store low-pass filtered scalar
c
                           gth3r(x,z,y) = dthr(x,z,y,ith)-gth3r(x,z,y)
                           gth3i(x,z,y) = dthi(x,z,y,ith)-gth3i(x,z,y)
c
c     Low-pass filter scalar in x/z and store high-pass filter
c
                           gth3r(x,z,y) = dthr(x,z,y,ith) -
     &                          lpfxz(x,zb)*gth3r(x,z,y)
                           gth3i(x,z,y) = dthi(x,z,y,ith) -
     &                          lpfxz(x,zb)*gth3i(x,z,y)
                        end do
                     end do
                  end do
               else
                  call stopnow(47432)
               end if
c
c     Put into gur,gui
c
               call putxy(gth3r,gth3i,zbp,5+ith+scalar,gur,gui)
            end do
         end if
         if (iles.eq.3.and.cs.eq.0) then
c
c     Save also the HPF dtheta/dy with secondary HPF
c
            do ith=1,scalar
c
c     Compute HPF filtered dtheta/dy gth3 from dth
c     Get gth3
c
               do y=1,nyp
                  do z=1,mbz
                     do x=1,memnx
                        gth3r(x,z,y) = dthr(x,z,y,ith)
                        gth3i(x,z,y) = dthi(x,z,y,ith)
                     end do
                  end do
               end do

               if (ihighorder.eq.0) then
c
c     Only two-dimensional filtering
c
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gth3r(x,z,y) = dthr(x,z,y,ith) -
     &                          lpfxz2(x,zb)*gth3r(x,z,y)
                           gth3i(x,z,y) = dthi(x,z,y,ith) -
     &                          lpfxz2(x,zb)*gth3i(x,z,y)
                        end do
                     end do
                  end do
               else if (ihighorder.eq.1) then
c
c     Do it the iterative way
c
                  do i=1,iord+1
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              ggth3r(x,z,y) = filtxz2(x,zb)*gth3r(x,z,y)
                              ggth3i(x,z,y) = filtxz2(x,zb)*gth3i(x,z,y)
                           end do
                        end do
                     end do
                     call lpf_y(ggth3r,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     call lpf_y(ggth3i,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     do y=1,nyp
                        do z=1,mbz
                           do x=1,memnx
                              gth3r(x,z,y) = gth3r(x,z,y)-ggth3r(x,z,y)
                              gth3i(x,z,y) = gth3i(x,z,y)-ggth3i(x,z,y)
                           end do
                        end do
                     end do
                  end do
               else if (ihighorder.eq.2) then
c
c     Do it the old way, high-pass filter velocities in y
c
                  do i=1,iord+1
                     call hpf_y(gth3r,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                     call hpf_y(gth3i,gewy12,gewy22,memnx,nyp,mbz,
     &                    diags2,gplane)
                  end do
                  do y=1,nyp
                     do z=1,mbz
                        do x=1,memnx
c
c     Store low-pass filtered velocities
c
                           gth3r(x,z,y) = dthr(x,z,y,ith)-gth3r(x,z,y)
                           gth3i(x,z,y) = dthi(x,z,y,ith)-gth3i(x,z,y)
c
c     Low-pass filter velocities in x/z and store high-pass filter
c
                           gth3r(x,z,y) = dthr(x,z,y,ith) -
     &                          lpfxz2(x,zb)*gth3r(x,z,y)
                           gth3i(x,z,y) = dthi(x,z,y,ith) -
     &                          lpfxz2(x,zb)*gth3i(x,z,y)
                        end do
                     end do
                  end do
               else
                  call stopnow(47432)
               end if
c
c     Put into taur,taui
c
               call putxy(gth3r,gth3i,zbp,6+ith+scalar,taur,taui)
            end do
         end if
      end do

      end subroutine prhs
