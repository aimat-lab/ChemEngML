c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/steadywallrough.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine steadywallrough(bu1,bu2,wallur,wallui,
     &     wallwr,wallwi,v_wall,wlvr,wlvi,
     &     prexn,prey,prezn,presn,prean,
     &     alfa,beta,zl,xsc,zsc,dstar,xst,xlen,delx,
     &     h_rough,xstart,xend,xrise,xfall,zfunc,q,psi,tpsi,taylor4,
     &     rghfil,roughfile,monitor,monfile,my_node)
c
c     Calculates the boundary condition at y=0 resulting from 
c     surface roughness. The latter is modelled in terms of a Taylor
c     series (projection of the no-slip conditions from the desired
c     bump contour to the "wall" y=0)
c
c     Taylor series is truncated at 1st order or at 4th order (if
c     taylor4==T)
c
c     The projected wall-boundary conditions are calculated using 
c     the BASE FLOW (bu1,bu2). Hence, this routine is called only once
c     before the time-step loop begins.
c
c     CALLED by bla.f (if wbci==-1)
c
c     CALLS vchbf, vchbb,dcheb,rdcheb,fft2df
c
      implicit none

      include 'par.f'

#ifdef MPI
      include 'mpif.h'
#endif

c
c     Parameters
c
      integer nxm,nxn,nzn,lenx
      parameter(nxm=nxp/2+1,nxn=nx/2+1,nzn=nz/2+1)
      parameter (lenx=nx/8)

      real pi
      parameter (pi = 3.1415926535897932385)
c
c     Global variables
c
      real bu1(nxm,nyp,3),bu2(nxm,nyp,3)
      
      integer xst,xlen,zfunc,my_node

      real zl,xsc,zsc,q,psi,tpsi,dstar,delx
      real h_rough,xstart,xend,xrise,xfall 

      real prexn(nx+15),prezn(nz*2+15),presn(nz+2+15)
      real prean(nz*3/4+15),prey(nyp*2+15),w(nxm,nyp)

      real alfa(nx/2*mbz),beta(nz)

      logical v_wall,taylor4,rghfil,monitor
      character(80) roughfile,monfile
c
c     Output
c
      real wallur(nxm,memnz),wallui(nxm,memnz)
      real wlvr(nxm,memnz),wlvi(nxm,memnz)
      real wallwr(nxm,memnz),wallwi(nxm,memnz)
c
c     Local variables
c
      integer x,xsh,xind,y,z,zb,i

      real wr(nxn*mby,nz),wi(nxn*mby,nz)

      real lwallur(nxn*mby,nz),lwallui(nxn*mby,nz)
      real lwallvr(nxn*mby,nz),lwallvi(nxn*mby,nz)
      real lwallwr(nxn*mby,nz),lwallwi(nxn*mby,nz)
      
      real hfunc(lenx,nz,2)
      real xc11,xc22,fx1(lenx),fx2(lenx)
      real zc,zshift1,zshift2,step,tmpr1,tmpr2
      real delty,norm,hd

      real app1(nyp),app2(nyp),app21(nyp),app22(nyp)
      real app31(nyp),app32(nyp),app41(nyp),app42(nyp)
      real du1wall(lenx,3),du2wall(lenx,3)
      real d2u1wall(lenx,3),d2u2wall(lenx,3)
      real d3u1wall(lenx,3),d3u2wall(lenx,3)
      real d4u1wall(lenx,3),d4u2wall(lenx,3)
      
     
      if (my_node.eq.0) then
         write(*,*)'--> Wall-roughness B.C.s set (from initial flow).'
         if (psi.eq.90) then
            write(*,*)'! Please set psi < 90 degrees !'
            stop
         end if
      end if
c
c     Angle by which roughness crests are turned
c
      psi  = pi*psi/180
      tpsi = tan(psi)
c
c     Initialize the wall B.C.s:
c
      do z=1,nz
         do x=1,nxn
            lwallur(x,z)=0.
            lwallui(x,z)=0.
            lwallvr(x,z)=0.
            lwallvi(x,z)=0.
            lwallwr(x,z)=0.
            lwallwi(x,z)=0.
         end do
      end do
c
c     Calculation of the vel. derivatives at y=0
c
      delty=2./real(nyp-1)      ! y step width (internal scaling)

      do i=1,3
         do x=1,xlen 
            xind=x+xst-1
            do y=1,nyp
               app1(y)=bu1(xind,y,i) 
               app2(y)=bu2(xind,y,i)
            end do
            
            call vchbf(app1,w,nyp,1,1,1,prey)
            call vchbf(app2,w,nyp,1,1,1,prey)
c
c     Normalise
c
            do y=1,nyp
               app1(y)=app1(y)*delty
               app2(y)=app2(y)*delty
            end do
            
            call rdcheb(app1,nyp,1,1) !1st deriv. of vel. profs.
            call rdcheb(app2,nyp,1,1)
            
            if (taylor4) then
               
               call dcheb(app21,app1,nyp,1,1) ! 2nd derivative
               call dcheb(app22,app2,nyp,1,1)
               
               call dcheb(app31,app21,nyp,1,1) ! 3rd derivative
               call dcheb(app32,app22,nyp,1,1)
               
               call dcheb(app41,app31,nyp,1,1) ! 4th derivative
               call dcheb(app42,app32,nyp,1,1)
               
               call vchbb(app21,w,nyp,1,1,1,prey)
               call vchbb(app22,w,nyp,1,1,1,prey)
               
               call vchbb(app31,w,nyp,1,1,1,prey)
               call vchbb(app32,w,nyp,1,1,1,prey)
               
               call vchbb(app41,w,nyp,1,1,1,prey)
               call vchbb(app42,w,nyp,1,1,1,prey)
               
               d2u1wall(x,i)=app21(nyp) !2nd derivative
               d2u2wall(x,i)=app22(nyp)
               
               d3u1wall(x,i)=app31(nyp) !3rd derivative
               d3u2wall(x,i)=app32(nyp)
               
               d4u1wall(x,i)=app41(nyp) !4th derivative
               d4u2wall(x,i)=app42(nyp)
            end if
            call vchbb(app1,w,nyp,1,1,1,prey)
            call vchbb(app2,w,nyp,1,1,1,prey)
            
            du1wall(x,i)=app1(nyp) !1st derivative
            du2wall(x,i)=app2(nyp)    
         end do
      end do
c
c     Bump function and projected BCs
c
      xsh = xst-0.5*nxn
      if (xsh.le.0) xsh = xsh+nxn      

      do z=1,nz
         if (zfunc.eq.0) then
            zc = zl
            q = 0.25
         else if (zfunc.eq.1) then
            zc=zl*real(z-nzn)/real(nz)+zsc 
         else
            write(*,*) 'Sorry, only zfunc==0 or 1 are supported.'
            stop
         end if

         do x=1,xlen
            xind=x+xsh-1

            xc11=xstart+(2*x-2)*delx+xsc
            fx1(x)=step((xc11-xstart)/xrise)-
     &           step((xc11-xend)/xfall+1)
            
            xc22 = xstart+(2*x-1)*delx+xsc
            fx2(x)=step((xc22-xstart)/xrise)-
     &           step((xc22-xend)/xfall+1)

            zshift1 = tpsi*(xc11-xstart)
            zshift2 = tpsi*(xc22-xstart)
            tmpr1  = sin(q*2*pi*(zc-zshift1)/zl)
            tmpr2  = sin(q*2*pi*(zc-zshift2)/zl) 
                       
            hfunc(x,z,1)=h_rough*fx1(x)*tmpr1
            hfunc(x,z,2)=h_rough*fx2(x)*tmpr2

            lwallur(xind,z)=-hfunc(x,z,1)*du1wall(x,1) !"roughness
            lwallui(xind,z)=-hfunc(x,z,2)*du2wall(x,1) !projection"

            lwallwr(xind,z)=-hfunc(x,z,1)*du1wall(x,3)
            lwallwi(xind,z)=-hfunc(x,z,2)*du2wall(x,3)

            if (taylor4) then !Take 2nd, 3rd, 4th deriv. into account
               lwallur(xind,z)=lwallur(xind,z)-
     &              0.5*(hfunc(x,z,1)**2)*d2u1wall(x,1)-
     &              (1./6.)*(hfunc(x,z,1)**3)*d3u1wall(x,1)-
     &              (1./24.)*(hfunc(x,z,1)**4)*d4u1wall(x,1) 
               lwallui(xind,z)=lwallui(xind,z)-
     &              0.5*(hfunc(x,z,2)**2)*d2u2wall(x,1)-
     &              (1./6.)*(hfunc(x,z,2)**3)*d3u2wall(x,1)-
     &              (1./24.)*(hfunc(x,z,2)**4)*d4u2wall(x,1)

               lwallwr(xind,z)=lwallwr(xind,z)-
     &              0.5*(hfunc(x,z,1)**2)*d2u1wall(x,3)-
     &              (1./6.)*(hfunc(x,z,1)**3)*d3u1wall(x,3)-
     &              (1./24.)*(hfunc(x,z,1)**4)*d4u1wall(x,3) 
               lwallwi(xind,z)=lwallwi(xind,z)-
     &              0.5*(hfunc(x,z,2)**2)*d2u2wall(x,3)-
     &              (1./6.)*(hfunc(x,z,2)**3)*d3u2wall(x,3)-
     &              (1./24.)*(hfunc(x,z,2)**4)*d4u2wall(x,3)  
            end if

            if (v_wall) then
               lwallvr(xind,z)=-hfunc(x,z,1)*du1wall(x,2)
               lwallvi(xind,z)=-hfunc(x,z,2)*du2wall(x,2)
               if (taylor4) then !Take 2nd, 3rd, 4th deriv. into account
                  lwallvr(xind,z)=lwallvr(xind,z)-
     &                 0.5*(hfunc(x,z,1)**2)*d2u1wall(x,2)-
     &                 (1./6.)*(hfunc(x,z,1)**3)*d3u1wall(x,2)-
     &                 (1./24.)*(hfunc(x,z,1)**4)*d4u1wall(x,2) 
                  lwallvi(xind,z)=lwallvi(xind,z)-
     &                 0.5*(hfunc(x,z,2)**2)*d2u2wall(x,2)-
     &                 (1./6.)*(hfunc(x,z,2)**3)*d3u2wall(x,2)-
     &                 (1./24.)*(hfunc(x,z,2)**4)*d4u2wall(x,2)
               end if
             end if
          end do
       end do
c
c     Write u,v,w values (at wall, where bump height max.) on screen:
c
       if (my_node .eq. 0) then
          x = 0.5*(2*xsh+xlen-0.25)
          if (zfunc.eq.1) then
             if (q.ne.0)   z=nz/(4*q)+nzn
          else
             z = 1
          end if
          write(*,*)'--> u,v,w|0 at h_max:'
          write(*,118) lwallur(x,z),lwallvr(x,z),lwallwr(x,z)
c
c     Write the roughness function (where h(x,z)==hmax) into a file?
c
          if (rghfil) then
             hd=0.
         
             open(unit=122,status='replace',file=roughfile,
     &            form='formatted')
             
             do x=1,xst-1
                xc11=(2*x-2)*delx+xsc
                xc22=(2*x-1)*delx+xsc
                write(122,119) xc11/dstar,(hd,  z=1,nz)
                write(122,119) xc22/dstar,(hd,  z=1,nz)
             end do
             
             do x=1,xlen
                xc11=xstart+(2*x-2)*delx+xsc
                xc22=xstart+(2*x-1)*delx+xsc
                write(122,119) xc11/dstar,(hfunc(x,z,1)/dstar, 
     &               z=1,nz)
                write(122,119) xc22/dstar,(hfunc(x,z,2)/dstar,
     &               z=1,nz)
             end do
             
             do x=xst+xlen-1,nxn
                xc11=(2*x-2)*delx+xsc
                xc22=(2*x-1)*delx+xsc
                write(122,119) xc11/dstar,(hd, z=1,nz) 
                write(122,119) xc22/dstar,(hd, z=1,nz)
             end do
             
             close(122)
          end if
       end if
       
 118   format(3F22.16)
 119   format(9F12.6)
c
c     Transform to Fourier space
c
      call sfft2df(lwallur,lwallui,.true.,1,prexn,prezn,presn,prean,
     &      wr,wi)
      call sfft2df(lwallwr,lwallwi,.false.,1,prexn,prezn,presn,prean,
     &     wr,wi)
      if (v_wall) call sfft2df(lwallvr,lwallvi,.true.,1,
     &     prexn,prezn,presn,prean,wr,wi)
c
c     Normalization
c
      norm=real(nx*nz)
      do z=1,nz/2
         do x=1,nx/2
            lwallur(x,z)=lwallur(x,z)/norm
            lwallui(x,z)=lwallui(x,z)/norm
            lwallwr(x,z)=lwallwr(x,z)/norm
            lwallwi(x,z)=lwallwi(x,z)/norm
         end do
      end do 

      if (nfzsym.eq.0) then
         do z=nz/2+1,nz
            do x=1,nx/2
               lwallur(x,z)=lwallur(x,z)/norm
               lwallui(x,z)=lwallui(x,z)/norm
               lwallwr(x,z)=lwallwr(x,z)/norm
               lwallwi(x,z)=lwallwi(x,z)/norm
            end do
         end do
      end if
c
c     If bump condition for v is also prescribed
c
      if (v_wall) then
         do z=1,nz/2
            do x=1,nx/2
               lwallvr(x,z)=lwallvr(x,z)/norm
               lwallvi(x,z)=lwallvi(x,z)/norm
            end do
         end do 

         if (nfzsym.eq.0) then
            do z=nz/2+1,nz
               do x=1,nx/2
                  lwallvr(x,z)=lwallvr(x,z)/norm
                  lwallvi(x,z)=lwallvi(x,z)/norm
               end do
            end do
         end if
      end if
c
c     From this point the subroutine is parallelized
c     Obtain Dv and eta
c
      do z=1,memnz
         zb=my_node*memnz+z
c
c     Re{Dv},Im{Dv},Re{eta} and Im{eta}
c
         do x=1,nx/2
            wallur(x,z)=alfa(x)*lwallui(x,zb)+beta(zb)*lwallwi(x,zb)
            wallui(x,z)=-alfa(x)*lwallur(x,zb)-beta(zb)*lwallwr(x,zb)
            wallwr(x,z)=alfa(x)*lwallwi(x,zb)-beta(zb)*lwallui(x,zb)
            wallwi(x,z)=-alfa(x)*lwallwr(x,zb)+beta(zb)*lwallur(x,zb)
         end do
      end do 
c
c     Re{v},Im{v}
c
      if (v_wall) then
         do z=1,memnz
            zb=my_node*memnz+z
            do x=1,nx/2
               wlvr(x,z)=lwallvr(x,zb)
               wlvi(x,z)=lwallvi(x,zb) 
            end do
         end do
      end if
c
c     In (1,1), i.e. (alfa,beta)=(0,0) the values of u and w
c     and not Dv and eta are put back (not necessary for v)
c
      if (my_node.eq.0) then
         wallur(1,1)=lwallur(1,1)
         wallwr(1,1)=lwallwr(1,1)
      end if
c
c     Roughness BCs written to a file? (done in updatewallrough);
c     Here, the file is opened.
c
      if (monitor) then
         if (my_node .eq. 0) then
            open(unit=120,status='replace',file=monfile,
     &           form='formatted')
         end if
      end if

      end subroutine steadywallrough
