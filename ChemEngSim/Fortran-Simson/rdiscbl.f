c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/rdiscbl.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine rdiscbl(ur,ui,u0low,u0upp,w0low,w0upp,du0upp,re,
     &     pr,xl,zl,t,xs,dstar,fltype,bstart,bslope,rlam,m1,spanv,
     &     spat,namnin,varsiz,m,boxr,boxi,w3,urx,nxtmp,my_node_world)
c
c     Reads m variables from file namnin and puts then into ur
c
      implicit none

      include 'par.f'
#ifdef MPI
      include 'mpif.h'
#endif
      character*80 namnin
      integer m,fltype,nxtmp
      real ur(memnx,memny,memnz,memnxyz),ui(memnx,memny,memnz,memnxyz)
      real u0low,u0upp,w0low,w0upp,du0upp,m1(scalar)
      real re,pr(scalar),xl,zl,t,xs,dstar,bstart,bslope,spanv,rlam
      logical pou,varsiz,spat
      real urx(nx)
      real boxr(nx/2,mbz,nyp),boxi(nx/2,mbz,nyp)
      real w3(nx/2,mbz,nyp)
      character(len=5) ch

      integer x,y,z,i,zb,zs,nxz,nxr,zu,nprocin
      integer nxin,nypin,nzcin,nfzsin
      real preyin(2*nyp+15),prey(2*nyp+15)

      integer my_node_world,zb_t,node_t,zbb,xb_t,ii

      integer iotype,kode

#ifdef MPI   
      integer ierror
#endif


      if (index(namnin,'NONE').eq.1.and.len_trim(namnin).eq.4) then
         re = 16403.
         pou = .false.
         xl = 6.2831853070000
         zl = 3.141592654000
         t = 0.
         xs = 0.
         
         nxin = nx
         nypin = nyp
         nzcin = nz
         nfzsin = 0
         
         fltype= 1
         rlam  = 0.
         dstar = 1.
         spanv = 0.
         
         ur = 0.
         ui = 0.
         return
         
      end if
 

c
c     Determine read/write types
c      0:  normal 
c      1:  multiple files (XXX.uu)
c      2:  direct access  (XXX.vv)
c      3:  MPI-IO         (XXX.ww)
c
      iotype = 0
      if (namnin(len_trim(namnin)-2:len_trim(namnin)).eq.".uu") then
         iotype = 1
      end if
      if (namnin(len_trim(namnin)-2:len_trim(namnin)).eq.".vv") then
         iotype = 2
      end if
      if (namnin(len_trim(namnin)-2:len_trim(namnin)).eq.".ww") then
         iotype = 3
      end if

      if (iotype.eq.0) then

c
c     Open input file, read file header
c
      open(unit=12,file=namnin,status='old',form='unformatted')
c      
c     Description of parameters:
c     re      Reynolds number (rescaled, real: re*dstar)
c     pr      Prandtl number (only read for scalar=1)
c     pou     Not used anymore, always set to false
c     xl      streamwise length (rescaled, real: xl/dstar)
c     zl      spanwise width (rescaled, real: xl/dstar)
c     t       time (rescaled, real: t/dstar)
c     xs      shift since t zero (0 for spatial cases)
c     nxin    collocation points in x
c     nypin   collocation points in y
c     nzcin   collocation points in z
c     nfzsin  symmetry flag (0: no symmetry)
c     fltype  flow type
c            -2: temporal Falkner-Skan-Cooke boundary layer
c            -1: temporal Falkner-Skan boundary layer
c             1: temporal Poiseuille flow
c             2: temporal Couette flow
c             3: temporal Blasius boundary layer
c             4: spatial Poiseuille flow
c             5: spatial Couette flow
c             6: spatial Blasius
c             7: spatial Falkner-Skan
c             8: spatial Falkner-Skan-Cooke
c             9: spatial parallel boundary layer
c     dstar   length scale (=2/yl)  
c
      if (scalar.ge.1) then
c
c     Read Prandtl number
c
         m1 = 0.
         read(12,err=2001) re,pou,xl,zl,t,xs,(pr(i),m1(i),i=1,scalar)

         do i=1,scalar
            if (m1(i).eq.0.or.abs(m1(i)-0.5).lt.1.e-13) then
            else
               if (my_node_world.eq.0) then
                  write(*,*) 'Variation of temperature profile m1'
                  write(*,*) 'not implemented.',m1(i),' Scalar no.',i
               end if
               call stopnow(6654)
            end if
         end do
      else
c
c     Ignore Prandtl number and set dummy value
c
         read(12,err=2001) re,pou,xl,zl,t,xs
      end if
      read(12) nxin,nypin,nzcin,nfzsin
      fltype= 0
      rlam  = 0.
      dstar = 0.
      spanv = 0.
      read(12,err=1010) fltype,dstar
 1010 continue
c
c     Write flow type to standard output
c
      if (my_node_world.eq.0) then
         write(*,*) 'Reading initial file'
         write(*,*) '  filename    : ',trim(namnin)
         write(*,*) '  m           : ',m
         if (scalar.ge.1) then
            do i=1,scalar
               write(*,*) '  Re        : ',re*dstar,
     &              ' Pr :',pr(i),' m1 :',m1(i)
            end do
         else
            write(*,*) '  Re          : ',re*dstar
         end if
         if (fltype.eq.-2) write(*,*) 
     &        '  fltype      : -2. ',
     &        'Temporal Falkner-Skan-Cooke boundary layer'
         if (fltype.eq.-1) write(*,*) 
     &        '  fltype      : -1. Temporal Falkner-Skan boundary layer'
         if (fltype.eq.0) write(*,*) 
     &        '  fltype      : 0. No base flow'
         if (fltype.eq.1) write(*,*) 
     &        '  fltype      : 1. Temporal Poiseuille flow'
         if (fltype.eq.2) write(*,*) 
     &        '  fltype      : 2. Temporal Couette flow'
         if (fltype.eq.3) write(*,*) 
     &        '  fltype      : 3. Temporal Blasius boundary layer'
         if (fltype.eq.4) write(*,*) 
     &        '  fltype      : 4. Spatial Poiseuille flow'
         if (fltype.eq.5) write(*,*) 
     &        '  fltype      : 5. Spatial Couette flow'
         if (fltype.eq.6) write(*,*) 
     &        '  fltype      : 6. Spatial Blasius boundary layer'
         if (fltype.eq.7) write(*,*) 
     &        '  fltype      : 7. Spatial Falkner-Skan boundary layer'
         if (fltype.eq.8) write(*,*) 
     &        '  fltype      : 8. ',
     &        'Spatial Falkner-Skan-Cooke boundary layer'
         if (fltype.eq.9) write(*,*) 
     &        '  fltype      : 9. Spatial parallel boundary layer'
      end if

      if (fltype.lt.-3.or.fltype.gt.9.or.fltype.eq.0) then
         write(*,*) 'The input file does not contain'
         write(*,*) 'the correct type of flow, now: ',fltype
         stop
      end if

c      if (spat.and.fltype.lt.4.or.(.not.spat.and.fltype.ge.4)) then
      if (spat) then
         if (fltype.eq.-2.or.fltype.eq.-1.or.fltype.eq.1.or.
     &       fltype.eq. 2.or.fltype.eq. 3) then
            write(*,*) 'Conflicting variables. Spatial flow but '//
     &                 'temporal flow type.'
            write(*,*) 'Change spat in bla.i or use other flow field.'
            stop
         end if
      else
         if (fltype.eq.4.or.fltype.eq.5.or.fltype.eq.6.or.
     &       fltype.eq.7.or.fltype.eq.8.or.fltype.eq.9) then
            write(*,*) 'Conflicting variables. Temporal flow but '//
     &                 'spatial flow type.'
            write(*,*) 'Change spat in bla.i or use other flow field.'
            stop
         end if
      end if
c
c     Ensure that dstar is correctly defined in channel and Couette flow cases (=1)
c
      if (fltype.eq.1.or.fltype.eq.2.or.
     &    fltype.eq.4.or.fltype.eq.5) then
          dstar=1.
      end if
c
c     Read additional info for specific flow types
c
      if (fltype.eq.-1) read(12) rlam
      if (fltype.eq.-2) read(12) rlam,spanv
      if (fltype.eq.6) then
         read(12) bstart,bslope
         rlam=0.0
         spanv=0.0
      end if
      if (fltype.ge.7) read(12) bstart,bslope,rlam,spanv
      

      if (my_node_world.gt.0) then
         close(unit=12)
      end if

      nxz=nx/2*mbz
      nxtmp=nxin+nfxd*nxin/2
c
c     Check file info
c
      if (nxin.ne.nx.or.nypin.ne.nyp.or.nzcin.ne.nzc.or.
     &     nfzsin.ne.nfzsym) then
         if (my_node_world.eq.0) then
            write(*,*) 'Input file has a size other than program'
            write(*,'(a,4i5)') '   File parameters:    ',
     &           nxin,nypin,nzcin,nfzsin
            write(*,'(a,4i5)') '   Program parameters: ',
     &           nx,nyp,nzc,nfzsym
            write(*,*) '   (nx,nyp,nzc,nfzsym)'
         end if
         if (.not.varsiz) then
            call stopnow(453565)
         end if
         if (nypin.gt.nyp) then
            if (my_node_world.eq.0) then
               write(*,*) 'Resolution cannot be reduced in y-direction'
            end if
            call stopnow(326576)
         end if
         if (nfzsin.ne.nfzsym) then
            if (my_node_world.eq.0) then
               write(*,*) 'Symmetry cannot be changed'
            end if
            call stopnow(6764)
         end if
      end if
     
      nxr=min(nx,nxin)
      if (nypin.lt.nyp) then
         call vcosti(nypin,preyin,0)
      end if
      call vcosti(nyp,prey,0)
     
      do i=1,m
         if (i.ge.4) then
c     
c     This is the scalar
c     
            ii = 8+pressure+3*(i-4)
         else
c     
c     These are the velocities
c     
            ii = i
         end if
         
         do zb=1,nzc,mbz
            if (my_node_world.eq.0) then
               write(*,*) 'reading ',i,zb
            end if

            do z=zb,zb+mbz-1

               zb_t=mod(zb-1,memnz)+1

               do y=1,nypin
c
c     If expanding in z-direction and record is new, pad with zeroes
c
                  if ((nfzsym.eq.0.and.z.gt.(nzcin+1)/2
     &                 .and.z.le.nz-nzcin/2)
     &                 .or.(nfzsym.eq.1.and.z.gt.nzcin)) then
                     do x=1,nx
                        urx(x) = 0.0
                     end do
                  else
c     
c     Read an x-vector (fixed y and z)
c     
                     if (my_node_world.eq.0) then
                        read(12) (urx(x),x=1,nxr)
                     end if
                  end if
#ifdef MPI
                  if (nproc.gt.1) then
                     call mpi_barrier(mpi_comm_world,ierror)
                     call mpi_bcast(urx,nxr,mpi_double_precision,0,
     &                    mpi_comm_world,ierror)
c                     call mpi_barrier(mpi_comm_world,ierror)
                  end if
#endif

c
c     Set odd-ball mode in z to zero 
c     (new grid: nz/2+1=nz-nz/2+1, old grid: nz-nzin/2+1)
c
                  if (z.eq.nz-nzcin/2+1) then
                     do x=1,nx
                        urx(x) = 0.0
                     end do
                  end if


c
c     If expanding in x-direction, padding with zeros
c
                  do x=nxin+1,nx
                     urx(x)=0.0
                  end do                     
c
c     Copy x-vector into correct processor
c     
                  do x=1,nx/2
                     node_t = (zb-1)/memnz*nprocx+(x-1)/memnx
                     xb_t = mod(x-1,memnx)+1
                     if (my_node_world.eq.node_t) then
                        ur(xb_t,y,zb_t,ii) = urx(2*x-1)
                        ui(xb_t,y,zb_t,ii) = urx(2*x)
                     end if
                  end do
c     
c     Set boundary velocities to mean in the streamwise direction (0/0-mode)
c     (could be not accurate for varying freestream velocity)
c     If will however be reset in bflow (simulations with fringe), but then
c     according to the value at the inflow (x=0)
c
                  if (zb.eq.1.and.y.eq.1    .and.i.eq.1) u0upp=urx(1)
                  if (zb.eq.1.and.y.eq.1    .and.i.eq.3) w0upp=urx(1)
                  if (zb.eq.1.and.y.eq.nypin.and.i.eq.1) u0low=urx(1)
                  if (zb.eq.1.and.y.eq.nypin.and.i.eq.3) w0low=urx(1)
                  
               end do
c
c     If contracting in z-direction, then skip records on file
c
               if (z.eq.nz/2.or.nz.eq.1) then
                  do zs=nzc+1,nzcin
                     if (my_node_world.eq.0) then
                        do y=1,nypin
                           read(12) (urx(x),x=1,nxr)
                        end do
                     end if
                  end do
               end if
               
            end do
         end do
c
c     If expanding in y-direction, then Chebyshev transform, expand and return
c     Here, we do it directly on ur,ui after reading in
c
         if (nypin.lt.nyp) then
            do z=1,memnz
               call vchbf(ur(1,1,z,ii),w3,nypin,memnx,memnx,1,preyin)
               call vchbf(ui(1,1,z,ii),w3,nypin,memnx,memnx,1,preyin)
               do y=1,nypin
                  do x=1,memnx
                     ur(x,y,z,ii) = ur(x,y,z,ii)*(2./real(nypin-1))
                     ui(x,y,z,ii) = ui(x,y,z,ii)*(2./real(nypin-1))
                  end do
               end do
               do y=nypin+1,nyp
                  do x=1,memnx
                     ur(x,y,z,ii) = 0.
                     ui(x,y,z,ii) = 0.
                  end do
               end do
               call vchbb(ur(1,1,z,ii),w3,nyp,memnx,memnx,1,prey)
               call vchbb(ui(1,1,z,ii),w3,nyp,memnx,memnx,1,prey)
            end do
         end if
         
      end do
c
c     Close file
c
      if (my_node_world.eq.0) then
         close(unit=12)
      end if

      else if (iotype.eq.1) then


         open(unit=12,file=trim(namnin)//'-00000',status='old',
     &        form='unformatted')
c      
c     Description of parameters:
c     re      Reynolds number (rescaled, real: re*dstar)
c     pr      Prandtl number (only read for scalar=1)
c     pou     Not used anymore, always set to false
c     xl      streamwise length (rescaled, real: xl/dstar)
c     zl      spanwise width (rescaled, real: xl/dstar)
c     t       time (rescaled, real: t/dstar)
c     xs      shift since t zero (0 for spatial cases)
c     nxin    collocation points in x
c     nypin   collocation points in y
c     nzcin   collocation points in z
c     nfzsin  symmetry flag (0: no symmetry)
c     fltype  flow type
c            -2: temporal Falkner-Skan-Cooke boundary layer
c            -1: temporal Falkner-Skan boundary layer
c             1: temporal Poiseuille flow
c             2: temporal Couette flow
c             3: temporal Blasius boundary layer
c             4: spatial Poiseuille flow
c             5: spatial Couette flow
c             6: spatial Blasius
c             7: spatial Falkner-Skan
c             8: spatial Falkner-Skan-Cooke
c             9: spatial parallel boundary layer
c     dstar   length scale (=2/yl)  
c
         if (scalar.ge.1) then
c
c     Read Prandtl number
c
            read(12,err=2001) re,pou,xl,zl,t,xs,(pr(i),m1(i),i=1,scalar)
            
            do i=1,scalar
               if (m1(i).eq.0.or.abs(m1(i)-0.5).lt.1.e-13) then
               else
                  if (my_node_world.eq.0) then
                     write(*,*) 'Variation of temperature profile m1'
                     write(*,*) 'not implemented.',m1(i),' Scalar no.',i
                  end if
                  call stopnow(6654)
               end if
            end do
         else
c
c     Ignore Prandtl number and set dummy value
c
            read(12,err=2001) re,pou,xl,zl,t,xs
         end if
         read(12) nxin,nypin,nzcin,nfzsin
         fltype= 0
         rlam  = 0.
         dstar = 0.
         spanv = 0.
         read(12,err=1011) fltype,dstar
 1011    continue
c
c     Write flow type to standard output
c     
         if (my_node_world.eq.0) then
            write(*,*) 'Reading initial (multiple) file'
            write(*,*) '  filename    : ',trim(namnin)
            write(*,*) '  m           : ',m
            if (scalar.ge.1) then
               do i=1,scalar
                  write(*,*) '  Re        : ',re*dstar,
     &                 ' Pr :',pr(i),' m1 :',m1(i)
               end do
            else
               write(*,*) '  Re          : ',re*dstar
            end if
            if (fltype.eq.-2) write(*,*) 
     &           '  fltype      : -2. ',
     &           'Temporal Falkner-Skan-Cooke boundary layer'
            if (fltype.eq.-1) write(*,*) 
     &           '  fltype      : -1. Temporal Falkner-Skan '//
     &           'boundary layer'
            if (fltype.eq.0) write(*,*) 
     &           '  fltype      : 0. No base flow'
            if (fltype.eq.1) write(*,*) 
     &           '  fltype      : 1. Temporal Poiseuille flow'
            if (fltype.eq.2) write(*,*) 
     &           '  fltype      : 2. Temporal Couette flow'
            if (fltype.eq.3) write(*,*) 
     &           '  fltype      : 3. Temporal Blasius boundary layer'
            if (fltype.eq.4) write(*,*) 
     &           '  fltype      : 4. Spatial Poiseuille flow'
            if (fltype.eq.5) write(*,*) 
     &           '  fltype      : 5. Spatial Couette flow'
            if (fltype.eq.6) write(*,*) 
     &           '  fltype      : 6. Spatial Blasius boundary layer'
            if (fltype.eq.7) write(*,*) 
     &           '  fltype      : 7. Spatial Falkner-Skan '//
     &           'boundary layer'
            if (fltype.eq.8) write(*,*) 
     &           '  fltype      : 8. ',
     &           'Spatial Falkner-Skan-Cooke boundary layer'
            if (fltype.eq.9) write(*,*) 
     &           '  fltype      : 9. Spatial parallel boundary layer'
         end if
         
         if (fltype.lt.-3.or.fltype.gt.9.or.fltype.eq.0) then
            write(*,*) 'The input file does not contain'
            write(*,*) 'the correct type of flow, now: ',fltype
            stop
         end if
         
c      if (spat.and.fltype.lt.4.or.(.not.spat.and.fltype.ge.4)) then
         if (spat) then
            if (fltype.eq.-2.or.fltype.eq.-1.or.fltype.eq.1.or.
     &           fltype.eq. 2.or.fltype.eq. 3) then
               write(*,*) 'Conflicting variables. Spatial flow but '//
     &              'temporal flow type.'
               write(*,*) 'Change spat in bla.i or '//
     &              'use other flow field.'
               stop
            end if
         else
            if (fltype.eq.4.or.fltype.eq.5.or.fltype.eq.6.or.
     &           fltype.eq.7.or.fltype.eq.8.or.fltype.eq.9) then
               write(*,*) 'Conflicting variables. Temporal flow but '//
     &              'spatial flow type.'
               write(*,*) 'Change spat in bla.i or use '//
     &              'other flow field.'
               stop
            end if
         end if
c     
c     Ensure that dstar is correctly defined in channel and Couette flow cases (=1)
c
         if (fltype.eq.1.or.fltype.eq.2.or.
     &        fltype.eq.4.or.fltype.eq.5) then
            dstar=1.
         end if
c
c     Read additional info for specific flow types
c
         if (fltype.eq.-1) read(12) rlam
         if (fltype.eq.-2) read(12) rlam,spanv
         if (fltype.eq.6) then
            read(12) bstart,bslope
            rlam=0.0
            spanv=0.0
         end if
         if (fltype.ge.7) read(12) bstart,bslope,rlam,spanv
         read(12) nprocin

         if (my_node_world.eq.0) then
            write(*,*) 'expecting ',nprocin,' fields.'
         end if
c
c     Read wall velocities (if present)
c
         read(12,end=1444,err=1444) u0low,u0upp,w0low,w0upp,du0upp
         if (my_node_world.eq.0) then
            write(*,*) 'reading u0low... from file'
         end if
         goto 1445

 1444    continue
         u0low  = 0.
         u0upp  = 0.
         w0low  = 0.
         w0upp  = 0.
         du0upp = 0.
         if (my_node_world.eq.0) then
            write(*,*) 'setting u0low... to default values'
         end if

 1445    continue
c
c     Close file
c
         close(12)


         close(12)

         
         nxz=nx/2*mbz
         nxtmp=nxin+nfxd*nxin/2
c     
c     Check file info
c     
         if (nxin.ne.nx.or.nypin.ne.nyp.or.nzcin.ne.nzc.or.
     &        nfzsin.ne.nfzsym) then
            if (my_node_world.eq.0) then
               write(*,*) 'Input file has a size other than program'
               write(*,'(a,4i5)') '   File parameters:    ',
     &              nxin,nypin,nzcin,nfzsin
               write(*,'(a,4i5)') '   Program parameters: ',
     &              nx,nyp,nzc,nfzsym
               write(*,*) '   (nx,nyp,nzc,nfzsym)'
            end if
            call stopnow(453565)
         end if
         if (nproc.ne.nprocin) then
            if (my_node_world.eq.0) then
               write(*,*) 'nproc not same as in header file'
               write(*,*) nproc,nprocin
            end if
            call stopnow(434432)
         end if
      

         write(ch,'(i5.5)') my_node_world+1
         open(unit=12,file=trim(namnin)//'-'//ch,status='old',
     &        form='unformatted')
         do i=1,m
c     
c     Choose which field to read
c     
            if (i.ge.4) then
c     
c     This is the scalar
c     
               ii = 8+pressure+3*(i-4)
            else
c     
c     These are the velocities
c     
               ii = i
            end if
            read(12) ur(:,:,:,ii),ui(:,:,:,ii)
         end do
         close(12)
         
      else if (iotype.eq.2) then
         write(*,*) 'Direct-access files not yet implemented'
         call stopnow(95885)
      else if (iotype.eq.3) then
         write(*,*) 'MPI-IO not yet implemented'
         call stopnow(95886)
      end if






c
c     This is for temporal simulations with derivative bc
c
      du0upp=0.0
c
c     u0upp etc. will be (possibly) overwritten in bflow.f
c
      if (my_node_world.eq.0) then
         write(*,*) '  u0low,w0low : ',
     &        u0low,w0low
         write(*,*) '  u0upp,w0upp : ',
     &        u0upp,w0upp
         write(*,*) '        spanv : ',spanv
      end if

      return

 2001 continue
      if (my_node_world.eq.0) then
         write(*,*) 'Error reading file header, line 1:'
         write(*,*) 're = ',re
         write(*,*) 'pou= ',pou
         write(*,*) 'xl = ',xl
         write(*,*) 'zl = ',zl
         write(*,*) 't  = ',t
         write(*,*) 'xs = ',xs
         do i=1,scalar
            write(*,*) 'pr = ',pr(i),i
            write(*,*) 'm1 = ',m1(i),i
         end do
      end if
      call stopnow(2001)

      end subroutine rdiscbl
