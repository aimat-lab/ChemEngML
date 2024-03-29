c ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/wdiscbl.f $
c $LastChangedDate: 2013-09-16 07:23:16 +0200 (Mo, 16. Sep 2013) $
c $LastChangedBy: alex $
c $LastChangedRevision: 2 $
c
c ***********************************************************************
      subroutine wdiscbl(ur,ui,re,pr,m1,xl,zl,t,xs,dstar,fltype,
     &     bstart,bslope,rlam,spanv,namnut,m,gall,
     &     boxr,boxi,urx,alfa,zs,beta,my_node_world,
     &     u0low,u0upp,w0low,w0upp,du0upp)
c
c     Writes m variables from ur,ui to file namnut
c
      implicit none

      include 'par.f'
#ifdef MPI
      include 'mpif.h'
#endif

      character*80 namnut
      integer m,fltype
      real ur(memnx,memny,memnz,memnxyz),ui(memnx,memny,memnz,memnxyz)
      real re,pr(scalar),xl,zl,t,xs,dstar
      real bstart,bslope,rlam,spanv,m1(scalar)
      logical gall
      real urx(nx)
      real boxr(memnx,mbz,nyp),boxi(memnx,mbz,nyp)
      real alfa(nx/2*mbz),zs,beta(nz)
      real urtemp(memnx),uitemp(memnx)
      integer x,y,z,i,zb,ii,iii,zb_t,j
      real uw0low
      character(len=5) ch

      integer iotype
      real u0low,u0upp,w0low,w0upp,du0upp
      
c
c     MPI
c
      integer my_node_world,zbp,ip
#ifdef MPI
      integer ierror

      if (nproc.gt.1) call mpi_barrier(mpi_comm_world,ierror)
#endif      


      if (index(namnut,'NONE').eq.1.and.len_trim(namnut).eq.4) then
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
      if (namnut(len_trim(namnut)-2:len_trim(namnut)).eq.".uu") then
         iotype = 1
      end if
      if (namnut(len_trim(namnut)-2:len_trim(namnut)).eq.".vv") then
         iotype = 2
      end if
      if (namnut(len_trim(namnut)-2:len_trim(namnut)).eq.".ww") then
         iotype = 3
      end if

      if (iotype.eq.0) then


      if (my_node_world.eq.0) then
c
c     Write file header
c     Data is shifted so that xs,zs=0 for all fields
c     Any existing file will be overwritten
c
         open(unit=11,file=namnut,form='unformatted')
         rewind(11)
         if (scalar.ge.1) then
            write(11) re,.false.,xl,zl,t,0.,(pr(i),m1(i),i=1,scalar)
         else
            write(11) re,.false.,xl,zl,t,0.
         end if
         write(11) nx,nyp,nzc,nfzsym
         write(11) fltype,dstar
         if (fltype.eq.-1) write(11) rlam
         if (fltype.eq.-2) write(11) rlam,spanv
         if (fltype.ge.4) write(11) bstart,bslope,rlam,spanv
      end if

      do i=1,m
c
c     Choose which field to write
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


       
         do ip=0,nprocz-1
            do zb_t=1,memnz
               do y=1,nyp
                  do iii=0,nprocx-1
#ifdef MPI 
                     if (nproc.gt.1.and.my_node_world.eq.
     &                    ip*nprocx+iii.and.(ip+iii).ne.0) then
c     
c     Send complete field to processor 0
c
                        call mpi_send(ur(1,y,zb_t,ii),memnx,
     &                       mpi_double_precision,0,ip+iii+100,
     &                       mpi_comm_world,ierror)
                        call mpi_send(ui(1,y,zb_t,ii),memnx,
     &                       mpi_double_precision,0,ip+iii+200,
     &                       mpi_comm_world,ierror)
                     else
                        do x=1,memnx
                           urtemp(x)=ur(x,y,zb_t,ii)
                           uitemp(x)=ui(x,y,zb_t,ii)
                        end do
                     end if
c     
c     Receive individual fields from processors >0
c
                     if (my_node_world.eq.0.and.nproc.gt.1
     &                    .and.(ip+iii).ne.0) then
                        call mpi_recv(urtemp,memnx,
     &                       mpi_double_precision,
     &                       ip*nprocx+iii,ip+iii+100,
     &                       mpi_comm_world,mpi_status_ignore,ierror)
                        call mpi_recv(uitemp,memnx,
     &                       mpi_double_precision,
     &                       ip*nprocx+iii,ip+iii+200,
     &                       mpi_comm_world,mpi_status_ignore,ierror)
                     end if 
c
c     Wait until communication is finished
c
                     if (nproc.gt.1) then 
                        call mpi_barrier(mpi_comm_world,ierror)
                     end if
#endif
                     if (nproc.eq.1) then
                        do x=1,memnx
                           urtemp(x)=ur(x,y,zb_t,ii)
                           uitemp(x)=ui(x,y,zb_t,ii)
                        end do
                     end if
c
c     Shift data so that xs,zs=0 for all fields
c     Note: No shift for Couette flow!
c
          
                     zb=zb_t+ip*memnz
                     if (my_node_world.eq.0) then
                        if (fltype.ne.2.and.fltype.ne.5) then
                           call xysh(urtemp,uitemp,xs,zs,
     &                          alfa,beta,zb,iii)
                        end if
                        do x=1,memnx
                           urx(2*iii*memnx+2*x-1)= urtemp(x)
                           urx(2*iii*memnx+2*x)  = uitemp(x)
                        end do
                     end if
                  end do
                 
                  if (my_node_world.eq.0) then    
                     write(11) urx
                  end if
               end do
            end do
         end do
      end do

      if (my_node_world.eq.0) then
c
c     close the file and flush...
c
         call cflush(11)
         close(unit=11)
      end if



      else if (iotype.eq.1) then

c
c     Write multiple files per process
c
         if (my_node_world.eq.0) then
c
c     Write header
c
            open(unit=11,file=trim(namnut)//'-00000',form='unformatted')
            rewind(11)
            if (scalar.ge.1) then
               write(11) re,.false.,xl,zl,t,0.,(pr(i),m1(i),i=1,scalar)
            else
               write(11) re,.false.,xl,zl,t,0.
            end if
            write(11) nx,nyp,nzc,nfzsym
            write(11) fltype,dstar
            if (fltype.eq.-1) write(11) rlam
            if (fltype.eq.-2) write(11) rlam,spanv
            if (fltype.ge.4) write(11) bstart,bslope,rlam,spanv
            write(11) nproc
            write(11) u0low,u0upp,w0low,w0upp,du0upp
            close(unit=11)
         end if

         do j=1,nproc
            
c
c     Write data
c     
            if (my_node_world.eq.j-1) then
               write(ch,'(i5.5)') my_node_world+1
               open(unit=11,
     &              file=trim(namnut)//'-'//ch,form='unformatted')
               do i=1,m
c     
c     Choose which field to write
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
                  write(11) ur(:,:,:,ii),ui(:,:,:,ii)
               end do
               close(11)
            end if
            call mpi_barrier(mpi_comm_world,ierror)
         end do
      else if (iotype.eq.2) then
         write(*,*) 'Direct-access files not yet implemented'
         call stopnow(95885)
      else if (iotype.eq.3) then
         write(*,*) 'MPI-IO not yet implemented'
         call stopnow(95886)
      end if



c      if (my_node_world.eq.0) then
c
c     If one wants a status file whether a given velocity
c     file has been written, uncomment the following:
c
c         open(unit=11,file=trim(namnut)//'.written')
c         close(unit=11)
c      end if

      end subroutine wdiscbl
