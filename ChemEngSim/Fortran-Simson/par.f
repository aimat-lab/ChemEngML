C ***********************************************************************
c
c $HeadURL: svn://129.13.162.100/simson/blaq/par.f $
c $LastChangedDate: 2016-02-08 18:24:41 +0100 (Mon, 08. Feb 2016) $
c $LastChangedBy: alex $
c $LastChangedRevision: 95 $
c
c ***********************************************************************
c
c     par.f contains size of problem
c
      integer nx,ny,nz,nby,nbz,mby,mbz
      integer nfxd,nfyd,nfzd,nfzsym
      integer nxp,nyp,nzp,nzc,nzat,nzst,nzd,nzpc,nzd_new
      integer npls,nzstat0,nystat0
      integer memnx,memny,memnz,memnxyz
      integer mbox2,mbox3,nproc,nprocx,nprocz,nthread,nxys,nxysth
      integer osnf,mcorr,mbla,mser,msamp,scalar,pressure
c
c     Adjustable parameters:
c     ======================
c
c     Number of spectral modes
c
C flow in x-direction
C      parameter (nx=1152,ny=301,nz=576)
C      parameter (nx=768,ny=301,nz=384)
C      parameter (nx=768,ny=301,nz=384)
C      parameter (nx=4096,ny=301,nz=384)
C flow in z-direction
C      parameter (nx=864,ny=865,nz=1024)
      parameter (nx=256,ny=129,nz=4)
C      parameter (nx=1024,ny=385,nz=768)
C      parameter (nx=128,ny=257,nz=256)
c
c     Number of processors (MPI)
c
      parameter (nprocx=2)
      parameter (nprocz=nprocx)
      parameter (nproc=nprocz*nprocx)
c
c     Number of threads (OpenMP)
c
      parameter (nthread=1)
c
c     Statistics
c
      parameter (nxys  =  96)
      parameter (nxysth=  35)
      parameter (mcorr =  500)
      parameter (mser  =  20)
      parameter (msamp =  16)
      parameter (npls  = 12)
      parameter (nzstat0 = 3)
      parameter (nystat0 = 5)
c
c     Pressure (0/1)
c
      parameter (pressure=1)
c
c     Passive scalars
c
      parameter (scalar=1)
c
c     Dealiazing flags
c
      parameter (nfxd=1,nfyd=0,nfzd=1)
c
c     Number of waves for freestream OS-eigenmodes
c
      parameter (osnf = 0)
c
c     Number of points in base flow
c
      parameter (mbla = 20001)
c
c     Temporally disabled parameters: (don't change them!)
c     ====================================================
c
c     Symmetry flag
c
      parameter (nfzsym=0)
c
c     Boxsize
c
      parameter (mby=1,mbz=1)
c
c     Computed parameters: (don't change them!)
c     =========================================
c
      parameter (nxp=nx+nfxd*nx/2,nyp=ny+nfyd*ny/2,nzp=nz+nfzd*nz/2)
      parameter (nzc=nz-nfzsym*nz/2)
      parameter (nzd=nzp-(nzp/2-2)*nfzsym)
      parameter (nzpc=nzp-(nzp/2-1)*nfzsym)
      parameter (nzd_new=(nzp/nprocz+min(1,mod(nzp,nprocz)))*nprocz)
c
c     Symmetric transform lengths
c
      parameter (nzat=nzp/2-1,nzst=nzp/2+1)
c
c     Number of boxes
c
      parameter (nby=(nyp+mby-1)/mby,nbz=nzc/mbz)
c
c     Core storage size
c
      parameter (memnx=nx/(2*nprocx))
      parameter (memnz=nzc/nprocz)
c
c     The next line has to be used if the mpi_alltoall implementation
c     in getpxz and putpxz is employed
      parameter (memny=(nyp/nprocz+1)*nprocz)
c     Otherwise, use the next line:
c      parameter (memny=nyp)
c
c     Arrangement of main storage:
c
c     1-3:   velocities
c     4-5:   vorticities
c     6-7:   partial right-hand sides
c     8  :   pressure (if compiled accordingly)
c     9-11:  theta, dtheta/dy, prhs (scalar 1, if compiled accordingly)
c     12-:   repeated entries 9-11 for additional scalars
c
      parameter (memnxyz=7+pressure+scalar*3)
c
c     Boxsize
c
      parameter (mbox2=(nxp/2+1)*nzd_new/nprocz*mby*nthread)
      parameter (mbox3=memnx*nyp*mbz*nthread)
c
c     End of par.f
c
c ***********************************************************************
