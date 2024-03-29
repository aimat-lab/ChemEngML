# Example bla.i file
# Periodic channel flow with constant mass flux
#
20070716        Version of input file
init.u
end.u
1000000.00        Maximum simulation time (tmax)
-40             Maximum iterations (maxit)
-200            Maximum CPU time (cpumax)
171000          Maximum wall-clock time (wallmax)
.true.          Save intermediate files (write_inter)
0               Time step (dt)
4               1/3/4 number of RK stages (nst)
0.8             CFL number (cflmaxin)
10.0D0          Box length (xl)
.false.         Variable size (varsiz)
.0              Rotation rate (rot)
.true.          Constant mass flux (cflux)
#.false.         Constant mass flux (cflux)
#180.00          Re_tau
.false.         Perturbation mode (pert)
0               Boundary condition number (ibc)
0               Boundary condition scalar (tbc)
0.0D0           Theta_low
1.0D0           Theta_up
.false.         Chebyshev integration method (cim)
.false.         Galilei transformation (gall)
.false.         Constant wall-suction (suction)
.false.         Spatial simulation (spat)
0.              Temporal development vel. (cdev)
.false.         Read sgs.i (sgs)
0               SFD (isfd)
0               MHD (imhd)
10               Type of localized disturbance (loctyp)
.false.         Trip forcing (tripf)
0               Boundary condition at lower wall (wbci)
4               CFL calculation interval (icfl)
0               Amplitude calculation interval (iamp)
.false.         y-dependent statistics (longli)
0               Extremum calculation interval (iext)
64              xy-statistics calculation interval (ixys)
xy.stat
1024            xy-statistics saving interval (ixyss)
50.             Start time for statistics (txys)
.false.         Two-point correlations in z (corrf)
.false.         Two-point correlations in x (corrf_x)
.false.         Time series (serf)
0               Number of 3D fields to save (msave)
0               Number of wavenumbers to save (mwave)
0               Number of planes to save (npl)
1               Number of planes x pourya
1
filenotused.stat
0               Number of planes z pourya
endstring
