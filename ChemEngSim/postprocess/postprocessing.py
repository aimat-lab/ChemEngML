import os
import numpy as np
from math import pi
import struct
from pathlib import Path
from scipy.linalg import toeplitz


def chebdif(N, M):
    L = np.eye(N, dtype=bool)

    n1 = int(np.floor(N / 2))
    n2 = int(np.ceil(N / 2))

    k = np.linspace(0, N - 1, N, dtype=int)
    th = k * pi / (N - 1)

    x = np.sin(pi * np.linspace(N - 1, 1 - N, N) / (2 * (N - 1)))

    T = np.tile(th / 2, [N, 1]).T

    dx = 2 * np.sin(T.T + T) * np.sin(T.T - T)

    dx = np.concatenate((dx[0:n1, :], -np.flipud(np.fliplr(dx[0:n2, :]))))

    dx[L] = np.ones(N)

    c = (toeplitz((-1) ** k))

    c = np.float64(c)

    c[0, :] = c[0, :] * 2.0
    c[-1, :] = c[-1, :] * 2.0

    c[:, 0] = c[:, 0] / 2.0
    c[:, -1] = c[:, -1] / 2.0

    z = 1 / dx
    z[L] = np.zeros(N)

    d = np.eye(N)

    dm = np.zeros([N, N, M])

    for ell in range(M):
        d = (ell + 1) * z * (c * np.tile(np.diag(d), [N, 1]) - d)
        d[L] = -np.sum(d, axis=1)
        dm[:, :, ell] = d

    return x, dm


def postprocess(pathtofolder):
    # reading header
    with open(os.path.join(pathtofolder, 'xy.stat'), mode='rb') as file:  # b is important -> binary
        file_content = file.read()

    re = struct.unpack("d", file_content[4:12])
    xl = struct.unpack("d", file_content[16:24])
    pr = struct.unpack("d", file_content[48:56])

    nx = struct.unpack("i", file_content[113:117])
    ny = struct.unpack("i", file_content[117:121])

    nxys = struct.unpack("i", file_content[165:169])
    nxysth = struct.unpack("i", file_content[169:173])

    re = float(re[0])
    xl = float(xl[0])
    pr = float(pr[0])

    nx = int(nx[0])
    ny = int(ny[0])
    nxys = int(nxys[0])
    nxysth = int(nxysth[0])

    count = nx * ny

    # wall-normal coordindate
    y = np.zeros([ny, ])
    for i in range(0, ny):
        y[i] = 1 - np.cos(pi * i / (ny - 1))

    # reading geometry
    my_file = Path(os.path.join(pathtofolder, 'ibm.bin'))
    if my_file.is_file():
        with open(os.path.join(pathtofolder, 'ibm.bin'), mode='rb') as file:
            file_content2 = file.read()

        rgh_nxp = struct.unpack("i", file_content2[0:4])
        rgh_nyp = struct.unpack("i", file_content2[4:8])

        rgh_nxp = int(rgh_nxp[0])
        rgh_nyp = int(rgh_nyp[0])

        ibm_count = rgh_nxp * rgh_nyp

        ibm = (np.reshape(struct.unpack(str(ibm_count) + 'd', file_content2[20:20 + ibm_count * 8]),
                          [rgh_nyp, rgh_nxp]))
    else:
        rgh_nxp = int(nx * 3 / 2)
        rgh_nyp = ny
        ibm = np.zeros([rgh_nyp, rgh_nxp])

    ibm_x = (np.mean(ibm, axis=1)) * xl

    ibm_lw = np.trapz(ibm_x[0:int((ny + 1) / 2)], y[0:int((ny + 1) / 2)]) / xl
    ibm_uw = 2 - np.trapz(ibm_x[int((ny + 1) / 2):ny], y[int((ny + 1) / 2):ny]) / xl

    if ibm_lw != 0:
        ibm_lw = y[y < ibm_lw][-1]

    if ibm_uw != 2.0:
        ibm_uw = y[y > ibm_uw][0]

    delta_eff = (ibm_uw - ibm_lw) / 2

    # reading data
    var = np.zeros([ny, nx, nxys + nxysth])
    nstart = 193
    for i in range(0, nxys + nxysth - 1):
        var[:, :, i] = np.flipud(
            np.reshape(
                struct.unpack(str(count) + 'd',
                              file_content[nstart + i * (count * 8 + 8):nstart + i * (count * 8 + 8) + 8 * count]),
                [ny, nx]
            )
        )

    # computing fluctuations and assigning velocities and temperature
    u = var[:, :, 0]
    v = var[:, :, 1]

    theta = var[:, :, nxys]
    vtheta_fluct = var[:, :, nxys + 3] - var[:, :, 1] * var[:, :, nxys]

    u = np.roll(u, int(nx / 2), axis=1)
    v = np.roll(v, int(nx / 2), axis=1)
    theta = np.roll(theta, int(nx / 2), axis=1)

    vtheta_fluct = np.roll(vtheta_fluct, int(nx / 2), axis=1)

    # reading history file
    hist = np.genfromtxt(os.path.join(pathtofolder, 'history.out'))

    dpdx_mean = np.mean(hist[-1, 1])
    flow_mean = np.mean(hist[-1, 2])

    # computing integral quantities
    u_bulk = flow_mean / delta_eff
    tauw_dpdx = -dpdx_mean * delta_eff

    Cf = 2 * tauw_dpdx / u_bulk ** 2

    thetamean = np.mean(theta, axis=1)

    dthetady = np.zeros([ny, ])
    for i in range(1, ny - 1):
        dthetady[i] = (thetamean[i + 1] - thetamean[i - 1]) / (y[i + 1] - y[i - 1])

    dthetady[0] = (thetamean[1] - thetamean[0]) / (y[1] - y[0])
    dthetady[-1] = (thetamean[-1] - thetamean[-2]) / (y[-1] - y[-2])

    q_lam = dthetady
    q_coh = -pr * re * np.mean(
        (theta - np.tile(np.mean(theta, axis=1), (nx, 1)).T) * (v - np.tile(np.mean(v, axis=1), (nx, 1)).T), axis=1
    )
    q_turb = -pr * np.mean(vtheta_fluct, axis=1) * re

    q_tot = q_lam + q_coh + q_turb
    q_tot_mean = np.trapz(q_tot[ibm_x == 0], y[ibm_x == 0])

    u1d = np.mean(u, axis=1)
    umaxn = np.argmax(u1d)

    D_h = 2 * 2 * delta_eff
    D_h_l = 2 * 2 * (y[umaxn] - ibm_lw)
    D_h_u = 2 * 2 * (ibm_uw - y[umaxn])

    re_dh = u_bulk * re * D_h

    q = q_tot_mean / (np.max(y[ibm_x == 0]) - np.min(y[ibm_x == 0]))

    u_b_l = np.trapz(u1d[0:umaxn + 1], y[0:umaxn + 1]) / (y[umaxn] - ibm_lw)
    u_b_u = np.trapz(u1d[umaxn:], y[umaxn:]) / (ibm_uw - y[umaxn])

    theta_b_l = np.trapz(np.mean(theta[0:umaxn + 1, :], axis=1) * np.mean(u[0:umaxn + 1, :], axis=1),
                         y[0:umaxn + 1]) / u_b_l / (y[umaxn] - ibm_lw)
    theta_b_u = np.trapz((1 - np.mean(theta[umaxn:, :], axis=1)) * np.mean(u[umaxn:, :], axis=1), y[umaxn:]) / u_b_u / (
                ibm_uw - y[umaxn])

    Nu_l = (D_h_l) * q / theta_b_l
    Nu_u = (D_h_u) * q / theta_b_u
    Nu = np.mean([Nu_l, Nu_u])

    St = Nu / re_dh / pr

    return Cf, St
