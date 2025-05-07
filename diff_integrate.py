import fill_with_density as fd
import numpy as np
import math


from scipy.fft import idctn,dctn


PI=3.14159265358979323846264338327950288419716939937510
#Function to calculate the velocity at the grid points (x, y) with x =0.5, 1.5, ..., lx-0.5 and y = 0.5, 1.5, ..., ly-0.5 at time t.
def diff_calcv(t):
    dlx=float(fd.lx) # We must typecast to prevent integer division by lx or ly.
    dly=float(fd.ly)
    global rho,gridvx,gridvy
    #Fill rho with Fourier coefficients.
    for i in range(fd.lx):
        for j in range(fd.ly):
            rho[i*fd.ly+j]=math.exp((- (i/dlx)*(i/dlx) - (j/dly)*(j/dly)) * t)*fd.rho_ft[i*fd.ly+j]
    #Replace rho by cosine Fourier backtransform in both variables. rho[i*ly+j] is the density at position (i+1/2, j+1/2).
    rho=idctn(rho, norm='ortho', type=3)

    #We temporarily insert the Fourier coefficients for the x- and y-components of the flux vector in the arrays gridvx and gridvy.
    for i in range(fd.lx-1):
        for j in range(fd.ly):
            gridvx[i * fd.ly + j] =fd.rho_ft[(i + 1) * fd.ly + j] * (i + 1) *math.exp((- ((i + 1) / dlx) * ((i + 1) / dlx) - (j / dly) * (j / dly)) * t) / (PI * dlx)
    for j in range(fd.ly):
        gridvx[(fd.lx - 1) * fd.ly + j] = 0.0
    for i in range(fd.lx):
        for j in range(fd.ly-1):
            gridvy[i * fd.ly + j] =fd.rho_ft[i * fd.ly + j + 1] * (j + 1) *math.exp((- (i / dlx) * (i / dlx) - ((j + 1) / dly) * ((j + 1) / dly)) * t) / (PI * dly)
    for i in range(fd.lx):
        gridvy[i*fd.ly + fd.ly - 1] = 0.0

    #Compute the flux vector and temporarily store the result in gridvx and gridvy.
    gridvx=idctn(gridvx, norm='ortho', type=3)
    gridvy=idctn(gridvy, norm='ortho', type=3)

    #The velocity is the flux divided by the density.
    for i in range(fd.lx):
        for j in range(fd.ly):
            if rho[i*fd.ly + j] <= 0.0:
                print(f'ERROR: division by zero in diff_calcv()')
                print(f'rho[{i}, {j}] ={rho[i*fd.ly + j]}')
            gridvx[i * fd.ly + j] /= rho[i * fd.ly + j]
            gridvy[i * fd.ly + j] /= rho[i * fd.ly + j]

#Function to integrate the equations of motion with the diffusion method.
def diff_integrate():

    ABS_TOL=min(fd.lx,fd.ly)*1e-6
    INC_AFTER_ACC=1.1
    DEC_AFTER_NOT_ACC=0.75
    CONV_MAX_CHANGE=min(fd.lx,fd.ly)*1e-9
    MAX_ITER=10000
    MIN_T=1e3
    MAX_T=1e12

    global rho,gridvx,gridvy
    rho = np.zeros(fd.lx * fd.ly, dtype=float)
    gridvx = np.zeros(fd.lx * fd.ly, dtype=float)
    gridvy = np.zeros(fd.lx * fd.ly, dtype=float)

    global eul
    eul = np.zeros((fd.lx * fd.ly, 2), dtype=float)

    # mid[i*ly+j] will be the new displacement proposed by the midpoint method (see comment below for the formula)
    global mid
    mid = np.zeros((fd.lx * fd.ly, 2), dtype=float)

    #(vx_intp, vy_intp) will be the velocity at position (proj.x, proj.y) at time t.
    global vx_intp, vy_intp
    vx_intp = np.zeros(fd.lx * fd.ly, dtype=float)
    vy_intp = np.zeros(fd.lx * fd.ly, dtype=float)

    #(vx_intp_half, vy_intp_half) will be the velocity at the midpoint proj.x + 0.5*delta_t*vx_intp, proj.y + 0.5*delta_t*vy_intp) at time t + 0.5*delta_t.
    global vy_intp_half,vx_intp_half
    vx_intp_half = np.zeros(fd.lx * fd.ly, dtype=float)
    vy_intp_half = np.zeros(fd.lx * fd.ly, dtype=float)

    t = 0.0
    delta_t = 1e-2  # Initial time step.
    iter = 0

    import main_ as m
    import ffb_integrate as fi

    while True:
        diff_calcv(t)

        # We know, either because of the initialization or because of the check at the end of the last iteration, that (proj.x[k], proj.y[k])
        # is inside the rectangle [0, lx] x [0, ly]. This fact guarantees that interpol() is given a point that cannot cause it to fail.
        for k in range(fd.lx * fd.ly):
            vx_intp[k] = fi.interpol(m.proj[k][0], m.proj[k][1], gridvx, 'x')
            vy_intp[k] = fi.interpol(m.proj[k][0], m.proj[k][1], gridvy, 'y')

        accept = False
        while not accept:
            for k in range(fd.lx * fd.ly):
                eul[k][0] = m.proj[k][0] + vx_intp[k] * delta_t
                eul[k][1] = m.proj[k][1] + vx_intp[k] * delta_t
            # Use "explicit midpoint method".
            # x <- x + delta_t * v_x(x + 0.5*delta_t*v_x(x,y,t), y + 0.5*delta_t*v_y(x,y,t),  t + 0.5*delta_t)  and similarly for y.
            diff_calcv(t + 0.5*delta_t)

            # Make sure we do not pass a point outside [0, lx] x [0, ly] to interpol(). Otherwise decrease the time step below and try again.
            accept = True
            for k in range(fd.lx * fd.ly):
                if m.proj[k][0] + 0.5 * delta_t * vx_intp[k] < 0.0 or m.proj[k][0] + 0.5 * delta_t * vx_intp[k] > fd.lx or \
                        m.proj[k][1] + 0.5 * delta_t * vy_intp[k] < 0.0 or m.proj[k][1] + 0.5 * delta_t * vy_intp[
                    k] > fd.ly:
                    accept = False
                    delta_t *= DEC_AFTER_NOT_ACC
                    break
            if accept:
                for k in range(fd.lx * fd.ly):
                    vx_intp_half[k] = fi.interpol(m.proj[k][0] + 0.5 * delta_t * vx_intp[k],
                                               m.proj[k][1] + 0.5 * delta_t * vy_intp[k],
                                               gridvx, 'x')
                    vy_intp_half[k] = fi.interpol(m.proj[k][0] + 0.5 * delta_t * vx_intp[k],
                                               m.proj[k][1] + 0.5 * delta_t * vy_intp[k],
                                               gridvy, 'y')
                    mid[k][0] = m.proj[k][0] + vx_intp_half[k] * delta_t
                    mid[k][1] = m.proj[k][1] + vy_intp_half[k] * delta_t

                    # Do not accept the integration step if the maximum squared difference between the Euler and midpoint proposals exceeds
                    # ABS_TOL. Neither should we accept the integration step if one  of the positions wandered out of the boundaries. If it happened, decrease the time step.
                    if (mid[k][0] - eul[k][0]) * (mid[k][0] - eul[k][0]) + (mid[k][1] - eul[k][1]) * (
                            mid[k][1] - eul[k][1]) > ABS_TOL or mid[k][0] < 0.0 or mid[k][0] > fd.lx or mid[k][
                        1] < 0.0 or mid[k][1] > fd.ly:
                        accept = False
                        delta_t *= DEC_AFTER_NOT_ACC
                        break
        #What is the maximum change in squared displacements between this and the previous integration step?
        max_change = 0.0
        for k in range(fd.lx * fd.ly):
            max_change=max((mid[k][0]-m.proj[k][0])*(mid[k][0]-m.proj[k][0]) +(mid[k][1]-m.proj[k][1])*(mid[k][1]-m.proj[k][1]),max_change)

        if iter%10==0:
            print(f'iter ={iter},t={t},delta_t={delta_t}')
        #When we get here, the integration step was accepted.
        t += delta_t
        iter += 1
        for k in range(fd.lx * fd.ly):
            m.proj[k][0] = mid[k][0]
            m.proj[k][1] = mid[k][1]
        delta_t *= INC_AFTER_ACC

        if not (max_change > CONV_MAX_CHANGE and t < MAX_T and iter < MAX_ITER) or not t < MIN_T:
            break
