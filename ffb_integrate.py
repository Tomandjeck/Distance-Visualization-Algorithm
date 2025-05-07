import sys
import torch
import math
import fill_with_density as fd
import numpy as np
import globals as g
import ps_figure as pf
from torch.autograd import Function
import pandas as pd

def dst_1d(x):
    """
       实现一维 DST-I，如果 norm='ortho'，则进行正交化
    """
    N=x.shape[0]
    #将一维向量转换为列向量（二维张量），形状为 (N, 1)
    k = torch.arange(N,dtype=torch.float64).view(-1,1) # (N, 1)
    n = torch.arange(N , dtype=torch.float64).view(1,-1) # (1, N)

    #factor 是一个 N x N 的系数矩阵，它每个元素的值为公式中的正弦项。
    factor=torch.sin(math.pi*(k+1)*(n+1)/(N+1))
    # 根据公式计算 DST-I 的输出（逐行应用DST-I）
    # 对于矩阵中的K行j列元素，它的值等于factor的k行与矩阵x的j列元素乘积的加权和
    y = 2 * torch.matmul(factor.float(), x.float())

    # 正交化：缩放因子
    scale_factor=torch.sqrt(torch.tensor(2 * (N + 1), dtype=torch.float64))

    y=y/scale_factor

    return y

def dct_I(x):
    N=x.shape[0]

    #构建一个(N,N-2)的矩阵
    k=torch.arange(0,N).view(-1,1) #(N, 1) 形状
    n=torch.arange(1,N-1).view(1,N-2)  # (1, N-2) 形状，注意 n 的范围是 1 到 N-2

    # 计算系数矩阵 factor，factor的形状为(N,N-2)
    factor = torch.cos(math.pi * k * n / (N - 1)).float()

    #x的形状为(N,N),选取x的形状为(N-2,N)
    #两矩阵相乘得到y的形状为(N,N)=(N,N-2) x (N-2,N)
    y=2*torch.matmul(factor,x[1:N-1,:].float())

    #计算公式去和部分
    x_0 = x[0, :]  # 大小为 (N,)，第 0 行
    x_N_1 = x[N - 1, :] # 大小为 (N,)，第 N-1 行
    # 生成矩阵 y，大小为 (N, N)
    y_add = torch.zeros(N, N)
    for k in range(N):
        sign_term = (-1) ** k  # 计算 (-1)^k
        y_add[k,:]=x_0+sign_term*x_N_1
    #对求和部分矩阵整体乘以根号2
    y_add=y_add*math.sqrt(2)

    #两个矩阵求和
    y_result=y+y_add

    # 计算系数
    f_0_N_minus_1 = 0.5 * math.sqrt(1 / (N - 1))  # 对于 k = 0 或 k = N-1
    f_other = 0.5 * math.sqrt(2 / (N - 1))  # 对于其他行

    # 创建一个和 x 大小相同的系数矩阵 f，用于保存每行的系数
    f = torch.ones(N, N)  # 初始化为全 1 矩阵，大小为 (N, 1)

    # 设置第 0 行和第 N-1 行的系数
    f[0, :] = f_0_N_minus_1
    f[N - 1, :] = f_0_N_minus_1

    # 设置其他行的系数，至此得到系数矩阵
    f[1:N - 1, :] = f_other

    #两个矩阵相乘，得到最终结果
    y_result=y_result*f

    return  y_result

#Function to initialize the Fourier transforms of gridvx[] and gridvy[] at very point on the lx-times-ly grid at t = 0.
# After this function has finished, we do not need to do any further Fourier transforms for this round of integration
#函数用于在 t = 0 时初始化 lx-times-ly 网格上每个点的 gridvx[] 和 gridvy[] 的傅里叶变换。
def init_gridv(rho_ft,grid_fluxx_init,grid_fluxy_init,lx, ly): #用于在 t = 0 时网格上的任意点处初始化 gridvx[] 和 gridvy[]
    dlx = float(g.L)  # We must typecast to prevent integer division by lx or ly.
    dly = float(g.L)  #我们必须进行类型转换。 否则，分母中的比率将为零。

    #There is a bit less typing later on if we divide now by 4*lx*ly because then the REDFT01 transform of rho_ft[0] will
    # directly be the mean of rho_init[].
    #如果我们现在除以 4*lx*ly，那么稍后的输入会少一些，因为 rho_ft[0] 的 REDFT01 变换将直接是 rho_init[] 的平均值。
    #g.gridvx可视化时值太小了，所以把这个值放大
    rho_ft = rho_ft /(4 * lx * ly) #4 * fd.lx * fd.ly

    #We temporarily insert the Fourier coefficients for the x- and y-components of the flux vector in the arrays grid_fluxx_init[] and
    #grid_fluxy_init[].
    #我们临时将流量向量的 x 和 y 分量的傅里叶系数插入数组 grid_fluxx_init[] 和 grid_fluxy_init[] 中。
    #grid_fluxx_init,grid_fluxy_init存放的是流量向量的 x 和 y 分量的傅里叶系数


    for i in range(lx-1):
        di=float(i)
        for j in range(ly):
            grid_fluxx_init[i, j] =-rho_ft[(i+1), j] /(torch.pi * ((di+1)/dlx + (j/(di+1)) * (j/dly) * (dlx/dly)))
    for j in range(ly):
        grid_fluxx_init[(lx-1), j]=0.0
    for i in range(lx):
        di=float(i)
        for j in range(ly-1):
            grid_fluxy_init[i, j]=-rho_ft[i, j + 1] /(torch.pi * ((di/(j+1)) * (di/dlx) * (dly/dlx) + (j+1)/dly))
    for i in range(lx):
        grid_fluxy_init[i, ly - 1]=0.0

    #使用傅里叶正弦和余弦表示速度分量，x速度分量用余弦表示，y速度分量用正弦表示
    #执行傅里叶变换，将流量向量从频率域转换回时域
    #grid_fluxx_init 和 grid_fluxy_init 是存储计算得到的流量向量的傅里叶变换系数
    #grid_fluxx_init = transform_fluxx(grid_fluxx_init)
    dst_rows=dst_1d(grid_fluxx_init)
    grid_fluxx_init=dct_I(dst_rows.T).T
    # import ps_figure as pf
    # pf.display(grid_fluxx_init)
    #grid_fluxy_init = transform_fluxy(grid_fluxy_init)

    dct_rows = dct_I(grid_fluxy_init)
    grid_fluxy_init = dst_1d(dct_rows.T).T
    # import ps_figure as pf
    # pf.display(grid_fluxy_init)
    #可视化
    # pf.new_picture_grid_flux_init(grid_fluxx_init)
    # pf.new_picture_grid_flux_init(grid_fluxy_init)
    return rho_ft,grid_fluxx_init,grid_fluxy_init

#Function to calculate the velocity at the grid points (x, y) with x = 0.5, 1.5, ..., lx-0.5 and y = 0.5, 1.5, ..., ly-0.5 at time t.
#函数功能计算时间 t 时网格点 (x, y) 处的速度的函数，其中 x = 0.5, 1.5, ..., lx-0.5 且 y = 0.5, 1.5, ..., ly-0.5。
def ffb_calcv(t,rho_init,rho_ft,grid_fluxx_init,grid_fluxy_init):

    #如果 rho 非常接近零，可能会引发错误
    rho = rho_ft[0, 0] + (1.0 - t) * (rho_init - rho_ft[0, 0])#+g.epsilon
    g.gridvx = -grid_fluxx_init / rho
    g.gridvy = -grid_fluxy_init / rho


    #pf.test_gridv(g.gridvx,g.gridvy)
    # pf.new_picture_gridv()

#Function to bilinearly interpolate a numerical array grid[0..lx*ly-1] whose entries are numbers for the positions:x = (0.5, 1.5, ..., lx-0.5), y = (0.5, 1.5, ..., ly-0.5)
# .The final argument "zero" can take two possible values: 'x' or 'y'. If zero==x, the interpolated function is forced to return 0 if x=0 or x=lx
# .This option is suitable fo interpolating from gridvx because there can be no flow through the boundary. If zero==y, the interpolation returns 0
# if y=0 or y=ly, suitable for gridvy. The unconstrained boundary will be  determined by continuing the function value at 0.5 (or lx-0.5 or ly-0.5)
#all the way to the edge (i.e. the slope is 0 consistent with a cosine transform).
#用于在一个给定的数值数组 grid 中进行插值。x和y是坐标的点，grid是x和y所在的网格
#函数的目的是为了在这个网格上对任意的 (x, y) 坐标进行插值计算。
#双线性插值的基本思想是在两个方向上分别进行线性插值。首先在 x 方向上对 y0 和 y1 进行插值，然后在 y 方向上对这两个结果进行插值，从而得到最终的插值结果。
# (x0, y0): x 和 y 坐标周围的左下角网格点。
# (x0, y1): 左上角网格点。
# (x1, y0): 右下角网格点。
# (x1, y1): 右上角网格点。
#计算每个网格点的贡献:
#对于每个网格点 (xi, yi)（其中 xi 是 x0 或 x1，yi 是 y0 或 y1），根据 x 和 y 的坐标，以及 zero 参数的设置，确定该点的值。如果 x 或 y 落在边界上，根据 zero 参数的值，可能会强制使网格点的值为 0。
#使用双线性插值公式，结合上述四个网格点的值以及 x 和 y 相对于这些点的位置，计算出 (x, y) 处的插值值。

def interpol(x,y,grid,zero,lx, ly):
    # 如果 x 或 y 是 pandas.Series，将其转换为 NumPy 数组
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    # 如果 x 或 y 是 NumPy 数组，将其转换为 PyTorch 张量
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    if torch.any(x < 0) or torch.any(x > lx) or torch.any(y < 0) or torch.any(y > ly):
        raise ValueError("ERROR: coordinate outside bounding box in interpol_batch()")
    if zero not in ['x', 'y']:
        raise ValueError("ERROR: unknown argument zero in interpol_batch().")

    x0 = torch.clamp(torch.floor(x + 0.5) - 0.5, 0, lx - 1)
    x1 = torch.clamp(torch.floor(x + 0.5) + 0.5, 0, lx - 1)
    y0 = torch.clamp(torch.floor(y + 0.5) - 0.5, 0, ly - 1)
    y1 = torch.clamp(torch.floor(y + 0.5) + 0.5, 0, ly - 1)

    # Avoid divide by zero by setting zero distance to a small number
    x1 = torch.where(x1 == x0, x0 + 1, x1)
    y1 = torch.where(y1 == y0, y0 + 1, y1)

    #On a scale from 0 to 1, how far is x (or y) away from x0 (or y0)? 1 means x=x1.
    delta_x = (x - x0) / (x1 - x0)
    delta_y = (y - y0) / (y1 - y0)

    #Function value at (x0, y0)

    def get_value(xp, yp):
        values = torch.zeros_like(xp, dtype=grid.dtype)
        mask = (x >= 0.5) & (y >= 0.5)
        xp_clipped = torch.clamp(xp[mask].to(torch.int64), 0, grid.shape[0] - 1)
        yp_clipped = torch.clamp(yp[mask].to(torch.int64), 0, grid.shape[1] - 1)
        # values[mask] = grid[xp_clipped, yp_clipped].to(values.dtype)
        # 转换 grid 为 PyTorch 张量以确保返回的是张量
        grid_tensor = torch.from_numpy(grid) if isinstance(grid, np.ndarray) else grid
        values[mask] = grid_tensor[xp_clipped, yp_clipped].to(values.dtype)
        return values

    fx0y0 = get_value(x0, y0)
    fx0y1 = get_value(x0, y1)
    fx1y0 = get_value(x1, y0)
    fx1y1 = get_value(x1, y1)

    return (1-delta_x)*(1-delta_y)*fx0y0 + (1-delta_x)*delta_y*fx0y1+ delta_x*(1-delta_y)*fx1y0 + delta_x*delta_y*fx1y1
#Function to integrate the equations of motion with the fast flow-based method.
#将运动方程与基于快速流的方法进行整合的函数。
def ffb_integrate(rho_init,rho_ft,proj,lx, ly):

    DEC_AFTER_NOT_ACC=0.75#0.75 因为存在先下降到比较低的值又反弹的情况
    ABS_TOL=min(lx,ly)*1e-3
    INC_AFTER_ACC=1.5 #1.1

    #为傅里叶变换分配内存。
    grid_fluxx_init = torch.zeros((g.L, g.L), dtype=torch.float64)
    grid_fluxy_init = torch.zeros((g.L, g.L), dtype=torch.float64)

    #eul[i*ly+j] will be the new position of proj[i*ly+j] proposed by a simple Euler step: move a full time interval delta_t with the
    # velocity at time t and position (proj[i*ly+j].x, proj[i*ly+j].y).
    #eul[i*ly+j] 将是由简单欧拉步骤提出的 proj[i*ly+j] 的新位置：以时间 t 处的速度和位置 (proj[i*ly+ j].x，proj[i*ly+j].y),移动一个完整的时间间隔 delta_t。。

    eul=torch.zeros((g.L, g.L, 2), dtype=torch.float64) #相较于构建三维数组，这种方式效率更高，但是它不够直观。

    #mid[i*ly+j] will be the new displacement proposed by the midpoint method (see comment below for the formula)
    #mid[i*ly+j] 将是 中点法 提出的新位移
    mid=torch.zeros((g.L, g.L, 2), dtype=torch.float64)

    #(vx_intp, vy_intp) will be the velocity at position (proj.x, proj.y) at time t.
    #(vx_intp, vy_intp) 将是时间 t 位置 (proj.x, proj.y) 处的速度。

    vx_intp = torch.zeros((g.L, g.L), dtype=torch.float64)
    vy_intp = torch.zeros((g.L, g.L), dtype=torch.float64)

    #(vx_intp_half, vy_intp_half) will be the velocity at the midpoint proj.x + 0.5*delta_t*vx_intp, proj.y + 0.5*delta_t*vy_intp) at time t + 0.5*delta_t.
    #(vx_intp_half, vy_intp_half) 将是时间 t + 0.5*delta_t 时中点 proj.x + 0.5*delta_t*vx_intp, proj.y + 0.5*delta_t*vy_intp) 处的速度。

    vx_intp_half = torch.zeros((g.L, g.L), dtype=torch.float64)
    vy_intp_half = torch.zeros((g.L, g.L), dtype=torch.float64)

    #Initialize grids for vx and vy using Fourier transforms
    #使用傅里叶变换初始化 vx 和 vy 的网格;
    rho_ft,grid_fluxx_init,grid_fluxy_init=init_gridv(rho_ft,grid_fluxx_init,grid_fluxy_init,lx, ly)

    t=0.0
    delta_t = 1e-2 #Initial time step. 1e-2
    iter = 0

    #Integrate
    while t <1: #0.586 1
        ffb_calcv(t,rho_init,rho_ft,grid_fluxx_init,grid_fluxy_init)

        # pf.new_picture_gridv()

        # We know, either because of the initialization or because of the check at the end of the last iteration, that (proj.x[k], proj.y[k])
        # is inside the rectangle [0, lx] x [0, ly]. This fact guarantees that interpol() is given a point that cannot cause it to fail.
        #gridvx,gridvy中存放了t时刻的x和y方向网格速度
        #当前网格点位置(g.proj[i*fd.ly+j][0], g.proj[i*fd.ly+j][1])的速度分量，并存储在 vx_intp 和 vy_intp 中
        # 使用NumPy的向量化操作创建坐标网格
        x_coords = proj[:,:, 0]
        y_coords = proj[:,:, 1]

        # 批量插值计算
        vx_intp = interpol(x_coords, y_coords, g.gridvx, 'x',lx, ly)
        vy_intp = interpol(x_coords, y_coords, g.gridvy, 'y',lx, ly)

        #pf.new_picture_vintp(vx_intp,vy_intp)

        accept = False
        while not accept:
            # vx_intp 和 vy_intp 被用于更新网格点的位置。使用简单的欧拉方法计算新位置

            eul[:,:, 0] = proj[:,:, 0] + vx_intp * delta_t
            eul[:,:, 1] = proj[:,:, 1] + vy_intp * delta_t

            #print(f"eul requires_grad : {eul.requires_grad}")
            #pf.new_picture_EulAndMid(eul,'e')

            # Use "explicit midpoint method".
            # x <- x + delta_t * v_x(x + 0.5*delta_t*v_x(x,y,t), y + 0.5*delta_t*v_y(x,y,t),  t + 0.5*delta_t)  and similarly for y.

            #显式中点方法计算新位置
            ffb_calcv(t + 0.5 * delta_t,rho_init,rho_ft,grid_fluxx_init,grid_fluxy_init)

            # Make sure we do not pass a point outside [0, lx] x [0, ly] to interpol(). Otherwise decrease the time step below and try again.
            accept = True

            # 使用向量化操作计算 proj_ij_x 和 proj_ij_y
            proj_ij_x = proj[:,:, 0] + 0.5 * delta_t * vx_intp
            proj_ij_y = proj[:,:, 1] + 0.5 * delta_t * vy_intp

            # print(f"proj_ij_x requires_grad : {proj_ij_x.requires_grad}")
            # 检查是否有超出边界的情况
            accept = torch.all((0.0 <= proj_ij_x) & (proj_ij_x <= lx) & (0.0 <= proj_ij_y) & (proj_ij_y <= ly))
            if not accept:
                delta_t *= DEC_AFTER_NOT_ACC

            if accept:
                #记录因为(mid[i,j,0] - eul[i,j,0]) **2 + (mid[i,j,1] - eul[i,j,1]) **2 > ABS_TOL的次数
                #记录因为0.0<= mid[i,j,0] <=fd.lx and 0.0 <= mid[i,j,1] <=fd.ly的次数
                #用来存储(mid[i,j,0] - eul[i,j,0]) **2 + (mid[i,j,1] - eul[i,j,1]) **2结果

                # 计算在中点位置的速度分量
                vx_intp_half = interpol(proj_ij_x, proj_ij_y, g.gridvx, 'x',lx, ly)
                vy_intp_half = interpol(proj_ij_x, proj_ij_y, g.gridvy, 'y',lx, ly)

                # print(f"vx_intp_half requires_grad : {vx_intp_half.requires_grad}")
                # 使用显式中点方法，根据中点位置的速度分量计算新位置
                mid[:, :, 0] = proj[:,:, 0] + vx_intp_half * delta_t
                mid[:, :, 1] = proj[:,:, 1] + vy_intp_half * delta_t

                # mid_x = g.proj[:,:,0] + vx_intp_half * delta_t
                # mid_y = g.proj[:,:, 1] + vy_intp_half * delta_t
                # mid=torch.stack((mid_x,mid_y),dim=-1)

                #print(f"mid requires_grad : {mid.requires_grad}")
                # 计算误差和检查边界条件
                diff_squared = (mid[:, :, 0] - eul[:, :, 0]) ** 2 + (mid[:, :, 1] - eul[:, :, 1]) ** 2
                within_tolerance = diff_squared <= ABS_TOL
                within_bounds = (mid[:, :, 0] >= 0.0) & (mid[:, :, 0] <= lx) & (mid[:, :, 1] >= 0.0) & (
                            mid[:, :, 1] <= ly)
                accept = torch.all(within_tolerance & within_bounds)

            # pf.new_picture_EulAndMid(mid,'m')
            if not accept:
                delta_t *= DEC_AFTER_NOT_ACC

        # if iter % 3 == 0:
        #     print(f'iter ={iter},t={t},delta_t={delta_t}')

        t += delta_t
        iter += 1

        # swap pointers for next iteration
        projtmp = proj
        #print(f"g.proj requires_grad : {g.proj.requires_grad}")
        #print(f"g.projtmp requires_grad : {g.projtmp.requires_grad}")
        # 使用向量化操作更新g.proj
        proj=mid
        #print(f"g.proj requires_grad : {g.proj.requires_grad}")
        mid=projtmp
        #print(f"mid requires_grad : {mid.requires_grad}")

        delta_t *= INC_AFTER_ACC  # Try a larger step size next time.
    return proj,projtmp