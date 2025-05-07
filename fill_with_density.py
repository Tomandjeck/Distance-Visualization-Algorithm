import math
import pandas as pd
import ps_figure as pf

import globals as g
import torch.fft
from torch.autograd import Function
import torch
from torch_dct import dct,idct

#
# def dct_1d(x,norm="ortho"):
#     """
#       实现一维 DCT-I，用于支持梯度计算，并且考虑 norm='ortho'
#     """
#
#     N=x.shape[0]
#     X=torch.zeros_like(x)
#
#     if norm=="ortho":
#         scale=torch.sqrt(torch.tensor(2.0/(N-1)))
#     else :
#         scale=1.0
#
#     # 计算 DCT-I
#     for k in range(N):
#         if k==0 or k==N-1:
#             factor=0.5*torch.sqrt(torch.tensor(1.0/(N-1)))
#         else:
#             factor=0.5*scale
#
#         X[k]=factor * (x[0] + (-1)**k*x[N-1])*torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
#
#         for n in range(1,N-1):
#             X[k]+=2*factor*x[n]*torch.cos(torch.tensor(math.pi*k*n/(N-1), dtype=torch.float64))
#
#     return X
#
# def dct_2d(matrix,norm="ortho"):
#     """
#         对二维矩阵应用 DCT-I，考虑 norm='ortho'
#     """
#
#     # 对每一行应用 DCT-I
#     dct_rows=torch.stack([dct_1d(row,norm) for row in matrix])
#
#     # 对每一列应用 DCT-I
#     dct_2d_result = torch.stack([dct_1d(col, norm) for col in dct_rows.T]).T
#
#     return dct_2d_result
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

# def dct_1d(x,norm="ortho"):
#     """
#       实现一维 DCT-I，用于支持梯度计算，并且考虑 norm='ortho'
#     """
#
#     N=x.shape[0]
#     X=torch.zeros_like(x)
#
#     if norm=="ortho":
#         scale=torch.sqrt(torch.tensor(2.0/(N-1)))
#     else :
#         scale=1.0
#
#     factor_0_N = 0.5 * torch.sqrt(torch.tensor(1.0 / (N - 1)))
#     sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
#     cos_vals = torch.cos(torch.pi * torch.arange(1, N - 1).float() / (N - 1)).unsqueeze(0)  # 预计算 cos 值
#     n_indices = torch.arange(1, N - 1)
#     # 计算 DCT-I
#     for k in range(N):
#         if k==0 or k==N-1:
#             factor=0.5*torch.sqrt(torch.tensor(1.0/(N-1)))
#         else:
#             factor=0.5*scale
#
#         X[k]=factor * (x[0] + (-1)**k*x[N-1])*torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
#
#         for n in range(1,N-1):
#             X[k]+=2*factor*x[n]*torch.cos(torch.tensor(math.pi*k*n/(N-1), dtype=torch.float64))
#
#     return X
#
# def dct_2d(matrix,norm="ortho"):
#     """
#         对二维矩阵应用 DCT-I，考虑 norm='ortho'
#     """
#
#     # 对每一行应用 DCT-I
#     dct_rows=torch.stack([dct_1d(row,norm) for row in matrix])
#
#     # 对每一列应用 DCT-I
#     dct_2d_result = torch.stack([dct_1d(col, norm) for col in dct_rows.T]).T
#
#     return dct_2d_result
#Function to change coordinates from (minx, miny, maxx, maxy) to (0, 0, LX, LY).
#根据给定的边界和比例因子来调整地图的尺寸和位置，确保地图在一个特定的矩形框架内合适地显示
def rescale_map(g_x,g_y):

    #在地图和矩形边界之间留出足够空间的最小尺寸。
    #用于调整一个矩形映射区域的边界，PADDING、rm.map_maxx、rm.map_maxy、rm.map_minx、rm.map_miny 都是已给定的常数，分别代表填充比例和矩形映射区域的最大及最小x和y坐标值。
    #new_maxx 和 new_minx 计算的是调整后的矩形的最大和最小X坐标。new_maxy 和 new_miny 计算的是调整后的矩形的最大和最小Y坐标。
    #这些新坐标是通过原始坐标加上或减去原始宽度或高度的一部分（由PADDING因子决定）来计算得出的。如果PADDING为正值，新的矩形区域会比原始区域大，反之，则会更小。
    #计算方式是先通过1.0 ± PADDING因子扩大或缩小原始坐标，然后取平均值来定位新坐标。这种方法确保了矩形中心点不变，而整体大小根据PADDING值进行调整。
    g.new_maxx = 0.5 * ((1.0 + g.PADDING) * g.map_maxx + (1.0 - g.PADDING) * g.map_minx)
    g.new_minx = 0.5 * ((1.0 - g.PADDING) * g.map_maxx + (1.0 + g.PADDING) * g.map_minx)
    g.new_maxy = 0.5 * ((1.0 + g.PADDING) * g.map_maxy + (1.0 - g.PADDING) * g.map_miny)
    g.new_miny = 0.5 * ((1.0 - g.PADDING) * g.map_maxy + (1.0 + g.PADDING) * g.map_miny)
    #decide whether to keep the original length as width or height
    if (g.map_maxx-g.map_minx) > (g.map_maxy-g.map_miny): #如果条件成立，意味着矩形的宽度大于高度
        lx=g.L
        g.latt_const=(g.new_maxx-g.new_minx) / g.L
        ly=1 << int(math.ceil(math.log2((g.new_maxy - g.new_miny) / g.latt_const)))
        #adjust new bounding box coordiantes
        g.new_maxy = 0.5 * (g.map_maxy + g.map_miny) + 0.5 * ly * g.latt_const
        g.new_miny = 0.5 * (g.map_maxy + g.map_miny) - 0.5 * ly * g.latt_const
    else:#反之，矩形的宽度小于高度
        ly = g.L
        g.latt_const = (g.new_maxy - g.new_miny) / g.L
        lx = 1 << int(math.ceil(math.log2((g.new_maxx - g.new_minx) / g.latt_const)))
        # adjust new bounding box coordiantes
        g.new_maxx = 0.5 * (g.map_maxx + g.map_minx) + 0.5 * lx * g.latt_const
        g.new_minx = 0.5 * (g.map_maxx + g.map_minx) - 0.5 * lx * g.latt_const

    #重新缩放所有多边形坐标
    for i in range(g.n_poly):
        for j in range(int(g.n_polycorn[i])):
            g.polycorn[i][j][0] = (g.polycorn[i][j][0] - g.new_minx) / g.latt_const
            g.polycorn[i][j][1] = (g.polycorn[i][j][1] - g.new_miny) / g.latt_const

    g_x=(g_x-g.new_minx)/g.latt_const
    g_y=(g_y-g.new_miny)/g.latt_const

    #If we wish to plot the inverse transform, we must save the original
    #polygon coordinates.

    return  g_x,g_y,lx,ly


#Function to set values of inside[][], used in set_inside_values_for_polygon() below. It sets the value in inside[][]
#for all x-values between poly_minx and the x-value (of the point on the line connecting the given two coordinates) that corresponds to the
#current y-value l.


def set_inside_value_at_y(region,pk,pn,l,poly_minx,inside): #,inside
    intersection = (pn[0] - 0.5 - (pk[0] - 0.5)) * (l - (pk[1] - 0.5)) / (pn[1] - 0.5 - (pk[1] - 0.5)) + (pk[0] - 0.5)
    poly_minx_int = int(poly_minx)
    intersection_ceil = math.ceil(intersection)

    inside[poly_minx_int:intersection_ceil, l] = region - inside[poly_minx_int:intersection_ceil, l] - 1

#Function that takes two polygon coordinates and loops over the y-values between the two input y-coordinates.
# It updates the value of inside[][] for all points between polyminx (for this polygon) and the x-value at all
# coordinates on the horizontal line to the left of the line segment connecting the input coordinates.
def set_inside_values_between_points(region,pk,pn,poly_minx,inside):  #,inside
    #Loop over all integer y-values between the two input y-coordinates.
    min_y = math.ceil(min(pn[1], pk[1]) - 0.5)
    max_y = math.ceil(max(pn[1] - 0.5, pk[1] - 0.5))
    for l in range(min_y, max_y):
        set_inside_value_at_y(region, pk, pn, l, poly_minx, inside)

#Function to set values in inside[][] for a particular polygon in a region.
#输入的是某个区域中的多边形
def set_inside_values_for_polygon(region, n_polycorn, polycorn, inside): #,inside
    #region, n_polycorn, polycorn, inside = args
    poly_minx = min(poly[0] for poly in polycorn)

    #poly_minx中存放的是最小的x值
    #Loop over all pairs of consecutive coordinates of polygon.
    for k in range(int(n_polycorn)):
        n = k - 1 if k > 0 else n_polycorn - 1
        n=int(n)
        # 在某个区域中的某个多边形，处理多边形里面两个点之间的关系
        set_inside_values_between_points(region, polycorn[k], polycorn[n], poly_minx, inside) #, inside


#Function to determine if a grid point at (x+0.5, y+0.5) is inside one of the polygons and, if yes,in which region.
#The result is stored in the array xyhalfshift2reg[x][y]. If (x+0.5, y+0.5) is outside all polygons, then xyhalfshift2reg[x][y] = -1.
# If it is inside region i, then xyhalfshift2reg[x][y] = i.

def interior(lx, ly):
    xyhalfshift2reg = torch.full((lx, ly), -1,dtype=torch.int)

    for i in range(g.n_reg):
        for j in range(g.n_polyinreg[i]):
            #polyinreg[i][j]的格式为：[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [16]...]
            #poly得到的是多边形的编号
            #i存放的是哪个区域编号，n_polycorn中存放的是多边形含有的顶点数量，polycorn存放的具体顶点，xyhalfshift2reg是矩阵
            poly=g.polyinreg[i][j]

            set_inside_values_for_polygon(i,g.n_polycorn[poly],g.polycorn[poly],xyhalfshift2reg)
    return xyhalfshift2reg



#Function to smoothen the density near polygon edges.
#通过对密度函数进行平滑处理，可以减少高频噪声和细节，这样在随后的计算中可以使用较大的时间步长而不影响计算的稳定性和精度。
#高斯模糊可以减少密度函数中的极端值和快速变化的区域，这样在进行数值积分时，能够更快地达到收敛。
#初始平滑处理能够提高数值方法的稳定性，避免在计算过程中出现不必要的振荡和数值误差，从而提高结果的可靠性。
def gaussian_blur(rho_init,rho_ft,lx, ly):
    #，通过在计算开始之前应用适度宽度的高斯模糊，可以加速收敛过程，并确保计算结果的准确性和稳定性。
    #g.gridvx可视化时值太小了，所以把这个值放大

    rho_init=rho_init/(4 *lx * ly)


    # 1,输入密度函数(rho_init)，进行傅里叶变换，得到傅里叶系数(rho_ft)
    # 2，利用傅里叶系数计算每个点的速度分量
    # 3，傅里叶变换只需要开始时计算一次，然后每一步计算中使用这些结果
    #傅里叶变换允许将空间域的问题转换为频率域，在频率域进行计算更高效。

    # 在列上应用 DCT-II
    rho_ft = dct(rho_init, norm='ortho')
    # 交换维度以在行上应用 DCT-II
    rho_ft = dct(rho_ft.T, norm='ortho').T

    # import ps_figure as pf
    # pf.display(g.rho_ft)
    #
    # import ps_figure as pf
    # pf.new_picture_rho_ft()

    #Now perform the Gaussian blur.
    #使用高斯模糊，调整rho_ft

    prefactor = -0.5 * g.BLUR_WIDTH * g.BLUR_WIDTH * math.pi * math.pi

    # 生成比例数组
    scale_i = torch.arange(lx, dtype=torch.float64) / lx
    scale_j = torch.arange(ly, dtype=torch.float64) / ly

    # 利用广播生成比例网格
    scale_i_grid, scale_j_grid = torch.meshgrid(scale_i, scale_j, indexing='ij')

    # 计算指数部分
    exp_factor = torch.exp(prefactor * (scale_i_grid ** 2 + scale_j_grid ** 2))

    # 应用指数因子到rho_ft
    rho_ft = rho_ft *exp_factor

    # import ps_figure as pf
    # pf.display(g.rho_ft)

    # 对 rho_ft 进行二维逆 DCT（DCT-III） 9.25比较成功的版本
    # g.rho_init=dct_2d(g.rho_ft,norm="ortho")
    temp=dct_I(rho_ft)
    rho_init =dct_I(temp.T).T
    # import ps_figure as pf
    # pf.display(g.rho_init)

    return  rho_init




#Function to fill the lx-times-ly-grid with the initial density.It reads the input target areas and produces a
# .eps image of the input polygons. It also performs a Gaussian blur on the input density. This function  should only be called once,
# namely before the first round of integration.Afterwards use fill_with_density2() below.
#map_file_name:存放的处理之后的json数据；area_file_name：存放的是想可视化的数据
def fill_with_density1(population_data_tensor,population_data_id_tensor,target_area):

    import read_map as rm
    #Read the coordinates.
    g_x,g_y=rm.read_map()

    #缩放前的地图
    # import ps_figure as pf
    # pf.figure_polycorn()

    #Fit the map on an (lx)*(ly)-square grid.
    g_x,g_y,lx,ly=rescale_map(g_x,g_y)

    #缩放后的地图
    # import ps_figure as pf
    # pf.figure_polycorn(g_x,g_y)

    # if eps:
    #     import ps_figure as pf
    #     pf.figure_cartcorn_coordiante() #暂定

    if g.n_reg==1:
        target_area[0]=1.0
        return True

    init_area = torch.zeros(g.n_reg, dtype=torch.float64)
    dens = torch.zeros(g.n_reg, dtype=torch.float64)

    #确定网格点位于哪些区域内。
    xyhalfshift2reg=interior(lx, ly)
    # import ps_figure as pf
    # pf.new_picture_xyhalfshift2reg()

    # 获取csv文件当中的region data数据
    # df = pd.read_csv(area_file_name,encoding='gbk') #,encoding='gbk'
    # area = df['Region Data'].values
    # id=df['Region Id'].values
    # g.population_data_tensor= g.population_data_tensor*1
    area = population_data_tensor
    id = population_data_id_tensor

    #print(f"area requires_grad : {area.requires_grad}")

    for i in range(g.n_reg):
        if id[i] != -1 and area[i] != -1:
            if id[i] >g.max_id or g.region_id_inv[int(id[i].item())]<0:
                print(f"id[{i}],g.max_id={g.max_id} or g.region_id_inv[int(id[i].item())]={g.region_id_inv[int(id[i].item())]}")
                print("ERROR: Identifier %d in area-file does not match")
                exit(1)
            target_area[g.region_id_inv[int(id[i].item())]]=area[i]
        elif id[i] != -1 and area[i] == -1.0:
            if area[i] == 'NA':
                target_area[g.region_id_inv[int(id[i].item())]]=-2.0
                g.region_na[g.region_id_inv[int(id[i].item())]]=1

    for i in range(g.n_reg):
        if target_area[i] <= 0.0 and target_area[i] != -2.0:
            target_area[i]=0.000001
            print(f"ERROR: No target area for region {g.region_id[i]}.")
            # exit(1)

    #Replace target areas equal to zero by a small positive value.
    tmp_tot_target_area=torch.tensor(0.0,dtype=target_area.dtype)
    tot_init_area=0.0
    na_ctr=0

    #print(f"tmp_tot_target_area grad_fn : {tmp_tot_target_area.grad_fn}")

    #计算每个多边形的面积
    areas=rm.polygon_area(g.polycorn)

    #用于取ploy的面积的值
    ploys=0
    for i in range(g.n_reg):
        if g.region_na[i] == 1:
            na_ctr+=1
        else:
            tmp_tot_target_area+=target_area[i]

        for j in range(g.n_polyinreg[i]):
            init_area[i] += areas[ploys] #rm.polygon_area(g.n_polycorn[g.polyinreg[i][j]],g.polycorn[g.polyinreg[i][j]])
            ploys+=1
        tot_init_area += init_area[i] #面积之和

    #tmp_tot_target_area.requires_grad_()
    #print(f"tmp_tot_target_area grad_fn : {tmp_tot_target_area.grad_fn}")

    # 用于取ploy的周长的值
    ploys = 0
    perimeters=rm.polygon_perimeter()
    region_perimeter = torch.zeros(g.RIGIONS, dtype=torch.float64)
    for i in range(g.n_reg):
        for j in range(g.n_polyinreg[i]):
            region_perimeter[i]+=perimeters[ploys]
            ploys+=1

    first_region=1
    total_NA_ratio=0

    for i in range(g.n_reg):
        if g.region_na[i]==1:
            total_NA_ratio+=init_area[i]/tot_init_area

    total_NA_area=(total_NA_ratio*tmp_tot_target_area)/(1-total_NA_ratio)
    tmp_tot_target_area+=total_NA_area


    for i in range(g.n_reg):
        # Set target area for regions with NA values
        if g.region_na[i]==1:
            if first_region==1:
                print("Setting area for NA regions:")
                first_region=0
            target_area[i]=(init_area[i]/tot_init_area)/total_NA_ratio*total_NA_area
            print(f"{g.region_id}:{target_area[i]}")


    #Increase target area for regions which will be too small in order to speed
    #up cartogram generation process. This happens when -n flag is not set
    if g.use_perimeter_threshold == True:
        # print("Note: Enlarging extremely small regions using scaled")
        # print("perimeter threshold. Areas for these regions will be")
        # print("scaled up. To disable this, please add the -n flag.")
        region_small=torch.zeros(g.n_reg, dtype=torch.int)
        region_threshold=torch.zeros(g.n_reg, dtype=torch.float64)
        region_threshold_area=torch.zeros(g.n_reg, dtype=torch.float64)
        region_small_ctr=0
        tot_region_small_area=0
        total_perimeter=torch.sum(region_perimeter)
        total_threshold=0
        for i in range(g.n_reg):
            region_threshold[i]=max((region_perimeter[i]/total_perimeter)*g.MIN_PERIMETER_FAC,0.00025)
            if (target_area[i]/tmp_tot_target_area < region_threshold[i]):
                region_small[i]=1
                region_small_ctr+=1
                tot_region_small_area+=target_area[i]
        for i in range(g.n_reg):
            if region_small[i] ==1:
                total_threshold+=region_threshold[i]
        total_threshold_area=(total_threshold*(tmp_tot_target_area-tot_region_small_area))/(1-total_threshold)
        if region_small_ctr > 0:
            print("Enlarging small regions:")

        for i in range(g.n_reg):
            if region_small[i] == 1:
                region_threshold_area[i]=(region_threshold[i]/total_threshold)*total_threshold_area
                old_target_area=target_area[i]
                target_area[i]=region_threshold_area[i]
                tmp_tot_target_area+=target_area[i]
                tmp_tot_target_area-=old_target_area
                print(f"{g.region_id[i]}:{target_area[i]}")

        if region_small_ctr>0:
            print("\n")
        else:
            print("No regions below minimum threshold.")
    else:
        #If -n flag is set, regions with zero area will be replaced by MIN_POP_FAC * min_area
        print("Note: Not using scaled perimeter threshold.\n\n")
        min_area=torch.min(target_area[target_area > 0.0])

        for i in range(g.n_reg):
            if target_area[i] == 0.0:
                target_area[i] = min_area*g.MIN_POP_FAC


    #计算所有的密度
    #人口密度公式：人口密度=某地人口数/该地区土地面积
    dens = target_area / init_area
    tot_target_area = torch.sum(target_area)
    avg_dens=tot_target_area/tot_init_area #计算平均密度

    # print(f"dens grad_fn : {dens.grad_fn}")
    # print(f"tot_target_area grad_fn : {tot_target_area.grad_fn}")
    # print(f"avg_dens grad_fn : {avg_dens.grad_fn}")

    #Digitize the density 输入密度的傅里叶变换。将密度数字化。
    #数字化密度
    # 暂时考虑使用numpy来给rho_ft,rho_init初始化，如果后续使用到需要找一个可以平替fftw的库：pyfftw
    rho_ft = torch.zeros((g.L, g.L), dtype=torch.float64)
    rho_init = torch.full((lx, ly), avg_dens.item(), dtype=torch.float64)
    mask = xyhalfshift2reg != -1
    rho_init[mask] = dens[xyhalfshift2reg[mask].to(torch.long)]

    #用于绘制rho_init里面填充密度的情况
    # import ps_figure as pf
    # pf.new_picture_rho_init()

    # 高斯模糊
    # 平滑密度分布，以避免多边形边缘周围出现不受控制的扭曲。
    rho_init=gaussian_blur(rho_init,rho_ft,lx, ly)
    #print(f"g.rho_init grad_fn : {g.rho_init.grad_fn}")
    # import ps_figure as pf
    # pf.display(g.rho_init)

    # import ps_figure as pf
    # pf.new_picture_rho_ft()

    # # 看看高斯模糊之后的rho_init
    # import ps_figure as pf
    # pf.new_picture_gaussian_blur_rho_init()
    #pf.new_picture_gaussian_blur_rho_init_polycorn(map_file_name)

    # 在列上应用 DCT-II
    rho_ft = dct(rho_init, norm='ortho')
    # 交换维度以在行上应用 DCT-II
    rho_ft = dct(rho_ft.T, norm='ortho').T

    # print(f"g.rho_ft grad_fn : {g.rho_ft.grad_fn}")
    # print(f"g.rho_init grad_fn: {g.rho_init.grad_fn}")

    # import ps_figure as pf
    # pf.display(g.rho_ft)

    # import ps_figure as pf
    # pf.new_picture_rho_ft()

    return False,g_x,g_y,target_area,rho_init,rho_ft,lx, ly

import copy
#Function to fill the lx-times-ly grid with density *after* the first round of integration. The main differences compared to   fill_with_density1() are that fill_with_density2()
def fill_with_density2(target_area,lx, ly):

    import read_map as rm
    for i in range(g.n_poly):
        for j in range(g.n_polycorn[i]):
            g.polycorn[i][j] = g.cartcorn[i][j]
    #g.polycorn = copy.deepcopy(g.cartcorn)

    tmp_area=torch.zeros(g.n_reg,dtype=torch.float64)

    #Determine inside which regions the grid points are located.
    xyhalfshift2reg=interior(lx, ly)

    # 计算每个多边形的面积
    areas = rm.polygon_area(g.polycorn)
    ploys=0

    #Calculate all region areas and densities up to this point in the algorithm.
    for i in range(g.n_reg):
        for j in range(g.n_polyinreg[i]):
            tmp_area[i] += areas[ploys]  # rm.polygon_area(g.n_polycorn[g.polyinreg[i][j]],g.polycorn[g.polyinreg[i][j]])
            ploys += 1

    dens=target_area/tmp_area

    #计算平均密度
    tot_tmp_area = torch.sum(tmp_area)
    tot_target_area = torch.sum(target_area)
    avgDensity = tot_target_area / tot_tmp_area

    rho_init = torch.full((lx, ly), avgDensity.item(),dtype=avgDensity.dtype)
    mask = xyhalfshift2reg != -1
    rho_init[mask] = dens[xyhalfshift2reg[mask].long()]

    # 用于绘制rho_init里面填充密度的情况
    # import ps_figure as pf
    # pf.new_picture_rho_init()

    # 在列上应用 DCT-II
    rho_ft = dct(rho_init, norm='ortho')
    # 交换维度以在行上应用 DCT-II
    rho_ft = dct(rho_ft.T, norm='ortho').T

    return rho_init,rho_ft






