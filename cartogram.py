import fill_with_density as fd

import read_map as rm
import main_ as m
import numpy as np
import torch
import globals as g

#Function to project the polygons in the input .gen file.
def project(g_x,g_y,proj,lx, ly):
    import ffb_integrate as fi
    #The displacement vector (xdisp[i*ly+j], ydisp[i*ly+j]) is the point that was initially at (i+0.5, j+0.5). We work with (xdisp, ydisp)
    #instead of (proj.x, proj.y) so that we can use the function interpol() defined in integrate.c.
    #位移向量 (xdisp[i*ly+j], ydisp[i*ly+j]) 是最初位于 (i+0.5, j+0.5) 的点。
    xdisp=torch.zeros((lx, ly), dtype=torch.float64)
    ydisp =torch.zeros((lx, ly), dtype=torch.float64)

    # 使用NumPy的向量化操作进行赋值
    i, j = torch.meshgrid(torch.arange(lx), torch.arange(ly), indexing='ij')
    #i, j = torch.meshgrid(torch.arange(g.L), torch.arange(g.L), indexing='ij')
    xdisp = proj[:,:, 0] - i - 0.5
    ydisp = proj[:,:, 1] - j - 0.5

    # 将所有多边形坐标转换为NumPy数组torch.cat
    # all_coords = np.concatenate(g.polycorn)
    g_polycorn_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in g.polycorn]
    all_coords = torch.cat(g_polycorn_tensors, dim=0)
    x_coords = all_coords[:, 0]
    y_coords = all_coords[:, 1]

    # 批量插值计算
    x_interpolated = fi.interpol(x_coords, y_coords, xdisp, 'x',lx, ly)
    y_interpolated = fi.interpol(x_coords, y_coords, ydisp, 'y',lx, ly)

   # print(f"x_interpolated requires_grad : {x_interpolated.requires_grad}")
    # 投影坐标
    projected_coords = all_coords + torch.vstack((x_interpolated, y_interpolated)).T
    # all_coords = torch.from_numpy(all_coords).to(x_interpolated.device)
    # projected_coords = all_coords + torch.stack((x_interpolated, y_interpolated),dim=-1)

    x_intep_point = fi.interpol(g_x, g_y, xdisp, 'x',lx, ly)
    y_intep_point = fi.interpol(g_x, g_y, ydisp, 'y',lx, ly)

    g_x = g_x + x_intep_point
    g_y = g_y + y_intep_point

    # print(f"g.x requires_grad : {g.x.grad_fn}")
    # print(f"g.y requires_grad : {g.y.grad_fn}")

    # 更新g.cartcorn
    start_idx = 0
    for i in range(g.n_poly):
        end_idx = start_idx + g.n_polycorn[i]
        g.cartcorn[i] = projected_coords[start_idx:end_idx].tolist()
        start_idx = end_idx
    return g_x,g_y


#Function to return the maximum absolute relative area error. The relative area error is defined by:   area_on_cartogram / target_area - 1.
#The function also updates the arrays cart_area[] and area_err[] that are  passed by reference.
def max_area_err(corn,sum_cart_area,target_area):
    #cart_area记录了变形后cartogram图，各个区域面积的大小
    areas = rm.polygon_area(corn)
    # 记录的是当前各个区域地图的面积
    cart_area = torch.zeros(g.RIGIONS, dtype=torch.float64)
    ploys = 0
    for i in range(g.n_reg):
        cart_area[i]=torch.tensor(0.0)
        for j in range(g.n_polyinreg[i]):
            cart_area[i]+=areas[ploys]
            ploys += 1
    #sum_target_area 统计的是未缩放和重定位时各个区域的人口之和
    sum_target_area = torch.sum(target_area)

    #print(f"sum_target_area requires_grad : {sum_target_area.requires_grad}")

    #sum_cart_area统计的是cartogram图各个区域面积之和
    sum_cart_area = torch.sum(cart_area)

    #Objective area in cartogram units.
    # for i in range(g.n_reg):
    #     #(sum_cart_area) / sum_target_area得到了放缩的倍数，obj_area得到每个区域的目标面积
    #     obj_area=g.target_area[i] * (sum_cart_area) / sum_target_area #obj_area = 区域人口 * 总面积 / 总人口 表示最终根据人口这个区域应该占据多少面积
    #     area_err[i] = cart_area[i] / obj_area - 1.0 #用于测算实际面积与目标面积之间的误差
    obj_area = target_area * (sum_cart_area / sum_target_area)
    # 相对面积误差，相对面积误差定义：area_on_cartogram/target_area - 1
    area_err = torch.zeros(g.RIGIONS, dtype=torch.float64)
    area_err[:] = cart_area / obj_area - 1.0

    # print(f"obj_area grad_fn : {obj_area.grad_fn}")
    # print(f"g.area_err grad_fn: {g.area_err.grad_fn}")

    #print(f"g.area_err requires_grad : {g.area_err.requires_grad}")

    max = torch.max(torch.abs(area_err))

    #print(f"max requires_grad : {max.requires_grad}")

    return max,sum_cart_area

