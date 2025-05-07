import torch
import numpy as np
import fill_with_density as fd
import diff_integrate as di
import ffb_integrate as fi
import cartogram as c
import globals as g
import pandas as pd
from math import radians, cos, sin, asin, sqrt
# 定义记录变量值的函数
def record_mae_value(value,path):


    # 新记录
    new_record = pd.DataFrame({'mae': [value]})

    #追加记录
    df=pd.concat([df,new_record],ignore_index=True)

    # 保存数据框到Excel文件
    with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        # 如果文件已存在，则追加数据
        if 'Sheet1' in writer.book.sheetnames:
            existing_df = pd.read_excel(path)
            combined_df = pd.concat([existing_df, new_record], ignore_index=True)
            combined_df.to_excel(writer, index=False, sheet_name='Sheet1')
        else:
            df.to_excel(writer, index=False, sheet_name='Sheet1')

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance

def convert_points(End_x,End_y,latt_const,minx,miny):
    # # 地球半径（米）
    # R = 6378137
    #
    # # 转换为PyTorch张量
    # x_points = End_x * latt_const + minx
    # y_points = End_y * latt_const + miny
    # # 转换为经纬度
    # longitudes = x_points / R
    # latitudes = torch.atan(torch.sinh(y_points / R))
    #
    # # 将弧度转换为度
    # longitudes = torch.rad2deg(longitudes)
    # latitudes = torch.rad2deg(latitudes)
    #
    # # 将经纬度转换为弧度
    # longitudes_rad = torch.deg2rad(longitudes)
    # latitudes_rad = torch.deg2rad(latitudes)
    #
    # # 计算所有点对的经度和纬度差（使用广播机制）
    # lon_diff =  longitudes_rad.unsqueeze(1) -  longitudes_rad
    # lat_diff = latitudes_rad.unsqueeze(1) - latitudes_rad
    #
    # a = torch.sin(lat_diff / 2) ** 2 + torch.cos(latitudes_rad).unsqueeze(1) * torch.cos(latitudes_rad.unsqueeze((0))) * torch.sin(
    #     lon_diff / 2) ** 2
    #
    # # 输出 a 的最小值和最大值
    # sqrt_a = torch.sqrt(torch.clamp_min(a+ 1e-10, min=0.0))
    #
    # # 计算最终距离矩阵（所有点对之间的距离）
    # distance_matrix = 2 * torch.asin(torch.clamp(sqrt_a, min=0.0, max=1.0)) * 6371*1000
    # geo_distance=torch.triu(distance_matrix,diagonal=1)
    # geo_distance=geo_distance+geo_distance.T
    # # 将距离单位转换为公里，若需要
    # geo_distance = torch.round(geo_distance / 1000 * 1000) / 1000  # 保留三位小数

    R = 6378137

    # 转换为PyTorch张量
    x_points = End_x * latt_const + minx
    y_points = End_y * latt_const + miny
    # 转换为弧度
    longitudes = x_points / R
    latitudes = torch.atan(torch.sinh(y_points / R))

    num_points = len(x_points)
    distance_matrix = torch.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i + 1, num_points):
            lon_diff = longitudes[i] - longitudes[j]
            lat_diff = latitudes[i] - latitudes[j]

            a = torch.sin(lat_diff / 2) ** 2 + torch.cos(latitudes[i]) * torch.cos(latitudes[j]) * torch.sin(
                lon_diff / 2) ** 2
            sqrt_a = torch.sqrt(a)
            distance = 2 * torch.asin(sqrt_a) * 6371 * 1000
            # 填充上三角矩阵，考虑对称性
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    kilo_distance = distance_matrix / 1000

    return kilo_distance


def main(population_data_tensor,population_data_id_tensor,iteration,device,args):
    #Read the original polygon coordinates. If there is more than one region, fill the lx-times-ly grid with density and print a map.
    #rho_ft[] will be filled with the Fourier transform of the initial  density.
    #shp_file = '../面状变形图和bokeh系统/Last-work/data/china_provinces_3857.shp'
    #geojson_file = './data/china_provinces_3857.shp'
    MaeExcelPath='./mae.xlsx'

    # df=pd.DataFrame(columns=['mae'])

    diff = False
    eps = False

    #map_file 是处理之后的json_file;csv_file_name存放你想可视化的信息：region id;region data;

    #area_file_name = './output_citys.csv'
    #area_file_name = './output.csv'

    # 目标区域面积，例如你想做人口地图，里面的就是人口数据
    target_area = torch.zeros(g.RIGIONS, dtype=torch.float64).to(device)
    #Read the original polygon coordinates. If there is more than one region, fill the lx-times-ly grid with density and print a map.
    #rho_ft[] will be filled with the Fourier transform of the initial density.
    #rho_ft[] 将用初始密度的傅里叶变换填充。
    BoolVal,g_x,g_y,target_area,rho_init,rho_ft,lx, ly=fd.fill_with_density1(population_data_tensor,population_data_id_tensor,target_area)
    if BoolVal:
        print('WARNING: There is only one region. The output cartogram will')
        print('simply be an affine transformation of the input map')


    #存储的是初始的全部的面积
    init_tot_area = 0
    #max保存的是相对面积误差的最大值
    max,init_tot_area = c.max_area_err(g.polycorn,init_tot_area,target_area)

    print(f"max : {max}")

    if max <= g.MAX_PERMITTED_AREA_ERROR:
        if eps:
            import ps_figure as pf
            pf.figure_cartcorn_coordiante() #暂定
        print('ERROR')
        return 0

    #投影位置
    #g.proj = np.zeros((g.L * g.L, 2), dtype=np.float64)#.tolist()
    g.cartcorn = [torch.zeros((int(g.n_polycorn[i]), 2), dtype=torch.float64) for i in range(g.n_poly)]
    #proj[i*ly+j] will store the current position of the point that started at (i+0.5, j+0.5).
    projinit=torch.zeros((g.L, g.L, 2), dtype=torch.float64)#.tolist()

    i, j = torch.meshgrid(torch.arange(g.L), torch.arange(g.L), indexing='ij')
    # projinit = torch.stack((i + 0.5, j + 0.5), dim=-1).to(torch.float64)
    # projinit = torch.nn.Parameter(projinit)

    projinit[:, :, 0] = i + 0.5
    projinit[:, :, 1] = j + 0.5

    #projinit.requires_grad_()

    proj=projinit

    # print(f"projinit requires_grad : {projinit.requires_grad}")
    # print(f"g.proj requires_grad : {g.proj.requires_grad}")

    #可视化proj
    # import ps_figure as pf
    # pf.new_picture_proj()

    #print("Starting integration 1")

    #原始图像
    # import ps_figure as pf
    # pf.figure_polycorn_coordiante()

    #运动方程的首次积分。
    if not diff:
        proj,projtmp=fi.ffb_integrate(rho_init,rho_ft,proj,lx, ly)
    else:
        di.diff_integrate()
    #FALSE because we do not need to project the graticule.

    g_x,g_y=c.project(g_x,g_y,proj,lx, ly) #FALSE，因为我们不需要投影经纬网。
    # import ps_figure as pf
    # pf.figure_cartcorn()
    # import ps_figure as pf
    # pf.figure_cartcorn_coordiante()

    #init_tot_area记录了地图最开始的面积，cart_tot_area记录了此时制图的面积
    cart_tot_area =0
    mae,cart_tot_area=c.max_area_err(g.cartcorn,cart_tot_area,target_area)
    #记录mae变化的值

    #record_mae_value(mae,MaeExcelPath)

    print(f'max. abs. area error:{mae}')

    g.integration=1

    count=0

    proj2 = torch.zeros((g.L, g.L, 2), dtype=torch.float64)
    #用来记录拐点
    #MAE_turning_point=0
    #torch.autograd.set_detect_anomaly(True)

    while mae > g.MAX_PERMITTED_AREA_ERROR: #g.MAX_PERMITTED_AREA_ERROR
        rho_init2, rho_ft2 = fd.fill_with_density2(target_area, lx, ly)

        count += 1

        projtmp = proj2
        proj2 = proj
        proj = projtmp

        proj=projinit

        g.integration +=1

        #print(f'starting integration {g.integration}\n')
        if not diff:
            proj,projtmp=fi.ffb_integrate(rho_init2,rho_ft2,proj,lx, ly)
        else:
            di.diff_integrate()

        g_x,g_y=c.project(g_x,g_y,proj,lx, ly)
        # if count%20==0:
        #     import ps_figure as pf
        #     pf.figure_cartcorn()

        projtmp=proj
        proj=proj2
        proj2=projtmp

        mae,cart_tot_area=c.max_area_err(g.cartcorn,cart_tot_area,target_area)

        # 记录mae变化的值
        #record_mae_value(mae, MaeExcelPath)
        print(f'max. abs. area error:{mae}')

    #重新缩放所有区域，使其与积分开始前的总面积完美匹配。
    #correction_factor是矫正系数
    correction_factor = np.sqrt(init_tot_area / cart_tot_area)
    #print(f'correction_factor = {correction_factor}')
    for i in range(g.n_poly):
        g.cartcorn[i] = np.array(g.cartcorn[i])
        g.cartcorn[i][:, 0] = correction_factor * (g.cartcorn[i][:, 0] - 0.5 * lx) + 0.5 * lx
        g.cartcorn[i][:, 1] = correction_factor * (g.cartcorn[i][:, 1] - 0.5 * ly) + 0.5 * ly
    #g.x的requires_grad为true，且不是叶子节点
    End_x=correction_factor.item() * (g_x - 0.5 * lx) + 0.5 * lx
    End_y=correction_factor.item() * (g_y - 0.5 * ly) + 0.5 * ly

    import ps_figure as pf
    pf.new_picture_cartcorn(g_x,g_y,args)
    # pf.new_picture_density()
    geo_distance = convert_points(End_x,End_y,g.latt_const,g.new_minx,g.new_miny)
    return geo_distance

import psutil,os
import matplotlib.pyplot as plt
import threading
import time

def monitor_memory(process,memory_usage_list,interval=0.1):
    while True:
        current_memory=process.memory_info().rss/(1024*1024)
        memory_usage_list.append(current_memory)
        time.sleep(interval)

if __name__=='__main__':
    # 获取当前进程的进程ID
    pid = os.getpid()

    process=psutil.Process(pid)

    memory_usage_list=[]

    monitor_thread=threading.Thread(target=monitor_memory,args=(process,memory_usage_list))
    monitor_thread.daemon=True
    monitor_thread.start()

    # record start time
    start = time.time()

    main()

    # Record end time
    end = time.time()


    elapsed_time = end - start

    # Print the difference between start and end time
    print("The time of execution of above program is:", elapsed_time, "s")
    print(f"Max memory usage: {max(memory_usage_list):.2f} MB")
    plt.figure()
    plt.plot(memory_usage_list)
    plt.xlabel('Time (intervals)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.show()

    # # # 使用 cProfile 对 main() 函数进行性能分析,并将结果保存到 'profiling_results' 文件中
    # cProfile.run('m.main()', 'profiling_results')
    # #
    # # # 打印性能分析结果到控制台
    # p = pstats.Stats('profiling_results')
    # #strip_dirs() 用以除去文件名前的路径信息;sort_stats(key,[…]) 用以排序profile的输出;print_stats([restriction,…]) 把Stats报表输出到stdout
    # #‘cumulative’	cumulative time
    # p.strip_dirs().sort_stats('tottime').print_stats() #cumulative：cumtime；calls：ncalls；file：filename
    # os.system('snakeviz profiling_results')






