import geopandas as gpd
import globals as g
import numpy as np
import matplotlib.pyplot as plt
import read_map as rm
import random

def figure_polycorn(g_x,g_y):

    import matplotlib.pyplot as plt
    import networkx as nx

    # 绘制这些点
    fig,ax=plt.subplots()

    temp=rm.polygon_json
    j=0
    for i in range(g.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[g.polycorn[j]]]
        while j + 1 < len(g.polygon_id) and g.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([g.polycorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1
    pp = gpd.GeoDataFrame.from_features(temp['features'])

    # pp.plot(ax=ax,facecolor='none', edgecolor='#1f77b4')
    # ax.scatter(g.x.detach().numpy(),g.y.detach().numpy(),s=5,c='r')
    # ax.set_title("rescale map")
    # plt.show()

    from shapely.validation import make_valid
    # 确保所有几何对象都是有效的
    pp['geometry'] = pp['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

    # Create a graph where each polygon is a node
    G = nx.Graph()
    for idx, row in pp.iterrows():
        G.add_node(idx)

    # Add edges between adjacent polygons
    for idx, row in pp.iterrows():
        for other_idx, other_row in pp.iterrows():
            if idx != other_idx and row['geometry'].touches(other_row['geometry']):
                G.add_edge(idx, other_idx)
    print(G)

    # Apply greedy coloring algorithm
    colors = nx.coloring.greedy_color(G, strategy="largest_first")

    # Define color palette (at least 4 colors for 4-color theorem)
    #color_palette = ['#E7298A', '#7570B3', '#66A61E', '#D95F20']
    #color_palette = ['#E7298A', '#7570B3', '#66A61E','#D95F20','#D95F20','#1B9E77','#D95F02','#E6AB02' ] #'#D95F20','#1B9E77','#D95F02','#E6AB02'
    if g.mode:
        color_map = ['#E7298A', '#66A61E', '#7570B3', '#D95F20', '#E7298A', '#66A61E', '#7570B3', '#66A61E', '#E7298A',
                     '#7570B3', '#66A61E', '#E7298A', '#E7298A', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#E7298A',
                     '#66A61E', '#D95F20', '#E7298A', '#66A61E', '#E7298A', '#7570B3', '#66A61E', '#7570B3', '#7570B3',
                     '#66A61E', '#D95F20', '#D95F20', '#E7298A']
    else:
        color_map = ['#D95F20', '#7570B3', '#66A61E', '#D95F20', '#66A61E', '#D95F20', '#7570B3', '#7570B3', '#66A61E', '#E7298A', '#E7298A', '#66A61E', '#D95F20', '#7570B3', '#D95F20', '#D95F20', '#66A61E', '#E7298A', '#7570B3', '#E7298A', '#66A61E', '#E7298A', '#7570B3', '#66A61E', '#66A61E', '#D95F20', '#E7298A', '#66A61E', '#E7298A', '#7570B3', '#7570B3', '#E7298A', '#E7298A', '#66A61E', '#7570B3', '#66A61E', '#7570B3', '#66A61E', '#E7298A', '#66A61E', '#D95F20', '#7570B3', '#66A61E', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#D95F20', '#7570B3', '#E7298A', '#D95F20', '#7570B3', '#66A61E', '#1B9E77', '#E7298A', '#D95F20', '#7570B3', '#D95F20', '#66A61E', '#E7298A', '#E7298A', '#E7298A', '#E7298A', '#D95F20', '#66A61E', '#7570B3', '#66A61E', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#E7298A']

    # Create a color map for each polygon
    #color_map = [color_palette[colors[i] % len(color_palette)] for i in range(len(pp))]

    pp.plot(ax=ax,facecolor=color_map, edgecolor='#1f77b4')
    ax.scatter(g_x.detach(), g_y.detach(), c='white', s=10, linewidth=1, zorder=3)
    ax.scatter(g_x.detach(), g_y.detach(), c='#B4EBAF', s=2, zorder=4)
    ax.set_title("cartogram")
    plt.savefig(f'./出图/result/result.png', format='png', dpi=300)
    plt.show()

def figure_cartcorn():

    import read_map as rm
    import geopandas as gpd
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    temp = rm.polygon_json
    j = 0
    for i in range(g.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[g.cartcorn[j]]]
        while j + 1 < len(g.polygon_id) and g.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([g.cartcorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1
    pp = gpd.GeoDataFrame.from_features(temp['features'])
    pp.plot(ax=ax,facecolor='none', edgecolor='#1f77b4')
    ax.scatter(g.x, g.y, s=5, c='r')
    plt.show()
def figure_cartcorn_end():
    import main_ as m
    import read_map as rm
    import geopandas as gpd
    import matplotlib.pyplot as plt

    temp = rm.polygon_json
    j = 0
    for i in range(rm.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[m.cartcorn[j]]]
        while j + 1 < len(rm.polygon_id) and rm.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([m.cartcorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1
    pp = gpd.GeoDataFrame.from_features(temp['features'])
    pp.plot()
    plt.show()

def figure_cartcorn_coordiante():

    import matplotlib.pyplot as plt
    #绘制这些点
    for poly in g.cartcorn:
        poly=np.array(poly)
        plt.scatter(poly[:,0],poly[:,1],s=1)
    plt.xlabel('X coordinate')  # x轴标签
    plt.ylabel('Y coordinate')  # y轴标签
    plt.title('Plot of Coordinates')  # 图表标题
    plt.show()  # 显示图表

def figure_polycorn_coordiante():

    import matplotlib.pyplot as plt
    #绘制这些点
    for poly in g.polycorn:
        poly=np.array(poly)
        plt.scatter(poly[:,0],poly[:,1],s=1)
    plt.xlabel('X coordinate')  # x轴标签
    plt.ylabel('Y coordinate')  # y轴标签
    plt.title('Plot of Coordinates')  # 图表标题
    plt.show()  # 显示图表

def figure_gridAndDensity():
    import cartogram as  c
    import main_ as m
    import matplotlib.pyplot as plt
    import fill_with_density as fd
    # c.project(False)

    plt.figure()
    # 绘制这些点
    for poly in m.cartcorn:
        plt.scatter(poly[:, 0], poly[:, 1], s=1)

    # 设置图的大小和网格

    #plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

    # 绘制128x128的网格，并在每个网格中心填写数据
    grid_size = fd.lx
    xlim = plt.xlim()
    ylim = plt.ylim()
    x_step = (xlim[1] - xlim[0]) / grid_size
    y_step = (ylim[1] - ylim[0]) / grid_size

#原来代码,用于绘制grid
def figure_xyhalfshift2reg_grid():
    import read_map as rm
    import matplotlib.pyplot as plt
    import process_json as pj
    import fill_with_density as fd

    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(10 * 3, 6 * 3)  # 设置图像大小

    temp = pj.polygon_json
    j = 0
    for i in range(rm.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[rm.polycorn[j]]]
        while j + 1 < len(rm.polygon_id) and rm.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([rm.polycorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1
    pp = gpd.GeoDataFrame.from_features(temp['features'])
    pp.plot(ax=ax)

    grid_size = fd.lx
    xlim = plt.xlim()
    ylim = plt.ylim()
    x_step = (xlim[1] - xlim[0]) / grid_size
    y_step = (ylim[1] - ylim[0]) / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            x = xlim[0] + i * x_step
            y = ylim[0] + j * y_step
            value = fd.grid[i, j] #density  grid
            # 假设填写的数据为'i,j'
            plt.text(x + x_step / 2, y + y_step / 2, f'{int(value)}', ha='center', va='center', fontsize=5)  # value:.1f
    plt.title("grid")
    plt.show()

def figure_xyhalfshift2reg_density():
    import read_map as rm
    import matplotlib.pyplot as plt
    import process_json as pj
    import fill_with_density as fd

    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(10 * 3, 6 * 3)  # 设置图像大小

    temp = pj.polygon_json
    j = 0
    for i in range(rm.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[rm.polycorn[j]]]
        while j + 1 < len(rm.polygon_id) and rm.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([rm.polycorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1
    pp = gpd.GeoDataFrame.from_features(temp['features'])
    pp.plot(ax=ax)

    grid_size = fd.lx
    xlim = plt.xlim()
    ylim = plt.ylim()
    x_step = (xlim[1] - xlim[0]) / grid_size
    y_step = (ylim[1] - ylim[0]) / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            x = xlim[0] + i * x_step
            y = ylim[0] + j * y_step
            value = fd.density[i, j] #density  grid
            # 假设填写的数据为'i,j'
            plt.text(x + x_step / 2, y + y_step / 2, f'{int(value)}', ha='center', va='center', fontsize=5)  # value:.1f
    plt.title("density")
    plt.show()

def figure_xyhalfshift2reg_gridvxy(count):
    import read_map as rm
    import matplotlib.pyplot as plt
    import process_json as pj
    import fill_with_density as fd
    import numpy as np

    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(10 * 3, 6 * 3)  # 设置图像大小

    temp = pj.polygon_json
    j = 0
    for i in range(rm.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[rm.polycorn[j]]]
        while j + 1 < len(rm.polygon_id) and rm.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([rm.polycorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1
    pp = gpd.GeoDataFrame.from_features(temp['features'])
    pp.plot(ax=ax)

    grid_size = fd.lx
    xlim = plt.xlim() #用于设置x轴的显示范围
    ylim = plt.ylim()
    # print(f'xlim is {xlim}')
    # print(f'ylim is {ylim}')
    x_step = (xlim[1] - xlim[0]) / grid_size
    y_step = (ylim[1] - ylim[0]) / grid_size

    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         x = xlim[0] + i * x_step
    #         y = ylim[0] + j * y_step
    #         value = fd.density[i, j] #density  grid
    #         # 假设填写的数据为'i,j'
    #         plt.text(x + x_step / 2, y + y_step / 2, f'{int(value)}', ha='center', va='center', fontsize=5)  # value:.1f

    # 绘制向量场
    import ffb_integrate as fi
    #从两个一维数组生成两个二维矩阵，这些矩阵对应于第一个一维数组的所有x坐标和第二个一维数组的所有y坐标。
    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1], x_step), np.arange(ylim[0], ylim[1], y_step))
    #plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, fi.gridvx, fi.gridvy, color='r', scale=1) #5 1 15 35

    #plt.title(f"flow vector filed {fi.count}")

    # 保存图片
    #plt.savefig(f'C:\\Users\\1\\Documents\\研究生\\研究生笔记\\人口流动数据的城市间综合距离估算及空间化\\项目复现\\出图结果\\2024-4-10\\output_image {fi.count}.png')
    #plt.show()


    # 绘制向量场
    #plt.figure(figsize=(10, 10))
    #plt.quiver(X, Y, gridvx, gridvy, scale=1, units='xy')
    # plt.xlim(-1, lx)
    # plt.ylim(-1, ly)
    plt.title(f'Vector Field {count}')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid()
    #print(f'count is {fi.count}')
    #plt.savefig(f'C:\\Users\\1\\Desktop\\picture\\gridvxy_output_image {count}.png')

    plt.show()


def figure_xyhalfshift2reg_vxy_intp(count):
    import read_map as rm
    import matplotlib.pyplot as plt
    import process_json as pj
    import fill_with_density as fd
    import numpy as np

    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(10 * 3, 6 * 3)  # 设置图像大小

    temp = pj.polygon_json
    j = 0
    for i in range(rm.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[rm.polycorn[j]]]
        while j + 1 < len(rm.polygon_id) and rm.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([rm.polycorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1
    pp = gpd.GeoDataFrame.from_features(temp['features'])
    pp.plot(ax=ax)

    grid_size = fd.lx
    xlim = plt.xlim()
    ylim = plt.ylim()
    # print(f'xlim is {xlim}')
    # print(f'ylim is {ylim}')
    x_step = (xlim[1] - xlim[0]) / grid_size
    y_step = (ylim[1] - ylim[0]) / grid_size

    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         x = xlim[0] + i * x_step
    #         y = ylim[0] + j * y_step
    #         value = fd.density[i, j] #density  grid
    #         # 假设填写的数据为'i,j'
    #         plt.text(x + x_step / 2, y + y_step / 2, f'{int(value)}', ha='center', va='center', fontsize=5)  # value:.1f

    # 绘制向量场
    import ffb_integrate as fi
    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1], x_step), np.arange(ylim[0], ylim[1], y_step))
    plt.quiver(X, Y, fi.vx_intp, fi.vy_intp, color='r', scale=35) #5 1 15 35

    plt.title(f"vxy_intp flow vector filed {count}")

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid()
    # 保存图片
    #plt.savefig(f'C:\\Users\\1\\Documents\\研究生\\研究生笔记\\人口流动数据的城市间综合距离估算及空间化\\项目复现\\出图结果\\2024-4-10\\vxy_intp_output_image {count}.png')
    plt.show()

#改良版代码，用于绘制density和向量场图
def figure_xyhalfshift2reg(count):
    import read_map as rm
    import matplotlib.pyplot as plt
    import process_json as pj
    import fill_with_density as fd
    import numpy as np

    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(10 * 3, 6 * 3)  # 设置图像大小

    # 绘制这些点
    # fig =plt.figure(figsize=(10 * 2, 6 * 2))

    temp = pj.polygon_json
    j = 0
    for i in range(rm.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[rm.polycorn[j]]]
        while j + 1 < len(rm.polygon_id) and rm.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([rm.polycorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1
    pp = gpd.GeoDataFrame.from_features(temp['features'])
    pp.plot(ax=ax)

    grid_size = fd.lx
    xlim = plt.xlim()
    ylim = plt.ylim()
    x_step = (xlim[1] - xlim[0]) / grid_size
    y_step = (ylim[1] - ylim[0]) / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            x = xlim[0] + i * x_step
            y = ylim[0] + j * y_step
            value = fd.density[i, j] #density  grid
            # 假设填写的数据为'i,j'
            plt.text(x + x_step / 2, y + y_step / 2, f'{int(value)}', ha='center', va='center', fontsize=5)  # value:.1f

    # 绘制向量场
    import ffb_integrate as fi
    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1], x_step), np.arange(ylim[0], ylim[1], y_step))
    plt.quiver(X, Y, fi.gridvx, fi.gridvy, color='r', scale=5)

    plt.title(f"gridvx flow vector filed {count}")

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid()
    plt.savefig(f'C:\\Users\\1\\Documents\\研究生\\研究生笔记\\人口流动数据的城市间综合距离估算及空间化\\项目复现\\出图结果\\2024-4-10\\gridvxy宏观\\gridvxy_output_image {count}.png')
    plt.show()

def figure_xyhalfshift2reg2():
    import read_map as rm
    import matplotlib.pyplot as plt
    import process_json as pj
    import fill_with_density as fd


    grid_size = fd.lx
    xlim = plt.xlim()
    ylim = plt.ylim()
    x_step = (xlim[1] - xlim[0]) / grid_size
    y_step = (ylim[1] - ylim[0]) / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            x = xlim[0] + i * x_step
            y = ylim[0] + j * y_step
            value = fd.grid[i, j]
            # 假设填写的数据为'i,j'
            plt.text(x + x_step / 2, y + y_step / 2, f'{int(value)}', ha='center', va='center', fontsize=5)  # value:.1f

    plt.show()

#用于可视化xyhalfshift2reg中每个格子表示哪个区域
def new_picture_xyhalfshift2reg():
    # 对xyhalfshift2reg逆时针旋转90度
    # matrix = np.array(g.xyhalfshift2reg)
    # rotated_matrix = np.rot90(matrix, k=1)
    # g.xyhalfshift2reg = rotated_matrix.tolist()

    # 可视化xyhalfshift2reg
    xyhalfshift2reg = g.xyhalfshift2reg.detach()
    array = np.array(xyhalfshift2reg)
    # 创建热图
    plt.imshow(array, cmap="viridis", interpolation='none')
    # 添加颜色条
    plt.colorbar()
    # 保存图片
    plt.savefig(f'./出图/xyhalfshift2reg/heatmap{g.integration}.png', format='png', dpi=300)

    plt.title("xyhalfshift2reg picture")

    # 显示热图
    plt.show()

    # 关闭当前图表以防止重叠
    plt.close()

#可视化速度场gridvx，gridvy的每个格子的合向量
def new_picture_gridv():
    g.gridv_count += 1

        # 可视化xyhalfshift2reg
    array = np.array(g.xyhalfshift2reg)

        # 创建一个新的图形和轴
        #fig 和 ax 分别是 Figure 对象和 Axes 对象。Figure 是整个图形的容器，包含所有的绘图元素；Axes 是一个子区域，其中包含实际的绘图
    fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制热图
    heatmap = ax.imshow(array, cmap="viridis", interpolation='none')

        # 添加颜色条
    plt.colorbar(heatmap, ax=ax)

        # 创建网格点
    x = np.arange(0, g.L, 1)
    y = np.arange(0,  g.L, 1)
    X, Y = np.meshgrid(x, y)

    #因为gridvx,gridvy的值太小了，画图看不到
    gridvx=g.gridvx*1000
    gridvy=g.gridvy*1000


    # 绘制向量场
    ax.quiver(X, Y, gridvx, gridvy, scale=1, scale_units='xy', angles='xy')

    plt.title("gridv picture")

    # 保存图片
    #integration记录是第几轮，count记录是多少个
    plt.savefig(f'./出图/gridv/gridv-{g.integration}-{g.gridv_count}.png', format='png', dpi=300)



    # 显示图形
    #plt.show()

    plt.close()
    g.gridv_count+=1

def test_gridv(gridvx,gridvy):
    # # 创建热图
    # plt.imshow(gridv, cmap="viridis", interpolation='none')
    # # 添加颜色条
    # plt.colorbar()
    #
    # plt.title("grid_flux_init picture")
    #
    # plt.show()

    # 计算两个分量的向量和
    vector_sum = np.sqrt(gridvx ** 2 + gridvy ** 2)

    # 创建热图
    plt.imshow(vector_sum, cmap="viridis", interpolation='none')
    # 添加颜色条
    plt.colorbar()

    # 设置图像标题
    plt.title("Vector Sum Magnitude Heatmap")

    # 显示图像
    plt.show()

#可视化速度场gridvx，gridvy的每个格子的合向量
def new_picture_vintp(vx_intp,vy_intp):
    # 可视化xyhalfshift2reg
    array = np.array(g.xyhalfshift2reg)

    # 创建一个新的图形和轴
    #fig 和 ax 分别是 Figure 对象和 Axes 对象。Figure 是整个图形的容器，包含所有的绘图元素；Axes 是一个子区域，其中包含实际的绘图
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制热图
    heatmap = ax.imshow(array, cmap="viridis", interpolation='none')

    # 添加颜色条
    plt.colorbar(heatmap, ax=ax)

    # 创建网格点
    x = np.arange(0, g.L, 1)
    y = np.arange(0,  g.L, 1)
    X, Y = np.meshgrid(x, y)

    # 绘制向量场
    ax.quiver(X, Y, vx_intp, vy_intp, scale=1, scale_units='xy', angles='xy')

    plt.title("vintp picture")

    # 保存图片
    #integration记录是第几轮，count记录是多少个
    plt.savefig(f'./出图/vintp/vintp-{g.integration}-{g.vintp_count}.png', format='png', dpi=300)
    g.vintp_count+=1

    # 显示图形
    #plt.show()

    plt.close()


#用于可视化rho_init中每个格子表示值
def new_picture_rho_init():
    # 打印rho_init
    display(g.rho_init)

    # 创建热图
    plt.imshow(g.rho_init.detach(), cmap="viridis", interpolation='none') #plasma viridis 'inferno'、'magma' tab10
    # 添加颜色条
    plt.colorbar()

    plt.title("rho_init picture tensor")

    # 保存图片
    # integration记录是第几轮，count记录是多少个
    plt.savefig(f'./出图/rho_init/rho_init-{g.integration}-{g.rho_init_count}.png', format='png', dpi=300)
    g.rho_init_count+=1
    # 显示热图
    plt.show()

    # 关闭当前图表以防止重叠
    plt.close()

#用于可视化高斯模糊之后的rho_init中每个格子表示值
def new_picture_gaussian_blur_rho_init():
    # #没有叠加底图rho_init的代码
    # # 创建热图
    # plt.imshow(g.rho_init, cmap="viridis", interpolation='none')
    # # 添加颜色条
    # plt.colorbar()

    # 叠加底图rho_init的代码
    # 绘制叠加图像
    fig, ax = plt.subplots()

    # 底图
    array = np.array(g.xyhalfshift2reg.detach())
    ax.imshow(array, cmap="viridis", interpolation='none')

    # 叠加图
    im = ax.imshow(g.rho_init.detach(), cmap="viridis", alpha=0.5)

    # 添加颜色条
    fig.colorbar(im, ax=ax)

    plt.title("gaussian_blur_rho_init picture tensor")

    # 保存图片
    # integration记录是第几轮，count记录是多少个
    plt.savefig(f'./出图/gaussian_blur_rho_init/gaussian_blur_rho_init-{g.integration}-{g.gaussian_blur_rho_init_count}.png', format='png', dpi=300)
    # g.gaussian_blur_rho_init_count+=1
    # 显示热图
    plt.show()

    # 关闭当前图表以防止重叠
    plt.close()


def new_picture_gaussian_blur_rho_init_polycorn(polygon_json):

    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.affinity import scale
    from shapely.affinity import rotate

    # 绘制多边形边界
    temp = polygon_json
    j = 0
    for i in range(g.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[g.polycorn[j]]]
        while j + 1 < len(g.polygon_id) and g.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([g.polycorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1

    pp = gpd.GeoDataFrame.from_features(temp['features'])

    # # 获取图像中心坐标
    # center_x = np.mean([point[0] for poly in g.polycorn for point in poly])
    # center_y = np.mean([point[1] for poly in g.polycorn for point in poly])
    # center = (center_x, center_y)
    #
    # # 对多边形进行旋转
    # pp['geometry'] = pp['geometry'].apply(lambda geom: rotate(geom, angle=180, origin=center))

    # 获取图像的 y 坐标中心
    min_y, max_y = pp.total_bounds[1], pp.total_bounds[3]
    center_y = (min_y + max_y) / 2

    # 对多边形进行上下翻转
    pp['geometry'] = pp['geometry'].apply(lambda geom: scale(geom, xfact=1, yfact=-1, origin=(0, center_y)))

    # 创建绘图
    fig, ax = plt.subplots()

    # 绘制多边形边界
    pp.plot(ax=ax, facecolor='none', edgecolor='#1f77b4')

    # 绘制 rho_init 图像
    im = ax.imshow(g.rho_init, cmap="viridis", alpha=0.5)

    # 添加颜色条
    fig.colorbar(im, ax=ax)

    plt.show()
import torch
# 设置 PyTorch 的输出精度，保留 8 位小数
torch.set_printoptions(precision=8)
def display(arr):
    for i in range(128):
        for j in range(128):
            print(f"{arr[i,j]}({i},{j}) ",end="")
        print("\n")
#用于可视化rho_ft中每个格子表示值
def new_picture_rho_ft():
    # 创建热图
    plt.imshow(g.rho_ft.detach(), cmap="viridis", interpolation='none')
    # 添加颜色条
    plt.colorbar()

    plt.title("rho_ft picture tensor")

    # 保存图片
    # integration记录是第几轮，count记录是多少个
    # plt.savefig(f'./出图/rho_ft/rho_ft-{g.integration}-{g.rho_ft_count}.png', format='png', dpi=300)
    # g.rho_ft_count+=1
    # 显示热图
    plt.show()

    # 关闭当前图表以防止重叠
    plt.close()

#用于可视化grid_flux_init中每个格子表示值
def new_picture_grid_flux_init(grid_flux_init):
    # 创建热图
    plt.imshow(grid_flux_init.detach(), cmap="viridis", interpolation='none')
    # 添加颜色条
    plt.colorbar()

    plt.title("grid_flux_init picture")

    # 保存图片
    # integration记录是第几轮，count记录是多少个
    # plt.savefig(f'./出图/rho_ft/rho_ft-{g.integration}-{g.rho_ft_count}.png', format='png', dpi=300)
    # g.rho_ft_count+=1
    # 显示热图
    plt.show()

    # 关闭当前图表以防止重叠
    plt.close()

#用于可视化在xyhalfshift2reg上的proj中每个格子表示值
def new_picture_proj():
    # 可视化xyhalfshift2reg
    array = np.array(g.xyhalfshift2reg)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 10))

    # 创建热图
    heatmap=ax.imshow(array, cmap="viridis", interpolation='none')

    # 添加颜色条
    plt.colorbar(heatmap,ax=ax)

    # 提取 x 和 y 坐标
    x = g.proj[:, :, 0]
    y = g.proj[:, :, 1]

    # 绘制散点图
    ax.scatter(x, y, s=0.1, c='red', alpha=1)

    # 设置标题和标签
    plt.title('Visualization of proj Data')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # # 设置轴的比例相同
    plt.gca().set_aspect('equal', adjustable='box')

    # 设置背景颜色为白色
    plt.gca().set_facecolor('white')

    # 保存图片
    # integration记录是第几轮，count记录是多少个
    plt.savefig(f'./出图/proj/proj-{g.integration}-{g.proj_count}.png', format='png', dpi=300)

    # 显示图形
    #plt.show()

    plt.close()

#绘制eul的结果
def new_picture_EulAndMid(eul,ch):
    # 可视化xyhalfshift2reg
    array = np.array(g.xyhalfshift2reg)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 10))

    # 创建热图
    heatmap=ax.imshow(array, cmap="viridis", interpolation='none')

    # 添加颜色条
    plt.colorbar(heatmap,ax=ax)

    # 提取 x 和 y 坐标
    x = eul[:,:, 0].flatten()
    y = eul[:,:, 1].flatten()

    # 绘制散点图
    ax.scatter(x, y, s=0.1, c='red', alpha=1)

    if ch=='e':
        # 设置标题和标签
        plt.title('Visualization of eul Data')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        # # 设置轴的比例相同
        plt.gca().set_aspect('equal', adjustable='box')

        # 设置背景颜色为白色
        plt.gca().set_facecolor('white')

        # 保存图片
        # integration记录是第几轮，count记录是多少个
        plt.savefig(f'./出图/eul/eul-{g.integration}-{g.eul_count}.png', format='png', dpi=300)
        g.eul_count+=1
    elif ch=='m':
        # 设置标题和标签
        plt.title('Visualization of mid Data')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        # # 设置轴的比例相同
        plt.gca().set_aspect('equal', adjustable='box')

        # 设置背景颜色为白色
        plt.gca().set_facecolor('white')

        # 保存图片
        # integration记录是第几轮，count记录是多少个
        plt.savefig(f'./出图/mid/mid-{g.integration}-{g.mid_count}.png', format='png', dpi=300)
        g.mid_count += 1
    # 显示图形
    #plt.show()

    plt.close()
#绘制出图的密度分布
def new_picture_density():
    # 方法1：绘制热图
    # 创建热图
    # plt.imshow(g.rho_init, cmap="viridis", interpolation='none')
    # # 添加颜色条
    # plt.colorbar()
    #
    # plt.title('Density Matrix Heatmap')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.savefig(f'./出图/result/Density_Matrix_Heatmap.png', format='png', dpi=300)
    # #plt.show()

    # 方法2：绘制直方图
    plt.figure(figsize=(8, 6))
    plt.hist(g.rho_init.flatten(), bins=20, color='skyblue', edgecolor='black')
    plt.title('Density Distribution Histogram')
    plt.xlabel('Density Value')
    plt.ylabel('Frequency')
    plt.savefig(f'./出图/result/Density_Distribution_Histogram.png', format='png', dpi=300)
    #plt.show()

    # 方法3：统计摘要
    mean_density = np.mean(g.rho_init)
    median_density = np.median(g.rho_init)
    std_density = np.std(g.rho_init)

    print(f"Mean Density: {mean_density}")
    print(f"Median Density: {median_density}")
    print(f"Standard Deviation of Density: {std_density}")

def new_picture_cartcorn(g_x,g_y,args):

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import networkx as nx

    fig,ax=plt.subplots()
    temp = rm.polygon_json
    j = 0
    for i in range(g.n_reg):
        temp['features'][i]['geometry']['coordinates'] = [[g.cartcorn[j]]]
        while j + 1 < len(g.polygon_id) and g.polygon_id[j + 1] == i:
            temp['features'][i]['geometry']['coordinates'].append([g.cartcorn[j + 1]])
            j += 1

        temp['features'][i]['geometry']['type'] = 'MULTIPOLYGON'
        j += 1
    pp = gpd.GeoDataFrame.from_features(temp['features'])
    from shapely.validation import make_valid
    # 确保所有几何对象都是有效的
    pp['geometry'] = pp['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

    # Create a graph where each polygon is a node
    G = nx.Graph()
    for idx, row in pp.iterrows():
        G.add_node(idx)

    # Add edges between adjacent polygons
    for idx, row in pp.iterrows():
        for other_idx, other_row in pp.iterrows():
            if idx != other_idx and row['geometry'].touches(other_row['geometry']):
                G.add_edge(idx, other_idx)


    # Apply greedy coloring algorithm
    colors = nx.coloring.greedy_color(G, strategy="largest_first")

    # Define color palette (at least 4 colors for 4-color theorem)
    #color_palette = ['#E7298A', '#7570B3', '#66A61E', '#D95F20']
    color_palette = ['#E7298A', '#7570B3', '#66A61E', '#D95F20','#1B9E77','#D95F02','#E6AB02']# ,'#1B9E77','#D95F02','#E6AB02'

    # Create a color map for each polygon
    # color_map = [color_palette[colors[i] % len(color_palette)] for i in range(len(pp))]
    # print(color_map)
    # print(len(color_map))
    if g.mode:
        color_map = ['#E7298A', '#66A61E', '#7570B3', '#D95F20', '#E7298A', '#66A61E', '#7570B3', '#66A61E', '#E7298A',
                     '#7570B3', '#66A61E', '#E7298A', '#E7298A', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#E7298A',
                     '#66A61E', '#D95F20', '#E7298A', '#66A61E', '#E7298A', '#7570B3', '#66A61E', '#7570B3', '#7570B3',
                     '#66A61E', '#D95F20', '#D95F20', '#E7298A']
    else:
        color_map = ['#D95F20', '#7570B3', '#66A61E', '#D95F20', '#66A61E', '#7570B3', '#D95F20', '#7570B3', '#66A61E', '#E7298A', '#E7298A', '#66A61E', '#1B9E77', '#7570B3', '#D95F20', '#D95F20', '#66A61E', '#D95F20', '#7570B3', '#E7298A', '#66A61E', '#E7298A', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#D95F20', '#66A61E', '#E7298A', '#7570B3', '#7570B3', '#66A61E', '#E7298A', '#66A61E', '#7570B3', '#E7298A', '#7570B3', '#66A61E', '#E7298A', '#66A61E', '#D95F20', '#7570B3', '#66A61E', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#D95F20', '#7570B3', '#E7298A', '#D95F20', '#7570B3', '#66A61E', '#1B9E77', '#E7298A', '#D95F20', '#7570B3', '#D95F20', '#66A61E', '#E7298A', '#E7298A', '#E7298A', '#E7298A', '#D95F20', '#66A61E', '#7570B3', '#66A61E', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#E7298A', '#66A61E', '#7570B3', '#66A61E', '#E7298A', '#D95F20', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#7570B3', '#66A61E', '#1B9E77', '#E7298A', '#7570B3', '#7570B3', '#7570B3', '#E7298A', '#E7298A', '#D95F20', '#66A61E', '#E7298A', '#66A61E', '#E7298A', '#D95F20', '#7570B3', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#E7298A', '#7570B3', '#E7298A', '#D95F20', '#D95F20', '#E7298A', '#7570B3', '#D95F20', '#E7298A', '#1B9E77', '#66A61E', '#E7298A', '#66A61E', '#7570B3', '#7570B3', '#7570B3', '#E7298A', '#66A61E', '#E7298A', '#D95F20', '#D95F20', '#D95F20', '#E7298A', '#1B9E77', '#E7298A', '#E7298A', '#D95F20', '#E7298A', '#D95F20', '#7570B3', '#66A61E', '#7570B3', '#66A61E', '#7570B3', '#E7298A', '#66A61E', '#7570B3', '#E7298A', '#66A61E', '#D95F20', '#1B9E77', '#7570B3', '#E7298A', '#7570B3', '#7570B3', '#E7298A', '#D95F20', '#7570B3', '#66A61E', '#D95F20', '#7570B3', '#D95F20', '#D95F20', '#7570B3', '#E7298A', '#1B9E77', '#66A61E', '#7570B3', '#66A61E', '#D95F20', '#E7298A', '#66A61E', '#66A61E', '#E7298A', '#1B9E77', '#E7298A', '#66A61E', '#D95F20', '#7570B3', '#66A61E', '#D95F20', '#E7298A', '#7570B3', '#E7298A', '#E7298A', '#7570B3', '#7570B3', '#1B9E77', '#7570B3', '#7570B3', '#66A61E', '#D95F20', '#1B9E77', '#E7298A', '#66A61E', '#7570B3', '#E7298A', '#D95F20', '#66A61E', '#D95F20', '#1B9E77', '#7570B3', '#7570B3', '#66A61E', '#E7298A', '#66A61E', '#66A61E', '#7570B3', '#66A61E', '#7570B3', '#7570B3', '#7570B3', '#66A61E', '#E7298A', '#E7298A', '#66A61E', '#D95F20', '#E7298A', '#7570B3', '#66A61E', '#D95F20', '#D95F20', '#E7298A', '#66A61E', '#D95F20', '#E7298A', '#D95F20', '#7570B3', '#E7298A', '#1B9E77', '#7570B3', '#E7298A', '#D95F20', '#7570B3', '#66A61E', '#1B9E77', '#7570B3', '#7570B3', '#1B9E77', '#66A61E', '#D95F20', '#66A61E', '#66A61E', '#D95F20', '#E7298A', '#66A61E', '#66A61E', '#7570B3', '#D95F20', '#D95F20', '#7570B3', '#E7298A', '#66A61E', '#7570B3', '#E7298A', '#7570B3', '#D95F20', '#E7298A', '#66A61E', '#7570B3', '#E7298A', '#E7298A', '#7570B3', '#66A61E', '#D95F20', '#7570B3', '#D95F20', '#66A61E', '#D95F20', '#7570B3', '#D95F20', '#7570B3', '#E7298A', '#E7298A', '#D95F20', '#66A61E', '#7570B3', '#66A61E', '#66A61E', '#66A61E', '#E7298A', '#7570B3', '#E7298A', '#1B9E77', '#1B9E77', '#66A61E', '#7570B3', '#D95F20', '#D95F20', '#66A61E', '#7570B3', '#E7298A', '#66A61E', '#E7298A', '#D95F20', '#E7298A', '#1B9E77', '#66A61E', '#E7298A', '#7570B3', '#7570B3', '#7570B3', '#66A61E', '#7570B3', '#D95F20', '#7570B3', '#7570B3', '#D95F20', '#66A61E', '#E7298A', '#66A61E', '#E7298A', '#7570B3', '#7570B3', '#66A61E', '#66A61E', '#D95F20', '#D95F20', '#7570B3', '#E7298A', '#66A61E', '#E7298A', '#1B9E77', '#D95F20', '#1B9E77', '#D95F20', '#E7298A', '#D95F20', '#7570B3', '#1B9E77', '#66A61E', '#7570B3', '#E7298A', '#D95F20', '#D95F20', '#66A61E', '#7570B3', '#7570B3', '#D95F20', '#66A61E', '#E7298A', '#D95F20', '#66A61E', '#7570B3', '#1B9E77', '#D95F20', '#66A61E', '#D95F20', '#66A61E', '#66A61E', '#66A61E', '#1B9E77', '#66A61E', '#E7298A', '#D95F20', '#66A61E', '#7570B3', '#E7298A', '#E7298A', '#7570B3', '#66A61E', '#E7298A', '#D95F20', '#D95F20', '#66A61E', '#E7298A', '#E7298A', '#E7298A', '#7570B3', '#E7298A', '#7570B3', '#7570B3', '#7570B3', '#E7298A', '#E7298A', '#E7298A', '#E7298A', '#E7298A']

    pp.plot(ax=ax,facecolor=color_map, edgecolor='#1f77b4')
    ax.scatter(g_x.detach(), g_y.detach(), c='white', s=10, linewidth=1, zorder=3)
    ax.scatter(g_x.detach(), g_y.detach(), c='#B4EBAF', s=2, zorder=4)
    ax.set_title("cartogram")

    # 固定坐标范围
    ax.set_xlim([15, 115])
    ax.set_ylim([15, 115])

    plt.savefig(f'./出图/result/citys/citys_result{args}.png', format='png', dpi=300)
    #plt.show()

def new_picture_result(svg_file):
    from svgpathtools import svg2paths2
    import matplotlib.patches as patches
    # 读取SVG文件
    paths, attributes, svg_attributes = svg2paths2(svg_file)

    fig, ax = plt.subplots()

    # 解析每一个路径
    for path in paths:
        for segment in path:
            if segment.__class__.__name__ == 'Line':
                line = patches.FancyArrowPatch((segment.start.real, segment.start.imag),
                                               (segment.end.real, segment.end.imag),
                                               mutation_scale=1, color='black')
                ax.add_patch(line)
            elif segment.__class__.__name__ == 'CubicBezier':
                verts = [(segment.start.real, segment.start.imag),
                         (segment.control1.real, segment.control1.imag),
                         (segment.control2.real, segment.control2.imag),
                         (segment.end.real, segment.end.imag)]
                codes = [patches.Path.MOVETO,
                         patches.Path.CURVE4,
                         patches.Path.CURVE4,
                         patches.Path.CURVE4]
                path_data = list(zip(codes, verts))
                path_patch = patches.PathPatch(patches.Path(*zip(*path_data)), facecolor='none', edgecolor='black')
                ax.add_patch(path_patch)
            # 可以扩展到其他类型的SVG路径段

    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('SVG Visualization')
    plt.show()

def picture_targetAndDistance(geo_distance_normalized,target_distance,error,args):
    from scipy.stats import gaussian_kde
    mask_distance=torch.triu(torch.ones_like(geo_distance_normalized,dtype=bool),diagonal=1)
    mask_target = torch.triu(torch.ones_like(target_distance, dtype=bool),diagonal=1)

    uper_distance_element=geo_distance_normalized[mask_distance]
    uper_target_element=target_distance[mask_target]


    # 转为 NumPy 数组
    uper_distance_array = uper_distance_element.detach().numpy()
    uper_target_array = uper_target_element.detach().numpy()

    # 计算点的密度
    xy = np.vstack([uper_target_array, uper_distance_array])
    kde = gaussian_kde(xy)(xy)  # 使用高斯核密度估计
    # 对密度进行归一化，映射到颜色范围
    kde_normalized = (kde - kde.min()) / (kde.max() - kde.min())

    fig,ax=plt.subplots(figsize=(10,6),dpi=300)
    # 绘制散点图，使用颜色表示密度
    x=np.arange(0,1.1,0.1)
    y=x
    plt.plot(x,y,'k-')
    scatter=plt.scatter(uper_target_array,uper_distance_array,s=10,c=kde_normalized,cmap='viridis')

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Density')

    ax.set_aspect('auto')

    plt.xlabel('target_distance')
    plt.ylabel('geo_distance')
    plt.title(f'Error: {error}')

    # 固定坐标范围
    ax.set_xlim([-500, 17000])
    ax.set_ylim([-200, 5000])

    if g.mode:
        plt.savefig(f'./出图/result/province_+500/targetAndDistance_result.png', format='png', dpi=300)
    else:
        plt.savefig(f'./出图/result/citys/targetAndDistance_result{args}.png', format='png', dpi=300)
    #plt.show()