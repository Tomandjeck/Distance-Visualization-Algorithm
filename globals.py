import numpy as np
import pyfftw
import torch

#确定的最小面积阈值
MIN_PERIMETER_FAC=0.025
#将区域 0 替换为该值的最小值。
MIN_POP_FAC=0.2
#用于平滑密度的高斯模糊宽度
BLUR_WIDTH=0.8 #增大这个值会使模糊效果更强，减小这个值会使模糊效果减弱。
#地图与边框之间的间距
PADDING=1.5 #2.5 1.08
#四图书的区域最多相差绝对误差
MAX_PERMITTED_AREA_ERROR=5#0.01 41.5->20->10->5->1->0.01 0.8 1 1.4 0.61 4.02  7
#绘制格子的多少
L=128#Maximum dimension of the FFT lattice is L x L 512 256 128 64
#用于记录缩放之后又在变形之前的 polygon 坐标
origcorn=[]

#原始地图的四个点坐标的值
map_maxx=0
map_maxy=0
map_minx=0
map_miny=0

new_maxx=0
new_minx=0
new_maxy=0
new_miny=0

latt_const=0

#记录有多少条线
n_poly=0
# 计算顶点数，记录了每条线的点数
n_polycorn= []
# 存储顶点，记录了每条线具体的点坐标
polycorn= []
#记录当前线条属于的多边形id号
polygon_id= []

#记录区域的数量
n_reg = 0
RIGIONS=369 #31 369 72
#记录区域的ID
region_id=np.zeros(RIGIONS,dtype=int)
region_na=np.zeros(RIGIONS, dtype=int)
#记录每个区域的周长
region_id_inv=[]

#记录了这个大的区域块有多少条线构成
n_polyinreg=[]
#记录了每条线属于哪个区域块的具体编号
polyinreg=[]
#记录区域最大的id号
max_id=0

#记录是否使用边界阈值
use_perimeter_threshold = True

epsilon = 1e-10

#proj存储开始于(i+0.5,j+0.5)运动之后点的位置
cartcorn=[]

integration=0
#设定一个flag便于其他程序计数
gridv_count=0
rho_init_count=0
rho_ft_count=0
proj_count=0
gaussian_blur_rho_init_count=0
vintp_count=0
eul_count=0
mid_count=0

gridvx = torch.zeros((L, L), dtype=torch.float64)  # rho_ft=np.zeros((lx, ly), dtype=np.double)
gridvy = torch.zeros((L, L), dtype=torch.float64)

#1表示为省级，0表示城市级别
mode=0

