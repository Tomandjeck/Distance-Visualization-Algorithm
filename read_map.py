import numpy as np
import sys
import globals as g
import geopandas as gpd
import multiprocessing as mp
import torch
import pandas as pd
import json
from shapely.geometry import Polygon,MultiPolygon
import matplotlib.pyplot as plt
from shapely.geometry import Point
AREA_THRESHOLD=1e-12

def output_province(ids):
    provinces = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区',
                 '辽宁省', '吉林省', '黑龙江省',
                 '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省',
                 '河南省', '湖北省', '湖南省', '广东省', '广西自治区', '海南省',
                 '重庆市', '四川省', '贵州省', '云南省', '西藏自治区',
                 '陕西省', '甘肃省', '青海省', '宁夏自治区', '新疆自治区']

    population = [2188, 1373, 7448, 3480.48, 2400, 4229.4, 2375.37, 3125, 2489.43, 8505.4, 6540, 6113, 4187, 4517.4,
                  10169.99, 9883, 7448, 6622, 12684, 5037, 1020.46, 3212.43, 8372, 3852, 4690, 366, 3954, 2490.02, 594,
                  725, 2589]

    cities_df = pd.DataFrame({
        'Region Id': ids,  # 与 GeoJSON 中的 ID 对应
        'Region Data': population,  # 对应的人口数据
        'Region Name': provinces  # 城市名称
    })

    cities_df.to_csv('output.csv', index=False)

def output_citys(ids):
    import re

    citys = """'北京市':110000,
                '天津市':120000,
                '石家庄市':130100,'唐山市':130200,'秦皇岛市':130300,'邯郸市':130400,'邢台市':130500,'保定市':130600,'张家口市':130700,'承德市':130800,'沧州市':130900,'廊坊市':131000,'衡水市':131100,
                '太原市':140100,'大同市':140200,'阳泉市':140300,'长治市':140400,'晋城市':140500,'朔州市':140600,'晋中市':140700,'运城市':140800,'忻州市':140900,'临汾市':141000,'吕梁市':141100,
                '呼和浩特市':150100,'包头市':150200,'乌海市':150300,'赤峰市':150400,'通辽市':150500,'鄂尔多斯市':150600,'呼伦贝尔市':150700,'巴彦淖尔市':150800,'乌兰察布市':150900,'兴安盟':152200,'锡林郭勒盟':152500,'阿拉善盟':152900,
                '沈阳市':210100,'大连市':210200,'鞍山市':210300,'抚顺市':210400,'本溪市':210500,'丹东市':210600,'锦州市':210700,'营口市':210800,'阜新市':210900,'辽阳市':211000,'盘锦市':211100,'铁岭市':211200,'朝阳市':211300,'葫芦岛市':211400,
                '长春市':220100,'吉林市':220200,'四平市':220300,'辽源市':220400,'通化市':220500,'白山市':220600,'松原市':220700,'白城市':220800,'延边朝鲜族自治州':222400,
                '哈尔滨市':230100,'齐齐哈尔市':230200,'鸡西市':230300,'鹤岗市':230400,'双鸭山市':230500,'大庆市':230600,'伊春市':230700,'佳木斯市':230800,'七台河市':230900,'牡丹江市':231000,'黑河市':231100,'绥化市':231200,'大兴安岭地区':232700,
                '上海市':310000,
                '南京市':320100,'无锡市':320200,'徐州市':320300,'常州市':320400,'苏州市':320500,'南通市':320600,'连云港市':320700,'淮安市':320800,'盐城市':320900,'扬州市':321000,'镇江市':321100,'泰州市':321200,'宿迁市':321300,
                '杭州市':330100,'宁波市':330200,'温州市':330300,'嘉兴市':330400,'湖州市':330500,'绍兴市':330600,'金华市':330700,'衢州市':330800,'舟山市':330900,'台州市':331000,'丽水市':331100,
                '合肥市':340100,'芜湖市':340200,'蚌埠市':340300,'淮南市':340400,'马鞍山市':340500,'淮北市':340600,'铜陵市':340700,'安庆市':340800,'黄山市':341000,'滁州市':341100,'阜阳市':341200,'宿州市':341300,'六安市':341500,'亳州市':341600,'池州市':341700,'宣城市':341800,
                '福州市':350100,'厦门市':350200,'莆田市':350300,'三明市':350400,'泉州市':350500,'漳州市':350600,'南平市':350700,'龙岩市':350800,'宁德市':350900,
                '南昌市':360100,'景德镇市':360200,'萍乡市':360300,'九江市':360400,'新余市':360500,'鹰潭市':360600,'赣州市':360700,'吉安市':360800,'宜春市':360900,'抚州市':361000,'上饶市':361100,
                '济南市':370100,'青岛市':370200,'淄博市':370300,'枣庄市':370400,'东营市':370500,'烟台市':370600,'潍坊市':370700,'济宁市':370800,'泰安市':370900,'威海市':371000,'日照市':371100,'临沂市':371300,'德州市':371400,'聊城市':371500,'滨州市':371600,'菏泽市':371700,
                '郑州市':410100,'开封市':410200,'洛阳市':410300,'平顶山市':410400,'安阳市':410500,'鹤壁市':410600,'新乡市':410700,'焦作市':410800,'濮阳市':410900,'许昌市':411000,'漯河市':411100,'三门峡市':411200,'南阳市':411300,'商丘市':411400,'信阳市':411500,'周口市':411600,'驻马店市':411700,'济源市':419001,
                '武汉市':420100,'黄石市':420200,'十堰市':420300,'宜昌市':420500,'襄阳市':420600,'鄂州市':420700,'荆门市':420800,'孝感市':420900,'荆州市':421000,'黄冈市':421100,'咸宁市':421200,'随州市':421300,'恩施土家族苗族自治州':422800,'仙桃市':429004,'潜江市':429005,'天门市':429006,'神农架林区':429021,
                '长沙市':430100,'株洲市':430200,'湘潭市':430300,'衡阳市':430400,'邵阳市':430500,'岳阳市':430600,'常德市':430700,'张家界市':430800,'益阳市':430900,'郴州市':431000,'永州市':431100,'怀化市':431200,'娄底市':431300,'湘西土家族苗族自治州':433100,
                '广州市':440100,'韶关市':440200,'深圳市':440300,'珠海市':440400,'汕头市':440500,'佛山市':440600,'江门市':440700,'湛江市':440800,'茂名市':440900,'肇庆市':441200,'惠州市':441300,'梅州市':441400,'汕尾市':441500,'河源市':441600,'阳江市':441700,'清远市':441800,'东莞市':441900,'中山市':442000,'潮州市':445100,'揭阳市':445200,'云浮市':445300,
                '南宁市':450100,'柳州市':450200,'桂林市':450300,'梧州市':450400,'北海市':450500,'防城港市':450600,'钦州市':450700,'贵港市':450800,'玉林市':450900,'百色市':451000,'贺州市':451100,'河池市':451200,'来宾市':451300,'崇左市':451400,
                '海口市':460100,'三亚市':460200,'三沙市':460300,'儋州市':460400,'五指山市':469001,'琼海市':469002,'文昌市':469005,'万宁市':469006,'东方市':469007,'定安县':469021,'屯昌县':469022,'澄迈县':469023,'临高县':469024,'白沙黎族自治县':469025,'昌江黎族自治县':469026,'乐东黎族自治县':469027,'陵水黎族自治县':469028,'保亭黎族苗族自治县':469029,'琼中黎族苗族自治县':469030,
                '重庆市':500000,
                '成都市':510100,'自贡市':510300,'攀枝花市':510400,'泸州市':510500,'德阳市':510600,'绵阳市':510700,'广元市':510800,'遂宁市':510900,'内江市':511000,'乐山市':511100,'南充市':511300,'眉山市':511400,'宜宾市':511500,'广安市':511600,'达州市':511700,'雅安市':511800,'巴中市':511900,'资阳市':512000,'阿坝藏族羌族自治州':513200,'甘孜藏族自治州':513300,'凉山彝族自治州':513400,
                '贵阳市':520100,'六盘水市':520200,'遵义市':520300,'安顺市':520400,'毕节市':520500,'铜仁市':520600,'黔西南布依族苗族自治州':522300,'黔东南苗族侗族自治州':522600,'黔南布依族苗族自治州':522700,
                '昆明市':530100,'曲靖市':530300,'玉溪市':530400,'保山市':530500,'昭通市':530600,'丽江市':530700,'普洱市':530800,'临沧市':530900,'楚雄彝族自治州':532300,'红河哈尼族彝族自治州':532500,'文山壮族苗族自治州':532600,'西双版纳傣族自治州':532800,'大理白族自治州':532900,'德宏傣族景颇族自治州':533100,'怒江傈僳族自治州':533300,'迪庆藏族自治州':533400,
                '拉萨市':540100,'日喀则市':540200,'昌都市':540300,'林芝市':540400,'山南市':540500,'那曲市':540600,'阿里地区':542500,
                '西安市':610100,'铜川市':610200,'宝鸡市':610300,'咸阳市':610400,'渭南市':610500,'延安市':610600,'汉中市':610700,'榆林市':610800,'安康市':610900,'商洛市':611000,
                '兰州市':620100,'嘉峪关市':620200,'金昌市':620300,'白银市':620400,'天水市':620500,'武威市':620600,'张掖市':620700,'平凉市':620800,'酒泉市':620900,'庆阳市':621000,'定西市':621100,'陇南市':621200,'临夏回族自治州':622900,'甘南藏族自治州':623000,
                '西宁市':630100,'海东市':630200,'海北藏族自治州':632200,'黄南藏族自治州':632300,'海南藏族自治州':632500,'果洛藏族自治州':632600,'玉树藏族自治州':632700,'海西蒙古族藏族自治州':632800,
                '银川市':640100,'石嘴山市':640200,'吴忠市':640300,'固原市':640400,'中卫市':640500,
                '乌鲁木齐市':650100,'克拉玛依市':650200,'吐鲁番市':650400,'哈密市':650500,'昌吉回族自治州':652300,'博尔塔拉蒙古自治州':652700,'巴音郭楞蒙古自治州':652800,'阿克苏地区':652900,'克孜勒苏柯尔克孜自治州':653000,'喀什地区':653100,'和田地区':653200,'伊犁哈萨克自治州':654000,'塔城地区':654200,'阿勒泰地区':654300,
                '石河子市':659001,'阿拉尔市':659002,'图木舒克市':659003,'五家渠市':659004,'北屯市':659005,'铁门关市':659006,'双河市':659007,'可克达拉市':659008,'昆玉市':659009,
                '台湾省':710000,'香港特别行政区':810000,'澳门特别行政区':820000"""
    # 使用正则表达式提取键名
    keys = re.findall(r"'([^']+)'", citys)
    res=[]
    for i in keys:
        res.append(i)

    population = [2188, 1373, 7448, 3480.48, 2400, 4229.4, 2375.37, 3125, 2489.43, 8505.4, 6540, 6113, 4187, 4517.4,
                  10169.99, 9883, 7448, 6622, 12684, 5037, 1020.46, 3212.43, 8372, 3852, 4690, 366, 3954, 2490.02, 594,
                  725, 2589]

    cities_df = pd.DataFrame({
        #'Region Id': ids,  # 与 GeoJSON 中的 ID 对应
       # 'Region Data': population,  # 对应的人口数据
        'Region Name': res  # 城市名称
    })

    cities_df.to_csv('output_citys.csv', index=False)

def output_city_beta(ids):
    df=pd.read_csv('./output_citys.csv',encoding='gbk')
    df['Region Id']=ids
    df.to_csv('output_citys.csv', index=False,encoding='gbk')

def deal_with_province():
    shp_file = './data/MapData/china_provinces_3857.shp'
    gdf = gpd.read_file(shp_file)
    gdf = gdf[:31]  # 少了台湾，香港，澳门

    df = pd.read_csv("./data/TableData/china_province_coords.csv",encoding='gbk')
    df = df[:31]
    geometry = [Point(xy) for xy in zip(df['lng'], df['lat'])]

    gdf1 = gpd.GeoDataFrame(df, geometry=geometry)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf1.crs = "EPSG:4326"
    gdf1 = gdf1.to_crs("EPSG:3857")
    # 从 GeoDataFrame 提取 X 和 Y 坐标
    x_coords = gdf1.geometry.x.to_numpy()  # 转换为 NumPy 数组
    y_coords = gdf1.geometry.y.to_numpy()  # 转换为 NumPy 数组
    g_x = torch.tensor(x_coords, dtype=torch.float64)
    g_y = torch.tensor(y_coords, dtype=torch.float64)  # , requires_grad=True

    gdf1['x'] = gdf1.geometry.x
    gdf1['y'] = gdf1.geometry.y


    gdf1.to_csv("./data/TableData/province_points.csv")
    # gdf.plot(ax=ax)
    # ax.scatter(g.x, g.y, c='white', s=10, linewidth=1, zorder=3)
    # ax.scatter(g.x, g.y, c='#B4EBAF', s=2, zorder=4)
    #
    # plt.show()
    return gdf,g_x,g_y

def deal_with_citys():
    shp_file = './data/MapData/china_citys_3857.shp'
    gdf = gpd.read_file(shp_file)
    # 截取部分数据出来实验
    #gdf=gdf[:362]


    df = pd.read_csv("./data/TableData/citys_coordinate_wgs84_369城市.csv",encoding='gbk')
    #截取部分数据出来实验
    #df=df[:362]

    geometry = [Point(xy) for xy in zip(df['lng'], df['lat'])]
    gdf1 = gpd.GeoDataFrame(df, geometry=geometry)
    #fig, ax = plt.subplots(figsize=(10, 10))
    gdf1.crs = "EPSG:4326"
    gdf1 = gdf1.to_crs("EPSG:3857")
    # 从 GeoDataFrame 提取 X 和 Y 坐标
    x_coords = gdf1.geometry.x.to_numpy()  # 转换为 NumPy 数组
    y_coords = gdf1.geometry.y.to_numpy()  # 转换为 NumPy 数组
    g_x = torch.tensor(x_coords, dtype=torch.float64)
    g_y = torch.tensor(y_coords, dtype=torch.float64) #, requires_grad=True

    gdf1['x'] = gdf1.geometry.x
    gdf1['y'] = gdf1.geometry.y
    gdf1.to_csv("./data/TableData/citys_points.csv")
    # gdf.plot(ax=ax)
    # ax.scatter(g.x, g.y, c='white', s=10, linewidth=1, zorder=3)
    # #ax.scatter(g.x, g.y, c='#B4EBAF', s=2, zorder=4)
    # ax.scatter(g.x, g.y, c='red', s=2, zorder=4)
    # plt.show()
    # plt.close(fig)

    return gdf,g_x,g_y


def read_geojson():
    if g.mode:
        gdf,g_x,g_y = deal_with_province()
    else:
        gdf,g_x,g_y = deal_with_citys()

    global polygon_json
    polygon = gdf.geometry.to_json()
    polygon_json = json.loads(polygon)

    #个矩形覆盖了地理空间对象的所有坐标点。
    g.map_minx, g.map_miny, g.map_maxx, g.map_maxy=gdf.total_bounds
    #This for loop counts the number of polygons in the geojson file

    #变量初始化
    g.n_poly = 0
    g.n_polycorn = []
    g.polycorn = []
    g.polygon_id = []

    #使用geopandas处理数据
    for feature_iterator in gdf.itertuples():
        geometry=feature_iterator.geometry
        if geometry.geom_type=='Polygon':
            g.n_poly+=1
        elif geometry.geom_type=='MultiPolygon':
            g.n_poly+= len(geometry.geoms)
        else:
            print('Error: Region contains geometry other than polygons and multipolygons.1')
            sys.exit(1)

    # This for loop counts the number of polygon corners in the geojson file
    # n_polycorn = []
    # polycorn = [] #存储数据格式为：[[[1,2],[1,3],...],[[1,2],[1,3],...],...]
    # polygon_id = []

    polyctr = 0

    for feature_iterator in gdf.itertuples():
        geometry = feature_iterator.geometry
        feature_id = int(feature_iterator.Index)  # 使用索引作为ID，如果存在ID字段，替换为feature_iterator.ID

        if geometry.geom_type == 'Polygon':
            linear_ring = np.array(geometry.exterior.coords, dtype=np.float64)
            g.n_polycorn.append(len(linear_ring))  # 计算顶点数
            g.polycorn.append(linear_ring)  # 存储顶点

            if not np.array_equal(linear_ring[0], linear_ring[-1]):
                print(f'WARNING: {polyctr + 1}-th polygon does not close upon itself')
                print(f'Identifier {feature_id}, first point({linear_ring[0][0]}, {linear_ring[0][1]})')
            polyctr += 1
            g.polygon_id.append(feature_id)

        elif geometry.geom_type == 'MultiPolygon':
            for polygon in geometry.geoms:
                linear_ring = np.array(polygon.exterior.coords, dtype=np.float64)
                g.n_polycorn.append(len(linear_ring))  # 计算顶点数
                g.polycorn.append(linear_ring)  # 存储顶点
                polyctr += 1
                g.polygon_id.append(feature_id)


                if not np.array_equal(linear_ring[0], linear_ring[-1]):
                    print(f'WARNING: {polyctr + 1}-th polygon does not close upon itself')
                    print(f'Identifier {feature_id}, first point({linear_ring[0][0]}, {linear_ring[0][1]})')

        else:
            print('Error: Region contains geometry other than polygons and multipolygons.2')
            sys.exit(1)
    return g_x,g_y

def polygon_area(polycorn):
    areas = torch.zeros(g.n_poly,dtype=torch.float64)
    count=0
    for linear_ring in polycorn:
        polygon = Polygon(linear_ring)
        areas[count]=polygon.area
        count+=1
    return areas


def polygon_perimeter():
    perimeters = torch.zeros(g.n_poly, dtype=torch.float64)
    count = 0
    for coords in g.polycorn:
        polygon = Polygon(coords)
        perimeters[count]=polygon.length
        count += 1
    return perimeters

#remove tiny polygons
def remove_tiny_polygons():
    #Find out whether there are any tiny polygons.
    # poly_has_tiny_area=[abs(polygon_area(int(n_polycorn_),polycorn_))<AREA_THRESHOLD*(g.map_maxx - g.map_minx)*(g.map_maxy - g.map_miny)
    #                     for n_polycorn_,polycorn_ in zip(g.n_polycorn,g.polycorn)]
    poly_has_tiny_area = [
        Polygon(polycorn_).area < AREA_THRESHOLD * (g.map_maxx - g.map_minx) * (g.map_maxy - g.map_miny)
        for polycorn_ in g.polycorn
    ]
    n_non_tiny_poly = 0
    for i in range(g.n_poly):
        if  not poly_has_tiny_area[i]:
            n_non_tiny_poly+=1


    if n_non_tiny_poly<g.n_poly:
        print("Removing tiny polygons.")

        # If there are tiny polygons, we replace the original polygons by the  subset of non-tiny polygons.
        n_non_tiny_polycorn=[]
        non_tiny_polygon_id=[]
        n_non_tiny_poly=0
        for poly_indx in range(g.n_poly):
            if not poly_has_tiny_area[poly_indx]:
                n_non_tiny_polycorn.append(g.n_polycorn[poly_indx])
                non_tiny_polygon_id.append(g.polygon_id[poly_indx])
                n_non_tiny_poly+=1

        n_non_tiny_poly=0
        non_tiny_polycorn=[]
        for poly_indx in range(g.n_poly):
            if not poly_has_tiny_area[poly_indx]:
                non_tiny_polycorn.append(g.polycorn[poly_indx])
                n_non_tiny_poly+=1

        g.n_poly=n_non_tiny_poly
        g.polygon_id=np.array(non_tiny_polygon_id)
        g.n_polycorn=np.array(n_non_tiny_polycorn,dtype=np.float64)
        g.polycorn=np.array(non_tiny_polycorn,dtype=object)



#Function to make regions from polygons.
def make_region():

    poly_is_hole = np.zeros(g.n_poly, dtype=bool)
    areas = polygon_area(g.polycorn)
    for i in range(g.n_poly):
        if areas[i] < 0:
            poly_is_hole[i] = True

        else:
            poly_is_hole[i] = False  # 共有0个hole

    # Count the number of regions

    # n_reg = 0 #有31个区域
    g.max_id = min_id = g.polygon_id[0]
    for i in range(g.n_poly):
        if g.polygon_id[i]==-9999: #-99999 is a special ID. Such polygons will be assigned to the same region as the polygon immediately before it.
            continue
        if g.polygon_id[i]>g.max_id or g.polygon_id[i]<min_id:
            g.n_reg+=1
        else:
            repeat=False
            for j in range(i):
                if g.polygon_id[i]==g.polygon_id[j]:
                    repeat=True
                    break
            if not repeat:
                g.n_reg+=1
        g.max_id=max(g.max_id,g.polygon_id[i])
        min_id=min(min_id,g.polygon_id[i])
    if min_id<0:
        print(f'ERROR: Negative region identifier {min_id}')
        sys.exit(1)

    #Match region IDs
    g.n_reg = 0

    g.max_id = min_id = g.polygon_id[0]
    for j in range(g.n_poly):
        if g.polygon_id[j]==-9999:
            continue
        if g.polygon_id[j]>g.max_id or g.polygon_id[j]<min_id:
            #g.region_id.append(g.polygon_id[j])
            g.region_id[g.n_reg]=g.polygon_id[j]
            g.n_reg+=1
        else:
            repeat=False
            for i in range(j):
                if g.polygon_id[j]==g.polygon_id[i]:
                    repeat=True
                    break
            if not repeat:
                #g.region_id.append(g.polygon_id[j])
                g.region_id[g.n_reg] = g.polygon_id[j]
                g.n_reg+=1
        g.max_id=max(g.max_id,g.polygon_id[j])
        min_id=min(min_id,g.polygon_id[j])

    #g.region_id=np.array(g.region_id)
    #region_id[i] takes as input C's internal identifier for the region and assumes the value of the ID in the .gen file.
    #region_id_inv[i] does the opposite. Its input is an ID from the .gen file. Its value is the internal identifier used by C.

    g.region_id_inv = np.full(g.max_id + 1, -1, dtype=int)
    for i in range(g.n_reg):
        g.region_id_inv[g.region_id[i]]=i

    #Which polygons contribute to which region?
    last_id=g.polygon_id[0]

    g.n_polyinreg = np.zeros(g.n_reg, dtype=int)

    for j in range(g.n_poly):
        if g.polygon_id[j]!= -9999:
            g.n_polyinreg[g.region_id_inv[g.polygon_id[j]]]+=1
            last_id=g.polygon_id[j]
        else:
            g.n_polyinreg[g.region_id_inv[last_id]]+=1

    g.polyinreg = np.empty(g.n_reg, dtype=object)
    for j in range(g.n_reg):
        g.polyinreg[j] = np.zeros(g.n_polyinreg[j], dtype=int)

    g.n_polyinreg.fill(0)

    for j in range(g.n_poly):
        if g.polygon_id[j]!=-9999:
            region_index=g.region_id_inv[g.polygon_id[j]]
            g.polyinreg[region_index][g.n_polyinreg[region_index]]=j
            g.n_polyinreg[region_index]+=1
            last_id=g.polygon_id[j]
        else:
            region_index = g.region_id_inv[last_id]
            g.polyinreg[region_index][g.n_polyinreg[region_index]] = j
            g.n_polyinreg[region_index] += 1


def read_map():
    # polygon_json 是 geojson格式
    #原项目在这里的功能是检查map_file_name的文件扩展名，来确定属于那种文件类型，并做出相应的处理文件。
    #读取geojson、json格式的文件
    #read_geojson(map_file_name)
    g_x,g_y=read_geojson()
    # this map has no tiny area that need to be removed
    remove_tiny_polygons()
    make_region()
    # print(g.polycorn)
    # g.polycorn = [torch.tensor(arr, dtype=torch.float32) for arr in g.polycorn]

    print(f"{g.n_poly} polygons, {g.n_reg} regions.")

    return g_x,g_y
