import geopandas as gpd
import json
import pandas as pd


def process_json1(ShpFile):
    province = gpd.read_file(ShpFile)
    gdf = province[:31]  # 少了台湾，香港，澳门

    ids=gdf['FID'].tolist()

    polygon = gdf.geometry.to_json()
    polygon_json = json.loads(polygon)
    provinces=['北京市','天津市','河北省','山西省','内蒙古自治区',
     '辽宁省','吉林省','黑龙江省',
    '上海市','江苏省','浙江省','安徽省','福建省','江西省','山东省',
    '河南省','湖北省','湖南省','广东省','广西自治区','海南省',
     '重庆市','四川省','贵州省','云南省','西藏自治区',
    '陕西省','甘肃省','青海省','宁夏自治区','新疆自治区']

    population = [2188, 1373, 7448, 3480.48, 2400, 4229.4, 2375.37, 3125, 2489.43, 8505.4, 6540, 6113, 4187, 4517.4,
                  10169.99, 9883, 7448, 6622, 12684, 5037, 1020.46, 3212.43, 8372, 3852, 4690, 366, 3954, 2490.02, 594,
                  725, 2589]

    cities_df = pd.DataFrame({
        'Region Id': ids,  # 与 GeoJSON 中的 ID 对应
        'Region Data': population,  # 对应的人口数据
        'Region Name': provinces  # 城市名称
    })

    cities_df.to_csv('output.csv',index=False)

    return  polygon_json

def process_json(ShpFile):
    province = gpd.read_file(ShpFile)
    gdf = province[:31]  # 少了台湾，香港，澳门

    ids=gdf['FID'].tolist()

    polygon = gdf.geometry.to_json()
    polygon_json = json.loads(polygon)
    provinces=['北京市','天津市','河北省','山西省','内蒙古自治区',
     '辽宁省','吉林省','黑龙江省',
    '上海市','江苏省','浙江省','安徽省','福建省','江西省','山东省',
    '河南省','湖北省','湖南省','广东省','广西自治区','海南省',
     '重庆市','四川省','贵州省','云南省','西藏自治区',
    '陕西省','甘肃省','青海省','宁夏自治区','新疆自治区']

    population = [2188, 1373, 7448, 3480.48, 2400, 4229.4, 2375.37, 3125, 2489.43, 8505.4, 6540, 6113, 4187, 4517.4,
                  10169.99, 9883, 7448, 6622, 12684, 5037, 1020.46, 3212.43, 8372, 3852, 4690, 366, 3954, 2490.02, 594,
                  725, 2589]

    cities_df = pd.DataFrame({
        'Region Id': ids,  # 与 GeoJSON 中的 ID 对应
        'Region Data': population,  # 对应的人口数据
        'Region Name': provinces  # 城市名称
    })

    cities_df.to_csv('output.csv',index=False)

    return  polygon_json
