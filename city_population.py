import pandas as pd

df=pd.read_csv('./output_citys.csv', encoding='gbk')

df['population']=None

for index,row in df.iterrows():
    if row['Region Name'] in df['城市'].values:

        # 找到对应的 2010 年人口数据
        population = df.loc[df['城市'] == row['Region Name'], '2010'].values[0]
        # 将人口数据赋值给 Population 列
        df.at[index, 'Population'] = population
# 保存结果到新的 CSV 文件
df.to_csv('./output_citys.csv', index=False)