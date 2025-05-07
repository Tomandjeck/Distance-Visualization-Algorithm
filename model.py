import torch.nn as nn
import pandas as pd
import torch.optim
import main_ as m
import numpy as np
import globals as g
import matplotlib.pyplot as plt

def L1_regularization(param,lambda_reg):
    L1_norm=0
    if param.requires_grad:
        L1_norm=torch.sum(torch.abs(param))
    return lambda_reg*L1_norm

def L2_regularization(param,lambda_reg):
    L2_norm = 0
    if param.requires_grad:
        L2_norm = torch.sum(param**2)
    return lambda_reg * L2_norm

def L1_L2_regularization(param,lambda_reg1,lambda_reg2):
    L1_norm = 0
    L2_norm = 0
    if param.requires_grad:
        L1_norm = torch.sum(torch.abs(param))
        L2_norm = torch.sum(param**2)
    return lambda_reg1 * L1_norm + lambda_reg2 * L2_norm

def calculate_error(geo_distance,target_distance):
    #不取对角线的值，设置为0
    geo_distance_upper = torch.triu(geo_distance,diagonal=1)
    target_distance_upper = torch.triu(target_distance,diagonal=1)

    # error=0.15 * torch.sum(g.population_data_tensor ** 2) + torch.sum((geo_distance_upper - target_distance_upper) ** 2)
    loss =  torch.sum((geo_distance_upper - target_distance_upper) ** 2)
    regulation = 0 #L1_L2_regularization(g.population_data_tensor,0.01,0.01)

    error = loss + regulation
    return error

def record_error(iter,loss):
    df1=pd.DataFrame({"iter":[iter]})
    df2=pd.DataFrame({"val":[loss]})
    dataFrame = df1.join(df2)
    dataFrame.to_csv("./loss_data.csv",mode='a')

def polt_loss(data):
    import re
    input_data=f"[{data}]"
    # 输入字符串
#     input_data = """
#   [tensor(46641.82812500, grad_fn=<AddBackward0>), tensor(46579.11328125, grad_fn=<AddBackward0>), tensor(46810.24609375, grad_fn=<AddBackward0>), tensor(46791.84375000, grad_fn=<AddBackward0>), tensor(46732.96484375, grad_fn=<AddBackward0>), tensor(46674.15234375, grad_fn=<AddBackward0>), tensor(46592.95312500, grad_fn=<AddBackward0>), tensor(46534.14843750, grad_fn=<AddBackward0>), tensor(46484.01171875, grad_fn=<AddBackward0>), tensor(46442.62109375, grad_fn=<AddBackward0>), tensor(46382.70703125, grad_fn=<AddBackward0>), tensor(46334.45703125, grad_fn=<AddBackward0>), tensor(46275.94921875, grad_fn=<AddBackward0>)]
# """

    # 提取浮点数
    float_numbers = re.findall(r"\d+\.\d+", input_data)
    loss = [float(num) for num in float_numbers]

    if g.mode:
        data = pd.read_excel("./data/TableData/province_target_loss.xlsx")
    else:
        data=pd.read_excel("./data/TableData/citys_target_loss.xlsx")
    df = pd.DataFrame(data)


    dictory = dict(val=loss)
    df1 = pd.DataFrame(dictory)

    df=pd.concat([df,df1],ignore_index=True)
    if g.mode:
        df.to_excel("./data/TableData/province_target_loss.xlsx",index=False)
    else:
        df.to_excel("./data/TableData/citys_target_loss.xlsx", index=False)

    population = df["val"]
    iteration = [i for i in range(len(df))]
    plt.plot(iteration,population)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid()
    plt.show()


def save_population(population):
    # population=[ 3187.35000000,  1972.35000000, 11147.35000000,  5179.83000000,
    #      3600.65000000,  6328.75000000,  3574.72000000,  4624.35000000,
    #      3688.78000000, 12704.75000000,  9739.35000000,  9112.35000000,
    #      6186.35000000,  6716.75000000, 15169.34000000, 14882.35000000,
    #     10947.35000000,  9921.35000000, 18683.35000000,  7536.35000001,
    #      1521.11000000,  4811.78000000, 12372.65000000,  5851.35000001,
    #      6990.65000000,   725.35000000,  5953.35000000,  3690.67000000,
    #       894.65000000,  1025.65000000,  3789.65000000]
    if g.mode:
        file = pd.read_csv("./data/TableData/province_data.csv")
    else:
        file = pd.read_csv("./data/TableData/numbers_column.csv")
    df=pd.DataFrame(file)
    df["population"]=population
    if g.mode:
        df.to_csv("./data/TableData/province_data.csv",index=False)
    else:
        df.to_csv("./data/TableData/numbers_column.csv", index=False)

#主函数
def optimize_population_data(target_distance,args,learn_rate=0.7,max_iteration=1):
    if g.mode:
        #output中存放了region id和region data,region name
        df1 = pd.read_csv("./data/TableData/output.csv", encoding='utf-8')
        df1 = df1[:31]
        population_data_id = df1['Region Id']
        #df2 = pd.read_csv("./province_numbers_column.csv")
        df2 = pd.read_csv("./data/TableData/province_data.csv") #./data/TableData/province_target_data.csv   ./data/TableData/province_data.csv
        #截取部分数据出来实验
        population_data = df2["population"]

        #population_data = df1["Region Data"]
        # min_value = population_data.min()
        # max_value = population_data.max()
        # population_data = (population_data - min_value) / (max_value - min_value)
    else:
        df1 = pd.read_csv("./data/TableData/output_citys.csv",encoding='gbk')
        #df1 = df1[:362]
        population_data_id = df1['Region Id']

        df2 = pd.read_csv("./data/TableData/numbers_column.csv")
        # df2 = pd.read_csv("./test_data.csv")
        # 截取部分数据出来实验
        #df2=df2[:362]
        population_data = df2["population"]
    # population_data = df2["Region Data"]
    # min_value = population_data.min()
    # max_value = population_data.max()
    # population_data = (population_data - min_value) / (max_value - min_value)
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将Pandas Series转换为PyTorch张量
    #g.population_data_tensor是叶子节点
    #g.population_data_tensor = torch.tensor(population_data.values, dtype=torch.float32, device=device)
    # g.population_data_id_tensor = torch.tensor(population_data_id.values, dtype=torch.float32)
    # optimizer = torch.optim.Adam([g.population_data_tensor],lr=learn_rate) #Adam
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # geo_distance = m.main()
    # geo_distance_normalized = (geo_distance - torch.min(geo_distance)) / (torch.max(geo_distance) - torch.min(geo_distance))
    # geo_distance_normalized = geo_distance_normalized.detach()
    # geo_distance_normalized.requires_grad_()
    #
    # print(f"geo_distance_normalized is {geo_distance_normalized}")
    # optimizer = torch.optim.Adam([geo_distance_normalized], lr=learn_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # 检查 GPU 是否可用
    device = torch.device("cpu")

    new_data=population_data.values
    new_id=population_data_id.values
    loss=[]
    for iteration in range(max_iteration):
        new_data=np.where(new_data>0,new_data,0.001)

        # 每次迭代重新设置 requires_grad
        population_data_tensor = torch.tensor(new_data,requires_grad=True)
        population_data_id_tensor = torch.tensor(new_id).to(device)
        # 每次迭代重新创建优化器
        optimizer = torch.optim.Adam([population_data_tensor], lr=learn_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        #torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        #获得了变形图之后各个城市在经纬度的两两之间的距离
        #print(f"population is {population_data_tensor}")
        geo_distance = m.main(population_data_tensor,population_data_id_tensor,iteration,device,args)


        # 归一化 geo_distance 和 target_distance
        #geo_distance_normalized = (geo_distance - torch.min(geo_distance)) / (torch.max(geo_distance) - torch.min(geo_distance))

        #获得变形图和目标城市之间距离的误差
        error = calculate_error(geo_distance,target_distance)
        #torch.sum( (geo_distance- target_distance)**2)本质是正则化项，需要乘以一个正则化权重调整对误差的影响

        #可视化目标距离与计算距离
        import ps_figure as pf
        pf.picture_targetAndDistance(geo_distance,target_distance,error,args)

        print(f"Iteration {iteration}: Error = {error}")

        loss.append(error)
        print(loss)

        if error.item() < 1e-6:  # 误差达到可接受范围
            break
            
        #计算梯度
        error.backward()

        # 更新参数
        optimizer.step()
        scheduler.step()  # 更新学习率

        new_data=population_data_tensor.detach().numpy()
        new_id=population_data_id_tensor.detach().numpy()

        polt_loss(error)
        save_population(new_data)

    return

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--arg1", type=str, required=True)
args = parser.parse_args()
args=args.arg1

#word2vec计算出来城市间的距离
if g.mode:
    # df=pd.read_excel("./citys_data.xlsx")
    # target_distances=df.to_numpy()
    # target_distances = target_distances[:31, :31]
    # target_distances = np.load("./converted_geo_distance.npy")
    target_distances = np.load("./data/NumpyData/population_target_distance.npy")
    target_distances = target_distances[:31, :31]

else:
    target_distances = np.load("./data/NumpyData/converted_geo_distance_369城市.npy") #../距离估算及线状变形/整合版/data/生成数据/NumpyData/converted_geo_distance.npy
    target_distances=target_distances[:,:] #:72,:72

    # df=pd.DataFrame(target_distances)
    # df.to_excel("./citys_data.xlsx",index=False)
#保存数据
# df=pd.DataFrame(target_distances)
# df.to_excel("./citys_data.xlsx",index=False)
target_distances_tensor = torch.tensor(target_distances, dtype=torch.float32,requires_grad=True)
#target_distances_tensor_normalized = (target_distances_tensor-torch.min(target_distances_tensor)) / (torch.max(target_distances_tensor)-torch.min(target_distances_tensor))

# 优化人口数据
optimize_population_data(target_distances_tensor,args)

