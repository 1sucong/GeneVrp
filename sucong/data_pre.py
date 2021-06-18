# from sucong.train_predict import Demand
import pandas as pd
import json
import numpy as np
import os
from matplotlib import pyplot as plt
import random

print(os.path.abspath(__file__))
loc_data = pd.read_csv('./sucong/CNprovince.csv')
loc_data['city'].fillna(loc_data['province'],inplace = True)
loc_list = [[n1,n2] for n1,n2 in zip(loc_data['longitude'],loc_data['latitude'])]
city_dict = {k:v for k,v in zip(loc_data['city'],loc_list)}
with open('./sucong/data/city_dict.json', 'w') as fp:
    json.dump(city_dict, fp)

# 生成假数据进行测试
# loc_test = loc_data.sample(frac=0.50)[['city','longitude']]
# loc_test['city_id'] = loc_test['city']
# loc_test['sales'] = np.random.randint(40,200, loc_test.shape[0])
# 读取补货单转json
# demand_data = loc_test

# 读取补货数据，合并到市粒度
demand_data = pd.read_csv('./sucong/data/res.csv')
demand_data = demand_data[demand_data['item_sku_id']==361132]
demand_data = demand_data.sample(200)
demand_data = demand_data[['dim_city_name','order_value']]
demand_data = demand_data.groupby('dim_city_name')['order_value'].sum()
demand_data.columns = ['dim_city_name','order_value']
demand_data = demand_data.reset_index()

init_json = {}
init_json['NodeCoor'] = []
init_json['Demand'] = []
print(demand_data.index)

# 生成json文件，将坐标，需求分别存入
for i in range(len(demand_data)):
    if demand_data.loc[i].at['dim_city_name'] in city_dict and demand_data.loc[i].at['order_value']>0:
        init_json['NodeCoor'].append(city_dict[demand_data.loc[i].at['dim_city_name']])
        init_json['Demand'].append(random.randint(400, 1000))
print(len(init_json['NodeCoor']))

# 画图
data = np.array(init_json['NodeCoor'])
x, y = data.T
plt.scatter(x,y,color = 'red')

init_json['center'] = [[108, 33],[119, 40],[111, 24]]

data1 = np.array(init_json['center'])
x1, y1 = data1.T
plt.scatter(x1,y1,color = 'blue')
plt.show()
init_json['MaxLoad'] = 4000
init_json['ServiceTime'] = 1
with open('./sucong/data/raw_data_test.json', 'w') as fp:
    json.dump(init_json, fp)