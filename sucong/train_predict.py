import norm_model
import json
import numpy as np

with open('./sucong/data/raw_data.json', 'r') as fp1:
    dataDict = json.load(fp1)

def calDist(pos1, pos2):
    '''计算距离的辅助函数，根据给出的坐标pos1和pos2，返回两点之间的距离
    输入： pos1, pos2 -- (x,y)元组
    输出： 欧几里得距离'''
    return np.sqrt((pos1[0] - pos2[0])*(pos1[0] - pos2[0]) + (pos1[1] - pos2[1])*(pos1[1] - pos2[1]))

lencenter = len(dataDict['center'])
lennode = len(dataDict['NodeCoor'])

centerDict = {k:v for k,v in zip(range(1,lencenter+1),dataDict['center'])}
nodeDict = {k:v for k,v in zip(range(lencenter+1,lennode+lencenter+1),dataDict['NodeCoor'])}
fullDict = {}
fullDict.update(centerDict)
fullDict.update(nodeDict)
print('初始化节点映射字典完成')

for i in range(len(dataDict['center'])):
    globals()['cluster'+str(i+1)] = [dataDict['center'][i]]
    globals()['demand'+str(i+1)] = [0]

for j in range(len(dataDict['NodeCoor'])):
    mindis = float('inf')
    index = 0
    for k in range(len(dataDict['center'])):
        if mindis > calDist(dataDict['NodeCoor'][j],dataDict['center'][k]):
            mindis = calDist(dataDict['NodeCoor'][j],dataDict['center'][k])
            index = k+1
    globals()['cluster'+str(index)].append(dataDict['NodeCoor'][j])
    globals()['demand'+str(index)].append(dataDict['Demand'][j])
    
finalroute = []
for i in range(1,lencenter+1):
    NodeCoor = eval('cluster'+str(i))
    Demand = eval('demand'+str(i))
    MaxLoad = dataDict['MaxLoad']
    print('开启模型调用，当前调用中心序列为{0}'.format(i))
    globals()['deap'+str(i)] = norm_model.DeapVrp(NodeCoor,Demand,MaxLoad,fullDict)
    globals()['route'+str(i)] = globals()['deap'+str(i)].predict()
    finalroute = finalroute+globals()['route'+str(i)]
    print('结束模型调用，当前调用中心序列为{0}'.format(i))
    print('--------------------------')
print(finalroute)