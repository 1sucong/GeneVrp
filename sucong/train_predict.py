from json import encoder
from typing import final
import norm_model
import json
import numpy as np
import random
import string

with open('./sucong/data/raw_data_test.json', 'r') as fp1:
    dataDict = json.load(fp1)
with open('./sucong/data/city_dict.json', 'r') as citydict:
    cityDict = json.load(citydict)
key_list = list(cityDict.keys())
val_list = list(cityDict.values())


def calDist(pos1, pos2):
    '''计算距离的辅助函数，根据给出的坐标pos1和pos2，返回两点之间的距离
    输入： pos1, pos2 -- (x,y)元组
    输出： 欧几里得距离'''
    return np.sqrt((pos1[0] - pos2[0])*(pos1[0] - pos2[0]) + (pos1[1] - pos2[1])*(pos1[1] - pos2[1]))

lencenter = len(dataDict['center'])
lennode = len(dataDict['NodeCoor'])

fullDemand = dataDict["Demand"]

centerDict = {k:v for k,v in zip(range(1,lencenter+1),dataDict['center'])}
nodeDict = {k:v for k,v in zip(range(lencenter+1,lennode+lencenter+1),dataDict['NodeCoor'])}
fullDict = {}
fullDict.update(centerDict)
fullDict.update(nodeDict)
print('初始化节点映射字典完成')

for i in range(len(dataDict['center'])):
    globals()['cluster'+str(i+1)] = []
    print(dataDict['center'])
    globals()['cluster'+str(i+1)].append(dataDict['center'][i])
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

list1 = [x for x in finalroute if x[0]==1]
list2 = [x for x in finalroute if x[0]==2]
list3 = [x for x in finalroute if x[0]==3]
for i in range(len(list1)):
    list1[i] = list1[i][1:-1]
for i in range(len(list2)):
    list2[i] = list2[i][1:-1]
for i in range(len(list3)):
    list3[i] = list3[i][1:-1]


dictres1 = []
for j in list1:
    listtmp = []
    for l in j:
        listtmp.append(fullDemand[l-4])
    for i in range(len(j)):
        j[i] = key_list[val_list.index(fullDict[j[i]])]
    plandict = {k:v for k,v in zip(j,listtmp)}
    dicttmp = {}
    dicttmp['plan'] = plandict
    dicttmp['carid'] = ''.join(random.choice(string.ascii_letters) for x in range(3))
    dictres1.append(dicttmp)
print(dictres1)

dictres2 = []
for j in list2:
    listtmp = []
    for l in j:
        listtmp.append(fullDemand[l-4])
    for i in range(len(j)):
        j[i] = key_list[val_list.index(fullDict[j[i]])]
    plandict = {k:v for k,v in zip(j,listtmp)}
    dicttmp = {}
    dicttmp['plan'] = plandict
    dicttmp['carid'] = ''.join(random.choice(string.ascii_letters) for x in range(3))
    dictres2.append(dicttmp)
print(dictres2)


dictres3 = []
for j in list3:
    listtmp = []
    for l in range(len(j)):
        listtmp.append(fullDemand[l-4])
    for i in range(len(j)):
        j[i] = key_list[val_list.index(fullDict[j[i]])]
    plandict = {k:v for k,v in zip(j,listtmp)}

    dicttmp = {}
    dicttmp['plan'] = plandict
    dicttmp['carid'] = ''.join(random.choice(string.ascii_letters) for x in range(3))
    dictres3.append(dicttmp)

print(dictres3)

finaljson = {}
finaljson['中心1号仓库'] = dictres1
finaljson['中心2号仓库'] = dictres2
finaljson['中心3号仓库'] = dictres3
print(finaljson)


with open('./sucong/result/361132_1.json', 'w',encoding='utf8') as fp2:
    dataDict = json.dump(finaljson,fp2,ensure_ascii=False)
