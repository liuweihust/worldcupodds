# coding=utf-8
from mpl_toolkits import mplot3d
#matplotlib inline
#import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import sys
from datasets import dataset_factory
from datasets import SoccerRawDb
from datasets import utils

"""
Blow code will show the real result,offer result and pred result along 2018's match
"""
data = SoccerRawDb.load_allrawdata()
data2018 = data[ data['Year']==2018 ]
x = data2018['No']
y_offer = utils.TopOffer(data2018)
y_result = utils.Score2Res(data2018)

odds = SoccerRawDb.load_oddsdata()
y_odds = odds['Odds1'].values
x_odds = odds['No'].values

plt.plot(x, y_offer, 'bo-',label='Offer')
plt.plot(x, y_result, 'go-',label='result')
plt.plot(x_odds, y_odds, 'r',label='Odds')
plt.legend()
plt.show()

"""
Blow code compute the margin rate: 利润给出率，也就是博彩公司的利润率，
也就是说假如赔率相对公平，那么理论上买任何一个赔率的概率是相同的，那么这种情况下，
博彩公司的利润率
"""
dataset = dataset_factory.get_dataset('310')
(train_x, train_y) = dataset.get_split('train','../data/')

arr = train_x.as_matrix()
hwmax,deumax,gwmax=np.max(arr,axis=0)
offers = train_x.as_matrix()
results = (2-train_y).as_matrix()
oddidx=np.where(offers == np.min(arr,axis=1).reshape([-1,1]) )[1]

for i in range(0,len(offers)):
    usum= 1/offers[i][0] + 1/offers[i][1] + 1/offers[i][2]
    print("%d:%f"%(i,usum))

"""
Blow code snippet compute the company's highest probability of single match,
Compare with the real reault and print correct and wrong odds
"""
correct_idx=np.where(results.reshape([-1,])==oddidx)[0]
wrong_idx=np.where(results.reshape([-1,])!=oddidx)[0]

for i in range(0,len(correct_idx)):
    print("%d:Offer %f,%f,%f,odds=%d,result=%d"%
        (i,offers[correct_idx[i]][0],offers[correct_idx[i]][1],offers[correct_idx[i]][2],oddidx[correct_idx[i]],results[correct_idx[i]]))

for i in range(0,len(wrong_idx)):
    print("%d:Offer %f,%f,%f,odds=%d,result=%d"%(i,offers[wrong_idx[i]][0],offers[wrong_idx[i]][1],offers[wrong_idx[i]][2],oddidx[wrong_idx[i]],results[wrong_idx[i]]))

"""
Blow code will show the matchness of the odds offer and real result,
3 axis represent the HostWin,Deuce,GuestWin offers, and red points means correct,
blue ones means wrong
"""

fig = plt.figure() 
ax = plt.axes(projection='3d') 
ax.set_xlabel('HostWin') 
ax.set_ylabel('Deuce') 
ax.set_zlabel('GuestWin');

for c, m, idx in [('r', 'o', correct_idx), ('b', '^', wrong_idx)]:
    ax.scatter(offers[idx,0], offers[idx,1], offers[idx,2], c=c, marker=m)

plt.show()
