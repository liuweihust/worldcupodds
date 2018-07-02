from mpl_toolkits import mplot3d
#matplotlib inline
#import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import sys
from datasets import dataset_factory
 
#if len(sys.argv) < 2:
#    exit

dataset = dataset_factory.get_dataset('310')
(train_x, train_y) = dataset.get_split('train','../data/')

#print(train_x,train_y)
arr = train_x.as_matrix()
hwmax,deumax,gwmax=np.max(arr,axis=0)

offers = train_x.as_matrix()
results = (2-train_y).as_matrix()
oddidx=np.where(offers == np.min(arr,axis=1).reshape([-1,1]) )[1]

correct_idx=np.where(results.reshape([-1,])==oddidx)[0]
wrong_idx=np.where(results.reshape([-1,])!=oddidx)[0]

for i in range(0,len(offers)):
    usum= 1/offers[i][0] + 1/offers[i][1] + 1/offers[i][2]
    print("%d:%f"%(i,usum))

for i in range(0,len(correct_idx)):
    print("%d:Offer %f,%f,%f,odds=%d,result=%d"%
        (i,offers[correct_idx[i]][0],offers[correct_idx[i]][1],offers[correct_idx[i]][2],oddidx[correct_idx[i]],results[correct_idx[i]]))

for i in range(0,len(wrong_idx)):
    print("%d:Offer %f,%f,%f,odds=%d,result=%d"%(i,offers[wrong_idx[i]][0],offers[wrong_idx[i]][1],offers[wrong_idx[i]][2],oddidx[wrong_idx[i]],results[wrong_idx[i]]))


fig = plt.figure() 
ax = plt.axes(projection='3d') 
ax.set_xlabel('HostWin') 
ax.set_ylabel('Deuce') 
ax.set_zlabel('GuestWin');

for c, m, idx in [('r', 'o', correct_idx), ('b', '^', wrong_idx)]:
    ax.scatter(offers[idx,0], offers[idx,1], offers[idx,2], c=c, marker=m)

plt.show()
