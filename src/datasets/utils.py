import pandas as pd
import numpy as np
import math

RESULT_NAMES = ['GuestWin','Deuce','HostWin']#0,1,2

def Score2Res(rec):
    rownum = rec.shape[0]
    data = pd.DataFrame(data=np.ones(rownum) , columns=['Res'],dtype=np.int32)

    for i in range(rownum):
        if rec['HostGoal'].values[i] > rec['GuestGoal'].values[i]:
            data['Res'][i] = 2
        elif rec['HostGoal'].values[i] < rec['GuestGoal'].values[i]:
            data['Res'][i] = 0 
        #1 is preset
    return data

def Score2Res90(rec):
    rownum = rec.shape[0]
    data = pd.DataFrame(data=np.ones(rownum) , columns=['Res90'],dtype=np.int32)

    for i in range(rownum):
        if rec['Comments'].values[i]!=('NRM'):
            continue
    
        if rec['HostGoal'].values[i] > rec['GuestGoal'].values[i]:
            data['Res90'][i] = 2 
        elif rec['HostGoal'].values[i] < rec['GuestGoal'].values[i]:
            data['Res90'][i] = 0 
        #1 is preset
    return data

def TopOffer(offers):
        res=offers[RESULT_NAMES].values
        oddidx=np.where(res==np.min(res,axis=1).reshape([-1,1]) )[1]
        return oddidx
