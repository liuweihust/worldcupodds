import pandas as pd
import numpy as np
import math
import tensorflow as tf
from utils import Score2Res
from utils import Score2Res90

CSV_INPUT_COLUMN_NAMES = ['Year','Month','Day','Time','No','Host','Guest','HostWin','Deuce','GuestWin','HostGoal','GuestGoal','Comments']
CSV_NATIONS_COLUMN_NAMES = [ 'Host','Guest']
RESULT_NAMES = ['GuestWin','Deuce','HostWin']#0,1,2

GroupFile='worldcup_group.csv'
FinalFile='worldcup_final.csv'
predfiles='worldcup_pred.csv'
oddsfile='worldcup_odds.csv'

evalfiles='worldcup-eval.csv'

def PrintDict():
    for i,v in enumerate(RESULT_NAMES):
        print("ID:%d = Name:%s"%(i,v))

def get_split(mode,data_dir='../data/'):
    if mode=='train':
        return load_traindata(data_dir='../data/')
    elif mode=='eval':
        return load_evaldata(data_dir='../data/')
    elif mode=='pred':
        return load_preddata(data_dir='../data/')

def load_data(data_dir='../data/',csvfile=None):
    data = pd.read_csv(data_dir+csvfile, header=0)
    return data

def load_grouprawdata(data_dir='../data/'):
    return load_data(data_dir=data_dir,csvfile=GroupFile)

def load_finalrawdata(data_dir='../data/'):
    return load_data(data_dir=data_dir,csvfile=FinalFile)

def load_allrawdata(data_dir='../data/'):
    groupdata = load_grouprawdata(data_dir)
    finaldata = load_finalrawdata(data_dir)
    excludegroupdata = groupdata[CSV_INPUT_COLUMN_NAMES]
    excludefinaldata = finaldata[CSV_INPUT_COLUMN_NAMES]

    result_x = pd.concat([excludegroupdata,excludefinaldata], axis=0)
    return result_x

def load_preddata(data_dir='../data/'):
    return load_data(data_dir=data_dir,csvfile=predfiles)

def load_oddsdata(data_dir='../data/'):
    return load_data(data_dir=data_dir,csvfile=oddsfile)

if __name__ == '__main__':
    train_x = load_allrawdata()
    print(train_x)

    preddata = load_preddata()
    print(preddata)

    oddsdata = load_oddsdata()
    print(oddsdata)

