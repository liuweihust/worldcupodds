import pandas as pd
import numpy as np
import math
import tensorflow as tf
import Soccer310Db
import Groups
import sys
import copy

#trainfiles='worldcup.csv'
grouppointfile='grouppoint.csv'

"""
    {
        2006:{
            'A':{
                    0:[0,0,0,0],
                    1:[3,3,0,0],
                    2:[6,6,0,0],
                    3:[9,6,3,0]
                },
            'B':{
                    0:[0,0,0,0],
                    1:[3,3,0,0],
                    2:[6,3,3,0],
                    3:[9,4,4,0]
                },
            },
        2010:{
            'A':{
                    0:[0,0,0,0],
                    1:[3,3,0,0],
                    2:[6,6,0,0],
                    3:[9,6,3,0]
                },
            'B':{
                    0:[0,0,0,0],
                    1:[3,3,0,0],
                    2:[6,3,3,0],
                    3:[9,4,4,0]
                },
            },
    }
"""

def GetHostGuestPts(hscore,gscore):
    if hscore>gscore:
        return 3,0
    elif hscore==gscore:
        return 1,1
    else:
        return 0,3

TeamNames=['T1','T2','T3','T4']

class GroupsPoints():
    def __init__(self,data_dir='../data/'):
        self._data_dir = data_dir
        self._matches = None
        self._groups =  Groups.Groups(self._data_dir)
        self._points = {}
        self._dataframe=None

    def loaddata(self):
        self._dataframe = pd.read_csv(self._data_dir+grouppointfile, header=0,index_col=0)

    def RebuildPoints(self):
        self._matches =  Soccer310Db.load_grouptraindata(self._data_dir)
        self._groups.LoadFile()

        self._matches.sort_values(by=['Year','Month','Day'],inplace=True)
        rownum = self._matches.shape[0]

        DFPoints = pd.DataFrame(data=np.zeros([rownum,len(TeamNames)]),
                            columns=TeamNames,dtype=np.int32)
        DFYear = pd.DataFrame(data=np.zeros(rownum),
                            columns=['Year'],dtype=np.int32)
        DFGroup = pd.DataFrame(data=['' for i in range(rownum)],
                            columns=['Group'],dtype=str)
        DFRound = pd.DataFrame(data=np.zeros(rownum),
                            columns=['Round'],dtype=np.int32)
        DFNo = pd.DataFrame(data=np.zeros(rownum),
                            columns=['No'],dtype=np.int32)

        matchnum={}
        prevyear=0
        for i in range(rownum):
            year = self._matches['Year'][i]
            host = self._matches['Host'][i]
            guest = self._matches['Guest'][i]
            group = self._groups.GetGroup(host,year)
            group2 = self._groups.GetGroup(guest,year)
            
            if year != prevyear:
                matchno=1
            else:
                matchno += 1
            prevyear=year
            DFNo['No'][i]=matchno
            DFYear['Year'][i]=year

            if group!=group2:
                continue

            nations = self._groups.GetGroupMem(year,group)
 
            if not self._points.has_key(year):
                self._points[year]={}
                matchnum[year]={}
                groups = self._groups.GetGroupNames(year)
                for gi in groups:
                    matchnum[year][gi]=0
                    self._points[year][gi]={}
                    self._points[year][gi][0]={}
                    for ni in self._groups.GetGroupMem(year,gi):
                        self._points[year][gi][0][ni]=0
            
            ground=int(matchnum[year][group]/2)+1
            DFGroup['Group'][i]=group
            DFRound['Round'][i]=ground

            if matchnum[year][group] %2 == 0:
                self._points[year][group][ground]=copy.deepcopy(self._points[year][group][ground-1])
         
            for j in range(len(TeamNames)): 
                DFPoints[ TeamNames[j] ][i] = self._points[year][group][ground][ nations[j] ] 

            hp,gp = GetHostGuestPts(self._matches['HostGoal'][i],self._matches['GuestGoal'][i])
            self._points[year][group][ground][host]+=hp
            self._points[year][group][ground][guest]+=gp
            matchnum[year][group] += 1

        self._dataframe = pd.concat([DFNo,DFYear,DFGroup,DFRound,DFPoints], axis=1)
        print(self._dataframe)
    
    def GetAllPointsData(self):
        return self._dataframe

    def GetPoints(self,year,group,ground,T1,T2,T3):
        #return self._points[year][group][ground]
        return self._dataframe[year][group][ground]
        #return [0,0,0,0]

    def PrintPoints(self):
        for year in self._points.keys():
            print("Year:%d"%year)
            for group in self._points[year].keys():
                print(" Group:%s"%group)
                sys.stdout.write("                ")
                for nation in self._groups.GetGroupMem(year,group):
                   sys.stdout.write("%18s"%nation)
                sys.stdout.write("\n")
                for ground in self._points[year][group]:
                    sys.stdout.write("     round %4d:"%ground)
                    for nation in self._groups.GetGroupMem(year,group):
                       sys.stdout.write("%18d"%self._points[year][group][ground][nation])
                    sys.stdout.write("\n")
    
    def PrintFramePoints(self):
        print(self._dataframe)

    def ExportFile(self,csvfile,WithMatch=False):
        if WithMatch:
            res = pd.concat([self._dataframe,self._matches],axis=1)
            res.to_csv(self._data_dir+csvfile) 
        else:
            self._dataframe.to_csv(self._data_dir+csvfile) 

if __name__ == '__main__':
    gp = GroupsPoints()
    #gp.loaddata()

    gp.RebuildPoints()
    #gp.PrintPoints()
    
    gp.ExportFile('1grouppoint.csv',WithMatch=False)

    #gp.PrintFramePoints()
    a=gp.GetAllPointsData()
    print(a)
    """
    year=2018
    group='B'
    ground=2
    M1='Portugal'
    M2='Spain'
    pts = gp.GetPoints(year=year,group=group,ground=ground,T1=M1,T2=M2,T3=M2)
    print("Year:%d Group:%s Round:%d Team:%s vs %s pts="%(year,group,ground,M1,M2))
    print(pts) 
    """
